

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.constrain import *
from src.utils.BCA import *
import math
from src.utils.unettool2 import *
from src.utils.zoomnet import *
from src.utils.NAT import *
from uniformer.uniformer import uniformer_small


def weight_init(module):
    for n, m in module.named_children():
        print('initialize: ' + n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, (nn.ReLU, nn.AdaptiveAvgPool2d, nn.Softmax, nn.Dropout2d)):
            pass
        else:
            m.initialize()

class Yblock(nn.Module):
    def __init__(self):
        super(Yblock, self).__init__()
        self.up=UpsampleDeterministic(2)
        self.convA1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bnA1 = nn.BatchNorm2d(64)

        self.convB1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bnB1 = nn.BatchNorm2d(64)

        self.convAB = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bnAB = nn.BatchNorm2d(64)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.convg = nn.Conv2d(128, 128, 1)
        self.sftmax = nn.Softmax(dim=1)

    #  self.dropout = nn.Dropout2d(p=0.1)

    def forward(self, x, y):
        if x.size()[2:] != y.size()[2:]:
            # y = F.interpolate(y, size=x.size()[2:], mode='bilinear', align_corners=True)
            y=self.up(y)
        fuze = torch.mul(x, y)
        y = F.relu(self.bnB1(self.convB1(fuze + y)), inplace=True)
        x = F.relu(self.bnA1(self.convA1(fuze + x)), inplace=True)
        x = torch.cat((x, y), 1)
        y = self.convg(self.gap(x))
        x = torch.mul(self.sftmax(y) * y.shape[1], x)
        #   x = self.dropout(x)
        x = F.relu(self.bnAB(self.convAB(x)), inplace=True)
        return x

    def initialize(self):
        weight_init(self)
        print("Yblock module init")



class FcnNet(nn.Module):
    def __init__(self,
                 num_classes: int = 1,
                 pretrain_backbone: bool = False,
                 ):
        super(FcnNet, self).__init__()

        backbone = uniformer_small(pretrained=True)

        weights_dict = torch.load('/raid/csh/IRL2a/uniformer_small_tl_384.pth')
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        backbone.load_state_dict(weights_dict, strict=False)
        self.backbone = backbone
        self.u1 = Yblock()
        self.u2 = Yblock()
        self.u3 = Yblock()
        self.u4 = Yblock()
        self.u5 = Yblock()
        self.u6 = Yblock()

        self.translayer = TransLayer2(out_c=64)
        self.out_conv4 = OutConv(64, num_classes)
        self.out_conv3 = OutConv(64, num_classes)
        self.out_conv1 = OutConv(64, num_classes)
        self.out_conv2 = OutConv(64, num_classes)
        self.out_conv12=OutConv(64, num_classes)
        self.out_conv = OutConv(64, num_classes)

        self.up2 = UpsampleDeterministic(2)
        self.up4 = UpsampleDeterministic(4)
        self.up8 = UpsampleDeterministic(8)
        self.up16 = UpsampleDeterministic(16)

    def forward(self, x: torch.Tensor, hh=512, ww=512,t=True) -> Dict[str, torch.Tensor]:
        input_shape = x.shape[-2:]
        layers = self.backbone(x)
        x1 = layers['layer1']
        x2 = layers['layer2']
        x3 = layers['layer3']
        x4 = layers['layer4']

        x1_f,x2_f,x3_f,x4_f=self.translayer(x1,x2,x3,x4)



        x4o = self.out_conv4(x4_f)
        x1o = self.out_conv1(x1_f)
        x12 = self.u1(x1_f, x2_f)
        x12o = self.out_conv12(x12)
        x23 = self.u2(x2_f, x3_f)
        x34 = self.u3(x3_f, x4_f)
        x3 = self.out_conv3(x34)
        x13 = self.u4(x12, x23)
        x24 = self.u5(x23, x34)

        x14 = self.u6(x13, x24)
        x14 = self.out_conv(x14)

        if t:
            x1 = self.up4(x1o)
            x4 = self.up2(self.up16(x4o))
            x3 = self.up16(x3)
            x = self.up4(x14)
            # x12o =  F.interpolate(x12o, size=(ww, hh), mode="bilinear", align_corners=False)
            x12o = self.up4(x12o)
        else:
            x1 = F.interpolate(x1o, size=(ww, hh), mode="bilinear", align_corners=False)
            x4 = F.interpolate(x4o, size=(ww, hh), mode="bilinear", align_corners=False)
            x3 = F.interpolate(x3, size=(ww, hh), mode="bilinear", align_corners=False)
            x = F.interpolate(x14, size=(ww, hh), mode="bilinear", align_corners=False)
            x12o = F.interpolate(x12o, size=(ww, hh), mode="bilinear", align_corners=False)

        return {"out": x, 'x3': x3, 'x4': x4, 'edge': x12o, 'edge2': x1}



class FcncsNet(nn.Module):
    def __init__(self,
                 num_classes: int = 1,
                 pretrain_backbone: bool = False,
                 ):
        super(FcncsNet, self).__init__()
        backbone = convnext_tiny(1)

        if pretrain_backbone:
            weights_dict = \
                model_zoo.load_url('https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth')[
                    'model']
            for k in list(weights_dict.keys()):
                if "head" in k:
                    del weights_dict[k]
            backbone.load_state_dict(weights_dict, strict=False)

        self.backbone = backbone
        # self.head = _DAHead(768)
        self.fcnhead = FCNHead(768, 1)
        # self.head2 = AlignHead(768, fpn_dim=96)
        # self.conv_last = nn.Sequential(
        #     conv3x3_bn_relu(4 * 96, 96, 1),
        #     nn.Conv2d(96, 1, kernel_size=1)
        # )
        # self.up4 = Up(192,96)
        # self.fft1=ResBlock_do_fft_bench(96)
        # self.fft2 = ResBlock_do_fft_bench(192)
        # self.fft3 = ResBlock_do_fft_bench(384)
        # self.fre1=MultiSpectralAttentionLayer(96,128,128)
        # self.fre2=MultiSpectralAttentionLayer(192,64,64)
        # self.fre3=MultiSpectralAttentionLayer(384,32,32)
        # self.out_conv = OutConv(96, num_classes)
        self.sc=PatchTrans(768,16)
        # self.sc2=PatchTrans(768,16)
        # self.sc3=PatchTrans(768,16)
        # self.sc4=PatchTrans(768,16)
    def forward(self, x: torch.Tensor, h=512, w=512) -> Dict[str, torch.Tensor]:
        input_shape = x.shape[-2:]

        layers = self.backbone(x)
        # x1 = layers['layer1']
        # x2 = layers['layer2']
        # x3 = layers['layer3']
        x4 = layers['layer4']
        x4=self.sc(x4)
        # x4=self.sc2(x4)
        # x4=self.sc3(x4)
        # x4=self.sc4(x4)
        # x4=self.sc2(x4)
        # x4=self.sc3(x4)
        # x4=self.sc4(x4)
        # x4 = self.head(x4)+x4
        # x4 = self.head(x4)
        # x_ = [x1, x2, x3, x4]

        x = self.fcnhead(x4)
        # x = self.up4(x, x1)
        # logits = self.fcnhead(x4)
        # x = F.interpolate(logits, size=input_shape, mode="bilinear", align_corners=False)
        x = F.interpolate(x, size=(w, h), mode='bilinear', align_corners=False)
        return {"out": x}

if __name__ == "__main__":
    import os

    model = FcnNet(pretrain_backbone=True).cuda()
    # checkpoint = torch.load('/disk/csh/forgerydetection4/save_weights_fcnscnoiseu_50/best_model1.pth',
    #                         map_location='cpu')
    # model.load_state_dict(checkpoint['model'])
    #
    # model.eval()
    x=torch.rand(2,3,384,384).cuda()
    # input = torch.load('/disk/csh/forgerydetection4/a.pth',map_location="cuda:0")
    output = model(x, 384, 384)

    print(output['out'].size())
    # print(out3.size())
    # print(output['out'].size())
    # print(output['edge'].size())

# print(out3.size())
    # print(output['out'].size())
    # print(output['edge'].size())
