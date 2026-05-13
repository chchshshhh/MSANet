

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.utils.model_zoo as model_zoo

from src.utils.constrain import *
from src.utils.BCA import *
import math
from src.utils.unettool2 import *
from src.utils.NAT import *
from uniformer.uniformer import uniformer_small



class PPM(nn.ModuleList):
    def __init__(self, pool_sizes, in_channels, out_channels):
        super(PPM, self).__init__()
        self.pool_sizes = pool_sizes
        self.in_channels = in_channels
        self.out_channels = out_channels
        for pool_size in pool_sizes:
            self.append(
                nn.Sequential(
                    nn.AdaptiveMaxPool2d(pool_size),
                    nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1),
                )
            )

    def forward(self, x):
        out_puts = []
        for ppm in self:
            ppm_out = nn.functional.interpolate(ppm(x), size=(x.size(2), x.size(3)), mode='bilinear',
                                                align_corners=True)
            out_puts.append(ppm_out)
        return out_puts


class PPMHEAD(nn.Module):
    def __init__(self, in_channels, out_channels, pool_sizes=[1, 2, 3, 6], num_classes=31):
        super(PPMHEAD, self).__init__()
        self.pool_sizes = pool_sizes
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.psp_modules = PPM(self.pool_sizes, self.in_channels, self.out_channels)
        self.final = nn.Sequential(
            nn.Conv2d(self.in_channels + len(self.pool_sizes) * self.out_channels, 4 * self.out_channels,
                      kernel_size=1),
            nn.BatchNorm2d(4 * self.out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.psp_modules(x)
        out.append(x)
        out = torch.cat(out, 1)
        out = self.final(out)
        return out


class FPNHEAD(nn.Module):
    def __init__(self, channels=512):
        super(FPNHEAD, self).__init__()
        self.PPMHead = PPMHEAD(in_channels=512, out_channels=128)

        self.Conv_fuse1 = nn.Sequential(
            nn.Conv2d(320, 320, 1),
            nn.BatchNorm2d(320),
            nn.ReLU()
        )
        self.Conv_fuse1_ = nn.Sequential(
            nn.Conv2d(320 + 512, 320, 1),
            nn.BatchNorm2d(320),
            nn.ReLU()
        )
        self.Conv_fuse2 = nn.Sequential(
            nn.Conv2d(128, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.Conv_fuse2_ = nn.Sequential(
            nn.Conv2d(320 + 128, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.Conv_fuse3 = nn.Sequential(
            nn.Conv2d(64, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.Conv_fuse3_ = nn.Sequential(
            nn.Conv2d(128 + 64, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.fuse_all = nn.Sequential(
            nn.Conv2d(512+320+128+64, channels // 4, 1),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU()
        )
        self.up2=UpsampleDeterministic(2)

    def forward(self, input_fpn):
        x1 = self.PPMHead(input_fpn[-1])

        x = self.up2(x1)
        x = torch.cat([x, self.Conv_fuse1(input_fpn[-2])], dim=1)
        x2 = self.Conv_fuse1_(x)

        x = self.up2(x2)
        x = torch.cat([x, self.Conv_fuse2(input_fpn[-3])], dim=1)
        x3 = self.Conv_fuse2_(x)

        x = self.up2(x3)
        x = torch.cat([x, self.Conv_fuse3(input_fpn[-4])], dim=1)
        x4 = self.Conv_fuse3_(x)

        x1 = self.up2(self.up2(self.up2(x1)))
        x2 = self.up2(self.up2(x2))
        x3 = self.up2(x3)

        x = self.fuse_all(torch.cat([x1, x2, x3, x4], 1))

        return x





class FcnNet(nn.Module):
    def __init__(self,
                 num_classes: int = 1,
                 pretrain_backbone: bool = False,
                 ):
        super(FcnNet, self).__init__()
        # backbone = convnext_tiny(1)
        # # backbone2= convnext_tiny(1)
        # if pretrain_backbone:
        #     weights_dict = \
        #         model_zoo.load_url('https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth')[
        #             'model']
        #     for k in list(weights_dict.keys()):
        #         if "head" in k:
        #             del weights_dict[k]
        #     backbone.load_state_dict(weights_dict, strict=False)
        #     # backbone2.load_state_dict(weights_dict,strict=False)
        # self.backbone = backbone
        backbone = uniformer_small(pretrained=True)
        # backbone2= convnext_tiny(1)
        # if pretrain_backbone:
        weights_dict = torch.load('/raid/csh/IRL2a/uniformer_small_tl_384.pth')
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        # self.model.load_state_dict(weights_dict, strict=False)
        backbone.load_state_dict(weights_dict, strict=False)
        self.backbone = backbone
        self.decoder=FPNHEAD()
        # self.backbone2 = backbone2
        # self.constraint=BayarConv2d(in_channels=1, out_channels=3, padding=2)
        # self.head = _DAHead(768)

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
        # self.trans1=DoubleConv(96,96)
        # self.trans2=DoubleConv(192,96)
        # self.trans3=DoubleConv(384,96)
        # self.trans4=DoubleConv(768,96)
        # self.up1 = Up2(192, 96)
        # self.up2 = Up2(192, 96)
        # self.up3 = Up2(192, 96)
        # self.nat=NATBlock(768)
        # self.sc=PatchTrans(768,16)
        # self.sc2=PatchTrans(768,16)
        # self.conv_l = DoubleConv(768, 384, 384)
        # self.bca=BCA(768,768,256)
        self.out_conv = OutConv(128, num_classes)
        self.upf=UpsampleDeterministic(4)
        # self.sc3=PatchTrans(768,16)
        # self.sc4=PatchTrans(768,16)
        # self.sc3=PatchTrans(768,16)
        # self.sc4=PatchTrans(768,16)
    def forward(self, x: torch.Tensor, h=384, w=384,t=True) -> Dict[str, torch.Tensor]:
        input_shape = x.shape[-2:]
        layers = self.backbone(x)
        x1 = layers['layer1']
        x2 = layers['layer2']
        x3 = layers['layer3']
        x4 = layers['layer4']
        x=self.decoder([x1,x2,x3,x4])
        x=self.out_conv(x)
        # x1=self.trans1(x1)
        # x2=self.trans2(x2)
        # x3=self.trans3(x3)
        # x4=self.trans4(x4)
        #
        # x = self.up1(x4, x3)
        # x = self.up2(x, x2)
        # x = self.up3(x, x1)
        # x = self.out_conv(x)


        x = self.upf(x)
        return {"out": x}



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
