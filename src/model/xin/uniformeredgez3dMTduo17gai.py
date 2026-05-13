

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
# from src.convnext_transformer import convnext_tiny,convnext_base
from src.utils.f3net import FAD_Head
import torch.utils.model_zoo as model_zoo
from src.utils.constrain import *
from src.utils.BCA import *
import math
from src.utils.unettool2 import *
from src.utils.reco import *
from src.utils.NAT import *
from einops import rearrange
from src.utils.fresod import *
from src.utils.zoomnet import *
from torchvision.transforms import RandomHorizontalFlip,RandomVerticalFlip
from uniformer.uniformer import uniformer_small
from src.utils.f3net import FAD_Head1,FAD_Head2

def batch_norm(X, eps=0.001):
    # extract the dimensions
    N, C, H, W = X.shape
    device=X.device
    # mini-batch mean
    mean = X.mean(axis=(0, 2, 3)).to(device)
    # mini-batch variance
    variance = ((X - mean.view((1, C, 1, 1))) ** 2).mean(axis=(0, 2, 3)).to(device)
    # normalize
    X = (X - mean.reshape((1, C, 1, 1))) * 1.0 / torch.pow((variance.view((1, C, 1, 1)) + eps), 0.5)
    return X.to(device)

class AnomalyDetector(nn.Module):
    # ********** IMAGE MANIPULATION TRACE FEATURE EXTRACTOR *********
    def __init__(self, eps=10 ** (-6), with_GRU=False):
        super(AnomalyDetector, self).__init__()

        self.eps = eps
        self.relu = nn.ReLU()

        self.with_GRU = with_GRU

        # ********** LOCAL ANOMALY DETECTOR *********

        self.adaptation = nn.Conv2d(64, 64, 1, 1, padding=0, bias=False)

        self.sigma_F = nn.Parameter(torch.zeros((1, 64, 1, 1)), requires_grad=True)

        # self.pool31 = nn.AvgPool2d(31, stride=1, padding=15, count_include_pad=False)
        # self.pool15 = nn.AvgPool2d(15, stride=1, padding=7, count_include_pad=False)
        # self.pool7 = nn.AvgPool2d(7, stride=1, padding=3, count_include_pad=False)

        # if not (self.with_GRU):
        #     self.conv_lstm = ConvLSTM(input_dim=64,
        #                               hidden_dim=8,
        #                               kernel_size=(7, 7),
        #                               num_layers=1,
        #                               batch_first=False,
        #                               bias=True,
        #                               return_all_layers=False)
        # else:
        #     self.conv_gru = ConvGRU(input_dim=64,
        #                             hidden_dim=8,
        #                             kernel_size=(7, 7),
        #                             num_layers=1,
        #                             batch_first=False,
        #                             bias=True,
        #                             return_all_layers=False)

        # self.end = nn.Sequential(nn.Conv2d(8, 1, 7, 1, padding=3), nn.Sigmoid())

    def forward(self, IMTFE_output):
        _, _, H, W = IMTFE_output.shape


        self.GlobalPool = nn.AvgPool2d((H, W), stride=1)


        ## Local Anomaly Feature Extraction
        X_adapt = self.adaptation(IMTFE_output)
        X_adapt = batch_norm(X_adapt)

        # Z-pool concatenation
        mu_T = self.GlobalPool(X_adapt)
        sigma_T = torch.sqrt(self.GlobalPool(torch.square(X_adapt - mu_T)))
        sigma_T = torch.max(sigma_T, self.sigma_F + self.eps)
        inv_sigma_T = torch.pow(sigma_T, -1)
        zpoolglobal = torch.abs((mu_T - X_adapt) * inv_sigma_T)


        return zpoolglobal

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data, gain=0.02)

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class PixelAttention(nn.Module):
    def __init__(self, in_channels):
        super(PixelAttention, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 1, 1, 0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64,1,1,0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pa = SpatialAttention()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x_srm = self.srm(x)
        fea = self.conv(x)
        att_map = self.pa(fea)

        return att_map


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

class project(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(project, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=1,stride=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),

        )


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
        backbone1 = uniformer_small(pretrained=True)
        backbone2 = uniformer_small(pretrained=True)
        backbone3 = uniformer_small(pretrained=True)
        # backbone2= convnext_tiny(1)
        # if pretrain_backbone:
        weights_dict = torch.load('/raid/csh/IRL2a/uniformer_small_tl_384.pth')
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        # self.model.load_state_dict(weights_dict, strict=False)
        backbone1.load_state_dict(weights_dict, strict=False)
        backbone2.load_state_dict(weights_dict, strict=False)
        backbone3.load_state_dict(weights_dict, strict=False)

        self.backbone1 = backbone1
        self.backbone2 = backbone2
        self.backbone3 = backbone3

        self.u1 = Yblock()
        self.u2 = Yblock()
        self.u3 = Yblock()
        self.u4 = Yblock()
        self.u5 = Yblock()
        self.u6 = Yblock()
        # self.fad=FAD_Head(384)
        self.flip1 = RandomHorizontalFlip(p=1)
        self.flip2 = RandomVerticalFlip(p=1)
        self.out_conv4 = OutConv(64, num_classes)
        self.out_conv3 = OutConv(64, num_classes)
        self.out_conv1 = OutConv(64, num_classes)
        self.out_conv = OutConv(64, num_classes)
        self.out_conv12 = OutConv(64, num_classes)
        self.translayer1 = TransLayer2(out_c=64)
        self.translayer2 = TransLayer2(out_c=64)
        self.translayer3 = TransLayer2(out_c=64)
        self.prject1 = project(64, 64)
        self.prject2 = project(64, 64)
        self.prject3 = project(64, 64)
        self.prject4 = project(64, 64)
        self.ad = AnomalyDetector()

        # self.ad2 = AnomalyDetector()
        # self.ad3 = AnomalyDetector()
        # self.ad4 = AnomalyDetector()
        self.att=PixelAttention(64)
        # self.att2=PixelAttention(64)
        # self.att3=PixelAttention(64)
        # self.att4=PixelAttention(64)

        # self.siu1 = SIU4(64)
        # self.siu2 = SIU4(64)
        # self.siu3 = SIU4(64)
        # self.siu4 = SIU4(64)
        self.fad1=FAD_Head1(384)
        self.fad2=FAD_Head2(384)
        self.siu1 = SIUonly3(64)
        self.siu2 = SIUonly3(64)
        self.siu3 = SIUonly3(64)
        self.siu4 = SIUonly3(64)

        self.out_conv = OutConv(64, num_classes)
        self.up16 = UpsampleDeterministic(16)
        self.up4 = UpsampleDeterministic(4)
        self.up2 = UpsampleDeterministic(2)
    def forward(self, x1: torch.Tensor,x2,x3, hh=384, ww=384,t=True) -> Dict[str, torch.Tensor]:
        input_shape = x2.shape[-2:]
        # x_c=rgb2gray(x)
        # x2=self.fad2(x2)
        x3=self.fad2(x3)
        layerl = self.backbone1(x1)
        layerm = self.backbone2(x2)
        layerh = self.backbone3(x3)
        # x_c=self.constraint(x_c)
        # c_layers=self.backbone2(x_c)
        xl1 = layerl['layer1']
        xl2 = layerl['layer2']
        xl3 = layerl['layer3']
        xl4 = layerl['layer4']
        xm1 = layerm['layer1']
        xm2 = layerm['layer2']
        xm3 = layerm['layer3']
        xm4 = layerm['layer4']
        xh1 = layerh['layer1']
        xh2 = layerh['layer2']
        xh3 = layerh['layer3']
        xh4 = layerh['layer4']
        xl1, xl2, xl3, xl4 = self.translayer1(xl1, xl2, xl3, xl4)
        xm1, xm2, xm3, xm4 = self.translayer2(xm1, xm2, xm3, xm4)
        xh1, xh2, xh3, xh4 = self.translayer3(xh1, xh2, xh3, xh4)

        feat2 = self.siu1(xl1, self.flip1(xm1), xh1)
        feat3 = self.siu2(xl2, self.flip1(xm2), xh2)
        feat4= self.siu3(xl3, self.flip1(xm3), xh3)
        feat5 = self.siu4(xl4, self.flip1(xm4), xh4)




        x1_f = feat2
        x2_f = feat3
        x3_f = feat4
        x4_f = feat5
        # x4_ad=self.ad1(x4_f)
        # x3_ad=self.ad2(x3_f)
        # x2_ad=self.ad3(x2_f)
        # x1_ad=self.ad4(x1_f)


        # x1_p = self.prject1(feat2)
        # x2_p = self.prject2(feat3)
        # x3_p = self.prject3(feat4)
        # x4_p = self.prject4(feat5)
        # list=[]
        # list.append(x1_p)
        # list.append(x2_p)
        # list.append(x3_p)
        # list.append(x4_p)
        #



        # x4_f=x4_f*self.att(x4_ad)+x4_f
        # x3_f = x3_f * self.att3(x3_ad) + x3_f
        # x2_f = x2_f * self.att2(x2_ad) + x2_f
        # x1_f = x1_f * self.att1(x1_ad) + x1_f

        # x3_f=x3_f*self.att3(x3_p)+x3_f
        # x2_f=x2_f*self.att2(x2_p)+x2_f
        # x1_f=x1_f*self.att1(x1_p)+x1_f

        # x4_f=x4_p+x4_f
        # x3_f=x3_p+x3_f
        # x2_f=x2_p+x2_f
        # x1_f=x1_p+x1_f

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
        # x14a=self.ad(x14)
        # x14=x14*self.att(x14a)+x14
        # x = self.d4(x4_f)
        # x = self.up4(x)
        # x = self.d3(x + x3_f)
        # x = self.up3(x)
        # x = self.d2(x + x2_f)
        # x = self.up2(x)
        # x = self.d1(x + x1_f)
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

        return {"out": x, 'x3': x3, 'x4': x4,  'edge': x12o, 'edge2': x1}
        # return {"out": x,  'x3': x3,'feature':list,'edge2':x12o}

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
    x=torch.rand(2,3,256,256).cuda()
    x0=torch.rand(2,3,128,128).cuda()
    x2=torch.rand(2,3,384,384).cuda()
    x3=torch.rand(2,192,32,32).cuda()
    # input = torch.load('/disk/csh/forgerydetection4/a.pth',map_location="cuda:0")
    output = model(x2,x2,x2, 384, 384)
    print(output['out'])

    # print(output['out'].shape())
    import torch
    from torchvision.models import resnet50
    from fvcore.nn import FlopCountAnalysis, parameter_count_table

    # 创建resnet50网络
    model =FcnNet().cuda()

    # 分析parameters
    def print_model_parm_nums(model):
        total = sum([param.nelement() for param in model.parameters()])
        print('  + Number of params: %.2fM' % (total / 1e6))
    print_model_parm_nums(model)
    # print(out3.size())
    # print(output['out'].size())
    # print(output['edge'].size())

# print(out3.size())
# print(output['out'].size())
# print(output['edge'].size())