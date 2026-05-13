

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.convnext_transformer import convnext_tiny,convnext_base
import torch.utils.model_zoo as model_zoo
from src.fad import *
from src.fft import *
from src.fratt import *
from src.utils.constrain import *
from src.utils.BCA import *
import math
from src.utils.unettool2 import *
from src.utils.NAT import *
from uniformer.uniformer import uniformer_small
from src.swinformer_backbone_seg import Swinformer
from src.utils.zoomnet import *
from torchvision.transforms import RandomHorizontalFlip,RandomVerticalFlip
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

class PPMHEAD2(nn.Module):
    def __init__(self, in_channels, out_channels, pool_sizes=[1, 2, 3, 6], num_classes=31):
        super(PPMHEAD2, self).__init__()
        self.pool_sizes = pool_sizes
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.psp_modules = PPM(self.pool_sizes, self.in_channels, self.out_channels)
        self.final = nn.Sequential(
            nn.Conv2d(self.in_channels + len(self.pool_sizes) * self.out_channels,64,
                      kernel_size=1),
            nn.BatchNorm2d(64),
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

class FPNHEAD2(nn.Module):
    def __init__(self, channels=512):
        super(FPNHEAD2, self).__init__()
        self.PPMHead = PPMHEAD2(in_channels=64, out_channels=64)

        self.Conv_fuse1 = nn.Sequential(
            nn.Conv2d(64, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.Conv_fuse1_ = nn.Sequential(
            nn.Conv2d(64 + 64, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.Conv_fuse2 = nn.Sequential(
            nn.Conv2d(64, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.Conv_fuse2_ = nn.Sequential(
            nn.Conv2d(64 + 64, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.Conv_fuse3 = nn.Sequential(
            nn.Conv2d(64, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.Conv_fuse3_ = nn.Sequential(
            nn.Conv2d(64 + 64, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.fuse_all = nn.Sequential(
            nn.Conv2d(64+64+64+64, 64, 1),
            nn.BatchNorm2d(64),
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

class SIU42(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.conv_l_pre_down = ConvBNReLU(in_dim, in_dim, 5, stride=1, padding=2)
        self.conv_l_post_down = ConvBNReLU(in_dim, in_dim, 3, 1, 1)

        self.conv_m = ConvBNReLU(in_dim, in_dim, 3, 1, 1)

        # self.dup = UpsampleDeterministic()
        self.conv_s_pre_up = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_s_post_up = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.trans = nn.Sequential(
            ConvBNReLU(2 * in_dim, in_dim, 1),
            ConvBNReLU(in_dim, in_dim, 3, 1, 1),
            ConvBNReLU(in_dim, in_dim, 3, 1, 1),
            nn.Conv2d(in_dim, 2, 1),
        )

    def forward(self, s, m, return_feats=False):
        """l,m,s表示大中小三个尺度，最终会被整合到m这个尺度上"""
        # tgt_size = m.shape[2:]
        # 尺度缩小
        # l = self.conv_l_pre_down(l)

        # l = F.adaptive_max_pool2d(l, tgt_size) + F.adaptive_avg_pool2d(l, tgt_size)
        # l = self.conv_l_post_down(l)
        # 尺度不变
        # m = self.conv_m(m)
        # 尺度增加(这里使用上采样之后卷积的策略)
        # s = self.conv_s(s)
        # s=self.conv_s_pre_up(s)
        # s = cus_sample(s, mode="size", factors=m.shape[2:])
        # s = self.dup(s)
        # s = self.conv_s_post_up(s)
        attn = self.trans(torch.cat([m, s], dim=1))
        attn_m, attn_s = torch.softmax(attn, dim=1).chunk(2, dim=1)
        lms = attn_m * m + attn_s * s

        # if return_feats:
        #     return lms, dict(attn_l=attn_l, attn_m=attn_m, attn_s=attn_s, l=l, m=m, s=s)
        return lms

from src.mtdr.mscale import PatchTrans

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
        self.decoder=FPNHEAD2()
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
        self.flip1 = RandomHorizontalFlip(p=1)
        self.flip2 = RandomVerticalFlip(p=1)
        self.translayer1 = TransLayer2(out_c=64)
        self.translayer2 = TransLayer2(out_c=64)
        self.translayer3 = TransLayer2(out_c=64)
        self.siu1 = SIU42(64)
        self.siu2 = SIU42(64)
        self.siu3 = SIU42(64)
        self.siu4 = SIU42(64)
        self.prject1 = project(64, 64)
        self.prject2 = project(64, 64)
        self.prject3 = project(64, 64)
        self.prject4 = project(64, 64)

        self.out_conv = OutConv(64, num_classes)
        self.upf=UpsampleDeterministic(4)
        # self.sc3=PatchTrans(768,16)
        # self.sc4=PatchTrans(768,16)
        # self.sc3=PatchTrans(768,16)
        # self.sc4=PatchTrans(768,16)
    def forward(self, x1: torch.Tensor,x2,x3, h=384, w=384,t=True) -> Dict[str, torch.Tensor]:
        input_shape = x1.shape[-2:]
        layerl = self.backbone(x1)
        layerm = self.backbone(x2)
        layerh = self.backbone(x3)
        # x_c=self.constraint(x_c)
        # c_layers=self.backbone2(x_c)
        xl1 = layerl['layer1']
        xl2 = layerl['layer2']
        xl3 = layerl['layer3']
        xl4 = layerl['layer4']
        # xm1 = layerm['layer1']
        # xm2 = layerm['layer2']
        # xm3 = layerm['layer3']
        # xm4 = layerm['layer4']
        xh1 = layerh['layer1']
        xh2 = layerh['layer2']
        xh3 = layerh['layer3']
        xh4 = layerh['layer4']
        xl1, xl2, xl3, xl4 = self.translayer1(xl1, xl2, xl3, xl4)
        # xm1, xm2, xm3, xm4 = self.translayer2(xm1, xm2, xm3, xm4)
        xh1, xh2, xh3, xh4 = self.translayer3(xh1, xh2, xh3, xh4)

        feat2 = self.siu1(xl1, self.flip2(xh1))
        feat3 = self.siu2(xl2, self.flip2(xh2))
        feat4 = self.siu3(xl3, self.flip2(xh3))
        feat5 = self.siu4(xl4, self.flip2(xh4))

        x1_p = self.prject1(feat2)
        x2_p = self.prject2(feat3)
        x3_p = self.prject3(feat4)
        x4_p = self.prject4(feat5)
        list = []
        list.append(x1_p)
        list.append(x2_p)
        list.append(x3_p)
        list.append(x4_p)

        x=self.decoder([feat2,feat3,feat4,feat5])
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
