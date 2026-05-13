

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
# from src.convnext_transformer import convnext_tiny,convnext_base

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
            ppm_out = nn.functional.interpolate(ppm(x), size=(x.size(2), x.size(3)))
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








class FcnNet(nn.Module):
    def __init__(self,
                 num_classes: int = 1,
                 pretrain_backbone: bool = False,
                 ):
        super(FcnNet, self).__init__()
        backbone = uniformer_small(pretrained=True,img_size=512)
        weights_dict = torch.load('/raid/csh/IRL2a/uniformer_small_tl_384.pth')
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]

        backbone.load_state_dict(weights_dict, strict=False)

        self.backbone = backbone

        self.decoder=FPNHEAD()

        self.out_conv = OutConv(128, num_classes)
        # self.upf=UpsampleDeterministic(4)

    def forward(self, x1: torch.Tensor,x2,x3, hh=512, ww=512,t=True) -> Dict[str, torch.Tensor]:
        input_shape = x2.shape[-2:]

        layerl = self.backbone(x1)

        xl1 = layerl['layer1']
        xl2 = layerl['layer2']
        xl3 = layerl['layer3']
        xl4 = layerl['layer4']


        x=self.decoder([xl1,xl2,xl3,xl4])
        x=self.out_conv(x)
        x =F.interpolate(x,[512,512])
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
    x2=torch.rand(2,3,512,512).cuda()
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
