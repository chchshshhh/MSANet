import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
            if m.weight is None:
                pass
            elif m.bias is not None:
                nn.init.zeros_(m.bias)
            else:
                nn.init.ones_(m.weight)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, (nn.ReLU, nn.ReLU6, nn.Upsample, Parameter, nn.AdaptiveAvgPool2d, nn.Sigmoid)):
            pass
        else:
            m.initialize()

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

    def initialize(self):
        weight_init(self)


class RF2B(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RF2B, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1)
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0))
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, (1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, (3, 1), padding=(1, 0))
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, (1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, (3, 1), padding=(1, 0))
        )

        self.branch4 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, (1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, (3, 1), padding=(1, 0))
        )
        self.conv = nn.Conv2d(in_channel, out_channel, 1)

        self.conv_cat = nn.Conv2d(out_channel*4, out_channel, 3, padding=1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(self.conv(x) + x1)
        x3 = self.branch3(self.conv(x) + x2)
        x4 = self.branch4(self.conv(x) + x3)
        x_cat = self.conv_cat(torch.cat((x1, x2, x3, x4), dim=1))

        x = self.relu(x0 + x_cat)
        return x

    def initialize(self):
        weight_init(self)


class Rblock(nn.Module):
    def __init__(self,inplanes,outplanes):
        super(Rblock,self).__init__()
        self.squeeze1 = nn.Sequential(nn.Conv2d(inplanes, outplanes, kernel_size=3,stride=1,dilation=2,padding=2), nn.BatchNorm2d(96), nn.ReLU(inplace=True))
        self.squeeze2 = nn.Sequential(nn.Conv2d(inplanes, outplanes, kernel_size=3,stride=1,padding=1), nn.BatchNorm2d(96), nn.ReLU(inplace=True))
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.convg = nn.Conv2d(192,192,1)
        self.sftmax = nn.Softmax(dim=1)
        self.convAB = nn.Conv2d(192, 96, kernel_size=3, stride=1, padding=1)
        self.bnAB   = nn.BatchNorm2d(96)

    def forward(self, x,z):
        z = self.squeeze1(x) if z is None else self.squeeze1(x+z)
        x = self.squeeze2(x)
        x = torch.cat((x,z),1)
        y = self.convg(self.gap(x))
        x = torch.mul(self.sftmax(y)*y.shape[1],x)
        x = F.relu(self.bnAB(self.convAB(x)),inplace=True)
        return x,z
    def initialize(self):
        weight_init(self)
        print("Rblock module init")