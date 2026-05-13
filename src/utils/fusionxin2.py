import torch
import torch.nn as nn



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
        x = self.relu(x)
        return x

class TransBasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=2, stride=2, padding=0, dilation=1, bias=False):
        super(TransBasicConv2d, self).__init__()
        self.Deconv = nn.ConvTranspose2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.Deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)



class HAIM(nn.Module):
    def __init__(self, in_channel):
        super(HAIM, self).__init__()
        self.relu = nn.ReLU(True)


        self.rgb_branch1 = BasicConv2d(in_channel, in_channel//4, 3, padding=1, dilation=1)
        self.rgb_branch2 = BasicConv2d(in_channel, in_channel//4, 3, padding=3, dilation=3)
        self.rgb_branch3 = BasicConv2d(in_channel, in_channel//4, 3, padding=5, dilation=5)
        self.rgb_branch4 = BasicConv2d(in_channel, in_channel//4, 3, padding=7, dilation=7)

        self.d_branch1 = BasicConv2d(in_channel, in_channel//4, 3, padding=1, dilation=1)
        self.d_branch2 = BasicConv2d(in_channel, in_channel//4, 3, padding=3, dilation=3)
        self.d_branch3 = BasicConv2d(in_channel, in_channel//4, 3, padding=5, dilation=5)
        self.d_branch4 = BasicConv2d(in_channel, in_channel//4, 3, padding=7, dilation=7)

        self.rgb_branch1_sa = SpatialAttention()
        self.rgb_branch2_sa = SpatialAttention()
        self.rgb_branch3_sa = SpatialAttention()
        self.rgb_branch4_sa = SpatialAttention()

        self.rgb_branch1_ca = ChannelAttention(in_channel // 4)
        self.rgb_branch2_ca = ChannelAttention(in_channel // 4)
        self.rgb_branch3_ca = ChannelAttention(in_channel // 4)
        self.rgb_branch4_ca = ChannelAttention(in_channel // 4)

        self.r_branch1_sa = SpatialAttention()
        self.r_branch2_sa = SpatialAttention()
        self.r_branch3_sa = SpatialAttention()
        self.r_branch4_sa = SpatialAttention()

        self.r_branch1_ca = ChannelAttention(in_channel // 4)
        self.r_branch2_ca = ChannelAttention(in_channel // 4)
        self.r_branch3_ca = ChannelAttention(in_channel // 4)
        self.r_branch4_ca = ChannelAttention(in_channel // 4)

        self.ca = ChannelAttention(in_channel)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         m.weight.data.normal_(std=0.01)
        #         m.bias.data.fill_(0)

    def forward(self, x_rgb, x_d):
        x1_rgb = self.rgb_branch1(x_rgb)
        x2_rgb = self.rgb_branch2(x_rgb)
        x3_rgb = self.rgb_branch3(x_rgb)
        x4_rgb = self.rgb_branch4(x_rgb)

        x1_d = self.d_branch1(x_d)
        x2_d = self.d_branch2(x_d)
        x3_d = self.d_branch3(x_d)
        x4_d = self.d_branch4(x_d)

        x1_rgb_ca = x1_rgb.mul(self.rgb_branch1_ca(x1_rgb))
        x1_d_sa = x1_d.mul(self.rgb_branch1_sa(x1_rgb_ca))
        x1_d = x1_d + x1_d_sa
        x1_d_ca = x1_d.mul(self.r_branch1_ca(x1_d))
        x1_rgb_sa = x1_rgb.mul(self.r_branch1_sa(x1_d_ca))
        x2_rgb = x2_rgb + x1_rgb_sa

        x2_rgb_ca = x2_rgb.mul(self.rgb_branch2_ca(x2_rgb))
        x2_d_sa = x2_d.mul(self.rgb_branch2_sa(x2_rgb_ca))
        x2_d = x2_d + x2_d_sa
        x2_d_ca = x2_d.mul(self.r_branch2_ca(x2_d))
        x2_rgb_sa = x2_rgb.mul(self.r_branch2_sa(x2_d_ca))
        x3_rgb = x3_rgb + x2_rgb_sa

        x3_rgb_ca = x3_rgb.mul(self.rgb_branch3_ca(x3_rgb))
        x3_d_sa = x3_d.mul(self.rgb_branch3_sa(x3_rgb_ca))
        x3_d = x3_d + x3_d_sa
        x3_d_ca = x3_d.mul(self.r_branch3_ca(x3_d))
        x3_rgb_sa = x3_rgb.mul(self.r_branch3_sa(x3_d_ca))
        x4_rgb = x4_rgb + x3_rgb_sa

        x4_rgb_ca = x4_rgb.mul(self.rgb_branch4_ca(x4_rgb))
        x4_d_sa = x4_d.mul(self.rgb_branch4_sa(x4_rgb_ca))
        x4_d = x4_d + x4_d_sa
        x4_d_ca = x4_d.mul(self.r_branch4_ca(x4_d))
        x4_rgb_sa = x4_rgb.mul(self.r_branch4_sa(x4_d_ca))


        y = torch.cat((x1_rgb_sa, x2_rgb_sa, x3_rgb_sa, x4_rgb_sa), 1)
        y_ca = y.mul(self.ca(y))

        z = y_ca + x_rgb
        # then try z = y_ca + x_rgb + x_d, choose the better performance

        return z


if __name__ == "__main__":
    import os

    model = HAIM(96).cuda()
    model.eval()
    input = torch.rand(2, 96, 128, 128).cuda()
    # dct = torch.rand(2, 48, 128, 128).cuda()
    output = model(input,input)

    print(output.size())
    # print(out3.size())
    # print(output['out'].size())