from torch import nn
import torch


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class FA(nn.Module):
    def __init__(self, channel):
        super(FA, self).__init__()
        # self.conv1 = nn.Conv2d(num_node, num_node, kernel_size=1)
        # self.relu = nn.ReLU(inplace=True)
        # self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)
        self.vector1 = torch.nn.Parameter(torch.rand(1,channel), requires_grad=True)
        self.vector2 = torch.nn.Parameter(torch.rand(1,channel), requires_grad=True)
        # self.vector1 = torch.rand(1, channel)
        # self.vector2 = torch.rand(1, channel)
        self.conv = nn.Conv2d(2 * channel, 4 * channel, kernel_size=3, padding=1, bias=False)

    def forward(self, rgb, fre):
        # h = self.conv1(x.permute(0, 2, 1)).permute(0, 2, 1)
        # h = h - x
        # h = self.relu(self.conv2(h))
        b, c, h, w = rgb.size()
        feature = torch.cat([rgb, fre], dim=1)
        feature = self.conv(feature)
        feature_c = torch.chunk(feature, 4, 1)
        feature1 = feature_c[0].view(b, h * w, c)
        feature2 = feature_c[1].view(b, h * w, c)
        feature3 = feature_c[2].view(b, h * w, c)
        feature4 = feature_c[3].view(b, h * w, c)
        f1 = torch.matmul(feature1, feature2.transpose(-2, -1))
        f2 = torch.matmul(feature3, feature4.transpose(-2, -1))
        final_f1 = torch.matmul(f1, rgb.view(b,c,h*w).view(b,h*w,c)) * self.vector1
        final_f2 = torch.matmul(f2,fre.view(b,c,h*w).view(b,h*w,c)) * self.vector2
        final=(final_f1 + final_f2).view(b,c,h,w)
        return final


if __name__ == "__main__":
    import os

    model = FA(96).cuda()
    model.eval()
    input = torch.rand(2, 96, 128, 128).cuda()
    dct = torch.rand(2, 96, 128, 128).cuda()
    output = model(input, dct)

    print(output.size())
    # print(out3.size())
