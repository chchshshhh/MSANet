import math

import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        # B, N, C = x.shape
        # x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        # x = x.flatten(2).transpose(1, 2)

        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict


class FeedForward2D(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(FeedForward2D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channel, out_channel, kernel_size=3, padding=2, dilation=2
            ),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class FeedForward2Dg(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(FeedForward2Dg, self).__init__()
        self.dconv1= nn.Conv2d(in_channel, out_channel, 3, 1, 1, bias=True, groups=out_channel)
        self.dconv2 = nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=True, groups=out_channel)

        self.br1 = nn.Sequential(
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.br2=nn.Sequential(
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2, inplace=True),
        )


    def forward(self, x):
        x = self.dconv1(x)+x
        x=self.br1(x)
        x=self.dconv2+x
        x=self.br2(x)
        return x



class GlobalFilter(nn.Module):
    def __init__(self, dim=32, h=80, w=41, fp32fft=True):
        super().__init__()
        self.complex_weight = nn.Parameter(
            torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02
        )
        self.w = w
        self.h = h
        self.fp32fft = fp32fft

    def forward(self, x):
        b, _, a, b = x.size()
        x = x.permute(0, 2, 3, 1).contiguous()

        if self.fp32fft:
            dtype = x.dtype
            x = x.to(torch.float32)

        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm="ortho")

        if self.fp32fft:
            x = x.to(dtype)

        x = x.permute(0, 3, 1, 2).contiguous()

        return x


class FreqBlock(nn.Module):
    def __init__(self, dim, h=80, w=41, fp32fft=True):
        super().__init__()
        self.filter = GlobalFilter(dim, h=h, w=w, fp32fft=fp32fft)
        self.feed_forward = FeedForward2D(in_channel=dim, out_channel=dim)

    def forward(self, x):
        x = x + self.feed_forward(self.filter(x))
        return x


class FreqBlock2(nn.Module):
    def __init__(self, dim, h=80, w=41, fp32fft=True):
        super().__init__()
        self.filter = GlobalFilter(dim, h=h, w=w, fp32fft=fp32fft)
        self.feed_forward = FeedForward2Dg(in_channel=dim, out_channel=dim)

    def forward(self, x):
        x = x + self.feed_forward(self.filter(x))
        return x