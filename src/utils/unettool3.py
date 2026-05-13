import torch
import torch.nn as nn
import torch.nn.functional as F
class UpsampleDeterministic(nn.Module):
    def __init__(self,upscale=2):
        super(UpsampleDeterministic, self).__init__()
        self.upscale = upscale

    def forward(self, x):
        '''
        x: 4-dim tensor. shape is (batch,channel,h,w)
        output: 4-dim tensor. shape is (batch,channel,self.upscale*h,self.upscale*w)
        '''
        return x[:, :, :, None, :, None]\
        .expand(-1, -1, -1, self.upscale, -1, self.upscale)\
        .reshape(x.size(0), x.size(1), x.size(2)\
                 *self.upscale, x.size(3)*self.upscale)



class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32,mid_channels),
            nn.ReLU(inplace=True)
            # nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True)
        )

class SingleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(SingleConv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32,mid_channels),
            nn.ReLU(inplace=True),

        )
class project(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(project, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=1,stride=1, bias=False),
            nn.GroupNorm(32,mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),

        )

  # (0): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
  #   (1): ReLU(inplace=True)
  #   (2): BatchNorm2d(256, eps=1e-05, momentum=0.0003, affine=True, track_running_stats=True)
  #   (3): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
  # )

class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.up = UpsampleDeterministic(upscale=2)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class Up2(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up2, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # self.up = UpsampleDeterministic(upscale=2)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            # nn.Dropout(0.2),
            nn.Dropout(0.5),
            # nn.Dropout(0.2),
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )
