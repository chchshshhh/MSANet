import torch
from torch import nn

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias
    )


class FB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, bias=False, act=nn.ReLU(True)):
        super(FB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if i == 0:
                modules_body.append(act)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class Fmoudle(nn.Module):
    def __init__(self, channel):
        super(Fmoudle, self).__init__()
        # modules_body = []
        # for i in range(2):
        #     modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        #     if i == 0:
        #         modules_body.append(act)
        # self.body = nn.Sequential(*modules_body)
        self.channel=channel
        self.fusion=FB(default_conv,2*self.channel,1)

    def forward(self, x,y):
        xy=torch.cat([x,y],dim=1)

        f = self.fusion(xy)
        x_f, y_f = torch.split(f,self.channel, 1)
        x=x+x_f
        y=y+y_f
        return x,y



