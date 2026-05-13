import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import upsample
import torch.nn.functional as F
from src.fratt import *

class MultiSpectralAttentionPooling(torch.nn.Module):
    def __init__(self, channel, dct_h, dct_w, reduction = 16, freq_sel_method = 'top16'):
        super(MultiSpectralAttentionPooling, self).__init__()
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]
        # make the frequencies in different sizes are identical to a 7x7 frequency space
        # eg, (2,2) in 14x14 is identical to (1,1) in 7x7

        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n,c,h,w = x.shape
        x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
            # If you have concerns about one-line-change, don't worry.   :)
            # In the ImageNet models, this line will never be triggered.
            # This is for compatibility in instance segmentation and object detection.
        y = self.dct_layer(x_pooled).view(n, c, 1, 1)


        return y


class TGMandTRM(nn.Module):
    def __init__(self, h, C=96):
        super(TGMandTRM, self).__init__()
        self.rank = 64
        self.ps = [1, 1, 1, 1]
        self.h = h
        self.c=C
        conv1_1, conv1_2, conv1_3 = self.ConvGeneration(self.rank, h)

        self.conv1_1 = conv1_1
        self.conv1_2 = conv1_2
        self.conv1_3 = conv1_3

        # self.lam = torch.ones(self.rank, requires_grad=True)
        self.lam = torch.nn.Parameter(torch.ones(self.rank,requires_grad=True))

        self.pool = nn.AdaptiveAvgPool2d(self.ps[0])
        # self.pool=MultiSpectralAttentionPooling(C,h,h)
        # self.pool=MultiSpectralAttentionPooling(C,h,h)
        self.poolf=MultiSpectralAttentionPooling(C,h,h)

        self.fusion = nn.Sequential(
            nn.Conv2d(2*C, C, 1, padding=0, bias=False),
            nn.BatchNorm2d(C),
            # nn.Sigmoid(),
            nn.ReLU(True),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out')


    def forward(self, x):
        b, c, height, width = x.size()
        C = self.poolf(x)
        H = self.pool(x.permute(0, 3, 1, 2).contiguous())
        W = self.pool(x.permute(0, 2, 3, 1).contiguous())
        self.lam.data = F.softmax(self.lam,-1)
        lam = torch.chunk(self.lam, dim=0, chunks=self.rank)
        list = []
        for i in range(0, self.rank):
            list.append(lam[i]*self.TukerReconstruction(b, self.h , self.ps[0], self.conv1_1[i](C), self.conv1_2[i](H), self.conv1_3[i](W)))
        tensor1 = sum(list)
        tensor1 = torch.cat((x , F.relu_(x * tensor1)), 1)
        tensor1=self.fusion(tensor1)
        return tensor1

    def ConvGeneration(self, rank, h):
        conv1 = []
        n = 1
        for _ in range(0, rank):
            conv1.append(nn.Sequential(
                nn.Conv2d(self.c, self.c // n, kernel_size=1, bias=False),
                nn.Sigmoid(),
            ))
        conv1 = nn.ModuleList(conv1)

        conv2 = []
        for _ in range(0, rank):
            conv2.append(nn.Sequential(
                nn.Conv2d(h, h // n, kernel_size=1, bias=False),
                nn.Sigmoid(),
            ))
        conv2 = nn.ModuleList(conv2)

        conv3 = []
        for _ in range(0, rank):
            conv3.append(nn.Sequential(
                nn.Conv2d(h, h // n, kernel_size=1, bias=False),
                nn.Sigmoid(),
            ))
        conv3 = nn.ModuleList(conv3)

        return conv1, conv2, conv3

    def TukerReconstruction(self, batch_size, h, ps, feat, feat2, feat3):
        b = batch_size
        C = feat.view(b, -1, ps)
        H = feat2.view(b, ps, -1)
        W = feat3.view(b, ps * ps, -1)
        CHW = torch.bmm(torch.bmm(C, H).view(b, -1, ps * ps), W).view(b, -1, h, h)
        return CHW
# model=TGMandTRM(64,C=96)
# a=torch.rand(2,96,64,64)
# out=model(a)
# print(out.size())