import torch.nn as nn
import functools
import math
import torch
from torch.nn.parameter import Parameter

class StyleAttentionLayer(nn.Module):
    def __init__(self, channel):
        super(StyleAttentionLayer, self).__init__()

        self.scale_mean = Parameter(torch.Tensor(channel))
        self.scale_std = Parameter(torch.Tensor(channel))
        self.shift = Parameter(torch.Tensor(channel))

        self.scale_mean.data.fill_(0)
        self.scale_std.data.fill_(0)
        self.shift.data.fill_(0)

        setattr(self.scale_mean, 'affine_gate', True)
        setattr(self.scale_std, 'affine_gate', True)
        setattr(self.shift, 'affine_gate', True)

        self.activation = nn.Sigmoid()

    def calc_mean_std(self, feat, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    def forward(self, x):
        b, c, _, _ = x.size()

        feat_mean, feat_std = self.calc_mean_std(x)
        gate = feat_mean * self.scale_mean[None, :, None, None] + \
                feat_std * self.scale_std[None, :, None, None] + \
                self.shift[None, :, None, None]
        gate = self.activation(gate)

        return x * gate 

class SELayer(nn.Module):
    def __init__(self, channel):
        super(SELayer, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.activation = nn.Sigmoid()

        self.reduction = 16

        self.fc = nn.Sequential(
                nn.Linear(channel, channel // self.reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // self.reduction, channel),
        )

    def forward(self, x):
        b, c, _, _ = x.size()

        avg_y = self.avgpool(x).view(b, c)

        gate = self.fc(avg_y).view(b, c, 1, 1)
        gate = self.activation(gate)

        return x * gate 
