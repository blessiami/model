import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DyConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 n_kernels=4, att_channels=4, att_bias=True):
        super(DyConv, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
                                     padding_mode='zeros')
        self.n_kernels = n_kernels

        self.weight = nn.Parameter(torch.Tensor(n_kernels, out_channels, in_channels // groups, *self.kernel_size))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(n_kernels, out_channels))
        else:
            self.register_parameter('bias', None)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.attention = nn.Sequential(nn.Linear(in_channels, att_channels, bias=att_bias),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(att_channels, n_kernels, bias=att_bias),
                                       nn.ReLU(inplace=True))

        self._init_params()

    def _init_params(self):
        for i in range(self.n_kernels):
            nn.init.kaiming_uniform_(self.weight[i], a=math.sqrt(5))
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, input, temperature=30):
        att = self.avgpool(input)
        att = torch.flatten(att, 1)
        att = self.attention(att)

        o = []
        for i in range(self.n_kernels):
            bias = self.bias[i] if self.bias is not None else self.bias
            x = F.conv2d(input, self.weight[i], bias, self.stride, self.padding, self.dilation, self.groups)
            o.append((att[:, i].unsqueeze(1).unsqueeze(2).unsqueeze(3) * x).unsqueeze(0))

        o = torch.sum(torch.cat(o, dim=0), dim=0)

        return o
