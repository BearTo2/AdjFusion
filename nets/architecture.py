# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.nn.utils.spectral_norm as spectral_norm
from nets.normalization import SPADE


class Ada_SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, dilation=1, ic = 3):
        super().__init__()
        # Attributes
        #self.learned_shortcut = (fin != fout)
        fmiddle = fin

        # create conv layers
        self.pad = nn.ReflectionPad2d(dilation)
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=0, dilation=dilation)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=0, dilation=dilation)
        
        # apply spectral norm if specified
        self.conv_0 = spectral_norm(self.conv_0)
        self.conv_1 = spectral_norm(self.conv_1)
        
        # define normalization layers
        spade_config_str = 'spadesyncbatch3x3'
        
        self.norm_0 = SPADE(spade_config_str, fin, ic, PONO=False)
        self.norm_1 = SPADE(spade_config_str, fmiddle, ic, PONO=False)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg1):
        x_s = self.shortcut(x, seg1)
        
        x = self.norm_0(x, seg1)
        #return x
        x = self.actvn(x)
        x = self.pad(x)
        dx = self.conv_0(x)
        dx = self.conv_1(self.pad(self.actvn(self.norm_1(dx, seg1))))

        out = x_s + dx

        return out

    def shortcut(self, x, seg1):
        x_s = x
        
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


# ResNet block used in pix2pixHD
# We keep the same architecture as pix2pixHD.
class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer, activation=nn.ReLU(False), kernel_size=3):
        super().__init__()

        pw = (kernel_size - 1) // 2
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(pw),
            norm_layer(nn.Conv2d(dim, dim, kernel_size=kernel_size)),
            activation,
            nn.ReflectionPad2d(pw),
            norm_layer(nn.Conv2d(dim, dim, kernel_size=kernel_size))
        )

    def forward(self, x):
        y = self.conv_block(x)
        out = x + y
        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)