# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import re
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.sync_batchnorm import SynchronizedBatchNorm2d
import torch.nn.utils.spectral_norm as spectral_norm

def PositionalNorm2d(x, epsilon=1e-5):
    # x: B*C*W*H normalize in C dim
    mean = x.mean(dim=1, keepdim=True)
    std = x.var(dim=1, keepdim=True).add(epsilon).sqrt()
    output = (x - mean) / std
    return output


class SPADE(nn.Module):
    def __init__(self, config_text, norm_nc=32, label_nc=3, PONO=False):
        super().__init__()

        assert config_text.startswith('spade')
        parsed = re.search('spade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))
        self.pad_type = 'nozero'
        '''
        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'syncbatch':
        '''
        #self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=True)
        '''
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)
        '''
        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        pw = ks // 2
        #if self.pad_type != 'zero':
        self.mlp_shared = nn.Sequential(
            nn.ReflectionPad2d(pw),
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=0),
            nn.ReLU()
        )
        self.pad = nn.ReflectionPad2d(pw)
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=0)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=0)
        '''
        else:
            self.mlp_shared = nn.Sequential(
                    nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
                    nn.ReLU()
                )
            self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
            self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        '''

    def forward(self, x, segmap, similarity_map=None):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)
        
        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        #if self.pad_type != 'zero':
        gamma = self.mlp_gamma(self.pad(actv))
        beta = self.mlp_beta(self.pad(actv))
        '''
        else:
            gamma = self.mlp_gamma(actv)
            beta = self.mlp_beta(actv)
        

        if similarity_map is not None:
            similarity_map = F.interpolate(similarity_map, size=gamma.size()[2:], mode='nearest')
            gamma = gamma * similarity_map
            beta = beta * similarity_map
        '''
        # apply scale and bias
        # print (normalized.shape)
        # print (gamma.shape)
        out = normalized * (1 + gamma) + beta

        return out



def get_nonspade_norm_layer():
    # helper function to get # output channels of the previous layer
    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)

    # this function will be returned
    def add_norm_layer(layer):
        
        layer = spectral_norm(layer)

        # remove bias in the previous layer, which is meaningless
        # since it has no effect after normalization
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)

        norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)

        return nn.Sequential(layer, norm_layer)

    return add_norm_layer

