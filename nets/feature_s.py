
import cv2
import kornia
import numpy
import torchvision.models as models
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import random
from nets.normalization import get_nonspade_norm_layer
from nets.architecture import Ada_SPADEResnetBlock as Ada_SPADEResnetBlock


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class predictor(nn.Module):
    def __init__(self, ndf=8):
        super(predictor, self).__init__()
        self.Dense = nn.Sequential(
            nn.Linear(ndf*24, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        #print(x.shape)
        return self.Dense(x)


class Control_Fusion_module(nn.Module):
    '''
    基于注意力的自适应特征聚合 Fusion_Module
    '''

    def __init__(self, channels=64, r=4):
        super(Control_Fusion_module, self).__init__()
        inter_channels = int(channels // r)

        self.apt = nn.Sequential(
            nn.Conv2d(channels+1, 1, kernel_size=3, padding=1)
        )
        model = [nn.Conv2d(channels *3 + 1, channels *2, 3, stride=1, padding=1),
                 nn.InstanceNorm2d(channels *2),
                 nn.GELU(),
                 nn.Conv2d(channels *2, channels *2, 3, stride=1, padding=1),
                 nn.InstanceNorm2d(channels *2),
                 nn.GELU(),
                 nn.Conv2d(channels *2, channels, 3, stride=1, padding=1),
                 nn.Sigmoid()
                 ]
        self.M_fuse = nn.Sequential(*model)
        self.channel_agg = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=1, stride=1, padding=0),
            LayerNorm2d(channels),
            nn.ReLU(inplace=True),
            )

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=32),
            LayerNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=32),
            LayerNorm2d(channels), 
            nn.ReLU(inplace=True),            
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=32),
            LayerNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2, global_alpha):
        _, c, _, _ = x1.shape
        input = torch.cat([x1, x2], dim=1)
        input = self.channel_agg(input) ## 进行特征压缩 因为只计算一个特征的权重
        local_att = self.local_att(input)  ## 局部注意力 即spatial attention
        M = self.M_fuse(torch.cat([global_alpha,x1,x2,local_att], dim=1))
        xo = M.mul(x1) + (1 - M).mul(x2) ## fusion results ## 特征聚合
        return xo


class Fusion_module(nn.Module):
    '''
    基于注意力的自适应特征聚合 Fusion_Module
    '''

    def __init__(self, channels=64, r=4):
        super(Fusion_module, self).__init__()
        inter_channels = int(channels // r)

        self.Recalibrate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2 * channels, 2 * inter_channels, kernel_size=1, stride=1, padding=0),
            LayerNorm2d(2 * inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * inter_channels, 2 * channels, kernel_size=1, stride=1, padding=0),
            LayerNorm2d(2 * channels),
            nn.Sigmoid(),
        )

        self.channel_agg = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=1, stride=1, padding=0),
            LayerNorm2d(channels),
            nn.ReLU(inplace=True),
            )

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            LayerNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            LayerNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            LayerNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            LayerNorm2d(channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        _, c, _, _ = x1.shape
        input = torch.cat([x1, x2], dim=1)
        recal_w = self.Recalibrate(input)
        recal_input = recal_w * input ## 先对特征进行一步自校正
        recal_input = recal_input + input
        x1, x2 = torch.split(recal_input, c, dim =1)
        agg_input = self.channel_agg(recal_input) ## 进行特征压缩 因为只计算一个特征的权重
        local_w = self.local_att(agg_input)  ## 局部注意力 即spatial attention
        global_w = self.global_att(agg_input) ## 全局注意力 即channel attention
        w = self.sigmoid(local_w * global_w) ## 计算特征x1的权重 
        xo = w * x1 + (1 - w) * x2 ## fusion results ## 特征聚合
        return xo


class SDFM_control(nn.Module):
    def __init__(self, in_C, out_C):
        super(SDFM_control, self).__init__()
        '''
        self.RGBobj1_1 = BBasicConv2d(in_C, out_C, 3, 1, 1)
        self.RGBobj1_2 = BBasicConv2d(out_C, out_C, 3, 1, 1)        
        self.RGBspr = BBasicConv2d(out_C, out_C, 3, 1, 1)        

        self.Infobj1_1 = BBasicConv2d(in_C, out_C, 3, 1, 1)
        self.Infobj1_2 = BBasicConv2d(out_C, out_C, 3, 1, 1)        
        self.Infspr = BBasicConv2d(out_C, out_C, 3, 1, 1) 
        '''
        self.obj_fuse = Control_Fusion_module(channels=out_C)  
        

    def forward(self, rgb, depth, global_alpha):
        '''
        rgb_sum = self.RGBobj1_2(self.RGBobj1_1(rgb))
        rgb_obj = self.RGBspr(rgb_sum)
        Inf_sum = self.Infobj1_2(self.Infobj1_1(depth))
        Inf_obj = self.Infspr(Inf_sum)
        '''
        out = self.obj_fuse(rgb, depth, global_alpha)
        return out
        

class SDFM(nn.Module):
    def __init__(self, in_C, out_C):
        super(SDFM, self).__init__()
        '''
        self.RGBobj1_1 = BBasicConv2d(in_C, out_C, 3, 1, 1)
        self.RGBobj1_2 = BBasicConv2d(out_C, out_C, 3, 1, 1)        
        self.RGBspr = BBasicConv2d(out_C, out_C, 3, 1, 1)        

        self.Infobj1_1 = BBasicConv2d(in_C, out_C, 3, 1, 1)
        self.Infobj1_2 = BBasicConv2d(out_C, out_C, 3, 1, 1)        
        self.Infspr = BBasicConv2d(out_C, out_C, 3, 1, 1) 
        '''
        self.obj_fuse = Fusion_module(channels=out_C)  
        

    def forward(self, rgb, depth):
        '''
        rgb_sum = self.RGBobj1_2(self.RGBobj1_1(rgb))
        rgb_obj = self.RGBspr(rgb_sum)
        Inf_sum = self.Infobj1_2(self.Infobj1_1(depth))
        Inf_obj = self.Infspr(Inf_sum)
        '''
        out = self.obj_fuse(rgb, depth)
        return out
        
        
class SIM(nn.Module):

    def __init__(self, norm_nc, label_nc, nhidden=64):

        super().__init__()

        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False, track_running_stats=False)
        # The dimension of the intermediate embedding space. Yes, hardcoded.
        # nhidden = 128
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Sequential(
            nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1), 
            nn.Sigmoid()
        )
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.ln = LayerNorm2d(norm_nc)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def forward(self, x, segmap):
        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)
        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = self.up(segmap)
        actv = self.mlp_shared(segmap)
        #actv = segmap
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        # apply scale and bias
        out = self.ln(normalized * (1 + gamma)) + beta

        return out


class ATN(nn.Module):
#   [(x,mask),(y,mask)]=>weight
    def __init__(self, ndf=8):
        super(ATN, self).__init__()
        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        
        self.predictor = predictor(ndf=8)
        
        resnet_raw_model1 = models.resnet34(pretrained=True)
        resnet_raw_model2 = models.resnet34(pretrained=True)
        resnet_raw_model3 = models.resnet34(pretrained=True)
        self.mask_deep_layer1 = nn.Sequential(nn.Conv2d(ndf * 4, ndf * 8, kw, stride=2, padding=1))
        self.mask_deep_bn1 = resnet_raw_model1.bn1
        self.mask_deep_relu = resnet_raw_model1.relu
        self.mask_deep_maxpool = resnet_raw_model1.maxpool
        self.mask_deep_layer2 = resnet_raw_model1.layer1
        
        self.x_deep_layer1 = nn.Sequential(nn.Conv2d(ndf * 4, ndf * 8, kw, stride=2, padding=1))
        self.x_deep_bn1 = resnet_raw_model2.bn1
        self.x_deep_relu = resnet_raw_model2.relu
        self.x_deep_maxpool = resnet_raw_model2.maxpool
        self.x_deep_layer2 = resnet_raw_model2.layer1
        
        self.y_deep_layer1 = nn.Sequential(nn.Conv2d(ndf * 4, ndf * 8, kw, stride=2, padding=1))
        self.y_deep_bn1 = resnet_raw_model3.bn1
        self.y_deep_relu = resnet_raw_model3.relu
        self.y_deep_maxpool = resnet_raw_model3.maxpool
        self.y_deep_layer2 = resnet_raw_model3.layer1
        
        self.x_mark_fus_2 = SDFM(ndf*8, 64)
        self.x_mark_fus_1 = SDFM(ndf*8, 64)
        self.x_mark_fus = SDFM(32, 32)
        
        self.y_mark_fus_2 = SDFM(ndf*8, 64)
        self.y_mark_fus_1 = SDFM(ndf*8, 64)
        self.y_mark_fus = SDFM(32, 32)
        
        self.x_y_fus_2 = SDFM_control(ndf*8, 64)
        self.x_y_fus_1 = SDFM_control(ndf*8, 64)
        self.x_y_fus = SDFM_control(32, 32)
        
        self.local_att = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32),
            nn.BatchNorm2d(32), 
            nn.ReLU(inplace=True),            
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32),
            nn.BatchNorm2d(32), 
        )
        self.apt = nn.Sequential(
            nn.Conv2d(33, 1, kernel_size=3, padding=1)
        )
        self.SIM1 = SIM(norm_nc=32, label_nc=64, nhidden=32)
        self.SIM2 = SIM(norm_nc=64, label_nc=64, nhidden=64)
        
        model = [nn.Conv2d(ndf * 8 + 1, ndf * 8, kw, stride=1, padding=pw),
                 nn.InstanceNorm2d(ndf * 8),
                 nn.GELU(),
                 nn.Conv2d(ndf * 8, ndf * 8, kw, stride=1, padding=pw),
                 nn.InstanceNorm2d(ndf * 8),
                 nn.GELU(),
                 nn.Conv2d(ndf * 8, ndf * 4, kw, stride=1, padding=pw),
                 nn.Sigmoid()
                 ]
        self.M_fuse = nn.Sequential(*model)

        self.max = nn.AdaptiveMaxPool2d((1,1))
        self.MAX = 0
        self.MIN = 1
        self.tot = 0
        self.cnt = 0
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        #self.feat_fuse = nn.Conv2d(ndf*8, ndf*4, (3, 3), (1, 1), 1)

    def forward(self, x, y, mask_x, mask_y, alpha, name):
        #mask = self.encode(mask)
        # x, y = x * self.mask_apt(mask), y * self.mask_apt(mask)
        # x, y = self.SCAM_cross(x, y)
        mask_x1, mask_y1 = self.mask_deep_layer1(mask_x), self.mask_deep_layer1(mask_y)
        mask_x1, mask_y1 = self.mask_deep_relu(self.mask_deep_bn1(mask_x1)), self.mask_deep_relu(self.mask_deep_bn1(mask_y1)) # 8
        mask_x2, mask_y2 = self.mask_deep_layer2(self.mask_deep_maxpool(mask_x1)), self.mask_deep_layer2(self.mask_deep_maxpool(mask_y1)) # 16
        
        x1, y1 = self.x_deep_layer1(x), self.y_deep_layer1(y)
        x1, y1 = self.x_deep_relu(self.x_deep_bn1(x1)), self.y_deep_relu(self.y_deep_bn1(y1))
        x2, y2 = self.x_deep_layer2(self.x_deep_maxpool(x1)), self.y_deep_layer2(self.y_deep_maxpool(y1))
        #print(x1.shape, x2.shape, x3.shape)
        flag1 = abs(alpha+1)
        if flag1 <= 0.01:
            '''
            mask_x_var, mask_x_mean = torch.std_mean(mask_x3, dim=[2, 3], keepdim=False)
            mask_y_var, mask_y_mean = torch.std_mean(mask_y3, dim=[2, 3], keepdim=False)
            mask_x_max, mask_y_max = self.max(mask_x3).squeeze(2).squeeze(2), self.max(mask_y3).squeeze(2).squeeze(2)
            x_var, x_mean = torch.std_mean(x3, dim=[2, 3], keepdim=False)
            y_var, y_mean = torch.std_mean(y3, dim=[2, 3], keepdim=False)
            x_max, y_max = self.max(x3).squeeze(2).squeeze(2), self.max(y3).squeeze(2).squeeze(2)
            '''
            x_var, x_mean = torch.std_mean(x, dim=[2, 3], keepdim=False)
            y_var, y_mean = torch.std_mean(y, dim=[2, 3], keepdim=False)
            x_max, y_max = self.max(x).squeeze(2).squeeze(2), self.max(y).squeeze(2).squeeze(2)
            #alpha = self.predictor(torch.cat([x_var, x_mean, x_max, mask_x_var, mask_x_mean, mask_x_max], dim=1), torch.cat([y_var, y_mean, y_max, mask_y_var, mask_y_mean, mask_y_max],dim=1))
            alpha = self.predictor(torch.cat([x_var, x_mean, x_max, y_var, y_mean, y_max], dim=1))
            
            alpha = alpha.view(alpha.shape[0], alpha.shape[1], 1, 1)
            global_alpha = alpha.clone().detach()
            #print(global_alpha)
            x_mark_2, y_mark_2 = self.x_mark_fus_2(x2,mask_x2), self.y_mark_fus_2(y2,mask_y2)
            alpha_2 = alpha.expand_as(x2[:,0:1,:,:])
            x_y_fus_2 = self.x_y_fus_2(x_mark_2, y_mark_2, alpha_2.detach())
            x_mark_1, y_mark_1 = self.x_mark_fus_1(x1,mask_x1), self.y_mark_fus_1(y1,mask_y1)
            alpha_1 = alpha.expand_as(x1[:,0:1,:,:])
            x_y_fus_1 = self.x_y_fus_1(x_mark_1, y_mark_1, alpha_1.detach())
            x_mark_1, y_mark_1, x_mark_2, y_mark_2, x_mark_3, y_mark_3 = '', '', '', '', '', ''
            x_mark, y_mark = self.x_mark_fus(x,mask_x), self.y_mark_fus(y,mask_y)
            alpha = alpha.expand_as(x[:,0:1,:,:])
            x_y_fus = self.x_y_fus(x_mark, y_mark, alpha.detach())
            seg_1 = self.SIM2(x_y_fus_1, x_y_fus_2)
            seg_feat = self.SIM1(x_y_fus, seg_1)
            x_y_fus, x_y_fus_1, x_y_fus_2, x_y_fus_3 = '', '', '', ''
            local_alpha = self.local_att(seg_feat)
            local_alpha = self.sigmoid(self.apt(torch.cat([alpha,local_alpha],dim=1)))
            #alpha = alpha.expand_as(x[:,0:1,:,:])
            #print(alpha.shape, local_alpha.shape)

        else:
            alpha = torch.tensor(alpha).view(1,1).cuda()
            alpha = alpha.float()
            
            #print(alpha.shape)
            alpha = alpha.view(alpha.shape[0], alpha.shape[1], 1, 1)
            global_alpha = alpha.clone().detach()
            
            x_mark_2, y_mark_2 = self.x_mark_fus_2(x2,mask_x2), self.y_mark_fus_2(y2,mask_y2)
            alpha_2 = alpha.expand_as(x2[:,0:1,:,:])
            x_y_fus_2 = self.x_y_fus_2(x_mark_2, y_mark_2, alpha_2.detach())
            x_mark_1, y_mark_1 = self.x_mark_fus_1(x1,mask_x1), self.y_mark_fus_1(y1,mask_y1)
            alpha_1 = alpha.expand_as(x1[:,0:1,:,:])
            x_y_fus_1 = self.x_y_fus_1(x_mark_1, y_mark_1, alpha_1.detach())
            x_mark_1, y_mark_1, x_mark_2, y_mark_2, x_mark_3, y_mark_3 = '', '', '', '', '', ''
            x_mark, y_mark = self.x_mark_fus(x,mask_x), self.y_mark_fus(y,mask_y)
            alpha = alpha.expand_as(x[:,0:1,:,:])
            x_y_fus = self.x_y_fus(x_mark, y_mark, alpha.detach())
            seg_1 = self.SIM2(x_y_fus_1, x_y_fus_2)
            seg_feat = self.SIM1(x_y_fus, seg_1)
            x_y_fus, x_y_fus_1, x_y_fus_2, x_y_fus_3 = '', '', '', ''
            local_alpha = self.local_att(seg_feat)
            local_alpha = self.sigmoid(self.apt(torch.cat([alpha,local_alpha],dim=1)))
            #alpha = alpha.expand_as(x[:,0:1,:,:])
            #print(alpha.shape, local_alpha.shape)
        
        M = self.M_fuse(torch.cat([local_alpha, x, y], dim=1))
        return M.mul(x) + (1 - M).mul(y), global_alpha


class Generator(nn.Module):
    def __init__(self):
        # TODO: kernel=4, concat noise, or change architecture to vgg feature pyramid
        super().__init__()
        self.cnt = 0
        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ndf = 8
        nf = 8
        
        model = [nn.Conv2d(1, ndf, kw, stride=1, padding=pw),
                 LayerNorm2d(ndf),
                 nn.GELU(),
                 nn.Conv2d(ndf, ndf*2, kw, stride=1, padding=pw),
                 LayerNorm2d(ndf*2),
                 nn.GELU(),
                 nn.Conv2d(ndf*2, ndf*4, kw, stride=1, padding=pw),
                 LayerNorm2d(ndf*4),
                 nn.GELU(),
                 nn.Conv2d(ndf*4, ndf*4, kw, stride=1, padding=pw),
                 LayerNorm2d(ndf*4),
                 nn.GELU()]
        self.y_encode = nn.Sequential(*model)
        
        model = [nn.Conv2d(3, ndf*2, kw, stride=1, padding=pw),
                 LayerNorm2d(ndf*2),
                 nn.GELU(),
                 nn.Conv2d(ndf*2, ndf*8, kw, stride=1, padding=pw),
                 LayerNorm2d(ndf*8),
                 nn.GELU(),
                 nn.Conv2d(ndf*8, ndf*16, kw, stride=1, padding=pw),
                 LayerNorm2d(ndf*16),
                 nn.GELU(),
                 nn.Conv2d(ndf*16, ndf*8, kw, stride=1, padding=pw),
                 LayerNorm2d(ndf*8),
                 nn.GELU(),
                 nn.Conv2d(ndf*8, ndf*4, kw, stride=1, padding=pw),
                 LayerNorm2d(ndf*4),
                 nn.GELU()]
        self.mark_encode = nn.Sequential(*model)
        
        model = [nn.Conv2d(3, ndf, kw, stride=1, padding=pw),
                 LayerNorm2d(ndf),
                 nn.GELU(),
                 nn.Conv2d(ndf * 1, ndf * 2, kw, stride=1, padding=pw),
                 LayerNorm2d(ndf * 2),
                 nn.GELU(),
                 nn.Conv2d(ndf * 2, ndf * 4, kw, stride=1, padding=pw),
                 LayerNorm2d(ndf * 4),
                 nn.GELU(),
                 nn.Conv2d(ndf * 4, ndf * 4, kw, stride=1, padding=pw),
                 LayerNorm2d(ndf * 4),
                 nn.GELU()]
        self.ir_layer = nn.Sequential(*model)
        model = [nn.Conv2d(3, ndf, kw, stride=1, padding=pw),
                 LayerNorm2d(ndf),
                 nn.GELU(),
                 nn.Conv2d(ndf * 1, ndf * 2, kw, stride=1, padding=pw),
                 LayerNorm2d(ndf * 2),
                 nn.GELU(),
                 nn.Conv2d(ndf * 2, ndf * 4, kw, stride=1, padding=pw),
                 LayerNorm2d(ndf * 4),
                 nn.GELU(),
                 nn.Conv2d(ndf * 4, ndf * 4, kw, stride=1, padding=pw),
                 LayerNorm2d(ndf * 4),
                 nn.GELU()]
        self.vi_layer = nn.Sequential(*model)
        
        self.actvn = nn.GELU()
        
        self.CNN_res_deeper2 = Ada_SPADEResnetBlock(4 * nf, 4 * nf, dilation=4, ic=3)
        self.CNN_res_degridding0 = nn.Sequential(nn.Conv2d(ndf * 4, ndf * 4, 3, stride=1, padding=2, dilation=2),
                                             LayerNorm2d(ndf*4)
        )
        
        #self.inv_ir = Inv_Generator()
        #self.inv_vi = Inv_Generator()
        
        self.model_M = ATN(ndf=ndf)
        
        model = [nn.Conv2d(ndf * 8, ndf * 4, kw, stride=1, padding=pw),
                 nn.InstanceNorm2d(ndf * 4),
                 nn.LeakyReLU(0.2, False),
                 nn.Conv2d(ndf * 4, ndf * 4, kw, stride=1, padding=pw),
                 nn.InstanceNorm2d(ndf * 4),
                 nn.LeakyReLU(0.2, False),
                 nn.Conv2d(ndf * 4, ndf * 4, kw, stride=1, padding=pw),
                 nn.Sigmoid()]
        self.feat_fuse = nn.Sequential(*model)
        self.dense = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ndf * 4 // (1 << i), ndf * 4 // (1 << i), (3, 3), (1, 1), 1),
                LayerNorm2d(ndf * 4 // (1 << i)),
                nn.Conv2d(ndf * 4 // (1 << i), ndf * 4 // (1 << (i + 1)), (3, 3), (1, 1), 1),
                LayerNorm2d(ndf * 4 // (1 << (i + 1))),
                nn.GELU()
            ) for i in range(2)
        ])

        self.y_gen = nn.Sequential(
            nn.Conv2d(ndf, 1, (3, 3), (1, 1), 1),
            nn.Sigmoid(),
        )
        
        self.mark_fuse = nn.Conv2d(ndf*12, ndf*4, (3, 3), (1, 1), 1)
        #self.AT_mark = nn.Conv2d(ndf*4, ndf*4, (3, 3), (1, 1), 1)
        #self.apt = nn.Conv2d(ndf*4, ndf*4, (3, 3), (1, 1), 1)
        self.cbcr_dense = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ndf * 4 // (1 << i), ndf * 4 // (1 << i), (3, 3), (1, 1), 1),
                LayerNorm2d(ndf * 4 // (1 << i)),
                nn.Conv2d(ndf * 4 // (1 << i), ndf * 4 // (1 << (i + 1)), (3, 3), (1, 1), 1),
                LayerNorm2d(ndf * 4 // (1 << (i + 1))),
                nn.GELU()
            ) for i in range(2)
        ])
        self.cbcr_gen = nn.Sequential(
            nn.Conv2d(ndf, 2, (3, 3), (1, 1), 1),
            nn.Sigmoid(),
        )
        

    def generate_mask(self, shape, left, top, window_size, mask_ratio):
        N, D, H, W = shape
        window_number = left.shape[0]

        mask = torch.ones(H*W).cuda()

        for l in left:
            r = l+window_size-1
            for t in top:
                b = t + window_size-1
                num = int(window_size**2 * mask_ratio)
                #print(num)
                d = torch.randperm(window_size**2, device=mask.device)[:num]
                y = d % 7
                d = d // 7
                d = (d + t -1) * W + l + y
                mask[d] = 0
        mask = mask.repeat(N, D, 1)
        mask = mask.reshape(N, D, H, W)
        return mask

    def forward(self, ir=None, vi=None, ir_mark=None, vi_mark=None, alpha=0.5, beta=0.5, cnt=0):
        '''
        B,C,H,W = ir.shape
        patch_size = 14
        w, h = ir.shape[-2] - ir.shape[-2] % patch_size, ir.shape[-1] - ir.shape[-1] % patch_size
        ir_x, vi_x, en_x = ir[:, :, :w, :h], vi[:, :, :w, :h], en[:, :, :w, :h]
        w_featmap = ir.shape[-2] // patch_size #73
        h_featmap = ir.shape[-1] // patch_size #54
        ir_x, vi_x, en_x = self.dinov2_vits14.forward_features(ir_x), self.dinov2_vits14.forward_features(vi_x), self.dinov2_vits14.forward_features(en_x)
        ir_x, vi_x, en_x = ir_x['x_norm_patchtokens'].unsqueeze(1), vi_x['x_norm_patchtokens'].unsqueeze(1), en_x['x_norm_patchtokens'].unsqueeze(1)
        ir_x, vi_x, en_x = ir_x.transpose(1,3), vi_x.transpose(1,3), en_x.transpose(1,3)
        nh = ir_x.shape[1]
        ir_x, vi_x, en_x = ir_x[:, :, :, 0].reshape(B, nh, -1), vi_x[:, :, :, 0].reshape(B, nh, -1), en_x[:, :, :, 0].reshape(B, nh, -1)
        ir_x, vi_x, en_x = ir_x.reshape(B, nh, w_featmap, h_featmap), vi_x.reshape(B, nh, w_featmap, h_featmap), en_x.reshape(B, nh, w_featmap, h_featmap)
        if cnt % 5000 == 100000:
            self.See_it(ir_x, 'ir_x_res' + str(cnt))
            self.See_it(vi_x, 'vi_x_res' + str(cnt))
            self.See_it(en_x, 'en_x_res' + str(cnt))
        ir_x, vi_x, en_x = self.up(ir_x), self.up(vi_x), self.up(en_x)
        
        ir_x = torch.nn.functional.interpolate(torch.cuda.FloatTensor(ir_x), size=(H, W), scale_factor=None,
                                               mode='bicubic', align_corners=None)
        vi_x = torch.nn.functional.interpolate(torch.cuda.FloatTensor(vi_x), size=(H, W), scale_factor=None,
                                               mode='bicubic', align_corners=None)
        en_x = torch.nn.functional.interpolate(torch.cuda.FloatTensor(en_x), size=(H, W), scale_factor=None,
                                               mode='bicubic', align_corners=None)
        #print(ir_mark.shape, vi_mark.shape)
        
        ir_x = self.res_deeper2(ir_x, ir_mark)
        ir_x = self.res_degridding0(ir_x)
        vi_x = self.res_deeper2(vi_x, vi_mark)
        vi_x = self.res_degridding0(vi_x)
        en_x = self.res_deeper2(en_x, vi_mark)
        en_x = self.res_degridding0(en_x)
        '''
        
        ir_cnn, vi_cnn = self.ir_layer(ir), self.vi_layer(vi)
        ir_cnn, vi_cnn = self.CNN_res_deeper2(ir_cnn, ir_mark), self.CNN_res_deeper2(vi_cnn, vi_mark)
        ir_cnn, vi_cnn = self.CNN_res_degridding0(ir_cnn), self.CNN_res_degridding0(vi_cnn)
        #ir_x, vi_x, en_x = self.mer_ir(ir_x, ir_cnn, ), self.mer_vi(vi_x, vi_cnn, ), self.mer_en(en_x, en_cnn, )
        if cnt % 5000 == 10000:
            self.See_it(ir_x, 'ir_x_' + str(cnt))
            self.See_it(vi_x, 'vi_x_' + str(cnt))
        #print(vi_x.shape, ir_x.shape, en_x.shape, vi_mark.shape, ir_mark.shape)
        ir_mark = self.mark_encode(ir_mark)
        vi_mark = self.mark_encode(vi_mark)
        #vi_x = self.model_S(vi_cnn, en_cnn, vi_mark, vi_mark, beta, 'S')
        feat, global_alpha = self.model_M(ir_cnn, vi_cnn, ir_mark, vi_mark, alpha, 'M')
        #ir_cnn, vi_cnn = '', ''
        # feat = torch.cat([feat, invariant_feat], dim=1)
        feat = self.dense[0](feat)
        feat = self.dense[1](feat)
        
        res_y = self.y_gen(feat)
        
        feat = self.y_encode(res_y.detach())
        S = self.feat_fuse(torch.cat([feat, vi_cnn.detach()],dim=1))
        feat = S.mul(feat) + (1 - S).mul(vi_cnn)
        vi_x = ''
        
        #feat = self.apt(feat) * self.AT_mark(max_mark) + feat
        
        for i in range(2):
            feat = self.cbcr_dense[i](feat)
        
        res_cbcr = self.cbcr_gen(feat)
        
        feat = ''
        
        return res_y, res_cbcr, global_alpha

    def See_it(self, a, layer_name):
        images_per_row = 8
        layer_activation = a.cpu().detach().numpy()
        n_feature = layer_activation.shape[1]  # 每层输出的特征层数
        size1 = layer_activation.shape[-2]  # 每层的特征大小
        size2 = layer_activation.shape[-1]  # 每层的特征大小
        # print(layer_activation.shape, size1, size2, n_feature)
        n_cols = n_feature // images_per_row  # 特征图平铺的行数
        display_grid = np.zeros((size1 * n_cols, images_per_row * size2))  # 每层图片大小
        for col in range(n_cols):  # 行扫描
            for row in range(images_per_row):  # 平铺每行
                # print(layer_activation.shape)
                # print(col*images_per_row+row)
                channel_image = layer_activation[0, col * images_per_row + row, :, :]  # 写入col*images_per_row+row特征层
                channel_image -= channel_image.mean()  # 标准化处理，增加可视化效果
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                # print(channel_image.shape)
                # print(display_grid[col*size1:(col+1)*size1, row*size2:(row+1)*size2].shape)
                display_grid[col * size1:(col + 1) * size1, row * size2:(row + 1) * size2] = channel_image  # 写入大图中
        scale = 1. / max(size1, size2)  # 每组图缩放系数
        plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
        # plt.title(layer_name)
        plt.grid(False)
        plt.imsave('logs/m3fd/DINO_sam_mer/img/' + layer_name + '.jpg', display_grid)
        display_grid, layer_activation,  = '', ''


class Inv_Generator(nn.Module):
    """
    Use to generate infrared or visible images.
    fus -> ir or vi
    """

    def __init__(self, dim1: int = 8, depth1: int = 3, depth2: int = 2):
        super(Inv_Generator, self).__init__()
        self.depth1 = depth1
        self.depth2 = depth2
        
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, dim1, (9, 9), (1, 1), 4),
            nn.LeakyReLU(0.01)
        )

        self.dense1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim1 * (i + 1), dim1, (3, 3), (1, 1), 1),
                nn.InstanceNorm2d(dim1),
                nn.Conv2d(dim1, dim1, (3, 3), (1, 1), 1),
                nn.InstanceNorm2d(dim1),
                nn.LeakyReLU(0.01)
            ) for i in range(depth1)
        ])
        
        self.dense2 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim1 * 4 // (1 << i), dim1 * 4 // (1 << i), (3, 3), (1, 1), 1),
                nn.InstanceNorm2d(dim1 * 4 // (1 << i)),
                nn.Conv2d(dim1 * 4 // (1 << i), dim1 * 4 // (1 << (i + 1)), (3, 3), (1, 1), 1),
                nn.InstanceNorm2d(dim1 * 4 // (1 << (i + 1))),
                nn.LeakyReLU(0.01)
            ) for i in range(depth2)
        ])

        self.gen = nn.Sequential(
            nn.Conv2d(dim1, 3, (3, 3), (1, 1), 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        
        x = self.encoder1(x)
        for i in range(self.depth1):
            t = self.dense1[i](x)
            x = torch.cat([x, t], dim=1)
        
        for i in range(self.depth2):
            x = self.dense2[i](x)
        x = self.gen(x)

        return x


class GaussianBlurConv(nn.Module):
    def __init__(self, channels=3):
        super(GaussianBlurConv, self).__init__()
        self.channels = channels
        self.weight_3 = nn.Parameter(data=self.get_gaussian_kernel(size=3), requires_grad=False)
        self.weight_5 = nn.Parameter(data=self.get_gaussian_kernel(size=5), requires_grad=False)
        self.weight_7 = nn.Parameter(data=self.get_gaussian_kernel(size=7), requires_grad=False)

    def __call__(self, x):
        x_3 = F.conv2d(x, self.weight_3, padding=1, groups=self.channels)
        x_5 = F.conv2d(x, self.weight_5, padding=2, groups=self.channels)
        x_7 = F.conv2d(x, self.weight_7, padding=3, groups=self.channels)
        # print(x_3.shape, x_5.shape, x_7.shape)
        return x_3 + x_5 + x_7

    def get_gaussian_kernel(self, size=3):  # 获取高斯kerner 并转为tensor ，size 可以改变模糊程度
        kernel = cv2.getGaussianKernel(size, 0).dot(cv2.getGaussianKernel(size, 0).T)
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel = np.repeat(kernel, self.channels, axis=0)
        kernel = kernel.cuda()
        kernel = torch.nn.Parameter(data=kernel, requires_grad=False)
        return kernel