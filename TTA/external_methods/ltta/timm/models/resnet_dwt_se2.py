"""PyTorch ResNet

This started as a copy of https://github.com/pytorch/vision 'resnet.py' (BSD-3-Clause) with
additional dropout and dynamic global avg/max pool.

ResNeXt, SE-ResNeXt, SENet, and MXNet Gluon stem/downsample variants, tiered stems added by Ross Wightman

Copyright 2019, Ross Wightman
"""
import math
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

import random

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import CIFAR10_DWT_1_MEAN_STD
from .helpers import build_model_with_cfg, checkpoint_seq
from .layers import PatchEmbed, DropBlock2d, DropPath, AvgPool2dSame, BlurPool2d, GroupNorm, create_attn, get_attn, create_classifier, create_classifier2, trunc_normal_
from .registry import register_model
import dann
import dwt_net as D
import torch.nn.functional as F

## for DWT 
from .vision_transformer import Up_Attention
import torch_dwt as tdwt

__all__ = ['ResNet_DWT_SE2', 'BasicBlock31', 'Bottleneck31', 'SEModule4',  'SEModule5']  # model_registry will add each entrypoint fn to this


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv1', 'classifier': 'fc',
        **kwargs
    }


default_cfgs = {
    # ResNet and Wide ResNet
    'resnet18': _cfg(url='https://download.pytorch.org/models/resnet18-5c106cde.pth'),
    'resnet26': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet26-9aa10e23.pth',
        interpolation='bicubic'),
    'resnet50': _cfg(
        # https://download.pytorch.org/models/resnet50-19c8e357.pth (RESNET)
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_a1_0-14fe96d1.pth',
        #url='https://download.pytorch.org/models/resnet50-19c8e357.pth',
        interpolation='bicubic', crop_pct=0.95),
    'resnet101': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet101_a1h-36d3f2aa.pth',
        interpolation='bicubic', crop_pct=0.95),
    'resnet50_gn': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_gn_a1h2-8fe6c4d0.pth',
        crop_pct=0.94, interpolation='bicubic'),
}

class SEModule5(nn.Module):

    def __init__(self, channels=64, reduction_=1, loss_option=0):
        super(SEModule5, self).__init__()
        
        self.eps = 1e-9
        self.loss_option = loss_option
        reduction = int(reduction_ // 10)
        mean_true = int(reduction_ % 10)

        self.mean_analysis = None
        self.std_analysis = None

        print(f'Reduction: {reduction} GT_OPTION: {mean_true}')

        if mean_true == 0:
            self.gt_lst = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5] # ALL Median
        elif mean_true == 1: 
            self.gt_lst = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] # All Highest
        elif mean_true == 2:
            self.gt_lst = [self.eps, self.eps, self.eps, self.eps, self.eps, self.eps, self.eps, self.eps, self.eps, self.eps, self.eps, self.eps, self.eps, self.eps, self.eps, self.eps] # All Lowest
        elif mean_true == 3:
            self.gt_lst = [1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, self.eps, 1.0, 0.5, 0.5, self.eps, 1.0, self.eps, self.eps, self.eps] # LL is More Important
        elif mean_true == 4:
            self.gt_lst = [self.eps, self.eps, self.eps, self.eps, self.eps, 0.5, 0.5, 1.0, self.eps, 0.5, 0.5, 1.0, self.eps, 1.0, 1.0, 1.0] # HH is More Important
            

        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, (channels * 2), kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        

    def adaptive_std(self, std):
        threshold = 0.9
        valid_std_mask = std >= threshold
        
        std = (std * valid_std_mask) + self.eps
        
        gt_lst = torch.ones((128, 64, 1, 1)).cuda()
        gt_lst = (gt_lst * valid_std_mask) + self.eps

        return std, gt_lst


    def forward(self, x, frequency_index, gt_lst=None,is_mean=None):
        module_input = x
        x = x.mean((2, 3), keepdim=True)
        x = self.fc1(x) # Excitation
        x = self.relu(x)
        x = self.fc2(x) # scaling 
        x = self.sigmoid(x)

        mean = x[:, ::2]  # 모든 배치의 짝수 위치 (평균)
        std = x[:, 1::2]  # 모든 배치의 홀수 위치 (표준편차)

        self.mean_analysis = mean
        self.std_analysis = std

        if self.training:
            if is_mean != None:
                target_mean = is_mean[frequency_index]
            else:
                target_mean = self.gt_lst[frequency_index]
            batch_size = mean.size(0)  
            target_tensor = torch.full((batch_size, 64, 1), target_mean).squeeze(-1).cuda().detach()
            mean_mean = mean.mean(dim=(2,3))
            if self.loss_option % 10 == 0:
                g_loss = (-torch.log(self._gaussian_dist_pdf(mean, std, frequency_index, gt_lst=gt_lst)) / 2).mean()
            else:
                g_loss = (-torch.log(self._gaussian_dist_pdf(mean, std, frequency_index, gt_lst=gt_lst)) / 2).mean() + F.mse_loss(mean_mean,target_tensor)
            #target_tensor = target_tensor.unsqueeze(-1).unsqueeze(-1)
            #stand_value = (mean - target_tensor) / (std)
            #return (module_input * stand_value), [g_loss, mean]
            return (module_input * mean), [g_loss, mean]

        else: # Validation 
            target_mean = self.gt_lst[frequency_index]
            batch_size = mean.size(0)  
            target_tensor = torch.full((batch_size, 64, 1), target_mean).squeeze(-1).cuda().detach()
            # mean_mean = mean.mean(dim=(2,3))
            # target_tensor = target_tensor.unsqueeze(-1).unsqueeze(-1)
            # stand_value = (mean - target_tensor) / (std)
            return (module_input * mean), mean#[mean, std]
        
    
    def _gaussian_dist_pdf(self, data_point, var, freq_idx, gt_lst=None):
        
        if gt_lst == None:
            gt = self.gt_lst[freq_idx]
        elif isinstance(gt_lst, list):
            gt = gt_lst[freq_idx]
        else:
            gt = gt_lst
        var = var.clone() + self.eps
        pdf_value = torch.exp(- (data_point - gt) ** 2.0 / var / 2.0) / torch.sqrt(2.0 * np.pi * var)
        return torch.clamp(pdf_value, min=self.eps)


class SEModule4(nn.Module):

    def __init__(self, channels=64, reduction_=1, loss_option=0):
        super(SEModule4, self).__init__()
        
        self.eps = 1e-9
        self.loss_option = loss_option
        reduction = int(reduction_ // 10)
        mean_true = int(reduction_ % 10)
        
        self.mean_true = mean_true

        self.mean_analysis = None
        self.std_analysis = None
        self.mse_loss = nn.MSELoss()
        print(f'Reduction: {reduction} GT_OPTION: {mean_true}')
        #self.gt_lst = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] # All Highest 
        self.gt_lst = [self.eps, self.eps, self.eps, self.eps, self.eps, self.eps, self.eps, self.eps, self.eps, self.eps, self.eps, self.eps, self.eps, self.eps, self.eps, self.eps] # All Highest            

        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.relu = nn.ReLU(inplace=False) # For Squeeze and Mean, std
        self.fc2 = nn.Conv2d(channels // reduction, (channels * 4), kernel_size=1)
        self.sigmoid = nn.Sigmoid() # For Attention and Uncertainty
        

    def norm(self,input_tensor):
        # 인스턴스 정규화 수행
        b, c, h, w = input_tensor.shape
        input_tensor = input_tensor.view(b, c, -1)

        mean_input = input_tensor.mean(dim=2, keepdim=True)
        std_input = input_tensor.std(dim=2, keepdim=True)

        return ((input_tensor - mean_input) / std_input).view(b, c, h, w)

    def adain(self,input_tensor, mean, std, is_norm=False):
        # 인스턴스 정규화 수행
        b, c, h, w = input_tensor.shape
        
        if is_norm:
            normalized_tensor = self.norm(input_tensor)
            #input_tensor = input_tensor.view(b, c, -1)
            normalized_tensor = normalized_tensor.view(b, c, h, w)


        # 주어진 mean과 std로 조정
        normalized_tensor = input_tensor
        mean = mean.view(b, c, 1, 1)
        std = std.view(b, c, 1, 1)

        adain_output = std * normalized_tensor + mean
        
        return adain_output

    def forward(self, x, frequency_index, gt_lst=None,is_mean=None):
        module_input = x
        mean_gt = x.mean((2, 3), keepdim=True).clone().detach()
        std_gt = x.std((2, 3), keepdim=True).clone().detach()
        #print('SHAPE check {} {}'.format(mean_gt.shape, std_gt.shape))
        #x = self.norm(x) # Normalization 
        x = x.mean((2, 3), keepdim=True)
        x = self.fc1(x) # Excitation
        x = self.relu(x)
        x = self.fc2(x) # scaling 
        #x = self.sigmoid(x)
        #print('SElf. fc2 shape {}'.format(x.shape))

        mean = self.relu(x[:, 0::4]) # 첫 번째 1/4 부분
        std = self.sigmoid(x[:, 1::4]) # 두 번째 1/4 부분
        attention = self.sigmoid(x[:, 2::4])  # 세 번째 1/4 부분
        uncertainty = self.sigmoid(x[:, 3::4])  # 네 번째 1/4 부분

        # attention = self.sigmoid(x[:, 0::2])  # 세 번째 1/4 부분
        # uncertainty = self.sigmoid(x[:, 1::2])  # 네 번째 1/4 부분

        #self.mean_analysis = mean
        #self.std_analysis = std

        if self.training:
            # if self.loss_option % 10 == 0:
            #     g_loss = (-torch.log(self._gaussian_dist_pdf(mean, std, frequency_index, gt_lst=gt_lst)) / 2).mean()
            # return (module_input * mean), [g_loss, mean]

            
            
            # print('MEAN STD Check => {} {}'.format(mean[0][0], std[0][0]))
            # print('MEAN STD GT Check => {} {}'.format(mean_gt[0][0], std_gt[0][0]))
            # print('LOSS Check => {} {}'.format(mean_loss, std_loss))
            
            if self.mean_true == 0:
                g_loss = (-torch.log(self._gaussian_dist_pdf(attention, uncertainty, frequency_index, gt_lst=self.gt_lst)) / 2).mean()
                mean_loss = self.mse_loss(mean, mean_gt)
                std_loss = self.mse_loss(std, std_gt)
                total_loss = g_loss + mean_loss + std_loss
                return (self.adain(module_input, mean=mean, std=std) * attention), [total_loss]
            elif self.mean_true == 1:
                mean_loss = self.mse_loss(mean, mean_gt)
                std_loss = self.mse_loss(std, std_gt)
                total_loss = mean_loss + std_loss
                return (self.adain(module_input, mean=mean, std=std)), [total_loss]
            elif self.mean_true == 2:
                mean_loss = (-torch.log(self._gaussian_dist_pdf(mean, attention, frequency_index, gt_lst=mean_gt)) / 2).mean()
                std_loss = (-torch.log(self._gaussian_dist_pdf(std, uncertainty, frequency_index, gt_lst=mean_gt)) / 2).mean()
                total_loss = mean_loss + std_loss
                return (self.adain(module_input, mean=mean, std=std)), [total_loss]
            elif self.mean_true == 3:
                g_loss = (-torch.log(self._gaussian_dist_pdf(attention, uncertainty, frequency_index, gt_lst=self.gt_lst)) / 2).mean()
                mean_loss = self.mse_loss(mean, mean_gt)
                std_loss = self.mse_loss(std, std_gt)
                total_loss = g_loss + mean_loss + std_loss
                return (module_input * attention), [total_loss]
            #total_loss = g_loss 
            #total_loss = mean_loss + std_loss
            return (self.adain(module_input, mean=mean, std=std)), [total_loss]
            #return (module_input * attention), [total_loss, g_loss]
        else: # Validation 
            if self.mean_true == 0:
                return (self.adain(module_input, mean=mean, std=std) * attention),  mean#[mean, std]
            elif self.mean_true == 3:
                return (module_input * attention),  mean#[mean, std]
            else:
                return (self.adain(module_input, mean=mean, std=std)),  mean#[mean, std]
            #return (module_input * attention), [total_loss, g_loss]
        
    
    def _gaussian_dist_pdf(self, data_point, var, freq_idx, gt_lst=None):
        if isinstance(gt_lst, list):
            gt = gt_lst[freq_idx]
        else:
            gt = gt_lst
        var = var.clone() + self.eps
        pdf_value = torch.exp(- (data_point - gt) ** 2.0 / var / 2.0) / torch.sqrt(2.0 * np.pi * var)
        return torch.clamp(pdf_value, min=self.eps)

def get_padding(kernel_size, stride, dilation=1):
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


def create_aa(aa_layer, channels, stride=2, enable=True):
    if not aa_layer or not enable:
        return nn.Identity()
    return aa_layer(stride) if issubclass(aa_layer, nn.AvgPool2d) else aa_layer(channels=channels, stride=stride)

class BasicBlock31(nn.Module):
    expansion = 1

    def __init__(
            self, inplanes, planes, stride=1, downsample=None, cardinality=1, base_width=64,
            reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, norm_affine=True, 
            attn_layer=None, aa_layer=None, drop_block=None, drop_path=None):
        super(BasicBlock31, self).__init__()
        assert cardinality == 1, 'BasicBlock only supports cardinality of 1'
        assert base_width == 64, 'BasicBlock does not support changing base width'
        first_planes = planes // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        self.conv1 = nn.Conv2d(
            inplanes, first_planes, kernel_size=3, stride=1 if use_aa else stride, padding=first_dilation,
            dilation=first_dilation, bias=False)
        
        self.bn1 = norm_layer(first_planes, affine=norm_affine)

        self.drop_block = drop_block() if drop_block is not None else nn.Identity()
        self.act1 = act_layer(inplace=True)
        self.aa = create_aa(aa_layer, channels=first_planes, stride=stride, enable=use_aa)

        self.conv2 = nn.Conv2d(first_planes, outplanes, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
        
        self.bn2 = norm_layer(outplanes, affine=norm_affine)

        self.se = create_attn(attn_layer, outplanes)

        self.act2 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_path = drop_path

    def zero_init_last(self):
        nn.init.zeros_(self.bn2.weight)

    def forward(self, x, seq=0):
        shortcut = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.drop_block(x)
        x = self.act1(x)
        x = self.aa(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
           shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act2(x)

        return x


class Bottleneck31(nn.Module):
    expansion = 4

    def __init__(
            self, inplanes, planes, stride=1, downsample=None, cardinality=1, base_width=64,
            reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, norm_affine=True,
            attn_layer=None, aa_layer=None, drop_block=None, drop_path=None, se_dwt_level=[0,0,0,0]):  # se_dwt_level Args =>  [self.dwt_level[0], self.dwt_bn[0], self.dwt_bn[1]]
        super(Bottleneck31, self).__init__()

        self.dwt_level = se_dwt_level[0]
        self.dwt_bn = se_dwt_level[1]
        self.reduction_option = se_dwt_level[2]
        dwt_conv_layer = list()
        dwt_ada_layer = list()

        if self.dwt_level == 0:
            self.split_count = 0
        elif self.dwt_level == 1:
            self.split_count = 4
        elif self.dwt_level == 2:
            self.split_count = 16
        self.nll_loss = None
        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        self.conv1 = nn.Conv2d(inplanes, first_planes, kernel_size=1, bias=False)
        
        self.bn1 = norm_layer(first_planes, affine=norm_affine)
        self.act1 = act_layer(inplace=True)

        if self.dwt_level == 0:
            self.conv2 = nn.Conv2d(
                first_planes, width, kernel_size=3, stride=1 if use_aa else stride,
                padding=first_dilation, dilation=first_dilation, groups=cardinality, bias=False)
        else:
            for idx in range(self.split_count):
                dwt_conv_layer.append(nn.Conv2d(first_planes, width, kernel_size=3, stride=1 if use_aa else stride, padding=first_dilation, dilation=first_dilation, groups=cardinality, bias=False).cuda())
                dwt_ada_layer.append(SEModule4(channels=width, reduction_=self.reduction_option))

        self.dwt_conv_layer = nn.ModuleList(dwt_conv_layer)
        self.dwt_ada_layer = nn.ModuleList(dwt_ada_layer)

        self.bn2 = norm_layer(width, affine=norm_affine)
        
        self.drop_block = drop_block() if drop_block is not None else nn.Identity()
        self.act2 = act_layer(inplace=True)
        self.aa = create_aa(aa_layer, channels=width, stride=stride, enable=use_aa)

        self.conv3 = nn.Conv2d(width, outplanes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes, affine=norm_affine)

        self.se = create_attn(attn_layer, outplanes)

        self.act3 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_path = drop_path

    def zero_init_last(self):
        nn.init.zeros_(self.bn3.weight)

        #Bottleneck Layer
    def dwt_rearrange(self, if_map):
        split_tensor_lst = list()

        if self.dwt_level == 1:
            tdwt.get_dwt_level1(if_map, split_tensor_lst)
        elif self.dwt_level == 2:
            tdwt.get_dwt_level2(if_map, split_tensor_lst)

        # dwt_ada_layer 는 항상 1개만 사용하도록 한다. 
        if self.dwt_bn == 0: 
            output_tensor_lst = [self.dwt_conv_layer[i](split_tensor_lst[i]) for i in range(self.split_count)]
            for i in range(self.split_count):
                output_tensor_lst[i], nll_loss = self.dwt_ada_layer[0](output_tensor_lst[i],i)
                if i == 0:
                    self.nll_loss = nll_loss
                else:
                    self.nll_loss += nll_loss
        else:
            output_tensor_lst = [self.dwt_conv_layer[0](split_tensor_lst[i]) for i in range(self.split_count)]
            for i in range(self.split_count):
                output_tensor_lst[i], nll_loss = self.dwt_ada_layer[0](output_tensor_lst[i],i)
                if i == 0:
                    self.nll_loss = nll_loss
                else:
                    self.nll_loss += nll_loss

        if self.dwt_level == 1:
            return tdwt.get_dwt_level1_inverse(output_tensor_lst)
        elif self.dwt_level == 2:
            return tdwt.get_dwt_level2_inverse(output_tensor_lst)

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        if self.dwt_level == 0:
            x = self.conv2(x)
        else:
            x = self.dwt_rearrange(x)

        x = self.bn2(x)
        x = self.drop_block(x)
        x = self.act2(x)
        x = self.aa(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act3(x)
        
        return x


def downsample_conv(
        in_channels, out_channels, kernel_size, stride=1, dilation=1, first_dilation=None, norm_layer=None):
    norm_layer = norm_layer or nn.BatchNorm2d
    kernel_size = 1 if stride == 1 and dilation == 1 else kernel_size
    first_dilation = (first_dilation or dilation) if kernel_size > 1 else 1
    p = get_padding(kernel_size, stride, first_dilation)

    return nn.Sequential(*[
        nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=p, dilation=first_dilation, bias=False),
        norm_layer(out_channels)
    ])


def downsample_avg(
        in_channels, out_channels, kernel_size, stride=1, dilation=1, first_dilation=None, norm_layer=None):
    norm_layer = norm_layer or nn.BatchNorm2d
    avg_stride = stride if dilation == 1 else 1
    if stride == 1 and dilation == 1:
        pool = nn.Identity()
    else:
        avg_pool_fn = AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
        pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)

    return nn.Sequential(*[
        pool,
        nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
        norm_layer(out_channels)
    ])


def drop_blocks(drop_prob=0.):
    return [
        None, None,
        partial(DropBlock2d, drop_prob=drop_prob, block_size=5, gamma_scale=0.25) if drop_prob else None,
        partial(DropBlock2d, drop_prob=drop_prob, block_size=3, gamma_scale=1.00) if drop_prob else None]


def make_blocks(
        block_fn, channels, block_repeats, inplanes, reduce_first=1, output_stride=32,
        down_kernel_size=1, avg_down=False, drop_block_rate=0., drop_path_rate=0., se_dwt_level=[0,0,0,0], **kwargs):
    stages = []
    feature_info = []
    net_num_blocks = sum(block_repeats)
    net_block_idx = 0
    net_stride = 4
    dilation = prev_dilation = 1
    for stage_idx, (planes, num_blocks, db) in enumerate(zip(channels, block_repeats, drop_blocks(drop_block_rate))):
        stage_name = f'layer{stage_idx + 1}'  # never liked this name, but weight compat requires it
        stride = 1 if stage_idx == 0 else 2
        if net_stride >= output_stride:
            dilation *= stride
            stride = 1
        else:
            net_stride *= stride

        downsample = None
        if stride != 1 or inplanes != planes * block_fn.expansion:
            down_kwargs = dict(
                in_channels=inplanes, out_channels=planes * block_fn.expansion, kernel_size=down_kernel_size,
                stride=stride, dilation=dilation, first_dilation=prev_dilation, norm_layer=kwargs.get('norm_layer'))
            downsample = downsample_avg(**down_kwargs) if avg_down else downsample_conv(**down_kwargs)

        block_kwargs = dict(reduce_first=reduce_first, dilation=dilation, drop_block=db, **kwargs)
        blocks = []
        for block_idx in range(num_blocks):
            downsample = downsample if block_idx == 0 else None
            stride = stride if block_idx == 0 else 1
            block_dpr = drop_path_rate * net_block_idx / (net_num_blocks - 1)  # stochastic depth linear decay rule
            if stage_idx == (se_dwt_level[3]-1) and block_idx == 0:
                print(f'stage_dix {stage_idx} blocks idx: {block_idx}')
                blocks.append(block_fn(
                    inplanes, planes, stride, downsample, first_dilation=prev_dilation,
                    drop_path=DropPath(block_dpr) if block_dpr > 0. else None, se_dwt_level=se_dwt_level, **block_kwargs))
            else:
                blocks.append(block_fn(
                    inplanes, planes, stride, downsample, first_dilation=prev_dilation,
                    drop_path=DropPath(block_dpr) if block_dpr > 0. else None, **block_kwargs))
            prev_dilation = dilation
            inplanes = planes * block_fn.expansion
            net_block_idx += 1
        stages.append((stage_name, nn.Sequential(*blocks)))
        feature_info.append(dict(num_chs=inplanes, reduction=net_stride, module=stage_name))

    return stages, feature_info
    

# Learnable parameter를 위하여 설계됨.
class ResNet_DWT_SE2(nn.Module):
    def __init__(
            self, block, layers, num_classes=1000, in_chans=3, output_stride=32, global_pool='avg',
            cardinality=1, base_width=64, stem_width=64, stem_type='', replace_stem_pool=False, block_reduce_first=1, no_skip=False, aux_header=False,
            dwt_kernel_size=[0,0,0], dwt_level=[2,2,2], dwt_bn=[0,0,0], deep_format=False, 
            down_kernel_size=1, avg_down=False, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, aa_layer=None,
            drop_rate=0.0, drop_path_rate=0., drop_block_rate=0., zero_init_last=True, block_args=None, mvar=False, meta_option=0, **kwargs):
        super(ResNet_DWT_SE2, self).__init__()
        block_args = block_args or dict()
        assert output_stride in (8, 16, 32)
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.grad_checkpointing = False
        self.in_channel = in_chans
        self.meta_option = meta_option
        self.loss_option = aux_header

        self.mean_analysis = None
        self.std_analysis = None

        # DWT Options #
        self.dwt_kernel_size = dwt_kernel_size
        self.dwt_level = dwt_level
        self.dwt_bn = dwt_bn
        
        self.nll_loss = None
    
        dwt_conv_layer = list()
        dwt_ada_layer = list()
        dwt_bn_lst = list()
        print(f'Resnet_DWT initialize -> Kenrel: {self.dwt_kernel_size} / Level: {self.dwt_level} / BN: {self.dwt_bn} IS LOSS MEAN ? ==> {self.loss_option % 10} / IS COSINE ? ==> {self.loss_option // 10}' )

        if self.dwt_kernel_size[0] == 7:
            self.conv1_padding = 3
        elif self.dwt_kernel_size[0] == 5:
            self.conv1_padding = 2
        elif self.dwt_kernel_size[0] == 3:
            self.conv1_padding = 1
        else:
            self.conv1_padding = 3
            

        if self.dwt_level[0] == 1:
            self.split_count = 4
            self.split_im_size = 56
        elif self.dwt_level[0] == 2:
            self.split_count = 16
            self.split_im_size = 28
        elif self.dwt_level[0] == 3:
            self.split_count = 64
            self.split_im_size = 14
        elif self.dwt_level[0] == 0:
            self.split_count = 0
        else:
            assert "DWT_ LEVEL Assertion not (0, 1, 2)"


        inplanes = 64

        SE_NET = SEModule4

        if self.meta_option == 0:
            SE_NET_CH = inplanes * self.split_count
        elif meta_option == 1:
            SE_NET_CH = inplanes
        # elif meta_option == 2:
        #     SE_NET = SEModule3

        if self.dwt_kernel_size[0] == 0:
            self.conv1 = nn.Conv2d(in_chans, inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_chans, inplanes, kernel_size=7, stride=2, padding=3, bias=False)

            if self.dwt_level[0] == 1:
                for idx in range(self.split_count):
                    dwt_conv_layer.append(nn.Conv2d(in_chans, inplanes, kernel_size=self.dwt_kernel_size[0], stride=2, padding=self.conv1_padding, bias=False).cuda())
                    dwt_ada_layer.append(SE_NET(channels=SE_NET_CH, reduction_=self.dwt_bn[1]))#self.dwt_bn[1]))
                    #dwt_bn_lst.append(norm_layer(inplanes))
                    if self.dwt_bn[0] == 1:
                        break
            elif self.dwt_level[0] == 2:
                for idx in range(self.split_count):
                    dwt_conv_layer.append(nn.Conv2d(in_chans, inplanes, kernel_size=self.dwt_kernel_size[0], stride=2, padding=self.conv1_padding, bias=False).cuda())
                    dwt_ada_layer.append(SE_NET(channels=SE_NET_CH, reduction_=self.dwt_bn[1]))#self.dwt_bn[1]))
                    #dwt_bn_lst.append(norm_layer(inplanes))
                    if self.dwt_bn[0] == 1:
                        break

            elif self.dwt_level[0] == 3: 
                for idx in range(self.split_count):
                    dwt_conv_layer.append(nn.Conv2d(in_chans, inplanes, kernel_size=self.dwt_kernel_size[0], stride=2, padding=self.conv1_padding, bias=False).cuda())
            else:     
                assert 'DWT LEVEL Aseertion'
            self.dwt_conv_layer = nn.ModuleList(dwt_conv_layer)
            self.dwt_ada_layer = nn.ModuleList(dwt_ada_layer)
            #self.dwt_bn_layer = nn.ModuleList(dwt_bn_lst)

        self.bn1 = norm_layer(inplanes)
        self.act1 = act_layer(inplace=True)
        self.feature_info = [dict(num_chs=inplanes, reduction=2, module='act1')]

        # Stem pooling. The name 'maxpool' remains for weight compatibility.
        if replace_stem_pool:
            self.maxpool = nn.Sequential(*filter(None, [
                nn.Conv2d(inplanes, inplanes, 3, stride=1 if aa_layer else 2, padding=1, bias=False),
                create_aa(aa_layer, channels=inplanes, stride=2) if aa_layer is not None else None,
                norm_layer(inplanes),
                act_layer(inplace=True)
            ]))
        else:
            if aa_layer is not None:
                if issubclass(aa_layer, nn.AvgPool2d):
                    self.maxpool = aa_layer(2)
                else:
                    self.maxpool = nn.Sequential(*[
                        nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                        aa_layer(channels=inplanes, stride=2)])
            else:
                self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Feature Blocks
        if self.dwt_bn[2] == 0:
            self.se_dwt_level = [0,0,0,0]
        else:
            self.se_dwt_level = [self.dwt_level[0], self.dwt_bn[0], self.dwt_bn[1], self.dwt_bn[2]] #LEVEL? FIXED? REDUCTION? POSITION?
        header_in_channel = 512
        channels = [64, 128, 256, header_in_channel]
        stage_modules, stage_feature_info = make_blocks(
            block, channels, layers, inplanes, cardinality=cardinality, base_width=base_width,
            output_stride=output_stride, reduce_first=block_reduce_first, avg_down=avg_down,
            down_kernel_size=down_kernel_size, act_layer=act_layer, norm_layer=norm_layer, aa_layer=aa_layer,
            drop_block_rate=drop_block_rate, drop_path_rate=drop_path_rate, se_dwt_level = self.se_dwt_level, **block_args)
        for stage in stage_modules:
            self.add_module(*stage)  # layer1, layer2, etc
        self.feature_info.extend(stage_feature_info)

        # Head (Pooling and Classifier)
        self.num_features = header_in_channel * block.expansion
        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)        

        self.init_weights(zero_init_last=zero_init_last, dwt_kernel_size=self.dwt_kernel_size[0])

    @torch.jit.ignore
    def init_weights(self, zero_init_last=True, dwt_kernel_size=0):
        if dwt_kernel_size == 1:
            trunc_normal_(self.pos_embed, std=.02)

        if dwt_kernel_size != 0 and dwt_kernel_size != 1:
            if len(self.dwt_conv_layer) != 0:
                for idx, item in enumerate(self.dwt_conv_layer):
                    nn.init.kaiming_normal_(item.weight, mode='fan_out', nonlinearity='relu')
            
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        if zero_init_last:
            for m in self.modules():
                if hasattr(m, 'zero_init_last'):
                    m.zero_init_last()

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(stem=r'^conv1|bn1|maxpool', blocks=r'^layer(\d+)' if coarse else r'^layer(\d+)\.(\d+)')
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self, name_only=False):
        return 'fc' if name_only else self.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)
            
    def get_first_output_fm(self):
        return self.first_output_fm

    def get_nll_loss(self):
        return self.nll_loss

    #Renew    
    def dwt_rearrange(self, if_map, dwt_ratio, dwt_quant=1, dwt_drop=False, w=None):
        split_tensor_lst = list()

        if self.dwt_level[0] == 1:
            tdwt.get_dwt_level1(if_map, split_tensor_lst)
        elif self.dwt_level[0] == 2:
            tdwt.get_dwt_level2(if_map, split_tensor_lst)
        elif self.dwt_level[0] == 3:
            tdwt.get_dwt_level3(if_map, split_tensor_lst)
        
        output_tensor_lst = [self.dwt_conv_layer[0](split_tensor_lst[i]) for i in range(self.split_count)]
        means_lst = list()
        if self.meta_option == 0:
            output_tensor = torch.cat(output_tensor_lst,dim=1)
            output_tensor, nll_loss_ = self.dwt_ada_layer[0](output_tensor,-1)
            output_tensor_lst = list(torch.chunk(output_tensor, self.split_count, dim=1))
        else:
            for i in range(self.split_count):
                output_tensor_lst[i], nll_loss_ = self.dwt_ada_layer[0](output_tensor_lst[i],i)

        for i in range(self.split_count):
            if self.training == True:
                nll_loss = nll_loss_[0]
                #means_lst.append(nll_loss_[1])
                if i == 0:
                    self.nll_loss = nll_loss
                else:
                    self.nll_loss += nll_loss
            else:
                if i == 0:
                    self.nll_loss = list()
                    self.nll_loss.append(nll_loss_)
                else:
                    self.nll_loss.append(nll_loss_)

        if self.parallel:
            return output_tensor_lst
        else:
            if self.dwt_level[0] == 1:
                return tdwt.get_dwt_level1_inverse(output_tensor_lst)
            elif self.dwt_level[0] == 2:
                return tdwt.get_dwt_level2_inverse(output_tensor_lst)
            elif self.dwt_level[0] == 3:
                return tdwt.get_dwt_level3_inverse(output_tensor_lst)

    def disable_grad_sequential(self, sequential_module):
        for module in sequential_module:
            for param in module.parameters():
                param.requires_grad = False
    
    def print_disabled_grad_sequantial(self, sequential_module):
        for name, param in self.layer1.named_parameters():
            print(f"{name}: requires_grad_ = {param.requires_grad}")
    
    def disable_grad_item(self, module):
            for param in module.parameters():
                param.requires_grad = False
        
    def forward_features(self, x, dwt_ratio=None, dwt_quant=1, dwt_drop=False, analysis=False, result_lst=[], fName=None):
        if self.dwt_kernel_size[0] == 0:
            x = self.conv1(x)
        else:
            if self.dwt_bn[2] == 0:
                x = self.dwt_rearrange(x, dwt_ratio, dwt_quant=dwt_quant, dwt_drop=dwt_drop)
                if self.parallel:
                    x = torch.cat(x, dim=0)
            else:
                x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)

        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq([self.layer1, self.layer2, self.layer3, self.layer4], x, flatten=True)
        else:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
        return x
    
    def forward_head(self, x, pre_logits: bool = False):
        if isinstance(x,list):
            par_x = list()
            for item in x:
                x = self.global_pool(item)
                if self.drop_rate:
                    x = F.dropout(x, p=float(self.drop_rate), training=self.training)
                par_x.append(self.fc(x))
            return par_x
        else:
            x = self.global_pool(x)
            if self.drop_rate:
                x = F.dropout(x, p=float(self.drop_rate), training=self.training)
            return x if pre_logits else self.fc(x)

    def forward(self, x, dwt_ratio=None, ena_dwt_ratio=False, dwt_quant=1, dwt_drop=False, analysis=False, result_lst=[], fName=None, is_weight=0., is_test=False, is_mean=None):
        self.is_test = is_test
        self.is_mean = is_mean
        if is_weight > 0:
            self.parallel = True
        else:
            self.parallel = False

        if self.training and self.is_test:
            self.disable_grad_item(self.bn1)
            self.disable_grad_item(self.act1)
            self.disable_grad_item(self.maxpool)
            
            self.disable_grad_sequential(self.layer1)
            self.disable_grad_sequential(self.layer2)
            self.disable_grad_sequential(self.layer3)
            self.disable_grad_sequential(self.layer4)
        else:
            x = self.forward_features(x, None, dwt_quant=dwt_quant, dwt_drop=dwt_drop, analysis=analysis, result_lst=result_lst, fName=fName)
        
        return self.forward_head(x)

        

def _create_resnet(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(ResNet_DWT_SE2, variant, pretrained, **kwargs)

@register_model
def resnet50_dwt_se4(pretrained=False, aux_header=False, no_skip=False, dwt_kernel_size=[0, 0, 0], dwt_level=[2, 2, 2], dwt_bn=[0, 0, 0], deep_format=False, mvar=False, meta_option=0, **kwargs):
    """Constructs a ResNet-50 model. [3, 4, 6, 3]
    """
    model_args = dict(block=Bottleneck31, layers=[3, 4, 6, 3], aux_header=aux_header, no_skip=no_skip, dwt_kernel_size=dwt_kernel_size, dwt_level=dwt_level, dwt_bn=dwt_bn, deep_format=False, mvar=mvar, meta_option=0, **kwargs)
    return _create_resnet('resnet50', pretrained, **model_args)

@register_model
def resnet50_dwt_se5(pretrained=False, aux_header=False, no_skip=False, dwt_kernel_size=[0, 0, 0], dwt_level=[2, 2, 2], dwt_bn=[0, 0, 0], deep_format=False, mvar=False, meta_option=1, **kwargs):
    """Constructs a ResNet-50 model. [3, 4, 6, 3]
    """
    model_args = dict(block=Bottleneck31, layers=[3, 4, 6, 3], aux_header=aux_header, no_skip=no_skip, dwt_kernel_size=dwt_kernel_size, dwt_level=dwt_level, dwt_bn=dwt_bn, deep_format=False, mvar=mvar, meta_option=1, **kwargs)
    return _create_resnet('resnet50', pretrained, **model_args)