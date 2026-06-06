"""PyTorch ResNet

This started as a copy of https://github.com/pytorch/vision 'resnet.py' (BSD-3-Clause) with
additional dropout and dynamic global avg/max pool.

ResNeXt, SE-ResNeXt, SENet, and MXNet Gluon stem/downsample variants, tiered stems added by Ross Wightman

Copyright 2019, Ross Wightman
"""
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import CIFAR10_DWT_1_MEAN_STD
from .helpers import build_model_with_cfg, checkpoint_seq
from .layers import PatchEmbed, DropBlock2d, DropPath, AvgPool2dSame, BlurPool2d, GroupNorm, create_attn, get_attn, create_classifier, create_classifier2, trunc_normal_
from .registry import register_model
import dann
import dwt_net as D

## for DWT 
from .vision_transformer import Up_Attention
import torch_dwt as tdwt

__all__ = ['ResNet_DWT', 'BasicBlock2', 'Bottleneck2', 'FeatureMappingLayer', 'EffectLayer']  # model_registry will add each entrypoint fn to this


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
}


def get_padding(kernel_size, stride, dilation=1):
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


def create_aa(aa_layer, channels, stride=2, enable=True):
    if not aa_layer or not enable:
        return nn.Identity()
    return aa_layer(stride) if issubclass(aa_layer, nn.AvgPool2d) else aa_layer(channels=channels, stride=stride)
    
class FeatureMappingLayer(nn.Module):
    def __init__(self, id, input_shape):
        super(FeatureMappingLayer, self).__init__()

        self.fc1 = nn.Linear(input_shape, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 2048)
        self.fc4 = nn.Linear(2048, input_shape)

    def forward(self, x):
        x = nn.ReLU()(self.fc1(x))
        x = nn.ReLU()(self.fc2(x))
        x = nn.ReLU()(self.fc3(x))
        x = self.fc4(x)
        return x
    
class EffectLayer(nn.Module):
    def __init__(self, input_shape):
        super(EffectLayer, self).__init__()

        self.fc1 = nn.Linear(input_shape, input_shape*10)
        self.fc2 = nn.Linear(input_shape*10, 1)

    def forward(self, x):
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x

class BasicBlock2(nn.Module):
    expansion = 1

    def __init__(
            self, inplanes, planes, stride=1, downsample=None, cardinality=1, base_width=64,
            reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
            attn_layer=None, aa_layer=None, drop_block=None, drop_path=None):
        super(BasicBlock2, self).__init__()
        assert cardinality == 1, 'BasicBlock only supports cardinality of 1'
        assert base_width == 64, 'BasicBlock does not support changing base width'
        first_planes = planes // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        self.conv1 = nn.Conv2d(
            inplanes, first_planes, kernel_size=3, stride=1 if use_aa else stride, padding=first_dilation,
            dilation=first_dilation, bias=False)
        
        self.bn1 = norm_layer(first_planes)


        self.drop_block = drop_block() if drop_block is not None else nn.Identity()
        self.act1 = act_layer(inplace=True)
        self.aa = create_aa(aa_layer, channels=first_planes, stride=stride, enable=use_aa)

        self.conv2 = nn.Conv2d(
            first_planes, outplanes, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
        
        self.bn2 = norm_layer(outplanes)

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


class Bottleneck2(nn.Module):
    expansion = 4

    def __init__(
            self, inplanes, planes, stride=1, downsample=None, cardinality=1, base_width=64,
            reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
            attn_layer=None, aa_layer=None, drop_block=None, drop_path=None):
        super(Bottleneck2, self).__init__()

        self.bn1 = list()
        self.bn2 = list()
        self.bn3 = list()

        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        self.conv1 = nn.Conv2d(inplanes, first_planes, kernel_size=1, bias=False)
        
        self.bn1 = norm_layer(first_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(
            first_planes, width, kernel_size=3, stride=1 if use_aa else stride,
            padding=first_dilation, dilation=first_dilation, groups=cardinality, bias=False)
        
        self.bn2 = norm_layer(width)
        
        self.drop_block = drop_block() if drop_block is not None else nn.Identity()
        self.act2 = act_layer(inplace=True)
        self.aa = create_aa(aa_layer, channels=width, stride=stride, enable=use_aa)

        self.conv3 = nn.Conv2d(width, outplanes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes)

        self.se = create_attn(attn_layer, outplanes)

        self.act3 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_path = drop_path

    def zero_init_last(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x, seq=0):
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
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
        down_kernel_size=1, avg_down=False, drop_block_rate=0., drop_path_rate=0., **kwargs):
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
class ResNet_DWT(nn.Module):
    def __init__(
            self, block, layers, num_classes=1000, in_chans=3, output_stride=32, global_pool='avg',
            cardinality=1, base_width=64, stem_width=64, stem_type='', replace_stem_pool=False, block_reduce_first=1, no_skip=False, aux_header=False,
            dwt_kernel_size=[0,0,0], dwt_level=[2,2,2], dwt_bn=[0,0,0], deep_format=False, 
            down_kernel_size=1, avg_down=False, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, aa_layer=None,
            drop_rate=0.0, drop_path_rate=0., drop_block_rate=0., zero_init_last=True, block_args=None, mvar=False, **kwargs):
        super(ResNet_DWT, self).__init__()
        block_args = block_args or dict()
        assert output_stride in (8, 16, 32)
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.grad_checkpointing = False
        self.mvar = mvar
        self.dwt_fix = aux_header
        self.no_skip = False
        self.dwt_kernel_size = dwt_kernel_size
        self.dwt_level = dwt_level
        self.dwt_bn = dwt_bn
        self.in_channel = in_chans
        self.deep_format = False
                
        self.dwt_mstd_lst = list()
        dwt_conv_layer = list()
        print(f'Resnet_DWT initialize -> Kenrel: {self.dwt_kernel_size} / Level: {self.dwt_level} / BN: {self.dwt_bn}')
        self.dwt_drop_layer = torch.nn.Dropout2d(p=0.5)

        if self.dwt_level[0] == 1:
            self.split_count = 4
        elif self.dwt_level[0] == 2:
            self.split_count = 16
        elif self.dwt_level[0] == 3:
            self.split_count = 64
        else:
            self.split_count = 1
        
        custom_img_size = 224
        
        self.unbind_count = int((custom_img_size/(2**(self.dwt_level[0]+1))) ** 2) # 224 is img size 
        self.root_unbind_count = int((224/(2**(self.dwt_level[0]+1))))

        self.unbind_output = int((custom_img_size/4) ** 2) # 224 is img size
        self.root_unbind_output = int((custom_img_size/4)) # 224 is img size
        # Stem
        deep_stem = 'deep' in stem_type
        inplanes = stem_width * 2 if deep_stem else 64
        self.depth_ch = inplanes #self.split_count * in_chans
        if self.dwt_bn[1] == 1:
            self.depth_norm = nn.BatchNorm2d(self.depth_ch)
        elif self.dwt_bn[1] == 2:
            self.depth_norm = nn.InstanceNorm2d(self.depth_ch, affine=False, track_running_stats=True)
        elif self.dwt_bn[1] == 3:
            self.depth_norm = nn.InstanceNorm2d(self.depth_ch, affine=True, track_running_stats=True) #IBN
        elif self.dwt_bn[1] == 4:
            self.depth_norm = nn.LayerNorm(self.depth_ch) # Layer Normalization
        else:
            self.depth_norm = None

        
        
        if deep_stem:
            stem_chs = (stem_width, stem_width)
            if 'tiered' in stem_type:
                stem_chs = (3 * (stem_width // 4), stem_width)
            self.conv1 = nn.Sequential(*[
                nn.Conv2d(in_chans, stem_chs[0], 3, stride=2, padding=1, bias=False),
                norm_layer(stem_chs[0]),
                act_layer(inplace=True),
                nn.Conv2d(stem_chs[0], stem_chs[1], 3, stride=1, padding=1, bias=False),
                norm_layer(stem_chs[1]),
                act_layer(inplace=True),
                nn.Conv2d(stem_chs[1], inplanes, 3, stride=1, padding=1, bias=False)])
        else:
            if self.dwt_kernel_size[0] == 0:
                self.conv1 = nn.Conv2d(in_chans, inplanes, kernel_size=7, stride=2, padding=3, bias=False)
            elif self.dwt_kernel_size[0] == 1:
                self.depthwise_conv = nn.Conv2d(self.depth_ch, self.depth_ch, kernel_size=3, stride=2, padding=1, groups= self.depth_ch, bias=False)
                self.pos_embed = nn.Parameter(torch.randn(1, self.depth_ch, self.unbind_count) * .02)
                num_heads = 4
                qkv_bias = True
                attn_drop = 0.
                proj_drop = 0.
                drop = 0.
                self.attn = Up_Attention(self.unbind_count, self.unbind_output, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
                self.pointwise_conv = nn.Conv2d(self.depth_ch, 64, kernel_size=1, stride=1, padding=0, bias=True)
            else:
                if self.dwt_fix==False:
                    self.DWT = D.DWT()
                    
                if self.dwt_level[0] == 1:
                    for idx in range(4):
                        dwt_conv_layer.append(nn.Conv2d(in_chans, inplanes, kernel_size=self.dwt_kernel_size[0], stride=2, padding=3, bias=False).cuda())
                elif self.dwt_level[0] == 2:
                    for idx in range(16):
                        dwt_conv_layer.append(nn.Conv2d(in_chans, inplanes, kernel_size=self.dwt_kernel_size[0], stride=2, padding=3, bias=False).cuda())
                elif self.dwt_level[0] == 3: 
                    for idx in range(64):
                        dwt_conv_layer.append(nn.Conv2d(in_chans, inplanes, kernel_size=self.dwt_kernel_size[0], stride=2, padding=3, bias=False).cuda())
                else:     
                    for idx in range(16):
                        dwt_conv_layer.append(nn.Conv2d(in_chans, inplanes, kernel_size=self.dwt_kernel_size[0], stride=2, padding=3, bias=False).cuda())
                self.dwt_conv_layer = nn.ModuleList(dwt_conv_layer)

        self.bn1 = nn.BatchNorm2d(inplanes)
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
        header_in_channel = 512
        channels = [64, 128, 256, header_in_channel]
        stage_modules, stage_feature_info = make_blocks(
            block, channels, layers, inplanes, cardinality=cardinality, base_width=base_width,
            output_stride=output_stride, reduce_first=block_reduce_first, avg_down=avg_down,
            down_kernel_size=down_kernel_size, act_layer=act_layer, norm_layer=norm_layer, aa_layer=aa_layer,
            drop_block_rate=drop_block_rate, drop_path_rate=drop_path_rate, **block_args)
        for stage in stage_modules:
            self.add_module(*stage)  # layer1, layer2, etc
        self.feature_info.extend(stage_feature_info)

        # Head (Pooling and Classifier)
        self.num_features = header_in_channel * block.expansion
        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

        self.AUX_COUNT=16
        mapping_layer_lst = []
        for i in range(self.AUX_COUNT):
            mapping_layer_lst.append(FeatureMappingLayer(i, header_in_channel))
        self.mapping_layer_lst = nn.ModuleList(mapping_layer_lst)

        self.effect_layer = EffectLayer(num_classes)

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
    
    def dwt_depthwise_convolution(self, if_map):
        split_tensor_lst = list()
        if self.dwt_level[0] == 1:
            tdwt.get_dwt_level1(if_map, split_tensor_lst, x_dwt_rate=None)
        elif self.dwt_level[0] == 2:
            tdwt.get_dwt_level2(if_map, split_tensor_lst, x_dwt_rate=None, x_quant=0)
        elif self.dwt_level[0] == 3:
            tdwt.get_dwt_level3(if_map, split_tensor_lst, x_dwt_rate=None)

        B,_,_,_ = split_tensor_lst[0].shape
        x = torch.cat(split_tensor_lst,dim=1)

        x = self.depthwise_conv(x)
        if self.depth_norm != None:
            x = self.depth_norm(x)
        x = x.reshape(B,self.depth_ch,-1)
        x = x + self.pos_embed
        x = self.attn(x)
        x = x.reshape(B,self.depth_ch, self.root_unbind_count,self.root_unbind_count)
        x = self.pointwise_conv(x)
        return x 
        
        
    def dwt_forward_lev2(self, x):
        LL, hs = self.DWT(x)
        B, N, C, H, W = hs.shape
        hs = hs.reshape(B, N*C, H, W)
        LH, HL, HH= torch.split(hs, self.in_channel, dim=1)
        
        LL_LL, LL_HS = self.DWT(LL)
        B, N, C, H, W = LL_HS.shape
        LL_HS = LL_HS.reshape(B, N*C, H, W)
        LL_LH, LL_HL, LL_HH = torch.split(LL_HS, self.in_channel, dim=1)
        
        LH_LL, LH_HS = self.DWT(LH)
        B, N, C, H, W = LH_HS.shape
        LH_HS = LH_HS.reshape(B, N*C, H, W)
        LH_LH, LH_HL, LH_HH = torch.split(LH_HS, self.in_channel, dim=1)
        
        HL_LL, HL_HS = self.DWT(HL)
        B, N, C, H, W = HL_HS.shape
        HL_HS = HL_HS.reshape(B, N*C, H, W)
        HL_LH, HL_HL, HL_HH = torch.split(HL_HS, self.in_channel, dim=1)
        
        HH_LL, HH_HS = self.DWT(HH)
        B, N, C, H, W = HH_HS.shape
        HH_HS = HH_HS.reshape(B, N*C, H, W)
        HH_LH, HH_HL, HH_HH = torch.split(HH_HS, self.in_channel, dim=1)

        if self.dwt_bn[0] == 0:
            LL_LL = self.dwt_conv_layer[0](LL_LL)
            LL_LH = self.dwt_conv_layer[1](LL_LH)
            LL_HL = self.dwt_conv_layer[2](LL_HL)
            LL_HH = self.dwt_conv_layer[3](LL_HH)
            LH_LL = self.dwt_conv_layer[4](LH_LL)
            LH_LH = self.dwt_conv_layer[5](LH_LH)
            LH_HL = self.dwt_conv_layer[6](LH_HL)
            LH_HH = self.dwt_conv_layer[7](LH_HH)
            HL_LL = self.dwt_conv_layer[8](HL_LL)
            HL_LH = self.dwt_conv_layer[9](HL_LH)
            HL_HL = self.dwt_conv_layer[10](HL_HL)
            HL_HH = self.dwt_conv_layer[11](HL_HH)
            HH_LL = self.dwt_conv_layer[12](HH_LL)
            HH_LH = self.dwt_conv_layer[13](HH_LH)
            HH_HL = self.dwt_conv_layer[14](HH_HL)
            HH_HH = self.dwt_conv_layer[15](HH_HH)
        else:
            LL_LL = self.dwt_conv_layer[0](LL_LL)
            LL_LH = self.dwt_conv_layer[0](LL_LH)
            LL_HL = self.dwt_conv_layer[0](LL_HL)
            LL_HH = self.dwt_conv_layer[0](LL_HH)
            LH_LL = self.dwt_conv_layer[0](LH_LL)
            LH_LH = self.dwt_conv_layer[0](LH_LH)
            LH_HL = self.dwt_conv_layer[0](LH_HL)
            LH_HH = self.dwt_conv_layer[0](LH_HH)
            HL_LL = self.dwt_conv_layer[0](HL_LL)
            HL_LH = self.dwt_conv_layer[0](HL_LH)
            HL_HL = self.dwt_conv_layer[0](HL_HL)
            HL_HH = self.dwt_conv_layer[0](HL_HH)
            HH_LL = self.dwt_conv_layer[0](HH_LL)
            HH_LH = self.dwt_conv_layer[0](HH_LH)
            HH_HL = self.dwt_conv_layer[0](HH_HL)
            HH_HH = self.dwt_conv_layer[0](HH_HH)
        
        if self.dwt_bn[1] == 2 or self.dwt_bn[1] == 3 or self.dwt_bn[1] == 4:
            LL_LL = self.depth_norm(LL_LL)
            LL_LH = self.depth_norm(LL_LH)
            LL_HL = self.depth_norm(LL_HL)
            LL_HH = self.depth_norm(LL_HH)
            LH_LL = self.depth_norm(LH_LL)
            LH_LH = self.depth_norm(LH_LH)
            LH_HL = self.depth_norm(LH_HL)
            LH_HH = self.depth_norm(LH_HH)
            HL_LL = self.depth_norm(HL_LL)
            HL_LH = self.depth_norm(HL_LH)
            HL_HL = self.depth_norm(HL_HL)
            HL_HH = self.depth_norm(HL_HH)
            HH_LL = self.depth_norm(HH_LL)
            HH_LH = self.depth_norm(HH_LH)
            HH_HL = self.depth_norm(HH_HL)
            HH_HH = self.depth_norm(HH_HH)

        LL_HS = torch.stack([LL_LH, LL_HL, LL_HH], dim=2)
        LH_HS = torch.stack([LH_LH, LH_HL, LH_HH], dim=2)
        HL_HS = torch.stack([HL_LH, HL_HL, HL_HH], dim=2)
        HH_HS = torch.stack([HH_LH, HH_HL, HH_HH], dim=2)

        LL = self.DWT(LL_LL, LL_HS)
        LH = self.DWT(LH_LL, LH_HS)
        HL = self.DWT(HL_LL, HL_HS)
        HH = self.DWT(HH_LL, HH_HS)
        
        HS = torch.stack([LH, HL, HH], dim=2)
        out = self.DWT(LL, HS)
        return out
    
    def dwt_forward_lev1(self, x):
        LL, hs = self.DWT(x)
        B, N, C, H, W = hs.shape
        hs = hs.reshape(B, N*C, H, W)
        LH, HL, HH= torch.split(hs, self.in_channel, dim=1)
        self.DWT(LL)
        self.DWT(LH)
        self.DWT(HL)
        self.DWT(HH)
        if self.dwt_bn[0] == 0:
            LL = self.dwt_conv_layer[0](LL)
            LH = self.dwt_conv_layer[1](LH)
            HL = self.dwt_conv_layer[2](HL)
            HH = self.dwt_conv_layer[3](HH)
        else: 
            LL = self.dwt_conv_layer[0](LL)
            LH = self.dwt_conv_layer[0](LH)
            HL = self.dwt_conv_layer[0](HL)
            HH = self.dwt_conv_layer[0](HH)
        
        if self.dwt_bn[1] == 2 or self.dwt_bn[1] == 3 or self.dwt_bn[1] == 4:
            LL = self.depth_norm(LL)
            LH = self.depth_norm(LH)
            HL = self.depth_norm(HL)
            HH = self.depth_norm(HH)
        HS = torch.stack([LH, HL, HH], dim=2)

        out = self.DWT(LL, HS)
        return out
    #Renew    
    def dwt_rearrange(self, if_map, dwt_ratio, dwt_quant=1, dwt_drop=False):
        split_tensor_lst = list()
        
        #dwt_ratio = True
        #dwt_quant = 1
        
        if self.dwt_level[0] == 1:
            tdwt.get_dwt_level1(if_map, split_tensor_lst, x_dwt_rate=dwt_ratio)
        elif self.dwt_level[0] == 2:
            tdwt.get_dwt_level2(if_map, split_tensor_lst, x_dwt_rate=dwt_ratio, x_quant=dwt_quant)
        elif self.dwt_level[0] == 3:
            tdwt.get_dwt_level3(if_map, split_tensor_lst, x_dwt_rate=dwt_ratio)
        else:
            tdwt.get_dwt_level2(if_map, split_tensor_lst, x_dwt_rate=dwt_ratio)

        output_tensor_lst = [(self.dwt_conv_layer[i](split_tensor_lst[i])) for i in range(len(split_tensor_lst))]                

        if self.dwt_level[0] == 1:
            return tdwt.get_dwt_level1_inverse(output_tensor_lst)
        elif self.dwt_level[0] == 2:
            return tdwt.get_dwt_level2_inverse(output_tensor_lst)
        elif self.dwt_level[0] == 3:
            return tdwt.get_dwt_level3_inverse(output_tensor_lst)
        else:
            return tdwt.get_dwt_level2_inverse(output_tensor_lst)
        
    
    def forward_features(self, x, dwt_ratio=None, dwt_quant=1, dwt_drop=False, analysis=False, result_lst=[], fName=None):
        print('CSV File Name: ',fName)
        mean_lst = [-0.057057466357946396,0.7060386538505554,-5.179277650313452e-06,1.3232263326644897,-2.6167762712248077e-07,0.3743395209312439,-0.3285554051399231,1.7560757896717405e-06,0.13485145568847656,-2.691614645300433e-05,-0.7639787197113037,-0.547653079032898,1.1218091249465942,0.000133162597194314,-0.09875902533531189,-0.10571078211069107,-0.6295934319496155,-1.46388840675354,-1.1325898170471191,0.9556585550308228,0.5111185312271118,-1.417383074760437,0.8356536626815796,0.48882389068603516,-0.5815690159797668,-0.5565015077590942,0.025108128786087036,-0.5024105310440063,0.06762072443962097,0.9878159165382385,-0.21401464939117432,-0.6457175016403198,-0.5600462555885315,-0.8545172810554504,-1.4227516651153564,1.1650350093841553,7.925930276542204e-07,-0.5668885707855225,2.2486156581180694e-07,0.5485429763793945,-1.0599725246429443,-0.46619516611099243,0.5351068377494812,-0.3255155086517334,-0.5290671586990356,0.725121021270752,-0.8621638417243958,-0.5418109893798828,-3.7242932648950955e-07,0.16436195373535156,0.22313937544822693,0.8684818744659424,0.368825227022171,-0.46606945991516113,1.1795960664749146,0.6122359037399292,-0.12616026401519775,-0.8574968576431274,1.3128681182861328,0.3180176615715027,0.6363632082939148,-2.29079008102417,1.6352388858795166,0.11971855163574219]
        std_lst = [0.7807341814041138,1.6964383125305176,6.5610702222329564e-06,1.8405921459197998,3.712130194344354e-07,0.7240822315216064,1.5632123947143555,2.364981583014014e-06,0.8748623132705688,3.2679992727935314e-05,1.7368030548095703,1.3506052494049072,1.4743995666503906,0.00016176854842342436,3.256566047668457,0.43791449069976807,1.2816191911697388,2.0981597900390625,1.45108962059021,2.2317657470703125,1.7198412418365479,1.8310530185699463,1.8871062994003296,0.6753318905830383,0.87575364112854,2.452942371368408,3.147238254547119,1.6581344604492188,1.96791672706604,1.3749144077301025,1.0491232872009277,1.0249533653259277,1.426124095916748,1.433781385421753,2.157956123352051,1.5517756938934326,1.120892193284817e-06,0.8860875368118286,2.6819213871931424e-07,0.7746370434761047,1.3534035682678223,1.1506304740905762,1.1455103158950806,1.110863447189331,1.5047214031219482,1.819545865058899,1.1201128959655762,1.0203936100006104,4.941667839375441e-07,0.9299454689025879,0.5674835443496704,1.378926396369934,1.5735379457473755,0.7560486197471619,1.484763503074646,0.9142564535140991,1.3960485458374023,2.169797658920288,1.8008086681365967,0.9810318946838379,1.2246129512786865,2.9010114669799805,2.0490593910217285,1.0187582969665527]

        if self.dwt_kernel_size[0] == 0:
            if analysis:
                x = self.conv1(x)
                def mean_std(feature):
                    return feature.mean(dim=(0,-2,-1)), feature.std(dim=(0,-2,-1))
                mean, std = mean_std(x)
                result_lst.append([mean,std])
            else:
                x = self.conv1(x)
                #x = self.adain(x, mean_lst, std_lst, fName=fName)
        # elif self.dwt_kernel_size[0] == 1:
        #     x = self.dwt_depthwise_convolution(x)
        else:
            if self.dwt_fix:
                x = self.dwt_rearrange(x, dwt_ratio, dwt_quant=dwt_quant, dwt_drop=dwt_drop)
            else:
                if self.dwt_level[0] == 1:
                    x = self.dwt_forward_lev1(x)
                elif self.dwt_level[0] == 2:
                    x = self.dwt_forward_lev2(x)
                else:
                    print("Assertion Error\n")
            
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
    
    def adain(self, content_features, style_mean_list, style_std_list, fName=None):
        #fName = None
        if fName is None:
            style_mean = torch.tensor(style_mean_list).cuda()
            style_std = torch.tensor(style_std_list).cuda()
        else:
            import pandas as pd

            # CSV 파일 경로
            file_path = fName

            # header=None 옵션을 사용하여 컬럼명 행이 없음을 지정
            data = pd.read_csv(file_path, header=None)

            # DataFrame의 각 행을 리스트로 변환
            rows = [list(row) for _, row in data.iterrows()]

            # 3개의 행을 별도의 리스트로 저장
            style_mean_list, style_std_list = rows
            style_mean = torch.tensor(style_mean_list).cuda()
            style_std = torch.tensor(style_std_list).cuda()


        # Compute mean and std of content features
        content_mean = content_features.mean([2,3], keepdim=True)
        content_std = content_features.var([2,3], keepdim=True) + 1e-5 # adding epsilon for numerical stability

        # Normalize content features
        content_normalized = (content_features - content_mean) / content_std

        # Scale and shift with style's statistics
        stylized_content = content_normalized * style_std[None, :, None, None] + style_mean[None, :, None, None]

        return stylized_content
    
    def forward_head(self, x, pre_logits: bool = False):
        x = self.global_pool(x)
        if self.drop_rate:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        return x if pre_logits else self.fc(x)

    def forward(self, x, dwt_ratio=None, ena_dwt_ratio=False, dwt_quant=1, dwt_drop=False, analysis=False, result_lst=[], fName=None):
        x = self.forward_features(x, None, dwt_quant=dwt_quant, dwt_drop=dwt_drop,analysis=analysis, result_lst=result_lst, fName=fName)
        if analysis:
            return
        else:
            return self.forward_head(x)

def _create_resnet(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(ResNet_DWT, variant, pretrained, **kwargs)

@register_model
def resnet18_dwt(pretrained=False, aux_header=False, no_skip=False, dwt_kernel_size=[0, 0, 0], dwt_level=[2, 2, 2], dwt_bn=[0, 0, 0], deep_format=False, mvar=False, **kwargs):
    """Constructs a ResNet-18 model.
    """
    model_args = dict(block=BasicBlock2, layers=[2, 2, 2, 2], aux_header=aux_header, no_skip=no_skip, dwt_kernel_size=dwt_kernel_size, dwt_level=dwt_level, dwt_bn=dwt_bn, deep_format=False, mvar=mvar, **kwargs)
    return _create_resnet('resnet18', pretrained, **model_args)
