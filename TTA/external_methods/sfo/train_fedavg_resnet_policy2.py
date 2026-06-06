#!/usr/bin/env python3
"""FedAvg Training — ResNet with Parallel Channel/Spatial Attention (SA3) + AA Policy Clients

Federated Averaging over 15 ImageNet-C corruption clients PLUS N AutoAugment
policy clients for ResNet models.

Uses SpatialAttention3 (tanh-based, single-channel mask in [-1, 1]) instead of
SpatialAttention2.  No domain CE or Information Maximization losses — only
classification CE is used for training.

Only channel_attn and spatial_attn parameters (and optionally embedding params)
are trainable via --train-mode. The ResNet backbone and head remain frozen.

Uses cosine LR scheduling across rounds.
"""
import argparse
import copy
import csv
import importlib
import logging
import math
import os
import random
import time
from collections import OrderedDict
from contextlib import suppress
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.optim.swa_utils import AveragedModel

from torchvision.datasets import ImageFolder

from timm import utils
from timm.data import create_dataset, create_loader, resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import set_fast_norm
from timm.models import create_model, safe_model_name, resume_checkpoint

import numpy as np

from timm.data.auto_augment import (
    rand_augment_transform, augment_and_mix_transform, auto_augment_transform,
    AugMixSLOp, augmix_sl_ops_v2, _HPARAMS_DEFAULT,
)
from timm.models.vision_transformer import SpatialAttention3, MoEChannelAttention
from train_aug_classifier import AugClassifier, NUM_AUG_TRANSFORMS

_logger = logging.getLogger('fedavg_resnet_policy2')


_SAFE_MAG_CAPS = {
    'NegativeIntensity': 0.8,
    'SolarizeIncreasing': 0.7,
}

SL_GROUP_POLICIES = {
    'sl_noise': {
        'NegativeIntensity': 0.60,
        'SaltAndPepperIncreasing': 0.25,
        'PositiveIntensity': 0.15,
    },
    'sl_blur': {
        'GaussianBlurIncreasing': 0.35,
        'NegativeIntensity': 0.30,
        'SharpnessV2': 0.20,
        'PositiveIntensity': 0.15,
    },
    'sl_weather': {
        'NegativeIntensity': 0.45,
        'PositiveIntensity': 0.20,
        'SaltAndPepperIncreasing': 0.15,
        'SaturationV2': 0.10,
        'SharpnessV2': 0.10,
    },
    'sl_digital': {
        'NegativeIntensity': 0.40,
        'PositiveIntensity': 0.15,
        'PosterizeIncreasing': 0.15,
        'SharpnessV2': 0.15,
        'SolarizeIncreasing': 0.10,
        'SaturationV2': 0.05,
    },
}


class GroupAugMixSLTransform:
    """AugMix-style transform with group-specific inclusion probabilities.

    Each transform is independently included via Bernoulli sampling.
    At least one transform is always applied.  Magnitude is sampled
    from [cap * 0.5, cap] where cap comes from _SAFE_MAG_CAPS.
    """

    def __init__(self, policy_name: str, hparams=None):
        if policy_name not in SL_GROUP_POLICIES:
            raise ValueError(f'Unknown SL group policy: {policy_name}. '
                             f'Available: {list(SL_GROUP_POLICIES.keys())}')
        self.policy_name = policy_name
        self.inclusion = SL_GROUP_POLICIES[policy_name]

        hparams = hparams or _HPARAMS_DEFAULT
        all_ops = {op.name: op for op in augmix_sl_ops_v2(hparams=hparams)}
        self.ops = [(name, all_ops[name], prob) for name, prob in self.inclusion.items()]

    def __call__(self, img):
        selected = [(name, op) for name, op, prob in self.ops
                    if np.random.random() < prob]

        if not selected:
            name, op, _ = self.ops[np.random.randint(len(self.ops))]
            selected = [(name, op)]

        for name, op in selected:
            cap = _SAFE_MAG_CAPS.get(name, 1.0)
            mag = np.random.uniform(cap * 0.5, cap)
            img = op(img, mag)

        return img

    def __repr__(self):
        return f'{self.__class__.__name__}(policy={self.policy_name})'


CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
    'snow', 'frost', 'fog', 'brightness',
    'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
]

CORRUPTION_GROUPS = {
    'noise': ['gaussian_noise', 'shot_noise', 'impulse_noise'],
    'blur': ['defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur'],
    'weather': ['snow', 'frost', 'fog', 'brightness'],
    'digital': ['contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'],
}
CORRUPTION_GROUP_NAMES = list(CORRUPTION_GROUPS.keys())


def _get_base_model(model):
    m = model
    if hasattr(m, 'module'):
        m = m.module
    if hasattr(m, '_orig_mod'):
        m = m._orig_mod
    return m


def _is_auxiliary_param(param_name: str) -> bool:
    name = param_name.lower()
    aux_keywords = (
        'se_module', 'sam_module',
        'channel_attn', 'spatial_attn',
        'se_module_last', 'sam_module_last',
        'channel_attn_last', 'spatial_attn_last',
    )
    return any(kw in name for kw in aux_keywords)


def _is_embedding_param(param_name: str) -> bool:
    name = param_name.lower()
    return name.startswith(('conv1',))


def _is_head_param(param_name: str) -> bool:
    name = param_name.lower()
    return name.startswith(('fc.', 'global_pool.'))


def _is_norm_param(param_name: str) -> bool:
    """Match BatchNorm/GroupNorm params in backbone layers (bn1, bn2, bn3)."""
    name = param_name.lower()
    for kw in ('.bn1.', '.bn2.', '.bn3.', '.norm.'):
        if kw in name:
            return True
    return False


def _freeze_bn(model: nn.Module):
    """Set all BatchNorm/InstanceNorm layers to eval mode to prevent
    running_mean/running_var updates during training."""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                          nn.SyncBatchNorm)):
            m.eval()


def _freeze_aug_classifier(model: nn.Module):
    """Keep AugClassifier in eval mode after model.train()."""
    base = _get_base_model(model)
    if getattr(base, 'aug_classifier', None) is not None:
        base.aug_classifier.eval()


def _is_aug_classifier_param(param_name: str) -> bool:
    return 'aug_classifier' in param_name.lower()


def _apply_train_mode(model: nn.Module, train_mode: int) -> int:
    if train_mode == 0:
        for param in model.parameters():
            param.requires_grad = True
        for name, param in model.named_parameters():
            if _is_head_param(name):
                param.requires_grad = False
    elif train_mode == 1:
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if _is_auxiliary_param(name) or _is_embedding_param(name):
                param.requires_grad = True
    elif train_mode == 2:
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if _is_auxiliary_param(name):
                param.requires_grad = True
    elif train_mode == 3:
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if _is_auxiliary_param(name) or _is_embedding_param(name) or _is_norm_param(name):
                param.requires_grad = True
    else:
        raise ValueError(f'Unsupported train_mode: {train_mode}')

    for name, param in model.named_parameters():
        if _is_aug_classifier_param(name):
            param.requires_grad = False

    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable_count == 0:
        raise ValueError(
            f'No trainable params for train_mode={train_mode}. '
            'Check --parallel-attention / --use-se-module.'
        )
    return trainable_count


def get_trainable_state(model):
    """Extract state_dict of trainable parameters only."""
    return OrderedDict(
        (name, param.data.clone())
        for name, param in model.named_parameters()
        if param.requires_grad
    )


def set_trainable_state(model, state):
    """Load state_dict into trainable parameters only.
    Handles 'module.' prefix from DataParallel checkpoints."""
    cleaned = OrderedDict()
    for k, v in state.items():
        cleaned[k.replace('module.', '', 1) if k.startswith('module.') else k] = v

    with torch.no_grad():
        for name, param in model.named_parameters():
            key = name.replace('module.', '', 1) if name.startswith('module.') else name
            if key in cleaned:
                param.data.copy_(cleaned[key])


def fedavg_aggregate(global_state, client_states):
    avg_state = OrderedDict()
    num_clients = len(client_states)
    for key in global_state.keys():
        avg_state[key] = sum(cs[key] for cs in client_states) / num_clients
    return avg_state


def cosine_lr(base_lr, min_lr, current_round, total_rounds):
    """Cosine annealing LR for the current round."""
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * current_round / total_rounds))


def create_client_loader(corruption, args, data_config, model_dtype, device):
    data_dir = os.path.join(args.data_root, corruption, str(args.severity))
    input_img_mode = args.input_img_mode or ('RGB' if data_config['input_size'][0] == 3 else 'L')

    dataset = create_dataset(
        '', root=data_dir, split=args.val_split, is_training=True,
        class_map=args.class_map, download=False,
        batch_size=args.batch_size, seed=args.seed,
        input_img_mode=input_img_mode,
    )
    loader = create_loader(
        dataset, input_size=data_config['input_size'],
        batch_size=args.batch_size, is_training=True, no_aug=True,
        num_workers=args.workers,
        mean=data_config['mean'], std=data_config['std'],
        pin_memory=args.pin_mem,
        img_dtype=model_dtype or torch.float32,
        device=device, distributed=False,
        use_prefetcher=not args.no_prefetcher,
    )
    return loader


def create_eval_loader(corruption, args, data_config, model_dtype, device):
    data_dir = os.path.join(args.data_root, corruption, str(args.severity))
    input_img_mode = args.input_img_mode or ('RGB' if data_config['input_size'][0] == 3 else 'L')

    dataset = create_dataset(
        '', root=data_dir, split=args.val_split, is_training=False,
        class_map=args.class_map, download=False,
        batch_size=args.batch_size, input_img_mode=input_img_mode,
    )
    loader = create_loader(
        dataset, input_size=data_config['input_size'],
        batch_size=args.batch_size, is_training=False,
        interpolation=data_config['interpolation'],
        num_workers=args.workers, crop_pct=data_config['crop_pct'],
        mean=data_config['mean'], std=data_config['std'],
        pin_memory=args.pin_mem,
        img_dtype=model_dtype or torch.float32,
        device=device, distributed=False,
        use_prefetcher=not args.no_prefetcher,
    )
    return loader


def validate_aa_policies(policies, img_size=224):
    """Validate that all AA policy strings are parseable.
    Raises ValueError with details on the first invalid policy."""
    from PIL import Image
    dummy = Image.new('RGB', (img_size, img_size))
    for p in policies:
        if p.startswith('sl_'):
            if p not in SL_GROUP_POLICIES:
                raise ValueError(
                    f'Unknown SL group policy "{p}". '
                    f'Available: {list(SL_GROUP_POLICIES.keys())}')
            GroupAugMixSLTransform(p)(dummy)
            continue
        try:
            hparams = {'img_mean': (124, 116, 104)}
            if p.startswith('rand'):
                t = rand_augment_transform(p, hparams=hparams)
            elif p.startswith('augmix'):
                t = augment_and_mix_transform(p, hparams=hparams)
            else:
                t = auto_augment_transform(p, hparams=hparams)
            t(dummy)
        except Exception as e:
            raise ValueError(
                f'Invalid AA policy "{p}": {e}\n'
                f'Examples: v0, rand-m9-n3-mstd0.5, augmix-m5-w4-d2, '
                f'sl_noise, sl_blur, sl_weather, sl_digital'
            ) from e
    return True


def _sl_training_transform(policy_name, data_config):
    """Build a training transform with GroupAugMixSLTransform for sl_* policies."""
    from timm.data.transforms import RandomResizedCropAndInterpolation, MaybeToTensor
    from torchvision import transforms

    img_size = data_config['input_size'][-1]
    mean = data_config['mean']
    std = data_config['std']

    return transforms.Compose([
        RandomResizedCropAndInterpolation(img_size, interpolation='random'),
        transforms.RandomHorizontalFlip(p=0.5),
        GroupAugMixSLTransform(policy_name),
        MaybeToTensor(),
        transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
    ])


def create_policy_client_loader(aa_policy, args, data_config, model_dtype, device):
    """Create a training loader on clean ImageNet with the given AA policy.

    For sl_* policies (sl_noise, sl_blur, sl_weather, sl_digital), uses
    GroupAugMixSLTransform with Bernoulli inclusion sampling.
    For standard policies (v0, rand-*, augmix-*), delegates to timm.
    """
    input_img_mode = args.input_img_mode or ('RGB' if data_config['input_size'][0] == 3 else 'L')

    if aa_policy.startswith('sl_'):
        root = os.path.join(args.clean_data_dir, args.clean_split)
        transform = _sl_training_transform(aa_policy, data_config)
        dataset = ImageFolder(root, transform=transform)
        return torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=args.pin_mem, drop_last=True,
        )

    dataset = create_dataset(
        '', root=args.clean_data_dir, split=args.clean_split, is_training=True,
        class_map=args.class_map, download=False,
        batch_size=args.batch_size, seed=args.seed,
        input_img_mode=input_img_mode,
    )
    return create_loader(
        dataset, input_size=data_config['input_size'],
        batch_size=args.batch_size, is_training=True,
        no_aug=False, auto_augment=aa_policy,
        num_workers=args.workers,
        mean=data_config['mean'], std=data_config['std'],
        pin_memory=args.pin_mem,
        img_dtype=model_dtype or torch.float32,
        device=device, distributed=False,
        use_prefetcher=not args.no_prefetcher,
    )


def local_train_one_epoch(model, loader, optimizer, args, device, model_dtype):
    """Single-epoch training with classification CE only (no domain/IM losses)."""
    model.train()
    _freeze_aug_classifier(model)
    if args.train_mode in (1, 2):
        _freeze_bn(model)

    losses_m = utils.AverageMeter()
    top1_m = utils.AverageMeter()

    for batch_idx, (images, target) in enumerate(loader):
        images = images.to(device=device, dtype=model_dtype or torch.float32)
        target = target.to(device=device)

        output = model(images)
        loss = F.cross_entropy(output, target)

        optimizer.zero_grad()
        loss.backward()
        if args.clip_grad is not None:
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            nn.utils.clip_grad_norm_(trainable_params, args.clip_grad)
        optimizer.step()

        acc1, = utils.accuracy(output.detach(), target, topk=(1,))
        losses_m.update(loss.item(), images.shape[0])
        top1_m.update(acc1.item(), images.shape[0])

        if batch_idx % args.log_interval == 0:
            _logger.info(
                f'    [{batch_idx:>4d}/{len(loader)}]  '
                f'loss={losses_m.val:.4f}({losses_m.avg:.4f})  '
                f'acc@1={top1_m.val:.2f}%({top1_m.avg:.2f}%)')

    return losses_m.avg, top1_m.avg


@torch.no_grad()
def evaluate(model, loader, args, device, model_dtype):
    model.eval()

    losses_m = utils.AverageMeter()
    top1_m = utils.AverageMeter()
    top5_m = utils.AverageMeter()
    prefetcher = not args.no_prefetcher

    for images, target in loader:
        if not prefetcher:
            images = images.to(device=device, dtype=model_dtype)
            target = target.to(device=device)

        output = model(images)
        if isinstance(output, (tuple, list)):
            output = output[0]
        loss = F.cross_entropy(output, target)

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        losses_m.update(loss.item(), images.shape[0])
        top1_m.update(acc1.item(), images.shape[0])
        top5_m.update(acc5.item(), images.shape[0])

    return losses_m.avg, top1_m.avg, top5_m.avg


# ── Argument parsing ──

config_parser = parser = argparse.ArgumentParser(description='Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE')

parser = argparse.ArgumentParser(description='FedAvg — ResNet Parallel Attention (SA3) Training')

group = parser.add_argument_group('Data')
group.add_argument('--data-root', type=str, default='/home/oem/servers/imagenet-c')
group.add_argument('--severity', type=int, default=5)
group.add_argument('--val-split', type=str, default='validation')
group.add_argument('--class-map', default='', type=str)
group.add_argument('--input-img-mode', default=None, type=str)
group.add_argument('--clean-data-dir', type=str, default='/data/imagenet/imagenet',
                   help='Clean ImageNet root for AA policy clients')
group.add_argument('--clean-split', type=str, default='val',
                   help='Split of clean ImageNet to use for policy clients')
group.add_argument('--aa-policies', type=str, nargs='+', default=None,
                   help='AA policy strings for additional clients '
                        '(e.g. v0 rand-m9-n3-mstd0.5 augmix-m5-w4-d2)')

group = parser.add_argument_group('Model')
group.add_argument('--model', default='resnet50', type=str)
group.add_argument('--resume', type=str, default=None)
group.add_argument('--resume-round', type=str, default=None,
                   help='Resume FedAvg from a round checkpoint (e.g. ./output/.../round_007.pth). '
                        'Restores trainable state, round number, and best accuracy.')
group.add_argument('--resume-aug', type=str, default=None,
                   help='Load frozen AugClassifier checkpoint; stage1 mean residual '
                        '(32→mean→1, 112x112) is added to conv1 features')
group.add_argument('--num-classes', type=int, default=1000)
group.add_argument('--input-size', default=[3, 224, 224], nargs=3, type=int)
group.add_argument('--parallel-attention', action='store_true', default=True)
group.add_argument('--vit-kernel-size', type=int, default=7)
group.add_argument('--spatial-group-size', type=int, default=1)
group.add_argument('--sam-norm-type', type=int, default=0, choices=[0, 1, 2, 3, 4],
                   help='SpatialAttention norm: 0=Identity, 1=BN, 2=IN(affine), 3=IN, 4=GN')
group.add_argument('--use-se-module', action='store_true', default=False)
group.add_argument('--use-sam-module', type=int, default=-1)
group.add_argument('--reverse-se', action='store_true', default=False)
group.add_argument('--train-mode', type=int, default=1, choices=[0, 1, 2, 3],
                   help='0=all(head frozen), 1=aux+embedding, 2=aux only, '
                        '3=aux+embedding+backbone norms')

group = parser.add_argument_group('SpatialAttention3')
group.add_argument('--detach', action='store_true', default=False,
                   help='Detach spatial attention input (block backbone gradient through spatial path)')
group.add_argument('--residual', action='store_true', default=False,
                   help='Use residual fusion: x + x*channel_attn*spatial_attn instead of x*channel_attn*spatial_attn')
group.add_argument('--rand-cor', action='store_true', default=False,
                   help='Each round randomly picks 1 of 4 corruption groups (noise/blur/weather/digital) '
                        'instead of using all 15 corruptions')
group.add_argument('--var-feature', action='store_true', default=False,
                   help='Use 5-channel spatial input (mean,min,max,std,median) instead of 2 (mean,max)')
group.add_argument('--moe-channel', action='store_true', default=False,
                   help='Replace ChannelAttention with MoEChannelAttention '
                        '(8 expert heads weighted by AugClassifier probs). Requires --resume-aug.')

group = parser.add_argument_group('FedAvg')
group.add_argument('--rounds', type=int, default=10)
group.add_argument('--warmup-aa', type=int, default=0,
                   help='Warmup rounds using only --aa-policies clients before main FedAvg. '
                        'Total rounds = warmup_aa + rounds. LR cosine spans the total.')
group.add_argument('--local-epochs', type=int, default=1)
group.add_argument('--swa', action='store_true', default=False,
                   help='Enable SWA (temporal weight averaging across FedAvg rounds)')
group.add_argument('--swa-start-frac', type=float, default=0.5,
                   help='Start SWA from this fraction of total rounds (default: 0.5)')

group = parser.add_argument_group('Optimizer')
group.add_argument('--lr', type=float, default=5e-3)
group.add_argument('--min-lr', type=float, default=1e-5)
group.add_argument('--weight-decay', type=float, default=1e-4)
group.add_argument('--clip-grad', type=float, default=1.0)

group = parser.add_argument_group('Training')
group.add_argument('-b', '--batch-size', type=int, default=64)
group.add_argument('--device', default='cuda', type=str)
group.add_argument('--seed', type=int, default=42)
group.add_argument('-j', '--workers', type=int, default=4)
group.add_argument('--pin-mem', action='store_true', default=False)
group.add_argument('--no-prefetcher', action='store_true', default=False)
group.add_argument('--amp', action='store_true', default=False)
group.add_argument('--amp-dtype', default='float16', type=str)
group.add_argument('--model-dtype', default=None, type=str)
group.add_argument('--log-interval', type=int, default=50)
group.add_argument("--local_rank", default=0, type=int)
group.add_argument('--device-modules', default=None, type=str, nargs='+')

group = parser.add_argument_group('Output')
group.add_argument('--output-dir', type=str, default=None,
                   help='Output directory (default: ./output/fedavg_resnet_sa3_k{vit_kernel_size}_tm{train_mode})')

group = parser.add_argument_group('Compatibility (unused)')
group.add_argument('--drop', type=float, default=0.0)
group.add_argument('--drop-path', type=float, default=None)
group.add_argument('--drop-block', type=float, default=None)
group.add_argument('--bn-momentum', type=float, default=None)
group.add_argument('--bn-eps', type=float, default=None)


def _parse_args():
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)
    args = parser.parse_args(remaining)
    return args


def main():
    utils.setup_default_logging()
    args = _parse_args()

    aa_policies = args.aa_policies or []

    if args.output_dir is None:
        norm_suffix = f'_sn{args.sam_norm_type}' if args.sam_norm_type != 0 else ''
        swa_suffix = '_swa' if args.swa else ''
        policy_suffix = f'_aa{len(aa_policies)}' if aa_policies else ''
        detach_suffix = '_det' if args.detach else ''
        residual_suffix = '_res' if args.residual else ''
        randcor_suffix = '_rc' if args.rand_cor else ''
        vf_suffix = '_vf' if args.var_feature else ''
        aug_suffix = '_aug' if args.resume_aug else ''
        moe_suffix = '_moe' if args.moe_channel else ''
        args.output_dir = (f'./output/fedavg_resnet_sa3_k{args.vit_kernel_size}'
                           f'_tm{args.train_mode}_le{args.local_epochs}'
                           f'{norm_suffix}{policy_suffix}'
                           f'{detach_suffix}{residual_suffix}{randcor_suffix}{vf_suffix}{aug_suffix}{moe_suffix}{swa_suffix}')

    if args.device_modules:
        for module in args.device_modules:
            importlib.import_module(module)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)
    utils.random_seed(args.seed, 0)

    model_dtype = None
    if args.model_dtype:
        model_dtype = getattr(torch, args.model_dtype)

    # ── Create model ──
    model_kwargs = {}
    if args.use_sam_module != -1 or args.parallel_attention:
        model_kwargs['sam_kernel_size'] = args.vit_kernel_size
        model_kwargs['spatial_group_size'] = args.spatial_group_size

    in_chans = args.input_size[0] if args.input_size else 3

    model = create_model(
        args.model,
        num_classes=args.num_classes,
        in_chans=in_chans,
        parallel_attention=args.parallel_attention,
        use_se_module=args.use_se_module,
        use_sam_module=args.use_sam_module,
        reverse_se_sam=args.reverse_se,
        sam_norm_type=args.sam_norm_type,
        var_feature=args.var_feature,
        **model_kwargs,
    )
    model.to(device=device, dtype=model_dtype)

    if args.resume:
        _logger.info(f'Loading checkpoint: {args.resume}')
        ckpt = torch.load(args.resume, map_location='cpu')
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        model_sd = model.state_dict()
        skipped = [k for k in state_dict
                   if k in model_sd and state_dict[k].shape != model_sd[k].shape]
        for k in skipped:
            del state_dict[k]
        if skipped:
            _logger.info(f'  Skipped (shape mismatch): {skipped}')
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            _logger.info(f'  Missing keys ({len(missing)}): {missing}')
        if unexpected:
            _logger.info(f'  Unexpected keys ({len(unexpected)}): {unexpected}')

    # ── Replace SpatialAttention with SpatialAttention3 ──
    base = _get_base_model(model)
    if isinstance(base.conv1, nn.Sequential):
        ch = base.conv1[-1].out_channels
    else:
        ch = base.conv1.out_channels
    base.spatial_attn = SpatialAttention3(
        kernel_size=args.vit_kernel_size, channels=ch,
        var_feature=args.var_feature,
    ).to(device=device, dtype=model_dtype)
    _logger.info(f'SpatialAttention3 installed: channels={ch}, '
                 f'kernel={args.vit_kernel_size}, var_feature={args.var_feature}')

    # ── Attention fusion options ──
    if args.parallel_attention:
        base = _get_base_model(model)
        base.attn_detach = args.detach
        base.attn_residual = args.residual
        _logger.info(f'Attention fusion: detach={args.detach}, residual={args.residual}')

    # ── Load frozen AugClassifier (stage1 mean residual, no DeconvAdapter) ──
    if args.resume_aug:
        base = _get_base_model(model)
        _logger.info(f'Loading AugClassifier: {args.resume_aug}')
        aug_ckpt = torch.load(args.resume_aug, map_location='cpu')
        aug_embed_dim = aug_ckpt.get('embed_dim', 768)
        aug_num_transforms = aug_ckpt.get('num_transforms', NUM_AUG_TRANSFORMS)
        _logger.info(f'  embed_dim={aug_embed_dim}, num_transforms={aug_num_transforms}')

        aug_cls = AugClassifier(
            in_chans=in_chans,
            num_classes=aug_num_transforms,
            embed_dim=aug_embed_dim,
        )
        aug_sd = aug_ckpt['state_dict'] if 'state_dict' in aug_ckpt else aug_ckpt
        aug_cls.load_state_dict(aug_sd, strict=True)
        aug_cls.to(device=device, dtype=model_dtype)
        aug_cls.eval()
        for p in aug_cls.parameters():
            p.requires_grad = False
        _logger.info(f'  AugClassifier frozen: {sum(p.numel() for p in aug_cls.parameters()):,} params')
        _logger.info(f'  Stage1 mean residual: (32,112,112) → mean → (1,112,112) broadcast to conv1')

        base.aug_classifier = aug_cls

    # ── MoE Channel Attention ──
    if args.moe_channel:
        if not args.resume_aug:
            raise ValueError('--moe-channel requires --resume-aug (AugClassifier provides expert weights)')
        base = _get_base_model(model)
        old_ca = base.channel_attn
        if isinstance(base.conv1, nn.Sequential):
            ch = base.conv1[-1].out_channels
        else:
            ch = base.conv1.out_channels
        num_experts = aug_num_transforms
        moe_ca = MoEChannelAttention(
            channels=ch, num_experts=num_experts,
        ).to(device=device, dtype=model_dtype)
        moe_ca.init_from_channel_attn(old_ca)
        base.channel_attn = moe_ca
        moe_params = sum(p.numel() for p in base.channel_attn.parameters())
        _logger.info(f'MoEChannelAttention installed: channels={ch}, '
                     f'num_experts={num_experts}, params={moe_params:,} '
                     f'(initialized from pretrained ChannelAttention)')

    trainable_count = _apply_train_mode(model, args.train_mode)
    total_params = sum(p.numel() for p in model.parameters())
    _logger.info(f'Model loaded: {safe_model_name(args.model)}, '
                 f'trainable: {trainable_count:,}/{total_params:,} '
                 f'(train_mode={args.train_mode})')

    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    _logger.info(f'Trainable parameter groups ({len(trainable_names)}):')
    for n in trainable_names:
        _logger.info(f'  {n}')

    # ── DataParallel for multi-GPU ──
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        _logger.info(f'DataParallel enabled: {torch.cuda.device_count()} GPUs')

    data_config = resolve_data_config(vars(args) | {'pretrained': False},
                                      model=_get_base_model(model), verbose=True)

    # ── Output dir ──
    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, 'summary.csv')

    # ── Validate AA policies ──
    if aa_policies:
        _logger.info(f'Validating {len(aa_policies)} AA policies...')
        validate_aa_policies(aa_policies, img_size=args.input_size[1])
        for i, p in enumerate(aa_policies):
            _logger.info(f'  Policy {i}: {p}')

    if args.rand_cor:
        _logger.info('Random corruption group mode: 1 of 4 groups per round '
                     f'({", ".join(CORRUPTION_GROUP_NAMES)})')

    total_clients = len(CORRUPTIONS) + len(aa_policies)

    # ── SWA setup ──
    total_rounds = args.warmup_aa + args.rounds
    swa_model = None
    swa_start = max(1, int(total_rounds * args.swa_start_frac) + 1)
    if args.swa:
        swa_model = AveragedModel(model, device=device)
        _logger.info(f'SWA enabled: averaging from global round {swa_start}/{total_rounds}')

    # ── FedAvg training loop ──
    total_rounds = args.warmup_aa + args.rounds
    _logger.info(f'\nStarting FedAvg (SA3): {total_rounds} total rounds '
                 f'({args.warmup_aa} warmup-aa + {args.rounds} main), '
                 f'{len(CORRUPTIONS)} corruption + {len(aa_policies)} policy = '
                 f'{total_clients} clients, '
                 f'local_epochs={args.local_epochs}, lr={args.lr}→{args.min_lr} (cosine)')
    if args.warmup_aa > 0:
        if not aa_policies:
            _logger.warning('--warmup-aa > 0 but no --aa-policies specified; warmup skipped')
        else:
            _logger.info(f'  Warmup-AA: rounds 1–{args.warmup_aa} use only AA policy clients')

    best_mean_acc1 = 0.0
    start_round = 1

    if args.resume_round:
        _logger.info(f'Resuming from round checkpoint: {args.resume_round}')
        rnd_ckpt = torch.load(args.resume_round, map_location='cpu')
        set_trainable_state(model, rnd_ckpt['state_dict'])
        start_round = rnd_ckpt['round'] + 1
        best_mean_acc1 = rnd_ckpt.get('mean_acc1', 0.0)
        _logger.info(f'  Restored round {rnd_ckpt["round"]}, '
                     f'best_mean_acc1={best_mean_acc1:.3f}%, '
                     f'resuming from global round {start_round}')

    for global_rnd in range(start_round, total_rounds + 1):
        round_start = time.time()

        current_lr = cosine_lr(args.lr, args.min_lr, global_rnd - 1, total_rounds)
        is_warmup = global_rnd <= args.warmup_aa and len(aa_policies) > 0

        if is_warmup:
            phase_label = f'WU {global_rnd}/{args.warmup_aa}'
        else:
            main_rnd = global_rnd - args.warmup_aa
            phase_label = f'R {main_rnd}/{args.rounds}'

        _logger.info(f'\n{"="*60}')
        _logger.info(f'  [{phase_label}] Global round {global_rnd}/{total_rounds}  (lr={current_lr:.6f})')
        _logger.info(f'{"="*60}')

        global_state = get_trainable_state(model)
        client_states = []

        # ── Corruption clients (skipped during warmup-aa) ──
        if not is_warmup:
            if args.rand_cor:
                round_corruptions = [random.choice(group)
                                     for group in CORRUPTION_GROUPS.values()]
                _logger.info(f'  rand-cor: {round_corruptions}')
            else:
                round_corruptions = CORRUPTIONS

            for ci, corruption in enumerate(round_corruptions):
                set_trainable_state(model, global_state)

                trainable_params = [p for p in model.parameters() if p.requires_grad]
                optimizer = torch.optim.AdamW(
                    trainable_params, lr=current_lr, weight_decay=args.weight_decay,
                )

                loader = create_client_loader(corruption, args, data_config, model_dtype, device)
                for local_ep in range(args.local_epochs):
                    loss, acc1 = local_train_one_epoch(
                        model, loader, optimizer, args, device, model_dtype,
                    )

                _logger.info(f'  Client {ci:>2d} [{corruption:<22s}]  '
                             f'loss={loss:.4f}  acc@1={acc1:.2f}%')

                client_states.append(get_trainable_state(model))

        # ── AA policy clients (warmup-aa phase only) ──
        if is_warmup:
            for pi, policy in enumerate(aa_policies):
                set_trainable_state(model, global_state)

                trainable_params = [p for p in model.parameters() if p.requires_grad]
                optimizer = torch.optim.AdamW(
                    trainable_params, lr=current_lr, weight_decay=args.weight_decay,
                )

                loader = create_policy_client_loader(policy, args, data_config, model_dtype, device)
                for local_ep in range(args.local_epochs):
                    loss, acc1 = local_train_one_epoch(
                        model, loader, optimizer, args, device, model_dtype,
                    )

                _logger.info(f'  Client {pi:>2d} [aa:{policy:<19s}]  '
                             f'loss={loss:.4f}  acc@1={acc1:.2f}%')

                client_states.append(get_trainable_state(model))

        # FedAvg aggregation
        aggregated = fedavg_aggregate(global_state, client_states)
        set_trainable_state(model, aggregated)

        round_time = time.time() - round_start
        _logger.info(f'  Aggregated {len(client_states)} clients ({round_time:.1f}s)')

        if swa_model is not None and global_rnd >= swa_start:
            swa_model.update_parameters(model)
            _logger.info(f'  SWA updated (n_averaged={swa_model.n_averaged.item()})')

        # ── Evaluate ──
        _logger.info(f'\n  Evaluating round {global_rnd}...')
        eval_results = OrderedDict()
        acc1_sum = 0.0

        for corruption in CORRUPTIONS:
            eval_loader = create_eval_loader(corruption, args, data_config, model_dtype, device)
            eval_loss, eval_acc1, eval_acc5 = evaluate(
                model, eval_loader, args, device, model_dtype,
            )
            eval_results[corruption] = {'loss': eval_loss, 'top1': eval_acc1, 'top5': eval_acc5}
            acc1_sum += eval_acc1
            _logger.info(f'    {corruption:<22s}  Acc@1={eval_acc1:.3f}%  Acc@5={eval_acc5:.3f}%')

        mean_acc1 = acc1_sum / len(CORRUPTIONS)
        _logger.info(f'  [{phase_label}] Mean Acc@1: {mean_acc1:.3f}%  (lr={current_lr:.6f})')

        # ── Write summary CSV ──
        write_header = (global_rnd == 1) or (global_rnd == start_round and not os.path.exists(summary_path))
        with open(summary_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                header = ['round', 'phase', 'lr', 'mean_acc1'] + list(CORRUPTIONS)
                writer.writerow(header)
            phase = 'warmup' if is_warmup else 'main'
            row = [global_rnd, phase, f'{current_lr:.6f}', f'{mean_acc1:.3f}']
            for c in CORRUPTIONS:
                row.append(f'{eval_results[c]["top1"]:.3f}')
            writer.writerow(row)

        # ── Save checkpoint ──
        ckpt_data = {
            'round': global_rnd,
            'state_dict': get_trainable_state(model),
            'mean_acc1': mean_acc1,
            'lr': current_lr,
            'args': vars(args),
        }

        ckpt_path = os.path.join(args.output_dir, f'round_{global_rnd:03d}.pth')
        torch.save(ckpt_data, ckpt_path)

        if mean_acc1 > best_mean_acc1:
            best_mean_acc1 = mean_acc1
            best_path = os.path.join(args.output_dir, 'best.pth')
            torch.save(ckpt_data, best_path)
            _logger.info(f'  New best! Mean Acc@1: {mean_acc1:.3f}% → {best_path}')
        else:
            _logger.info(f'  Checkpoint saved: {ckpt_path}')

    # ── SWA final evaluation ──
    if swa_model is not None and swa_model.n_averaged.item() > 0:
        _logger.info(f'\n{"="*60}')
        _logger.info(f'  SWA evaluation (averaged global rounds {swa_start}–{total_rounds}, '
                     f'n={swa_model.n_averaged.item()})')
        _logger.info(f'{"="*60}')

        set_trainable_state(model, OrderedDict(
            (n, p.data.clone()) for n, p in swa_model.module.named_parameters()
            if p.requires_grad))

        swa_eval = OrderedDict()
        swa_acc1_sum = 0.0
        for corruption in CORRUPTIONS:
            eval_loader = create_eval_loader(corruption, args, data_config, model_dtype, device)
            _, s_acc1, s_acc5 = evaluate(model, eval_loader, args, device, model_dtype)
            swa_eval[corruption] = {'top1': s_acc1, 'top5': s_acc5}
            swa_acc1_sum += s_acc1
            _logger.info(f'    {corruption:<22s}  Acc@1={s_acc1:.3f}%  Acc@5={s_acc5:.3f}%')

        swa_mean = swa_acc1_sum / len(CORRUPTIONS)
        _logger.info(f'  SWA Mean Acc@1: {swa_mean:.3f}%')

        with open(summary_path, 'a', newline='') as f:
            writer = csv.writer(f)
            row = ['swa', 'swa', '', f'{swa_mean:.3f}']
            for c in CORRUPTIONS:
                row.append(f'{swa_eval[c]["top1"]:.3f}')
            writer.writerow(row)

        swa_ckpt = {
            'round': f'swa_{swa_start}-{total_rounds}',
            'state_dict': get_trainable_state(model),
            'mean_acc1': swa_mean,
            'args': vars(args),
        }
        swa_path = os.path.join(args.output_dir, 'swa_best.pth')
        torch.save(swa_ckpt, swa_path)
        _logger.info(f'  SWA checkpoint saved: {swa_path}')

    _logger.info(f'\n{"="*60}')
    _logger.info(f'  Training complete. Summary: {summary_path}')
    _logger.info(f'  Best Mean Acc@1: {best_mean_acc1:.3f}%')
    _logger.info(f'{"="*60}')


if __name__ == '__main__':
    main()
