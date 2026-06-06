#!/usr/bin/env python3
"""FedAvg Training — Direct Parallel Channel/Spatial Attention + AA Policy Clients

Federated Averaging over 15 ImageNet-C corruption clients PLUS N AutoAugment
policy clients.  Policy clients train on clean ImageNet with the specified
augmentation policy applied.

Only channel_attn and spatial_attn parameters (and optionally embedding params)
are trainable via --train-mode. The ViT backbone and head remain frozen.

Uses cosine LR scheduling across rounds.
"""
import argparse
import copy
import csv
import importlib
import logging
import math
import os
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

from timm.data.auto_augment import rand_augment_transform, augment_and_mix_transform, auto_augment_transform
from timm.models.vision_transformer import SpatialAttention2

_logger = logging.getLogger('fedavg2_policy')

CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
    'snow', 'frost', 'fog', 'brightness',
    'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
]


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
    return name.startswith((
        'patch_embed', 'pos_embed', 'cls_token', 'reg_token', 'norm_pre',
        'conv1', 'bn1', 'act1',
    ))


def _apply_train_mode(model: nn.Module, train_mode: int) -> int:
    if train_mode == 0:
        for param in model.parameters():
            param.requires_grad = True
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
    else:
        raise ValueError(f'Unsupported train_mode: {train_mode}')

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
    """Load state_dict into trainable parameters only."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in state:
                param.data.copy_(state[name])


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
    """Validate that all AA policy strings are parseable by timm."""
    from PIL import Image
    dummy = Image.new('RGB', (img_size, img_size))
    for p in policies:
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
                f'Examples: v0, original, rand-m9-n3-mstd0.5, augmix-m5-w4-d2'
            ) from e
    return True


class PairedCorruptionDataset(torch.utils.data.Dataset):
    """Return (clean_img, corrupt_img, label) from aligned clean/corrupt dirs."""

    def __init__(self, clean_root, corrupt_root, transform):
        self.clean = ImageFolder(clean_root)
        self.corrupt = ImageFolder(corrupt_root)
        self.transform = transform
        assert len(self.clean) == len(self.corrupt), (
            f'Size mismatch: clean={len(self.clean)} vs corrupt={len(self.corrupt)}')

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, idx):
        clean_pil, label = self.clean[idx]
        corrupt_pil, _ = self.corrupt[idx]
        return self.transform(clean_pil), self.transform(corrupt_pil), label


class PairedPolicyDataset(torch.utils.data.Dataset):
    """Return (clean_img, augmented_img, label) from the SAME image."""

    def __init__(self, root, clean_transform, aug_transform):
        self.dataset = ImageFolder(root)
        self.clean_transform = clean_transform
        self.aug_transform = aug_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        pil_img, label = self.dataset[idx]
        return self.clean_transform(pil_img), self.aug_transform(pil_img), label


def _eval_transform(data_config):
    return create_transform(
        input_size=data_config['input_size'],
        is_training=False,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        crop_pct=data_config['crop_pct'],
    )


def create_paired_corruption_loader(corruption, args, data_config):
    clean_root = os.path.join(args.clean_data_dir, args.clean_split)
    corrupt_root = os.path.join(args.data_root, corruption, str(args.severity))
    transform = _eval_transform(data_config)
    ds = PairedCorruptionDataset(clean_root, corrupt_root, transform)
    return torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=args.pin_mem, drop_last=True,
    )


def create_paired_policy_loader(aa_policy, args, data_config):
    root = os.path.join(args.clean_data_dir, args.clean_split)
    clean_tf = _eval_transform(data_config)
    aug_tf = create_transform(
        input_size=data_config['input_size'],
        is_training=True, no_aug=False, auto_augment=aa_policy,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
    )
    ds = PairedPolicyDataset(root, clean_tf, aug_tf)
    return torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=args.pin_mem, drop_last=True,
    )


def create_policy_client_loader(aa_policy, args, data_config, model_dtype, device):
    """Create a training loader on clean ImageNet with the given AA policy."""
    input_img_mode = args.input_img_mode or ('RGB' if data_config['input_size'][0] == 3 else 'L')

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


def compute_domain_loss(logits, domain_label):
    """Spatial-level domain classification CE loss."""
    B, N, H, W = logits.shape
    logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, N)
    labels = torch.full((logits_flat.shape[0],), domain_label,
                        device=logits.device, dtype=torch.long)
    return F.cross_entropy(logits_flat, labels)


def compute_im_loss(logits):
    """Information Maximization: minimize per-position entropy, maximize marginal entropy."""
    probs = F.softmax(logits, dim=1)
    log_probs = (probs + 1e-8).log()
    h_cond = -(probs * log_probs).sum(dim=1).mean()
    p_marg = probs.mean(dim=(0, 2, 3))
    h_marg = -(p_marg * (p_marg + 1e-8).log()).sum()
    return h_cond - h_marg


def local_train_one_epoch(model, loader, optimizer, args, device, model_dtype):
    model.train()

    losses_m = utils.AverageMeter()
    top1_m = utils.AverageMeter()
    prefetcher = not args.no_prefetcher

    for batch_idx, (images, target) in enumerate(loader):
        if not prefetcher:
            images = images.to(device=device, dtype=model_dtype)
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


def local_train_one_epoch_dual(model, paired_loader, optimizer,
                               args, device, model_dtype, current_round=1):
    """Train with paired (source, target, label) loader using domain CE + IM loss.

    During warm-up (round <= warmup_rounds), only CE is active.
    """
    model.train()

    base = _get_base_model(model)
    N = args.prop_size

    in_warmup = current_round <= args.warmup_rounds
    lam_dom = 0.0 if in_warmup else args.lambda_domain
    lam_im = 0.0 if in_warmup else args.lambda_im

    losses_m = utils.AverageMeter()
    cls_m = utils.AverageMeter()
    dom_m = utils.AverageMeter()
    im_m = utils.AverageMeter()
    top1_m = utils.AverageMeter()

    for batch_idx, (src_img, tgt_img, label) in enumerate(paired_loader):
        src_img = src_img.to(device=device, dtype=model_dtype)
        tgt_img = tgt_img.to(device=device, dtype=model_dtype)
        label = label.to(device=device)

        optimizer.zero_grad()

        out_src = model(src_img)
        src_logits = base.spatial_attn._logits
        src_cls = F.cross_entropy(out_src, label)
        src_dom = compute_domain_loss(src_logits, 0)
        src_im = compute_im_loss(src_logits)
        src_loss = src_cls + lam_dom * src_dom + lam_im * src_im
        src_loss.backward()

        out_tgt = model(tgt_img)
        tgt_logits = base.spatial_attn._logits
        tgt_cls = F.cross_entropy(out_tgt, label)
        tgt_dom = compute_domain_loss(tgt_logits, N - 1)
        tgt_im = compute_im_loss(tgt_logits)
        tgt_loss = tgt_cls + lam_dom * tgt_dom + lam_im * tgt_im
        tgt_loss.backward()

        if args.clip_grad is not None:
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            nn.utils.clip_grad_norm_(trainable_params, args.clip_grad)
        optimizer.step()

        total = (src_loss.item() + tgt_loss.item()) / 2
        cls_val = (src_cls.item() + tgt_cls.item()) / 2
        dom_val = (src_dom.item() + tgt_dom.item()) / 2
        im_val = (src_im.item() + tgt_im.item()) / 2
        acc1, = utils.accuracy(out_tgt.detach(), label, topk=(1,))
        bs = src_img.shape[0] + tgt_img.shape[0]
        losses_m.update(total, bs)
        cls_m.update(cls_val, bs)
        dom_m.update(dom_val, bs)
        im_m.update(im_val, bs)
        top1_m.update(acc1.item(), tgt_img.shape[0])

        if batch_idx % args.log_interval == 0:
            phase = 'WU' if in_warmup else 'FL'
            _logger.info(
                f'    [{batch_idx:>4d}/{len(paired_loader)}]({phase})  '
                f'loss={losses_m.val:.4f}({losses_m.avg:.4f})  '
                f'cls={cls_m.val:.4f}  dom={dom_m.val:.4f}  im={im_m.val:.4f}  '
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

parser = argparse.ArgumentParser(description='FedAvg — Parallel Attention + AA Policy Training')

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
group.add_argument('--model', default='vit_base_patch16_224', type=str)
group.add_argument('--resume', type=str,
                   default='./VIT_IMG_PAR/Normal_parallel_train_1_kernel_size_2/model_best.pth.tar')
group.add_argument('--num-classes', type=int, default=1000)
group.add_argument('--input-size', default=[3, 224, 224], nargs=3, type=int)
group.add_argument('--parallel-attention', action='store_true', default=True)
group.add_argument('--vit-kernel-size', type=int, default=2)
group.add_argument('--spatial-group-size', type=int, default=1)
group.add_argument('--sam-norm-type', type=int, default=0, choices=[0, 1, 2, 3, 4],
                   help='SpatialAttention norm: 0=Identity, 1=BN, 2=IN(affine), 3=IN, 4=GN')
group.add_argument('--vit-last', action='store_true', default=False)
group.add_argument('--vit-closed', type=str, default=None, choices=['same', 'diff'])
group.add_argument('--use-se-module', action='store_true', default=False)
group.add_argument('--use-sam-module', type=int, default=-1)
group.add_argument('--reverse-se', action='store_true', default=False)
group.add_argument('--train-mode', type=int, default=1, choices=[0, 1, 2],
                   help='0=all, 1=aux+embedding, 2=aux only')

group = parser.add_argument_group('SpatialAttention2')
group.add_argument('--prop-size', type=int, default=0.5,
                   help='N-class spatial attention prop_size (0=use original SpatialAttention, >0=SpatialAttention2)')
group.add_argument('--lambda-domain', type=float, default=0.5,
                   help='Weight for domain CE loss (default: 0.5)')
group.add_argument('--lambda-im', type=float, default=0.1,
                   help='Weight for Information Maximization loss (default: 0.1)')
group.add_argument('--warmup-rounds', type=int, default=10,
                   help='Number of CE-only warm-up rounds before enabling domain+IM (default: 10)')

group = parser.add_argument_group('FedAvg')
group.add_argument('--rounds', type=int, default=10)
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
                   help='Output directory (default: ./output/fedavg_direct_k{vit_kernel_size}_tm{train_mode})')

# Compatibility
group = parser.add_argument_group('Compatibility (unused)')
group.add_argument('--drop', type=float, default=0.0)
group.add_argument('--drop-path', type=float, default=None)
group.add_argument('--drop-block', type=float, default=None)
group.add_argument('--bn-momentum', type=float, default=None)
group.add_argument('--bn-eps', type=float, default=None)
group.add_argument('--vit-early-norm-types', type=int, nargs=4, default=None, choices=[0,1,2,3,4])


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
    use_sa2 = args.prop_size > 0

    if args.output_dir is None:
        norm_suffix = f'_sn{args.sam_norm_type}' if args.sam_norm_type != 0 else ''
        swa_suffix = '_swa' if args.swa else ''
        policy_suffix = f'_aa{len(aa_policies)}' if aa_policies else ''
        sa2_suffix = f'_p{args.prop_size}' if use_sa2 else ''
        wu_suffix = f'_wu{args.warmup_rounds}' if use_sa2 and args.warmup_rounds > 0 else ''
        args.output_dir = (f'./output/fedavg_direct_k{args.vit_kernel_size}'
                           f'_tm{args.train_mode}_le{args.local_epochs}'
                           f'{norm_suffix}{sa2_suffix}{wu_suffix}{policy_suffix}{swa_suffix}')

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
    vit_norm_kwargs = {
        'sam_kernel_size': args.vit_kernel_size,
        'spatial_group_size': args.spatial_group_size,
    }
    if args.vit_last:
        vit_norm_kwargs['vit_last'] = True
    if args.vit_closed is not None:
        vit_norm_kwargs['vit_closed'] = args.vit_closed
    if args.vit_early_norm_types is not None:
        vit_norm_kwargs['vit_early_norm_types'] = args.vit_early_norm_types

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
        **vit_norm_kwargs,
    )
    model.to(device=device, dtype=model_dtype)

    if args.resume:
        _logger.info(f'Loading checkpoint: {args.resume}')
        ckpt = torch.load(args.resume, map_location='cpu')
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            _logger.info(f'  Missing keys ({len(missing)}): {missing}')
        if unexpected:
            _logger.info(f'  Unexpected keys ({len(unexpected)}): {unexpected}')

    # ── Replace SpatialAttention with SpatialAttention2 if prop_size > 0 ──
    if use_sa2:
        base = _get_base_model(model)
        ch = base.embed_dim
        base.spatial_attn = SpatialAttention2(
            kernel_size=args.vit_kernel_size, channels=ch,
            prop_size=args.prop_size,
        ).to(device=device, dtype=model_dtype)
        _logger.info(f'SpatialAttention2 installed: prop_size={args.prop_size}, '
                     f'channels={ch}, kernel={args.vit_kernel_size}')

    trainable_count = _apply_train_mode(model, args.train_mode)
    total_params = sum(p.numel() for p in model.parameters())
    _logger.info(f'Model loaded: {safe_model_name(args.model)}, '
                 f'trainable: {trainable_count:,}/{total_params:,} '
                 f'(train_mode={args.train_mode})')

    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    _logger.info(f'Trainable parameter groups ({len(trainable_names)}):')
    for n in trainable_names:
        _logger.info(f'  {n}')

    data_config = resolve_data_config(vars(args) | {'pretrained': False}, model=model, verbose=True)

    # ── Output dir ──
    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, 'summary.csv')

    # ── Validate AA policies ──
    if aa_policies:
        _logger.info(f'Validating {len(aa_policies)} AA policies...')
        validate_aa_policies(aa_policies, img_size=args.input_size[1])
        for i, p in enumerate(aa_policies):
            _logger.info(f'  Policy {i}: {p}')

    total_clients = len(CORRUPTIONS) + len(aa_policies)

    # ── SWA setup ──
    swa_model = None
    swa_start = max(1, int(args.rounds * args.swa_start_frac) + 1)
    if args.swa:
        swa_model = AveragedModel(model, device=device)
        _logger.info(f'SWA enabled: averaging from round {swa_start}/{args.rounds}')

    # ── FedAvg training loop ──
    _logger.info(f'\nStarting FedAvg: {args.rounds} rounds, '
                 f'{len(CORRUPTIONS)} corruption + {len(aa_policies)} policy = '
                 f'{total_clients} clients, '
                 f'local_epochs={args.local_epochs}, lr={args.lr}→{args.min_lr} (cosine)')

    best_mean_acc1 = 0.0

    for rnd in range(1, args.rounds + 1):
        round_start = time.time()

        current_lr = cosine_lr(args.lr, args.min_lr, rnd - 1, args.rounds)
        _logger.info(f'\n{"="*60}')
        _logger.info(f'  Round {rnd}/{args.rounds}  (lr={current_lr:.6f})')
        _logger.info(f'{"="*60}')

        global_state = get_trainable_state(model)
        client_states = []

        # ── Corruption clients ──
        for ci, corruption in enumerate(CORRUPTIONS):
            set_trainable_state(model, global_state)

            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(
                trainable_params, lr=current_lr, weight_decay=args.weight_decay,
            )

            if use_sa2:
                paired_loader = create_paired_corruption_loader(
                    corruption, args, data_config)
                for local_ep in range(args.local_epochs):
                    loss, acc1 = local_train_one_epoch_dual(
                        model, paired_loader, optimizer,
                        args, device, model_dtype, current_round=rnd,
                    )
            else:
                loader = create_client_loader(corruption, args, data_config, model_dtype, device)
                for local_ep in range(args.local_epochs):
                    loss, acc1 = local_train_one_epoch(
                        model, loader, optimizer, args, device, model_dtype,
                    )

            _logger.info(f'  Client {ci:>2d} [{corruption:<22s}]  '
                         f'loss={loss:.4f}  acc@1={acc1:.2f}%')

            client_states.append(get_trainable_state(model))

        # ── AA policy clients ──
        for pi, policy in enumerate(aa_policies):
            set_trainable_state(model, global_state)

            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(
                trainable_params, lr=current_lr, weight_decay=args.weight_decay,
            )

            if use_sa2:
                paired_loader = create_paired_policy_loader(
                    policy, args, data_config)
                for local_ep in range(args.local_epochs):
                    loss, acc1 = local_train_one_epoch_dual(
                        model, paired_loader, optimizer,
                        args, device, model_dtype, current_round=rnd,
                    )
            else:
                loader = create_policy_client_loader(policy, args, data_config, model_dtype, device)
                for local_ep in range(args.local_epochs):
                    loss, acc1 = local_train_one_epoch(
                        model, loader, optimizer, args, device, model_dtype,
                    )

            ci_total = len(CORRUPTIONS) + pi
            _logger.info(f'  Client {ci_total:>2d} [aa:{policy:<19s}]  '
                         f'loss={loss:.4f}  acc@1={acc1:.2f}%')

            client_states.append(get_trainable_state(model))

        # FedAvg aggregation
        aggregated = fedavg_aggregate(global_state, client_states)
        set_trainable_state(model, aggregated)

        round_time = time.time() - round_start
        _logger.info(f'  Aggregated {len(client_states)} clients ({round_time:.1f}s)')

        if swa_model is not None and rnd >= swa_start:
            swa_model.update_parameters(model)
            _logger.info(f'  SWA updated (n_averaged={swa_model.n_averaged.item()})')

        # ── Evaluate ──
        _logger.info(f'\n  Evaluating round {rnd}...')
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
        _logger.info(f'  Round {rnd} Mean Acc@1: {mean_acc1:.3f}%  (lr={current_lr:.6f})')

        # ── Write summary CSV ──
        write_header = (rnd == 1)
        with open(summary_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                header = ['round', 'lr', 'mean_acc1'] + list(CORRUPTIONS)
                writer.writerow(header)
            row = [rnd, f'{current_lr:.6f}', f'{mean_acc1:.3f}']
            for c in CORRUPTIONS:
                row.append(f'{eval_results[c]["top1"]:.3f}')
            writer.writerow(row)

        # ── Save checkpoint ──
        ckpt_data = {
            'round': rnd,
            'state_dict': get_trainable_state(model),
            'mean_acc1': mean_acc1,
            'lr': current_lr,
            'args': vars(args),
        }

        ckpt_path = os.path.join(args.output_dir, f'round_{rnd:03d}.pth')
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
        _logger.info(f'  SWA evaluation (averaged rounds {swa_start}–{args.rounds}, '
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
            row = ['swa', '', f'{swa_mean:.3f}']
            for c in CORRUPTIONS:
                row.append(f'{swa_eval[c]["top1"]:.3f}')
            writer.writerow(row)

        swa_ckpt = {
            'round': f'swa_{swa_start}-{args.rounds}',
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
