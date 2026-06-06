#!/usr/bin/env python3
"""FedAvg Training — AuxNet2 for Per-Sample Normalization Affine Prediction

Given a pre-trained model with Parallel Channel/Spatial Attention (from a
previous FedAvg round), this script freezes the entire backbone and trains
a lightweight auxiliary network (AuxNet2) that:

  1. Takes channel_attn (B, C, 1, 1) and spatial_mask (B, C, H, W)
     from the frozen Parallel Attention modules.
  2. Predicts 12 affine parameters (scale + bias for each of the 6
     normalization values: mean_R, mean_G, mean_B, std_R, std_G, std_B).
  3. Re-normalizes the input images with the predicted mean/std and
     re-runs the frozen model for classification.

Training uses FedAvg across 15 ImageNet-C corruption clients with
cross-entropy loss.  Only AuxNet2 parameters are updated.
"""
import argparse
import csv
import importlib
import logging
import math
import os
import time
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from timm import utils
from timm.data import create_dataset, create_loader, resolve_data_config
from timm.models import create_model, safe_model_name

_logger = logging.getLogger('fedavg_auxnet2')

CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
    'snow', 'frost', 'fog', 'brightness',
    'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
]


# ---------------------------------------------------------------------------
#  Utilities
# ---------------------------------------------------------------------------

def _get_base_model(model):
    m = model
    if hasattr(m, 'module'):
        m = m.module
    if hasattr(m, '_orig_mod'):
        m = m._orig_mod
    return m


def cosine_lr(base_lr, min_lr, current_round, total_rounds):
    return min_lr + 0.5 * (base_lr - min_lr) * (
        1 + math.cos(math.pi * current_round / total_rounds))


def _freeze_bn(model):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                          nn.SyncBatchNorm)):
            m.eval()


# ---------------------------------------------------------------------------
#  Model creation helpers (shared with eval_fedavg.py)
# ---------------------------------------------------------------------------

def create_model_from_ckpt_args(ckpt_args, device, model_dtype=None):
    model_name = ckpt_args['model']
    is_vit = 'vit' in model_name.lower()

    model_kwargs = {}
    if ckpt_args.get('use_sam_module', -1) != -1 or ckpt_args.get('parallel_attention', False):
        model_kwargs['sam_kernel_size'] = ckpt_args.get('vit_kernel_size', 7)
        model_kwargs['spatial_group_size'] = ckpt_args.get('spatial_group_size', 1)

    if is_vit:
        if ckpt_args.get('vit_last', False):
            model_kwargs['vit_last'] = True
        if ckpt_args.get('vit_closed') is not None:
            model_kwargs['vit_closed'] = ckpt_args['vit_closed']
        if ckpt_args.get('vit_early_norm_types') is not None:
            model_kwargs['vit_early_norm_types'] = ckpt_args['vit_early_norm_types']

    in_chans = ckpt_args.get('input_size', [3, 224, 224])[0]

    model = create_model(
        model_name,
        num_classes=ckpt_args.get('num_classes', 1000),
        in_chans=in_chans,
        parallel_attention=ckpt_args.get('parallel_attention', True),
        use_se_module=ckpt_args.get('use_se_module', False),
        use_sam_module=ckpt_args.get('use_sam_module', -1),
        reverse_se_sam=ckpt_args.get('reverse_se', False),
        sam_norm_type=ckpt_args.get('sam_norm_type', 0),
        **model_kwargs,
    )
    model.to(device=device, dtype=model_dtype)
    return model


def load_model_with_fedavg(model, backbone_path, fedavg_state):
    if backbone_path:
        _logger.info(f'Loading backbone: {backbone_path}')
        ckpt = torch.load(backbone_path, map_location='cpu')
        sd = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            _logger.info(f'  Missing keys: {len(missing)}')
        if unexpected:
            _logger.info(f'  Unexpected keys: {len(unexpected)}')

    n = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in fedavg_state:
                param.data.copy_(fedavg_state[name])
                n += 1
    _logger.info(f'Overlaid {n} FedAvg-trained parameters')


# ---------------------------------------------------------------------------
#  Attention hook & architecture detection
# ---------------------------------------------------------------------------

def register_attention_hooks(model, use_spatial_logits=False):
    """Register forward hooks to capture Parallel Attention outputs.

    If use_spatial_logits is True, capture pre-softmax logits from
    spatial_attn.norm (shape: B, num_groups*2, H, W) instead of the
    post-softmax expanded mask.
    """
    base = _get_base_model(model)
    cache = {}

    def _ch_hook(module, inp, out):
        cache['channel'] = out.detach()

    def _sp_mask_hook(module, inp, out):
        _, mask = out
        cache['spatial'] = mask.detach()

    def _sp_logit_hook(module, inp, out):
        cache['spatial'] = out.detach()

    handles = [
        base.channel_attn.register_forward_hook(_ch_hook),
    ]
    if use_spatial_logits:
        handles.append(
            base.spatial_attn.norm.register_forward_hook(_sp_logit_hook))
    else:
        handles.append(
            base.spatial_attn.register_forward_hook(_sp_mask_hook))

    return cache, handles


def get_attention_dims(model, input_size, use_spatial_logits=False):
    """Return (channels, spatial_channels, spatial_size).

    channels       : channel_attn output dim (C)
    spatial_channels: spatial input channels to AuxNet2
        - use_spatial_logits=False → C  (expanded mask)
        - use_spatial_logits=True  → num_groups * 2  (pre-softmax logits)
    spatial_size   : H (= W) of spatial feature map
    """
    base = _get_base_model(model)
    num_groups = base.spatial_attn.num_groups

    if hasattr(base, 'patch_embed'):
        channels = base.embed_dim
        ps = base.patch_embed.patch_size
        ps = ps[0] if isinstance(ps, (tuple, list)) else ps
        spatial_size = input_size[1] // ps
    else:
        if isinstance(base.conv1, nn.Sequential):
            channels = base.conv1[-1].out_channels
        else:
            channels = base.conv1.out_channels
        spatial_size = input_size[1] // 2

    spatial_channels = num_groups * 2 if use_spatial_logits else channels
    return channels, spatial_channels, spatial_size


# ---------------------------------------------------------------------------
#  AuxNet2
# ---------------------------------------------------------------------------

class AuxNet2(nn.Module):
    """Predict per-sample normalization parameters (mean, std) via AdaIN-style
    output from Parallel Attention features.

    mean : sigmoid → [0, 1]          (valid pixel-mean range)
    std  : sigmoid × 0.5 → [0, 0.5]  (valid pixel-std range)

    Spatial input modes (--use-spatial-logits):
      False (default) : post-softmax mask (B, C, H, W), downsampled via
                         depthwise conv.
      True            : pre-softmax logits (B, num_groups*2, H, W), downsampled
                         via regular conv — richer gradient information.

    Optional LayerNorm (--use-aux-norm) before the FC backbone normalises
    unbounded logit scales across corruptions.
    """

    def __init__(self, channels, spatial_channels, spatial_size,
                 use_spatial_logits=False, use_aux_norm=False):
        super().__init__()

        ds = []
        s = spatial_size
        while s > 16:
            if use_spatial_logits:
                ds.append(nn.Conv2d(spatial_channels, spatial_channels, 3,
                                    stride=2, padding=1, bias=False))
            else:
                ds.append(nn.Conv2d(spatial_channels, spatial_channels, 3,
                                    stride=2, padding=1,
                                    groups=spatial_channels, bias=False))
            ds.append(nn.GELU())
            s = (s + 1) // 2
        self.spatial_down = nn.Sequential(*ds) if ds else nn.Identity()

        fc_in = channels + spatial_channels
        self.norm = nn.LayerNorm(fc_in) if use_aux_norm else nn.Identity()

        self.backbone = nn.Sequential(
            nn.Linear(fc_in, 64),
            nn.GELU(),
        )
        self.mean_head = nn.Linear(64, 3)
        self.std_head = nn.Linear(64, 3)

    def forward(self, channel_attn, spatial_feat):
        """
        channel_attn : (B, C, 1, 1)
        spatial_feat : (B, spatial_channels, H, W)
        returns      : new_mean (B, 3) ∈ [0, 1],
                       new_std  (B, 3) ∈ (0, 0.5]
        """
        s = self.spatial_down(spatial_feat)
        s_feat = s.mean(dim=(2, 3))
        c_feat = channel_attn.flatten(1)
        feat = self.norm(torch.cat([s_feat, c_feat], dim=1))
        feat = self.backbone(feat)

        new_mean = torch.sigmoid(self.mean_head(feat))
        new_std = torch.sigmoid(self.std_head(feat)) * 0.5
        return new_mean, new_std


# ---------------------------------------------------------------------------
#  Normalization helpers
# ---------------------------------------------------------------------------

def renormalize(images, old_mean, old_std, new_mean, new_std):
    """Convert images from (old_mean, old_std) normalisation to (new_mean, new_std).

    Algebraic trick — no raw-image loading required:
        raw = img * old_std + old_mean
        out = (raw - new_mean) / new_std
    """
    device = images.device
    om = torch.tensor(old_mean, device=device, dtype=images.dtype).view(1, 3, 1, 1)
    os_ = torch.tensor(old_std, device=device, dtype=images.dtype).view(1, 3, 1, 1)
    nm = new_mean.view(-1, 3, 1, 1)
    ns = new_std.view(-1, 3, 1, 1)
    return (images * os_ + om - nm) / ns


# ---------------------------------------------------------------------------
#  Data loaders
# ---------------------------------------------------------------------------

def create_client_loader(corruption, args, data_config, model_dtype, device):
    data_dir = os.path.join(args.data_root, corruption, str(args.severity))
    img_mode = args.input_img_mode or (
        'RGB' if data_config['input_size'][0] == 3 else 'L')

    dataset = create_dataset(
        '', root=data_dir, split=args.val_split, is_training=True,
        class_map='', download=False,
        batch_size=args.batch_size, seed=args.seed,
        input_img_mode=img_mode,
    )
    return create_loader(
        dataset, input_size=data_config['input_size'],
        batch_size=args.batch_size, is_training=True, no_aug=True,
        num_workers=args.workers,
        mean=data_config['mean'], std=data_config['std'],
        pin_memory=args.pin_mem,
        img_dtype=model_dtype or torch.float32,
        device=device, distributed=False,
        use_prefetcher=not args.no_prefetcher,
    )


def create_eval_loader(corruption, args, data_config, model_dtype, device):
    data_dir = os.path.join(args.data_root, corruption, str(args.severity))
    img_mode = args.input_img_mode or (
        'RGB' if data_config['input_size'][0] == 3 else 'L')

    dataset = create_dataset(
        '', root=data_dir, split=args.val_split, is_training=False,
        batch_size=args.batch_size, input_img_mode=img_mode,
    )
    return create_loader(
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


# ---------------------------------------------------------------------------
#  Train / eval loops
# ---------------------------------------------------------------------------

def local_train_one_epoch(model, aux_net, loader, optimizer,
                          attn_cache, args, device, model_dtype, data_config):
    aux_net.train()
    model.eval()

    default_mean = data_config['mean']
    default_std = data_config['std']
    prefetcher = not args.no_prefetcher

    losses_m = utils.AverageMeter()
    top1_m = utils.AverageMeter()

    for batch_idx, (images, target) in enumerate(loader):
        if not prefetcher:
            images = images.to(device=device, dtype=model_dtype)
            target = target.to(device=device)

        # Pass 1 (no grad): capture attention maps via hooks
        with torch.no_grad():
            _ = model(images)

        # AuxNet2 predicts new mean/std via AdaIN-style output
        new_mean, new_std = aux_net(attn_cache['channel'], attn_cache['spatial'])

        # Re-normalize and run Pass 2 (grad flows through renorm -> aux_net)
        images_re = renormalize(images, default_mean, default_std,
                                new_mean, new_std)
        output = model(images_re)
        if isinstance(output, (tuple, list)):
            output = output[0]
        loss = F.cross_entropy(output, target)

        optimizer.zero_grad()
        loss.backward()
        if args.clip_grad is not None:
            nn.utils.clip_grad_norm_(aux_net.parameters(), args.clip_grad)
        optimizer.step()

        acc1, = utils.accuracy(output.detach(), target, topk=(1,))
        losses_m.update(loss.item(), images.shape[0])
        top1_m.update(acc1.item(), images.shape[0])

        if batch_idx % args.log_interval == 0:
            dm = torch.tensor(default_mean, device=device)
            ds = torch.tensor(default_std, device=device)
            _logger.info(
                f'    [{batch_idx:>4d}/{len(loader)}]  '
                f'loss={losses_m.val:.4f}({losses_m.avg:.4f})  '
                f'acc@1={top1_m.val:.2f}%({top1_m.avg:.2f}%)  '
                f'μ={new_mean.mean(0).tolist()}  '
                f'σ={new_std.mean(0).tolist()}')

    return losses_m.avg, top1_m.avg


@torch.no_grad()
def evaluate_with_auxnet(model, aux_net, loader, attn_cache,
                         device, model_dtype, data_config, no_prefetcher):
    aux_net.eval()
    model.eval()

    default_mean = data_config['mean']
    default_std = data_config['std']

    losses_m = utils.AverageMeter()
    top1_m = utils.AverageMeter()
    top5_m = utils.AverageMeter()

    for images, target in loader:
        if no_prefetcher:
            images = images.to(device=device, dtype=model_dtype)
            target = target.to(device=device)

        _ = model(images)
        new_mean, new_std = aux_net(attn_cache['channel'], attn_cache['spatial'])
        images_re = renormalize(images, default_mean, default_std,
                                new_mean, new_std)

        output = model(images_re)
        if isinstance(output, (tuple, list)):
            output = output[0]
        loss = F.cross_entropy(output, target)
        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

        losses_m.update(loss.item(), images.shape[0])
        top1_m.update(acc1.item(), images.shape[0])
        top5_m.update(acc5.item(), images.shape[0])

    return losses_m.avg, top1_m.avg, top5_m.avg


# ---------------------------------------------------------------------------
#  Argument parsing
# ---------------------------------------------------------------------------

config_parser = parser = argparse.ArgumentParser(
    description='Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE')

parser = argparse.ArgumentParser(
    description='FedAvg — AuxNet2 Normalization Affine Training')

g = parser.add_argument_group('Checkpoint')
g.add_argument('--fedavg-ckpt', type=str, required=True,
               help='Path to best.pth from prior FedAvg training')
g.add_argument('--resume', type=str, default=None,
               help='Override backbone checkpoint path')

g = parser.add_argument_group('Data')
g.add_argument('--data-root', type=str, default='/home/oem/servers/imagenet-c')
g.add_argument('--severity', type=int, default=5)
g.add_argument('--val-split', type=str, default='validation')
g.add_argument('--input-img-mode', default=None, type=str)

g = parser.add_argument_group('AuxNet2')
g.add_argument('--use-spatial-logits', action='store_true', default=False,
               help='Use pre-softmax spatial logits instead of post-softmax mask')
g.add_argument('--use-aux-norm', action='store_true', default=False,
               help='Apply LayerNorm before FC backbone in AuxNet2')

g = parser.add_argument_group('FedAvg')
g.add_argument('--rounds', type=int, default=10)
g.add_argument('--local-epochs', type=int, default=1)

g = parser.add_argument_group('Optimizer')
g.add_argument('--lr', type=float, default=1e-3)
g.add_argument('--min-lr', type=float, default=1e-5)
g.add_argument('--weight-decay', type=float, default=1e-4)
g.add_argument('--clip-grad', type=float, default=1.0)

g = parser.add_argument_group('Training')
g.add_argument('-b', '--batch-size', type=int, default=64)
g.add_argument('--device', default='cuda', type=str)
g.add_argument('--seed', type=int, default=42)
g.add_argument('-j', '--workers', type=int, default=4)
g.add_argument('--pin-mem', action='store_true', default=False)
g.add_argument('--no-prefetcher', action='store_true', default=False)
g.add_argument('--log-interval', type=int, default=50)
g.add_argument('--device-modules', default=None, type=str, nargs='+')

g = parser.add_argument_group('Output')
g.add_argument('--output-dir', type=str, default=None)


def _parse_args():
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)
    return parser.parse_args(remaining)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    utils.setup_default_logging()
    args = _parse_args()

    if args.device_modules:
        for mod in args.device_modules:
            importlib.import_module(mod)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)
    utils.random_seed(args.seed, 0)

    # ── Load FedAvg checkpoint & build frozen model ──
    _logger.info(f'Loading FedAvg checkpoint: {args.fedavg_ckpt}')
    fedavg_ckpt = torch.load(args.fedavg_ckpt, map_location='cpu')
    ckpt_args = fedavg_ckpt['args']

    model = create_model_from_ckpt_args(ckpt_args, device)
    resume_path = args.resume or ckpt_args.get('resume')
    load_model_with_fedavg(model, resume_path, fedavg_ckpt['state_dict'])

    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    _freeze_bn(model)

    total_params = sum(p.numel() for p in model.parameters())
    _logger.info(f'Backbone frozen: {total_params:,} params (all requires_grad=False)')

    # ── Create AuxNet2 ──
    input_size = ckpt_args.get('input_size', [3, 224, 224])
    channels, spatial_channels, spatial_size = get_attention_dims(
        model, input_size, use_spatial_logits=args.use_spatial_logits)

    aux_net = AuxNet2(
        channels, spatial_channels, spatial_size,
        use_spatial_logits=args.use_spatial_logits,
        use_aux_norm=args.use_aux_norm,
    )
    aux_net.to(device)

    sp_mode = 'logits' if args.use_spatial_logits else 'mask'
    aux_params = sum(p.numel() for p in aux_net.parameters())
    _logger.info(
        f'AuxNet2: ch={channels}, spatial({sp_mode})={spatial_channels}ch×{spatial_size}, '
        f'norm={"LN" if args.use_aux_norm else "none"}, params={aux_params:,}')

    # ── Attention hooks ──
    attn_cache, hooks = register_attention_hooks(
        model, use_spatial_logits=args.use_spatial_logits)

    data_config = resolve_data_config(
        {'input_size': input_size, 'pretrained': False}, model=model, verbose=True)

    # ── Output directory ──
    if args.output_dir is None:
        model_tag = safe_model_name(ckpt_args['model'])
        suffixes = []
        if args.use_spatial_logits:
            suffixes.append('logit')
        if args.use_aux_norm:
            suffixes.append('ln')
        suffix = '_' + '_'.join(suffixes) if suffixes else ''
        args.output_dir = f'./output/fedavg_auxnet2_{model_tag}{suffix}'
    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, 'summary.csv')

    # ── FedAvg training loop ──
    _logger.info(f'\nStarting FedAvg AuxNet2: {args.rounds} rounds, '
                 f'{len(CORRUPTIONS)} clients, '
                 f'local_epochs={args.local_epochs}, '
                 f'lr={args.lr}→{args.min_lr} (cosine)')

    best_mean_acc1 = 0.0

    for rnd in range(1, args.rounds + 1):
        round_start = time.time()
        current_lr = cosine_lr(args.lr, args.min_lr, rnd - 1, args.rounds)

        _logger.info(f'\n{"="*60}')
        _logger.info(f'  Round {rnd}/{args.rounds}  (lr={current_lr:.6f})')
        _logger.info(f'{"="*60}')

        global_state = OrderedDict(
            (n, p.data.clone()) for n, p in aux_net.named_parameters())
        client_states = []

        for ci, corruption in enumerate(CORRUPTIONS):
            with torch.no_grad():
                for n, p in aux_net.named_parameters():
                    p.data.copy_(global_state[n])

            optimizer = torch.optim.AdamW(
                aux_net.parameters(),
                lr=current_lr,
                weight_decay=args.weight_decay,
            )

            loader = create_client_loader(
                corruption, args, data_config, None, device)

            for _ in range(args.local_epochs):
                loss, acc1 = local_train_one_epoch(
                    model, aux_net, loader, optimizer,
                    attn_cache, args, device, None, data_config)

            _logger.info(f'  Client {ci:>2d} [{corruption:<22s}]  '
                         f'loss={loss:.4f}  acc@1={acc1:.2f}%')

            client_states.append(OrderedDict(
                (n, p.data.clone()) for n, p in aux_net.named_parameters()))

        # ── FedAvg aggregation ──
        agg = OrderedDict()
        for key in global_state:
            agg[key] = sum(cs[key] for cs in client_states) / len(client_states)
        with torch.no_grad():
            for n, p in aux_net.named_parameters():
                p.data.copy_(agg[n])

        round_time = time.time() - round_start
        _logger.info(f'  Aggregated {len(client_states)} clients ({round_time:.1f}s)')

        # ── Evaluate ──
        _logger.info(f'\n  Evaluating round {rnd}...')
        eval_results = OrderedDict()
        acc1_sum = 0.0

        for corruption in CORRUPTIONS:
            eval_loader = create_eval_loader(
                corruption, args, data_config, None, device)
            _, eval_acc1, eval_acc5 = evaluate_with_auxnet(
                model, aux_net, eval_loader, attn_cache,
                device, None, data_config, args.no_prefetcher)
            eval_results[corruption] = {
                'top1': eval_acc1, 'top5': eval_acc5}
            acc1_sum += eval_acc1
            _logger.info(
                f'    {corruption:<22s}  '
                f'Acc@1={eval_acc1:.3f}%  Acc@5={eval_acc5:.3f}%')

        mean_acc1 = acc1_sum / len(CORRUPTIONS)
        _logger.info(
            f'  Round {rnd} Mean Acc@1: {mean_acc1:.3f}%  '
            f'(lr={current_lr:.6f})')

        # ── Summary CSV ──
        write_header = (rnd == 1)
        with open(summary_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(
                    ['round', 'lr', 'mean_acc1'] + list(CORRUPTIONS))
            row = [rnd, f'{current_lr:.6f}', f'{mean_acc1:.3f}']
            for c in CORRUPTIONS:
                row.append(f'{eval_results[c]["top1"]:.3f}')
            writer.writerow(row)

        # ── Save checkpoint ──
        ckpt_data = {
            'round': rnd,
            'aux_net_state_dict': OrderedDict(
                (n, p.data.clone()) for n, p in aux_net.named_parameters()),
            'mean_acc1': mean_acc1,
            'lr': current_lr,
            'args': vars(args),
            'aux_channels': channels,
            'aux_spatial_size': spatial_size,
        }

        ckpt_path = os.path.join(args.output_dir, f'round_{rnd:03d}.pth')
        torch.save(ckpt_data, ckpt_path)

        if mean_acc1 > best_mean_acc1:
            best_mean_acc1 = mean_acc1
            best_path = os.path.join(args.output_dir, 'best.pth')
            torch.save(ckpt_data, best_path)
            _logger.info(
                f'  New best! Mean Acc@1: {mean_acc1:.3f}% → {best_path}')
        else:
            _logger.info(f'  Checkpoint saved: {ckpt_path}')

    for h in hooks:
        h.remove()

    _logger.info(f'\n{"="*60}')
    _logger.info(f'  Training complete. Summary: {summary_path}')
    _logger.info(f'  Best Mean Acc@1: {best_mean_acc1:.3f}%')
    _logger.info(f'{"="*60}')


if __name__ == '__main__':
    main()
