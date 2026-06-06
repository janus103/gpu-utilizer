#!/usr/bin/env python3
"""Analyze SpatialAttention2 5-class distribution across clean and corrupted domains.

Measures the softmax probability distribution from SpatialAttention2._logits
for source (clean ImageNet) and each of 15 ImageNet-C corruptions.
Outputs a CSV with per-domain statistics and generates matplotlib visualizations.
"""
import argparse
import csv
import importlib
import logging
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from timm import utils
from timm.data import create_dataset, create_loader, resolve_data_config
from timm.models import create_model, safe_model_name
from timm.models.vision_transformer import SpatialAttention2

_logger = logging.getLogger('sa2_analysis')

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
    embedding_prefixes = (
        'patch_embed', 'pos_embed', 'cls_token', 'reg_token',
        'norm_pre', 'conv1', 'bn1', 'act1',
    )
    return name.startswith(embedding_prefixes)


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
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_trainable_state(model, state):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in state:
                param.data.copy_(state[name])


def create_corruption_loader(corruption, args, data_config, model_dtype, device):
    data_dir = os.path.join(args.data_root, corruption, str(args.severity))
    input_img_mode = args.input_img_mode or 'RGB'
    dataset = create_dataset(
        '', root=data_dir, split=args.val_split, is_training=False,
        class_map='', download=False,
        batch_size=args.batch_size, input_img_mode=input_img_mode,
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


def create_clean_loader(args, data_config, model_dtype, device):
    input_img_mode = args.input_img_mode or 'RGB'
    dataset = create_dataset(
        '', root=args.clean_data_dir, split=args.clean_split, is_training=False,
        class_map='', download=False,
        batch_size=args.batch_size, input_img_mode=input_img_mode,
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


@torch.no_grad()
def collect_sa2_stats(model, loader, device, model_dtype, base_model,
                      num_batches, prefetcher=True):
    """Run inference and accumulate SpatialAttention2 logit statistics."""
    model.eval()
    prop_size = base_model.spatial_attn.prop_size

    prob_sum = torch.zeros(prop_size, device=device)
    argmax_hist = torch.zeros(prop_size, device=device, dtype=torch.long)
    max_prob_sum = 0.0
    entropy_sum = 0.0
    total_positions = 0

    for batch_idx, (images, _target) in enumerate(loader):
        if batch_idx >= num_batches:
            break
        if not prefetcher:
            images = images.to(device=device, dtype=model_dtype)

        _ = model(images)
        logits = base_model.spatial_attn._logits  # (B, prop_size, H, W)
        probs = F.softmax(logits, dim=1)

        B, N, H, W = probs.shape
        n_pos = B * H * W

        prob_sum += probs.sum(dim=(0, 2, 3))  # (N,)
        argmax_classes = probs.argmax(dim=1)   # (B, H, W)
        for c in range(prop_size):
            argmax_hist[c] += (argmax_classes == c).sum()

        max_prob_sum += probs.max(dim=1).values.sum().item()

        log_probs = (probs + 1e-8).log()
        per_pos_entropy = -(probs * log_probs).sum(dim=1)  # (B, H, W)
        entropy_sum += per_pos_entropy.sum().item()

        total_positions += n_pos

        if (batch_idx + 1) % 10 == 0:
            _logger.info(f'    batch {batch_idx + 1}/{num_batches}')

    mean_probs = (prob_sum / total_positions).cpu().numpy()
    argmax_counts = argmax_hist.cpu().numpy()
    argmax_ratios = argmax_counts / argmax_counts.sum()
    mean_max_prob = max_prob_sum / total_positions
    mean_entropy = entropy_sum / total_positions

    p_marg = mean_probs
    marginal_entropy = -(p_marg * np.log(p_marg + 1e-8)).sum()

    return {
        'mean_probs': mean_probs,
        'argmax_counts': argmax_counts,
        'argmax_ratios': argmax_ratios,
        'mean_max_prob': mean_max_prob,
        'mean_entropy': mean_entropy,
        'marginal_entropy': marginal_entropy,
        'total_positions': total_positions,
    }


def write_csv(results, output_path, prop_size):
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['domain']
        header += [f'p_class{i}' for i in range(prop_size)]
        header += [f'hist_ratio{i}' for i in range(prop_size)]
        header += ['mean_max_prob', 'mean_entropy', 'marginal_entropy', 'total_positions']
        writer.writerow(header)

        for domain, stats in results.items():
            row = [domain]
            row += [f'{v:.6f}' for v in stats['mean_probs']]
            row += [f'{v:.6f}' for v in stats['argmax_ratios']]
            row += [
                f'{stats["mean_max_prob"]:.6f}',
                f'{stats["mean_entropy"]:.6f}',
                f'{stats["marginal_entropy"]:.6f}',
                str(stats['total_positions']),
            ]
            writer.writerow(row)


def plot_results(results, output_dir, prop_size):
    domains = list(results.keys())
    n_domains = len(domains)

    # --- 1. Stacked bar: mean class probability ---
    fig, ax = plt.subplots(figsize=(max(14, n_domains * 0.9), 6))
    x = np.arange(n_domains)
    bottoms = np.zeros(n_domains)
    colors = plt.cm.Set2(np.linspace(0, 1, prop_size))

    for c in range(prop_size):
        vals = [results[d]['mean_probs'][c] for d in domains]
        ax.bar(x, vals, bottom=bottoms, label=f'class {c}', color=colors[c], width=0.7)
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels(domains, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Mean probability')
    ax.set_title('SpatialAttention2: Mean class probability per domain')
    ax.legend(title='SA2 class', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'sa2_mean_probs_stacked.png'), dpi=150)
    plt.close(fig)

    # --- 2. Heatmap: argmax histogram ratios ---
    matrix = np.array([results[d]['argmax_ratios'] for d in domains])
    fig, ax = plt.subplots(figsize=(max(6, prop_size * 1.2), max(8, n_domains * 0.4)))
    im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd', vmin=0)
    ax.set_xticks(range(prop_size))
    ax.set_xticklabels([f'class {i}' for i in range(prop_size)])
    ax.set_yticks(range(n_domains))
    ax.set_yticklabels(domains, fontsize=9)
    ax.set_xlabel('SA2 argmax class')
    ax.set_title('SpatialAttention2: Argmax class ratio per domain')

    for i in range(n_domains):
        for j in range(prop_size):
            val = matrix[i, j]
            text_color = 'white' if val > 0.5 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=8, color=text_color)

    fig.colorbar(im, ax=ax, shrink=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'sa2_argmax_heatmap.png'), dpi=150)
    plt.close(fig)

    # --- 3. Bar chart: mean entropy and max prob ---
    fig, axes = plt.subplots(1, 2, figsize=(max(16, n_domains * 0.9), 5))

    ax1 = axes[0]
    vals_entropy = [results[d]['mean_entropy'] for d in domains]
    bar_colors = ['#2196F3' if d == 'clean' else '#FF9800' for d in domains]
    ax1.bar(x, vals_entropy, color=bar_colors, width=0.7)
    ax1.set_xticks(x)
    ax1.set_xticklabels(domains, rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('Mean per-position entropy')
    ax1.set_title('Per-position entropy (lower = sharper)')
    ax1.axhline(y=vals_entropy[0], color='#2196F3', linestyle='--', alpha=0.5, label='clean')
    ax1.legend()

    ax2 = axes[1]
    vals_maxp = [results[d]['mean_max_prob'] for d in domains]
    ax2.bar(x, vals_maxp, color=bar_colors, width=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(domains, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('Mean max probability (mask strength)')
    ax2.set_title('Mask strength (higher = more confident)')
    ax2.axhline(y=vals_maxp[0], color='#2196F3', linestyle='--', alpha=0.5, label='clean')
    ax2.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'sa2_entropy_maxprob.png'), dpi=150)
    plt.close(fig)

    # --- 4. Grouped bar: per-class probability comparison (clean vs each corruption) ---
    fig, ax = plt.subplots(figsize=(max(14, n_domains * 0.9), 6))
    bar_width = 0.15
    for c in range(prop_size):
        offsets = x + (c - prop_size / 2 + 0.5) * bar_width
        vals = [results[d]['mean_probs'][c] for d in domains]
        ax.bar(offsets, vals, width=bar_width, label=f'class {c}', color=colors[c])

    ax.set_xticks(x)
    ax.set_xticklabels(domains, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Mean probability')
    ax.set_title('SpatialAttention2: Per-class probability (grouped)')
    ax.legend(title='SA2 class', bbox_to_anchor=(1.02, 1), loc='upper left')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'sa2_grouped_probs.png'), dpi=150)
    plt.close(fig)

    _logger.info(f'Plots saved to {output_dir}')


def _parse_args():
    config_parser = argparse.ArgumentParser(description='Config', add_help=False)
    config_parser.add_argument('-c', '--config', default='', type=str, metavar='FILE')

    parser = argparse.ArgumentParser(
        description='Analyze SpatialAttention2 distribution across domains')

    group = parser.add_argument_group('Data')
    group.add_argument('--data-root', type=str, default='/home/oem/servers/imagenet-c')
    group.add_argument('--severity', type=int, default=5)
    group.add_argument('--val-split', type=str, default='validation')
    group.add_argument('--clean-data-dir', type=str, default='/data/imagenet/imagenet')
    group.add_argument('--clean-split', type=str, default='val')
    group.add_argument('--input-img-mode', default=None, type=str)

    group = parser.add_argument_group('Model')
    group.add_argument('--model', default='vit_base_patch16_224', type=str)
    group.add_argument('--resume', type=str,
                       default='./VIT_IMG_PAR/Normal_parallel_train_1_kernel_size_2/model_best.pth.tar')
    group.add_argument('--fedavg-ckpt', type=str, default=None)
    group.add_argument('--num-classes', type=int, default=1000)
    group.add_argument('--input-size', default=[3, 224, 224], nargs=3, type=int)
    group.add_argument('--parallel-attention', action='store_true', default=True)
    group.add_argument('--vit-kernel-size', type=int, default=2)
    group.add_argument('--spatial-group-size', type=int, default=1)
    group.add_argument('--vit-last', action='store_true', default=False)
    group.add_argument('--vit-closed', type=str, default=None, choices=['same', 'diff'])
    group.add_argument('--use-se-module', action='store_true', default=False)
    group.add_argument('--use-sam-module', type=int, default=-1)
    group.add_argument('--reverse-se', action='store_true', default=False)
    group.add_argument('--train-mode', type=int, default=1, choices=[0, 1, 2])
    group.add_argument('--sam-norm-type', type=int, default=0, choices=[0, 1, 2, 3, 4])
    group.add_argument('--vit-early-norm-types', type=int, nargs=4, default=None,
                       choices=[0, 1, 2, 3, 4])

    group = parser.add_argument_group('SpatialAttention2')
    group.add_argument('--prop-size', type=int, default=5)

    group = parser.add_argument_group('Runtime')
    group.add_argument('-b', '--batch-size', type=int, default=256)
    group.add_argument('-j', '--workers', type=int, default=4)
    group.add_argument('--device', default='cuda', type=str)
    group.add_argument('--model-dtype', default=None, type=str)
    group.add_argument('--pin-mem', action='store_true', default=False)
    group.add_argument('--no-prefetcher', action='store_true', default=False)
    group.add_argument('--seed', type=int, default=42)
    group.add_argument('--device-modules', default=None, type=str, nargs='+')
    group.add_argument('--num-batches', type=int, default=50,
                       help='Number of batches to process per domain (default: 50)')
    group.add_argument('--output-dir', type=str, default='./output/sa2_analysis',
                       help='Directory for CSV and plots')

    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    return parser.parse_args(remaining)


def main():
    utils.setup_default_logging()
    args = _parse_args()

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

    vit_norm_kwargs = {}
    if args.use_sam_module != -1 or args.parallel_attention:
        vit_norm_kwargs['sam_kernel_size'] = args.vit_kernel_size
        vit_norm_kwargs['spatial_group_size'] = args.spatial_group_size
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
        _logger.info(f'Loading base checkpoint: {args.resume}')
        ckpt = torch.load(args.resume, map_location='cpu')
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            _logger.info(f'  Missing keys ({len(missing)}): {missing}')
        if unexpected:
            _logger.info(f'  Unexpected keys ({len(unexpected)}): {unexpected}')

    use_sa2 = args.prop_size > 0
    if use_sa2:
        base = _get_base_model(model)
        ch = base.embed_dim
        base.spatial_attn = SpatialAttention2(
            kernel_size=args.vit_kernel_size, channels=ch,
            prop_size=args.prop_size,
        ).to(device=device, dtype=model_dtype)
        _logger.info(f'SpatialAttention2 installed: prop_size={args.prop_size}, '
                     f'channels={ch}, kernel={args.vit_kernel_size}')

    _apply_train_mode(model, args.train_mode)

    if args.fedavg_ckpt:
        _logger.info(f'Loading FedAvg checkpoint: {args.fedavg_ckpt}')
        ckpt = torch.load(args.fedavg_ckpt, map_location='cpu')
        fedavg_state = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        set_trainable_state(model, fedavg_state)
        if 'round' in ckpt:
            _logger.info(f'  Round: {ckpt["round"]}')
        if 'mean_acc1' in ckpt:
            _logger.info(f'  Saved mean_acc1: {ckpt["mean_acc1"]:.3f}%')

    base_model = _get_base_model(model)
    data_config = resolve_data_config(
        vars(args) | {'pretrained': False}, model=model, verbose=True,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    prefetcher = not args.no_prefetcher

    results = {}

    # --- Clean ---
    _logger.info(f'\n{"="*60}')
    _logger.info(f'Collecting stats: clean ({args.clean_data_dir}, split={args.clean_split})')
    _logger.info(f'{"="*60}')
    clean_loader = create_clean_loader(args, data_config, model_dtype, device)
    results['clean'] = collect_sa2_stats(
        model, clean_loader, device, model_dtype, base_model,
        args.num_batches, prefetcher=prefetcher,
    )
    mp = results['clean']['mean_probs']
    _logger.info(f'  clean mean_probs: {np.array2string(mp, precision=4)}')
    _logger.info(f'  clean mean_entropy: {results["clean"]["mean_entropy"]:.4f}')
    _logger.info(f'  clean mean_max_prob: {results["clean"]["mean_max_prob"]:.4f}')

    # --- 15 corruptions ---
    for corruption in CORRUPTIONS:
        _logger.info(f'\nCollecting stats: {corruption} (severity={args.severity})')
        loader = create_corruption_loader(
            corruption, args, data_config, model_dtype, device,
        )
        results[corruption] = collect_sa2_stats(
            model, loader, device, model_dtype, base_model,
            args.num_batches, prefetcher=prefetcher,
        )
        mp = results[corruption]['mean_probs']
        _logger.info(f'  {corruption} mean_probs: {np.array2string(mp, precision=4)}')
        _logger.info(f'  {corruption} mean_entropy: {results[corruption]["mean_entropy"]:.4f}')

    # --- Write CSV ---
    csv_path = os.path.join(args.output_dir, 'sa2_distribution.csv')
    write_csv(results, csv_path, args.prop_size)
    _logger.info(f'\nCSV saved: {csv_path}')

    # --- Plot ---
    plot_results(results, args.output_dir, args.prop_size)

    # --- Summary to console ---
    _logger.info(f'\n{"="*60}')
    _logger.info('SUMMARY')
    _logger.info(f'{"="*60}')
    _logger.info(f'{"domain":<22s}  {"p0":>6s} {"p1":>6s} {"p2":>6s} {"p3":>6s} {"p4":>6s}  '
                 f'{"maxP":>6s} {"H_pos":>6s} {"H_marg":>6s}')
    _logger.info('-' * 88)
    for domain, stats in results.items():
        mp = stats['mean_probs']
        _logger.info(
            f'{domain:<22s}  {mp[0]:6.4f} {mp[1]:6.4f} {mp[2]:6.4f} {mp[3]:6.4f} {mp[4]:6.4f}  '
            f'{stats["mean_max_prob"]:6.4f} {stats["mean_entropy"]:6.4f} '
            f'{stats["marginal_entropy"]:6.4f}'
        )
    _logger.info(f'{"="*60}')


if __name__ == '__main__':
    main()
