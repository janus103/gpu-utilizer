#!/usr/bin/env python3
"""Analyze SpatialAttention2 class 0/4 distribution split by correct/incorrect predictions.

For each domain, measures:
  - Per-image: what fraction of spatial positions are class-0 dominant vs class-4 dominant
  - Split by whether the model's top-1 prediction is correct or incorrect
  - Also reports entropy correlation with correctness
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

_logger = logging.getLogger('sa2_ci')

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


def _is_auxiliary_param(name: str) -> bool:
    n = name.lower()
    return any(kw in n for kw in (
        'se_module', 'sam_module', 'channel_attn', 'spatial_attn',
        'se_module_last', 'sam_module_last', 'channel_attn_last', 'spatial_attn_last',
    ))


def _is_embedding_param(name: str) -> bool:
    n = name.lower()
    return n.startswith((
        'patch_embed', 'pos_embed', 'cls_token', 'reg_token',
        'norm_pre', 'conv1', 'bn1', 'act1',
    ))


def _apply_train_mode(model: nn.Module, train_mode: int) -> int:
    if train_mode == 0:
        for p in model.parameters():
            p.requires_grad = True
    elif train_mode == 1:
        for p in model.parameters():
            p.requires_grad = False
        for n, p in model.named_parameters():
            if _is_auxiliary_param(n) or _is_embedding_param(n):
                p.requires_grad = True
    elif train_mode == 2:
        for p in model.parameters():
            p.requires_grad = False
        for n, p in model.named_parameters():
            if _is_auxiliary_param(n):
                p.requires_grad = True
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_trainable_state(model, state):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in state:
                param.data.copy_(state[name])


def _make_loader(data_dir, args, data_config, model_dtype, device, split):
    img_mode = args.input_img_mode or 'RGB'
    ds = create_dataset(
        '', root=data_dir, split=split, is_training=False,
        class_map='', download=False,
        batch_size=args.batch_size, input_img_mode=img_mode,
    )
    return create_loader(
        ds, input_size=data_config['input_size'],
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
def collect_correct_incorrect_stats(
    model, loader, device, model_dtype, base_model, num_batches, prefetcher=True,
):
    model.eval()

    correct_c0_fracs = []
    incorrect_c0_fracs = []
    correct_entropies = []
    incorrect_entropies = []
    correct_maxprobs = []
    incorrect_maxprobs = []
    n_correct = 0
    n_incorrect = 0

    for batch_idx, (images, target) in enumerate(loader):
        if batch_idx >= num_batches:
            break
        if not prefetcher:
            images = images.to(device=device, dtype=model_dtype)
            target = target.to(device=device)

        output = model(images)
        if isinstance(output, (tuple, list)):
            output = output[0]

        logits_sa2 = base_model.spatial_attn._logits  # (B, 5, H, W)
        probs = F.softmax(logits_sa2, dim=1)
        B, N, H, W = probs.shape

        argmax_cls = probs.argmax(dim=1)  # (B, H, W)
        c0_frac_per_img = (argmax_cls == 0).float().mean(dim=(1, 2))  # (B,)

        log_probs = (probs + 1e-8).log()
        entropy_per_pos = -(probs * log_probs).sum(dim=1)  # (B, H, W)
        mean_entropy_per_img = entropy_per_pos.mean(dim=(1, 2))  # (B,)

        max_prob_per_pos = probs.max(dim=1).values  # (B, H, W)
        mean_maxprob_per_img = max_prob_per_pos.mean(dim=(1, 2))  # (B,)

        pred = output.argmax(dim=1)  # (B,)
        correct_mask = (pred == target)  # (B,)

        for i in range(B):
            if correct_mask[i]:
                correct_c0_fracs.append(c0_frac_per_img[i].item())
                correct_entropies.append(mean_entropy_per_img[i].item())
                correct_maxprobs.append(mean_maxprob_per_img[i].item())
                n_correct += 1
            else:
                incorrect_c0_fracs.append(c0_frac_per_img[i].item())
                incorrect_entropies.append(mean_entropy_per_img[i].item())
                incorrect_maxprobs.append(mean_maxprob_per_img[i].item())
                n_incorrect += 1

        if (batch_idx + 1) % 10 == 0:
            _logger.info(f'    batch {batch_idx + 1}/{num_batches}')

    total = n_correct + n_incorrect
    acc = n_correct / total * 100 if total > 0 else 0

    def _safe_mean(lst):
        return np.mean(lst) if lst else 0.0

    return {
        'n_correct': n_correct,
        'n_incorrect': n_incorrect,
        'accuracy': acc,
        'correct_c0_frac': _safe_mean(correct_c0_fracs),
        'incorrect_c0_frac': _safe_mean(incorrect_c0_fracs),
        'correct_c4_frac': 1.0 - _safe_mean(correct_c0_fracs),
        'incorrect_c4_frac': 1.0 - _safe_mean(incorrect_c0_fracs),
        'correct_entropy': _safe_mean(correct_entropies),
        'incorrect_entropy': _safe_mean(incorrect_entropies),
        'correct_maxprob': _safe_mean(correct_maxprobs),
        'incorrect_maxprob': _safe_mean(incorrect_maxprobs),
        'all_correct_c0_fracs': correct_c0_fracs,
        'all_incorrect_c0_fracs': incorrect_c0_fracs,
    }


def write_csv(results, path):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([
            'domain', 'accuracy',
            'n_correct', 'n_incorrect',
            'correct_c0_frac', 'correct_c4_frac',
            'incorrect_c0_frac', 'incorrect_c4_frac',
            'correct_entropy', 'incorrect_entropy',
            'correct_maxprob', 'incorrect_maxprob',
        ])
        for domain, s in results.items():
            w.writerow([
                domain, f'{s["accuracy"]:.2f}',
                s['n_correct'], s['n_incorrect'],
                f'{s["correct_c0_frac"]:.6f}', f'{s["correct_c4_frac"]:.6f}',
                f'{s["incorrect_c0_frac"]:.6f}', f'{s["incorrect_c4_frac"]:.6f}',
                f'{s["correct_entropy"]:.6f}', f'{s["incorrect_entropy"]:.6f}',
                f'{s["correct_maxprob"]:.6f}', f'{s["incorrect_maxprob"]:.6f}',
            ])


def plot_results(results, output_dir):
    domains = list(results.keys())
    n = len(domains)
    x = np.arange(n)

    # --- 1. Grouped bar: correct vs incorrect class-0 fraction ---
    fig, ax = plt.subplots(figsize=(max(14, n * 0.9), 6))
    w = 0.35
    c0_correct = [results[d]['correct_c0_frac'] for d in domains]
    c0_incorrect = [results[d]['incorrect_c0_frac'] for d in domains]
    ax.bar(x - w / 2, c0_correct, w, label='Correct', color='#4CAF50')
    ax.bar(x + w / 2, c0_incorrect, w, label='Incorrect', color='#F44336')
    ax.set_xticks(x)
    ax.set_xticklabels(domains, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Class-0 dominant fraction (per image)')
    ax.set_title('SA2 Class-0 fraction: Correct vs Incorrect predictions')
    ax.legend()
    ax.axhline(y=results['clean']['correct_c0_frac'], color='#4CAF50',
               linestyle='--', alpha=0.4, label='clean correct')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'sa2_c0_correct_vs_incorrect.png'), dpi=150)
    plt.close(fig)

    # --- 2. Entropy: correct vs incorrect ---
    fig, ax = plt.subplots(figsize=(max(14, n * 0.9), 6))
    ent_correct = [results[d]['correct_entropy'] for d in domains]
    ent_incorrect = [results[d]['incorrect_entropy'] for d in domains]
    ax.bar(x - w / 2, ent_correct, w, label='Correct', color='#2196F3')
    ax.bar(x + w / 2, ent_incorrect, w, label='Incorrect', color='#FF9800')
    ax.set_xticks(x)
    ax.set_xticklabels(domains, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Mean per-position entropy')
    ax.set_title('SA2 Entropy: Correct vs Incorrect predictions')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'sa2_entropy_correct_vs_incorrect.png'), dpi=150)
    plt.close(fig)

    # --- 3. Max prob (mask strength): correct vs incorrect ---
    fig, ax = plt.subplots(figsize=(max(14, n * 0.9), 6))
    mp_correct = [results[d]['correct_maxprob'] for d in domains]
    mp_incorrect = [results[d]['incorrect_maxprob'] for d in domains]
    ax.bar(x - w / 2, mp_correct, w, label='Correct', color='#9C27B0')
    ax.bar(x + w / 2, mp_incorrect, w, label='Incorrect', color='#FF5722')
    ax.set_xticks(x)
    ax.set_xticklabels(domains, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Mean max probability (mask strength)')
    ax.set_title('SA2 Mask strength: Correct vs Incorrect predictions')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'sa2_maxprob_correct_vs_incorrect.png'), dpi=150)
    plt.close(fig)

    # --- 4. Histogram: per-image c0_frac distribution (correct vs incorrect) ---
    n_cols = 4
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    axes_flat = axes.flatten()
    for i, domain in enumerate(domains):
        ax = axes_flat[i]
        s = results[domain]
        bins = np.linspace(0, 1, 30)
        if s['all_correct_c0_fracs']:
            ax.hist(s['all_correct_c0_fracs'], bins=bins, alpha=0.6,
                    label='Correct', color='#4CAF50', density=True)
        if s['all_incorrect_c0_fracs']:
            ax.hist(s['all_incorrect_c0_fracs'], bins=bins, alpha=0.6,
                    label='Incorrect', color='#F44336', density=True)
        ax.set_title(domain, fontsize=9)
        ax.set_xlim(0, 1)
        if i == 0:
            ax.legend(fontsize=7)
    for i in range(n, len(axes_flat)):
        axes_flat[i].set_visible(False)
    fig.suptitle('Per-image class-0 fraction distribution: Correct vs Incorrect', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'sa2_c0_frac_histograms.png'), dpi=150)
    plt.close(fig)

    # --- 5. Summary: delta (incorrect - correct) c0_frac vs accuracy ---
    fig, ax = plt.subplots(figsize=(8, 6))
    accs = [results[d]['accuracy'] for d in domains]
    deltas = [results[d]['incorrect_c0_frac'] - results[d]['correct_c0_frac'] for d in domains]
    for i, d in enumerate(domains):
        color = '#2196F3' if d == 'clean' else '#FF9800'
        ax.scatter(deltas[i], accs[i], color=color, s=60, zorder=3)
        ax.annotate(d, (deltas[i], accs[i]), fontsize=7, ha='left', va='bottom')
    ax.set_xlabel('Delta class-0 frac (incorrect - correct)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Does class-0 fraction shift correlate with errors?')
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'sa2_delta_c0_vs_accuracy.png'), dpi=150)
    plt.close(fig)

    _logger.info(f'Plots saved to {output_dir}')


def _parse_args():
    config_parser = argparse.ArgumentParser(description='Config', add_help=False)
    config_parser.add_argument('-c', '--config', default='', type=str)

    parser = argparse.ArgumentParser(
        description='SA2 correct/incorrect distribution analysis')

    g = parser.add_argument_group('Data')
    g.add_argument('--data-root', type=str, default='/home/oem/servers/imagenet-c')
    g.add_argument('--severity', type=int, default=5)
    g.add_argument('--val-split', type=str, default='validation')
    g.add_argument('--clean-data-dir', type=str, default='/data/imagenet/imagenet')
    g.add_argument('--clean-split', type=str, default='val')
    g.add_argument('--input-img-mode', default=None, type=str)

    g = parser.add_argument_group('Model')
    g.add_argument('--model', default='vit_base_patch16_224', type=str)
    g.add_argument('--resume', type=str,
                   default='./VIT_IMG_PAR/Normal_parallel_train_1_kernel_size_2/model_best.pth.tar')
    g.add_argument('--fedavg-ckpt', type=str, default=None)
    g.add_argument('--num-classes', type=int, default=1000)
    g.add_argument('--input-size', default=[3, 224, 224], nargs=3, type=int)
    g.add_argument('--parallel-attention', action='store_true', default=True)
    g.add_argument('--vit-kernel-size', type=int, default=2)
    g.add_argument('--spatial-group-size', type=int, default=1)
    g.add_argument('--vit-last', action='store_true', default=False)
    g.add_argument('--vit-closed', type=str, default=None, choices=['same', 'diff'])
    g.add_argument('--use-se-module', action='store_true', default=False)
    g.add_argument('--use-sam-module', type=int, default=-1)
    g.add_argument('--reverse-se', action='store_true', default=False)
    g.add_argument('--train-mode', type=int, default=1, choices=[0, 1, 2])
    g.add_argument('--sam-norm-type', type=int, default=0, choices=[0, 1, 2, 3, 4])
    g.add_argument('--vit-early-norm-types', type=int, nargs=4, default=None,
                   choices=[0, 1, 2, 3, 4])
    g.add_argument('--prop-size', type=int, default=5)

    g = parser.add_argument_group('Runtime')
    g.add_argument('-b', '--batch-size', type=int, default=256)
    g.add_argument('-j', '--workers', type=int, default=4)
    g.add_argument('--device', default='cuda', type=str)
    g.add_argument('--model-dtype', default=None, type=str)
    g.add_argument('--pin-mem', action='store_true', default=False)
    g.add_argument('--no-prefetcher', action='store_true', default=False)
    g.add_argument('--seed', type=int, default=42)
    g.add_argument('--device-modules', default=None, type=str, nargs='+')
    g.add_argument('--num-batches', type=int, default=50)
    g.add_argument('--output-dir', type=str, default='./output/sa2_correct_incorrect')

    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            parser.set_defaults(**yaml.safe_load(f))
    return parser.parse_args(remaining)


def main():
    utils.setup_default_logging()
    args = _parse_args()

    if args.device_modules:
        for m in args.device_modules:
            importlib.import_module(m)

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
        args.model, num_classes=args.num_classes, in_chans=in_chans,
        parallel_attention=args.parallel_attention,
        use_se_module=args.use_se_module, use_sam_module=args.use_sam_module,
        reverse_se_sam=args.reverse_se, sam_norm_type=args.sam_norm_type,
        **vit_norm_kwargs,
    )
    model.to(device=device, dtype=model_dtype)

    if args.resume:
        _logger.info(f'Loading base checkpoint: {args.resume}')
        ckpt = torch.load(args.resume, map_location='cpu')
        sd = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        model.load_state_dict(sd, strict=False)

    if args.prop_size > 0:
        base = _get_base_model(model)
        ch = base.embed_dim
        base.spatial_attn = SpatialAttention2(
            kernel_size=args.vit_kernel_size, channels=ch,
            prop_size=args.prop_size,
        ).to(device=device, dtype=model_dtype)
        _logger.info(f'SpatialAttention2 installed: prop_size={args.prop_size}')

    _apply_train_mode(model, args.train_mode)

    if args.fedavg_ckpt:
        _logger.info(f'Loading FedAvg checkpoint: {args.fedavg_ckpt}')
        ckpt = torch.load(args.fedavg_ckpt, map_location='cpu')
        fedavg_state = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        set_trainable_state(model, fedavg_state)

    base_model = _get_base_model(model)
    data_config = resolve_data_config(
        vars(args) | {'pretrained': False}, model=model, verbose=True,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    prefetcher = not args.no_prefetcher
    results = {}

    # --- Clean ---
    _logger.info(f'\n{"="*60}')
    _logger.info('Collecting: clean')
    _logger.info(f'{"="*60}')
    clean_dir = args.clean_data_dir
    clean_loader = _make_loader(clean_dir, args, data_config, model_dtype, device,
                                split=args.clean_split)
    results['clean'] = collect_correct_incorrect_stats(
        model, clean_loader, device, model_dtype, base_model,
        args.num_batches, prefetcher=prefetcher,
    )

    # --- Corruptions ---
    for corruption in CORRUPTIONS:
        _logger.info(f'\nCollecting: {corruption}')
        c_dir = os.path.join(args.data_root, corruption, str(args.severity))
        loader = _make_loader(c_dir, args, data_config, model_dtype, device,
                              split=args.val_split)
        results[corruption] = collect_correct_incorrect_stats(
            model, loader, device, model_dtype, base_model,
            args.num_batches, prefetcher=prefetcher,
        )

    # --- CSV ---
    csv_path = os.path.join(args.output_dir, 'sa2_correct_incorrect.csv')
    write_csv(results, csv_path)
    _logger.info(f'\nCSV saved: {csv_path}')

    # --- Plot ---
    plot_results(results, args.output_dir)

    # --- Summary ---
    _logger.info(f'\n{"="*60}')
    _logger.info('SUMMARY: correct vs incorrect SA2 class-0 fraction & entropy')
    _logger.info(f'{"="*60}')
    _logger.info(
        f'{"domain":<22s}  {"acc%":>6s}  '
        f'{"c0_corr":>8s} {"c0_incr":>8s} {"delta":>8s}  '
        f'{"H_corr":>7s} {"H_incr":>7s} {"dH":>7s}'
    )
    _logger.info('-' * 90)
    for domain, s in results.items():
        delta_c0 = s['incorrect_c0_frac'] - s['correct_c0_frac']
        delta_h = s['incorrect_entropy'] - s['correct_entropy']
        _logger.info(
            f'{domain:<22s}  {s["accuracy"]:6.2f}  '
            f'{s["correct_c0_frac"]:8.4f} {s["incorrect_c0_frac"]:8.4f} '
            f'{delta_c0:+8.4f}  '
            f'{s["correct_entropy"]:7.4f} {s["incorrect_entropy"]:7.4f} '
            f'{delta_h:+7.4f}'
        )
    _logger.info(f'{"="*60}')


if __name__ == '__main__':
    main()
