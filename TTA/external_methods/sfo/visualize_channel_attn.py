#!/usr/bin/env python3
"""Visualize ChannelAttention (SE) scale factors across domains.

For a batch of images, extracts the 768-dim channel attention vector
from the clean image and all 15 corrupted versions, then produces:

1. Per-image heatmap: (16 domains × 768 channels) showing scale factors
2. Per-image difference heatmap: |corrupt - clean| for each corruption
3. Batch-averaged bar chart comparing mean channel activation per domain
4. Top-K most discriminative channels across domains

Usage:
  python visualize_channel_attn.py \\
      --checkpoint output/fedavg_direct_k2_tm1_le1_p5_wu10_aa5/best.pth \\
      --num-images 5 --output-dir ./vis_channel_classes
"""
import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from torchvision.datasets import ImageFolder
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.models import create_model
from timm.models.vision_transformer import SpatialAttention2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable

CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
    'snow', 'frost', 'fog', 'brightness',
    'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
]
DOMAIN_LABELS = ['clean'] + CORRUPTIONS

CORRUPTION_GROUPS = {
    'noise':   ['gaussian_noise', 'shot_noise', 'impulse_noise'],
    'blur':    ['defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur'],
    'weather': ['snow', 'frost', 'fog', 'brightness'],
    'digital': ['contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'],
}

GROUP_COLORS = {
    'clean':   '#2ca02c',
    'noise':   '#d62728',
    'blur':    '#1f77b4',
    'weather': '#ff7f0e',
    'digital': '#9467bd',
}


def _get_base_model(model):
    m = model
    if hasattr(m, 'module'):
        m = m.module
    if hasattr(m, '_orig_mod'):
        m = m._orig_mod
    return m


def _domain_to_group(domain_name):
    if domain_name == 'clean':
        return 'clean'
    for g, members in CORRUPTION_GROUPS.items():
        if domain_name in members:
            return g
    return 'digital'


def build_model(args, ckpt, device):
    saved_args = ckpt['args']
    prop_size = saved_args.get('prop_size', 5)
    vit_kernel_size = saved_args.get('vit_kernel_size', 2)
    model_name = saved_args.get('model', 'vit_base_patch16_224')

    model_kwargs = {}
    if saved_args.get('parallel_attention', True):
        model_kwargs['sam_kernel_size'] = vit_kernel_size
        model_kwargs['spatial_group_size'] = saved_args.get('spatial_group_size', 1)

    model = create_model(
        model_name,
        pretrained=True,
        num_classes=saved_args.get('num_classes', 1000),
        parallel_attention=saved_args.get('parallel_attention', True),
        use_se_module=saved_args.get('use_se_module', False),
        use_sam_module=saved_args.get('use_sam_module', -1),
        **model_kwargs,
    )

    if prop_size > 0:
        base = _get_base_model(model)
        embed_dim = base.embed_dim
        base.spatial_attn = SpatialAttention2(
            kernel_size=vit_kernel_size,
            channels=embed_dim,
            prop_size=prop_size,
        )

    state_dict = ckpt['state_dict']
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f'Loaded checkpoint: missing={len(missing)}, unexpected={len(unexpected)}')

    model.to(device)
    model.eval()
    return model


def get_eval_transform(model):
    data_config = resolve_data_config({}, model=model)
    transform = create_transform(
        input_size=data_config['input_size'],
        is_training=False,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        crop_pct=data_config['crop_pct'],
    )
    return transform, data_config


def get_channel_attn(model, tensor_batch, device):
    """Hook into channel_attn to capture scale factors.

    Returns:
        attn_vec: (B, C) float tensor — per-channel scale factors after ReLU
    """
    base = _get_base_model(model)
    captured = {}

    def _hook(module, inp, out):
        captured['attn'] = out.detach().cpu()

    handle = base.channel_attn.register_forward_hook(_hook)
    tensor_batch = tensor_batch.to(device)
    with torch.no_grad():
        _ = model(tensor_batch)
    handle.remove()

    attn = captured['attn']  # (B, C, 1, 1)
    return attn.squeeze(-1).squeeze(-1)  # (B, C)


def visualize_one_image(
    model, clean_dataset, corrupt_datasets, img_idx,
    transform, device, output_dir, img_size=(224, 224), top_k=30,
):
    img_path, label = clean_dataset.samples[img_idx]
    pil_img = Image.open(img_path).convert('RGB')
    class_name = os.path.basename(os.path.dirname(img_path))

    pil_images = [pil_img]
    for corr_name in CORRUPTIONS:
        c_path, _ = corrupt_datasets[corr_name].samples[img_idx]
        pil_images.append(Image.open(c_path).convert('RGB'))

    tensors = torch.stack([transform(p) for p in pil_images])
    attn_vecs = get_channel_attn(model, tensors, device)  # (16, 768)
    attn_np = attn_vecs.numpy()
    n_domains, n_channels = attn_np.shape

    clean_vec = attn_np[0]  # (768,)
    diff_np = attn_np[1:] - clean_vec[np.newaxis, :]  # (15, 768)

    # Relative change: (corrupt - clean) / (clean + eps)
    eps = 1e-8
    rel_diff = diff_np / (clean_vec[np.newaxis, :] + eps)  # (15, 768)

    # Per-channel std across domains to find discriminative channels
    std_across_domains = attn_np.std(axis=0)  # (768,)
    topk_idx = np.argsort(std_across_domains)[-top_k:][::-1]

    # ── Helper: styled yticks ──
    def _style_yticks(ax, labels):
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
        for i, dl in enumerate(labels):
            grp = _domain_to_group(dl)
            ax.get_yticklabels()[i].set_color(GROUP_COLORS[grp])
            ax.get_yticklabels()[i].set_fontweight('bold')

    def _top_cbar(fig, ax, im, label):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('top', size='5%', pad=0.08)
        cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
        cbar.set_label(label, fontsize=8, labelpad=3)
        cax.xaxis.set_ticks_position('top')
        cax.xaxis.set_label_position('top')
        cbar.ax.tick_params(labelsize=7)
        return cbar

    # ── Fig 1: Raw scale factors heatmap ──
    vmax = np.percentile(attn_np, 99)
    fig1, ax1 = plt.subplots(figsize=(18, 5))
    im1 = ax1.imshow(attn_np, aspect='auto', cmap='viridis', vmin=0, vmax=vmax,
                      interpolation='nearest')
    _style_yticks(ax1, DOMAIN_LABELS)
    ax1.set_xlabel(f'Channel index (0–{n_channels-1})', fontsize=9)
    ax1.set_title(
        f'Channel Attention Scale Factors — Image #{img_idx} ({class_name}, label={label})',
        fontsize=11, fontweight='bold')
    _top_cbar(fig1, ax1, im1, 'Scale Factor (after ReLU)')
    plt.tight_layout()
    fname1 = os.path.join(output_dir, f'channel_raw_{img_idx:05d}.png')
    plt.savefig(fname1, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig1)
    print(f'  Saved: {fname1}')

    # ── Fig 2: Relative difference heatmap (%) ──
    pct = rel_diff * 100  # percentage
    pct_lim = np.percentile(np.abs(pct), 98)
    pct_lim = max(pct_lim, 0.1)

    fig2, ax2 = plt.subplots(figsize=(18, 4.5))
    im2 = ax2.imshow(pct, aspect='auto', cmap='RdBu_r',
                      vmin=-pct_lim, vmax=pct_lim, interpolation='nearest')
    _style_yticks(ax2, CORRUPTIONS)
    ax2.set_xlabel(f'Channel index (0–{n_channels-1})', fontsize=9)
    ax2.set_title(
        f'Channel Attention Relative Δ% (corrupt − clean)/clean — Image #{img_idx}',
        fontsize=11, fontweight='bold')
    _top_cbar(fig2, ax2, im2, 'Relative Δ (%)')
    plt.tight_layout()
    fname2 = os.path.join(output_dir, f'channel_diff_{img_idx:05d}.png')
    plt.savefig(fname2, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig2)
    print(f'  Saved: {fname2}')

    # ── Fig 3: Top-K discriminative channels (zoomed raw values) ──
    attn_topk = attn_np[:, topk_idx]  # (16, top_k)

    fig3, ax3 = plt.subplots(figsize=(max(12, top_k * 0.5), 5.5))
    im3 = ax3.imshow(attn_topk, aspect='auto', cmap='viridis',
                      vmin=attn_topk.min(), vmax=attn_topk.max(),
                      interpolation='nearest')
    _style_yticks(ax3, DOMAIN_LABELS)
    ax3.set_xticks(range(top_k))
    ax3.set_xticklabels([f'ch{c}' for c in topk_idx], fontsize=6, rotation=60, ha='right')
    ax3.set_title(
        f'Top-{top_k} Most Variable Channels — Image #{img_idx}\n'
        f'(ranked by cross-domain σ, colorbar auto-scaled)',
        fontsize=10, fontweight='bold')
    _top_cbar(fig3, ax3, im3, 'Scale Factor (auto-scaled)')
    plt.tight_layout()
    fname3 = os.path.join(output_dir, f'channel_topk_{img_idx:05d}.png')
    plt.savefig(fname3, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig3)
    print(f'  Saved: {fname3}')

    # ── Fig 4: Top-K relative diff (%) zoomed ──
    reldiff_topk = pct[:, topk_idx]  # (15, top_k)
    rl = np.percentile(np.abs(reldiff_topk), 99)
    rl = max(rl, 0.1)

    fig4, ax4 = plt.subplots(figsize=(max(12, top_k * 0.5), 5))
    im4 = ax4.imshow(reldiff_topk, aspect='auto', cmap='RdBu_r',
                      vmin=-rl, vmax=rl, interpolation='nearest')
    _style_yticks(ax4, CORRUPTIONS)
    ax4.set_xticks(range(top_k))
    ax4.set_xticklabels([f'ch{c}' for c in topk_idx], fontsize=6, rotation=60, ha='right')
    ax4.set_title(
        f'Top-{top_k} Channels — Relative Δ% (corrupt − clean)/clean — Image #{img_idx}',
        fontsize=10, fontweight='bold')
    _top_cbar(fig4, ax4, im4, 'Relative Δ (%)')
    plt.tight_layout()
    fname4 = os.path.join(output_dir, f'channel_topk_diff_{img_idx:05d}.png')
    plt.savefig(fname4, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig4)
    print(f'  Saved: {fname4}')

    # ── Fig 5: Per-domain mean bar with zoomed y-axis ──
    mean_per_domain = attn_np.mean(axis=1)  # (16,)
    bar_colors = [GROUP_COLORS[_domain_to_group(d)] for d in DOMAIN_LABELS]

    fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(16, 4.5),
                                       gridspec_kw={'width_ratios': [1, 1]})

    x_pos = np.arange(n_domains)
    ax5a.bar(x_pos, mean_per_domain, color=bar_colors,
             edgecolor='black', linewidth=0.5, alpha=0.85)
    ax5a.set_xticks(x_pos)
    ax5a.set_xticklabels(DOMAIN_LABELS, fontsize=6, rotation=45, ha='right')
    ax5a.set_ylabel('Mean Scale Factor', fontsize=9)
    ax5a.set_title('Full Range', fontsize=9, fontweight='bold')
    ax5a.axhline(mean_per_domain[0], color='gray', ls='--', lw=0.8, alpha=0.6)

    # Zoomed view: center around clean mean
    y_center = mean_per_domain[0]
    y_range = max(np.abs(mean_per_domain - y_center).max() * 3, 1e-6)
    ax5b.bar(x_pos, mean_per_domain, color=bar_colors,
             edgecolor='black', linewidth=0.5, alpha=0.85)
    ax5b.set_xticks(x_pos)
    ax5b.set_xticklabels(DOMAIN_LABELS, fontsize=6, rotation=45, ha='right')
    ax5b.set_ylabel('Mean Scale Factor (zoomed)', fontsize=9)
    ax5b.set_ylim(y_center - y_range, y_center + y_range)
    ax5b.set_title('Zoomed (3× max deviation from clean)', fontsize=9, fontweight='bold')
    ax5b.axhline(y_center, color='gray', ls='--', lw=0.8, alpha=0.6)

    legend_patches = [Patch(facecolor=c, edgecolor='black', linewidth=0.5, label=g)
                      for g, c in GROUP_COLORS.items()]
    ax5b.legend(handles=legend_patches, fontsize=6, ncol=5, loc='upper right')

    fig5.suptitle(
        f'Mean Channel Attention per Domain — Image #{img_idx} ({class_name})',
        fontsize=10, fontweight='bold')
    plt.tight_layout()
    fname5 = os.path.join(output_dir, f'channel_bar_{img_idx:05d}.png')
    plt.savefig(fname5, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig5)
    print(f'  Saved: {fname5}')

    return attn_np


def visualize_aggregate(all_attn, indices, output_dir, top_k=30):
    """Aggregate across multiple images for a population-level view."""
    stacked = np.stack(all_attn, axis=0)  # (N_img, 16, 768)
    mean_attn = stacked.mean(axis=0)      # (16, 768)
    n_domains, n_channels = mean_attn.shape

    clean_mean = mean_attn[0]
    diff_mean = mean_attn[1:] - clean_mean[np.newaxis, :]
    eps = 1e-8
    rel_diff = diff_mean / (clean_mean[np.newaxis, :] + eps) * 100  # %

    std_across = mean_attn.std(axis=0)
    topk_idx = np.argsort(std_across)[-top_k:][::-1]

    def _style_yticks(ax, labels):
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
        for i, dl in enumerate(labels):
            grp = _domain_to_group(dl)
            ax.get_yticklabels()[i].set_color(GROUP_COLORS[grp])
            ax.get_yticklabels()[i].set_fontweight('bold')

    def _top_cbar(fig, ax, im, label):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('top', size='5%', pad=0.08)
        cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
        cbar.set_label(label, fontsize=8, labelpad=3)
        cax.xaxis.set_ticks_position('top')
        cax.xaxis.set_label_position('top')
        cbar.ax.tick_params(labelsize=7)

    n_img = len(indices)

    # ── Aggregate raw heatmap ──
    vmax = np.percentile(mean_attn, 99)
    fig, ax = plt.subplots(figsize=(18, 5))
    im = ax.imshow(mean_attn, aspect='auto', cmap='viridis', vmin=0, vmax=vmax,
                   interpolation='nearest')
    _style_yticks(ax, DOMAIN_LABELS)
    ax.set_xlabel(f'Channel index (0–{n_channels-1})', fontsize=9)
    ax.set_title(f'Aggregate Channel Attention (mean over {n_img} images)',
                 fontsize=11, fontweight='bold')
    _top_cbar(fig, ax, im, 'Mean Scale Factor')
    plt.tight_layout()
    fname = os.path.join(output_dir, 'channel_aggregate_raw.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f'  Saved: {fname}')

    # ── Aggregate relative diff (%) ──
    pct_lim = np.percentile(np.abs(rel_diff), 98)
    pct_lim = max(pct_lim, 0.1)
    fig2, ax2 = plt.subplots(figsize=(18, 4.5))
    im2 = ax2.imshow(rel_diff, aspect='auto', cmap='RdBu_r',
                     vmin=-pct_lim, vmax=pct_lim, interpolation='nearest')
    _style_yticks(ax2, CORRUPTIONS)
    ax2.set_xlabel(f'Channel index (0–{n_channels-1})', fontsize=9)
    ax2.set_title(f'Aggregate Relative Δ% (corrupt − clean)/clean, {n_img} images',
                  fontsize=11, fontweight='bold')
    _top_cbar(fig2, ax2, im2, 'Relative Δ (%)')
    plt.tight_layout()
    fname2 = os.path.join(output_dir, 'channel_aggregate_diff.png')
    plt.savefig(fname2, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig2)
    print(f'  Saved: {fname2}')

    # ── Aggregate top-K raw (auto-scaled) ──
    attn_topk = mean_attn[:, topk_idx]
    fig3, ax3 = plt.subplots(figsize=(max(12, top_k * 0.5), 5.5))
    im3 = ax3.imshow(attn_topk, aspect='auto', cmap='viridis',
                     vmin=attn_topk.min(), vmax=attn_topk.max(),
                     interpolation='nearest')
    _style_yticks(ax3, DOMAIN_LABELS)
    ax3.set_xticks(range(top_k))
    ax3.set_xticklabels([f'ch{c}' for c in topk_idx], fontsize=6, rotation=60, ha='right')
    ax3.set_title(f'Aggregate Top-{top_k} Most Variable Channels\n'
                  f'({n_img} images, ranked by cross-domain σ, auto-scaled)',
                  fontsize=10, fontweight='bold')
    _top_cbar(fig3, ax3, im3, 'Scale Factor (auto-scaled)')
    plt.tight_layout()
    fname3 = os.path.join(output_dir, 'channel_aggregate_topk.png')
    plt.savefig(fname3, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig3)
    print(f'  Saved: {fname3}')

    # ── Aggregate top-K relative diff (%) ──
    rd_topk = rel_diff[:, topk_idx]
    rl = max(np.percentile(np.abs(rd_topk), 99), 0.1)
    fig3b, ax3b = plt.subplots(figsize=(max(12, top_k * 0.5), 5))
    im3b = ax3b.imshow(rd_topk, aspect='auto', cmap='RdBu_r',
                       vmin=-rl, vmax=rl, interpolation='nearest')
    _style_yticks(ax3b, CORRUPTIONS)
    ax3b.set_xticks(range(top_k))
    ax3b.set_xticklabels([f'ch{c}' for c in topk_idx], fontsize=6, rotation=60, ha='right')
    ax3b.set_title(f'Aggregate Top-{top_k} — Relative Δ% ({n_img} images)',
                   fontsize=10, fontweight='bold')
    _top_cbar(fig3b, ax3b, im3b, 'Relative Δ (%)')
    plt.tight_layout()
    fname3b = os.path.join(output_dir, 'channel_aggregate_topk_diff.png')
    plt.savefig(fname3b, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig3b)
    print(f'  Saved: {fname3b}')

    # ── Aggregate per-domain bar (full + zoomed) ──
    pop_mean = mean_attn.mean(axis=1)
    bar_colors = [GROUP_COLORS[_domain_to_group(d)] for d in DOMAIN_LABELS]

    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(16, 4.5),
                                       gridspec_kw={'width_ratios': [1, 1]})
    x_pos = np.arange(n_domains)

    ax4a.bar(x_pos, pop_mean, color=bar_colors,
             edgecolor='black', linewidth=0.5, alpha=0.85)
    ax4a.set_xticks(x_pos)
    ax4a.set_xticklabels(DOMAIN_LABELS, fontsize=6, rotation=45, ha='right')
    ax4a.set_ylabel('Mean Scale Factor', fontsize=9)
    ax4a.set_title('Full Range', fontsize=9, fontweight='bold')
    ax4a.axhline(pop_mean[0], color='gray', ls='--', lw=0.8, alpha=0.6)

    y_c = pop_mean[0]
    y_r = max(np.abs(pop_mean - y_c).max() * 3, 1e-6)
    ax4b.bar(x_pos, pop_mean, color=bar_colors,
             edgecolor='black', linewidth=0.5, alpha=0.85)
    ax4b.set_xticks(x_pos)
    ax4b.set_xticklabels(DOMAIN_LABELS, fontsize=6, rotation=45, ha='right')
    ax4b.set_ylabel('Mean Scale Factor (zoomed)', fontsize=9)
    ax4b.set_ylim(y_c - y_r, y_c + y_r)
    ax4b.set_title('Zoomed (3× max deviation)', fontsize=9, fontweight='bold')
    ax4b.axhline(y_c, color='gray', ls='--', lw=0.8, alpha=0.6)

    legend_patches = [Patch(facecolor=c, edgecolor='black', linewidth=0.5, label=g)
                      for g, c in GROUP_COLORS.items()]
    ax4b.legend(handles=legend_patches, fontsize=6, ncol=5, loc='upper right')

    fig4.suptitle(f'Aggregate Mean Channel Attention per Domain ({n_img} images)',
                  fontsize=10, fontweight='bold')
    plt.tight_layout()
    fname4 = os.path.join(output_dir, 'channel_aggregate_bar.png')
    plt.savefig(fname4, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig4)
    print(f'  Saved: {fname4}')


def main():
    parser = argparse.ArgumentParser(
        description='Visualize ChannelAttention scale factors across domains')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--clean-data-dir', type=str, default='/data/imagenet/imagenet')
    parser.add_argument('--clean-split', type=str, default='val')
    parser.add_argument('--data-root', type=str, default='/home/oem/servers/imagenet-c')
    parser.add_argument('--severity', type=int, default=5)
    parser.add_argument('--num-images', type=int, default=10)
    parser.add_argument('--image-indices', type=int, nargs='+', default=None)
    parser.add_argument('--top-k', type=int, default=30,
                        help='Number of most discriminative channels to highlight')
    parser.add_argument('--output-dir', type=str, default='./vis_channel_classes')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)

    print(f'Loading checkpoint: {args.checkpoint}')
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model = build_model(args, ckpt, device)
    transform, data_config = get_eval_transform(model)
    img_size = data_config['input_size'][-2:]

    clean_root = os.path.join(args.clean_data_dir, args.clean_split)
    clean_dataset = ImageFolder(clean_root)
    print(f'Clean dataset: {len(clean_dataset)} images from {clean_root}')

    corrupt_datasets = {}
    for corr in CORRUPTIONS:
        corr_root = os.path.join(args.data_root, corr, str(args.severity))
        corrupt_datasets[corr] = ImageFolder(corr_root)

    if args.image_indices is not None:
        indices = args.image_indices
    else:
        indices = random.sample(range(len(clean_dataset)),
                                min(args.num_images, len(clean_dataset)))

    print(f'\nVisualizing {len(indices)} images: {indices}')
    all_attn = []
    for idx in indices:
        print(f'\n--- Image {idx} ---')
        attn_np = visualize_one_image(
            model, clean_dataset, corrupt_datasets, idx,
            transform, device, args.output_dir, img_size, args.top_k,
        )
        all_attn.append(attn_np)

    if len(all_attn) > 1:
        print(f'\n--- Aggregate over {len(indices)} images ---')
        visualize_aggregate(all_attn, indices, args.output_dir, args.top_k)

    print(f'\nAll done! Visualizations saved to {args.output_dir}/')


if __name__ == '__main__':
    main()
