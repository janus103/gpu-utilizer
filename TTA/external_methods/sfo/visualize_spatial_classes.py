#!/usr/bin/env python3
"""Visualize SpatialAttention2 per-position class assignments across domains.

For each sample, produces a (2 rows × 16 cols) grid:
  Row 0: original image / 15 corrupted images
  Row 1: argmax class map overlaid on the image

Columns: [clean, gaussian_noise, shot_noise, ..., jpeg_compression]

The class map has `prop_size` discrete classes (default 5).
Source domain is trained to be class 0, target domain class N-1,
and intermediate classes emerge via Information Maximization.

Usage:
  python visualize_spatial_classes.py \
      --checkpoint output/fedavg_direct_k2_tm1_le1_p5_wu10_aa5/best.pth \
      --num-images 5 --output-dir ./vis_spatial_classes
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
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
    'snow', 'frost', 'fog', 'brightness',
    'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
]

DOMAIN_LABELS = ['clean'] + CORRUPTIONS


def _get_base_model(model):
    m = model
    if hasattr(m, 'module'):
        m = m.module
    if hasattr(m, '_orig_mod'):
        m = m._orig_mod
    return m


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
    return model, prop_size


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


def load_image_from_dataset(dataset, idx):
    """Load a PIL image and label from an ImageFolder dataset."""
    path, label = dataset.samples[idx]
    pil_img = Image.open(path).convert('RGB')
    return pil_img, label, path


def get_class_map(model, tensor_batch, device):
    """Run forward pass and extract spatial attention logits → argmax class map.

    Returns:
        argmax_map: (B, H, W) int tensor — class index per spatial position
        probs:      (B, prop_size, H, W) float tensor — softmax probabilities
        logits:     (B, prop_size, H, W) float tensor — raw logits
    """
    tensor_batch = tensor_batch.to(device)
    with torch.no_grad():
        _ = model(tensor_batch)
    base = _get_base_model(model)
    logits = base.spatial_attn._logits  # (B, prop_size, H, W)
    probs = F.softmax(logits, dim=1)
    argmax_map = probs.argmax(dim=1)  # (B, H, W)
    return argmax_map.cpu(), probs.cpu(), logits.cpu()


def make_discrete_cmap(prop_size):
    """Create a discrete colormap for `prop_size` classes."""
    base_colors = [
        '#2166ac',  # class 0 (source) — blue
        '#67a9cf',  # class 1
        '#f7f7f7',  # class 2 (middle) — neutral/white
        '#ef8a62',  # class 3
        '#b2182b',  # class 4 (target) — red
    ]
    if prop_size <= len(base_colors):
        colors = base_colors[:prop_size]
    else:
        cmap_base = plt.cm.RdBu_r
        colors = [cmap_base(i / (prop_size - 1)) for i in range(prop_size)]

    cmap = ListedColormap(colors, name='spatial_class')
    bounds = np.arange(-0.5, prop_size, 1)
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm


def _upsample_map(arr, img_size):
    """Upsample a 2D float array to img_size using nearest interpolation."""
    return np.array(Image.fromarray(arr.astype(np.float32)).resize(
        (img_size[1], img_size[0]), Image.NEAREST))


def _class_intensity_rgb(argmax_up, maxprob_up, prop_size, img_size):
    """Build an RGB image where hue = argmax class color and brightness = softmax prob."""
    base_hex = ['#2166ac', '#67a9cf', '#aaaaaa', '#ef8a62', '#b2182b']
    rgb_img = np.zeros((*img_size, 3), dtype=np.float32)
    for ci in range(prop_size):
        mask = (argmax_up == ci)
        base_rgb = np.array(mcolors.to_rgb(base_hex[min(ci, len(base_hex) - 1)]))
        intensity = maxprob_up[mask, np.newaxis]  # (N, 1) in [0, 1]
        rgb_img[mask] = base_rgb * intensity
    return np.clip(rgb_img, 0, 1)


def visualize_one_image(
    model, clean_dataset, corrupt_datasets, img_idx,
    transform, device, prop_size, output_dir, img_size=(224, 224),
):
    """Create visualization for one image across all domains.

    Layout (3 rows × 16 cols):
      Row 0: original images
      Row 1: class × intensity map (hue = argmax class, brightness = softmax prob)
      Row 2: overlay on original image
    """
    base_hex = ['#2166ac', '#67a9cf', '#aaaaaa', '#ef8a62', '#b2182b']

    pil_img, label, img_path = load_image_from_dataset(clean_dataset, img_idx)
    class_name = os.path.basename(os.path.dirname(img_path))

    pil_images = [pil_img]
    for corr_name in CORRUPTIONS:
        c_path, _ = corrupt_datasets[corr_name].samples[img_idx]
        c_pil = Image.open(c_path).convert('RGB')
        pil_images.append(c_pil)

    tensors = torch.stack([transform(p) for p in pil_images])
    vis_images = [np.array(p.resize(img_size, Image.BILINEAR)) for p in pil_images]

    argmax_maps, probs, logits = get_class_map(model, tensors, device)

    H_feat, W_feat = argmax_maps.shape[1], argmax_maps.shape[2]
    n_domains = len(DOMAIN_LABELS)

    fig, axes = plt.subplots(
        3, n_domains,
        figsize=(2.8 * n_domains, 9.0),
        gridspec_kw={'hspace': 0.05, 'wspace': 0.04},
    )

    for col_idx in range(n_domains):
        argmax_np = argmax_maps[col_idx].numpy()
        max_prob = probs[col_idx].max(dim=0)[0].numpy()

        argmax_up = _upsample_map(argmax_np.astype(float), img_size)
        maxprob_up = _upsample_map(max_prob, img_size)

        ci_rgb = _class_intensity_rgb(argmax_up, maxprob_up, prop_size, img_size)

        # Row 0: image
        axes[0, col_idx].imshow(vis_images[col_idx])
        axes[0, col_idx].set_title(DOMAIN_LABELS[col_idx], fontsize=7, fontweight='bold')
        axes[0, col_idx].axis('off')

        # Row 1: class + intensity
        axes[1, col_idx].imshow(ci_rgb, interpolation='nearest')
        axes[1, col_idx].axis('off')

        # Row 2: overlay
        vis_f = vis_images[col_idx].astype(np.float32) / 255.0
        blended = vis_f * 0.45 + ci_rgb * 0.55
        axes[2, col_idx].imshow(np.clip(blended, 0, 1), interpolation='nearest')
        axes[2, col_idx].axis('off')

    axes[0, 0].set_ylabel('Image', fontsize=9, rotation=0, labelpad=50, va='center')
    axes[1, 0].set_ylabel('Class ×\nIntensity', fontsize=9, rotation=0, labelpad=50, va='center')
    axes[2, 0].set_ylabel('Overlay', fontsize=9, rotation=0, labelpad=50, va='center')

    # --- Legend bar at the TOP ---
    fig.subplots_adjust(top=0.86)

    # Class color patches + probability gradient bar
    legend_patches = []
    for ci in range(prop_size):
        lbl = f'{ci}'
        if ci == 0:
            lbl = '0 (src)'
        elif ci == prop_size - 1:
            lbl = f'{ci} (tgt)'
        legend_patches.append(Patch(
            facecolor=base_hex[min(ci, len(base_hex) - 1)], edgecolor='black',
            linewidth=0.5, label=lbl))
    fig.legend(
        handles=legend_patches, loc='upper left',
        bbox_to_anchor=(0.02, 0.97), ncol=prop_size,
        fontsize=8, title='Argmax Class', title_fontsize=9,
        frameon=True, fancybox=True, edgecolor='gray',
    )

    cbar_ax = fig.add_axes([0.55, 0.93, 0.35, 0.012])
    sm = plt.cm.ScalarMappable(cmap='gray_r', norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Softmax Probability (brightness)', fontsize=8, labelpad=3)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle(
        f'SpatialAttention2 — Image #{img_idx} ({class_name}, label={label})  |  '
        f'prop_size={prop_size}, feat={H_feat}×{W_feat}',
        fontsize=11, fontweight='bold', y=0.99,
    )

    fname = os.path.join(output_dir, f'spatial_class_{img_idx:05d}.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight', pad_inches=0.15)
    plt.close(fig)
    print(f'  Saved: {fname}')

    # --- Per-class probability distribution comparison ---
    fig2, axes2 = plt.subplots(1, n_domains, figsize=(2.5 * n_domains, 3))
    for col_idx in range(n_domains):
        p = probs[col_idx]  # (prop_size, H, W)
        mean_probs = p.mean(dim=(1, 2)).numpy()  # (prop_size,)
        bars = axes2[col_idx].bar(
            range(prop_size), mean_probs,
            color=[base_hex[min(i, len(base_hex) - 1)] for i in range(prop_size)],
            edgecolor='black', linewidth=0.5,
        )
        axes2[col_idx].set_ylim(0, 1)
        axes2[col_idx].set_title(DOMAIN_LABELS[col_idx], fontsize=7)
        axes2[col_idx].set_xticks(range(prop_size))
        axes2[col_idx].tick_params(labelsize=6)
        if col_idx == 0:
            axes2[col_idx].set_ylabel('Mean Prob', fontsize=8)

    fig2.suptitle(
        f'Mean Class Probability — Image #{img_idx}',
        fontsize=11, fontweight='bold',
    )
    plt.tight_layout()
    fname2 = os.path.join(output_dir, f'spatial_probs_{img_idx:05d}.png')
    plt.savefig(fname2, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig2)
    print(f'  Saved: {fname2}')


def main():
    parser = argparse.ArgumentParser(
        description='Visualize SpatialAttention2 class maps across domains')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to .pth checkpoint file')
    parser.add_argument('--clean-data-dir', type=str, default='/data/imagenet/imagenet',
                        help='Clean ImageNet root')
    parser.add_argument('--clean-split', type=str, default='val')
    parser.add_argument('--data-root', type=str, default='/home/oem/servers/imagenet-c',
                        help='ImageNet-C root')
    parser.add_argument('--severity', type=int, default=5)
    parser.add_argument('--num-images', type=int, default=5,
                        help='Number of images to visualize')
    parser.add_argument('--image-indices', type=int, nargs='+', default=None,
                        help='Specific image indices to visualize (overrides --num-images)')
    parser.add_argument('--output-dir', type=str, default='./vis_spatial_classes')
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
    model, prop_size = build_model(args, ckpt, device)
    transform, data_config = get_eval_transform(model)
    img_size = data_config['input_size'][-2:]

    print(f'prop_size={prop_size}, img_size={img_size}')

    clean_root = os.path.join(args.clean_data_dir, args.clean_split)
    clean_dataset = ImageFolder(clean_root)
    print(f'Clean dataset: {len(clean_dataset)} images from {clean_root}')

    corrupt_datasets = {}
    for corr in CORRUPTIONS:
        corr_root = os.path.join(args.data_root, corr, str(args.severity))
        corrupt_datasets[corr] = ImageFolder(corr_root)
        assert len(corrupt_datasets[corr]) == len(clean_dataset), (
            f'{corr}: {len(corrupt_datasets[corr])} != {len(clean_dataset)}')

    if args.image_indices is not None:
        indices = args.image_indices
    else:
        indices = random.sample(range(len(clean_dataset)), min(args.num_images, len(clean_dataset)))

    print(f'\nVisualizing {len(indices)} images: {indices}')
    for idx in indices:
        print(f'\n--- Image {idx} ---')
        visualize_one_image(
            model, clean_dataset, corrupt_datasets, idx,
            transform, device, prop_size, args.output_dir, img_size,
        )

    print(f'\nAll done! Visualizations saved to {args.output_dir}/')


if __name__ == '__main__':
    main()
