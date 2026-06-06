#!/usr/bin/env python3
"""Visualize SpatialAttention masks on original vs augmented validation images.

Runs a single mini-batch through the model and saves per-image visualizations
comparing the mask produced by original (clean) images versus augmented images.

Output layout per image (saved as PNG to ./result_img/):
  [original] [orig + mask overlay] [augmented] [aug + mask overlay]

Usage example (ViT):
  python visualize_mask.py --data-dir /path/to/imagenet \
      --model vit_base_patch16_224 --pretrained-path ./ckpt.pth \
      --parallel-attention --aa v0 -b 16

Usage example (ResNet):
  python visualize_mask.py --data-dir /path/to/imagenet \
      --model resnet26 --pretrained-path ./ckpt.pth \
      --parallel-attention --aa v0 -b 16
"""
import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from timm.data import create_dataset, resolve_data_config
from timm.data.auto_augment import auto_augment_policy, GEOMETRIC_OPS
from timm.data.transforms_factory import transforms_imagenet_train, transforms_imagenet_eval
from timm.models import create_model

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='Mask Visualization')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to dataset root')
    parser.add_argument('--dataset', default='', type=str)
    parser.add_argument('--val-split', default='validation', type=str)
    parser.add_argument('--model', default='vit_base_patch16_224', type=str)
    parser.add_argument('--pretrained', action='store_true', default=False)
    parser.add_argument('--pretrained-path', default=None, type=str)
    parser.add_argument('--initial-checkpoint', default='', type=str)
    parser.add_argument('--num-classes', type=int, default=None)
    parser.add_argument('-b', '--batch-size', type=int, default=16)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--aa', type=str, default='v0', help='AutoAugment policy (default: v0)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', default='./result_img', type=str)

    # Auxiliary module args (passed through to create_model)
    parser.add_argument('--use-se-module', action='store_true', default=False)
    parser.add_argument('--use-sam-module', type=int, default=-1)
    parser.add_argument('--reverse-se', action='store_true', default=False)
    parser.add_argument('--parallel-attention', action='store_true', default=False)
    parser.add_argument('--vit-early-norm-types', type=int, nargs=4, default=None)
    parser.add_argument('--vit-kernel-size', type=int, default=7)
    parser.add_argument('--spatial-group-size', type=int, default=1)
    parser.add_argument('--vit-last', action='store_true', default=False)
    parser.add_argument('--vit-closed', type=str, default=None, choices=['same', 'diff'])
    parser.add_argument('--model-kwargs', nargs='*', default={})

    parser.add_argument('--crop-pct', default=None, type=float)
    parser.add_argument('--input-size', default=None, nargs=3, type=int)
    parser.add_argument('--img-size', type=int, default=None)
    parser.add_argument('--in-chans', type=int, default=None)
    parser.add_argument('--interpolation', default='', type=str)
    parser.add_argument('--mean', type=float, nargs='+', default=None)
    parser.add_argument('--std', type=float, nargs='+', default=None)
    parser.add_argument('--gp', default=None, type=str)
    return parser.parse_args()


class VisDataset(torch.utils.data.Dataset):
    """Produces (original_tensor, augmented_tensor, original_pil_resized, aug_pil_resized, target).

    Geometric AA ops are shared between original and augmented;
    color AA ops are applied only to the augmented copy.
    Both copies go through the same crop/resize, so they are spatially aligned.
    """

    def __init__(self, dataset, primary_transform, aa_policy, final_transform, eval_transform):
        self.dataset = dataset
        self.dataset.transform = None
        self.primary_transform = primary_transform
        self.aa_policy = aa_policy
        self.final_transform = final_transform
        self.eval_transform = eval_transform

    @staticmethod
    def _is_degenerate(pil_img, threshold=5.0):
        """Check if a PIL image is nearly all-black (or all single-value)."""
        arr = np.array(pil_img)
        return arr.mean() < threshold

    def __getitem__(self, index):
        img, target = self.dataset[index]

        img = self.primary_transform(img)

        sub_policy = random.choice(self.aa_policy)
        geo_ops = [op for op in sub_policy if op.name in GEOMETRIC_OPS]
        color_ops = [op for op in sub_policy if op.name not in GEOMETRIC_OPS]

        for op in geo_ops:
            img = op(img)

        orig_pil = img.copy()

        # Apply color ops one-by-one; roll back if the result degenerates
        aug_img = img.copy()
        for op in color_ops:
            candidate = op(aug_img)
            if candidate.mode != 'RGB':
                candidate = candidate.convert('RGB')
            if self._is_degenerate(candidate):
                continue
            aug_img = candidate

        orig_tensor = self.final_transform(orig_pil)
        aug_tensor = self.final_transform(aug_img)

        orig_vis = np.array(orig_pil.resize((224, 224), Image.BILINEAR))
        aug_vis = np.array(aug_img.resize((224, 224), Image.BILINEAR))

        return orig_tensor, aug_tensor, orig_vis, aug_vis, target

    def __len__(self):
        return len(self.dataset)


def save_visualization(
    orig_vis, aug_vis, orig_mask, aug_mask, idx, output_dir, mean, std,
):
    """Save a single image's visualization as PNG.

    Layout: [original] [orig+mask] [augmented] [aug+mask]
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    titles = ['Original', 'Original + Mask', 'Augmented', 'Augmented + Mask']

    axes[0].imshow(orig_vis)
    axes[0].set_title(titles[0], fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(orig_vis)
    h = axes[1].imshow(orig_mask, cmap='jet', alpha=0.5, vmin=0, vmax=1)
    axes[1].set_title(titles[1], fontsize=12)
    axes[1].axis('off')

    axes[2].imshow(aug_vis)
    axes[2].set_title(titles[2], fontsize=12)
    axes[2].axis('off')

    axes[3].imshow(aug_vis)
    axes[3].imshow(aug_mask, cmap='jet', alpha=0.5, vmin=0, vmax=1)
    axes[3].set_title(titles[3], fontsize=12)
    axes[3].axis('off')

    plt.colorbar(h, ax=axes, shrink=0.8, label='Mask Intensity')
    plt.suptitle(f'Image {idx}: Spatial Attention Mask Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f'mask_vis_{idx:04d}.png'),
        dpi=150, bbox_inches='tight', pad_inches=0.1,
    )
    plt.close(fig)


def save_batch_grid(
    orig_vis_batch, aug_vis_batch, orig_masks, aug_masks, output_dir, mean, std,
):
    """Save a grid overview of the entire mini-batch."""
    B = len(orig_vis_batch)
    cols = 4
    rows = B
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    if rows == 1:
        axes = axes[np.newaxis, :]

    for i in range(B):
        axes[i, 0].imshow(orig_vis_batch[i])
        axes[i, 0].axis('off')
        if i == 0:
            axes[i, 0].set_title('Original', fontsize=11)

        axes[i, 1].imshow(orig_vis_batch[i])
        axes[i, 1].imshow(orig_masks[i], cmap='jet', alpha=0.5, vmin=0, vmax=1)
        axes[i, 1].axis('off')
        if i == 0:
            axes[i, 1].set_title('Original + Mask', fontsize=11)

        axes[i, 2].imshow(aug_vis_batch[i])
        axes[i, 2].axis('off')
        if i == 0:
            axes[i, 2].set_title('Augmented', fontsize=11)

        axes[i, 3].imshow(aug_vis_batch[i])
        axes[i, 3].imshow(aug_masks[i], cmap='jet', alpha=0.5, vmin=0, vmax=1)
        axes[i, 3].axis('off')
        if i == 0:
            axes[i, 3].set_title('Augmented + Mask', fontsize=11)

    plt.suptitle('Batch: Spatial Attention Mask - Original vs Augmented', fontsize=14)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, 'batch_grid.png'),
        dpi=120, bbox_inches='tight', pad_inches=0.1,
    )
    plt.close(fig)


def _get_base_model(model):
    m = model
    if hasattr(m, 'module'):
        m = m.module
    if hasattr(m, '_orig_mod'):
        m = m._orig_mod
    return m


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)

    # ---- Build model ----
    factory_kwargs = {}
    if args.pretrained_path:
        factory_kwargs['pretrained_cfg_overlay'] = dict(file=args.pretrained_path)

    vit_model = 'vit' in args.model.lower()
    vit_norm_kwargs = {}
    if args.vit_early_norm_types is not None:
        vit_norm_kwargs['vit_early_norm_types'] = args.vit_early_norm_types
    if vit_model and (args.use_sam_module != -1 or args.parallel_attention):
        vit_norm_kwargs['sam_kernel_size'] = args.vit_kernel_size
        vit_norm_kwargs['spatial_group_size'] = args.spatial_group_size
    if vit_model and args.vit_last:
        vit_norm_kwargs['vit_last'] = True
    if vit_model and args.vit_closed is not None:
        vit_norm_kwargs['vit_closed'] = args.vit_closed

    model_kwargs = {}
    if isinstance(args.model_kwargs, dict):
        model_kwargs = args.model_kwargs

    model = create_model(
        args.model,
        pretrained=args.pretrained,
        in_chans=args.in_chans or 3,
        num_classes=args.num_classes,
        checkpoint_path=args.initial_checkpoint,
        use_se_module=args.use_se_module,
        use_sam_module=args.use_sam_module,
        reverse_se_sam=args.reverse_se,
        parallel_attention=args.parallel_attention,
        global_pool=args.gp,
        **factory_kwargs,
        **vit_norm_kwargs,
        **model_kwargs,
    )
    model.to(device)
    model.eval()

    data_config = resolve_data_config(vars(args), model=model)
    img_size = data_config['input_size'][-2:]
    mean = data_config['mean']
    std = data_config['std']
    interpolation = data_config['interpolation']
    crop_pct = data_config['crop_pct']

    print(f'Model: {args.model}, img_size={img_size}, mean={mean}, std={std}')
    print(f'AA policy: {args.aa}, batch_size={args.batch_size}')

    # ---- Build dataset with paired original/augmented transforms ----
    dataset_val = create_dataset(
        args.dataset,
        root=args.data_dir,
        split=args.val_split,
        is_training=False,
        batch_size=args.batch_size,
    )

    primary_tf, _, final_tf = transforms_imagenet_train(
        img_size=img_size,
        scale=(0.08, 1.0),
        ratio=(3. / 4., 4. / 3.),
        hflip=0.5,
        vflip=0.,
        color_jitter=0.4,
        auto_augment=args.aa,
        interpolation='random',
        mean=mean,
        std=std,
        re_prob=0.,
        use_prefetcher=False,
        separate=True,
    )

    img_size_min = min(img_size) if isinstance(img_size, (tuple, list)) else img_size
    aa_hparams = dict(
        translate_const=int(img_size_min * 0.45),
        img_mean=tuple([min(255, round(255 * x)) for x in mean]),
    )
    aa_policy_name = args.aa.split('-')[0]
    aa_policy = auto_augment_policy(aa_policy_name, hparams=aa_hparams)

    eval_tf = transforms_imagenet_eval(
        img_size=img_size,
        crop_pct=crop_pct,
        interpolation=interpolation,
        mean=mean,
        std=std,
    )

    vis_dataset = VisDataset(dataset_val, primary_tf, aa_policy, final_tf, eval_tf)

    loader = torch.utils.data.DataLoader(
        vis_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )

    # ---- Run single mini-batch ----
    print('Running inference on first mini-batch...')
    orig_tensors, aug_tensors, orig_vis_np, aug_vis_np, targets = next(iter(loader))
    orig_tensors = orig_tensors.to(device)
    aug_tensors = aug_tensors.to(device)

    with torch.no_grad():
        _ = model(orig_tensors)
        base_model = _get_base_model(model)
        orig_masks_list = base_model.get_last_masks()

        _ = model(aug_tensors)
        aug_masks_list = base_model.get_last_masks()

    if orig_masks_list is None or aug_masks_list is None:
        print('ERROR: No spatial attention masks found. '
              'Make sure --parallel-attention is enabled and the model supports get_last_masks().')
        return

    orig_mask = orig_masks_list[0]  # (B, 1, H_mask, W_mask)
    aug_mask = aug_masks_list[0]

    B = orig_mask.shape[0]
    target_h, target_w = img_size

    orig_mask_up = F.interpolate(orig_mask, size=(target_h, target_w), mode='bilinear', align_corners=False)
    aug_mask_up = F.interpolate(aug_mask, size=(target_h, target_w), mode='bilinear', align_corners=False)

    orig_mask_np = orig_mask_up.squeeze(1).cpu().numpy()  # (B, H, W)
    aug_mask_np = aug_mask_up.squeeze(1).cpu().numpy()

    print(f'Mask stats - orig: min={orig_mask_np.min():.3f}, max={orig_mask_np.max():.3f}, '
          f'mean={orig_mask_np.mean():.3f}')
    print(f'Mask stats -  aug: min={aug_mask_np.min():.3f}, max={aug_mask_np.max():.3f}, '
          f'mean={aug_mask_np.mean():.3f}')

    # ---- Save individual visualizations ----
    orig_vis_list = []
    aug_vis_list = []
    orig_masks_resized = []
    aug_masks_resized = []

    for i in range(B):
        o_vis = orig_vis_np[i].numpy() if isinstance(orig_vis_np[i], torch.Tensor) else orig_vis_np[i]
        a_vis = aug_vis_np[i].numpy() if isinstance(aug_vis_np[i], torch.Tensor) else aug_vis_np[i]

        o_mask = orig_mask_np[i]
        a_mask = aug_mask_np[i]

        save_visualization(o_vis, a_vis, o_mask, a_mask, i, args.output_dir, mean, std)

        orig_vis_list.append(o_vis)
        aug_vis_list.append(a_vis)
        orig_masks_resized.append(o_mask)
        aug_masks_resized.append(a_mask)

    save_batch_grid(orig_vis_list, aug_vis_list, orig_masks_resized, aug_masks_resized,
                    args.output_dir, mean, std)

    # ---- Save raw mask comparison (difference map) ----
    diff_masks = np.abs(orig_mask_np - aug_mask_np)
    fig, axes = plt.subplots(2, min(B, 8), figsize=(2.5 * min(B, 8), 6))
    if min(B, 8) == 1:
        axes = axes[:, np.newaxis]
    n_show = min(B, 8)
    for i in range(n_show):
        axes[0, i].imshow(orig_mask_np[i], cmap='jet', vmin=0, vmax=1)
        axes[0, i].set_title(f'Orig #{i}', fontsize=8)
        axes[0, i].axis('off')

        axes[1, i].imshow(aug_mask_np[i], cmap='jet', vmin=0, vmax=1)
        axes[1, i].set_title(f'Aug #{i}', fontsize=8)
        axes[1, i].axis('off')

    plt.suptitle('Raw Masks: Original (top) vs Augmented (bottom)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'mask_comparison_raw.png'), dpi=150, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(1, n_show, figsize=(2.5 * n_show, 3))
    if n_show == 1:
        axes = [axes]
    for i in range(n_show):
        im = axes[i].imshow(diff_masks[i], cmap='hot', vmin=0, vmax=diff_masks.max())
        axes[i].set_title(f'|diff| #{i}', fontsize=8)
        axes[i].axis('off')
    plt.colorbar(im, ax=axes, shrink=0.8)
    plt.suptitle('Absolute Mask Difference (Original - Augmented)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'mask_diff.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f'\nVisualization saved to {args.output_dir}/')
    print(f'  - mask_vis_XXXX.png  : per-image (original vs augmented with overlay)')
    print(f'  - batch_grid.png     : full batch grid overview')
    print(f'  - mask_comparison_raw.png : raw mask heatmaps')
    print(f'  - mask_diff.png      : absolute difference between masks')


if __name__ == '__main__':
    main()
