#!/usr/bin/env python3
"""Visualize photometric augmentation effects across severity scales.

Uses PIL only (no matplotlib) to compose grid images.

Produces two images per source image:
1. Per-transform grid: rows=8 transforms, cols=severity levels 0.0→1.0
2. AugMix composite: cols=total severity sums 0.0→2.5, rows=random samples
"""
import argparse
import glob
import os
import random
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(__file__))
from timm.data.auto_augment import (
    _AUGMIX_SL_TRANSFORMS_V2,
    augmix_sl_ops_v2_deterministic,
    _HPARAMS_DEFAULT,
)

_SAFE_MAG_CAPS = {
    'NegativeIntensity': 0.8,
    'SolarizeIncreasing': 0.7,
}

FONT_SIZE = 20
HEADER_H = 36
PAD = 6


def _get_font(size=FONT_SIZE):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except (IOError, OSError):
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", size)
        except (IOError, OSError):
            return ImageFont.load_default()


def _text_img(text, w, h=HEADER_H, bg=(255, 255, 255), fg=(0, 0, 0), font=None):
    """Create a small PIL image with centered text."""
    img = Image.new('RGB', (w, h), bg)
    draw = ImageDraw.Draw(img)
    font = font or _get_font()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((w - tw) // 2, (h - th) // 2), text, fill=fg, font=font)
    return img


def load_random_image(data_dir, target_size=224):
    classes = sorted(os.listdir(data_dir))
    cls = random.choice(classes)
    cls_dir = os.path.join(data_dir, cls)
    imgs = glob.glob(os.path.join(cls_dir, '*.JPEG')) + \
           glob.glob(os.path.join(cls_dir, '*.jpeg')) + \
           glob.glob(os.path.join(cls_dir, '*.png'))
    img_path = random.choice(imgs)
    img = Image.open(img_path).convert('RGB')
    img = img.resize((target_size, target_size), Image.BICUBIC)
    return img, os.path.basename(img_path)


def compose_grid(cells, row_labels, col_labels, cell_size=224):
    """Compose a grid image from 2D list of PIL images with row/col headers."""
    n_rows = len(cells)
    n_cols = len(cells[0]) if n_rows > 0 else 0
    font = _get_font()

    label_w = 260
    total_w = label_w + n_cols * (cell_size + PAD) + PAD
    total_h = HEADER_H + n_rows * (cell_size + PAD) + PAD

    canvas = Image.new('RGB', (total_w, total_h), (255, 255, 255))

    for j, col_label in enumerate(col_labels):
        header = _text_img(col_label, cell_size, HEADER_H, font=font)
        canvas.paste(header, (label_w + j * (cell_size + PAD) + PAD, 0))

    for i in range(n_rows):
        y = HEADER_H + i * (cell_size + PAD) + PAD
        row_header = _text_img(row_labels[i], label_w, cell_size, font=font)
        canvas.paste(row_header, (0, y))

        for j in range(n_cols):
            x = label_w + j * (cell_size + PAD) + PAD
            cell = cells[i][j]
            if cell.size != (cell_size, cell_size):
                cell = cell.resize((cell_size, cell_size), Image.BICUBIC)
            canvas.paste(cell, (x, y))

    return canvas


_DISPLAY_NAMES = {
    'PositiveIntensity': 'Pos. Intensity',
    'NegativeIntensity': 'Neg. Intensity',
    'SaturationV2': 'Saturation',
    'SharpnessV2': 'Sharpness',
    'GaussianBlurIncreasing': 'GaussianBlur',
    'PosterizeIncreasing': 'Posterize',
    'SolarizeIncreasing': 'Solarize',
    'SaltAndPepperIncreasing': 'SaltAndPepper',
}


def visualize_per_transform(img, ops, severity_levels, output_path, cell_size=180):
    """Grid: rows=transforms, cols=severity levels."""
    cells = []
    row_labels = []
    col_labels = [f'Magnitude {sv:.2f}' for sv in severity_levels]

    for op in ops:
        row = []
        cap = _SAFE_MAG_CAPS.get(op.name, 1.0)
        for sv in severity_levels:
            row.append(op(img.copy(), min(sv, cap)))
        cells.append(row)
        name = _DISPLAY_NAMES.get(op.name, op.name)
        if cap < 1.0:
            name += f' (cap={cap})'
        row_labels.append(name)

    grid = compose_grid(cells, row_labels, col_labels, cell_size=cell_size)
    grid.save(output_path, quality=95)
    print(f'  Saved: {output_path}')


def visualize_augmix_composite(img, ops, sum_levels, n_samples=5, max_depth=3,
                                output_path='augmix_composite.png', cell_size=180):
    """Grid: rows=random samples, cols=total severity sums."""
    num_ops = len(ops)
    cells = []
    row_labels = []
    col_labels = ['Original'] + [f'sum={sv:.1f}' for sv in sum_levels]

    for row_idx in range(n_samples):
        row = [img.copy()]
        for total_sv in sum_levels:
            aug_img = img.copy()
            if total_sv == 0.0:
                row.append(aug_img)
                continue

            n_tf = random.randint(1, max_depth)
            selected = random.sample(range(num_ops), n_tf)

            if n_tf == 1:
                shares = np.array([1.0])
            else:
                shares = np.random.dirichlet(np.ones(n_tf))
            severities = shares * total_sv

            for k, idx in enumerate(selected):
                sv = float(severities[k])
                normalized = min(sv / 1.0, 1.0)
                cap = _SAFE_MAG_CAPS.get(ops[idx].name, 1.0)
                normalized = min(normalized, cap)
                aug_img = ops[idx](aug_img, normalized)

            row.append(aug_img)
        cells.append(row)
        row_labels.append(f'Sample {row_idx + 1}')

    grid = compose_grid(cells, row_labels, col_labels, cell_size=cell_size)
    grid.save(output_path, quality=95)
    print(f'  Saved: {output_path}')


def main():
    parser = argparse.ArgumentParser(description='Visualize augmentation severity scales')
    parser.add_argument('--data-dir', type=str, default='/data/imagenet/imagenet/val')
    parser.add_argument('--output-dir', type=str, default='./vis_aug_scales')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--cell-size', type=int, default=180)
    parser.add_argument('--n-images', type=int, default=3)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    ops = augmix_sl_ops_v2_deterministic(hparams=_HPARAMS_DEFAULT)
    print(f'Transforms ({len(ops)}): {[op.name for op in ops]}')

    severity_levels = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 1.0]
    augmix_sum_levels = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]

    for img_idx in range(args.n_images):
        img, fname = load_random_image(args.data_dir, target_size=args.img_size)
        stem = os.path.splitext(fname)[0]
        print(f'\nImage {img_idx + 1}/{args.n_images}: {fname}')

        visualize_per_transform(
            img, ops, severity_levels,
            output_path=os.path.join(args.output_dir, f'per_transform_{stem}.png'),
            cell_size=args.cell_size,
        )

        visualize_augmix_composite(
            img, ops, augmix_sum_levels,
            n_samples=5, max_depth=3,
            output_path=os.path.join(args.output_dir, f'augmix_composite_{stem}.png'),
            cell_size=args.cell_size,
        )

    print(f'\nAll visualizations saved to {args.output_dir}/')


if __name__ == '__main__':
    main()
