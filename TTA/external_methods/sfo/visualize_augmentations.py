#!/usr/bin/env python3
"""Visualize V2 Augmentations at different severity levels.

This script takes an input image and creates a grid visualization showing
each augmentation from the V2 policy at severity levels 0.2, 0.4, 0.6, 0.8, 1.0.

V2 Augmentations:
- IntensityIncreasing: Combined Brightness + Contrast
- SaturationIncreasing: Color/Saturation
- SharpnessIncreasing: Edge sharpness
- GaussianBlurIncreasing: Blur effect
- PosterizeIncreasing: Color quantization
- SolarizeIncreasing: Inversion based on threshold

Usage:
    python visualize_augmentations.py /path/to/image.jpg
    python visualize_augmentations.py /path/to/image.png --output /path/to/output.png
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Add timm to path if needed
sys.path.insert(0, str(Path(__file__).parent))

from timm.data.auto_augment import (
    augmix_sl_ops_v2_deterministic,
    get_augmix_sl_transform_names,
)


def visualize_augmentations(
    image_path: str,
    output_path: str = None,
    severity_levels: list = None,
    figsize_per_cell: tuple = (2.5, 1.5),
    dpi: int = 150,
) -> str:
    """Create a grid visualization of V2 augmentations at different severity levels.
    
    Args:
        image_path: Path to input image (.jpg or .png).
        output_path: Path for output image. If None, auto-generated.
        severity_levels: List of severity levels to apply. Default: [0.2, 0.4, 0.6, 0.8, 1.0].
        figsize_per_cell: Figure size per cell (width, height).
        dpi: Output image DPI.
    
    Returns:
        Path to the saved output image.
    """
    # Default severity levels
    if severity_levels is None:
        severity_levels = [0.2, 0.4, 0.6, 0.8, 1.0]
    
    # Validate input image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")
    
    # Load image
    img = Image.open(image_path).convert('RGB')
    
    # Get V2 augmentation ops (deterministic version for consistent visualization)
    aug_ops = augmix_sl_ops_v2_deterministic()
    transform_names = get_augmix_sl_transform_names(version=2)
    
    # Grid dimensions: rows = augmentations + 1 (original), cols = severity levels + 1 (original)
    n_augs = len(transform_names)
    n_levels = len(severity_levels)
    
    # Create figure
    # Rows: Original + each augmentation
    # Cols: Original + each severity level
    n_rows = n_augs + 1
    n_cols = n_levels + 1
    
    fig_width = figsize_per_cell[0] * n_cols
    fig_height = figsize_per_cell[1] * n_rows
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), dpi=dpi)
    
    # Remove space between subplots (minimal row spacing)
    fig.subplots_adjust(wspace=0.02, hspace=0.01)
    
    # First row: Original image repeated
    for col_idx in range(n_cols):
        ax = axes[0, col_idx]
        ax.imshow(np.array(img))
        ax.axis('off')
        
        if col_idx == 0:
            ax.set_title('Original', fontsize=10, fontweight='bold')
        else:
            ax.set_title(f'SL={severity_levels[col_idx-1]:.1f}', fontsize=10)
    
    # Remaining rows: Each augmentation at different severity levels
    for row_idx, (aug_op, aug_name) in enumerate(zip(aug_ops, transform_names), start=1):
        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]
            
            if col_idx == 0:
                # First column: show original for reference
                ax.imshow(np.array(img))
                ax.set_ylabel(aug_name, fontsize=9, rotation=0, ha='right', va='center')
                ax.yaxis.set_label_coords(-0.1, 0.5)
            else:
                # Apply augmentation at this severity level
                sl = severity_levels[col_idx - 1]
                try:
                    aug_img = aug_op(img.copy(), sl)
                    ax.imshow(np.array(aug_img))
                except Exception as e:
                    # If augmentation fails, show error message
                    ax.text(
                        0.5, 0.5, f'Error:\n{str(e)[:30]}',
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=8, color='red'
                    )
            
            ax.axis('off')
    
    # Add row labels on the left side
    for row_idx, aug_name in enumerate(transform_names, start=1):
        ax = axes[row_idx, 0]
        # Format augmentation name (remove 'Increasing' suffix for cleaner display)
        display_name = aug_name.replace('Increasing', '')
        ax.text(
            -0.02, 0.5, display_name,
            transform=ax.transAxes,
            fontsize=10,
            fontweight='bold',
            ha='right',
            va='center',
            rotation=0
        )
    
    # Add title
    fig.suptitle(
        'V2 Augmentations Visualization\n(Severity Level: 0.0 = No Change, 1.0 = Maximum)',
        fontsize=14,
        fontweight='bold',
        y=0.98
    )
    
    # Generate output path if not specified
    if output_path is None:
        input_stem = Path(image_path).stem
        output_path = str(Path(image_path).parent / f'{input_stem}_augmented_grid.png')
    
    # Save figure
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, facecolor='white')
    plt.close()
    
    print(f"Saved augmentation grid to: {output_path}")
    return output_path


def visualize_augmentations_compact(
    image_path: str,
    output_path: str = None,
    severity_levels: list = None,
    figsize: tuple = (16, 10),
    dpi: int = 150,
) -> str:
    """Create a compact grid visualization (augmentations as columns, severity as rows).
    
    This is an alternative layout where:
    - Columns: Each augmentation type
    - Rows: Each severity level
    
    Args:
        image_path: Path to input image (.jpg or .png).
        output_path: Path for output image. If None, auto-generated.
        severity_levels: List of severity levels to apply. Default: [0.2, 0.4, 0.6, 0.8, 1.0].
        figsize: Figure size (width, height).
        dpi: Output image DPI.
    
    Returns:
        Path to the saved output image.
    """
    # Default severity levels
    if severity_levels is None:
        severity_levels = [0.2, 0.4, 0.6, 0.8, 1.0]
    
    # Validate input image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")
    
    # Load image
    img = Image.open(image_path).convert('RGB')
    
    # Get V2 augmentation ops (deterministic version for consistent visualization)
    aug_ops = augmix_sl_ops_v2_deterministic()
    transform_names = get_augmix_sl_transform_names(version=2)
    
    n_augs = len(transform_names)
    n_levels = len(severity_levels)
    
    # Create figure: rows = severity levels, cols = augmentations + 1 (original)
    n_rows = n_levels
    n_cols = n_augs + 1
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=dpi)
    fig.subplots_adjust(wspace=0.02, hspace=0.1)
    
    for row_idx, sl in enumerate(severity_levels):
        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]
            
            if col_idx == 0:
                # First column: original image
                ax.imshow(np.array(img))
                if row_idx == 0:
                    ax.set_title('Original', fontsize=10, fontweight='bold')
                # Add severity level label on the left
                ax.set_ylabel(f'SL={sl:.1f}', fontsize=10, fontweight='bold')
            else:
                # Apply augmentation
                aug_op = aug_ops[col_idx - 1]
                aug_name = transform_names[col_idx - 1]
                
                try:
                    aug_img = aug_op(img.copy(), sl)
                    ax.imshow(np.array(aug_img))
                except Exception as e:
                    ax.text(
                        0.5, 0.5, f'Error',
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=8, color='red'
                    )
                
                if row_idx == 0:
                    # Add augmentation name as title (first row only)
                    display_name = aug_name.replace('Increasing', '')
                    ax.set_title(display_name, fontsize=10, fontweight='bold')
            
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
    
    # Add main title
    fig.suptitle(
        'V2 Augmentations at Different Severity Levels',
        fontsize=14,
        fontweight='bold',
        y=0.98
    )
    
    # Generate output path if not specified
    if output_path is None:
        input_stem = Path(image_path).stem
        output_path = str(Path(image_path).parent / f'{input_stem}_augmented_compact.png')
    
    # Save figure
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, facecolor='white')
    plt.close()
    
    print(f"Saved compact augmentation grid to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Visualize V2 Augmentations at different severity levels.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python visualize_augmentations.py input.jpg
    python visualize_augmentations.py input.png --output result.png
    python visualize_augmentations.py input.jpg --layout compact
    python visualize_augmentations.py input.jpg --severity 0.1 0.3 0.5 0.7 0.9 1.0
        """
    )
    
    parser.add_argument(
        'image_path',
        type=str,
        help='Path to input image (.jpg or .png)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Path for output image (default: <input>_augmented_grid.png)'
    )
    parser.add_argument(
        '--layout',
        type=str,
        choices=['grid', 'compact'],
        default='grid',
        help='Layout style: "grid" (default) or "compact"'
    )
    parser.add_argument(
        '--severity', '-s',
        type=float,
        nargs='+',
        default=[0.2, 0.4, 0.6, 0.8, 1.0],
        help='Severity levels to apply (default: 0.2 0.4 0.6 0.8 1.0)'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='Output image DPI (default: 150)'
    )
    
    args = parser.parse_args()
    
    # Validate severity levels
    for sl in args.severity:
        if not 0.0 <= sl <= 1.0:
            parser.error(f"Severity level must be between 0.0 and 1.0, got {sl}")
    
    # Choose visualization function based on layout
    if args.layout == 'compact':
        visualize_augmentations_compact(
            image_path=args.image_path,
            output_path=args.output,
            severity_levels=args.severity,
            dpi=args.dpi,
        )
    else:
        visualize_augmentations(
            image_path=args.image_path,
            output_path=args.output,
            severity_levels=args.severity,
            dpi=args.dpi,
        )


if __name__ == '__main__':
    main()
