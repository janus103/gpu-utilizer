#!/usr/bin/env python3
"""Evaluate AugClassifier on ImageNet-C: Predicted Augmentation Distribution per Corruption

Loads a trained AugClassifier weight (from run_aug_cls_fedavg.sh) and runs
inference on each of the 15 ImageNet-C corruptions. For each corruption,
collects the softmax probability distribution over the 8 augmentation types,
then generates:
  1. A CSV with per-corruption mean probability for each aug type
  2. A heatmap visualization (corruptions × aug types)
  3. Optionally, per-corruption histogram plots
"""
import argparse
import csv
import os
import logging

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from train_aug_classifier import AugClassifier, NUM_AUG_TRANSFORMS
from train_aug_classifier_resnet_fedavg import AugClassifierResNet
from timm.data.auto_augment import _AUGMIX_SL_TRANSFORMS_V2

_logger = logging.getLogger('eval_aug_cls_imagenet_c')

CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
    'snow', 'frost', 'fog', 'brightness',
    'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
]

AUG_NAMES = list(_AUGMIX_SL_TRANSFORMS_V2)

_DISPLAY_NAMES = {
    'PositiveIntensity': 'PosIntensity',
    'NegativeIntensity': 'NegIntensity',
    'SaturationV2': 'Saturation',
    'SharpnessV2': 'Sharpness',
    'GaussianBlurIncreasing': 'GaussBlur',
    'PosterizeIncreasing': 'Posterize',
    'SolarizeIncreasing': 'Solarize',
    'SaltAndPepperIncreasing': 'S&P',
}

DISPLAY_NAMES = [_DISPLAY_NAMES.get(n, n) for n in AUG_NAMES]


def create_corruption_loader(corruption_dir, img_size=224, batch_size=256,
                             num_workers=8, mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    dataset = ImageFolder(corruption_dir, transform=transform)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )
    return loader


@torch.no_grad()
def evaluate_corruption(model, loader, device):
    """Run inference and collect softmax probabilities + argmax predictions."""
    model.eval()
    all_probs = []
    all_preds = []

    for images, _ in loader:
        images = images.to(device=device, dtype=torch.float32)
        logits = model(images)
        probs = F.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)

        all_probs.append(probs.cpu().numpy())
        all_preds.append(preds.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    return all_probs, all_preds


def make_heatmap(results_dict, output_path, title='AugClassifier Predictions on ImageNet-C'):
    """Generate heatmap: rows=corruptions, cols=aug types."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    corruptions = list(results_dict.keys())
    num_c = len(corruptions)
    num_a = len(DISPLAY_NAMES)

    matrix = np.zeros((num_c, num_a))
    for i, corr in enumerate(corruptions):
        matrix[i] = results_dict[corr]['mean_prob']

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=matrix.max())

    ax.set_xticks(range(num_a))
    ax.set_xticklabels(DISPLAY_NAMES, rotation=45, ha='right', fontsize=10)
    ax.set_yticks(range(num_c))
    ax.set_yticklabels(corruptions, fontsize=10)

    for i in range(num_c):
        for j in range(num_a):
            val = matrix[i, j]
            color = 'white' if val > matrix.max() * 0.6 else 'black'
            ax.text(j, i, f'{val:.1f}%', ha='center', va='center',
                    fontsize=7, color=color)

    plt.colorbar(im, ax=ax, label='Mean Probability (%)', shrink=0.8)
    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xlabel('Predicted Augmentation Type', fontsize=11)
    ax.set_ylabel('ImageNet-C Corruption', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    _logger.info(f'Heatmap saved: {output_path}')


def make_bar_chart(results_dict, output_path):
    """Stacked bar chart: each corruption shows its predicted aug distribution."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    corruptions = list(results_dict.keys())
    num_c = len(corruptions)
    num_a = len(DISPLAY_NAMES)

    matrix = np.zeros((num_c, num_a))
    for i, corr in enumerate(corruptions):
        matrix[i] = results_dict[corr]['mean_prob']

    fig, ax = plt.subplots(figsize=(16, 8))

    x = np.arange(num_c)
    bar_width = 0.7
    cmap = plt.cm.Set3(np.linspace(0, 1, num_a))

    bottom = np.zeros(num_c)
    for j in range(num_a):
        ax.bar(x, matrix[:, j], bar_width, bottom=bottom,
               label=DISPLAY_NAMES[j], color=cmap[j])
        bottom += matrix[:, j]

    ax.set_xticks(x)
    ax.set_xticklabels(corruptions, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Probability (%)', fontsize=11)
    ax.set_title('AugClassifier Predicted Distribution per ImageNet-C Corruption', fontsize=13)
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    _logger.info(f'Bar chart saved: {output_path}')


def make_argmax_chart(results_dict, output_path):
    """Bar chart showing argmax (top-1 predicted aug) distribution per corruption."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    corruptions = list(results_dict.keys())
    num_c = len(corruptions)
    num_a = len(DISPLAY_NAMES)

    fig, axes = plt.subplots(3, 5, figsize=(22, 12))
    axes = axes.flatten()
    cmap = plt.cm.Set3(np.linspace(0, 1, num_a))

    for i, corr in enumerate(corruptions):
        ax = axes[i]
        pred_counts = results_dict[corr]['pred_counts']
        pred_pct = pred_counts / pred_counts.sum() * 100.0

        bars = ax.bar(range(num_a), pred_pct, color=cmap)
        ax.set_title(corr.replace('_', ' ').title(), fontsize=10, pad=3)
        ax.set_xticks(range(num_a))
        ax.set_xticklabels(DISPLAY_NAMES, rotation=60, ha='right', fontsize=6)
        ax.set_ylim(0, max(pred_pct.max() * 1.2, 20))
        ax.set_ylabel('%', fontsize=8)

        top_idx = pred_pct.argmax()
        bars[top_idx].set_edgecolor('red')
        bars[top_idx].set_linewidth(2)

    fig.suptitle('Argmax (Top-1) Predicted Aug Distribution per Corruption',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    _logger.info(f'Argmax chart saved: {output_path}')


def make_similarity_analysis(results_dict, output_path):
    """Compute and visualize pairwise cosine similarity between corruption distributions."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    corruptions = list(results_dict.keys())
    num_c = len(corruptions)

    matrix = np.zeros((num_c, len(DISPLAY_NAMES)))
    for i, corr in enumerate(corruptions):
        matrix[i] = results_dict[corr]['mean_prob']

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normed = matrix / norms
    sim = normed @ normed.T

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(sim, cmap='RdYlGn', vmin=0, vmax=1)

    ax.set_xticks(range(num_c))
    ax.set_xticklabels(corruptions, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(num_c))
    ax.set_yticklabels(corruptions, fontsize=9)

    for i in range(num_c):
        for j in range(num_c):
            color = 'white' if sim[i, j] < 0.4 else 'black'
            ax.text(j, i, f'{sim[i, j]:.2f}', ha='center', va='center',
                    fontsize=6, color=color)

    plt.colorbar(im, ax=ax, label='Cosine Similarity', shrink=0.8)
    ax.set_title('Pairwise Cosine Similarity of Predicted Aug Distributions', fontsize=13)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    _logger.info(f'Similarity matrix saved: {output_path}')


def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    parser = argparse.ArgumentParser(
        description='Evaluate AugClassifier on ImageNet-C corruptions')

    parser.add_argument('--checkpoint', type=str,
                        default='./output/aug_cls_joint/best.pth',
                        help='Path to AugClassifier checkpoint')
    parser.add_argument('--imagenet-c-root', type=str,
                        default='/home/oem/servers/imagenet-c',
                        help='Root directory of ImageNet-C')
    parser.add_argument('--severity', type=int, default=5,
                        help='Corruption severity level (1-5)')
    parser.add_argument('--model-type', type=str, default='vit',
                        choices=['vit', 'resnet'],
                        help='Model type: vit (AugClassifier, embed_dim=768) or '
                             'resnet (AugClassifierResNet, embed_dim=64)')
    parser.add_argument('--embed-dim', type=int, default=None,
                        help='Override embed_dim (auto-detected from checkpoint)')
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output-dir', type=str, default='./output/aug_cls_imagenet_c_eval')
    parser.add_argument('--max-samples', type=int, default=0,
                        help='Max samples per corruption (0 = all)')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # ── Load checkpoint ──
    _logger.info(f'Loading checkpoint: {args.checkpoint}')
    ckpt = torch.load(args.checkpoint, map_location='cpu')

    default_embed_dim = 64 if args.model_type == 'resnet' else 768
    embed_dim = args.embed_dim or ckpt.get('embed_dim', default_embed_dim)
    num_transforms = ckpt.get('num_transforms', NUM_AUG_TRANSFORMS)
    aug_names = ckpt.get('aug_names', AUG_NAMES)

    _logger.info(f'  model_type={args.model_type}, embed_dim={embed_dim}, num_transforms={num_transforms}')
    _logger.info(f'  aug_names={aug_names}')
    _logger.info(f'  best_avg_acc={ckpt.get("best_avg_acc", "N/A")}')

    # ── Create model ──
    if args.model_type == 'resnet':
        model = AugClassifierResNet(
            in_chans=3,
            num_classes=num_transforms,
            embed_dim=embed_dim,
        )
    else:
        model = AugClassifier(
            in_chans=3,
            num_classes=num_transforms,
            embed_dim=embed_dim,
        )

    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        _logger.warning(f'  Missing keys: {missing}')
    if unexpected:
        _logger.warning(f'  Unexpected keys: {unexpected}')

    model = model.to(device)
    model.eval()
    _logger.info(f'Model loaded successfully ({sum(p.numel() for p in model.parameters()):,} params)')

    # ── Evaluate each corruption ──
    results = {}
    all_rows = []

    _logger.info(f'\n{"="*70}')
    _logger.info(f'Evaluating on ImageNet-C (severity={args.severity})')
    _logger.info(f'{"="*70}\n')

    for ci, corruption in enumerate(CORRUPTIONS):
        corruption_dir = os.path.join(args.imagenet_c_root, corruption, str(args.severity))
        if not os.path.isdir(corruption_dir):
            _logger.warning(f'  [{ci+1:2d}/15] {corruption}: directory not found at {corruption_dir}, skipping')
            continue

        loader = create_corruption_loader(
            corruption_dir,
            img_size=args.img_size,
            batch_size=args.batch_size,
            num_workers=args.workers,
        )

        if args.max_samples > 0:
            total_samples = min(args.max_samples, len(loader.dataset))
        else:
            total_samples = len(loader.dataset)

        _logger.info(f'  [{ci+1:2d}/15] {corruption:<22s} ({total_samples} images) ...', )

        all_probs, all_preds = evaluate_corruption(model, loader, device)

        if args.max_samples > 0:
            all_probs = all_probs[:args.max_samples]
            all_preds = all_preds[:args.max_samples]

        mean_prob = all_probs.mean(axis=0) * 100.0
        std_prob = all_probs.std(axis=0) * 100.0

        pred_counts = np.zeros(num_transforms)
        for p in all_preds:
            pred_counts[p] += 1

        top1_aug = DISPLAY_NAMES[mean_prob.argmax()]
        entropy = -np.sum(all_probs.mean(axis=0) * np.log(all_probs.mean(axis=0) + 1e-10))

        results[corruption] = {
            'mean_prob': mean_prob,
            'std_prob': std_prob,
            'pred_counts': pred_counts,
            'entropy': entropy,
            'num_samples': len(all_probs),
        }

        prob_str = ' | '.join(f'{DISPLAY_NAMES[j]}:{mean_prob[j]:.1f}%' for j in range(num_transforms))
        _logger.info(f'           Top-1: {top1_aug}  Entropy: {entropy:.3f}')
        _logger.info(f'           {prob_str}')

        row = {
            'corruption': corruption,
            'num_samples': len(all_probs),
            'top1_aug': top1_aug,
            'entropy': f'{entropy:.4f}',
        }
        for j in range(num_transforms):
            row[f'prob_{DISPLAY_NAMES[j]}'] = f'{mean_prob[j]:.2f}'
            row[f'std_{DISPLAY_NAMES[j]}'] = f'{std_prob[j]:.2f}'
        for j in range(num_transforms):
            row[f'argmax_{DISPLAY_NAMES[j]}'] = f'{pred_counts[j]:.0f}'
        all_rows.append(row)

    # ── Save CSV ──
    csv_path = os.path.join(args.output_dir, 'aug_distribution_per_corruption.csv')
    if all_rows:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)
        _logger.info(f'\nCSV saved: {csv_path}')

    # ── Check similarity across corruptions ──
    _logger.info(f'\n{"="*70}')
    _logger.info('Cross-Corruption Similarity Analysis')
    _logger.info(f'{"="*70}')

    if len(results) >= 2:
        corr_names = list(results.keys())
        distributions = np.array([results[c]['mean_prob'] for c in corr_names])

        norms = np.linalg.norm(distributions, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normed = distributions / norms
        cos_sim = normed @ normed.T

        mean_sim = (cos_sim.sum() - np.trace(cos_sim)) / (len(corr_names) * (len(corr_names) - 1))
        min_sim = np.min(cos_sim[np.triu_indices_from(cos_sim, k=1)])
        max_sim = np.max(cos_sim[np.triu_indices_from(cos_sim, k=1)])

        _logger.info(f'  Mean pairwise cosine similarity: {mean_sim:.4f}')
        _logger.info(f'  Min: {min_sim:.4f}, Max: {max_sim:.4f}')

        global_mean = distributions.mean(axis=0)
        deviations = np.linalg.norm(distributions - global_mean, axis=1)
        _logger.info(f'\n  Per-corruption deviation from global mean distribution:')
        for i, c in enumerate(corr_names):
            _logger.info(f'    {c:<22s}  L2 deviation: {deviations[i]:.3f}')

        _logger.info(f'\n  Global mean distribution:')
        for j in range(num_transforms):
            _logger.info(f'    {DISPLAY_NAMES[j]:>12s}: {global_mean[j]:.2f}%')

    # ── Generate visualizations ──
    _logger.info(f'\n{"="*70}')
    _logger.info('Generating visualizations...')
    _logger.info(f'{"="*70}')

    model_label = 'ResNet' if args.model_type == 'resnet' else 'ViT'
    try:
        make_heatmap(results, os.path.join(args.output_dir, 'heatmap_aug_distribution.png'),
                     title=f'AugClassifier ({model_label}) Predictions on ImageNet-C')
        make_bar_chart(results, os.path.join(args.output_dir, 'stacked_bar_distribution.png'))
        make_argmax_chart(results, os.path.join(args.output_dir, 'argmax_per_corruption.png'))
        make_similarity_analysis(results, os.path.join(args.output_dir, 'corruption_similarity.png'))
    except Exception as e:
        _logger.warning(f'Visualization failed: {e}')
        _logger.info('CSV results are still saved successfully.')

    _logger.info(f'\nAll results saved to: {args.output_dir}/')


if __name__ == '__main__':
    main()
