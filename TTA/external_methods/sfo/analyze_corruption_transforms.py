#!/usr/bin/env python3
"""Analyze predicted augmentation transforms for ImageNet-C corruptions.

This script analyzes what augmentation transforms the trained Aug classifier
predicts for each corruption type in ImageNet-C.

Goals:
1. For each corruption, analyze the distribution of predicted transforms
2. Check if similar transforms are predicted for the same corruption
3. Analyze variance across different images and classes

Usage:
    python analyze_corruption_transforms.py \
        --fsc-path ./FSC/resnet50_FSC_stem.pth \
        --checkpoint ./output/STEM_KLDIV_ORTHOGONAL/best.pth.tar \
        --corruption-dir /home/oem/jin/datasets/imagenet-c \
        --severity 5

Output:
    Per-corruption transform distribution statistics
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from timm.data import create_dataset, create_loader, resolve_data_config
from timm.models import create_model
from timm.utils import setup_default_logging

# Import transform names
from timm.data import get_augmix_sl_transform_names, AUGMIX_SL_V2_NUM_TRANSFORMS

_logger = logging.getLogger("analyze_corruption_transforms")

# Default corruptions
DEFAULT_CORRUPTIONS = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze predicted transforms for ImageNet-C corruptions"
    )
    
    # Required paths
    parser.add_argument(
        "--fsc-path",
        type=str,
        default="./FSC/resnet50_FSC_stem.pth",
        help="Path to FSC file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./output/STEM_KLDIV_ORTHOGONAL/best.pth.tar",
        help="Path to trained Aug classifier checkpoint",
    )
    parser.add_argument(
        "--corruption-dir",
        type=str,
        default="/home/oem/jin/datasets/imagenet-c",
        help="Path to ImageNet-C root directory",
    )
    
    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="resnet50",
        help="Model architecture",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=True,
        help="Use pretrained model",
    )
    
    # Data options
    parser.add_argument(
        "--severity",
        type=int,
        default=5,
        help="Corruption severity level (1-5)",
    )
    parser.add_argument(
        "--corruptions",
        nargs="+",
        default=None,
        help="List of corruptions to analyze (default: all)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples per corruption (for quick testing)",
    )
    
    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use",
    )
    
    # Aug classifier options
    parser.add_argument(
        "--aug-classifier-hidden",
        nargs="+",
        type=int,
        default=[512, 256],
        help="Hidden dimensions for Aug classifier",
    )
    parser.add_argument(
        "--fsc-diff-mode",
        type=str,
        default="orthogonal",
        choices=["subtract", "orthogonal"],
        help="FSC difference computation mode",
    )
    parser.add_argument(
        "--sl-loss-type",
        type=str,
        default="kldiv",
        choices=["bce", "kldiv", "ce"],
        help="Loss type used during training (determines output activation)",
    )
    
    return parser.parse_args()


class StemFeatureExtractor(nn.Module):
    """Extract features from the first conv layer (stem) of a ResNet model."""
    
    def __init__(self, model):
        super().__init__()
        if hasattr(model, 'conv1'):
            self.conv1 = model.conv1
        else:
            raise ValueError("Model does not have conv1 layer")
        
        if hasattr(model, 'bn1'):
            self.bn1 = model.bn1
        else:
            self.bn1 = nn.Identity()
        
        if hasattr(model, 'act1'):
            self.act1 = model.act1
        else:
            self.act1 = nn.ReLU(inplace=True)
        
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        
    def forward(self, x):
        x = x.contiguous()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pool(x)
        x = x.flatten(1)
        return x


class AugClassifier(nn.Module):
    """Classifier that predicts augmentation types from FSC_diff."""
    
    def __init__(
        self,
        feature_dim: int,
        num_transforms: int = 7,
        hidden_dims: list = None,
        dropout: float = 0.1,
        use_sigmoid: bool = False,
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256]
        
        self.feature_dim = feature_dim
        self.num_transforms = num_transforms
        self.use_sigmoid = use_sigmoid
        
        layers = []
        in_dim = feature_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, num_transforms))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.mlp(x)
        if self.use_sigmoid:
            x = torch.sigmoid(x)
        return x


def compute_fsc_diff(features: torch.Tensor, fsc: torch.Tensor, mode: str = 'orthogonal') -> torch.Tensor:
    """Compute the difference between features and FSC centroids."""
    if mode == 'subtract':
        return features - fsc
    
    elif mode == 'orthogonal':
        diff = features - fsc
        fsc_norm_sq = (fsc * fsc).sum(dim=1, keepdim=True) + 1e-8
        proj_coeff = (diff * fsc).sum(dim=1, keepdim=True) / fsc_norm_sq
        diff_parallel = proj_coeff * fsc
        diff_orthogonal = diff - diff_parallel
        return diff_orthogonal
    
    else:
        raise ValueError(f"Unknown fsc_diff_mode: {mode}")


def analyze_corruption(
    corruption_name: str,
    data_path: Path,
    stem_extractor: nn.Module,
    aug_classifier: nn.Module,
    fsc_centroids: torch.Tensor,
    backbone: nn.Module,
    data_config: dict,
    args,
    device: torch.device,
    transform_names: List[str],
) -> Dict:
    """Analyze predicted transforms for a single corruption."""
    
    # Create dataset
    dataset = create_dataset(
        root=str(data_path),
        name="",
        split="validation",
    )
    
    if args.max_samples and len(dataset) > args.max_samples:
        indices = torch.randperm(len(dataset))[:args.max_samples].tolist()
        dataset = torch.utils.data.Subset(dataset, indices)
    
    loader = create_loader(
        dataset,
        input_size=data_config["input_size"],
        batch_size=args.batch_size,
        is_training=False,
        no_aug=True,
        use_prefetcher=True,
        interpolation=data_config["interpolation"],
        mean=data_config["mean"],
        std=data_config["std"],
        num_workers=args.workers,
        crop_pct=data_config["crop_pct"],
        crop_mode=data_config["crop_mode"],
        pin_memory=True,
        device=device,
    )
    
    # Collect predictions
    all_preds = []
    all_preds_prob = []  # Renamed: probability output (sigmoid or softmax)
    all_labels = []
    all_pred_classes = []
    
    stem_extractor.eval()
    aug_classifier.eval()
    backbone.eval()
    
    # Determine output activation based on loss type
    use_softmax = args.sl_loss_type == 'kldiv'
    
    first_batch = True
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"  {corruption_name}", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            
            # Get backbone predictions for FSC selection
            logits = backbone(images)
            pred_labels = logits.argmax(dim=1)
            
            # Extract stem features
            stem_features = stem_extractor(images).float()
            
            # Select FSC centroids based on predicted labels
            fsc_for_batch = fsc_centroids[pred_labels]
            
            # Compute FSC_diff
            fsc_diff = compute_fsc_diff(stem_features, fsc_for_batch, mode=args.fsc_diff_mode)
            
            # Get Aug classifier predictions
            aug_pred = aug_classifier(fsc_diff)
            
            # Apply appropriate activation based on loss type
            if use_softmax:
                aug_pred_prob = torch.softmax(aug_pred, dim=1)
            else:
                aug_pred_prob = torch.sigmoid(aug_pred)
            
            # Print first batch probabilities for debugging
            if first_batch:
                print(f"\n  === First Mini-Batch Probabilities ({corruption_name}) ===")
                print(f"  Activation: {'softmax' if use_softmax else 'sigmoid'}")
                print(f"  Batch size: {aug_pred_prob.shape[0]}")
                print(f"  Transform names: {transform_names}")
                print(f"  Raw logits (first 3 samples):")
                for i in range(min(3, aug_pred.shape[0])):
                    logits_str = ', '.join([f'{v:.3f}' for v in aug_pred[i].cpu().tolist()])
                    print(f"    Sample {i}: [{logits_str}]")
                print(f"  Probabilities (first 3 samples):")
                for i in range(min(3, aug_pred_prob.shape[0])):
                    prob_str = ', '.join([f'{v:.3f}' for v in aug_pred_prob[i].cpu().tolist()])
                    prob_sum = aug_pred_prob[i].sum().item()
                    print(f"    Sample {i}: [{prob_str}] (sum={prob_sum:.3f})")
                print()
                first_batch = False
            
            all_preds.append(aug_pred.cpu())
            all_preds_prob.append(aug_pred_prob.cpu())
            all_labels.append(labels.cpu())
            all_pred_classes.append(pred_labels.cpu())
    
    # Concatenate all predictions
    all_preds = torch.cat(all_preds, dim=0)
    all_preds_prob = torch.cat(all_preds_prob, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_pred_classes = torch.cat(all_pred_classes, dim=0)
    
    # Analyze predictions
    num_samples = all_preds.shape[0]
    num_transforms = all_preds.shape[1]
    
    # Per-transform statistics (using probability outputs)
    transform_stats = {}
    for t_idx, t_name in enumerate(transform_names):
        t_preds = all_preds_prob[:, t_idx]
        transform_stats[t_name] = {
            'mean': t_preds.mean().item(),
            'std': t_preds.std().item(),
            'min': t_preds.min().item(),
            'max': t_preds.max().item(),
            'median': t_preds.median().item(),
        }
    
    # Find dominant transform (highest mean prediction)
    mean_preds = all_preds_prob.mean(dim=0)
    dominant_idx = mean_preds.argmax().item()
    dominant_transform = transform_names[dominant_idx]
    
    # Per-sample dominant transform distribution
    sample_dominant = all_preds_prob.argmax(dim=1)
    dominant_counts = torch.bincount(sample_dominant, minlength=num_transforms)
    dominant_distribution = {
        transform_names[i]: dominant_counts[i].item() / num_samples * 100
        for i in range(num_transforms)
    }
    
    # Entropy of predictions (measure of uncertainty)
    # For KL div trained model, all_preds_prob is already softmax
    # For BCE trained model, compute softmax for entropy
    if use_softmax:
        probs = all_preds_prob
    else:
        probs = F.softmax(all_preds, dim=1)
    entropy = -(probs * (probs + 1e-8).log()).sum(dim=1)
    mean_entropy = entropy.mean().item()
    
    return {
        'corruption': corruption_name,
        'num_samples': num_samples,
        'transform_stats': transform_stats,
        'dominant_transform': dominant_transform,
        'dominant_distribution': dominant_distribution,
        'mean_entropy': mean_entropy,
        'mean_predictions': mean_preds.tolist(),
    }


def main():
    setup_default_logging()
    args = parse_args()
    
    device = torch.device(args.device)
    _logger.info(f"Using device: {device}")
    
    # Get transform names
    transform_names = get_augmix_sl_transform_names(version=2)
    num_transforms = len(transform_names)
    _logger.info(f"Transform names ({num_transforms}): {transform_names}")
    
    # Load FSC
    _logger.info(f"Loading FSC from: {args.fsc_path}")
    fsc_data = torch.load(args.fsc_path, map_location='cpu')
    fsc_centroids = fsc_data['centroids'].to(device)
    feature_dim = fsc_data['feature_dim']
    _logger.info(f"FSC loaded: {fsc_centroids.shape}, feature_dim={feature_dim}")
    
    # Create backbone model
    _logger.info(f"Creating model: {args.model}")
    backbone = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=1000,
    )
    backbone = backbone.to(device)
    backbone.eval()
    
    # Create stem feature extractor
    stem_extractor = StemFeatureExtractor(backbone)
    stem_extractor = stem_extractor.to(device)
    stem_extractor.eval()
    
    # Load checkpoint first to get correct architecture
    _logger.info(f"Loading Aug classifier from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # Get hidden_dims and other args from checkpoint if available
    hidden_dims = args.aug_classifier_hidden
    dropout = 0.1
    if 'args' in checkpoint:
        ckpt_args = checkpoint['args']
        if hasattr(ckpt_args, 'aug_classifier_hidden'):
            hidden_dims = ckpt_args.aug_classifier_hidden
            _logger.info(f"Using hidden_dims from checkpoint: {hidden_dims}")
        if hasattr(ckpt_args, 'aug_classifier_dropout'):
            dropout = ckpt_args.aug_classifier_dropout
        if hasattr(ckpt_args, 'fsc_diff_mode'):
            args.fsc_diff_mode = ckpt_args.fsc_diff_mode
            _logger.info(f"Using fsc_diff_mode from checkpoint: {args.fsc_diff_mode}")
        if hasattr(ckpt_args, 'sl_loss_type'):
            args.sl_loss_type = ckpt_args.sl_loss_type
            _logger.info(f"Using sl_loss_type from checkpoint: {args.sl_loss_type}")
    
    # Create Aug classifier with correct architecture
    aug_classifier = AugClassifier(
        feature_dim=feature_dim,
        num_transforms=num_transforms,
        hidden_dims=hidden_dims,
        dropout=dropout,
        use_sigmoid=False,
    )
    
    aug_classifier.load_state_dict(checkpoint['aug_classifier'])
    aug_classifier = aug_classifier.to(device)
    aug_classifier.eval()
    _logger.info(f"Aug classifier loaded (epoch {checkpoint.get('epoch', 'N/A')})")
    
    # Resolve data config
    data_config = resolve_data_config(vars(args), model=backbone, verbose=False)
    
    # Get corruptions to analyze
    corruptions = args.corruptions or DEFAULT_CORRUPTIONS
    corruption_dir = Path(args.corruption_dir)
    
    _logger.info(f"Analyzing {len(corruptions)} corruptions at severity {args.severity}")
    
    # Analyze each corruption
    results = []
    for corruption in corruptions:
        data_path = corruption_dir / corruption / str(args.severity)
        
        if not data_path.exists():
            _logger.warning(f"Skipping {corruption}: path not found ({data_path})")
            continue
        
        result = analyze_corruption(
            corruption_name=corruption,
            data_path=data_path,
            stem_extractor=stem_extractor,
            aug_classifier=aug_classifier,
            fsc_centroids=fsc_centroids,
            backbone=backbone,
            data_config=data_config,
            args=args,
            device=device,
            transform_names=transform_names,
        )
        results.append(result)
    
    # Print results
    print("\n" + "=" * 100)
    print("CORRUPTION TRANSFORM ANALYSIS RESULTS")
    print("=" * 100)
    print(f"FSC diff mode: {args.fsc_diff_mode}")
    print(f"Severity level: {args.severity}")
    print(f"Loss type: {args.sl_loss_type} (output activation: {'softmax' if args.sl_loss_type == 'kldiv' else 'sigmoid'})")
    print("=" * 100)
    
    # Summary table header
    header = f"{'Corruption':<20}"
    for t_name in transform_names:
        short_name = t_name.replace('Increasing', '')[:8]
        header += f"{short_name:>10}"
    header += f"{'Dominant':>15} {'Entropy':>8}"
    print(header)
    print("-" * 100)
    
    # Print each corruption result
    for result in results:
        row = f"{result['corruption']:<20}"
        for t_name in transform_names:
            mean_val = result['transform_stats'][t_name]['mean']
            row += f"{mean_val:>10.3f}"
        row += f"{result['dominant_transform'].replace('Increasing', ''):>15}"
        row += f"{result['mean_entropy']:>8.3f}"
        print(row)
    
    print("=" * 100)
    
    # Detailed per-corruption analysis
    print("\n" + "=" * 100)
    print("DETAILED ANALYSIS: Per-sample Dominant Transform Distribution (%)")
    print("=" * 100)
    
    header = f"{'Corruption':<20}"
    for t_name in transform_names:
        short_name = t_name.replace('Increasing', '')[:8]
        header += f"{short_name:>10}"
    print(header)
    print("-" * 100)
    
    for result in results:
        row = f"{result['corruption']:<20}"
        for t_name in transform_names:
            pct = result['dominant_distribution'].get(t_name, 0)
            row += f"{pct:>10.1f}"
        print(row)
    
    print("=" * 100)
    
    # Variance analysis
    print("\n" + "=" * 100)
    print("VARIANCE ANALYSIS: Standard Deviation of Transform Predictions")
    print("=" * 100)
    
    header = f"{'Corruption':<20}"
    for t_name in transform_names:
        short_name = t_name.replace('Increasing', '')[:8]
        header += f"{short_name:>10}"
    print(header)
    print("-" * 100)
    
    for result in results:
        row = f"{result['corruption']:<20}"
        for t_name in transform_names:
            std_val = result['transform_stats'][t_name]['std']
            row += f"{std_val:>10.3f}"
        print(row)
    
    print("=" * 100)
    
    # Summary statistics
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    
    # Check if corruptions map to specific transforms
    print("\nCorruption → Dominant Transform Mapping:")
    for result in results:
        dominant = result['dominant_transform'].replace('Increasing', '')
        dominant_pct = max(result['dominant_distribution'].values())
        print(f"  {result['corruption']:<20} → {dominant:<15} ({dominant_pct:.1f}% of samples)")
    
    # Overall entropy statistics
    entropies = [r['mean_entropy'] for r in results]
    print(f"\nMean Entropy across all corruptions: {np.mean(entropies):.3f} ± {np.std(entropies):.3f}")
    print("(Higher entropy = more uncertain/diverse predictions)")


if __name__ == "__main__":
    main()
