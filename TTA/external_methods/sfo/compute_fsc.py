#!/usr/bin/env python3
"""Compute Feature Space Centroid (FSC) for each class.

This script computes the mean feature vector (centroid) for each class using only
correctly classified samples from the training dataset. The pretrained model's 
2048-dimensional features are used to represent each image semantically.

Usage:
    python compute_fsc.py --data-dir /path/to/imagenet/train --model resnet50 --pretrained

Output:
    ./FSC/{model_name}_FSC.pth
"""

import argparse
import logging
from pathlib import Path

import torch
from tqdm import tqdm

from timm.data import create_dataset, create_loader, resolve_data_config
from timm.models import create_model, load_checkpoint
from timm.utils import setup_default_logging

_logger = logging.getLogger("compute_fsc")


def parse_args():
    parser = argparse.ArgumentParser(description="Compute Feature Space Centroid (FSC)")
    
    # Data arguments
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to training dataset (e.g., ImageNet train folder)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help='Dataset type + name ("<type>/<name>") (default: ImageFolder)',
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use (default: train)",
    )
    
    # Model arguments
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="resnet50",
        help="Model architecture (default: resnet50)",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Use pretrained model weights",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Path to model checkpoint (optional)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=1000,
        help="Number of classes (default: 1000 for ImageNet)",
    )
    
    # Processing arguments
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for inference (default: 256)",
    )
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=8,
        help="Number of data loading workers (default: 8)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use automatic mixed precision",
    )
    
    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./FSC",
        help="Output directory for FSC files (default: ./FSC)",
    )
    
    return parser.parse_args()


def extract_features_and_predictions(model, loader, device, amp=False):
    """Extract features and predictions from model.
    
    Args:
        model: The pretrained model.
        loader: DataLoader for the dataset.
        device: Device to use.
        amp: Whether to use automatic mixed precision.
    
    Returns:
        all_features: List of feature tensors [N, feature_dim].
        all_predictions: List of prediction tensors [N].
        all_targets: List of target tensors [N].
    """
    model.eval()
    
    all_features = []
    all_predictions = []
    all_targets = []
    
    # Get the classifier module (works for different model architectures)
    classifier = model.get_classifier() if hasattr(model, 'get_classifier') else model.fc
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(loader, desc="Extracting features")):
            images = images.to(device)
            targets = targets.to(device)
            
            with torch.cuda.amp.autocast(enabled=amp):
                # Extract features before the final classifier layer
                features = model.forward_features(images)
                features = model.forward_head(features, pre_logits=True)
                
                # Get predictions using the classifier
                logits = classifier(features)
                predictions = logits.argmax(dim=1)
            
            # Move to CPU to save GPU memory
            all_features.append(features.float().cpu())
            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())
    
    return (
        torch.cat(all_features, dim=0),
        torch.cat(all_predictions, dim=0),
        torch.cat(all_targets, dim=0),
    )


def compute_fsc(features, predictions, targets, num_classes):
    """Compute Feature Space Centroid for each class.
    
    Only uses correctly classified samples to compute the centroid.
    
    Args:
        features: Feature tensor [N, feature_dim].
        predictions: Prediction tensor [N].
        targets: Target tensor [N].
        num_classes: Number of classes.
    
    Returns:
        fsc: Dictionary containing:
            - 'centroids': Tensor [num_classes, feature_dim] - centroid for each class
            - 'counts': Tensor [num_classes] - number of correct samples per class
            - 'total_correct': Total number of correctly classified samples
            - 'total_samples': Total number of samples
    """
    feature_dim = features.shape[1]
    
    # Initialize accumulators
    class_feature_sum = torch.zeros(num_classes, feature_dim, dtype=torch.float64)
    class_counts = torch.zeros(num_classes, dtype=torch.long)
    
    # Find correctly classified samples
    correct_mask = (predictions == targets)
    correct_indices = correct_mask.nonzero(as_tuple=True)[0]
    
    _logger.info(f"Total samples: {len(targets)}")
    _logger.info(f"Correctly classified: {len(correct_indices)} ({100 * len(correct_indices) / len(targets):.2f}%)")
    
    # Accumulate features for each class (only correct predictions)
    for idx in tqdm(correct_indices, desc="Computing centroids"):
        cls = targets[idx].item()
        class_feature_sum[cls] += features[idx].double()
        class_counts[cls] += 1
    
    # Compute mean (centroid) for each class
    centroids = torch.zeros(num_classes, feature_dim, dtype=torch.float32)
    for cls in range(num_classes):
        if class_counts[cls] > 0:
            centroids[cls] = (class_feature_sum[cls] / class_counts[cls]).float()
        else:
            _logger.warning(f"Class {cls} has no correctly classified samples!")
    
    return {
        "centroids": centroids,
        "counts": class_counts,
        "total_correct": len(correct_indices),
        "total_samples": len(targets),
        "feature_dim": feature_dim,
    }


def main():
    setup_default_logging()
    args = parse_args()
    
    # Setup device
    device = torch.device(args.device)
    _logger.info(f"Using device: {device}")
    
    # Create model
    _logger.info(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
    )
    
    if args.checkpoint:
        _logger.info(f"Loading checkpoint: {args.checkpoint}")
        load_checkpoint(model, args.checkpoint)
    
    model = model.to(device)
    model.eval()
    
    param_count = sum(p.numel() for p in model.parameters())
    _logger.info(f"Model {args.model} created, param count: {param_count / 1e6:.2f}M")
    
    # Resolve data config from model
    data_config = resolve_data_config(vars(args), model=model, verbose=True)
    
    # Create dataset and loader
    _logger.info(f"Loading dataset from: {args.data_dir}")
    dataset = create_dataset(
        root=args.data_dir,
        name=args.dataset,
        split=args.split,
    )
    _logger.info(f"Dataset size: {len(dataset)}")
    
    loader = create_loader(
        dataset,
        input_size=data_config["input_size"],
        batch_size=args.batch_size,
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
    
    # Extract features and predictions
    _logger.info("Extracting features from model...")
    features, predictions, targets = extract_features_and_predictions(
        model, loader, device, amp=args.amp
    )
    
    # Compute FSC
    _logger.info("Computing Feature Space Centroids...")
    fsc = compute_fsc(features, predictions, targets, args.num_classes)
    
    # Add metadata
    fsc["model_name"] = args.model
    fsc["checkpoint"] = args.checkpoint if args.checkpoint else "pretrained"
    fsc["data_dir"] = args.data_dir
    
    # Create output directory and save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"{args.model}_FSC.pth"
    torch.save(fsc, output_path)
    
    _logger.info(f"FSC saved to: {output_path}")
    _logger.info(f"Centroid shape: {fsc['centroids'].shape}")
    _logger.info(f"Classes with valid centroids: {(fsc['counts'] > 0).sum().item()}/{args.num_classes}")
    
    # Print per-class statistics
    _logger.info("\nPer-class statistics (first 10 classes):")
    for cls in range(min(10, args.num_classes)):
        _logger.info(f"  Class {cls}: {fsc['counts'][cls].item()} correct samples")


if __name__ == "__main__":
    main()
