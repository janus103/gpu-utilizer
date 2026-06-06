#!/usr/bin/env python3
"""Inspect FSC (Feature Space Centroid) stem file contents.

This script displays the centroid vectors and statistics for all 1000 classes
stored in an FSC stem file.

Usage:
    python inspect_fsc_stem.py --fsc-path ./FSC/resnet50_FSC_stem.pth
    python inspect_fsc_stem.py --fsc-path ./FSC/resnet50_FSC_stem.pth --show-all
"""

import argparse
import torch
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect FSC stem file contents")
    parser.add_argument(
        "--fsc-path",
        type=str,
        default="./FSC/resnet50_FSC_stem.pth",
        help="Path to FSC file (default: ./FSC/resnet50_FSC_stem.pth)",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Show all 1000 classes (default: show first/last 10)",
    )
    parser.add_argument(
        "--num-values",
        type=int,
        default=5,
        help="Number of centroid values to display per class (default: 5)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load FSC
    print(f"Loading FSC from: {args.fsc_path}")
    fsc_data = torch.load(args.fsc_path, map_location='cpu')
    centroids = fsc_data['centroids']
    counts = fsc_data['counts']
    feature_dim = fsc_data['feature_dim']
    
    num_classes = centroids.shape[0]
    
    # Basic info
    print()
    print("=" * 70)
    print("FSC Stem File Information")
    print("=" * 70)
    print(f"Centroid shape: {centroids.shape}")
    print(f"Feature dimension: {feature_dim}")
    print(f"Number of classes: {num_classes}")
    print(f"Total correct samples used: {fsc_data.get('total_correct', 'N/A')}")
    print(f"Total samples: {fsc_data.get('total_samples', 'N/A')}")
    print(f"Model: {fsc_data.get('model_name', 'N/A')}")
    print(f"Feature source: {fsc_data.get('feature_source', 'N/A')}")
    print(f"Pooling: {fsc_data.get('pooling', 'N/A')}")
    print()
    
    # Centroid statistics
    print("=" * 70)
    print("Centroid Statistics")
    print("=" * 70)
    print(f"Mean value: {centroids.mean().item():.4f}")
    print(f"Std value: {centroids.std().item():.4f}")
    print(f"Min value: {centroids.min().item():.4f}")
    print(f"Max value: {centroids.max().item():.4f}")
    print()
    
    # L2 norms
    norms = centroids.norm(dim=1)
    print("L2 Norms:")
    print(f"  Mean: {norms.mean().item():.4f}")
    print(f"  Std: {norms.std().item():.4f}")
    print(f"  Min: {norms.min().item():.4f} (Class {norms.argmin().item()})")
    print(f"  Max: {norms.max().item():.4f} (Class {norms.argmax().item()})")
    print()
    
    # Pairwise cosine similarity
    print("=" * 70)
    print("Pairwise Cosine Similarity (between all class centroids)")
    print("=" * 70)
    centroids_norm = F.normalize(centroids, dim=1)
    cos_sim = torch.mm(centroids_norm, centroids_norm.t())
    mask = ~torch.eye(num_classes, dtype=bool)
    pairwise_cos = cos_sim[mask]
    
    print(f"Mean: {pairwise_cos.mean().item():.4f}")
    print(f"Std: {pairwise_cos.std().item():.4f}")
    print(f"Min: {pairwise_cos.min().item():.4f}")
    print(f"Max: {pairwise_cos.max().item():.4f}")
    print()
    
    high_sim_90 = (pairwise_cos > 0.9).sum().item()
    high_sim_95 = (pairwise_cos > 0.95).sum().item()
    high_sim_99 = (pairwise_cos > 0.99).sum().item()
    total_pairs = len(pairwise_cos)
    
    print(f"Pairs with cosine > 0.90: {high_sim_90:,} ({high_sim_90/total_pairs*100:.2f}%)")
    print(f"Pairs with cosine > 0.95: {high_sim_95:,} ({high_sim_95/total_pairs*100:.2f}%)")
    print(f"Pairs with cosine > 0.99: {high_sim_99:,} ({high_sim_99/total_pairs*100:.2f}%)")
    print()
    
    # Display centroids
    print("=" * 70)
    print(f"Class Centroids (showing first {args.num_values} values)")
    print("=" * 70)
    
    def print_class(idx):
        c = centroids[idx][:args.num_values].tolist()
        c_str = ', '.join([f'{v:.3f}' for v in c])
        print(f"Class {idx:4d}: count={counts[idx]:5d}, norm={norms[idx]:.2f}, centroid=[{c_str}, ...]")
    
    if args.show_all:
        for i in range(num_classes):
            print_class(i)
    else:
        print("First 10 classes:")
        print("-" * 70)
        for i in range(min(10, num_classes)):
            print_class(i)
        
        print()
        print("...")
        print()
        
        print("Last 10 classes:")
        print("-" * 70)
        for i in range(max(0, num_classes - 10), num_classes):
            print_class(i)
    
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Classes with valid centroids: {(counts > 0).sum().item()}/{num_classes}")
    print(f"Average samples per class: {counts.float().mean().item():.1f}")
    print(f"Min samples: {counts.min().item()} (Class {counts.argmin().item()})")
    print(f"Max samples: {counts.max().item()} (Class {counts.argmax().item()})")


if __name__ == "__main__":
    main()
