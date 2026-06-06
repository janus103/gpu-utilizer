#!/usr/bin/env python3
"""Summarize expert training results.

This script reads results.txt from each expert_{transform}_{method} directory
and creates a summary table.

Usage:
    python summarize_expert_results.py [--dirs DIR1 DIR2 ...] [--output OUTPUT_FILE]
    
    Default directories:
        /data/jin/new_weight
        /home/ubuntu/jin/SOBA/output
"""
import argparse
import os
import re
from collections import defaultdict


def parse_results_file(filepath):
    """Parse results.txt and extract epoch metrics."""
    epochs = []
    
    with open(filepath, 'r') as f:
        for line in f:
            # Match lines like: epoch=000, train_loss=..., val_top1=57.30, ...
            match = re.search(
                r'epoch=(\d+).*val_top1=([0-9.]+)',
                line
            )
            if match:
                epoch = int(match.group(1))
                val_top1 = float(match.group(2))
                epochs.append((epoch, val_top1))
    
    return epochs


def analyze_results(epochs):
    """Analyze parsed results and return summary statistics."""
    if not epochs:
        return None
    
    # Sort by epoch number
    epochs_sorted = sorted(epochs, key=lambda x: x[0])
    
    # Last epoch
    last_epoch, last_val_top1 = epochs_sorted[-1]
    
    # Top 3 by val_top1
    top3 = sorted(epochs, key=lambda x: x[1], reverse=True)[:3]
    
    # Last 10 epochs average
    last_10 = epochs_sorted[-10:] if len(epochs_sorted) >= 10 else epochs_sorted
    last_10_avg = sum(v for _, v in last_10) / len(last_10)
    
    return {
        'last_epoch': last_epoch,
        'last_val_top1': last_val_top1,
        'top3': top3,
        'last_10_avg': last_10_avg,
        'total_epochs': len(epochs_sorted),
    }


def main():
    parser = argparse.ArgumentParser(description='Summarize expert training results')
    parser.add_argument(
        '--dirs', nargs='+',
        default=['/data/jin/new_weight', '/home/ubuntu/jin/SOBA/output'],
        help='Directories to scan for expert_* folders'
    )
    parser.add_argument(
        '--output', '-o',
        default='/data/jin/new_weight/result_summary.txt',
        help='Output file path'
    )
    args = parser.parse_args()
    
    output_file = args.output
    
    # Find all expert directories from all base directories
    results = defaultdict(dict)  # results[transform][method] = analysis
    source_paths = {}  # Track where each result came from
    
    for base_dir in args.dirs:
        if not os.path.exists(base_dir):
            print(f'Warning: {base_dir} does not exist, skipping')
            continue
        
        print(f'\nScanning: {base_dir}')
        
        for dirname in sorted(os.listdir(base_dir)):
            if not dirname.startswith('expert_'):
                continue
            
            # Parse directory name: expert_{transform}_{method}
            parts = dirname.split('_')
            if len(parts) < 3:
                continue
            
            # Handle cases like "expert_GaussianBlurIncreasing_stem-all-bn"
            transform = parts[1]
            method = '_'.join(parts[2:])
            
            results_path = os.path.join(base_dir, dirname, 'results.txt')
            if not os.path.exists(results_path):
                print(f'  Warning: {results_path} not found')
                continue
            
            epochs = parse_results_file(results_path)
            analysis = analyze_results(epochs)
            
            if analysis:
                # Check if we already have this experiment from another directory
                if method in results[transform]:
                    existing = results[transform][method]
                    # Keep the one with more epochs or higher accuracy
                    if analysis['total_epochs'] > existing['total_epochs']:
                        results[transform][method] = analysis
                        source_paths[(transform, method)] = base_dir
                        print(f'  Updated: {dirname} ({analysis["total_epochs"]} epochs) - replaced older version')
                    else:
                        print(f'  Skipped: {dirname} ({analysis["total_epochs"]} epochs) - keeping version with {existing["total_epochs"]} epochs')
                else:
                    results[transform][method] = analysis
                    source_paths[(transform, method)] = base_dir
                    print(f'  Parsed: {dirname} ({analysis["total_epochs"]} epochs)')
    
    # Generate summary table
    transforms = sorted(results.keys())
    methods = sorted(set(m for t in results.values() for m in t.keys()))
    
    with open(output_file, 'w') as f:
        f.write('=' * 120 + '\n')
        f.write('Expert Training Results Summary\n')
        f.write('=' * 120 + '\n')
        f.write(f'Scanned directories:\n')
        for d in args.dirs:
            f.write(f'  - {d}\n')
        f.write('\n')
        
        # Table 1: Last Epoch val_top1
        f.write('-' * 80 + '\n')
        f.write('Table 1: Last Epoch val_top1 (%)\n')
        f.write('-' * 80 + '\n')
        
        # Header
        header = f'{"Transform":<30}'
        for method in methods:
            header += f'{method:>15}'
        f.write(header + '\n')
        f.write('-' * 80 + '\n')
        
        for transform in transforms:
            row = f'{transform:<30}'
            for method in methods:
                if method in results[transform]:
                    val = results[transform][method]['last_val_top1']
                    row += f'{val:>15.2f}'
                else:
                    row += f'{"N/A":>15}'
            f.write(row + '\n')
        f.write('\n')
        
        # Table 2: Top 3 Epochs by val_top1
        f.write('-' * 120 + '\n')
        f.write('Table 2: Top 3 Epochs by val_top1 (epoch: val_top1%)\n')
        f.write('-' * 120 + '\n')
        
        header = f'{"Transform":<30}'
        for method in methods:
            header += f'{method:>30}'
        f.write(header + '\n')
        f.write('-' * 120 + '\n')
        
        for transform in transforms:
            row = f'{transform:<30}'
            for method in methods:
                if method in results[transform]:
                    top3 = results[transform][method]['top3']
                    top3_str = ', '.join([f'{ep}:{val:.1f}' for ep, val in top3])
                    row += f'{top3_str:>30}'
                else:
                    row += f'{"N/A":>30}'
            f.write(row + '\n')
        f.write('\n')
        
        # Table 3: Last 10 Epochs Average
        f.write('-' * 80 + '\n')
        f.write('Table 3: Last 10 Epochs Average val_top1 (%)\n')
        f.write('-' * 80 + '\n')
        
        header = f'{"Transform":<30}'
        for method in methods:
            header += f'{method:>15}'
        f.write(header + '\n')
        f.write('-' * 80 + '\n')
        
        for transform in transforms:
            row = f'{transform:<30}'
            for method in methods:
                if method in results[transform]:
                    val = results[transform][method]['last_10_avg']
                    row += f'{val:>15.2f}'
                else:
                    row += f'{"N/A":>15}'
            f.write(row + '\n')
        f.write('\n')
        
        # Table 4: Best val_top1 (excluding warmup epochs 0-4)
        f.write('-' * 80 + '\n')
        f.write('Table 4: Best val_top1 (%) - excluding warmup (epoch < 5)\n')
        f.write('-' * 80 + '\n')
        
        header = f'{"Transform":<30}'
        for method in methods:
            header += f'{method:>15}'
        f.write(header + '\n')
        f.write('-' * 80 + '\n')
        
        for transform in transforms:
            row = f'{transform:<30}'
            for method in methods:
                if method in results[transform]:
                    top3 = results[transform][method]['top3']
                    # Filter out warmup epochs
                    non_warmup = [(ep, val) for ep, val in top3 if ep >= 5]
                    if non_warmup:
                        best_ep, best_val = non_warmup[0]
                        row += f'{best_val:>15.2f}'
                    else:
                        # All top 3 were in warmup, find best from all epochs
                        row += f'{"(warmup)":>15}'
                else:
                    row += f'{"N/A":>15}'
            f.write(row + '\n')
        f.write('\n')
        
        # Detailed per-experiment summary
        f.write('=' * 120 + '\n')
        f.write('Detailed Summary per Experiment\n')
        f.write('=' * 120 + '\n\n')
        
        for transform in transforms:
            for method in methods:
                if method not in results[transform]:
                    continue
                
                r = results[transform][method]
                src = source_paths.get((transform, method), 'unknown')
                f.write(f'>>> {transform} / {method}\n')
                f.write(f'    Source: {src}\n')
                f.write(f'    Total Epochs: {r["total_epochs"]}\n')
                f.write(f'    Last Epoch: {r["last_epoch"]} (val_top1={r["last_val_top1"]:.2f}%)\n')
                f.write(f'    Top 3: ')
                for i, (ep, val) in enumerate(r['top3']):
                    warmup_note = ' [warmup]' if ep < 5 else ''
                    f.write(f'#{i+1} ep{ep}={val:.2f}%{warmup_note}  ')
                f.write('\n')
                f.write(f'    Last 10 Avg: {r["last_10_avg"]:.2f}%\n')
                f.write('\n')
    
    print(f'\nSummary saved to: {output_file}')


if __name__ == '__main__':
    main()
