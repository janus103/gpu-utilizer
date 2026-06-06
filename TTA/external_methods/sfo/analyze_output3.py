"""Analyze and visualize output3 phase1 training results."""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['axes.titlesize'] = 13
matplotlib.rcParams['figure.dpi'] = 120

BASE = './output3'
EXPERIMENTS = {
    'phase1_resnet50_direct': 'ResNet50 (ZOA, default)',
    'phase1_resnet50_direct_fanout': 'ResNet50 (ZOA, fan_out)',
    'phase1_resnet50_direct_pretrained': 'ResNet50 (pretrained)',
    'phase1_vitb_direct': 'ViT-B (ZOA, fan_in)',
    'phase1_vitb_direct_fanout': 'ViT-B (ZOA, fan_out)',
    'phase1_vitb_direct_pretrained': 'ViT-B (pretrained, fan_in)',
}

CORRUPTION_COLS = [
    'val_IntensityIncreasing', 'val_SaturationIncreasing', 'val_SharpnessIncreasing',
    'val_GaussianBlurIncreasing', 'val_PosterizeIncreasing', 'val_SolarizeIncreasing',
    'val_SaltAndPepperIncreasing',
]


def parse_results(filepath):
    """Parse a phase1_results.txt file into structured data."""
    epochs, losses, lr_vals = [], [], []
    loss_components = {'L_cls': [], 'L_radius': [], 'L_angular': [], 'L_dist': []}
    val_epochs, val_clean, val_mean = [], [], []
    val_corruptions = {c: [] for c in CORRUPTION_COLS}

    with open(filepath, 'r') as f:
        header = f.readline().strip().split('\t')
        for line in f:
            parts = line.strip().split('\t')
            if not parts or parts[0] == '':
                continue
            epoch = int(parts[0])
            epochs.append(epoch)
            losses.append(float(parts[1]))
            loss_components['L_cls'].append(float(parts[2]))
            loss_components['L_radius'].append(float(parts[3]))
            loss_components['L_angular'].append(float(parts[4]))
            loss_components['L_dist'].append(float(parts[5]))
            lr_vals.append(float(parts[6]))

            # Check if validation data exists (non-empty columns)
            if len(parts) > 7 and parts[7] != '':
                val_epochs.append(epoch)
                val_clean.append(float(parts[7]))
                for i, col in enumerate(CORRUPTION_COLS):
                    val_corruptions[col].append(float(parts[8 + i]))
                val_mean.append(float(parts[15]))

    return {
        'epochs': epochs, 'losses': losses, 'lr': lr_vals,
        'loss_components': loss_components,
        'val_epochs': val_epochs, 'val_clean': val_clean, 'val_mean': val_mean,
        'val_corruptions': val_corruptions,
    }


def load_all():
    data = {}
    for exp_name, label in EXPERIMENTS.items():
        fpath = os.path.join(BASE, exp_name, 'phase1_results.txt')
        if os.path.exists(fpath):
            data[exp_name] = parse_results(fpath)
            data[exp_name]['label'] = label
    return data


def plot_all(data):
    resnet_exps = [k for k in data if 'resnet' in k]
    vit_exps = [k for k in data if 'vitb' in k]

    colors_resnet = {'phase1_resnet50_direct': '#1f77b4',
                     'phase1_resnet50_direct_fanout': '#ff7f0e',
                     'phase1_resnet50_direct_pretrained': '#2ca02c'}
    colors_vit = {'phase1_vitb_direct': '#d62728',
                  'phase1_vitb_direct_fanout': '#9467bd',
                  'phase1_vitb_direct_pretrained': '#8c564b'}

    # ======================================================================
    # Figure 1: Training Loss & Val Mean Accuracy (side by side for each arch)
    # ======================================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Phase1 Training Overview: output3', fontsize=15, fontweight='bold')

    # ResNet - Loss
    ax = axes[0, 0]
    for exp in resnet_exps:
        d = data[exp]
        ax.plot(d['epochs'], d['losses'], label=d['label'], color=colors_resnet[exp], linewidth=2)
    ax.set_title('ResNet50 - Training Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ResNet - Val Mean
    ax = axes[0, 1]
    for exp in resnet_exps:
        d = data[exp]
        if d['val_epochs']:
            ax.plot(d['val_epochs'], d['val_mean'], 'o-', label=d['label'],
                    color=colors_resnet[exp], linewidth=2, markersize=5)
    ax.set_title('ResNet50 - Validation Mean Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean Accuracy (%)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ViT - Loss
    ax = axes[1, 0]
    for exp in vit_exps:
        d = data[exp]
        ax.plot(d['epochs'], d['losses'], label=d['label'], color=colors_vit[exp], linewidth=2)
    ax.set_title('ViT-B - Training Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ViT - Val Mean
    ax = axes[1, 1]
    for exp in vit_exps:
        d = data[exp]
        if d['val_epochs']:
            ax.plot(d['val_epochs'], d['val_mean'], 'o-', label=d['label'],
                    color=colors_vit[exp], linewidth=2, markersize=5)
    ax.set_title('ViT-B - Validation Mean Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean Accuracy (%)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('./output3/fig1_overview.png', bbox_inches='tight')
    plt.close()

    # ======================================================================
    # Figure 2: Loss Components
    # ======================================================================
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    fig.suptitle('Loss Components Breakdown', fontsize=15, fontweight='bold')
    comp_names = ['L_cls', 'L_radius', 'L_angular', 'L_dist']
    comp_titles = ['Classification Loss', 'Radius Loss', 'Angular Loss', 'Distance Loss']

    for i, (comp, title) in enumerate(zip(comp_names, comp_titles)):
        # ResNet row
        ax = axes[0, i]
        for exp in resnet_exps:
            d = data[exp]
            ax.plot(d['epochs'], d['loss_components'][comp],
                    label=d['label'], color=colors_resnet[exp], linewidth=1.5)
        ax.set_title(f'ResNet50 - {title}')
        ax.set_xlabel('Epoch')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7)

        # ViT row
        ax = axes[1, i]
        for exp in vit_exps:
            d = data[exp]
            ax.plot(d['epochs'], d['loss_components'][comp],
                    label=d['label'], color=colors_vit[exp], linewidth=1.5)
        ax.set_title(f'ViT-B - {title}')
        ax.set_xlabel('Epoch')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig('./output3/fig2_loss_components.png', bbox_inches='tight')
    plt.close()

    # ======================================================================
    # Figure 3: Per-Corruption Validation Accuracy (last epoch)
    # ======================================================================
    short_names = ['Intensity', 'Saturation', 'Sharpness', 'GaussBlur', 'Posterize', 'Solarize', 'S&P']

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Per-Corruption Accuracy at Final Epoch', fontsize=15, fontweight='bold')

    x = np.arange(len(short_names))
    width = 0.25

    # ResNet
    ax = axes[0]
    for j, exp in enumerate(resnet_exps):
        d = data[exp]
        if d['val_epochs']:
            vals = [d['val_corruptions'][c][-1] for c in CORRUPTION_COLS]
            bars = ax.bar(x + j * width, vals, width, label=d['label'],
                          color=list(colors_resnet.values())[j], alpha=0.85)
    ax.set_xticks(x + width)
    ax.set_xticklabels(short_names, rotation=30, ha='right')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('ResNet50')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # ViT
    ax = axes[1]
    for j, exp in enumerate(vit_exps):
        d = data[exp]
        if d['val_epochs']:
            vals = [d['val_corruptions'][c][-1] for c in CORRUPTION_COLS]
            bars = ax.bar(x + j * width, vals, width, label=d['label'],
                          color=list(colors_vit.values())[j], alpha=0.85)
    ax.set_xticks(x + width)
    ax.set_xticklabels(short_names, rotation=30, ha='right')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('ViT-B')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('./output3/fig3_per_corruption.png', bbox_inches='tight')
    plt.close()

    # ======================================================================
    # Figure 4: Convergence Analysis (for resume feasibility)
    # ======================================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Convergence Analysis: Is Resume Training Worthwhile?', fontsize=15, fontweight='bold')

    # Val Mean Accuracy - last N epochs improvement
    for row, (exps, colors, arch) in enumerate([
        (resnet_exps, colors_resnet, 'ResNet50'),
        (vit_exps, colors_vit, 'ViT-B'),
    ]):
        # Left: Val mean trend with improvement rate annotation
        ax = axes[row, 0]
        for exp in exps:
            d = data[exp]
            if len(d['val_mean']) >= 2:
                ax.plot(d['val_epochs'], d['val_mean'], 'o-', label=d['label'],
                        color=colors[exp], linewidth=2, markersize=5)
                # Annotate last few improvements
                last_3 = d['val_mean'][-3:] if len(d['val_mean']) >= 3 else d['val_mean']
                if len(last_3) >= 2:
                    improvement = last_3[-1] - last_3[-2]
                    ax.annotate(f'{improvement:+.2f}%',
                                xy=(d['val_epochs'][-1], d['val_mean'][-1]),
                                fontsize=8, fontweight='bold', color=colors[exp],
                                xytext=(5, 5), textcoords='offset points')

        ax.set_title(f'{arch} - Val Mean Accuracy (with last-step improvement)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Mean Accuracy (%)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Right: Epoch-to-epoch improvement of val_mean
        ax = axes[row, 1]
        for exp in exps:
            d = data[exp]
            if len(d['val_mean']) >= 2:
                improvements = [d['val_mean'][i] - d['val_mean'][i - 1]
                                for i in range(1, len(d['val_mean']))]
                ep_pairs = d['val_epochs'][1:]
                ax.plot(ep_pairs, improvements, 'o-', label=d['label'],
                        color=colors[exp], linewidth=1.5, markersize=4)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_title(f'{arch} - Val Mean Improvement per Step')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Improvement (%)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('./output3/fig4_convergence_analysis.png', bbox_inches='tight')
    plt.close()

    # ======================================================================
    # Figure 5: Val Clean Accuracy
    # ======================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Validation Clean Accuracy', fontsize=15, fontweight='bold')

    ax = axes[0]
    for exp in resnet_exps:
        d = data[exp]
        if d['val_epochs']:
            ax.plot(d['val_epochs'], d['val_clean'], 'o-', label=d['label'],
                    color=colors_resnet[exp], linewidth=2, markersize=5)
    ax.set_title('ResNet50 - Val Clean')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Clean Accuracy (%)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for exp in vit_exps:
        d = data[exp]
        if d['val_epochs']:
            ax.plot(d['val_epochs'], d['val_clean'], 'o-', label=d['label'],
                    color=colors_vit[exp], linewidth=2, markersize=5)
    ax.set_title('ViT-B - Val Clean')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Clean Accuracy (%)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('./output3/fig5_val_clean.png', bbox_inches='tight')
    plt.close()

    # ======================================================================
    # Figure 6: Learning Rate Schedule
    # ======================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle('Learning Rate Schedule', fontsize=14, fontweight='bold')

    ax = axes[0]
    for exp in resnet_exps:
        d = data[exp]
        ax.plot(d['epochs'], d['lr'], label=d['label'], color=colors_resnet[exp], linewidth=1.5)
    ax.set_title('ResNet50')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for exp in vit_exps:
        d = data[exp]
        ax.plot(d['epochs'], d['lr'], label=d['label'], color=colors_vit[exp], linewidth=1.5)
    ax.set_title('ViT-B')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('./output3/fig6_lr_schedule.png', bbox_inches='tight')
    plt.close()


def print_summary(data):
    print("=" * 100)
    print("PHASE1 TRAINING RESULTS SUMMARY (output3)")
    print("=" * 100)

    for exp_name, d in data.items():
        label = d['label']
        total_epochs = d['epochs'][-1] + 1
        final_loss = d['losses'][-1]

        if d['val_mean']:
            best_val_mean = max(d['val_mean'])
            best_val_epoch = d['val_epochs'][d['val_mean'].index(best_val_mean)]
            final_val_mean = d['val_mean'][-1]
            final_val_clean = d['val_clean'][-1]
        else:
            best_val_mean = best_val_epoch = final_val_mean = final_val_clean = 'N/A'

        print(f"\n--- {label} ({exp_name}) ---")
        print(f"  Epochs: {total_epochs}, Final Loss: {final_loss:.4f}")
        print(f"  Final Val Clean: {final_val_clean:.2f}%, Final Val Mean: {final_val_mean:.2f}%")
        print(f"  Best Val Mean: {best_val_mean:.2f}% (epoch {best_val_epoch})")

        if len(d['val_mean']) >= 3:
            last3 = d['val_mean'][-3:]
            last3_epochs = d['val_epochs'][-3:]
            print(f"  Last 3 val_mean: {[f'{v:.2f}' for v in last3]} at epochs {last3_epochs}")
            improvement_last = last3[-1] - last3[-2]
            improvement_2nd = last3[-2] - last3[-3]
            print(f"  Improvement trend: {improvement_2nd:+.2f}% -> {improvement_last:+.2f}%")

    # Convergence analysis
    print("\n" + "=" * 100)
    print("CONVERGENCE / RESUME ANALYSIS")
    print("=" * 100)

    for exp_name, d in data.items():
        label = d['label']
        if len(d['val_mean']) < 3:
            continue

        improvements = [d['val_mean'][i] - d['val_mean'][i - 1] for i in range(1, len(d['val_mean']))]
        last_improvement = improvements[-1] if improvements else 0
        avg_last3_improvement = np.mean(improvements[-3:]) if len(improvements) >= 3 else np.mean(improvements)

        # Loss still decreasing?
        last_losses = d['losses'][-5:]
        loss_decreasing = last_losses[-1] < last_losses[0]

        # LR at end
        final_lr = d['lr'][-1]

        print(f"\n--- {label} ---")
        print(f"  Last val_mean improvement: {last_improvement:+.2f}%")
        print(f"  Avg last 3 improvements: {avg_last3_improvement:+.2f}%")
        print(f"  Loss still decreasing (last 5 epochs): {loss_decreasing} "
              f"({last_losses[0]:.4f} -> {last_losses[-1]:.4f})")
        print(f"  Final LR: {final_lr:.2e}")

        # Verdict
        if abs(last_improvement) < 0.5 and not loss_decreasing:
            verdict = "CONVERGED - Resume unlikely to help significantly"
        elif last_improvement > 0.5 or (loss_decreasing and avg_last3_improvement > 0.3):
            verdict = "STILL IMPROVING - Resume could help"
        elif loss_decreasing and abs(last_improvement) < 0.5:
            verdict = "SLOWING DOWN - Resume may help marginally (consider higher LR restart)"
        else:
            verdict = "MIXED - Consider resume with fresh LR schedule"

        print(f"  >>> Verdict: {verdict}")

    # Final comparison table
    print("\n" + "=" * 100)
    print("FINAL COMPARISON TABLE")
    print("=" * 100)
    print(f"{'Experiment':<35} {'Epochs':>7} {'Val Clean':>10} {'Val Mean':>10} {'Best Mean':>10} {'Final Loss':>11}")
    print("-" * 88)
    for exp_name, d in data.items():
        label = d['label']
        total = d['epochs'][-1] + 1
        fc = d['val_clean'][-1] if d['val_clean'] else 0
        fm = d['val_mean'][-1] if d['val_mean'] else 0
        bm = max(d['val_mean']) if d['val_mean'] else 0
        fl = d['losses'][-1]
        print(f"{label:<35} {total:>7} {fc:>9.2f}% {fm:>9.2f}% {bm:>9.2f}% {fl:>10.4f}")


if __name__ == '__main__':
    data = load_all()
    print_summary(data)
    plot_all(data)
    print("\nPlots saved to ./output3/fig1_overview.png ~ fig6_lr_schedule.png")
