#!/usr/bin/env python3
"""Quick test script for Entropy Minimization TTA."""
import argparse
import os

import torch
import torch.nn.functional as F
import yaml

from timm import utils
from timm.rl_module import RLEnvironmentEntropyTTA


def setup_reproducibility(seed):
    """Setup reproducibility with seed"""
    import random
    import numpy as np
    import torch
    import os
    
    print(f"\n{'='*80}")
    print(f"[REPRODUCIBILITY] Setting up deterministic behavior with seed={seed}")
    print(f"{'='*80}")
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    print(f"{'='*80}\n")
    return seed


def _parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='RL Training for Entropy Minimization TTA')
    
    # Config file
    parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                       help='YAML config file specifying default arguments')
    
    # Dataset parameters
    parser.add_argument('data', nargs='?', metavar='DIR', const=None,
                       help='path to dataset (positional is *deprecated*, use --data-dir)')
    parser.add_argument('--data-dir', metavar='DIR', help='path to dataset (root dir)')
    parser.add_argument('--dataset', metavar='NAME', default='',
                       help='dataset type + name ("<type>/<name>")')
    parser.add_argument('--train-split', metavar='NAME', default='train',
                       help='dataset train split (default: train)')
    parser.add_argument('--val-split', metavar='NAME', default='validation',
                       help='dataset validation split (default: validation)')
    parser.add_argument('--dataset-download', action='store_true', default=False,
                       help='Allow download of dataset')
    parser.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                       help='path to class to idx mapping file')
    parser.add_argument('--dataset-alias', default='imagenet', type=str,
                       help='alias of dataset (default: "imagenet")')
    
    # Model parameters
    parser.add_argument('--model', default='resnet50', type=str, metavar='MODEL',
                       help='Name of model to train (default: "resnet50")')
    parser.add_argument('--pretrained', action='store_true', default=False,
                       help='Start with pretrained version of specified network')
    parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                       help='Initialize model from this checkpoint (default: none)')
    parser.add_argument('--num-classes', type=int, default=None, metavar='N',
                       help='number of label classes (Model default if None)')
    parser.add_argument('--in-chans', type=int, default=None, metavar='N',
                       help='Image input channels (default: None => 3)')
    parser.add_argument('--input-size', default=None, nargs=3, type=int,
                       metavar='N N N', help='Input all image dimensions (d h w)')
    
    # RL-specific parameters
    parser.add_argument('--total-episodes', type=int, default=5000, metavar='N',
                       help='Total number of RL episodes to run (default: 5000)')
    parser.add_argument('--step-count-limit', type=int, default=5, metavar='N',
                       help='Step count limit per datapoint (default: 5)')
    parser.add_argument('--cls-penalty', type=float, default=2.0, metavar='N',
                       help='Classification penalty for wrong prediction (default: 2.0)')
    parser.add_argument('--action-scale-factor', type=float, default=1.0, metavar='N',
                       help='Scale factor for action space range (default: 1.0)')
    parser.add_argument('--rl-alpha', type=float, default=0.2, metavar='N',
                       help='Entropy regularization coefficient (alpha) (default: 0.2)')
    parser.add_argument('--rl-alpha-auto', action='store_true', default=False,
                       help='Enable automatic entropy tuning for alpha')
    
    # AutoEncoder parameters
    parser.add_argument('--ae-path', type=str, default=None,
                       help='Path to pretrained AutoEncoder model (default: None)')
    parser.add_argument('--squeeze-dim', type=int, default=32,
                       help='Dimension of the latent vector for AutoEncoder (default: 128)')
    
    # Output
    parser.add_argument('--output', default='./output/rl', type=str, metavar='PATH',
                       help='path to output folder (default: ./output/rl)')
    parser.add_argument('--experiment', default='temp', type=str, metavar='NAME',
                       help='name of train experiment, name of sub-folder for output')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                       help='random seed (default: 42)')
    parser.add_argument('--force-run-steps', action='store_true', default=False,
                       help='Do not stop when initial prediction is already correct; always run step loop.')
    parser.add_argument('--workers', type=int, default=4, metavar='N',
                       help='how many training processes to use (default: 4)')
    parser.add_argument('--pin-mem', action='store_true', default=False,
                       help='Pin CPU memory in DataLoader')
    parser.add_argument('--prefetcher', action='store_true', default=True,
                       help='Use fast prefetcher')
    parser.add_argument('--no-prefetcher', dest='prefetcher', action='store_false',
                       help='Disable fast prefetcher')
    
    # Parse config file first
    args_config, remaining = parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)
    
    # Parse args again so CLI flags override defaults (including defaults loaded from YAML config)
    args = parser.parse_args()
    
    # Handle positional data argument
    if args.data and not args.data_dir:
        args.data_dir = args.data
    
    return args


def main():
    """Run 50 trials; each trial runs 110 samples with a single action step.
    
    For each sample:
    - reset() to fetch one datapoint,
    - apply one random action (conv updated),
    - record (pred/gt) after the conv change.
    
    Prints per-trial results and the best success rate across all 50 trials.
    """
    utils.setup_default_logging()
    args = _parse_args()

    # Make the run deterministic and lightweight.
    setup_reproducibility(args.seed)
    args.batch_size = 1            # ensure a single sample is pulled
    args.total_episodes = 1        # keep output dirs simple
    args.step_count_limit = 1      # exactly one step per sample for this test

    exp_name = args.experiment or "test"
    output_dir = os.path.join(args.output, exp_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"[TEST] Output directory: {output_dir}")

    # Save args for traceability
    args_file = os.path.join(output_dir, 'args.yaml')
    with open(args_file, 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)
    print(f"[TEST] Saved arguments to {args_file}")

    # Initialize environment
    print(f"\n{'='*80}")
    print(f"[TEST] Initializing RL Environment (single-sample mode)")
    print(f"{'='*80}\n")
    env = RLEnvironmentEntropyTTA(args)

    # Action space and state size sanity
    action_space = env.get_action_space()
    state_size = env.get_state_size()
    print(f"[TEST] State size: {state_size}, Action space: {action_space.shape}")

    # Helper for prediction
    def get_prediction(env_obj):
        env_obj.model.eval()
        with torch.no_grad():
            out = env_obj.model(env_obj.current_input)
            logits = out[0] if isinstance(out, tuple) else out
            probs = F.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
        return pred, probs.squeeze(0).cpu()

    best_rate = 0.0
    trials = 50
    samples_per_trial = 110  # assume full dataset pass per trial

    for trial in range(1, trials + 1):
        preds = []
        successes = 0

        for _ in range(samples_per_trial):
            state = env.reset()
            assert state.shape[0] == state_size, f"State size mismatch: {state.shape[0]} vs {state_size}"
            gt = env.current_gt.item()

            # One-step action after reset
            action = action_space.sample()
            next_state, reward, done, truncated = env.step(action)
            assert next_state.shape[0] == state_size, f"Next state size mismatch: {next_state.shape[0]} vs {state_size}"

            pred, _ = get_prediction(env)
            correct = (pred == gt)
            successes += int(correct)
            preds.append(f"{pred}/{gt}")

        rate = successes / samples_per_trial
        best_rate = max(best_rate, rate)
        preds_str = ", ".join(preds)
        print(f"[TEST] Trial {trial}/{trials} | results: [{preds_str}] | success: {successes}/{samples_per_trial} ({rate*100:.1f}%)")

    print(f"[TEST] Best success rate over {trials} trials: {best_rate*100:.1f}%")


if __name__ == '__main__':
    main()

