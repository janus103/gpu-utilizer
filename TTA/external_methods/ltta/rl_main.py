#!/usr/bin/env python3
"""
RL Main Script for Entropy Minimization TTA
Implements algorithm1.md: representative samples × augmentation × N_step episodes
"""
import argparse
import copy
import itertools
import json
import os
import random
import time
from datetime import datetime

import numpy as np
import torch
import yaml

from timm import utils
from timm.rl_module import RLEnvironmentEntropyTTA, ReplayMemory, SAC


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


def _add_rl_args(args):
    """Add RL-specific arguments"""
    rl_args = copy.copy(args)
    
    rl_args.policy = "Gaussian"
    rl_args.eval = True
    rl_args.gamma = 0.99
    rl_args.tau = 0.005
    rl_args.lr = 0.0003
    
    # Alpha clamping
    rl_alpha = getattr(args, 'rl_alpha', 0.2)
    rl_args.alpha = max(0.0, min(1.0, rl_alpha))
    
    rl_args.automatic_entropy_tuning = getattr(args, 'rl_alpha_auto', False)
    rl_args.batch_size = 128
    rl_args.hidden_size = 512
    rl_args.updates_per_step = 1
    rl_args.target_update_interval = 1
    rl_args.replay_size = 5000
    rl_args.cuda = torch.cuda.is_available()
    
    # Calculate num_steps and start_steps based on episodes
    total_episodes = getattr(args, 'total_episodes', 5000)
    step_count_limit = getattr(args, 'step_count_limit', 5)
    
    # Estimate datapoints per episode (will be set after env.build_datapoint_list)
    # For now, use a placeholder
    rl_args.num_steps = 1000000  # Will be updated
    rl_args.start_steps = 10000  # Will be updated
    
    return rl_args


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
    """Main training loop"""
    utils.setup_default_logging()
    args = _parse_args()
    
    # Setup reproducibility
    setup_reproducibility(args.seed)
    
    # Create output directory
    if args.experiment:
        exp_name = args.experiment
    else:
        exp_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    output_dir = os.path.join(args.output, exp_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"[MAIN] Output directory: {output_dir}")
    
    # Save args
    args_file = os.path.join(output_dir, 'args.yaml')
    with open(args_file, 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)
    print(f"[MAIN] Saved arguments to {args_file}")
    
    # Initialize environment
    print(f"\n{'='*80}")
    print(f"[MAIN] Initializing RL Environment")
    print(f"{'='*80}\n")
    env = RLEnvironmentEntropyTTA(args)

    # Get number of datapoints per episode (from runtime loader)
    num_datapoints = env.get_total_datapoints()
    
    if num_datapoints == 0:
        # Fallback if loader length is unknown
        num_datapoints = 100 
        print(f"[MAIN] Warning: Could not determine total datapoints, setting virtual = {num_datapoints}")
    else:
        print(f"[MAIN] Total datapoints per episode: {num_datapoints}")
    
    # # 디버깅을 위해서 해당 위치에서 멈추기
    # import sys; 
    # sys.exit(0)

    # Get action space and state size
    action_space = env.get_action_space()
    state_size = env.get_state_size()
    print(f"[MAIN] State size: {state_size}, Action space: {action_space.shape}")
    
    # Setup RL args
    rl_args = _add_rl_args(args)
    
    # Calculate actual num_steps and start_steps
    steps_per_datapoint = args.step_count_limit
    steps_per_episode = num_datapoints * steps_per_datapoint
    rl_args.num_steps = args.total_episodes * steps_per_episode
    rl_args.start_steps = int(max(1, args.total_episodes * 0.15) * steps_per_episode)
    
    print(f"[MAIN] Steps per datapoint: {steps_per_datapoint}")
    print(f"[MAIN] Steps per episode: {steps_per_episode}")
    print(f"[MAIN] Total steps: {rl_args.num_steps}")
    print(f"[MAIN] Start steps: {rl_args.start_steps}")
    
    # Initialize SAC agent
    print(f"\n{'='*80}")
    print(f"[MAIN] Initializing SAC Agent")
    print(f"{'='*80}\n")
    agent = SAC(state_size, action_space, rl_args)
    print(f"[MAIN] Agent initialized")
    
    # Initialize replay memory
    memory = ReplayMemory(rl_args.replay_size, args.seed)
    print(f"[MAIN] Replay memory initialized (capacity: {rl_args.replay_size})")
    
    # Training loop
    print(f"\n{'='*80}")
    print(f"[MAIN] Starting Training Loop")
    print(f"{'='*80}\n")
    
    total_numsteps = 0
    updates = 0
    best_episode_reward = float('-inf')
    
    # Create checkpoint directory
    
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training log file
    fname = (
        f"training_log_"
        f"asf{args.action_scale_factor}_"   # action_scale_factor
        f"bs{rl_args.batch_size}_"          # batch_size (RL 배치)
        f"rs{rl_args.replay_size}_"         # replay_size
        f"step{args.step_count_limit}_"     # step_count_limit
        f"ep{args.total_episodes}.txt"      # total_episodes
        ).replace('.', 'p')  # 파일명에 '.' 대신 'p'로 치환하여 안전하게
    log_file = os.path.join(output_dir, fname)
    with open(log_file, 'w') as f:
        f.write(f"Training started at {datetime.now()}\n")
        f.write(f"Total episodes: {args.total_episodes}\n")
        f.write(f"Steps per episode: {steps_per_episode}\n")
        f.write(f"Total steps: {rl_args.num_steps}\n")
        f.write(f"{'='*80}\n\n")
    
    for episode in itertools.count(1):
        if episode > args.total_episodes:
            print(f"\n[MAIN] Reached target episodes ({args.total_episodes}). Stopping...")
            break
        
        episode_total_reward = 0.0
        episode_steps = 0
        
        # Iterate over all datapoints in episode
        for datapoint_idx in range(num_datapoints):
            # Reset environment for new datapoint (CRITICAL: resets norm params)
            state = env.reset()
            
            # Run N_step for this datapoint
            for step in range(args.step_count_limit):
                # Select action
                if rl_args.start_steps > total_numsteps:
                    action = action_space.sample()
                else:
                    action = agent.select_action(state)
                
                # Step environment
                next_state, reward, done, truncated = env.step(action)
                
                total_numsteps += 1
                episode_steps += 1
                episode_total_reward += reward
                
                # Store transition
                transition_done = done or truncated
                positive = (reward >= 0)
                memory.push(state, action, reward, next_state, transition_done, positive)
                
                # Update agent
                if len(memory) >= rl_args.batch_size:
                    for _ in range(rl_args.updates_per_step):
                        try:
                            agent.update_parameters(memory, rl_args.batch_size, updates)
                            updates += 1
                        except ValueError:
                            break
                
                state = next_state
                
                if done or truncated:
                    break
        
        # Episode finished
        avg_reward_per_step = episode_total_reward / max(episode_steps, 1)
        
        print(f"Episode {episode}/{args.total_episodes} | "
              f"Total steps: {total_numsteps} | "
              f"Episode reward: {episode_total_reward:.2f} | "
              f"Avg reward/step: {avg_reward_per_step:.4f}")
        
        # Log to file
        with open(log_file, 'a') as f:
            f.write(f"Episode {episode} | Total steps: {total_numsteps} | "
                   f"Episode reward: {episode_total_reward:.2f} | "
                   f"Avg reward/step: {avg_reward_per_step:.4f}\n")
        
        # Save actor checkpoint if best episode reward
        if episode_total_reward > best_episode_reward:
            best_episode_reward = episode_total_reward
            actor_ckpt_path = os.path.join(checkpoint_dir, 
                                         f'best_actor_ep{episode}_reward{episode_total_reward:.4f}.pth')
            agent.save_actor_checkpoint(actor_ckpt_path, episode_total_reward)
            print(f"[MAIN] New best episode reward: {best_episode_reward:.4f} -> Saved actor to {actor_ckpt_path}")
    
    print(f"\n{'='*80}")
    print(f"[MAIN] Training completed!")
    print(f"Best episode reward: {best_episode_reward:.4f}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

