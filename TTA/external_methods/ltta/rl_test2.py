#!/usr/bin/env python3
"""Quick test script for Entropy Minimization TTA (progressive elimination).

This is a copy of `rl_test.py` with the following change:
- Run repeated trials with random actions.
- If a sample is predicted correctly at least once, it is removed from future trials.

Goal: measure how many trials are needed until all (default: 110) samples become correct at least once.
"""
import argparse
import os
import json
import sys

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

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    print(f"{'='*80}\n")
    return seed


def _parse_args():
    """Parse command line arguments (kept compatible with rl_test.py)."""
    parser = argparse.ArgumentParser(description="RL Training for Entropy Minimization TTA")

    # Config file
    parser.add_argument(
        "-c",
        "--config",
        default="",
        type=str,
        metavar="FILE",
        help="YAML config file specifying default arguments",
    )

    # Dataset parameters
    parser.add_argument(
        "data",
        nargs="?",
        metavar="DIR",
        const=None,
        help="path to dataset (positional is *deprecated*, use --data-dir)",
    )
    parser.add_argument("--data-dir", metavar="DIR", help="path to dataset (root dir)")
    parser.add_argument(
        "--dataset",
        metavar="NAME",
        default="",
        help='dataset type + name ("<type>/<name>")',
    )
    parser.add_argument(
        "--train-split",
        metavar="NAME",
        default="train",
        help="dataset train split (default: train)",
    )
    parser.add_argument(
        "--val-split",
        metavar="NAME",
        default="validation",
        help="dataset validation split (default: validation)",
    )
    parser.add_argument(
        "--dataset-download",
        action="store_true",
        default=False,
        help="Allow download of dataset",
    )
    parser.add_argument(
        "--class-map",
        default="",
        type=str,
        metavar="FILENAME",
        help="path to class to idx mapping file",
    )
    parser.add_argument(
        "--dataset-alias",
        default="imagenet",
        type=str,
        help='alias of dataset (default: "imagenet")',
    )

    # Model parameters
    parser.add_argument(
        "--model",
        default="resnet50",
        type=str,
        metavar="MODEL",
        help='Name of model to train (default: "resnet50")',
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=False,
        help="Start with pretrained version of specified network",
    )
    parser.add_argument(
        "--initial-checkpoint",
        default="",
        type=str,
        metavar="PATH",
        help="Initialize model from this checkpoint (default: none)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        metavar="N",
        help="number of label classes (Model default if None)",
    )
    parser.add_argument(
        "--in-chans",
        type=int,
        default=None,
        metavar="N",
        help="Image input channels (default: None => 3)",
    )
    parser.add_argument(
        "--input-size",
        default=None,
        nargs=3,
        type=int,
        metavar="N N N",
        help="Input all image dimensions (d h w)",
    )

    # RL-specific parameters
    parser.add_argument(
        "--total-episodes",
        type=int,
        default=5000,
        metavar="N",
        help="Total number of RL episodes to run (default: 5000)",
    )
    parser.add_argument(
        "--step-count-limit",
        type=int,
        default=5,
        metavar="N",
        help="Step count limit per datapoint (default: 5)",
    )
    parser.add_argument(
        "--cls-penalty",
        type=float,
        default=2.0,
        metavar="N",
        help="Classification penalty for wrong prediction (default: 2.0)",
    )
    parser.add_argument(
        "--action-scale-factor",
        type=float,
        default=1.0,
        metavar="N",
        help="Scale factor for action space range (default: 1.0)",
    )
    parser.add_argument(
        "--rl-alpha",
        type=float,
        default=0.2,
        metavar="N",
        help="Entropy regularization coefficient (alpha) (default: 0.2)",
    )
    parser.add_argument(
        "--rl-alpha-auto",
        action="store_true",
        default=False,
        help="Enable automatic entropy tuning for alpha",
    )

    # AutoEncoder parameters
    parser.add_argument(
        "--ae-path",
        type=str,
        default=None,
        help="Path to pretrained AutoEncoder model (default: None)",
    )
    parser.add_argument(
        "--squeeze-dim",
        type=int,
        default=32,
        help="Dimension of the latent vector for AutoEncoder (default: 128)",
    )

    # Output
    parser.add_argument(
        "--output",
        default="./output/rl",
        type=str,
        metavar="PATH",
        help="path to output folder (default: ./output/rl)",
    )
    parser.add_argument(
        "--experiment",
        default="temp",
        type=str,
        metavar="NAME",
        help="name of train experiment, name of sub-folder for output",
    )

    # Misc
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="S",
        help="random seed (default: 42)",
    )
    parser.add_argument(
        "--force-run-steps",
        action="store_true",
        default=False,
        help="Do not stop when initial prediction is already correct; always run step loop.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        metavar="N",
        help="how many training processes to use (default: 4)",
    )
    parser.add_argument(
        "--pin-mem",
        action="store_true",
        default=False,
        help="Pin CPU memory in DataLoader",
    )
    parser.add_argument(
        "--prefetcher",
        action="store_true",
        default=True,
        help="Use fast prefetcher",
    )
    parser.add_argument(
        "--no-prefetcher",
        dest="prefetcher",
        action="store_false",
        help="Disable fast prefetcher",
    )

    # rl_test2 specific (optional; defaults keep the original CLI compatible)
    parser.add_argument(
        "--max-trials",
        type=int,
        default=5000,
        metavar="N",
        help="Max number of trials to run until all samples are solved (default: 5000)",
    )
    parser.add_argument(
        "--samples-per-trial",
        type=int,
        default=110,
        metavar="N",
        help="Number of samples to track/solve (default: 110)",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        metavar="N",
        help="Print summary every N trials (default: 1)",
    )
    parser.add_argument(
        "--print-results",
        action="store_true",
        default=False,
        help="Also print pred/gt pairs for the remaining samples each trial (may be very verbose).",
    )
    parser.add_argument(
        "--solve-criterion",
        type=str,
        default="any",
        choices=["any", "last"],
        help="When to drop a sample from future trials: "
             "'any' = drop if correct at least once within the step loop; "
             "'last' = drop only if correct at the last step (default: any).",
    )

    # Parse config file first
    args_config, _remaining = parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, "r") as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # Parse args again so CLI flags override defaults (including defaults loaded from YAML config)
    args = parser.parse_args()

    # Handle positional data argument
    if args.data and not args.data_dir:
        args.data_dir = args.data

    return args


def _cli_has_arg(flag_name: str) -> bool:
    """Return True if a CLI flag is explicitly present (supports --flag and --flag=value)."""
    prefix = flag_name + "="
    for a in sys.argv[1:]:
        if a == flag_name or a.startswith(prefix):
            return True
    return False


def main():
    """Run trials with random actions; remove samples once they become correct at least once."""
    utils.setup_default_logging()
    args = _parse_args()

    # Make the run deterministic and lightweight.
    args.batch_size = 1  # ensure a single sample is pulled if/when we use env.reset()
    args.total_episodes = 1
    # Keep rl_test.py compatibility: if user doesn't specify --step-count-limit, default to 1 step.
    # If they DO specify it, respect their value for multi-step testing.
    if not _cli_has_arg("--step-count-limit"):
        args.step_count_limit = 1

    # ---- START: print key runtime config as early as possible ----
    asf = float(getattr(args, "action_scale_factor", 1.0))
    max_step_size = int(getattr(args, "step_count_limit", 1))
    print(f"[TEST2] START CONFIG | action_scale_factor={asf} | max_step_size(step-count-limit)={max_step_size}")
    # ---- END: print key runtime config as early as possible ----

    setup_reproducibility(args.seed)

    exp_name = args.experiment or "test"
    output_dir = os.path.join(args.output, exp_name, "rl_test2")
    os.makedirs(output_dir, exist_ok=True)
    print(f"[TEST2] Output directory: {output_dir}")

    # Save args for traceability
    args_file = os.path.join(output_dir, "args.yaml")
    with open(args_file, "w") as f:
        yaml.dump(vars(args), f, default_flow_style=False)
    print(f"[TEST2] Saved arguments to {args_file}")

    # Initialize environment
    print(f"\n{'='*80}")
    print("[TEST2] Initializing RL Environment (single-sample mode)")
    print(f"{'='*80}\n")
    env = RLEnvironmentEntropyTTA(args)

    action_space = env.get_action_space()
    try:
        action_space.seed(args.seed)
    except Exception:
        # Some gym versions / space wrappers may not support seeding; it's OK (still random).
        pass

    state_size = env.get_state_size()
    print(f"[TEST2] State size: {state_size}, Action space: {action_space.shape}")
    solve_criterion = str(getattr(args, "solve_criterion", "any"))
    print(f"[TEST2] Step-count-limit: {args.step_count_limit} | solve_criterion: {solve_criterion}")

    # Helper for prediction
    def get_prediction(env_obj):
        env_obj.model.eval()
        with torch.no_grad():
            out = env_obj.model(env_obj.current_input)
            logits = out[0] if isinstance(out, tuple) else out
            probs = F.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
        return pred

    # ------------------------------------------------------------
    # 1) Capture a fixed set of samples (default: 110) once.
    # ------------------------------------------------------------
    target_n = int(getattr(args, "samples_per_trial", 110))
    total_available = env.get_total_datapoints()
    if total_available and total_available > 0:
        target_n = min(target_n, int(total_available))

    # Reset loader iteration so we capture a consistent set
    if hasattr(env, "runtime_buffer"):
        env.runtime_buffer = []
    if hasattr(env, "train_loader") and env.train_loader is not None:
        env.runtime_iterator = iter(env.train_loader)

    samples = []
    print(f"[TEST2] Capturing {target_n} samples to track (from dataset order)...")
    for _i in range(target_n):
        _ = env.reset()
        # Save tensors (keep on device; small for CIFAR10)
        # NOTE: clone() to avoid any potential loader/prefetcher tensor reuse.
        inp = env.current_input.detach().clone()
        gt = env.current_gt.detach().clone()
        samples.append((inp, gt))
    print(f"[TEST2] Captured {len(samples)} samples.")

    # ------------------------------------------------------------
    # 2) Trials with elimination: once correct at least once => removed
    # ------------------------------------------------------------
    solved = [False] * len(samples)
    solved_at = [-1] * len(samples)  # trial index when first solved

    max_trials = int(getattr(args, "max_trials", 5000))
    log_every = max(1, int(getattr(args, "log_every", 1)))
    print_results = bool(getattr(args, "print_results", False))

    trial = 0
    while trial < max_trials:
        remaining_idx = [i for i, s in enumerate(solved) if not s]
        remaining_n = len(remaining_idx)
        if remaining_n == 0:
            break

        trial += 1
        newly_solved = 0
        preds = []

        for i in remaining_idx:
            inp, gt = samples[i]

            # Reset model to original (per sample)
            env.reset_norm_params()

            # Set current datapoint
            env.current_input = inp
            env.current_gt = gt
            env.current_class_id = int(gt.item())

            # Reset step tracking
            env.step_count = 0
            env.cum_reward = 0.0
            env.prev_entropy = None
            env.initial_entropy = None
            env.initial_predicted_class = None

            gt_i = int(gt.item())

            solved_now = False
            steps = max(1, int(getattr(args, "step_count_limit", 1)))

            if solve_criterion == "any":
                # Drop if correct at least once within the step loop.
                for step_idx in range(steps):
                    action = action_space.sample()
                    next_state, _reward, _done, _truncated = env.step(action)
                    assert next_state.shape[0] == state_size, (
                        f"Next state size mismatch: {next_state.shape[0]} vs {state_size}"
                    )

                    pred = get_prediction(env)
                    if print_results and step_idx == steps - 1:
                        preds.append(f"{pred}/{gt_i}")
                    if pred == gt_i:
                        solved_now = True
                        break  # save compute
            else:
                # solve_criterion == "last"
                # Drop only if correct at the *last* step.
                pred = None
                for step_idx in range(steps):
                    action = action_space.sample()
                    next_state, _reward, _done, _truncated = env.step(action)
                    assert next_state.shape[0] == state_size, (
                        f"Next state size mismatch: {next_state.shape[0]} vs {state_size}"
                    )
                    if step_idx == steps - 1:
                        pred = get_prediction(env)
                if pred is None:
                    pred = get_prediction(env)
                if print_results:
                    preds.append(f"{pred}/{gt_i}")
                solved_now = (pred == gt_i)

            if solved_now and not solved[i]:
                solved[i] = True
                solved_at[i] = trial
                newly_solved += 1

        solved_count = sum(solved)
        remain_count = len(samples) - solved_count
        rate = newly_solved / max(remaining_n, 1)

        if (trial % log_every) == 0 or remain_count == 0:
            if print_results:
                preds_str = ", ".join(preds)
                print(
                    f"[TEST2] Trial {trial}/{max_trials} | remaining: {remaining_n} | "
                    f"results: [{preds_str}] | newly_solved: {newly_solved}/{remaining_n} ({rate*100:.1f}%) | "
                    f"cumulative_solved: {solved_count}/{len(samples)} | left: {remain_count}"
                )
            else:
                print(
                    f"[TEST2] Trial {trial}/{max_trials} | "
                    f"newly_solved: {newly_solved}/{remaining_n} ({rate*100:.1f}%) | "
                    f"cumulative_solved: {solved_count}/{len(samples)} | left: {remain_count}"
                )

    # Final summary
    solved_count = sum(solved)
    remain_count = len(samples) - solved_count
    if remain_count == 0:
        print(f"[TEST2] All {len(samples)} samples became correct at least once by trial {trial}.")
    else:
        print(
            f"[TEST2] Reached max_trials={max_trials} with {solved_count}/{len(samples)} solved "
            f"({remain_count} still unsolved)."
        )

    # Save solved_at for analysis
    solved_path = os.path.join(output_dir, "solved_at.json")
    with open(solved_path, "w") as f:
        json.dump(
            {
                "total_samples": len(samples),
                "max_trials": max_trials,
                "trials_ran": trial,
                "solved_count": solved_count,
                "unsolved_count": remain_count,
                "solved_at": solved_at,
            },
            f,
            indent=2,
        )
    print(f"[TEST2] Saved per-sample first-solved trial to {solved_path}")

    # ---- END: print key runtime config again at the end ----
    print(f"[TEST2] END CONFIG | action_scale_factor={asf} | max_step_size(step-count-limit)={max_step_size}")


if __name__ == "__main__":
    main()

