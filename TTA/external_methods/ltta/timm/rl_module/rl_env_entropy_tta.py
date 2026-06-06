"""
RL Environment for Entropy Minimization TTA
Implements algorithm1.md: representative samples × augmentation × N_step episodes
"""
import copy
import json
import os
import random
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import spaces
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from PIL import Image

from timm import utils
from timm.data import create_dataset, create_loader, resolve_data_config
from timm.models import create_model, safe_model_name, load_checkpoint


class AutoEncoder(nn.Module):
    """AutoEncoder for first conv kernel statistics compression"""
    def __init__(self, input_dim, latent_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def encode(self, x):
        """Encode input to latent vector"""
        return self.encoder(x)
    
    def decode(self, z):
        """Decode latent vector to reconstructed statistics"""
        return self.decoder(z)


class RLEnvironmentEntropyTTA:
    """
    Environment for RL-based entropy minimization TTA.
    
    Key features:
    - Representative sample selection (per class, correct + min entropy)
    - Initial conv feature map statistics (per class, channel-wise mean/std)
    - Datapoint change triggers norm parameter reset
    - State/action/reward definition for RL
    """
    
    def __init__(self, args):
        print(f"[RLEnvironment] Initializing...")
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model setup
        self.model = self._get_model(args)
        self.original_state_dict = copy.deepcopy(self.model.state_dict())
        print(f"[RLEnvironment] Model loaded, original state saved")
        
        # AutoEncoder setup
        self.ae_model = None
        self.ae_latent_dim = getattr(args, 'squeeze_dim', 32)
        self.first_conv = None
        self._setup_autoencoder(args)
        
        # Data config
        self.data_config = resolve_data_config(vars(args), model=self.model, verbose=True)
        
        # Representative samples and stats (kept for compatibility, not used in runtime-only mode)
        self.rep_samples: Dict[int, Dict] = {}
        self.stem_conv_stats: Dict[int, Dict] = {}
        self.rep_sample_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        
        # Train dataset/loader (runtime only)
        self.train_dataset = None
        self.train_loader = None
        self.runtime_buffer: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.runtime_iterator = None
        print("[RLEnvironment] Setting up runtime data loading mode.")
        self._setup_runtime_loader()
        
        # Current datapoint tracking
        self.current_class_id = None
        self.current_input = None
        self.current_gt = None
        self.cum_reward = 0.0
        
        # Step tracking
        self.step_count = 0
        self.step_count_limit = getattr(args, 'step_count_limit', 5)
        
        # Initial entropy (for reward calculation)
        self.initial_entropy = None
        self.prev_entropy = None
        self.initial_predicted_class = None
        
        # Seeded random generator
        seed = getattr(args, 'seed', 42)
        self.rng = random.Random(seed)
        print(f"[RLEnvironment] Initialized with seed={seed}")
    
    def _get_model(self, args):
        """Load model from checkpoint or pretrained"""
        in_chans = 3
        if args.in_chans is not None:
            in_chans = args.in_chans
        elif args.input_size is not None:
            in_chans = args.input_size[0]

        # If loading from checkpoint and num_classes isn't provided, try to infer it from checkpoint head shape.
        ckpt_path = getattr(args, 'initial_checkpoint', '')
        if args.num_classes is None and ckpt_path:
            try:
                ckpt = torch.load(ckpt_path, map_location='cpu')
                state_dict = None
                if isinstance(ckpt, dict):
                    for k in ('state_dict', 'model_state_dict', 'model', 'model_state'):
                        if k in ckpt and isinstance(ckpt[k], dict):
                            state_dict = ckpt[k]
                            break
                    if state_dict is None:
                        state_dict = ckpt
                if isinstance(state_dict, dict):
                    # normalize common prefixes
                    norm_sd = {}
                    for k, v in state_dict.items():
                        if not isinstance(k, str):
                            continue
                        nk = k
                        if nk.startswith('module.'):
                            nk = nk[len('module.'):]
                        if nk.startswith('model.'):
                            nk = nk[len('model.'):]
                        norm_sd[nk] = v
                    for head_key in ('fc.weight', 'head.fc.weight', 'classifier.weight', 'head.weight'):
                        w = norm_sd.get(head_key)
                        if isinstance(w, torch.Tensor) and w.ndim == 2 and w.shape[0] > 0:
                            args.num_classes = int(w.shape[0])
                            print(f"[RLEnvironment] Inferred num_classes={args.num_classes} from checkpoint ({head_key})")
                            break
            except Exception as e:
                print(f"[RLEnvironment] Warning: failed to infer num_classes from checkpoint: {e}")
        
        model = create_model(
            args.model,
            pretrained=args.pretrained,
            in_chans=in_chans,
            num_classes=args.num_classes,
            drop_rate=getattr(args, 'drop', 0.0),
            drop_path_rate=getattr(args, 'drop_path', None),
            drop_block_rate=getattr(args, 'drop_block', None),
            global_pool=getattr(args, 'gp', None),
            bn_momentum=getattr(args, 'bn_momentum', None),
            bn_eps=getattr(args, 'bn_eps', None),
            scriptable=getattr(args, 'torchscript', False),
            checkpoint_path=ckpt_path,
        )
        
        if args.num_classes is None:
            assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr'
            args.num_classes = model.num_classes
        
        model.to(device=self.device)
        model.eval()
        return model
    
    def _setup_autoencoder(self, args):
        """Setup AutoEncoder for first conv kernel statistics"""
        ae_path = getattr(args, 'ae_path', None)
        
        # Find first conv layer
        self.first_conv = None
        for name, m in self.model.named_modules():
            if isinstance(m, nn.Conv2d):
                self.first_conv = m
                print(f"[RLEnvironment] Found first Conv2d layer: {name}")
                break
        
        if self.first_conv is None:
            raise RuntimeError("Could not find first Conv2d layer in model")
        
        if ae_path and os.path.exists(ae_path):
            print(f"[RLEnvironment] Loading AutoEncoder from {ae_path}")
            weight = self.first_conv.weight.data  # [out_channels, in_channels, k, k]
            
            # Calculate input dimension: 2 * (out_channels * in_channels)
            # (mean + variance for each kernel)
            weight_spatial_flat = weight.view(weight.size(0), weight.size(1), -1)  # [out, in, k*k]
            input_dim = 2 * (weight.size(0) * weight.size(1))  # [out*in*2]
            
            # Load AutoEncoder
            self.ae_model = AutoEncoder(input_dim, self.ae_latent_dim).to(self.device)
            self.ae_model.load_state_dict(torch.load(ae_path, map_location=self.device))
            self.ae_model.eval()
            print(f"[RLEnvironment] AutoEncoder loaded: input_dim={input_dim}, latent_dim={self.ae_latent_dim}")
        else:
            if ae_path:
                print(f"[RLEnvironment] Warning: AE path provided but file not found: {ae_path}")
            print(f"[RLEnvironment] AutoEncoder not loaded. AE is required for this implementation.")
            print(f"[RLEnvironment] Please provide --ae-path argument with a valid AutoEncoder checkpoint.")
    
    def _get_first_conv_statistics(self):
        """Extract mean and variance statistics from first conv layer weights"""
        if self.first_conv is None:
            raise RuntimeError("First conv layer not found")
        
        weight = self.first_conv.weight.data  # [out_channels, in_channels, k, k]
        weight_spatial_flat = weight.view(weight.size(0), weight.size(1), -1)  # [out, in, k*k]
        
        # Calculate mean and variance along spatial dimension (dim=2)
        kernel_means = weight_spatial_flat.mean(dim=2, keepdim=True)  # [out, in, 1]
        kernel_vars = weight_spatial_flat.var(dim=2, keepdim=True)    # [out, in, 1]
        
        # Flatten to 1D
        kernel_means_1d = kernel_means.view(-1)  # [out*in]
        kernel_vars_1d = kernel_vars.view(-1)   # [out*in]
        
        # Concatenate: [out*in*2]
        stats = torch.cat([kernel_means_1d, kernel_vars_1d], dim=0)
        
        return stats, kernel_means, kernel_vars, weight_spatial_flat
    
    def _update_first_conv_from_statistics(self, rec_means_1d, rec_vars_1d, kernel_means_shape, weight_spatial_flat_shape):
        """Update first conv weights using reconstructed statistics"""
        if self.first_conv is None:
            raise RuntimeError("First conv layer not found")
        
        weight = self.first_conv.weight.data  # [out, in, k, k]
        weight_spatial_flat = weight.view(weight.size(0), weight.size(1), -1)  # [out, in, k*k]
        
        # Reshape reconstructed statistics back to [out, in, 1]
        rec_means = rec_means_1d.view(kernel_means_shape).to(weight.device)
        rec_vars = rec_vars_1d.view(kernel_means_shape).to(weight.device)
        
        # Clamp reconstructed stats to avoid explosions/NaNs
        EPS = 1e-6
        VAR_MIN, VAR_MAX = 1e-4, 1e2
        MEAN_MIN, MEAN_MAX = -5.0, 5.0
        rec_means = rec_means.clamp(MEAN_MIN, MEAN_MAX)
        rec_vars = rec_vars.clamp(VAR_MIN, VAR_MAX)
        
        # Normalize current weights
        kernel_means_orig = weight_spatial_flat.mean(dim=2, keepdim=True)
        kernel_vars_orig = weight_spatial_flat.var(dim=2, keepdim=True)
        weight_norm = (weight_spatial_flat - kernel_means_orig) / torch.sqrt(kernel_vars_orig + EPS)
        
        # Reconstruct with new statistics (std also clamped)
        std_new = torch.sqrt(torch.abs(rec_vars) + EPS).clamp(1e-3, 10.0)
        weight_new_flat = weight_norm * std_new + rec_means
        
        # Reshape back to original 4D shape
        weight_new = weight_new_flat.view_as(weight)
        
        # Update model weight
        self.first_conv.weight.data = weight_new
    
    def _setup_runtime_loader(self):
        """Setup dataloader for runtime mode (only mode now)."""
        print(f"[RLEnvironment] Setting up runtime dataloader...")
        
        # Create dataset (eval mode, but could be training split)
        dataset = create_dataset(
            self.args.dataset,
            root=self.args.data_dir,
            split=getattr(self.args, 'train_split', 'train'),
            is_training=False,
            class_map=getattr(self.args, 'class_map', ''),
            download=getattr(self.args, 'dataset_download', False),
            batch_size=1, 
        )
        
        # Create loader
        self.train_loader = create_loader(
            dataset,
            input_size=self.data_config['input_size'],
            batch_size=getattr(self.args, 'batch_size', 32), # Use args.batch_size for efficiency
            is_training=False, # No extra augmentation; rely on dataset normalization only
            interpolation=self.data_config['interpolation'],
            mean=self.data_config['mean'],
            std=self.data_config['std'],
            num_workers=getattr(self.args, 'workers', 4),
            distributed=False,
            crop_pct=self.data_config['crop_pct'],
            pin_memory=getattr(self.args, 'pin_mem', False),
            device=self.device,
            use_prefetcher=getattr(self.args, 'prefetcher', True),
        )
        self.runtime_iterator = iter(self.train_loader)
        print(f"[RLEnvironment] Runtime loader initialized. Len: {len(self.train_loader)}")

    def get_total_datapoints(self):
        """Return total number of datapoints (from runtime loader)."""
        # Try to get dataset length from loader
        if hasattr(self, 'train_loader') and self.train_loader is not None:
            if hasattr(self.train_loader, 'dataset') and hasattr(self.train_loader.dataset, '__len__'):
                return len(self.train_loader.dataset)
            else:
                # Fallback estimate
                return len(self.train_loader) * getattr(self.args, 'batch_size', 1)
        return 0

    def reset_norm_params(self):
        """Reset normalization parameters to original state (CRITICAL: called on datapoint change)"""
        self.model.load_state_dict(self.original_state_dict)
    
    def get_first_bn_layer(self):
        """Get first BatchNorm layer"""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                return module, name
        raise RuntimeError("No BatchNorm layer found")
    
    def get_bn_weight(self):
        """Get first BN weight (gamma)"""
        bn_layer, _ = self.get_first_bn_layer()
        return bn_layer.weight.data.clone()
    
    def set_bn_weight_delta(self, delta):
        """Apply delta to first BN weight: weight += delta"""
        bn_layer, _ = self.get_first_bn_layer()
        if not isinstance(delta, torch.Tensor):
            delta = torch.tensor(delta, dtype=bn_layer.weight.dtype, device=bn_layer.weight.device)
        bn_layer.weight.data += delta
    
    def get_state(self):
        """
        Build state vector:
        - AE latent from first conv kernel statistics
        - top-k (k=10) softmax scores (ascending)
        - normalized class positions for those top-k entries (idx / (C-1))
        - corresponding pre-softmax logits in the same order
        """
        if self.ae_model is None:
            raise RuntimeError("AutoEncoder not loaded. Please provide --ae-path")
        
        self.model.eval()
        self.ae_model.eval()
        
        with torch.no_grad():
            # 1) AE latent from conv stats
            stats, _, _, _ = self._get_first_conv_statistics()  # [out*in*2]
            stats_batch = stats.unsqueeze(0).to(self.device)    # [1, out*in*2]
            latent = self.ae_model.encode(stats_batch)          # [1, latent_dim]
            assert latent.shape == (1, self.ae_latent_dim), \
                f"Latent shape mismatch: expected (1, {self.ae_latent_dim}), got {latent.shape}"
            latent_vec = latent.squeeze(0)                      # [latent_dim]
            
            # 2) Model logits / probs on current input
            output = self.model(self.current_input)
            logits = output[0] if isinstance(output, tuple) else output  # [1, C]
            logits_vec = logits.squeeze(0)                                # [C]
            probs = F.softmax(logits_vec, dim=0)                          # [C]
            
            num_classes = logits_vec.shape[0]
            k = min(10, num_classes)
            
            # top-k ascending on probs
            topk_probs, topk_idx = torch.topk(probs, k=k, largest=False, sorted=True)  # [k]
            # normalized positions of the classes (before sorting): idx / (C-1)
            denom = max(num_classes - 1, 1)
            topk_pos = topk_idx.to(logits_vec.dtype) / denom                             # [k]
            # logits aligned to the same indices/order
            topk_logits = logits_vec[topk_idx]                                           # [k]
            
            # --- Simple scale/normalize for stability ---
            tiny_eps = 1e-6  # smaller epsilon as requested
            latent_scaled = latent_vec / (latent_vec.abs().mean() + tiny_eps)
            logits_scaled = torch.tanh(topk_logits)  # squash large logits
            probs_scaled = topk_probs                # already in [0,1]
            pos_scaled = topk_pos                    # already in [0,1]
            
            # concatenate
            state = torch.cat([latent_scaled, probs_scaled, pos_scaled, logits_scaled], dim=0)
        
        return state.detach().cpu()
    
    def compute_reward(self):
        """
        Step-based reward:
        - If prediction matches GT: +1, else -1
        - Scaled by (step_idx - 1); step_idx starts at 1 -> first step reward = 0
        - Cumulative reward is tracked within an episode (reset on new datapoint)
        """
        self.model.eval()
        
        with torch.no_grad():
            output = self.model(self.current_input)
            logits = output[0] if isinstance(output, tuple) else output
            probs = F.softmax(logits, dim=1)
            predicted = torch.argmax(probs, dim=1)
            
            is_correct = (predicted.item() == self.current_gt.item())
        
        # step index starts at 1; first step must give 0 reward
        step_idx = self.step_count + 1
        scale = max(step_idx - 1, 0)
        
        if scale == 0:
            reward_step = 0.0
        else:
            factor = 1.0 if is_correct else -1.0
            reward_step = factor * scale
        
        # accumulate within the datapoint episode
        self.cum_reward += reward_step
        
        return self.cum_reward
    
    def reset(self):
        """
        Reset environment for a new datapoint.
        CRITICAL: Must reset norm params when datapoint changes!
        """
        # Reset norm params (CRITICAL)
        self.reset_norm_params()

        # Runtime mode: fetch next sample from loader
        # Refill buffer if empty
        if len(self.runtime_buffer) == 0:
            try:
                batch = next(self.runtime_iterator)
            except StopIteration:
                # Restart iterator
                print("[RLEnvironment] Runtime iterator exhausted, restarting...")
                self.runtime_iterator = iter(self.train_loader)
                batch = next(self.runtime_iterator)
            
            # Unpack batch (PrefetchLoader returns (input, target))
            # Or regular loader returns (input, target)
            inputs, targets = batch[0], batch[1]
            # Extract label tensor from possibly nested structure (e.g., [labels, aux])
            if isinstance(targets, (list, tuple)):
                if len(targets) > 0 and torch.is_tensor(targets[0]):
                    targets_labels = targets[0]
                else:
                    targets_labels = torch.as_tensor(targets)
            else:
                targets_labels = targets
            
            # Split batch into individual samples
            # inputs: [B, C, H, W], targets: [B]
            B = inputs.size(0)
            for i in range(B):
                self.runtime_buffer.append((
                    inputs[i:i+1].to(self.device), # Keep as [1, C, H, W]
                    targets_labels[i].to(self.device)      # Scalar
                ))
        
        # Pop one sample
        inp, tgt = self.runtime_buffer.pop(0)
        self.current_input = inp
        self.current_gt = tgt
        self.current_class_id = int(tgt.item())
        
        # Reset step tracking
        self.step_count = 0
        self.prev_entropy = None
        self.initial_entropy = None
        self.initial_predicted_class = None
        self.cum_reward = 0.0
        
        # Compute initial entropy (after current_input is set)
        self.model.eval()
        with torch.no_grad():
            output = self.model(self.current_input)
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
            probs = F.softmax(logits, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            predicted = torch.argmax(probs, dim=1)
            
            self.initial_entropy = entropy.item()
            self.prev_entropy = self.initial_entropy
            self.initial_predicted_class = predicted.item()
            self.cum_reward = 0.0
        
        # Get initial state (after current_input is set)
        state = self.get_state()
        
        return state
    
    def step(self, action):
        """
        Apply action (latent delta) to adjust first conv kernel via AutoEncoder.
        Returns (next_state, reward, done, truncated)
        """
        if self.ae_model is None:
            raise RuntimeError("AutoEncoder not loaded. Please provide --ae-path")
        
        self.model.eval()
        self.ae_model.eval()
        
        with torch.no_grad():
            # Get current latent vector
            current_state = self.get_state()  # [latent_dim + 3*k]
            # Slice out only the latent portion (first ae_latent_dim elements)
            current_latent = current_state[:self.ae_latent_dim].to(self.device).unsqueeze(0)  # [1, latent_dim]
            
            # Convert action to tensor if needed
            if not isinstance(action, torch.Tensor):
                action = torch.tensor(action, dtype=current_latent.dtype, device=self.device)
            else:
                action = action.to(self.device)
            
            # Ensure action has correct shape
            if action.dim() == 0:
                action = action.unsqueeze(0)
            if action.dim() == 1 and action.shape[0] != self.ae_latent_dim:
                raise ValueError(f"Action shape mismatch: expected {self.ae_latent_dim}, got {action.shape[0]}")
            if action.dim() == 1:
                action = action.unsqueeze(0)  # [1, latent_dim]
            
            # Shape validation
            assert current_latent.shape == (1, self.ae_latent_dim), \
                f"Current latent shape mismatch: expected (1, {self.ae_latent_dim}), got {current_latent.shape}"
            assert action.shape == (1, self.ae_latent_dim), \
                f"Action shape mismatch: expected (1, {self.ae_latent_dim}), got {action.shape}"
            
            # Apply latent delta: latent_adj = latent + action
            adjusted_latent = current_latent + action  # [1, latent_dim]
            
            # Decode to get reconstructed statistics
            reconstructed_stats = self.ae_model.decode(adjusted_latent)  # [1, out*in*2]
            
            # Get original statistics shapes for reshaping
            _, kernel_means, kernel_vars, weight_spatial_flat = self._get_first_conv_statistics()
            kernel_means_shape = kernel_means.shape  # [out, in, 1]
            expected_stats_dim = 2 * kernel_means.numel()  # out*in*2
            
            # Shape validation
            assert reconstructed_stats.shape[1] == expected_stats_dim, \
                f"Reconstructed stats shape mismatch: expected (1, {expected_stats_dim}), got {reconstructed_stats.shape}"
            
            # Split reconstructed statistics back to means and vars
            split_idx = kernel_means.numel()  # out*in
            rec_means_1d = reconstructed_stats[0, :split_idx]  # [out*in]
            rec_vars_1d = reconstructed_stats[0, split_idx:]   # [out*in]
            
            assert rec_means_1d.shape[0] == split_idx, \
                f"Reconstructed means shape mismatch: expected {split_idx}, got {rec_means_1d.shape[0]}"
            assert rec_vars_1d.shape[0] == split_idx, \
                f"Reconstructed vars shape mismatch: expected {split_idx}, got {rec_vars_1d.shape[0]}"
            
            # Update first conv kernel weights
            self._update_first_conv_from_statistics(
                rec_means_1d, rec_vars_1d, kernel_means_shape, weight_spatial_flat.shape
            )
            
        # Increment step
        self.step_count += 1
        
        # Compute reward
        reward = self.compute_reward()
        
        # Get next state
        next_state = self.get_state()
        
        # Check termination
        done = False
        truncated = (self.step_count >= self.step_count_limit)
        
        return next_state, reward, done, truncated
    
    def get_action_space(self):
        """Get action space (latent delta range)"""
        if self.ae_model is None:
            raise RuntimeError("AutoEncoder not loaded. Please provide --ae-path")
        
        # Use a reasonable range for latent delta
        # Latent vectors are typically in a normalized range, so we use a small scale
        scale_factor = getattr(self.args, 'action_scale_factor', 0.001)  # Smaller default for latent space
        high = scale_factor
        low = -high
        
        action_dim = self.ae_latent_dim
        action_space = spaces.Box(low=low, high=high, shape=(action_dim,), dtype=np.float32)
        
        print(f"[RLEnvironment] Action space (latent delta): dim={action_dim}, range=[{low:.6f}, {high:.6f}]")
        return action_space
    
    def get_state_size(self):
        """Get state size = latent_dim + 3*k (k=10 or num_classes if smaller)"""
        if self.ae_model is None:
            raise RuntimeError("AutoEncoder not loaded. Please provide --ae-path")
        num_classes = int(self.args.num_classes)
        k = min(10, num_classes)
        return self.ae_latent_dim + 3 * k

