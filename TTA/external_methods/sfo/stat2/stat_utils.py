# stat_utils.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


# -----------------------
# Module path resolution
# -----------------------

def _get_child_module(m: nn.Module, key: str) -> nn.Module:
    """
    key can be:
      - attribute name: 'layer1'
      - numeric index for Sequential/ModuleList: '0'
    """
    if key.isdigit():
        idx = int(key)
        if isinstance(m, (nn.Sequential, nn.ModuleList)):
            return m[idx]
        # timm sometimes stores blocks in lists but not ModuleList; still handle
        children = list(m.children())
        return children[idx]
    if hasattr(m, key):
        return getattr(m, key)
    raise KeyError(f"Cannot resolve '{key}' in module {m.__class__.__name__}")


def get_module_by_path(model: nn.Module, path: str) -> nn.Module:
    """
    Resolve a dotted path like:
      - 'maxpool'
      - 'patch_embed'
      - 'layer1.0.conv1'
      - 'blocks.11' (ViT blocks)
    """
    if path == "" or path is None:
        raise ValueError("hook path is empty")
    cur = model
    for part in path.split("."):
        cur = _get_child_module(cur, part)
    return cur


# -----------------------
# Hook capture
# -----------------------

@dataclass
class HookCapture:
    tensor: Optional[torch.Tensor] = None

    def __call__(self, module, inp, out):
        self.tensor = out.detach()


def attach_hook(model: nn.Module, hook_path: str) -> Tuple[HookCapture, torch.utils.hooks.RemovableHandle]:
    cap = HookCapture()
    target = get_module_by_path(model, hook_path)
    handle = target.register_forward_hook(cap)
    return cap, handle


# -----------------------
# Feature formatting
# -----------------------

@torch.no_grad()
def to_bcn(t: torch.Tensor) -> torch.Tensor:
    """
    Convert captured tensor to (B,C,N).
    Supported:
      - CNN feature map: (B,C,H,W) -> (B,C,N)
      - ViT patch tokens: (B,N,C) -> (B,C,N)
      - ViT tokens w/ CLS: (B,1+N,C) -> (B,C,1+N) (you can later drop CLS if desired)
    """
    if t.dim() == 4:
        B, C, H, W = t.shape
        return t.view(B, C, H * W)
    if t.dim() == 3:
        # (B,N,C) -> (B,C,N)
        return t.transpose(1, 2).contiguous()
    raise ValueError(f"Unsupported tensor shape for to_bcn: {tuple(t.shape)}")


# -----------------------
# Stats: mu, var, Z, R
# -----------------------

@torch.no_grad()
def mu_var_Z(X: torch.Tensor, eps: float = 1e-5):
    """
    X: (B,C,N)
    mu: (B,C,1)
    var: (B,C,1)   (population var over N)
    Z: (B,C,N)     standardized like IN (no affine)
    """
    mu = X.mean(dim=-1, keepdim=True)
    var = ((X - mu) ** 2).mean(dim=-1, keepdim=True)
    Z = (X - mu) / torch.sqrt(var + eps)
    return mu, var, Z


@torch.no_grad()
def corr_from_Z(Z: torch.Tensor) -> torch.Tensor:
    """
    Z: (B,C,N) -> R: (B,C,C)
    R_b = (1/N) Z Z^T
    """
    B, C, N = Z.shape
    return torch.bmm(Z, Z.transpose(1, 2)) / float(N)


# -----------------------
# Running mean updater
# -----------------------

@torch.no_grad()
def running_update(running: Optional[torch.Tensor],
                   batch_mean: torch.Tensor,
                   n_before: int,
                   B: int) -> Tuple[torch.Tensor, int]:
    """
    running: tensor (no batch dim), e.g. (1,C,1) or (C,C)
    batch_mean: tensor same shape as running
    """
    if running is None:
        return batch_mean.clone().float(), n_before + B
    n_after = n_before + B
    running = (running * n_before + batch_mean.float() * B) / float(n_after)
    return running, n_after


# -----------------------
# Full step for mu/var/R
# -----------------------

@torch.no_grad()
def compute_batch_stats_from_captured(captured: torch.Tensor, eps: float = 1e-5, drop_cls: bool = False):
    """
    captured: output of hook
    returns:
      mu_b: (B,C,1)
      var_b: (B,C,1)
      R_b: (B,C,C)
    """
    X = to_bcn(captured)  # (B,C,N)
    if drop_cls and X.size(-1) > 1:
        # If tensor includes CLS as first token, it becomes N axis. For patch tokens:
        # captured tokens are often (B, 1+N, C) at later hooks. Here we drop first token.
        X = X[:, :, 1:]

    mu_b, var_b, Z = mu_var_Z(X, eps=eps)
    R_b = corr_from_Z(Z)
    return mu_b, var_b, R_b