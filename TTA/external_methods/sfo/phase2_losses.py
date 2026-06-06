#!/usr/bin/env python3
"""Inner-loss and guard helpers for Phase-2 coupled LoRA adaptation.

This version is aligned with Phase-1 DirectAugClassifier signals:
  - clean probability from aug head (class 0)
  - dist from dist head
  - z-norm from encoded feature
No FSC centroid is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class InnerLossConfig:
    """Configuration for task-coupled unlabeled inner objective."""

    w_clean_rel: float = 1.0
    w_dist_rel: float = 1.0
    w_znorm_rel: float = 1.0
    w_trust: float = 1e-3
    w_kl: float = 0.0

    p_clean_ref: float = 0.55
    dist_ref: float = 2.0
    znorm_ref: float = 2.0
    rel_mode: str = "relu"  # "relu" or "mse"
    normalize_rel: bool = True
    clean_index: int = 0


@dataclass
class StepGuardConfig:
    """Safety guard for TTA inner-step acceptance."""

    enabled: bool = True
    skip_clean_like: bool = True
    skip_tol: float = 0.05

    max_dist_rise: float = 0.10
    max_znorm_rise: float = 0.10
    max_coeff_norm: float = 1.0

    score_dist_penalty: float = 0.5
    score_znorm_penalty: float = 0.5
    min_score_improve: float = 1e-4


def _rel_term(gap: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "mse":
        return gap.pow(2)
    return F.relu(gap)


def compute_inner_loss(
    aug_logits: torch.Tensor,
    dist_out: torch.Tensor,
    z_flat: torch.Tensor,
    coeff: torch.Tensor,
    cfg: InnerLossConfig,
    clean_dist_ref: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute unlabeled inner objective and diagnostics from Phase-1 outputs."""
    eps = 1e-8
    probs = F.softmax(aug_logits, dim=1)
    p_clean = probs[:, cfg.clean_index]
    dist = dist_out
    znorm = z_flat.norm(dim=1)

    if cfg.normalize_rel:
        clean_gap = (cfg.p_clean_ref - p_clean) / (cfg.p_clean_ref + eps)
        dist_gap = (dist - cfg.dist_ref) / (cfg.dist_ref + eps)
        znorm_gap = (znorm - cfg.znorm_ref) / (cfg.znorm_ref + eps)
    else:
        clean_gap = cfg.p_clean_ref - p_clean
        dist_gap = dist - cfg.dist_ref
        znorm_gap = znorm - cfg.znorm_ref

    loss_clean = _rel_term(clean_gap, cfg.rel_mode).mean()
    loss_dist = F.relu(dist_gap).mean()
    loss_znorm = F.relu(znorm_gap).mean()
    loss_trust = coeff.pow(2).sum()
    loss_kl = torch.tensor(0.0, device=aug_logits.device)
    if cfg.w_kl > 0 and clean_dist_ref is not None:
        p = probs.clamp(min=eps)
        t = clean_dist_ref.unsqueeze(0).expand_as(p).clamp(min=eps)
        loss_kl = (p * (torch.log(p) - torch.log(t))).sum(dim=1).mean()

    total = (
        cfg.w_clean_rel * loss_clean
        + cfg.w_dist_rel * loss_dist
        + cfg.w_znorm_rel * loss_znorm
        + cfg.w_trust * loss_trust
        + cfg.w_kl * loss_kl
    )

    diag = {
        "p_clean": p_clean.mean().detach(),
        "dist": dist.mean().detach(),
        "znorm": znorm.mean().detach(),
        "entropy": (-(probs * torch.log(probs + eps)).sum(dim=1).mean()).detach(),
        "loss_clean_rel": loss_clean.detach(),
        "loss_dist_rel": loss_dist.detach(),
        "loss_znorm_rel": loss_znorm.detach(),
        "loss_trust": loss_trust.detach(),
        "loss_kl": loss_kl.detach(),
        "loss_total": total.detach(),
    }
    return total, diag


def is_clean_like(diag: Dict[str, float], cfg: InnerLossConfig, guard_cfg: StepGuardConfig) -> bool:
    """Return whether a batch is already close to clean references."""
    return (
        diag["p_clean"] >= cfg.p_clean_ref * (1.0 - guard_cfg.skip_tol)
        and diag["dist"] <= cfg.dist_ref * (1.0 + guard_cfg.skip_tol)
        and diag["znorm"] <= cfg.znorm_ref * (1.0 + guard_cfg.skip_tol)
    )


def stability_score(diag: Dict[str, float], cfg: InnerLossConfig, guard_cfg: StepGuardConfig) -> float:
    """Score used by accept/reject policy."""
    eps = 1e-8
    score = float(diag["p_clean"])
    score -= guard_cfg.score_dist_penalty * max(0.0, (float(diag["dist"]) - cfg.dist_ref) / (cfg.dist_ref + eps))
    score -= guard_cfg.score_znorm_penalty * max(0.0, (float(diag["znorm"]) - cfg.znorm_ref) / (cfg.znorm_ref + eps))
    return score


def violates_caps(
    diag: Dict[str, float],
    coeff_norm: float,
    cfg: InnerLossConfig,
    guard_cfg: StepGuardConfig,
) -> bool:
    dist_limit = cfg.dist_ref * (1.0 + guard_cfg.max_dist_rise)
    znorm_limit = cfg.znorm_ref * (1.0 + guard_cfg.max_znorm_rise)
    if float(diag["dist"]) > dist_limit:
        return True
    if float(diag["znorm"]) > znorm_limit:
        return True
    if guard_cfg.max_coeff_norm > 0 and coeff_norm > guard_cfg.max_coeff_norm:
        return True
    return False
