#!/usr/bin/env python3
"""Episode-mode scheduling utilities for Phase-2 coupled training/TTA."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import List, Optional

import torch


def _normalize_mode_name(mode: str) -> str:
    mode = mode.strip().lower()
    if mode in {"continue", "continual", "c"}:
        return "continual"
    if mode in {"static", "s"}:
        return "static"
    if mode == "mixed":
        return "mixed"
    raise ValueError(f"Unsupported mode: {mode}")


def parse_episode_pattern(pattern: str) -> List[str]:
    """Parse pattern like ``S,S,C,C,S`` into normalized mode names."""
    if not pattern:
        return []
    items = [p.strip() for p in pattern.split(",") if p.strip()]
    parsed = []
    for p in items:
        parsed.append(_normalize_mode_name(p))
    return parsed


@dataclass
class EpisodeSchedulerConfig:
    mode: str = "mixed"
    ratio_start: float = 0.2
    ratio_end: float = 0.5
    total_epochs: int = 100
    pattern: str = ""
    seed: int = 42


class EpisodeModeScheduler:
    """Select episode mode (static/continual) per step."""

    def __init__(self, cfg: EpisodeSchedulerConfig):
        self.cfg = cfg
        self.mode = _normalize_mode_name(cfg.mode)
        self.pattern = parse_episode_pattern(cfg.pattern)
        self.rng = random.Random(cfg.seed)

    def continual_ratio(self, epoch: int) -> float:
        if self.cfg.total_epochs <= 1:
            return float(self.cfg.ratio_end)
        t = max(0.0, min(1.0, epoch / float(self.cfg.total_epochs - 1)))
        r = self.cfg.ratio_start + (self.cfg.ratio_end - self.cfg.ratio_start) * t
        return float(max(0.0, min(1.0, r)))

    def mode_for_step(self, epoch: int, global_step: int) -> str:
        if self.mode in {"static", "continual"}:
            return self.mode
        if self.pattern:
            return self.pattern[global_step % len(self.pattern)]
        return "continual" if self.rng.random() < self.continual_ratio(epoch) else "static"


@dataclass
class CarryPolicy:
    """Policy for carrying adaptation coefficients between episodes."""

    decay: float = 1.0
    reset_every: int = 0
    scope: str = "within_domain"

    def __post_init__(self):
        self.scope = self.scope.strip().lower()
        if self.scope not in {"none", "within_episode", "within_domain"}:
            raise ValueError(f"Unsupported carry scope: {self.scope}")


class CarryController:
    """Apply decay/reset logic to carry state in continual mode."""

    def __init__(self, policy: CarryPolicy):
        self.policy = policy

    def should_force_reset(self, global_step: int) -> bool:
        return self.policy.reset_every > 0 and ((global_step + 1) % self.policy.reset_every == 0)

    def next_state(
        self,
        mode: str,
        global_step: int,
        prev_state: torch.Tensor,
        current_state: torch.Tensor,
    ) -> torch.Tensor:
        mode = _normalize_mode_name(mode)
        if mode != "continual" or self.policy.scope == "none":
            return torch.zeros_like(prev_state)
        if self.should_force_reset(global_step):
            return torch.zeros_like(prev_state)
        return current_state.detach() * float(self.policy.decay)

