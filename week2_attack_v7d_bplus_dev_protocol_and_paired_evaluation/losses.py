"""Frozen B-Native, native-FDP, and B+ losses. No model calls."""

from __future__ import annotations

import torch
import torch.nn.functional as F

XI = 1.0
ETA = 1e-3
TRAIN_RATIOS = (0.25, 0.35)


def decision_floor(clean_hard: torch.Tensor) -> torch.Tensor:
    lowered = torch.clamp(clean_hard - XI, min=ETA)
    return torch.minimum(clean_hard, lowered)


def soft_anchor(soft_margin: torch.Tensor, clean_hard: torch.Tensor) -> torch.Tensor:
    return F.softplus(decision_floor(clean_hard) - soft_margin).mean()


def decision_preservation(hard_margin: torch.Tensor, clean_hard: torch.Tensor) -> torch.Tensor:
    return F.softplus(clean_hard - hard_margin).mean()


def compressed_loss(fast_means: list[torch.Tensor]) -> torch.Tensor:
    if len(fast_means) != len(TRAIN_RATIOS):
        raise ValueError("compressed loss is only the train ratios 0.25 and 0.35")
    return torch.stack(fast_means).mean()


def conditional_gap(m_full: torch.Tensor, fast_means: list[torch.Tensor]) -> torch.Tensor:
    return -(m_full - compressed_loss(fast_means))


def method_loss(name: str, compressed: torch.Tensor, anchor: torch.Tensor, fdp: torch.Tensor, gap: torch.Tensor, lambda_full: float, lambda_gap: float) -> torch.Tensor:
    if name == "B-Native":
        return compressed + lambda_full * anchor
    if name == "native-FDP":
        return compressed + lambda_full * fdp
    if name == "B+":
        return compressed + lambda_full * fdp + lambda_gap * gap
    raise ValueError(name)
