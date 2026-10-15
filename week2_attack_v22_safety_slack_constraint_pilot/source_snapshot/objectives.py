"""Core attack loss and the two Full protection losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from week2_attack_v7b_native_cache_margin_admission.native_cache_margin import hard_margin_row, soft_margin_row


def core_loss(rows: list[torch.Tensor], token_ids: list[int], positions: list[int]) -> torch.Tensor:
    chosen = [soft_margin_row(rows[index], int(token_ids[index])) for index in positions]
    return torch.stack(chosen).mean()


def hard_margins(rows: list[torch.Tensor], token_ids: list[int]) -> torch.Tensor:
    return torch.stack([hard_margin_row(row, int(token)) for row, token in zip(rows, token_ids)])


def native_fdp(adv_hard: torch.Tensor, clean_hard: torch.Tensor) -> torch.Tensor:
    return F.softplus(clean_hard - adv_hard).mean()


def safety_tau(clean_hard: torch.Tensor) -> torch.Tensor:
    return torch.maximum(torch.full_like(clean_hard, 0.125), 0.5 * clean_hard)


def safety_hinge(adv_hard: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    return torch.relu(tau - adv_hard).mean()
