"""Four attack losses plus the shared per-token Full-FDP term.

Each attack loss is its own function. A training step calls exactly one of
them. There is no mode flag inside these functions.
"""

from __future__ import annotations

import torch

from week2_attack_v7b_native_cache_margin_admission.native_cache_margin import hard_margin_row, soft_margin_row
from week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation.losses import decision_preservation

from week2_attack_v20_fourway_content_ablation.selection import CoreSpanUnavailable


def _mean_soft(rows: list[torch.Tensor], token_ids: list[int]) -> torch.Tensor:
    if len(rows) != len(token_ids) or not rows:
        raise RuntimeError(f"attack rows {len(rows)} != tokens {len(token_ids)}")
    return torch.stack([soft_margin_row(row, int(token)) for row, token in zip(rows, token_ids)]).mean()


def _take(rows: list[torch.Tensor], token_ids: list[int], positions: list[int]) -> tuple[list[torch.Tensor], list[int]]:
    if any(index < 0 or index >= len(rows) or index >= len(token_ids) for index in positions):
        raise RuntimeError("selected position is outside the frozen reference")
    return [rows[index] for index in positions], [int(token_ids[index]) for index in positions]


def dynamic_sequence_loss(rows: list[torch.Tensor], generated_ids: list[int]) -> torch.Tensor:
    """L_dynamic on the FastV sequence just generated from the current image."""
    return _mean_soft(rows, [int(token) for token in generated_ids])


def fixed_all_loss(rows: list[torch.Tensor], reference_ids: list[int]) -> torch.Tensor:
    """L_fixed_all on the frozen clean FastV sequence, EOS included."""
    return _mean_soft(rows, [int(token) for token in reference_ids])


def fixed_noeos_loss(rows: list[torch.Tensor], reference_ids: list[int], positions: list[int]) -> torch.Tensor:
    """L_fixed_noEOS on the same frozen sequence after control tokens are removed."""
    selected_rows, selected_ids = _take(rows, reference_ids, positions)
    return _mean_soft(selected_rows, selected_ids)


def core_content_loss(rows: list[torch.Tensor], reference_ids: list[int], positions: list[int]) -> torch.Tensor:
    """L_content on the pre-registered core span only."""
    if not positions:
        raise CoreSpanUnavailable("CORE_SPAN_UNAVAILABLE")
    selected_rows, selected_ids = _take(rows, reference_ids, positions)
    return _mean_soft(selected_rows, selected_ids)


def full_fdp_loss(hard_margins: torch.Tensor, clean_hard: torch.Tensor) -> torch.Tensor:
    """Per-token decision preservation. One softplus term per Full token, then mean."""
    if hard_margins.shape != clean_hard.shape:
        raise RuntimeError("FDP token axis does not match the clean Full reference")
    return decision_preservation(hard_margins, clean_hard)


def selected_hard_margins(rows: list[torch.Tensor], token_ids: list[int], positions: list[int]) -> list[float]:
    values = []
    for index in positions:
        values.append(float(hard_margin_row(rows[index], int(token_ids[index])).detach()))
    return values


def combine(attack: torch.Tensor, fdp: torch.Tensor, lambda_full: float) -> torch.Tensor:
    return attack + float(lambda_full) * fdp
