"""v6b losses. The v6 hinge is not used."""

from __future__ import annotations

import math

import torch

from week2_attack_v4_answer_aligned.loss import answer_margin


def with_eos(tokens: torch.Tensor, eos_id: int) -> torch.Tensor:
    tokens = tokens.view(-1).to(dtype=torch.long)
    if tokens.numel() == 0 or int(tokens[-1]) != int(eos_id):
        eos = torch.tensor([int(eos_id)], dtype=torch.long, device=tokens.device)
        tokens = torch.cat([tokens, eos])
    return tokens


def soft_full_anchor(logits: torch.Tensor, token_ids: torch.Tensor, clean_margin: torch.Tensor) -> torch.Tensor:
    """mean softplus((m_clean - m_adv) / 1). Clean margin is a constant."""
    margin = answer_margin(logits, token_ids)
    if margin.shape != clean_margin.shape:
        raise ValueError(f"margin {tuple(margin.shape)} != clean {tuple(clean_margin.shape)}")
    return torch.nn.functional.softplus(clean_margin.detach() - margin).mean()


def page_soft_mass(scores: torch.Tensor, spans: list[tuple[int, int]]) -> torch.Tensor:
    if scores.ndim != 1:
        raise ValueError("scores must be a vector")
    return torch.stack([scores[start:end].sum() for start, end in spans])


def selection_l1(mass: torch.Tensor, clean_mass: torch.Tensor) -> torch.Tensor:
    return (mass - clean_mass.detach()).abs().sum()


def coefficient(comp_l2: float, term_l2: float) -> float:
    if not math.isfinite(comp_l2) or not math.isfinite(term_l2) or comp_l2 <= 1e-6 or term_l2 <= 1e-6:
        raise ValueError(f"calibration gradient not usable: comp={comp_l2} term={term_l2}")
    return comp_l2 / term_l2


def mechanism_label(ac_wrong: bool, ca_wrong: bool, aa_wrong: bool, valid: bool) -> str:
    if not valid:
        return "UNEXPLAINED_OR_INVALID"
    if ac_wrong and not ca_wrong:
        return "CONTENT_SUFFICIENT"
    if ca_wrong and not ac_wrong:
        return "SELECTION_SUFFICIENT"
    if (not ac_wrong) and (not ca_wrong) and aa_wrong:
        return "INTERACTION_REQUIRED"
    if ac_wrong and ca_wrong:
        return "BOTH_INDEPENDENTLY_SUFFICIENT"
    return "UNEXPLAINED_OR_INVALID"
