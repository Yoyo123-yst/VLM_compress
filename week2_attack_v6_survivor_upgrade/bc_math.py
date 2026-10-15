"""Frozen B/C losses. No coefficient is chosen here."""

from __future__ import annotations

import torch

from week2_attack_v4_answer_aligned.loss import answer_margin


def with_eos(tokens: torch.Tensor, eos_id: int) -> torch.Tensor:
    """Keep every visible greedy token and exactly one trailing EOS."""
    tokens = tokens.view(-1).to(dtype=torch.long)
    if tokens.numel() == 0 or int(tokens[-1]) != int(eos_id):
        eos = torch.tensor([int(eos_id)], dtype=torch.long)
        tokens = torch.cat([tokens, eos])
    return tokens


def kappa_from_clean_margin(clean_margin: torch.Tensor) -> torch.Tensor:
    """kappa_t = max(0, min(m_clean_t, 1)). Clean state only."""
    return torch.clamp(clean_margin.detach(), 0.0, 1.0)


def sequence_hinge(logits: torch.Tensor, token_ids: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    """Mean relu(kappa_t - m_adv_t), including the EOS position."""
    margin = answer_margin(logits, token_ids)
    if margin.shape != kappa.shape:
        raise ValueError(f"margin {tuple(margin.shape)} != kappa {tuple(kappa.shape)}")
    return torch.relu(kappa - margin).mean()


def page_soft_mass(scores: torch.Tensor, spans: list[tuple[int, int]]) -> torch.Tensor:
    """Sum of real last-query attention mass on each page's visual tokens."""
    if scores.ndim != 1:
        raise ValueError("scores must be a vector")
    masses = [scores[start:end].sum() for start, end in spans]
    return torch.stack(masses)


def selection_l1(mass: torch.Tensor, clean_mass: torch.Tensor) -> torch.Tensor:
    """One page-level stability term. Hard keep-set equality is not used."""
    return (mass - clean_mass.detach()).abs().sum()


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


def scale_from_grad(comp_l2: float, term_l2: float) -> float:
    if comp_l2 < 1e-6 or term_l2 < 1e-6:
        raise ValueError("gradient L2 below 1e-6; the one allowed fallback must be used first")
    return comp_l2 / term_l2
