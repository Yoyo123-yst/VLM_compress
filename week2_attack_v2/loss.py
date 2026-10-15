"""v2 losses for the aux-page FastV attack.

L_share is a one-sided squared deficit against target share q*.
The raw count deficit is divided by the keep count so the term stays on a
share scale; otherwise its square is ~1e6 and λ in {0.5, 1, 2, 5} cannot
balance KL(p_clean || p_adv).
"""

from __future__ import annotations

import math

import torch


def ceil_keep_count(n_visual: int, ratio: float) -> int:
    return max(1, min(n_visual, math.ceil(ratio * n_visual)))


def share_target_loss(
    visual_scores: torch.Tensor,
    aux_positions: list[int],
    keep_k: int,
    temperature: float,
    q_star: float = 0.6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One-sided (relu(q* - soft_aux_share))^2.

    soft_aux_share = sum_{j in aux} sigmoid((a_j - τ) / T) / keep_k.
    τ is the ⌈rN⌉-th largest visual score, detached.
    Returns (loss, soft_share).
    """
    if keep_k < 1 or keep_k > visual_scores.numel():
        raise ValueError(f"keep_k {keep_k} outside 1..{visual_scores.numel()}")
    tau = visual_scores.topk(keep_k).values[-1].detach()
    soft = torch.sigmoid((visual_scores[aux_positions] - tau) / temperature).sum()
    soft_share = soft / keep_k
    deficit = torch.relu(q_star - soft_share)
    return deficit * deficit, soft_share


def kl_clean_adv(log_p_clean: torch.Tensor, logits_adv: torch.Tensor) -> torch.Tensor:
    """Mean KL(p_clean || p_adv) over answer positions. p_clean is detached."""
    log_q = logits_adv.float().log_softmax(dim=-1)
    p = log_p_clean.detach().exp()
    return (p * (log_p_clean.detach() - log_q)).sum(dim=-1).mean()
