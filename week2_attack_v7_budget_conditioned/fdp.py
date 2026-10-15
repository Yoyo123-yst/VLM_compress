"""Full-path decision-cone preservation.

Soft margin is the attack loss. Hard margin is the certificate only.
tau_vocab, xi, and epsilon_margin are constants, not per-sample values.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from week2_attack_v7_budget_conditioned.constants import EPSILON_MARGIN, TAU_VOCAB, XI


def hard_margin(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """z[y] - max_{v != y} z[v], one value per token including EOS."""
    if logits.ndim != 2:
        raise ValueError(f"logits must be [T, V], got {tuple(logits.shape)}")
    if token_ids.ndim != 1 or int(token_ids.shape[0]) != int(logits.shape[0]):
        raise ValueError("token_ids must be [T] aligned with logits")
    index = token_ids.to(device=logits.device, dtype=torch.long).view(-1, 1)
    chosen = logits.gather(1, index).squeeze(1)
    rival = logits.clone()
    rival.scatter_(1, index, float("-inf"))
    return chosen - rival.max(dim=-1).values


def soft_margin(logits: torch.Tensor, token_ids: torch.Tensor, tau_vocab: float = TAU_VOCAB) -> torch.Tensor:
    """z[y] - tau * logsumexp_{v != y}(z[v] / tau)."""
    if tau_vocab <= 0:
        raise ValueError("tau_vocab must be positive")
    if logits.ndim != 2:
        raise ValueError(f"logits must be [T, V], got {tuple(logits.shape)}")
    if token_ids.ndim != 1 or int(token_ids.shape[0]) != int(logits.shape[0]):
        raise ValueError("token_ids must be [T] aligned with logits")
    index = token_ids.to(device=logits.device, dtype=torch.long).view(-1, 1)
    chosen = logits.gather(1, index).squeeze(1)
    rival = logits.clone()
    rival.scatter_(1, index, float("-inf"))
    return chosen - tau_vocab * torch.logsumexp(rival / tau_vocab, dim=-1)


def bar_soft(clean_soft: torch.Tensor, xi: float = XI, epsilon_margin: float = EPSILON_MARGIN) -> torch.Tensor:
    """One-sided threshold. Never exceeds the clean soft margin."""
    clean = clean_soft.detach()
    lowered = torch.maximum(clean - xi, torch.full_like(clean, epsilon_margin))
    return torch.minimum(clean, lowered)


def fdp_loss(
    clean_soft: torch.Tensor,
    adv_soft: torch.Tensor,
    xi: float = XI,
    epsilon_margin: float = EPSILON_MARGIN,
) -> torch.Tensor:
    if clean_soft.shape != adv_soft.shape:
        raise ValueError(f"clean {tuple(clean_soft.shape)} != adv {tuple(adv_soft.shape)}")
    if clean_soft.numel() < 1:
        raise ValueError("FDP requires at least one token, including EOS")
    bar = bar_soft(clean_soft, xi, epsilon_margin)
    if torch.any(bar > clean_soft.detach() + 1e-6):
        raise RuntimeError("bar_a_soft exceeds the clean soft margin")
    return F.softplus(bar - adv_soft).mean()


def certificate(clean_hard: torch.Tensor, epsilon_margin: float = EPSILON_MARGIN) -> str:
    if torch.any(clean_hard <= epsilon_margin):
        return "FULL_CONE_CERTIFICATE_UNAVAILABLE"
    return "FULL_CONE_CERTIFICATE_AVAILABLE"


def greedy_tokens(logits: torch.Tensor) -> torch.Tensor:
    return logits.argmax(dim=-1)


def sequence_margin(token_margins: torch.Tensor) -> torch.Tensor:
    """Pre-registered aggregate: mean over the clean Full tokens, including EOS."""
    return token_margins.mean()
