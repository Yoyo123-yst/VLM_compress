"""Answer-aligned losses.

The attack objective is the compressed teacher-forcing margin of the clean
reference answer. Lower is weaker support for that answer. aux share is not
an objective. evidence_idx is not an argument of any loss.
"""

from __future__ import annotations

import torch

# Fixed before any v4 attack result. A drop of this size, or a cross from
# positive to non-positive, counts as "margin moved" for SURROGATE_NONFLIP.
MARGIN_DROP = 1.0
LAMBDA_FULL = 1.0


def official_keep_count(n_visual: int, keep_ratio: float) -> int:
    """Same rule as adapters.fastv_select.official_keep_count.

    FastV uses round(n * keep_ratio), with keep_ratio = 1 - prune_rate.
    """
    if n_visual < 1:
        raise ValueError("n_visual must be positive")
    return max(1, min(n_visual, int(round(n_visual * keep_ratio))))


def answer_margin(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """Per-position margin z[y] - max_{v != y} z[v]."""
    if logits.ndim != 2:
        raise ValueError(f"logits must be [T, V], got {tuple(logits.shape)}")
    if token_ids.ndim != 1 or token_ids.shape[0] != logits.shape[0]:
        raise ValueError("token_ids must be [T] aligned with logits")
    index = token_ids.to(device=logits.device, dtype=torch.long).view(-1, 1)
    chosen = logits.gather(1, index).squeeze(1)
    rival_logits = logits.clone()
    rival_logits.scatter_(1, index, float("-inf"))
    rival = rival_logits.max(dim=-1).values
    return chosen - rival


def compression_margin_loss(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """Untruncated mean margin. Minimizing it weakens the reference answer."""
    return answer_margin(logits, token_ids).mean()


def kl_clean_adv(log_p_clean: torch.Tensor, logits_adv: torch.Tensor) -> torch.Tensor:
    """Mean KL(p_clean || p_adv). p_clean is detached."""
    log_q = logits_adv.float().log_softmax(dim=-1)
    log_p = log_p_clean.detach()
    p = log_p.exp()
    return (p * (log_p - log_q)).sum(dim=-1).mean()


def normalized_total(loss_comp: torch.Tensor, loss_full: torch.Tensor, scale_comp: float, scale_full: float) -> torch.Tensor:
    """λ_full = 1 after dividing each term by its frozen δ=0 gradient norm."""
    if scale_comp <= 0 or scale_full <= 0:
        raise ValueError("scales must be positive")
    return loss_comp / scale_comp + LAMBDA_FULL * loss_full / scale_full


def project_capped_simplex(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Euclidean projection onto {p | 0 <= p_i <= 1, sum p = k}.

    Phase-2 diagnostic only. The phase-1 attack does not call this.
    """
    if scores.ndim != 1:
        raise ValueError("scores must be a vector")
    n = int(scores.numel())
    if k < 1 or k > n:
        raise ValueError(f"k={k} outside 1..{n}")
    # Duchi-style bisection on the box-constrained simplex.
    lo = float(scores.min() - 1.0)
    hi = float(scores.max() + 1.0)
    target = float(k)
    for _ in range(80):
        tau = (lo + hi) / 2.0
        projected = torch.clamp(scores - tau, 0.0, 1.0)
        if float(projected.sum()) > target:
            lo = tau
        else:
            hi = tau
    return torch.clamp(scores - hi, 0.0, 1.0)


def quantize_in_ball(clean_uint8: torch.Tensor, adv_float: torch.Tensor, eps_u: int = 8) -> torch.Tensor:
    """Round to uint8 and clamp inside the discrete Linf ball of the original page."""
    raw = adv_float.detach().mul(255).round().clamp(0, 255)
    lo = (clean_uint8 - eps_u).clamp(0, 255)
    hi = (clean_uint8 + eps_u).clamp(0, 255)
    return raw.clamp(lo, hi).to(torch.uint8)


def output_valid(text: str) -> bool:
    return bool(text and text.strip())


def margin_moved(margin_now: float, margin_start: float) -> bool:
    if margin_start > 0.0 and margin_now <= 0.0:
        return True
    return margin_now <= margin_start - MARGIN_DROP


def classify_sample(
    clean_full: float,
    clean_fastv: float,
    adv_full: float,
    adv_fastv: float,
    margin_now: float,
    margin_start: float,
    valid: bool,
) -> str:
    """Mutually exclusive label from a real generate checkpoint."""
    if not valid:
        return "INVALID_OUTPUT"
    if adv_full < 0.5:
        return "FULL_LEAK"
    if clean_full >= 0.5 and clean_fastv >= 0.5 and adv_full >= 0.5 and adv_fastv < 0.5:
        return "SUCCESS_COMP_ONLY"
    if adv_fastv >= 0.5 and margin_moved(margin_now, margin_start):
        return "SURROGATE_NONFLIP"
    return "NO_EFFECT"


def select_best(rows: list[dict]) -> dict:
    """Pick a checkpoint. rows need the keys used below. Invalid rows are ineligible."""
    eligible = [
        row for row in rows
        if row["output_valid"] and float(row["anls_full"]) >= 0.5
    ]
    if not eligible:
        leaked = [row for row in rows if row["output_valid"]]
        pool = leaked or rows
        return min(pool, key=lambda row: int(row["step"]))
    return min(
        eligible,
        key=lambda row: (
            float(row["anls_fastv"]),
            float(row["margin"]),
            float(row["linf"]),
            int(row["step"]),
        ),
    )
