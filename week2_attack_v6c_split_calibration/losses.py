"""v6c losses. B is calibrated. C's beta comes from one fixed probe."""

from __future__ import annotations

import torch

from week2_attack_v6b_soft_full_anchor.soft_loss import (
    page_soft_mass,
    selection_l1,
    soft_full_anchor,
    with_eos,
)

LAMBDA_SOFT = 2.049824826423478
PROBE_DELTA = 1.0 / 255.0


def probe_pages(pages: list[torch.Tensor], aux_idx: int) -> list[torch.Tensor]:
    """Add +1/255 on the auxiliary page only, then clamp."""
    out = [page.detach().clone() for page in pages]
    out[aux_idx] = (out[aux_idx] + PROBE_DELTA).clamp(0, 1)
    return out
