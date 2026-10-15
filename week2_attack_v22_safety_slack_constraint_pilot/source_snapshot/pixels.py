"""Prepare-path pixels. The forward value is the processor tensor."""

from __future__ import annotations

import torch

from week2_pilot.attack import normalized_tiles


def straight_through_pixels(prepared: torch.Tensor, aux_idx: int, page: torch.Tensor) -> torch.Tensor:
    """Return pixels equal to `prepared`, with backward through normalized tiles.

    The processor tensor is the value the model sees. The tile function only
    carries the backward signal.
    """
    surrogate = normalized_tiles(page).to(dtype=prepared.dtype)
    forward = prepared[aux_idx].detach()
    replacement = forward + (surrogate - surrogate.detach())
    pages = []
    for index in range(prepared.shape[0]):
        if index == aux_idx:
            pages.append(replacement)
        else:
            pages.append(prepared[index].detach())
    pixels = torch.stack(pages)
    if not torch.equal(pixels.detach(), prepared.detach()):
        raise RuntimeError("straight-through forward left the processor tensor")
    return pixels
