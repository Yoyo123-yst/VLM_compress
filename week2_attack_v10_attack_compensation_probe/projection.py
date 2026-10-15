"""Intersection projection for an attack delta and a compensation delta.

Per channel, both deltas and their sum stay inside the uint8 box and the
8-level L-infinity ball. Phase 2 can freeze the attack delta.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

EPS_U = 8


def bounds(clean_u8: np.ndarray, attack_u8: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    clean = clean_u8.astype(np.int16)
    lo = np.maximum(-EPS_U, -clean)
    hi = np.minimum(EPS_U, 255 - clean)
    if attack_u8 is not None:
        attack = attack_u8.astype(np.int16)
        lo = np.maximum(lo, np.maximum(-EPS_U - attack, -clean - attack))
        hi = np.minimum(hi, np.minimum(EPS_U - attack, 255 - clean - attack))
    return lo, hi


def project_int(delta: np.ndarray, clean_u8: np.ndarray, attack_u8: np.ndarray | None = None) -> np.ndarray:
    lo, hi = bounds(clean_u8, attack_u8)
    return np.clip(np.rint(delta), lo, hi).astype(np.int16)


def project_pair_int(attack: np.ndarray, comp: np.ndarray, clean_u8: np.ndarray, freeze_attack: bool) -> tuple[np.ndarray, np.ndarray]:
    """Euclidean projection onto the integer intersection.

    With the attack frozen, only the compensation coordinate moves. That is the
    phase-2 slice. Otherwise both coordinates move to the nearest feasible pair.
    """
    attack = np.rint(attack).astype(np.int16)
    comp = np.rint(comp).astype(np.int16)
    if freeze_attack:
        return attack, project_int(comp, clean_u8, attack)
    lo, hi = bounds(clean_u8)
    best = np.full(attack.shape, np.inf)
    out_a = np.zeros_like(attack)
    out_c = np.zeros_like(comp)
    for ia in range(-EPS_U, EPS_U + 1):
        for ic in range(-EPS_U, EPS_U + 1):
            ok = (ia >= lo) & (ia <= hi) & (ic >= lo) & (ic <= hi) & (ia + ic >= lo) & (ia + ic <= hi)
            dist = (ia - attack) ** 2 + (ic - comp) ** 2
            dist = np.where(ok, dist, np.inf)
            better = dist < best
            out_a = np.where(better, ia, out_a)
            out_c = np.where(better, ic, out_c)
            best = np.where(better, dist, best)
    if not np.isfinite(best).all():
        raise RuntimeError("intersection projection found an empty pixel")
    return out_a.astype(np.int16), out_c.astype(np.int16)


def float_from_uint8(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def uint8_array(page: torch.Tensor) -> np.ndarray:
    array = page.detach().mul(255).round().clamp(0, 255).to(torch.uint8)
    return array.permute(1, 2, 0).cpu().numpy()


def apply_uint8(clean_u8: np.ndarray, delta_u8: np.ndarray) -> Image.Image:
    out = np.clip(clean_u8.astype(np.int16) + delta_u8.astype(np.int16), 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def snap_leaf(delta: torch.Tensor, clean_page: torch.Tensor, attack: torch.Tensor | None, freeze_attack: bool) -> torch.Tensor:
    """Project one leaf in float units of 1/255 and copy the result back."""
    clean_u8 = uint8_array(clean_page)
    values = delta.detach().mul(255).permute(1, 2, 0).cpu().numpy()
    attack_u8 = None if attack is None else attack.detach().mul(255).round().permute(1, 2, 0).cpu().numpy()
    projected = project_int(values, clean_u8, attack_u8 if freeze_attack else None)
    if not freeze_attack:
        projected = project_int(projected, clean_u8)
    snapped = torch.from_numpy(projected.astype(np.float32) / 255.0).permute(2, 0, 1).to(delta.device)
    with torch.no_grad():
        delta.copy_(snapped)
    return delta


def project_attack_leaf(delta: torch.Tensor, clean_page: torch.Tensor) -> None:
    snap_leaf(delta, clean_page, None, False)


def project_comp_leaf(delta: torch.Tensor, clean_page: torch.Tensor, attack: torch.Tensor) -> None:
    snap_leaf(delta, clean_page, attack, True)


def delta_norms(clean_u8: np.ndarray, delta_u8: np.ndarray) -> dict:
    diff = delta_u8.astype(np.float64)
    return {
        "linf_uint8": int(np.max(np.abs(diff))) if diff.size else 0,
        "linf": float(np.max(np.abs(diff)) / 255.0) if diff.size else 0.0,
        "l1": float(np.abs(diff).sum()),
        "l2": float(np.sqrt(np.square(diff).sum())),
    }
