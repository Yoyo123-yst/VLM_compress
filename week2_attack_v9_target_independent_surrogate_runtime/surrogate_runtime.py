"""Target-independent spatial surrogates on the frozen 768 LLaVA-OneVision layout.

These policies are not official compressors. They never read FastV.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch

PAGE_TOKENS = 3699
BASE = 729
BASE_SIDE = 27
HIGH_SIDE = 54
HIGH_STRIDE = 55
RATIOS = (0.25, 0.35)
GROUP_WIDTH = {0.25: 4, 0.35: 3}
BLOCK_SEED = 2026101109


@dataclass(frozen=True)
class PagePlan:
    source_local: list[int]
    groups: list[list[int]]
    merged: bool
    dropped: bool


def _require_ratio(ratio: float) -> float:
    if ratio not in RATIOS:
        raise ValueError(f"ratio {ratio} is outside the frozen set {RATIOS}")
    return float(ratio)


def _raster(side: int) -> list[tuple[int, int]]:
    return [(row, col) for row in range(side) for col in range(side)]


def _uniform_raster(side: int, ratio: float) -> list[int]:
    n = side * side
    k = max(1, int(round(n * ratio)))
    chosen = [(index * n) // k for index in range(k)]
    if len(chosen) != len(set(chosen)) or any(index < 0 or index >= n for index in chosen):
        raise RuntimeError("uniform raster indices are not a valid subset")
    return chosen


def _high_local(row: int, col: int) -> int:
    return BASE + row * HIGH_STRIDE + col


def _newline_local(row: int) -> int:
    return BASE + row * HIGH_STRIDE + HIGH_SIDE


def spatial_strided_drop(ratio: float) -> PagePlan:
    """Keep a fixed raster lattice. The rule uses only coordinates."""
    ratio = _require_ratio(ratio)
    base_keep = _uniform_raster(BASE_SIDE, ratio)
    high_keep = _uniform_raster(HIGH_SIDE, ratio)
    source: list[int] = list(base_keep)
    rows_kept = {index // HIGH_SIDE for index in high_keep}
    for row in range(HIGH_SIDE):
        for index in high_keep:
            if index // HIGH_SIDE == row:
                source.append(_high_local(row, index % HIGH_SIDE))
        if row in rows_kept:
            source.append(_newline_local(row))
    groups = [[index] for index in source]
    return PagePlan(source, groups, merged=False, dropped=True)


def local_spatial_merge(ratio: float) -> PagePlan:
    """Mean-pool fixed horizontal neighborhoods. No score is read."""
    ratio = _require_ratio(ratio)
    width = GROUP_WIDTH[ratio]
    source: list[int] = []
    groups: list[list[int]] = []

    def add_row(local_of, side: int) -> None:
        for row in range(side):
            col = 0
            while col < side:
                members = [local_of(row, col + offset) for offset in range(width) if col + offset < side]
                groups.append(members)
                source.append(members[0])
                col += width

    add_row(lambda row, col: row * BASE_SIDE + col, BASE_SIDE)
    for row in range(HIGH_SIDE):
        col = 0
        while col < HIGH_SIDE:
            members = [_high_local(row, col + offset) for offset in range(width) if col + offset < HIGH_SIDE]
            groups.append(members)
            source.append(members[0])
            col += width
        groups.append([_newline_local(row)])
        source.append(_newline_local(row))
    return PagePlan(source, groups, merged=True, dropped=False)


def seeded_block_drop(ratio: float, page_index: int, seed: int = BLOCK_SEED) -> PagePlan:
    """Drop whole 2x2 blocks from a pre-registered seed. Structured augmentation only."""
    ratio = _require_ratio(ratio)
    threshold = int(round(ratio * 2**32))

    def keep(grid: str, brow: int, bcol: int) -> bool:
        raw = f"{seed}|{page_index}|{grid}|{brow}|{bcol}".encode()
        value = int(hashlib.sha256(raw).hexdigest()[:8], 16)
        return value < threshold

    source: list[int] = []
    for brow in range((BASE_SIDE + 1) // 2):
        for bcol in range((BASE_SIDE + 1) // 2):
            if not keep("base", brow, bcol):
                continue
            for row in range(brow * 2, min(BASE_SIDE, brow * 2 + 2)):
                for col in range(bcol * 2, min(BASE_SIDE, bcol * 2 + 2)):
                    source.append(row * BASE_SIDE + col)
    rows_kept: set[int] = set()
    high_source: list[int] = []
    for brow in range((HIGH_SIDE + 1) // 2):
        for bcol in range((HIGH_SIDE + 1) // 2):
            if not keep("high", brow, bcol):
                continue
            for row in range(brow * 2, min(HIGH_SIDE, brow * 2 + 2)):
                for col in range(bcol * 2, min(HIGH_SIDE, bcol * 2 + 2)):
                    high_source.append(_high_local(row, col))
                    rows_kept.add(row)
    ordered = list(source)
    for row in range(HIGH_SIDE):
        ordered.extend(index for index in high_source if BASE <= index and (index - BASE) // HIGH_STRIDE == row)
        if row in rows_kept:
            ordered.append(_newline_local(row))
    if not ordered:
        raise RuntimeError("seeded block drop removed the whole page")
    return PagePlan(ordered, [[index] for index in ordered], merged=False, dropped=True)


def apply_plan(features: torch.Tensor, plan: PagePlan) -> torch.Tensor:
    if features.shape[0] != PAGE_TOKENS:
        raise RuntimeError(f"page features {features.shape[0]} != frozen {PAGE_TOKENS}")
    parts = []
    for group in plan.groups:
        gathered = features.index_select(0, torch.tensor(group, device=features.device))
        parts.append(gathered.mean(dim=0) if plan.merged else gathered[0])
    return torch.stack(parts, dim=0)


def index_hash(per_page: list[list[int]]) -> str:
    raw = ";".join(",".join(str(index) for index in page) for page in per_page).encode()
    return hashlib.sha256(raw).hexdigest()


POLICIES = {
    "SpatialStridedDrop": spatial_strided_drop,
    "LocalSpatialMerge": local_spatial_merge,
    "SeededBlockDrop": seeded_block_drop,
}
