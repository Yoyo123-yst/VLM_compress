"""Coordinate-only stochastic bottlenecks. Not an official compressor."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import torch

BASE_SIDE = 27
HIGH_SIDE = 54
BASE = BASE_SIDE * BASE_SIDE
HIGH_STRIDE = HIGH_SIDE + 1
PAGE_TOKENS = BASE + HIGH_SIDE * HIGH_STRIDE


@dataclass
class PagePlan:
    family: str
    ratio: float
    offset: int
    source_local: list[int]
    groups: list[list[int]]
    page_in: int
    page_out: int

    def rule_hash(self) -> str:
        payload = json.dumps(
            {"family": self.family, "ratio": self.ratio, "offset": self.offset, "groups": self.groups},
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()


def offset_of(sample_id: str, step: int, seed: int) -> int:
    raw = f"{seed}|{sample_id}|{step}".encode()
    return int(hashlib.sha256(raw).hexdigest()[:8], 16)


def _keep_count(n: int, ratio: float) -> int:
    if not 0.10 <= ratio <= 0.60:
        raise ValueError(f"retention {ratio} is outside Uniform(0.10, 0.60)")
    count = int(round(n * ratio))
    return max(1, min(n - 1, count))


def _strided(n: int, ratio: float, offset: int) -> list[int]:
    count = _keep_count(n, ratio)
    start = offset % n
    step = n / count
    chosen = []
    used = set()
    for index in range(count):
        slot = int((start + index * step) % n)
        while slot in used:
            slot = (slot + 1) % n
        used.add(slot)
        chosen.append(slot)
    return sorted(chosen)


def _high_local(row: int, col: int) -> int:
    return BASE + row * HIGH_STRIDE + col


def _newline(row: int) -> int:
    return BASE + row * HIGH_STRIDE + HIGH_SIDE


def spatial_strided_drop(ratio: float, offset: int) -> PagePlan:
    base_keep = _strided(BASE, ratio, offset)
    high_keep = _strided(HIGH_SIDE * HIGH_SIDE, ratio, offset)
    source = list(base_keep)
    groups = [[index] for index in base_keep]
    rows = {index // HIGH_SIDE for index in high_keep}
    for row in range(HIGH_SIDE):
        for index in high_keep:
            if index // HIGH_SIDE == row:
                local = _high_local(row, index % HIGH_SIDE)
                source.append(local)
                groups.append([local])
        if row in rows:
            local = _newline(row)
            source.append(local)
            groups.append([local])
    return PagePlan("SpatialStridedDrop", ratio, offset, source, groups, PAGE_TOKENS, len(source))


def local_spatial_merge(ratio: float, offset: int) -> PagePlan:
    width = max(2, int(round(1.0 / ratio)))
    source: list[int] = []
    groups: list[list[int]] = []

    def add(local_of, side: int) -> None:
        shift = offset % side
        for row in range(side):
            col = 0
            while col < side:
                members = []
                for delta in range(width):
                    src = col + delta
                    if src >= side:
                        break
                    members.append(local_of(row, (src + shift) % side))
                groups.append(members)
                source.append(members[0])
                col += width

    add(lambda row, col: row * BASE_SIDE + col, BASE_SIDE)
    for row in range(HIGH_SIDE):
        col = 0
        shift = offset % HIGH_SIDE
        while col < HIGH_SIDE:
            members = []
            for delta in range(width):
                src = col + delta
                if src >= HIGH_SIDE:
                    break
                members.append(_high_local(row, (src + shift) % HIGH_SIDE))
            groups.append(members)
            source.append(members[0])
            col += width
        newline = _newline(row)
        groups.append([newline])
        source.append(newline)
    return PagePlan("LocalSpatialMerge", ratio, offset, source, groups, PAGE_TOKENS, len(source))


FAMILIES = {
    "SpatialStridedDrop": spatial_strided_drop,
    "LocalSpatialMerge": local_spatial_merge,
}


def plan_for(family: str, ratio: float, sample_id: str, step: int, seed: int) -> PagePlan:
    return FAMILIES[family](ratio, offset_of(sample_id, step, seed))


def apply_plan(features: torch.Tensor, plan: PagePlan) -> torch.Tensor:
    if features.ndim != 2 or features.shape[0] != PAGE_TOKENS:
        raise RuntimeError(f"page features {tuple(features.shape)} are not {PAGE_TOKENS}")
    rows = []
    for group in plan.groups:
        index = torch.tensor(group, device=features.device)
        rows.append(features.index_select(0, index).mean(dim=0))
    return torch.stack(rows, dim=0)


def pack_sequence(model, input_ids: torch.Tensor, spans: list[tuple[int, int]], pages: list[torch.Tensor]) -> torch.Tensor:
    embed = model.get_input_embeddings()
    pieces = []
    cursor = 0
    for (start, end), feats in zip(spans, pages):
        if start > cursor:
            pieces.append(embed(input_ids[:, cursor:start]))
        pieces.append(feats.unsqueeze(0).to(dtype=pieces[0].dtype if pieces else feats.dtype))
        cursor = end
    if cursor < input_ids.shape[1]:
        pieces.append(embed(input_ids[:, cursor:]))
    packed = torch.cat(pieces, dim=1)
    text_kept = input_ids.shape[1] - sum(end - start for start, end in spans)
    visual = sum(page.shape[0] for page in pages)
    if packed.shape[1] != text_kept + visual:
        raise RuntimeError("packed length lost text or crossed a page boundary")
    return packed
