"""Coordinate-only multiscale pools. These are not official compressors."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

BASE = 729
HIGH = 54
PAGE_TOKENS = 3699
RATIOS = (0.25, 0.35, 0.50)
NOT_AN_OFFICIAL_COMPRESSOR = True


def bin_bounds(index: int, out_size: int, in_size: int = HIGH) -> tuple[int, int]:
    start = math.floor(index * in_size / out_size)
    end = math.ceil((index + 1) * in_size / out_size)
    if start < 0 or end > in_size or start >= end:
        raise RuntimeError(f"pool bin {(start, end)} is outside 0..{in_size}")
    return start, end


def choose_grid(ratio: float) -> dict:
    if ratio not in RATIOS:
        raise ValueError(f"{ratio} is outside the frozen ratio set")
    target = round(ratio * PAGE_TOKENS)
    candidates = []
    for height in range(1, HIGH + 1):
        for width in range(1, HIGH + 1):
            count = BASE + height * (width + 1)
            ratio_error = abs(count / PAGE_TOKENS - ratio)
            if ratio_error > 0.01 + 1e-12:
                continue
            candidates.append((abs(count - target), abs(height / width - 1.0), height, width, count, ratio_error))
    if not candidates:
        raise RuntimeError(f"no grid within 1% of ratio {ratio}")
    candidates.sort()
    deviation, aspect, height, width, count, ratio_error = candidates[0]
    return {
        "ratio": ratio,
        "target": target,
        "height": height,
        "width": width,
        "tokens": count,
        "deviation": deviation,
        "aspect_gap": aspect,
        "ratio_error": ratio_error,
    }


FROZEN_GRIDS = {ratio: choose_grid(ratio) for ratio in RATIOS}


def page_shift(seed: int, sample_id: str, step: int, offset_id: int, page_index: int) -> tuple[int, int]:
    raw = f"{seed}|{sample_id}|{step}|{offset_id}|{page_index}".encode()
    value = int(hashlib.sha256(raw).hexdigest()[:8], 16)
    return value % HIGH, (value // HIGH) % HIGH


@dataclass
class PagePlan:
    family: str
    ratio: float
    height: int
    width: int
    shift_row: int
    shift_col: int
    page_index: int
    groups: list[list[int]]
    page_in: int
    page_out: int

    def rule_hash(self) -> str:
        payload = json.dumps(
            {
                "family": self.family,
                "ratio": self.ratio,
                "height": self.height,
                "width": self.width,
                "shift_row": self.shift_row,
                "shift_col": self.shift_col,
                "page_index": self.page_index,
                "groups": self.groups,
            },
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()


def _groups(height: int, width: int, shift_row: int, shift_col: int) -> list[list[int]]:
    groups = [[index] for index in range(BASE)]
    covered = []
    for out_row in range(height):
        row_start, row_end = bin_bounds(out_row, height)
        source_rows = [((row - shift_row) % HIGH) for row in range(row_start, row_end)]
        for out_col in range(width):
            col_start, col_end = bin_bounds(out_col, width)
            members = []
            for row in range(row_start, row_end):
                for col in range(col_start, col_end):
                    source_row = (row - shift_row) % HIGH
                    source_col = (col - shift_col) % HIGH
                    members.append(BASE + source_row * (HIGH + 1) + source_col)
            groups.append(members)
            covered.extend(members)
        groups.append([BASE + source_row * (HIGH + 1) + HIGH for source_row in source_rows])
    high_ids = {BASE + row * (HIGH + 1) + col for row in range(HIGH) for col in range(HIGH)}
    got = set(covered)
    if got != high_ids or any(index < BASE or index >= PAGE_TOKENS for index in covered):
        raise RuntimeError("high-resolution bins leave the page or omit a cell")
    return groups


def make_plan(family: str, ratio: float, sample_id: str, step: int, offset_id: int, page_index: int, seed: int) -> PagePlan:
    grid = FROZEN_GRIDS[ratio]
    if family == "BasePreservingAdaptivePool":
        shift = (0, 0)
    elif family == "BasePreservingJitteredPool":
        shift = page_shift(seed, sample_id, step, offset_id, page_index)
    else:
        raise ValueError(family)
    groups = _groups(grid["height"], grid["width"], shift[0], shift[1])
    return PagePlan(family, ratio, grid["height"], grid["width"], shift[0], shift[1], page_index, groups, PAGE_TOKENS, grid["tokens"])


def apply_plan(features: torch.Tensor, plan: PagePlan) -> torch.Tensor:
    if features.ndim != 2 or features.shape[0] != PAGE_TOKENS:
        raise RuntimeError(f"expected [{PAGE_TOKENS}, D], got {tuple(features.shape)}")
    base = features[:BASE]
    rows = features[BASE:].reshape(HIGH, HIGH + 1, features.shape[-1])
    grid = rows[:, :HIGH, :]
    newlines = rows[:, HIGH, :]
    if plan.shift_row or plan.shift_col:
        grid = torch.roll(grid, shifts=(plan.shift_row, plan.shift_col), dims=(0, 1))
        newlines = torch.roll(newlines, shifts=plan.shift_row, dims=0)
    pooled = F.adaptive_avg_pool2d(grid.permute(2, 0, 1).unsqueeze(0), (plan.height, plan.width))
    pooled = pooled[0].permute(1, 2, 0)
    pieces = [base]
    for out_row in range(plan.height):
        start, end = bin_bounds(out_row, plan.height)
        pieces.append(pooled[out_row])
        pieces.append(newlines[start:end].mean(dim=0, keepdim=True))
    output = torch.cat(pieces, dim=0)
    if output.shape[0] != plan.page_out:
        raise RuntimeError(f"pooled length {output.shape[0]} != {plan.page_out}")
    return output


def pack_sequence(model, input_ids: torch.Tensor, spans: list[tuple[int, int]], pages: list[torch.Tensor]) -> torch.Tensor:
    embed = model.get_input_embeddings()
    pieces = []
    cursor = 0
    text_count = 0
    for (start, end), feats in zip(spans, pages):
        if start > cursor:
            pieces.append(embed(input_ids[:, cursor:start]))
            text_count += start - cursor
        pieces.append(feats.unsqueeze(0).to(dtype=pieces[0].dtype if pieces else feats.dtype))
        cursor = end
    if cursor < input_ids.shape[1]:
        pieces.append(embed(input_ids[:, cursor:]))
        text_count += input_ids.shape[1] - cursor
    packed = torch.cat(pieces, dim=1)
    visual = sum(page.shape[0] for page in pages)
    if text_count != input_ids.shape[1] - sum(end - start for start, end in spans):
        raise RuntimeError("text tokens were dropped or duplicated")
    if packed.shape[1] != text_count + visual:
        raise RuntimeError("packed sequence crossed a page boundary")
    return packed
