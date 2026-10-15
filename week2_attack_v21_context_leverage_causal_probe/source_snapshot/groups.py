"""Pure token-partition and local-effect rules. No model calls."""

from __future__ import annotations

import random

CONTEXT_TEXT = "Dr. or Mrs."
SCALES = (0.25, 0.5, 1.0, 2.0)
RANDOM_SEEDS = (1701, 1702, 1703)


def jaccard(left: set[int], right: set[int]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def partition(tokens: list[dict], core_positions: list[int], all_positions: list[int]) -> dict:
    """tokens are the clean FastV pieces in order. Positions refer to that sequence."""
    by_pos = {int(row["position"]): row for row in tokens}
    all_set = tuple(int(pos) for pos in all_positions)
    core = tuple(int(pos) for pos in core_positions)
    context = tuple(pos for pos in all_set if pos not in set(core))
    problems = []
    if not core:
        problems.append("empty_core")
    if not context:
        problems.append("empty_context")
    if set(core) & set(context):
        problems.append("overlap")
    if tuple(sorted(set(core) | set(context))) != tuple(sorted(all_set)):
        problems.append("union_is_not_s_all")
    if any(pos not in by_pos for pos in all_set):
        problems.append("position_missing")
    context_text = "".join(by_pos[pos]["piece"] for pos in context) if not problems else ""
    if not problems and context_text != CONTEXT_TEXT:
        problems.append(f"context_text:{context_text}")
    return {
        "ok": not problems,
        "problems": problems,
        "s_all": list(all_set),
        "s_core": list(core),
        "s_context": list(context),
        "context_text": context_text,
    }


def draw_control(universe: list[int], size: int, forbidden: list[int], seed: int) -> list[int] | None:
    if size <= 0 or size > len(universe):
        return None
    forbidden_set = tuple(sorted(forbidden))
    rng = random.Random(seed)
    for _ in range(256):
        picked = tuple(sorted(rng.sample(list(universe), size)))
        if picked != forbidden_set:
            return list(picked)
    return None


def reduced(row: dict, clean_margin: float) -> bool:
    return bool(
        row.get("full_exact")
        and row.get("linf_ok")
        and row.get("png_reloaded")
        and float(row["core_margin_natural"]) < float(clean_margin)
    )


def supporting_pair(context_rows: dict[float, dict], random_rows: dict[int, dict[float, dict]], clean_margin: float) -> dict:
    """Return whether CONTEXT beats every R_context on some adjacent legal pair."""
    scales = list(SCALES)
    for left, right in zip(scales, scales[1:]):
        if left not in context_rows or right not in context_rows:
            continue
        if not (reduced(context_rows[left], clean_margin) and reduced(context_rows[right], clean_margin)):
            continue
        stronger = True
        for seed, rows in random_rows.items():
            for scale in (left, right):
                other = rows.get(scale)
                if other is None or not other.get("full_exact"):
                    continue
                context_drop = float(clean_margin) - float(context_rows[scale]["core_margin_natural"])
                random_drop = float(clean_margin) - float(other["core_margin_natural"])
                if context_drop <= random_drop:
                    stronger = False
        if stronger:
            return {"supported": True, "scales": [left, right], "random_status": "COMPARED"}
    return {"supported": False, "scales": [], "random_status": "COMPARED" if random_rows else "NOT_CONSTRUCTED"}
