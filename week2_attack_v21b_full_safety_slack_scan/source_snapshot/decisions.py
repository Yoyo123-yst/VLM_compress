"""Radius labels and the final slack status. No model calls."""

from __future__ import annotations

RADII = (1, 2, 4, 6, 8)


def prefix_safe(legal: dict[int, bool]) -> int:
    passed = 0
    for radius in RADII:
        if not legal.get(radius, False):
            break
        passed = radius
    return passed


def largest_legal(legal: dict[int, bool]) -> int:
    values = [radius for radius, ok in legal.items() if ok]
    return max(values) if values else 0


def nonmonotonic(legal: dict[int, bool]) -> bool:
    failed = False
    for radius in RADII:
        if not legal.get(radius, False):
            failed = True
        elif failed:
            return True
    return False


def direction_summary(points: list[dict]) -> dict:
    legal = {int(row["requested_uint8_linf"]): bool(row["full_legal"]) for row in points if int(row["requested_uint8_linf"]) in RADII}
    prefix = prefix_safe(legal)
    progress = []
    for row in points:
        if int(row["requested_uint8_linf"]) == 0 or not row["full_legal"]:
            continue
        progress.append(bool(row["core_margin_decreased"] or row["fastv_text_changed"]))
    return {
        "legal": {str(radius): bool(legal.get(radius, False)) for radius in RADII},
        "largest_tested_legal_radius": largest_legal(legal),
        "prefix_safe_radius": prefix,
        "nonmonotonic_full_behavior": nonmonotonic(legal),
        "attack_progress_on_legal_points": any(progress),
        "quantization_mismatches": [int(row["requested_uint8_linf"]) for row in points if row.get("quantization_radius_mismatch")],
    }


def final_status(core: dict, all_row: dict) -> dict:
    prefixes = [core["prefix_safe_radius"], all_row["prefix_safe_radius"]]
    progress = core["attack_progress_on_legal_points"] or all_row["attack_progress_on_legal_points"]
    any_legal = core["largest_tested_legal_radius"] > 0 or all_row["largest_tested_legal_radius"] > 0
    if max(prefixes) >= 4 and progress:
        status = "FULL_SAFETY_SLACK_CONFIRMED"
        next_step = "SMALL_CONSTRAINED_COMPARISON_MAY_BE_DISCUSSED_LATER"
    elif not any_legal or not progress:
        status = "NO_USABLE_FULL_SAFETY_SLACK"
        next_step = "NO_MULTISTEP_COMPARISON_AUTHORIZED"
    else:
        status = "LIMITED_FULL_SAFETY_SLACK"
        next_step = "NO_MULTISTEP_COMPARISON_AUTHORIZED"
    return {"status": status, "next_step": next_step, "comparison_started": False}
