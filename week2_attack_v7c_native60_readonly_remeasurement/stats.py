"""Frozen summaries for the native60 remeasure. No model calls."""

from __future__ import annotations

import math
import random

BUDGETS = (0.50, 0.35, 0.25, 0.15)
BOOT_N = 10000
BOOT_SEED = 2026101101


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("empty")
    return sum(values) / len(values)


def median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n % 2:
        return ordered[n // 2]
    return 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])


def stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    center = mean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / (len(values) - 1))


def bootstrap_mean_ci(values: list[float], n: int = BOOT_N, seed: int = BOOT_SEED) -> tuple[float, float]:
    rng = random.Random(seed)
    size = len(values)
    stats = []
    for _ in range(n):
        draw = [values[rng.randrange(size)] for _ in range(size)]
        stats.append(mean(draw))
    stats.sort()
    lo = stats[int(0.025 * n)]
    hi = stats[int(0.975 * n) - 1]
    return lo, hi


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1))
    return min(1.0, 2 * tail / (2 ** n))


def sign_test(diffs: list[float]) -> dict:
    nonzero = [value for value in diffs if value != 0]
    positive = sum(value > 0 for value in nonzero)
    negative = sum(value < 0 for value in nonzero)
    return {
        "n_nonzero": len(nonzero),
        "n_positive": positive,
        "n_negative": negative,
        "n_zero_excluded": len(diffs) - len(nonzero),
        "p_two_sided": mcnemar(positive, negative),
    }


def sweet_outer(delta_by_r: dict[float, list[float]]) -> dict:
    n = len(delta_by_r[0.25])
    sweet = [0.5 * (delta_by_r[0.25][i] + delta_by_r[0.35][i]) for i in range(n)]
    outer = [0.5 * (delta_by_r[0.50][i] + delta_by_r[0.15][i]) for i in range(n)]
    diff = [sweet[i] - outer[i] for i in range(n)]
    lo, hi = bootstrap_mean_ci(diff)
    return {
        "sweet_mean": mean(sweet),
        "outer_mean": mean(outer),
        "paired_diff_mean": mean(diff),
        "paired_diff_ci95": [lo, hi],
        "sign_test": sign_test(diff),
        "paired_diffs": diff,
    }


def curve_supports_sweet_spot(strict_counts: dict[float, int]) -> bool:
    return (
        strict_counts[0.25] > strict_counts[0.50]
        and strict_counts[0.25] > strict_counts[0.15]
        and strict_counts[0.35] > strict_counts[0.50]
        and strict_counts[0.35] > strict_counts[0.15]
    )


def decide(alignment_ok: bool, immutable_ok: bool, delta_ci_low: dict[float, float], delta_mean: dict[float, float], sweet: dict, strict_counts: dict[float, int]) -> str:
    if not alignment_ok:
        return "BLOCKED_NATIVE60_ALIGNMENT_MISMATCH"
    if not immutable_ok:
        return "BLOCKED_PARENT_RAW_MUTATED"
    sweet_gap = sweet["sweet_mean"] > sweet["outer_mean"] and sweet["paired_diff_ci95"][0] > 0
    both_budgets = (
        delta_mean[0.25] > 0
        and delta_mean[0.35] > 0
        and delta_ci_low[0.25] > 0
        and delta_ci_low[0.35] > 0
    )
    if both_budgets and sweet_gap and curve_supports_sweet_spot(strict_counts):
        return "PASS_NATIVE60_SWEET_SPOT_AND_DIFFERENTIAL_SUPPORTED"
    if delta_ci_low[0.25] > 0 or delta_ci_low[0.35] > 0:
        return "PARTIAL_NATIVE60_DIFFERENTIAL_SUPPORTED_NO_CONFIRMED_SWEET_SPOT"
    return "NO_GO_NATIVE60_DIFFERENTIAL_NOT_SUPPORTED"
