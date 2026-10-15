"""Prospective selector. Ranking inputs are margin and step only."""

from __future__ import annotations

EPS = 8 / 255


def choose(candidates: list[dict]) -> dict | None:
    if not candidates:
        return None
    return sorted(candidates, key=lambda row: (row["margin"], row["step"]))[0]


def eligible(row: dict) -> bool:
    return bool(
        int(row["step"]) > 0
        and row["nonzero"]
        and row["output_valid"]
        and row["full_exact"]
        and row["linf_ok"]
        and float(row["pixel_linf_uint8"]) <= EPS + 1e-6
    )
