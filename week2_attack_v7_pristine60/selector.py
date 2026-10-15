"""Prospective feasible selector. Ranking uses compressed margin, then earlier step."""

from __future__ import annotations

EPS = 8 / 255
LEGAL_KEYS = ("step", "nonzero", "output_valid", "full_exact", "linf_ok", "pixel_linf_uint8", "margin")


def eligible(row: dict) -> bool:
    return bool(
        int(row["step"]) > 0
        and row["nonzero"]
        and row["output_valid"]
        and row["full_exact"]
        and row["linf_ok"]
        and float(row["pixel_linf_uint8"]) <= EPS + 1e-6
    )


def choose(candidates: list[dict]) -> dict | None:
    if not candidates:
        return None
    ranked = []
    for row in candidates:
        ranked.append({"margin": float(row["margin"]), "step": int(row["step"])})
    return sorted(ranked, key=lambda row: (row["margin"], row["step"]))[0]
