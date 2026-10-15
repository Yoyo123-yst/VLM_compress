"""Fixed checkpoint selector.

The selector sees legality bits and the group's own attack surrogate. It does
not see ANLS, semantic labels, or an attack-success flag.
"""

from __future__ import annotations


def is_legal(row: dict) -> bool:
    required = (
        "nonzero",
        "output_valid",
        "full_text_exact",
        "full_token_exact",
        "full_replay",
        "linf_ok",
        "png_reloadable",
    )
    return all(bool(row.get(key)) for key in required)


def choose_fixed(rows: list[dict]) -> dict | None:
    eligible = []
    for row in rows:
        if "anls" in row or "semantic_status" in row or "content_failure" in row:
            raise RuntimeError("selector input included a forbidden outcome field")
        if not is_legal(row):
            continue
        if "attack_surrogate" not in row or "checkpoint" not in row:
            raise RuntimeError("selector row is missing surrogate or checkpoint")
        eligible.append(row)
    if not eligible:
        return None
    return sorted(eligible, key=lambda row: (float(row["attack_surrogate"]), int(row["checkpoint"])))[0]
