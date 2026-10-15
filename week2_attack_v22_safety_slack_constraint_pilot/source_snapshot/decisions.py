"""Pre-registered selector and final gate. No model calls."""

from __future__ import annotations

FORBIDDEN = ("anls", "semantic_status", "content_failure", "fast_text", "fast_tokens")


def choose_selector(rows: list[dict]) -> dict:
    for row in rows:
        found = [key for key in FORBIDDEN if key in row]
        if found:
            raise RuntimeError(f"selector saw forbidden fields: {found}")
        if "l_core" not in row or "step" not in row or "legal" not in row:
            raise RuntimeError("selector row lacks step, legal, or l_core")
    legal = [row for row in rows if row["legal"]]
    if not legal:
        raise RuntimeError("no legal checkpoint")
    legal.sort(key=lambda row: (float(row["l_core"]), int(row["step"])))
    return {"step": int(legal[0]["step"]), "l_core": float(legal[0]["l_core"])}


def first_drop(points: list[dict], clean_mean: float) -> int | None:
    for row in sorted(points, key=lambda item: int(item["step"])):
        if float(row["core_hard_mean"]) < clean_mean:
            return int(row["step"])
    return None


def decide(t1: dict, t2: dict, t0_full_exact: bool) -> dict:
    """Return the pre-registered status for this one development sample."""
    clean = float(t1["clean_core_hard_mean"])
    drop1 = first_drop(t1["checkpoints"], clean)
    drop2 = first_drop(t2["checkpoints"], clean)
    mean1 = float(t1["selector_core_hard_mean"])
    mean2 = float(t2["selector_core_hard_mean"])
    faster = drop2 is not None and (drop1 is None or drop2 < drop1 or (drop2 == drop1 and mean2 < mean1))
    lower = mean2 < mean1
    failure = t2["semantic_status"] == "CONTENT_FAILURE"
    full_ok = bool(t2["full_exact"])
    if faster and full_ok and failure:
        status = "SAFETY_SLACK_HINGE_INITIAL_WITNESS"
    elif lower and full_ok and not failure:
        status = "SURROGATE_IMPROVED_ENDPOINT_NOT_ESTABLISHED"
    else:
        status = "NO_GO_SAFETY_SLACK_HINGE_ON_47450"
    return {
        "status": status,
        "t2_faster_core_drop": faster,
        "t2_selector_margin_lower": lower,
        "t0_selector_full_exact": bool(t0_full_exact),
        "second_sample_authorized": False,
        "comparison_started_beyond_pilot": False,
    }
