"""CPU decision rules for the v21a repair. No model and no file writes."""

from __future__ import annotations

PRIMARY_SCALES = (0.5, 1.0, 2.0)
PAIRS = ((0.5, 1.0), (1.0, 2.0))


def _same(values: list) -> bool:
    return len(values) > 0 and all(value == values[0] for value in values)


def classify_replays(runs: list[dict]) -> dict:
    """Classify five serial zero-perturbation replays.

    Each run has paths ORIGINAL, ZERO_PROJECTED, and ZERO_DIRECT_SAVE.
    Equality uses hashes recorded by the worker, not reloaded floats.
    """
    if len(runs) != 5:
        raise RuntimeError(f"expected 5 replays, got {len(runs)}")
    names = ("ORIGINAL", "ZERO_PROJECTED", "ZERO_DIRECT_SAVE")
    pixels_differ = False
    processor_differ = False
    for run in runs:
        original = run["paths"]["ORIGINAL"]
        for name in names[1:]:
            path = run["paths"][name]
            if path["rgb_sha256"] != original["rgb_sha256"] or int(path["uint8_linf"]) != 0:
                pixels_differ = True
            if path["processor_exact_hash"] != original["processor_exact_hash"]:
                processor_differ = True
    processor_hashes = [run["paths"]["ORIGINAL"]["processor_exact_hash"] for run in runs]
    keep_hashes = [run["paths"]["ORIGINAL"]["keep_hash"] for run in runs]
    ks = [run["paths"]["ORIGINAL"]["k"] for run in runs]
    tokens = [tuple(run["paths"]["ORIGINAL"]["generated_tokens"]) for run in runs]
    logits = [run["paths"]["ORIGINAL"]["natural_logits_hash"] for run in runs]
    frozen_logits = [run["paths"]["ORIGINAL"]["frozen_logits_hash"] for run in runs]
    margins = [run["paths"]["ORIGINAL"]["margin_hash"] for run in runs]
    model_same = _same(processor_hashes) and _same(keep_hashes) and _same(ks) and _same(tokens) and _same(logits) and _same(frozen_logits) and _same(margins)
    within_run_model = True
    file_differs = False
    for run in runs:
        original = run["paths"]["ORIGINAL"]
        for name in names[1:]:
            path = run["paths"][name]
            if path["processor_exact_hash"] != original["processor_exact_hash"]:
                within_run_model = False
            if path["keep_hash"] != original["keep_hash"] or path["k"] != original["k"]:
                within_run_model = False
            if path["natural_logits_hash"] != original["natural_logits_hash"] or path["margin_hash"] != original["margin_hash"]:
                within_run_model = False
            if path["file_sha256"] != original["memory_png_sha256"]:
                file_differs = True
    if pixels_differ:
        status = "BLOCKED_IMAGE_RECONSTRUCTION_MISMATCH"
    elif processor_differ or not _same(processor_hashes):
        status = "BLOCKED_PREPROCESSING_PATH_MISMATCH"
    elif not model_same or not within_run_model:
        status = "BLOCKED_FASTV_RUNTIME_NONREPEATABLE"
    elif file_differs:
        status = "ENCODING_ONLY_DIFFERENCE"
    else:
        status = "ZERO_REPLAY_EXACT"
    return {
        "status": status,
        "enter_phase_b": status in {"ZERO_REPLAY_EXACT", "ENCODING_ONLY_DIFFERENCE"},
        "processor_hashes_identical": _same(processor_hashes),
        "keep_hashes_identical": _same(keep_hashes),
        "model_results_identical": model_same and within_run_model and not processor_differ,
    }


def _rows(summary: dict, direction: str) -> list[dict]:
    return [summary["steps"][direction][f"{scale:.2f}"] for scale in PRIMARY_SCALES]


def _mean_down(row: dict, keep: str) -> bool:
    return bool(row["full_exact"] and row["png_reloaded"] and row[f"core_hard_mean_{keep}"] < row["clean_core_hard_mean"])


def _supporting_pairs(summary: dict, direction: str) -> list[tuple[float, float]]:
    found = []
    steps = summary["steps"][direction]
    for left, right in PAIRS:
        a = steps[f"{left:.2f}"]
        b = steps[f"{right:.2f}"]
        if _mean_down(a, "natural") and _mean_down(b, "natural") and _mean_down(a, "frozen") and _mean_down(b, "frozen"):
            found.append((left, right))
    return found


def _content_changed(summary: dict, direction: str) -> bool:
    return any(not row["core_name_intact"] for row in _rows(summary, direction))


def _combined_not_lowering(summary: dict, scales: tuple[float, float]) -> bool:
    for scale in scales:
        row = summary["steps"]["CONTEXT_COMBINED"][f"{scale:.2f}"]
        if row["core_hard_mean_natural"] < row["clean_core_hard_mean"]:
            return False
    return True


def _fdp_protects(summary: dict, scales: tuple[float, float]) -> bool:
    for scale in scales:
        row = summary["steps"]["FDP_ONLY"][f"{scale:.2f}"]
        if row["core_hard_mean_natural"] < row["clean_core_hard_mean"]:
            return False
    return True


def _single_token_only(summary: dict) -> bool:
    for row in _rows(summary, "CONTEXT_ATTACK_ONLY"):
        down = int(row["core_tokens_down_natural"])
        if down > 1:
            return False
        if row["core_hard_mean_natural"] < row["clean_core_hard_mean"]:
            return False
        if not row["core_name_intact"]:
            return False
    return any(int(row["core_tokens_down_natural"]) == 1 for row in _rows(summary, "CONTEXT_ATTACK_ONLY"))


def decide_directions(summary: dict) -> dict:
    core = _rows(summary, "CORE_ATTACK_ONLY")
    sign_error = all(
        row["core_hard_mean_natural"] > row["clean_core_hard_mean"] and row["attack_soft_loss_natural"] > row["clean_attack_soft_loss"]
        for row in core
    )
    proxy = all(
        row["attack_soft_loss_natural"] < row["clean_attack_soft_loss"] and row["core_hard_mean_natural"] > row["clean_core_hard_mean"]
        for row in core
    )
    if summary.get("toy_sign_ok") is False or sign_error:
        status = "GRADIENT_SIGN_OR_IMPLEMENTATION_ERROR"
    elif proxy:
        status = "CORE_OBJECTIVE_PROXY_MISMATCH"
    else:
        pairs = _supporting_pairs(summary, "CONTEXT_ATTACK_ONLY")
        changed = _content_changed(summary, "CONTEXT_ATTACK_ONLY")
        if pairs or changed:
            combined_pairs = _supporting_pairs(summary, "CONTEXT_COMBINED")
            if combined_pairs and all(row["full_exact"] for row in _rows(summary, "CONTEXT_COMBINED")):
                status = "CONTEXT_EFFECT_SURVIVES_FULL_CONSTRAINT"
            elif pairs and _combined_not_lowering(summary, pairs[0]) and _fdp_protects(summary, pairs[0]):
                status = "CONTEXT_EFFECT_BLOCKED_BY_FULL_CONSTRAINT"
            elif changed and not _content_changed(summary, "CONTEXT_COMBINED") and _fdp_protects(summary, PRIMARY_SCALES):
                status = "CONTEXT_EFFECT_BLOCKED_BY_FULL_CONSTRAINT"
            else:
                status = "NO_GO_CONTEXT_CAUSAL_EFFECT_STRONG"
        elif _single_token_only(summary):
            status = "WEAK_SINGLE_TOKEN_COUPLING_ONLY"
        else:
            status = "NO_GO_CONTEXT_CAUSAL_EFFECT_STRONG"
    return {
        "status": status,
        "context_pairs": _supporting_pairs(summary, "CONTEXT_ATTACK_ONLY"),
        "combined_pairs": _supporting_pairs(summary, "CONTEXT_COMBINED"),
        "phase2": "NOT_RUN",
        "extension": "NOT_JUSTIFIED",
    }
