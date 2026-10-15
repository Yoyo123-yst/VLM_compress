"""Paired holdout summary. best_strict is the same rule used on the development set."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
sys.path.insert(0, str(ROOT))
from week2_attack_v4_relaxed_full.analyze_lambda import (  # noqa: E402
    by_sample,
    load_jsonl,
    pick,
    summarize,
)

OUT = ROOT / "week2_attack_v4_relaxed_full"
WRONG = 0.5


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def mcnemar(b: int, c: int) -> float:
    """Two-sided exact test on discordant pairs. b = candidate only, c = baseline only."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1))
    return min(1.0, 2 * tail / (2 ** n))


def mean_ci(values: list[float]) -> tuple[float, float]:
    n = len(values)
    mu = sum(values) / n
    var = sum((v - mu) ** 2 for v in values) / (n - 1)
    se = math.sqrt(var / n)
    return (mu - 1.96 * se, mu + 1.96 * se)


def main():
    base_rows = []
    cand_rows = []
    for path in sorted((OUT / "runs" / "holdout_lambda1").glob("checkpoints_s*.jsonl")):
        base_rows.extend(load_jsonl(path))
    for path in sorted((OUT / "runs" / "holdout_lambda025").glob("checkpoints_s*.jsonl")):
        cand_rows.extend(load_jsonl(path))
    base = {sid: pick(rows) for sid, rows in by_sample(base_rows).items()}
    cand = {sid: pick(rows) for sid, rows in by_sample(cand_rows).items()}
    common = sorted(set(base) & set(cand))
    base_s = summarize({sid: base[sid] for sid in common}, "best_strict")
    cand_s = summarize({sid: cand[sid] for sid in common}, "best_strict")
    paired = []
    b = c = 0
    gaps = []
    for sid in common:
        left = base[sid]["best_strict"]
        right = cand[sid]["best_strict"]
        left_hit = left["quadrant"] == "fastv_wrong_full_right"
        right_hit = right["quadrant"] == "fastv_wrong_full_right"
        if right_hit and not left_hit:
            b += 1
        if left_hit and not right_hit:
            c += 1
        gaps.append(float(right["G"]) - float(left["G"]))
        paired.append({
            "sample_id": sid,
            "base_full": left["pred_full"], "base_fastv": left["pred_fastv"],
            "cand_full": right["pred_full"], "cand_fastv": right["pred_fastv"],
            "base_anls_full": left["anls_full"], "base_anls_fastv": left["anls_fastv"],
            "cand_anls_full": right["anls_full"], "cand_anls_fastv": right["anls_fastv"],
            "base_G": left["G"], "cand_G": right["G"], "G_diff": float(right["G"]) - float(left["G"]),
            "base_quadrant": left["quadrant"], "cand_quadrant": right["quadrant"],
        })
    n = len(common)
    strict_k = sum(row["cand_quadrant"] == "fastv_wrong_full_right" for row in paired)
    both_wrong = sum(row["cand_quadrant"] == "both_wrong" for row in paired)
    full_err = sum(float(row["cand_anls_full"]) < WRONG for row in paired)
    fast_err = sum(float(row["cand_anls_fastv"]) < WRONG for row in paired)
    gap = (fast_err - full_err) / n if n else None
    mean_g = sum(float(row["cand_G"]) for row in paired) / n if n else None
    incomplete = n < 20 or len(base) != len(cand)
    if incomplete:
        decision = "HOLDOUT_INCOMPLETE"
    elif strict_k >= 2 and gap and gap > 0 and mean_g and mean_g > 0 and full_err < fast_err and both_wrong < strict_k:
        if strict_k / n >= 0.5 and full_err == 0:
            decision = "HOLDOUT_SUPPORTS_STRICT_COMP_ONLY"
        else:
            decision = "HOLDOUT_SUPPORTS_COMPRESSION_AMPLIFICATION"
    elif mean_g and mean_g > 0 and gap and gap > 0:
        decision = "HOLDOUT_WEAK_SIGNAL"
    else:
        decision = "HOLDOUT_DOES_NOT_SUPPORT"
    # A single sample must not be the whole gap.
    if decision.startswith("HOLDOUT_SUPPORTS") and strict_k < 2:
        decision = "HOLDOUT_WEAK_SIGNAL"
    lo, hi = wilson(strict_k, n) if n else (None, None)
    g_lo, g_hi = mean_ci(gaps) if n > 1 else (None, None)
    report = {
        "stage": "D",
        "health": "incomplete" if incomplete else "complete",
        "completed_samples": n,
        "baseline_completed": len(base),
        "candidate_completed": len(cand),
        "invalid_outputs": base_s["invalid"] + cand_s["invalid"] if n else None,
        "strict_csfr": cand_s["strict_csfr"] if n else None,
        "full_error_rate": cand_s["full_error_rate"] if n else None,
        "fastv_error_rate": cand_s["fastv_error_rate"] if n else None,
        "error_rate_gap": cand_s["error_rate_gap"] if n else None,
        "mean_DF": cand_s["mean_DF"] if n else None,
        "mean_DC": cand_s["mean_DC"] if n else None,
        "mean_G": cand_s["mean_G"] if n else None,
        "baseline": base_s if n else None,
        "wilson_strict_csfr": [lo, hi],
        "mcnemar_p_exact": mcnemar(b, c) if n else None,
        "discordant_candidate_only": b,
        "discordant_baseline_only": c,
        "mean_G_difference_ci95_normal": [g_lo, g_hi],
        "decision": decision,
        "evidence": [
            f"paired n={n}",
            f"candidate strict {strict_k}/{n}" if n else "no paired rows",
            f"McNemar discordant candidate-only {b}, baseline-only {c}",
        ],
        "next_action": "write final report",
        "next_command": "",
    }
    dest = OUT / "results" / "holdout"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "decision.json").write_text(json.dumps(report, indent=2) + "\n")
    if paired:
        with (dest / "paired.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(paired[0]))
            writer.writeheader()
            writer.writerows(paired)
    print(json.dumps({"decision": decision, "n": n, "strict": strict_k if n else None, "mcnemar": report["mcnemar_p_exact"]}))


if __name__ == "__main__":
    main()
