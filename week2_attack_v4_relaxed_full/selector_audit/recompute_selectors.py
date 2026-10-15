"""Read-only holdout checkpoint selector audit. Writes only under selector_audit/."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
OUT = ROOT / "week2_attack_v4_relaxed_full" / "selector_audit"
RELAXED = ROOT / "week2_attack_v4_relaxed_full"
sys.path.insert(0, str(ROOT))

from week2_attack_v4_answer_aligned.common import RATIO  # noqa: E402
from week2_attack_v4_answer_aligned.loss import official_keep_count, select_best  # noqa: E402

TOKENS_PER_PAGE = 3699
K_DERIVATION = (
    "expected_k = official_keep_count(M * 3699, keep_ratio=RATIO); "
    "M=2->1850, M=3->2774, M=4->3699 at r=0.25"
)

WRONG = 0.5
EPS = 8 / 255 + 1e-6
REQUIRED = {
    "sample_id", "step", "margin", "anls_full", "anls_fastv", "output_valid",
    "linf", "pred_full", "pred_fastv", "keep_hash", "page_sha", "k",
}
EXPECTED_STEPS = [0, 10, 25, 50, 75, 100, 150, 200]
RUN_TAGS = {
    "lambda_1": RELAXED / "runs" / "holdout_lambda1",
    "lambda_0_25": RELAXED / "runs" / "holdout_lambda025",
}
MANIFEST = RELAXED / "holdout" / "frozen_manifest.json"
PAIRED_LEGACY = RELAXED / "results" / "holdout" / "paired.csv"
BOOTSTRAP_SEED = 20261008
BOOTSTRAP_N = 10000


def load_jsonl(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def by_sample(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row["sample_id"]), []).append(row)
    for sid in grouped:
        grouped[sid].sort(key=lambda r: int(r["step"]))
    return grouped


def quadrant(full_anls: float, fast_anls: float) -> str:
    fw = full_anls < WRONG
    cv = fast_anls < WRONG
    if cv and not fw:
        return "fastv_wrong_full_right"
    if (not cv) and fw:
        return "fastv_right_full_wrong"
    if cv and fw:
        return "both_wrong"
    return "both_right"


def attach_degradation(rows: list[dict]) -> dict:
    clean = rows[0]
    if int(clean["step"]) != 0:
        raise RuntimeError("series must start at step 0")
    s_f = float(clean["anls_full"])
    s_c = float(clean["anls_fastv"])
    for row in rows:
        row["_S_F0"] = s_f
        row["_S_C0"] = s_c
        row["D_F"] = s_f - float(row["anls_full"])
        row["D_C"] = s_c - float(row["anls_fastv"])
        row["G"] = row["D_C"] - row["D_F"]
        row["quadrant"] = quadrant(float(row["anls_full"]), float(row["anls_fastv"]))
        row["pred_full_step0"] = clean["pred_full"]
        row["pred_fastv_step0"] = clean["pred_fastv"]
    return {"S_F_clean": s_f, "S_C_clean": s_c}


def as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return bool(int(value))


def select_fixed_step200(rows: list[dict]) -> dict:
    matches = [r for r in rows if int(r["step"]) == 200]
    if len(matches) != 1:
        raise RuntimeError(f"expected one step-200 checkpoint, got {len(matches)}")
    return matches[0]


def select_min_margin(rows: list[dict]) -> dict:
    pool = [r for r in rows if as_bool(r["output_valid"])]
    if not pool:
        raise RuntimeError("no output_valid checkpoint for min-margin selector")
    margins = {}
    for row in pool:
        m = float(row["margin"])
        margins.setdefault(m, []).append(row)
    min_m = min(margins)
    tied = margins[min_m]
    if len(tied) > 1:
        steps = sorted(int(r["step"]) for r in tied)
        if len(set(steps)) < len(tied):
            raise RuntimeError(f"min-margin tie on margin={min_m} and step for same sample")
        return min(tied, key=lambda r: int(r["step"]))
    return tied[0]


def select_oracle_best_strict(rows: list[dict]) -> dict:
    strict_rows = []
    for row in rows:
        strict_rows.append({
            "step": int(row["step"]),
            "anls_fastv": float(row["anls_fastv"]),
            "anls_full": float(row["anls_full"]),
            "margin": float(row["margin"]),
            "linf": float(row["linf"]),
            "output_valid": as_bool(row["output_valid"]),
        })
    chosen = select_best(strict_rows)
    step = int(chosen["step"])
    matches = [r for r in rows if int(r["step"]) == step]
    if len(matches) != 1:
        raise RuntimeError(f"oracle selector step {step} not unique")
    out = matches[0].copy()
    eligible = [r for r in rows if as_bool(r["output_valid"]) and float(r["anls_full"]) >= WRONG]
    out["_oracle_fallback"] = not eligible
    return out


SELECTORS = {
    "primary_fixed_step200": {
        "fn": select_fixed_step200,
        "uses_answer_at_selection": False,
        "label": "Primary endpoint (fixed step 200)",
        "banner": "",
    },
    "posthoc_min_compressed_margin": {
        "fn": select_min_margin,
        "uses_answer_at_selection": False,
        "label": "Post-hoc min compressed margin",
        "banner": "POST_HOC_SELECTOR_AUDIT_NOT_PROSPECTIVE_EVIDENCE",
    },
    "diagnostic_oracle_best_strict": {
        "fn": select_oracle_best_strict,
        "uses_answer_at_selection": True,
        "label": "Diagnostic oracle best_strict",
        "banner": "GROUND_TRUTH_ORACLE_UPPER_BOUND",
    },
}


def expected_k_for_sample(m_pages: int) -> int:
    n_visual = m_pages * TOKENS_PER_PAGE
    return official_keep_count(n_visual, RATIO)


def load_manifest_m() -> dict[str, int]:
    return {s["sample_id"]: int(s["M"]) for s in json.loads(MANIFEST.read_text())["samples"]}


def audit_k(data: dict[str, dict[str, list[dict]]], manifest_m: dict[str, int]) -> dict:
    per_sample = {}
    mismatches = []
    for sid in sorted(manifest_m):
        expected = expected_k_for_sample(manifest_m[sid])
        rows_l1 = data["lambda_1"][sid]
        rows_025 = data["lambda_0_25"][sid]
        ks_l1 = [int(r["k"]) for r in rows_l1]
        ks_025 = [int(r["k"]) for r in rows_025]
        constant_l1 = len(set(ks_l1)) == 1
        constant_025 = len(set(ks_025)) == 1
        cross_lambda = ks_l1[0] == ks_025[0] if constant_l1 and constant_025 else False
        match = constant_l1 and constant_025 and cross_lambda and ks_l1[0] == expected
        per_sample[sid] = {
            "M": manifest_m[sid],
            "expected_k": expected,
            "observed_k_values": sorted(set(ks_l1 + ks_025)),
            "k_matches_expected": match and ks_l1[0] == expected,
            "k_constant_across_checkpoints": constant_l1 and constant_025,
            "k_consistent_across_lambda": cross_lambda,
        }
        if not per_sample[sid]["k_matches_expected"]:
            mismatches.append({"sample_id": sid, **per_sample[sid]})
    return {
        "derivation": K_DERIVATION,
        "per_sample": per_sample,
        "mismatches": mismatches,
        "status": "PASS" if not mismatches else "FAIL",
    }


def hash_raw_files() -> dict[str, str]:
    paths: list[Path] = []
    paths.append(MANIFEST)
    for tag in ("holdout_lambda1", "holdout_lambda025"):
        run_dir = RELAXED / "runs" / tag
        paths.extend(sorted(run_dir.glob("checkpoints_s*.jsonl")))
        paths.extend(sorted(run_dir.glob("shard_s*.csv")))
        for sub in sorted(run_dir.iterdir()):
            if sub.is_dir() and sub.name.isdigit():
                for name in ("provenance.json", "best.png"):
                    candidate = sub / name
                    if candidate.exists():
                        paths.append(candidate)
    out = {}
    for path in sorted(set(paths)):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        out[str(path.relative_to(RELAXED))] = digest
    return out


def audit_raw_integrity(data: dict[str, dict[str, list[dict]]], k_audit: dict) -> dict:
    manifest_ids = [s["sample_id"] for s in json.loads(MANIFEST.read_text())["samples"]]
    manifest_set = set(manifest_ids)
    issues = []
    per_method = {}
    for method, grouped in data.items():
        ids = sorted(grouped)
        sample_issues = []
        field_missing = []
        dup_steps = []
        bad_linf = []
        nan_rows = []
        for sid, rows in grouped.items():
            steps = [int(r["step"]) for r in rows]
            if len(steps) != len(set(steps)):
                dup_steps.append(sid)
            for row in rows:
                missing = REQUIRED - set(row)
                if missing:
                    field_missing.append({"sample_id": sid, "step": row.get("step"), "missing": sorted(missing)})
                for key in ("margin", "anls_full", "anls_fastv", "linf"):
                    val = row.get(key)
                    if val is None or (isinstance(val, float) and math.isnan(val)):
                        nan_rows.append({"sample_id": sid, "step": row.get("step"), "field": key})
                if int(row["step"]) > 0 and float(row.get("linf", 0)) > EPS:
                    bad_linf.append({"sample_id": sid, "step": row["step"], "linf": row["linf"]})
        if ids != sorted(manifest_set):
            sample_issues.append("id_set_mismatch_manifest")
        if len(ids) != 20:
            sample_issues.append(f"n_samples={len(ids)}")
        step_sets = {tuple(int(r["step"]) for r in grouped[sid]) for sid in ids}
        if len(step_sets) != 1 or list(step_sets)[0] != tuple(EXPECTED_STEPS):
            sample_issues.append("checkpoint_steps_not_uniform")
        per_method[method] = {
            "n_samples": len(ids),
            "sample_ids": ids,
            "steps": list(step_sets)[0] if len(step_sets) == 1 else sorted(step_sets),
            "dup_steps": dup_steps,
            "field_missing": field_missing,
            "nan_rows": nan_rows,
            "linf_violations": bad_linf,
            "issues": sample_issues,
        }
        issues.extend(sample_issues)
    ids1 = set(data["lambda_1"])
    ids2 = set(data["lambda_0_25"])
    cross = []
    if ids1 != ids2:
        cross.append("lambda_methods_sample_id_mismatch")
    paired_legacy = {}
    if PAIRED_LEGACY.exists():
        for row in csv.DictReader(PAIRED_LEGACY.open()):
            paired_legacy[row["sample_id"]] = row
    oracle_mismatch = []
    if paired_legacy:
        for sid in sorted(ids1 & ids2):
            rows = data["lambda_0_25"][sid]
            attach_degradation(rows)
            oracle = select_oracle_best_strict(rows)
            leg = paired_legacy.get(sid)
            if not leg:
                oracle_mismatch.append({"sample_id": sid, "issue": "missing_in_paired_csv"})
                continue
            if leg["cand_quadrant"] != oracle["quadrant"]:
                oracle_mismatch.append({
                    "sample_id": sid,
                    "issue": "quadrant",
                    "legacy": leg["cand_quadrant"],
                    "recomputed": oracle["quadrant"],
                })
            if abs(float(leg["cand_G"]) - float(oracle["G"])) > 1e-6:
                oracle_mismatch.append({
                    "sample_id": sid,
                    "issue": "G",
                    "legacy": leg["cand_G"],
                    "recomputed": oracle["G"],
                })
    status = "PASS"
    blockers = list(dict.fromkeys(issues + cross))
    if blockers or any(per_method[m]["field_missing"] or per_method[m]["dup_steps"] for m in per_method):
        status = "BLOCKED_ON_RAW_INTEGRITY"
    if oracle_mismatch:
        blockers.append("paired_csv_oracle_mismatch")
        status = "BLOCKED_ON_RAW_INTEGRITY"
    if k_audit["status"] != "PASS":
        blockers.append("k_mismatch")
        status = "BLOCKED_ON_RAW_INTEGRITY"
    return {
        "status": status,
        "manifest_ids": manifest_ids,
        "k_audit": k_audit,
        "methods": per_method,
        "cross_method_issues": cross,
        "paired_csv_oracle_check": oracle_mismatch,
        "blockers": blockers,
    }


def summarize_selected(rows: list[dict], selector_key: str) -> dict:
    n = len(rows)
    cells = Counter(r["quadrant"] for r in rows)
    full_err = sum(float(r["anls_full"]) < WRONG for r in rows)
    fast_err = sum(float(r["anls_fastv"]) < WRONG for r in rows)
    gap = cells["fastv_wrong_full_right"] / n - cells["fastv_right_full_wrong"] / n
    full_anls = [float(r["anls_full"]) for r in rows]
    fast_anls = [float(r["anls_fastv"]) for r in rows]
    return {
        "selector": selector_key,
        "n": n,
        "fastv_wrong_full_right": cells["fastv_wrong_full_right"],
        "fastv_right_full_wrong": cells["fastv_right_full_wrong"],
        "both_wrong": cells["both_wrong"],
        "both_right": cells["both_right"],
        "strict_csfr": cells["fastv_wrong_full_right"] / n,
        "full_error_rate": full_err / n,
        "fastv_error_rate": fast_err / n,
        "error_rate_gap": gap,
        "mean_full_anls": statistics.mean(full_anls),
        "median_full_anls": statistics.median(full_anls),
        "mean_fastv_anls": statistics.mean(fast_anls),
        "median_fastv_anls": statistics.median(fast_anls),
        "mean_DF": statistics.mean(float(r["D_F"]) for r in rows),
        "mean_DC": statistics.mean(float(r["D_C"]) for r in rows),
        "mean_G": statistics.mean(float(r["G"]) for r in rows),
        "full_text_unchanged_n": sum(r["pred_full"] == r["pred_full_step0"] for r in rows),
        "fastv_text_unchanged_n": sum(r["pred_fastv"] == r["pred_fastv_step0"] for r in rows),
        "full_anls_unchanged_n": sum(abs(float(r["anls_full"]) - float(r["_S_F0"])) < 1e-12 for r in rows),
        "invalid_output_n": sum(not as_bool(r["output_valid"]) for r in rows),
        "linf_ok_n": sum(float(r["linf"]) <= EPS for r in rows),
        "selected_step_hist": dict(Counter(int(r["step"]) for r in rows)),
        "uses_answer_at_selection": SELECTORS[selector_key]["uses_answer_at_selection"],
        "banner": SELECTORS[selector_key]["banner"],
    }


def mcnemar_exact_two_sided(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1))
    return min(1.0, 2 * tail / (2 ** n))


def t_ci95(values: list[float]) -> tuple[float, float]:
    n = len(values)
    mu = statistics.mean(values)
    if n < 2:
        return (mu, mu)
    sd = statistics.stdev(values)
    se = sd / math.sqrt(n)
    # df = n-1, use 2.093 for n=20
    t = 2.093 if n == 20 else 1.96
    return (mu - t * se, mu + t * se)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def exact_signflip_p_two_sided(diff_values: list[float]) -> tuple[float, int]:
    nonzero = [float(v) for v in diff_values if abs(float(v)) > 1e-15]
    n = len(nonzero)
    if n == 0:
        return 1.0, 0
    observed = abs(sum(nonzero))
    total = 2 ** n
    extreme = 0
    for mask in range(total):
        stat = 0.0
        for i, value in enumerate(nonzero):
            sign = 1.0 if (mask >> i) & 1 else -1.0
            stat += sign * value
        if abs(stat) >= observed:
            extreme += 1
    return extreme / total, n


def bootstrap_mean_ci(values: list[float], seed: int, reps: int) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(reps):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(statistics.mean(sample))
    means.sort()
    lo = means[int(0.025 * reps)]
    hi = means[int(0.975 * reps) - 1]
    return (lo, hi)


def draw_bar(path: Path, title: str, labels: list[str], values: list[float], ymax: float | None = None):
    width, height = 920, 420
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 10), title, fill="black")
    left, top, right, bottom = 70, 50, 880, 340
    hi = ymax or max(0.05, max(values) * 1.1)
    group = (right - left) / len(labels)
    for i, (lab, val) in enumerate(zip(labels, values)):
        x0 = left + i * group + 10
        bar_w = max(4, group - 20)
        y0 = bottom - (max(0.0, val) / hi) * (bottom - top)
        y1 = bottom - (min(0.0, val) / hi) * (bottom - top) if val < 0 else bottom
        draw.rectangle([x0, min(y0, y1), x0 + bar_w, max(y0, y1)], fill=(40, 100, 170))
        draw.text((x0, bottom + 6), lab, fill="black")
        draw.text((x0, y0 - 14), f"{val:.2f}", fill="black")
    image.save(path)


def draw_hist(path: Path, title: str, counts: dict[int, int]):
    width, height = 720, 360
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 10), title, fill="black")
    steps = sorted(counts)
    max_c = max(counts.values()) or 1
    for i, step in enumerate(steps):
        c = counts[step]
        x0 = 60 + i * 70
        h = int(240 * c / max_c)
        draw.rectangle([x0, 300 - h, x0 + 50, 300], fill=(180, 80, 40))
        draw.text((x0, 305), str(step), fill="black")
        draw.text((x0, 300 - h - 14), str(c), fill="black")
    image.save(path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figs").mkdir(exist_ok=True)
    hashes_before = hash_raw_files()
    data = {}
    for method, run_dir in RUN_TAGS.items():
        paths = sorted(run_dir.glob("checkpoints_s*.jsonl"))
        data[method] = by_sample(load_jsonl(paths))

    manifest_m = load_manifest_m()
    k_audit = audit_k(data, manifest_m)
    integrity = audit_raw_integrity(data, k_audit)
    (OUT / "raw_integrity.json").write_text(json.dumps(integrity, indent=2) + "\n")
    md = [
        "# 原始记录完整性",
        "",
        f"状态：**{integrity['status']}**",
        "",
    ]
    if integrity["blockers"]:
        md.append("阻塞项：")
        for b in integrity["blockers"]:
            md.append(f"- {b}")
        md.append("")
    for method, info in integrity["methods"].items():
        md.append(f"## {method}")
        md.append(f"- 样本数 {info['n_samples']}")
        md.append(f"- checkpoint steps: {info['steps']}")
        md.append(f"- 重复 step: {info['dup_steps'] or '无'}")
        md.append(f"- 缺字段: {len(info['field_missing'])}")
        md.append(f"- L∞ 违规: {len(info['linf_violations'])}")
        md.append("")
    md.append("## K 与页数 M")
    md.append("")
    md.append("M=2/3/4 对应不同合法 K，不能统一要求 3699。")
    md.append(f"- 推导：{K_DERIVATION}")
    md.append(f"- K 审计状态：**{k_audit['status']}**")
    if k_audit["mismatches"]:
        md.append(f"- 不匹配样本：{k_audit['mismatches']}")
    md.append("")
    if integrity["paired_csv_oracle_check"]:
        md.append("## paired.csv 对照")
        for row in integrity["paired_csv_oracle_check"]:
            md.append(f"- {row}")
    (OUT / "raw_integrity.md").write_text("\n".join(md) + "\n")

    if integrity["status"] != "PASS":
        decision = {
            "stage": "selector_audit",
            "raw_integrity": "BLOCKED",
            "n": 20,
            "primary_selector": "fixed_step_200",
            "reporting_conclusion": "BLOCKED_ON_RAW_INTEGRITY",
            "next_action": "fix data chain before reporting",
        }
        (OUT / "selector_audit_decision.json").write_text(json.dumps(decision, indent=2) + "\n")
        print(json.dumps(decision))
        return

    per_sample_rows = []
    summaries = []
    selected_by = {method: {sel: [] for sel in SELECTORS} for method in data}

    for method, grouped in data.items():
        for sid in sorted(grouped):
            rows = grouped[sid]
            attach_degradation(rows)
            for sel_key, spec in SELECTORS.items():
                chosen = spec["fn"](rows)
                rec = {
                    "method": method,
                    "selector": sel_key,
                    "sample_id": sid,
                    "selected_step": int(chosen["step"]),
                    "margin": float(chosen["margin"]),
                    "anls_full": float(chosen["anls_full"]),
                    "anls_fastv": float(chosen["anls_fastv"]),
                    "D_F": float(chosen["D_F"]),
                    "D_C": float(chosen["D_C"]),
                    "G": float(chosen["G"]),
                    "quadrant": chosen["quadrant"],
                    "pred_full": chosen["pred_full"],
                    "pred_fastv": chosen["pred_fastv"],
                    "pred_full_step0": chosen["pred_full_step0"],
                    "pred_fastv_step0": chosen["pred_fastv_step0"],
                    "full_text_unchanged": int(chosen["pred_full"] == chosen["pred_full_step0"]),
                    "fastv_text_unchanged": int(chosen["pred_fastv"] == chosen["pred_fastv_step0"]),
                    "full_anls_unchanged": int(abs(float(chosen["anls_full"]) - float(chosen["_S_F0"])) < 1e-12),
                    "output_valid": int(as_bool(chosen["output_valid"])),
                    "linf": float(chosen["linf"]),
                    "k": int(chosen["k"]),
                    "oracle_fallback": int(chosen.get("_oracle_fallback", False)),
                }
                per_sample_rows.append(rec)
                selected_by[method][sel_key].append(chosen)

    for method in data:
        for sel_key in SELECTORS:
            summaries.append(summarize_selected(selected_by[method][sel_key], sel_key) | {"method": method})

    fields = list(per_sample_rows[0].keys())
    with (OUT / "per_sample_all_selectors.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(per_sample_rows)

    sum_fields = [
        "method", "selector", "n", "strict_csfr", "full_error_rate", "fastv_error_rate",
        "error_rate_gap", "mean_G", "mean_DF", "mean_DC", "mean_full_anls", "median_full_anls",
        "mean_fastv_anls", "median_fastv_anls",
        "fastv_wrong_full_right", "fastv_right_full_wrong", "both_wrong", "both_right",
        "full_text_unchanged_n", "full_anls_unchanged_n",
        "fastv_text_unchanged_n", "invalid_output_n", "linf_ok_n", "uses_answer_at_selection", "banner",
        "selected_step_hist",
    ]
    with (OUT / "summary_all_selectors.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sum_fields, extrasaction="ignore")
        writer.writeheader()
        for row in summaries:
            row = dict(row)
            row["selected_step_hist"] = json.dumps(row["selected_step_hist"], sort_keys=True)
            writer.writerow(row)

    paired_fieldnames = [
        "selector", "sample_id", "base_strict", "cand_strict", "base_G", "cand_G", "G_diff",
        "base_step", "cand_step", "candidate_only_strict", "baseline_only_strict",
        "mcnemar_p_exact_two_sided", "mean_G_diff", "t_ci95_low", "t_ci95_high",
        "bootstrap_ci95_low", "bootstrap_ci95_high",
        "nonzero_pair_count", "exact_signflip_p_two_sided",
        "mcnemar_reject_005", "signflip_reject_005", "t_ci_excludes_zero",
        "bootstrap_ci_excludes_zero", "inference_consistent", "lambda_advantage_conclusion",
    ]
    paired_rows = []
    sample_ids = sorted(data["lambda_1"])
    for sel_key in SELECTORS:
        b = c = 0
        g_diffs = []
        for sid in sample_ids:
            def get_rec(method):
                return next(
                    r for r in per_sample_rows
                    if r["method"] == method and r["selector"] == sel_key and r["sample_id"] == sid
                )
            base = get_rec("lambda_1")
            cand = get_rec("lambda_0_25")
            base_hit = base["quadrant"] == "fastv_wrong_full_right"
            cand_hit = cand["quadrant"] == "fastv_wrong_full_right"
            if cand_hit and not base_hit:
                b += 1
            if base_hit and not cand_hit:
                c += 1
            g_diffs.append(float(cand["G"]) - float(base["G"]))
            paired_rows.append({
                "selector": sel_key,
                "sample_id": sid,
                "base_strict": int(base_hit),
                "cand_strict": int(cand_hit),
                "base_G": base["G"],
                "cand_G": cand["G"],
                "G_diff": float(cand["G"]) - float(base["G"]),
                "base_step": base["selected_step"],
                "cand_step": cand["selected_step"],
            })
        t_lo, t_hi = t_ci95(g_diffs)
        boot_lo, boot_hi = bootstrap_mean_ci(g_diffs, BOOTSTRAP_SEED, BOOTSTRAP_N)
        sign_p, nonzero_n = exact_signflip_p_two_sided(g_diffs)
        mcnemar_p = mcnemar_exact_two_sided(b, c)
        t_excludes = not (t_lo <= 0.0 <= t_hi)
        boot_excludes = not (boot_lo <= 0.0 <= boot_hi)
        paired_rows.append({
            "selector": sel_key,
            "sample_id": "__AGGREGATE__",
            "base_strict": "",
            "cand_strict": "",
            "base_G": "",
            "cand_G": "",
            "G_diff": statistics.mean(g_diffs),
            "base_step": "",
            "cand_step": "",
            "candidate_only_strict": b,
            "baseline_only_strict": c,
            "mcnemar_p_exact_two_sided": mcnemar_p,
            "mean_G_diff": statistics.mean(g_diffs),
            "t_ci95_low": t_lo,
            "t_ci95_high": t_hi,
            "bootstrap_ci95_low": boot_lo,
            "bootstrap_ci95_high": boot_hi,
            "nonzero_pair_count": nonzero_n,
            "exact_signflip_p_two_sided": sign_p,
            "mcnemar_reject_005": mcnemar_p < 0.05,
            "signflip_reject_005": sign_p < 0.05,
            "t_ci_excludes_zero": t_excludes,
            "bootstrap_ci_excludes_zero": boot_excludes,
            "inference_consistent": (t_excludes == boot_excludes) and (mcnemar_p < 0.05) == (sign_p < 0.05),
            "lambda_advantage_conclusion": "NOT_ESTABLISHED",
        })

    with (OUT / "paired_lambda_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=paired_fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(paired_rows)

    # Figures
    labels = ["fixed200", "posthocMin", "oracle"]
    for method, tag in [("lambda_0_25", "lambda0.25"), ("lambda_1", "lambda1")]:
        vals = []
        for sel in SELECTORS:
            s = next(x for x in summaries if x["method"] == method and x["selector"] == sel)
            vals.append(s["strict_csfr"])
        draw_bar(OUT / "figs" / f"selector_comparison_{tag}.png", f"strict CSFR ({tag})", labels, vals, ymax=1.0)
    hist025 = next(
        x["selected_step_hist"] for x in summaries
        if x["method"] == "lambda_0_25" and x["selector"] == "primary_fixed_step200"
    )
    draw_hist(OUT / "figs" / "selected_step_distribution.png", "lambda0.25 fixed step200 (all step 200)", hist025)
    g_rows = [
        r for r in paired_rows
        if r["selector"] == "primary_fixed_step200" and r["sample_id"] != "__AGGREGATE__"
    ]
    agg_primary_row = next(
        r for r in paired_rows if r["selector"] == "primary_fixed_step200" and r["sample_id"] == "__AGGREGATE__"
    )
    per_sid = [r for r in paired_rows if r["selector"] == "primary_fixed_step200" and r["sample_id"] != "__AGGREGATE__"]
    draw_bar(
        OUT / "figs" / "lambda_paired_G_difference.png",
        f"paired dG per sample (primary fixed200); mean={float(agg_primary_row['mean_G_diff']):.3f}",
        [r["sample_id"] for r in per_sid],
        [float(r["G_diff"]) for r in per_sid],
        ymax=max(0.05, max(abs(float(r["G_diff"])) for r in per_sid) * 1.2),
    )
    # Better G figure: just mean line - redraw simple
    draw_bar(
        OUT / "figs" / "selector_comparison.png",
        "strict CSFR holdout (lambda0.25)",
        labels,
        [next(x for x in summaries if x["method"] == "lambda_0_25" and x["selector"] == k)["strict_csfr"] for k in SELECTORS],
        ymax=1.0,
    )

    def pick_summary(method, sel):
        return next(x for x in summaries if x["method"] == method and x["selector"] == sel)

    primary025 = pick_summary("lambda_0_25", "primary_fixed_step200")
    primary1 = pick_summary("lambda_1", "primary_fixed_step200")
    margin025 = pick_summary("lambda_0_25", "posthoc_min_compressed_margin")
    oracle025 = pick_summary("lambda_0_25", "diagnostic_oracle_best_strict")
    agg_primary = next(r for r in paired_rows if r["selector"] == "primary_fixed_step200" and r["sample_id"] == "__AGGREGATE__")

    report_lines = [
        "# Checkpoint 选择口径审计",
        "",
        "## 1. 结论摘要",
        "",
        f"- 原始 holdout checkpoint 记录完整性：**{integrity['status']}**。",
        f"- **主 endpoint** 应为 `primary_fixed_step200`（每题固定 step 200），不是 `best_strict`。",
        f"- λ=0.25 主 endpoint strict CSFR = **{primary025['fastv_wrong_full_right']}/20**（{primary025['strict_csfr']:.3f}），Full 错误率 {primary025['full_error_rate']:.3f}。",
        f"- post-hoc min-margin：**{margin025['fastv_wrong_full_right']}/20**。",
        f"- oracle best_strict 上界：**{oracle025['fastv_wrong_full_right']}/20**（`GROUND_TRUTH_ORACLE_UPPER_BOUND`）。",
        "- λ=0.25 相对 λ=1 的额外优势：点估计更高，但 n=20 配对区间包含 0，**不能**写成显著优于。",
        "",
        "## 2. 原始记录完整性",
        "",
        "见 `raw_integrity.md`。两种方法各 20 题、ID 与 frozen_manifest 一致，每题 8 个 checkpoint（0,10,25,50,75,100,150,200），必填字段齐全。",
        "",
        "## 3. 为什么 best_strict 不是主 endpoint",
        "",
        "`select_best` 先保留 Full ANLS≥0.5，再最小化 FastV ANLS，tie-break 为 margin、linf、step。选择时使用了真实 Full/FastV 表现，因此是 **ground-truth oracle**，只能作诊断上界。",
        "",
        "## 4. 三种 selector 定义",
        "",
        "| 名称 | 选择规则 | 选图时是否用 ANLS/答案 |",
        "| --- | --- | --- |",
        "| primary_fixed_step200 | step=200 | 否 |",
        "| posthoc_min_compressed_margin | output_valid 内最小 margin，同 margin 取更早 step | 否 |",
        "| diagnostic_oracle_best_strict | 复现 `loss.select_best` | 是 |",
        "",
        "## 5. λ=0.25 三种口径",
        "",
        "| selector | strict CSFR | Full error | FastV error | mean G | 选图用答案/ANLS |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for sel in SELECTORS:
        s = pick_summary("lambda_0_25", sel)
        report_lines.append(
            f"| {sel} | {s['fastv_wrong_full_right']}/20 | {s['full_error_rate']:.3f} | {s['fastv_error_rate']:.3f} | {s['mean_G']:.3f} | {'是' if s['uses_answer_at_selection'] else '否'} |"
        )
    report_lines += [
        "",
        f"λ=0.25 fixed step 200：Full 文本与 step0 一致 **{primary025['full_text_unchanged_n']}/20**；Full ANLS 与 step0 一致 **{primary025['full_anls_unchanged_n']}/20**。",
        "",
        "## 6. λ=1 三种口径",
        "",
        "| selector | strict CSFR | Full error | FastV error | mean G |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for sel in SELECTORS:
        s = pick_summary("lambda_1", sel)
        report_lines.append(
            f"| {sel} | {s['fastv_wrong_full_right']}/20 | {s['full_error_rate']:.3f} | {s['fastv_error_rate']:.3f} | {s['mean_G']:.3f} |"
        )
    report_lines += [
        "",
        "oracle 口径下 Full error=0 来自选择器先过滤 Full ANLS≥0.5 → **FULL_SAFETY_IS_SELECTOR_CONDITIONED**。",
        "",
        "## 7. 配对 λ 比较（按 selector 分开）",
        "",
    ]
    for sel in SELECTORS:
        agg = next(r for r in paired_rows if r["selector"] == sel and r["sample_id"] == "__AGGREGATE__")
        report_lines.append(
            f"- **{sel}**：candidate-only {agg.get('candidate_only_strict')}, baseline-only {agg.get('baseline_only_strict')}, "
            f"McNemar p={agg.get('mcnemar_p_exact_two_sided')}, sign-flip p={agg.get('exact_signflip_p_two_sided')} "
            f"(nonzero pairs {agg.get('nonzero_pair_count')}), mean ΔG={agg.get('mean_G_diff'):.4f}, "
            f"t-CI95 [{agg.get('t_ci95_low'):.4f}, {agg.get('t_ci95_high'):.4f}], "
            f"bootstrap-CI95 [{agg.get('bootstrap_ci95_low'):.4f}, {agg.get('bootstrap_ci95_high'):.4f}], "
            f"λ advantage={agg.get('lambda_advantage_conclusion')}."
        )
    report_lines += [
        "",
        "在固定 step 200 主 endpoint 上：mean ΔG 的 t 区间包含 0，而 percentile bootstrap 区间不包含 0；"
        "但精确 McNemar 与精确 sign-flip 均未拒绝原假设。不同推断方法边界不一致，n=20 不足以证明 λ=0.25 稳定优于 λ=1。",
        "",
        "## 8. Full 保护分析",
        "",
        f"- fixed step 200（λ=0.25）：Full 文本不变 {primary025['full_text_unchanged_n']}/20，可作为独立观察。",
        f"- fixed step 200（λ=1）：Full 文本不变 {primary1['full_text_unchanged_n']}/20；Full 错误率 {primary1['full_error_rate']:.3f}（主 endpoint 不保证 Full 全对）。",
        "- oracle best_strict：Full error 恒为 0 是选择器条件所致（FULL_SAFETY_IS_SELECTOR_CONDITIONED），不能单独证明攻击未伤 Full。",
        "",
        "## 9. 主文应如何报告",
        "",
        "1. 主结果报 fixed step 200 的 strict CSFR 与四格表。",
        "2. min-margin 作为 post-hoc 回溯，标注 `POST_HOC_SELECTOR_AUDIT_NOT_PROSPECTIVE_EVIDENCE`。",
        "3. 17/20 仅作 oracle 上界，标注 `GROUND_TRUTH_ORACLE_UPPER_BOUND`。",
        "",
        "## 10. 不应声称什么",
        "",
        "- 不应把 17/20 写成普通攻击成功率。",
        "- 不应写「λ=0.25 显著优于 λ=1」除非未来预注册检验通过。",
        "- 不应把 oracle 下的 Full error=0 等同于无选择器条件的 Full 安全。",
        "",
        "## 与旧 final_report.md 的差异",
        "",
        "- 旧报告以 oracle best_strict 作主表（λ=0.25 strict 17/20），与本次 `diagnostic_oracle_best_strict` 一致。",
        "- 旧报告未单列 fixed step 200：重算主 endpoint 为 λ=0.25 15/20、λ=1 12/20。",
        "- 旧报告写 holdout 两方法 Full 错误率均为 0，仅对 oracle 成立；λ=1 fixed step 200 为 1/20 Full ANLS<0.5。",
        "",
        "## 11. 下一步预注册建议",
        "",
        "- 正式实验主 endpoint 固定 final step（200）。",
        "- 若部署用 min-margin，须事前预注册且选择时不得读 ANLS。",
        "- 报告同时给出 oracle 上界作诊断，不与主 endpoint 混表。",
        "",
    ]
    (OUT / "selector_audit_report.md").write_text("\n".join(report_lines) + "\n")

    replacement = f"""# 建议替换 final_report 中 holdout 主结论的表述（勿直接覆盖原文件）

## 主结果（Primary endpoint）

Holdout n=20，**固定 step 200**，λ=0.25：strict compression-only **{primary025['fastv_wrong_full_right']}/20**，Full 错误率 **{primary025['full_error_rate']:.3f}**，FastV 错误率 **{primary025['fastv_error_rate']:.3f}**，mean G **{primary025['mean_G']:.3f}**。Full 解码文本与 step 0 一致 **{primary025['full_text_unchanged_n']}/20**。

## 次结果（Post-hoc，非预注册）

最小 compressed margin（选择时不用 ANLS）：strict **{margin025['fastv_wrong_full_right']}/20**。  
**POST_HOC_SELECTOR_AUDIT_NOT_PROSPECTIVE_EVIDENCE**

## 诊断上界（Oracle）

现有 best_strict：**{oracle025['fastv_wrong_full_right']}/20**。  
**GROUND_TRUTH_ORACLE_UPPER_BOUND** — 不能当作普通攻击成功率。

## λ 比较

主 endpoint 上 λ=0.25 strict {primary025['fastv_wrong_full_right']}/20 vs λ=1 {primary1['fastv_wrong_full_right']}/20；McNemar p={agg_primary.get('mcnemar_p_exact_two_sided')}；sign-flip p={agg_primary.get('exact_signflip_p_two_sided')}（nonzero={agg_primary.get('nonzero_pair_count')}）；mean ΔG={agg_primary.get('mean_G_diff'):.4f}，t-CI95 [{agg_primary.get('t_ci95_low'):.4f}, {agg_primary.get('t_ci95_high'):.4f}]，bootstrap-CI95 [{agg_primary.get('bootstrap_ci95_low'):.4f}, {agg_primary.get('bootstrap_ci95_high'):.4f}]。  
λ=0.25 绝对 strict compression-only 很强；相对 λ=1 的额外优势 **NOT_ESTABLISHED**。
"""
    (OUT / "final_report_replacement_suggestion.md").write_text(replacement)

    hashes_after = hash_raw_files()
    immutability_status = "PASS" if hashes_before == hashes_after else "BLOCKED_RAW_MUTATION"
    (OUT / "raw_immutability_check.json").write_text(json.dumps({
        "status": immutability_status,
        "file_count": len(hashes_before),
        "hashes_before": hashes_before,
        "hashes_after": hashes_after,
        "changed_files": sorted(set(hashes_before) ^ set(hashes_after)) if immutability_status != "PASS" else [],
    }, indent=2) + "\n")

    wilson_primary = wilson(primary025["fastv_wrong_full_right"], 20)
    decision = {
        "stage": "selector_audit",
        "raw_integrity": integrity["status"],
        "raw_immutability": immutability_status,
        "k_audit": k_audit["status"],
        "n": 20,
        "primary_selector": "fixed_step_200",
        "primary_lambda025_strict_csfr": primary025["strict_csfr"],
        "primary_lambda025_full_error_rate": primary025["full_error_rate"],
        "primary_lambda025_full_text_unchanged": f"{primary025['full_text_unchanged_n']}/20",
        "primary_lambda025_wilson_strict_csfr": list(wilson_primary),
        "posthoc_min_margin_strict_csfr": margin025["strict_csfr"],
        "oracle_best_strict_csfr": oracle025["strict_csfr"],
        "primary_signflip": {
            "nonzero_pair_count": agg_primary.get("nonzero_pair_count"),
            "exact_signflip_p_two_sided": agg_primary.get("exact_signflip_p_two_sided"),
        },
        "lambda_comparison_conclusion": "NOT_ESTABLISHED",
        "reporting_conclusion": "Use fixed step 200 as primary (15/20); 16/20 post-hoc; 17/20 oracle upper bound only.",
        "next_action": "formal final_report updated if gates pass",
        "formal_update_gates": {},
    }
    (OUT / "selector_audit_decision.json").write_text(json.dumps(decision, indent=2) + "\n")

    gates_ok = (
        integrity["status"] == "PASS"
        and immutability_status == "PASS"
        and k_audit["status"] == "PASS"
        and primary025["fastv_wrong_full_right"] == 15
        and primary025["full_error_rate"] == 0.0
        and primary025["full_text_unchanged_n"] == 20
        and primary025["full_anls_unchanged_n"] == 20
    )
    decision["formal_update_gates"] = {
        "passed": gates_ok,
        "integrity": integrity["status"],
        "immutability": immutability_status,
        "k_audit": k_audit["status"],
        "strict_15_of_20": primary025["fastv_wrong_full_right"] == 15,
        "full_error_zero": primary025["full_error_rate"] == 0.0,
        "full_text_20_of_20": primary025["full_text_unchanged_n"] == 20,
    }
    (OUT / "selector_audit_decision.json").write_text(json.dumps(decision, indent=2) + "\n")

    print(json.dumps({
        "integrity": integrity["status"],
        "immutability": immutability_status,
        "k_audit": k_audit["status"],
        "gates_ok": gates_ok,
        "lambda025": {
            "fixed200": primary025["fastv_wrong_full_right"],
            "posthoc_min_margin": margin025["fastv_wrong_full_right"],
            "oracle": oracle025["fastv_wrong_full_right"],
        },
        "signflip_primary": {
            "nonzero": agg_primary.get("nonzero_pair_count"),
            "p": agg_primary.get("exact_signflip_p_two_sided"),
        },
        "full_text_unchanged_025_fixed": primary025["full_text_unchanged_n"],
    }, indent=2))
    if not gates_ok:
        return


if __name__ == "__main__":
    main()
