"""Score the frozen pristine60 gates. Does not launch attacks or change earlier trees."""

from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter
from pathlib import Path

from week2_attack_v7_pristine60.load_samples import manifest
from week2_attack_v7_pristine60.parents import parent_hashes

V7 = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project/week2_attack_v7_pristine60")
WRONG = 0.5
EPS = 8 / 255
BOOT_SEED = 2026100907


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1))
    return min(1.0, 2 * tail / (2 ** n))


def bootstrap_ci(values: list[float]) -> tuple[float, float]:
    rng = random.Random(BOOT_SEED)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(10000))
    return means[int(0.025 * 10000)], means[int(0.975 * 10000) - 1]


def read_csv_glob(pattern: str) -> list[dict]:
    rows = []
    for path in sorted(V7.glob(pattern)):
        with path.open() as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def read_replay(pattern: str) -> dict[str, dict]:
    found = {}
    for path in sorted(V7.glob(pattern)):
        for row in json.loads(path.read_text()):
            found[row["sample_id"]] = row
    return found


def truthy(value) -> bool:
    return str(value).lower() in {"1", "true", "yes"}


def g_value(clean_full: float, clean_fast: float, full: float, fast: float) -> float:
    return (clean_fast - fast) - (clean_full - full)


def strict_of(row: dict, spec: dict, replay_ok: bool) -> bool:
    if not row or row.get("status") not in {"SELECTED", ""}:
        return False
    if row.get("status") == "NO_FEASIBLE_RANDOM_CANDIDATE" or row.get("status") == "NO_FEASIBLE_NONZERO_CHECKPOINT":
        return False
    if row.get("status") == "SELECTED_CANDIDATE_REPLAY_FAILED":
        return False
    try:
        linf = float(row.get("pixel_linf_uint8") or row.get("uint8_linf"))
        anls_full = float(row["anls_full"])
        anls_fast = float(row["anls_fastv"])
    except (TypeError, ValueError):
        return False
    return (
        anls_full >= WRONG
        and anls_fast < WRONG
        and row["pred_full"] == spec["clean_full_pred"]
        and truthy(row["output_valid"])
        and replay_ok
        and 0 < linf <= EPS + 1e-6
        and float(spec["clean_full_anls"]) >= WRONG
        and float(spec["clean_fastv_anls"]) >= WRONG
    )


def error_of(row: dict, spec: dict, replay_ok: bool, is_strict: bool) -> str:
    if is_strict:
        return "STRICT_SUCCESS"
    if not row or row.get("status") in {"NO_FEASIBLE_NONZERO_CHECKPOINT", "NO_FEASIBLE_RANDOM_CANDIDATE"}:
        return row.get("status", "MISSING")
    if not truthy(row.get("output_valid", 0)):
        return "INVALID_OUTPUT"
    if row.get("pred_full") != spec["clean_full_pred"]:
        return "FULL_TEXT_CHANGED"
    try:
        if float(row["anls_full"]) < WRONG:
            return "FULL_ANLS_BELOW_0.5"
        linf = float(row.get("pixel_linf_uint8") or row.get("uint8_linf") or 0)
    except (TypeError, ValueError):
        return "UNREADABLE"
    if not (0 < linf <= EPS + 1e-6):
        return "LINF_OR_ZERO"
    if not replay_ok:
        return "SELECTED_CANDIDATE_REPLAY_FAILED"
    if float(row["anls_fastv"]) >= WRONG:
        return "FASTV_STILL_CORRECT"
    return "OTHER"


def main():
    specs = {row["sample_id"]: row for row in manifest()["samples"]}
    selected = {row["sample_id"]: row for row in read_csv_glob("attacks/selector/selected.csv")}
    replay = read_replay("attacks/replay/replay_s*.json")
    single = {row["sample_id"]: row for row in read_csv_glob("controls/single/**/single.csv")}
    traj = {row["sample_id"]: row for row in read_csv_glob("controls/trajectory/**/trajectory.csv")}
    if set(specs) != set(selected) or set(specs) != set(replay) or set(specs) != set(single) or set(specs) != set(traj):
        missing = {
            "selected": sorted(set(specs) - set(selected)),
            "replay": sorted(set(specs) - set(replay)),
            "single": sorted(set(specs) - set(single)),
            "trajectory": sorted(set(specs) - set(traj)),
        }
        (V7 / "final_decision.json").write_text(json.dumps({
            "decision": "BLOCKED_RUNTIME_OR_INTEGRITY_FAILURE",
            "missing": missing,
        }, indent=2) + "\n")
        print(json.dumps({"decision": "BLOCKED_RUNTIME_OR_INTEGRITY_FAILURE", "missing": missing}))
        return
    analysis = V7 / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    samples_out = []
    paired_traj = []
    paired_single = []
    errors = []
    for sid, spec in specs.items():
        row = selected[sid]
        replay_ok = bool(replay[sid].get("ok"))
        is_strict = strict_of(row, spec, replay_ok)
        full_exact = row.get("pred_full") == spec["clean_full_pred"] and row.get("status") == "SELECTED"
        linf = float(row["pixel_linf_uint8"]) if row.get("pixel_linf_uint8") else 0.0
        entry = {
            "sample_id": sid,
            "strict": int(is_strict),
            "full_exact": int(full_exact),
            "replay_ok": int(replay_ok),
            "nonzero": int(linf > 0),
            "linf_ok": int(0 < linf <= EPS + 1e-6),
            "invalid": int(not truthy(row.get("output_valid", 0))),
            "step": row.get("step", ""),
            "pred_full": row.get("pred_full", ""),
            "pred_fastv": row.get("pred_fastv", ""),
            "anls_full": row.get("anls_full", ""),
            "anls_fastv": row.get("anls_fastv", ""),
            "error": error_of(row, spec, replay_ok, is_strict),
        }
        samples_out.append(entry)
        errors.append({"sample_id": sid, "error": entry["error"]})
        g_attack = g_value(float(spec["clean_full_anls"]), float(spec["clean_fastv_anls"]), float(row["anls_full"]), float(row["anls_fastv"])) if row.get("anls_full") else None
        for dest, control, name in (
            (paired_traj, traj[sid], "trajectory"),
            (paired_single, single[sid], "single"),
        ):
            c_replay = truthy(control.get("replay_ok", 0)) and control.get("status") != "NO_FEASIBLE_RANDOM_CANDIDATE"
            if control.get("status") == "NO_FEASIBLE_RANDOM_CANDIDATE":
                c_strict = False
                g_control = None
            else:
                c_strict = strict_of({**control, "status": "SELECTED", "pixel_linf_uint8": control.get("uint8_linf")}, spec, c_replay)
                g_control = g_value(float(spec["clean_full_anls"]), float(spec["clean_fastv_anls"]), float(control["anls_full"]), float(control["anls_fastv"]))
            dest.append({
                "sample_id": sid,
                "attack_strict": int(is_strict),
                f"{name}_strict": int(c_strict),
                "G_attack": g_attack,
                "G_control": g_control,
                "G_diff": None if g_attack is None or g_control is None else g_attack - g_control,
            })
    n = len(samples_out)
    strict_n = sum(row["strict"] for row in samples_out)
    lo, hi = wilson(strict_n, n)
    b = sum(row["attack_strict"] and not row["trajectory_strict"] for row in paired_traj)
    c = sum(row["trajectory_strict"] and not row["attack_strict"] for row in paired_traj)
    b_s = sum(row["attack_strict"] and not row["single_strict"] for row in paired_single)
    c_s = sum(row["single_strict"] and not row["attack_strict"] for row in paired_single)
    gaps = [row["G_diff"] for row in paired_traj if row["G_diff"] is not None]
    gates = {
        "strict": strict_n >= 38,
        "wilson_lower_gt_0.5": lo > 0.5,
        "full_exact": sum(row["full_exact"] for row in samples_out) == n,
        "replay": sum(row["replay_ok"] for row in samples_out) == n,
        "nonzero": sum(row["nonzero"] for row in samples_out) == n,
        "linf": sum(row["linf_ok"] for row in samples_out) == n,
        "invalid0": sum(row["invalid"] for row in samples_out) == 0,
        "mcnemar_direction": b > c,
        "mcnemar_p": mcnemar(b, c) < 0.05,
    }
    cells = []
    for path in sorted(V7.glob("mechanism/cells_s*.json")):
        cells.extend(json.loads(path.read_text()))
    by_cell = {row["sample_id"]: row for row in cells}
    mechanism_rows = []
    for entry in samples_out:
        cell = by_cell.get(entry["sample_id"], {})
        mechanism_rows.append({
            "sample_id": entry["sample_id"],
            "strict": entry["strict"],
            "mechanism": cell.get("mechanism", "MISSING"),
            "error": entry["error"],
            "jaccard": cell.get("jaccard", ""),
            "anls_CC": cell.get("anls_CC", ""),
            "anls_CA": cell.get("anls_CA", ""),
            "anls_AC": cell.get("anls_AC", ""),
            "anls_AA": cell.get("anls_AA", ""),
        })
    strict_cells = [row for row in mechanism_rows if row["strict"] and row["mechanism"] != "MISSING"]
    content = sum(row["mechanism"] in {"CONTENT_SUFFICIENT", "BOTH_INDEPENDENTLY_SUFFICIENT"} for row in strict_cells)
    selection = sum(row["mechanism"] == "SELECTION_SUFFICIENT" for row in strict_cells)
    interaction = sum(row["mechanism"] == "INTERACTION_REQUIRED" for row in strict_cells)
    mech_n = len(strict_cells)
    repeat_rows = []
    repeat_selected = {row["sample_id"]: row for row in read_csv_glob("repeat10/selector/selected.csv")}
    repeat_replay = read_replay("repeat10/replay/replay_s*.json")
    for sid, second in repeat_selected.items():
        first = selected[sid]
        second_ok = bool(repeat_replay.get(sid, {}).get("ok"))
        first_strict = next(row["strict"] for row in samples_out if row["sample_id"] == sid)
        second_strict = int(strict_of(second, specs[sid], second_ok))
        repeat_rows.append({
            "sample_id": sid,
            "strict_agree": int(first_strict == second_strict),
            "full_exact_agree": int((first.get("pred_full") == specs[sid]["clean_full_pred"]) == (second.get("pred_full") == specs[sid]["clean_full_pred"])),
            "step_agree": int(str(first.get("step")) == str(second.get("step"))),
            "fastv_agree": int(first.get("pred_fastv") == second.get("pred_fastv")),
            "png_agree": int(first.get("png_sha256") == second.get("png_sha256")),
            "keep_agree": int(first.get("keep_hash") == second.get("keep_hash")),
            "both_strict": int(first_strict and second_strict),
            "first_only": int(first_strict and not second_strict),
            "second_only": int(second_strict and not first_strict),
            "neither": int(not first_strict and not second_strict),
        })
    repeat_n = len(repeat_rows)
    agree = sum(row["strict_agree"] for row in repeat_rows)
    repeat_label = "REPEATABILITY_DIAGNOSTIC_PASS" if repeat_n == 10 and agree >= 8 else "REPEATABILITY_WEAK"
    passed = all(gates.values()) and len(cells) == n and repeat_n == 10
    if not passed:
        decision = "PRISTINE60_ATTACK_NOT_SUPPORTED" if len(cells) == n and repeat_n == 10 else "BLOCKED_RUNTIME_OR_INTEGRITY_FAILURE"
    elif repeat_label == "REPEATABILITY_WEAK":
        decision = "PRISTINE60_SUPPORTS_COMP_ONLY_BUT_REPEATABILITY_WEAK"
    else:
        decision = "PRISTINE60_SUPPORTS_SOURCE_BOUNDED_COMP_ONLY_ATTACK"
    def write_csv(path: Path, rows: list[dict]):
        if not rows:
            path.write_text("")
            return
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    write_csv(analysis / "all_samples.csv", samples_out)
    write_csv(V7 / "all_samples.csv", samples_out)
    checkpoints = []
    for path in sorted(V7.glob("attacks/shard_s*/*/checkpoints.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                checkpoints.append({key: row.get(key) for key in ("sample_id", "step", "margin", "pred_full", "pred_fastv", "anls_full", "anls_fastv", "output_valid", "linf", "keep_hash", "k", "png_sha256")})
    write_csv(V7 / "all_checkpoints.csv", checkpoints)
    write_csv(V7 / "paired_attack_vs_random_trajectory.csv", paired_traj)
    write_csv(V7 / "paired_attack_vs_single_noise.csv", paired_single)
    write_csv(V7 / "repeatability10.csv", repeat_rows)
    write_csv(V7 / "mechanism_2x2.csv", mechanism_rows)
    write_csv(V7 / "error_taxonomy.csv", errors)
    oracle_n = 0
    step200_n = 0
    grouped = {}
    for row in checkpoints:
        grouped.setdefault(row["sample_id"], []).append(row)
    for sid, rows in grouped.items():
        spec = specs[sid]
        hit = False
        for row in rows:
            if int(row["step"]) == 0:
                continue
            faux = {
                "status": "SELECTED", "pred_full": row["pred_full"], "anls_full": row["anls_full"],
                "anls_fastv": row["anls_fastv"], "output_valid": row["output_valid"], "pixel_linf_uint8": row["linf"],
            }
            if strict_of(faux, spec, True):
                hit = True
            if int(row["step"]) == 200 and strict_of(faux, spec, True):
                step200_n += 1
        oracle_n += int(hit)
    stats = {
        "n": n,
        "primary_strict": strict_n,
        "wilson95": [lo, hi],
        "full_exact": sum(row["full_exact"] for row in samples_out),
        "replay": sum(row["replay_ok"] for row in samples_out),
        "nonzero": sum(row["nonzero"] for row in samples_out),
        "linf": sum(row["linf_ok"] for row in samples_out),
        "invalid": sum(row["invalid"] for row in samples_out),
        "full_text_change": sum(row["error"] == "FULL_TEXT_CHANGED" for row in samples_out),
        "trajectory_attack_only": b,
        "trajectory_noise_only": c,
        "trajectory_mcnemar_p": mcnemar(b, c),
        "single_attack_only": b_s,
        "single_noise_only": c_s,
        "single_mcnemar_p": mcnemar(b_s, c_s),
        "mean_G_diff_vs_trajectory": None if not gaps else sum(gaps) / len(gaps),
        "bootstrap_ci95_G_diff": None if not gaps else list(bootstrap_ci(gaps)),
        "fixed_step200_saved_generation": step200_n,
        "oracle_upper_bound_count": oracle_n,
        "oracle_label": "GROUND_TRUTH_ORACLE_UPPER_BOUND",
        "mechanism_among_strict": {
            "n": mech_n,
            "content_sufficient_inclusive": content,
            "content_wilson95": list(wilson(content, mech_n)) if mech_n else None,
            "selection_sufficient": selection,
            "selection_wilson95": list(wilson(selection, mech_n)) if mech_n else None,
            "interaction": interaction,
            "interaction_wilson95": list(wilson(interaction, mech_n)) if mech_n else None,
            "counts": dict(Counter(row["mechanism"] for row in strict_cells)),
        },
        "repeatability": {
            "label": repeat_label,
            "strict_agree": agree,
            "n": repeat_n,
            "both_strict": sum(row["both_strict"] for row in repeat_rows),
            "first_only": sum(row["first_only"] for row in repeat_rows),
            "second_only": sum(row["second_only"] for row in repeat_rows),
            "neither": sum(row["neither"] for row in repeat_rows),
        },
        "gates": gates,
        "decision": decision,
        "claim_allowed": "source-bounded compression-only attack" if passed else "none",
        "selection_stable_claim": False,
    }
    (V7 / "statistics.json").write_text(json.dumps(stats, indent=2) + "\n")
    (analysis / "statistics.json").write_text(json.dumps(stats, indent=2) + "\n")
    parents = parent_hashes()
    frozen_parents = json.loads((V7 / "FROZEN.json").read_text())["parent_hashes"]
    changed = sorted(rel for rel, digest in parents.items() if frozen_parents.get(rel) != digest)
    (V7 / "raw_immutability_check.json").write_text(json.dumps({
        "result": "PASS" if not changed else "FAIL",
        "n": len(parents),
        "changed": changed,
    }, indent=2) + "\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig_dir = V7 / "figures"
        fig_dir.mkdir(exist_ok=True)
        labels = ["attack", "trajectory", "single noise"]
        vals = [strict_n, sum(row["trajectory_strict"] for row in paired_traj), sum(row["single_strict"] for row in paired_single)]
        plt.figure(figsize=(6, 4))
        plt.bar(labels, vals, color=["#3c6e71", "#d9b08c", "#8c8c8c"])
        plt.ylabel("strict count")
        plt.ylim(0, 60)
        plt.tight_layout()
        plt.savefig(fig_dir / "strict_counts.png", dpi=120)
        plt.close()
        counts = Counter(row["mechanism"] for row in strict_cells)
        plt.figure(figsize=(8, 4))
        plt.bar(list(counts.keys()), list(counts.values()))
        plt.xticks(rotation=20, ha="right")
        plt.ylabel("strict successes")
        plt.tight_layout()
        plt.savefig(fig_dir / "mechanism_among_strict.png", dpi=120)
        plt.close()
    except Exception as exc:
        (V7 / "figures").mkdir(exist_ok=True)
        (V7 / "figures" / "FIGURES_SKIPPED.txt").write_text(str(exc) + "\n")
    (V7 / "final_decision.json").write_text(json.dumps({
        "decision": decision,
        "primary_strict": f"{strict_n}/{n}",
        "wilson95": [lo, hi],
        "repeatability": repeat_label,
        "selection_stable_claim": False,
        "gates": gates,
    }, indent=2) + "\n")
    print(json.dumps({"decision": decision, "strict": strict_n, "wilson_lo": lo, "repeat": repeat_label, "mcnemar": mcnemar(b, c)}))


if __name__ == "__main__":
    main()
