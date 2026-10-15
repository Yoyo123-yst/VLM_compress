"""Compare frozen lambda=1 checkpoints with a new lambda run. Same selectors on both."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
sys.path.insert(0, str(ROOT))
from week2_attack_v4_answer_aligned.loss import select_best  # noqa: E402

OUT = ROOT / "week2_attack_v4_relaxed_full"
V4_CKPT = ROOT / "week2_attack_v4_answer_aligned" / "results" / "checkpoints.jsonl"
STRICT_HELD = ["50743", "33923", "5169"]
STRICT_NEW = ["4912", "56776", "49242"]
WRONG = 0.5


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def by_sample(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["sample_id"], []).append(row)
    for sample_id in grouped:
        grouped[sample_id].sort(key=lambda row: int(row["step"]))
    return grouped


def quadrant(full_anls: float, fast_anls: float) -> str:
    full_wrong = full_anls < WRONG
    fast_wrong = fast_anls < WRONG
    if fast_wrong and not full_wrong:
        return "fastv_wrong_full_right"
    if (not fast_wrong) and full_wrong:
        return "fastv_right_full_wrong"
    if fast_wrong and full_wrong:
        return "both_wrong"
    return "both_right"


def degradation(rows: list[dict]) -> dict:
    clean = rows[0]
    if int(clean["step"]) != 0:
        raise RuntimeError("checkpoint series does not start at 0")
    s_f = float(clean["anls_full"])
    s_c = float(clean["anls_fastv"])
    for row in rows:
        row["D_F"] = s_f - float(row["anls_full"])
        row["D_C"] = s_c - float(row["anls_fastv"])
        row["G"] = row["D_C"] - row["D_F"]
        row["quadrant"] = quadrant(float(row["anls_full"]), float(row["anls_fastv"]))
    return {"S_F_clean": s_f, "S_C_clean": s_c}


def best_gap(rows: list[dict]) -> dict:
    valid = [row for row in rows if int(row["output_valid"])]
    pool = valid or rows
    return sorted(pool, key=lambda row: (-float(row["G"]), -float(row["anls_full"]), float(row["anls_fastv"]), int(row["step"])))[0]


def pick(rows: list[dict]) -> dict:
    degradation(rows)
    final = rows[-1]
    strict_rows = []
    for row in rows:
        strict_rows.append({
            "step": int(row["step"]),
            "anls_fastv": float(row["anls_fastv"]),
            "anls_full": float(row["anls_full"]),
            "margin": float(row["margin"]),
            "linf": float(row["linf"]),
            "output_valid": bool(int(row["output_valid"])),
            "label_if_chosen": row["label_if_chosen"],
        })
    strict = select_best(strict_rows)
    chosen_strict = next(row for row in rows if int(row["step"]) == int(strict["step"]))
    return {"final_step": final, "best_strict": chosen_strict, "best_gap_diagnostic": best_gap(rows)}


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def summarize(selected: dict[str, dict], key: str) -> dict:
    rows = [selected[sid][key] for sid in selected]
    cells = [row["quadrant"] for row in rows]
    n = len(rows)
    gap = cells.count("fastv_wrong_full_right") / n - cells.count("fastv_right_full_wrong") / n
    return {
        "n": n,
        "strict_csfr": cells.count("fastv_wrong_full_right") / n,
        "reverse_failure": cells.count("fastv_right_full_wrong") / n,
        "both_wrong": cells.count("both_wrong") / n,
        "both_right": cells.count("both_right") / n,
        "full_error_rate": sum(float(row["anls_full"]) < WRONG for row in rows) / n,
        "fastv_error_rate": sum(float(row["anls_fastv"]) < WRONG for row in rows) / n,
        "error_rate_gap": gap,
        "mean_DF": mean([float(row["D_F"]) for row in rows]),
        "mean_DC": mean([float(row["D_C"]) for row in rows]),
        "mean_G": mean([float(row["G"]) for row in rows]),
        "invalid": sum(not int(row["output_valid"]) for row in rows),
        "ids_strict": [row["sample_id"] for row in rows if row["quadrant"] == "fastv_wrong_full_right"],
    }


def decide(lam1: dict, lam025: dict, grad_dominated: bool) -> dict:
    strict = lam025["best_strict"]
    base = lam1["best_strict"]
    kept = sum(sid in strict["ids_strict"] for sid in STRICT_HELD)
    gained = sum(sid in strict["ids_strict"] and sid not in base["ids_strict"] for sid in STRICT_NEW)
    go = (
        strict["invalid"] == 0
        and strict["full_error_rate"] <= 1 / 6 + 1e-9
        and (strict["error_rate_gap"] >= 4 / 6 - 1e-9 or strict["strict_csfr"] >= 4 / 6 - 1e-9)
        and kept >= 2
        and gained >= 1
    )
    weak = (
        (strict["mean_G"] - base["mean_G"]) >= 0.05 - 1e-9
        or (strict["error_rate_gap"] - base["error_rate_gap"]) >= 1 / 6 - 1e-9
        or gained >= 1
        or grad_dominated
    )
    if go:
        decision = "RELAXED_GO_TO_HOLDOUT"
        action = "freeze holdout and run the paired comparison"
    elif weak:
        decision = "RELAXED_WEAK_BRANCH_RUN"
        action = "choose one stage C branch"
    else:
        decision = "RELAXED_NO_GO"
        action = "stop"
    return {
        "decision": decision,
        "next_action": action,
        "kept_of_three": kept,
        "new_strict": gained,
        "grad_full_dominates": grad_dominated,
    }


def draw_bars(path: Path, title: str, labels: list[str], series: list[tuple[str, list[float]]]):
    width, height = 900, 420
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 12), title, fill="black")
    colors = [(30, 90, 160), (180, 80, 40), (40, 130, 80)]
    left, top, right, bottom = 60, 50, 860, 360
    values = [value for _, seq in series for value in seq]
    hi = max(0.05, max(values))
    lo = min(0.0, min(values))
    span = hi - lo
    group = (right - left) / max(1, len(labels))
    bar = group / (len(series) + 1)
    for index, label in enumerate(labels):
        x0 = left + index * group
        draw.text((x0, bottom + 8), label, fill="black")
        for s_index, (_, seq) in enumerate(series):
            value = seq[index]
            y = bottom - (value - lo) / span * (bottom - top)
            zero = bottom - (0 - lo) / span * (bottom - top)
            xa = x0 + s_index * bar
            draw.rectangle([xa, min(y, zero), xa + bar - 4, max(y, zero)], fill=colors[s_index])
    image.save(path)


def main():
    tag = sys.argv[1]
    grad_path = OUT / "diagnostics" / "weighted_gradients.jsonl"
    dominated = 0
    if grad_path.exists():
        for line in grad_path.read_text().splitlines():
            row = json.loads(line)
            if row["image"] == "best" and float(row["full_over_comp"]) > 1.0:
                dominated += 1
    lam1 = {sid: pick(rows) for sid, rows in by_sample(load_jsonl(V4_CKPT)).items()}
    new_rows = []
    run_dir = OUT / "runs" / tag
    for path in sorted(run_dir.glob("checkpoints_s*.jsonl")):
        new_rows.extend(load_jsonl(path))
    lam025 = {sid: pick(rows) for sid, rows in by_sample(new_rows).items()}
    if set(lam025) != set(lam1):
        raise SystemExit(f"sample mismatch {sorted(lam025)} vs {sorted(lam1)}")
    summary = {
        "lambda_1": {key: summarize(lam1, key) for key in ("final_step", "best_strict", "best_gap_diagnostic")},
        "lambda_0_25": {key: summarize(lam025, key) for key in ("final_step", "best_strict", "best_gap_diagnostic")},
    }
    verdict = decide(summary["lambda_1"], summary["lambda_0_25"], dominated >= 4)
    per_sample = []
    for sid in ["50743", "4912", "56776", "49242", "33923", "5169"]:
        for name, chosen in (("lambda_1", lam1), ("lambda_0_25", lam025)):
            for rule in ("final_step", "best_strict", "best_gap_diagnostic"):
                row = chosen[sid][rule]
                per_sample.append({
                    "method": name, "rule": rule, "sample_id": sid,
                    "step": row["step"], "pred_full": row["pred_full"], "pred_fastv": row["pred_fastv"],
                    "anls_full": row["anls_full"], "anls_fastv": row["anls_fastv"],
                    "D_F": row["D_F"], "D_C": row["D_C"], "G": row["G"],
                    "quadrant": row["quadrant"], "margin": row["margin"],
                    "keep_hash": row["keep_hash"], "aux_share": row["aux_share"],
                    "evidence_keep": row["evidence_keep"], "evidence_share": row["evidence_share"],
                    "linf": row["linf"], "output_valid": row["output_valid"],
                })
    dest = OUT / "results" / tag
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / "per_sample.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_sample[0]))
        writer.writeheader()
        writer.writerows(per_sample)
    with (dest / "checkpoints.csv").open("w", newline="") as handle:
        fields = ["sample_id", "step", "anls_full", "anls_fastv", "D_F", "D_C", "G", "quadrant", "pred_full", "pred_fastv", "margin", "output_valid"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in new_rows:
            degradation(by_sample(new_rows)[row["sample_id"]])
        for sid, rows in by_sample(new_rows).items():
            degradation(rows)
            for row in rows:
                writer.writerow({key: row[key] for key in fields})
    (OUT / "figs").mkdir(exist_ok=True)
    labels = ["50743", "4912", "56776", "49242", "33923", "5169"]
    g1 = [float(lam1[sid]["best_strict"]["G"]) for sid in labels]
    g2 = [float(lam025[sid]["best_strict"]["G"]) for sid in labels]
    draw_bars(OUT / "figs" / "lambda_comparison.png", "G at best_strict: lambda 1 vs 0.25", labels, [("lambda 1", g1), ("lambda 0.25", g2)])
    df = [float(lam025[sid]["best_strict"]["D_F"]) for sid in labels]
    dc = [float(lam025[sid]["best_strict"]["D_C"]) for sid in labels]
    draw_bars(OUT / "figs" / "full_vs_fastv_degradation.png", "lambda 0.25 best_strict degradation", labels, [("D_F", df), ("D_C", dc)])
    counts = [
        summary["lambda_0_25"]["best_strict"]["strict_csfr"] * 6,
        summary["lambda_0_25"]["best_strict"]["reverse_failure"] * 6,
        summary["lambda_0_25"]["best_strict"]["both_wrong"] * 6,
        summary["lambda_0_25"]["best_strict"]["both_right"] * 6,
    ]
    image = Image.new("RGB", (720, 360), "white")
    draw = ImageDraw.Draw(image)
    names = ["FV wrong Full right", "FV right Full wrong", "both wrong", "both right"]
    for index, (name, count) in enumerate(zip(names, counts)):
        draw.rectangle([40, 40 + index * 70, 40 + count * 80, 90 + index * 70], fill=(30, 90, 160))
        draw.text((50 + count * 80, 55 + index * 70), f"{name}: {int(count)}", fill="black")
    image.save(OUT / "figs" / "error_quadrants.png")
    decision = {
        "stage": "B",
        "health": "PASS" if summary["lambda_0_25"]["best_strict"]["invalid"] == 0 else "invalid",
        "completed_samples": 6,
        "invalid_outputs": summary["lambda_0_25"]["best_strict"]["invalid"],
        "replay_pass": None,
        "strict_csfr": summary["lambda_0_25"]["best_strict"]["strict_csfr"],
        "full_error_rate": summary["lambda_0_25"]["best_strict"]["full_error_rate"],
        "fastv_error_rate": summary["lambda_0_25"]["best_strict"]["fastv_error_rate"],
        "error_rate_gap": summary["lambda_0_25"]["best_strict"]["error_rate_gap"],
        "mean_DF": summary["lambda_0_25"]["best_strict"]["mean_DF"],
        "mean_DC": summary["lambda_0_25"]["best_strict"]["mean_DC"],
        "mean_G": summary["lambda_0_25"]["best_strict"]["mean_G"],
        "decision": verdict["decision"],
        "evidence": [
            f"lambda1 strict CSFR {summary['lambda_1']['best_strict']['strict_csfr']}",
            f"lambda0.25 strict CSFR {summary['lambda_0_25']['best_strict']['strict_csfr']}",
            f"mean G {summary['lambda_1']['best_strict']['mean_G']} -> {summary['lambda_0_25']['best_strict']['mean_G']}",
            f"gap {summary['lambda_1']['best_strict']['error_rate_gap']} -> {summary['lambda_0_25']['best_strict']['error_rate_gap']}",
            f"kept {verdict['kept_of_three']}/3 new {verdict['new_strict']}",
            f"best checkpoints with full grad L2 > comp grad L2: {dominated}/6",
        ],
        "next_action": verdict["next_action"],
        "next_command": "",
        "summary": summary,
        "verdict": verdict,
    }
    (dest / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    lines = ["# lambda 0.25", "", f"决定：{verdict['decision']}", ""]
    lines.append("主比较用 best_strict，和 v4 同一条规则。final_step 与 best_gap_diagnostic 只作对照。")
    lines.append("")
    for method, block in summary.items():
        row = block["best_strict"]
        lines.append(
            f"- {method} best_strict: strict CSFR {row['strict_csfr']:.3f}, Full error {row['full_error_rate']:.3f}, "
            f"FastV error {row['fastv_error_rate']:.3f}, gap {row['error_rate_gap']:.3f}, "
            f"mean D_F {row['mean_DF']:.3f}, D_C {row['mean_DC']:.3f}, G {row['mean_G']:.3f}"
        )
    (dest / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"decision": verdict["decision"], "kept": verdict["kept_of_three"], "new": verdict["new_strict"]}, indent=2))


if __name__ == "__main__":
    main()
