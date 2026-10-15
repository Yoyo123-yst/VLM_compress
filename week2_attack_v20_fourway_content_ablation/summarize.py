"""Paired summaries and the pre-registered decision."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
HERE = ROOT / "week2_attack_v20_fourway_content_ablation"
METHODS = ("A", "B", "C", "D")


def read(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def legal_independent(row: dict) -> bool:
    ratio = row["ratios"]["0.25"]
    return bool(
        row["uint8_linf"] > 0
        and ratio["valid"]
        and row["full_text_exact"]
        and row["full_token_exact"]
        and row["full_replay"]
        and row["linf_ok"]
        and row["png_reloadable"]
        and row["uint8_linf"] <= 8
    )


def success_row(row: dict) -> bool:
    return legal_independent(row) and row["ratios"]["0.25"]["semantic_status"] == "CONTENT_FAILURE"


def choose(train: dict, official: dict) -> dict:
    by_checkpoint = {row["checkpoint"]: row for row in official["checkpoints"]}
    candidates = []
    for row in train.get("checkpoints", []):
        other = by_checkpoint.get(row["checkpoint"])
        if other is None:
            continue
        if not legal_independent(other):
            continue
        candidates.append((float(row["attack_surrogate"]), int(row["checkpoint"]), other, row))
    if not candidates:
        return {"checkpoint": None, "official": None, "train": None}
    surrogate, checkpoint, other, train_row = sorted(candidates)[0]
    return {"checkpoint": checkpoint, "attack_surrogate": surrogate, "official": other, "train": train_row}


def noise_rate(sample_id: str) -> tuple[str, float | None, int]:
    payload = read(HERE / "noise_controls" / sample_id / "noise.json")
    if payload is None or payload.get("status") != "DONE":
        return "NOT_MEASURED", None, 0
    flags = [success_row(row) for row in payload["replicates"]]
    if not flags:
        return "NOT_MEASURED", None, 0
    return "MEASURED", sum(flags) / len(flags), len(flags)


def main() -> None:
    admission = read(HERE / "admission.json")
    if admission is None:
        raise SystemExit("admission.json missing")
    admitted = admission["admitted_ids"]
    run = HERE / "full_dev_fourway_v1"
    eval_root = HERE / "independent_eval"
    per_sample = []
    paired_rows = []
    cost_rows = []
    for sample_id in admission["requested_samples"]:
        entry = {"sample_id": sample_id, "admission": next(row["status"] for row in admission["samples"] if row["sample_id"] == sample_id)}
        methods = {}
        for method in METHODS:
            train = read(run / "per_run" / sample_id / f"{method}.json")
            official = read(eval_root / sample_id / f"{method}.json")
            if train is None:
                methods[method] = {"status": "NOT_RUN"}
                continue
            if official is None:
                methods[method] = {"status": train.get("status", "NOT_MEASURED"), "independent_eval": "NOT_RUN"}
                continue
            chosen = choose(train, official)
            official_row = chosen["official"]
            label = None if official_row is None else official_row["ratios"]["0.25"]["semantic_status"]
            methods[method] = {
                "status": train.get("status"),
                "selector_checkpoint": chosen["checkpoint"],
                "attack_surrogate": chosen.get("attack_surrogate"),
                "semantic_status": label if label is not None else "NO_LEGAL_CHECKPOINT",
                "content_failure": bool(official_row and success_row(official_row)),
                "full_text_exact": None if official_row is None else official_row["full_text_exact"],
                "full_token_exact": None if official_row is None else official_row["full_token_exact"],
                "full_replay": None if official_row is None else official_row["full_replay"],
                "anls": None if official_row is None else official_row["ratios"]["0.25"]["anls"],
                "fastv_text": None if official_row is None else official_row["ratios"]["0.25"]["text"],
                "same_positions_as_noeos": train.get("same_positions_as_noeos"),
                "png_sha256": None if official_row is None else official_row["png_sha256"],
                "keep_hash": None if official_row is None else official_row["ratios"]["0.25"]["keep_hash"],
                "keep_jaccard": None if official_row is None else official_row["ratios"]["0.25"]["keep_jaccard_vs_clean"],
                "counters": train.get("counters", "NOT_MEASURED"),
                "accepted": train.get("accepted", "NOT_MEASURED"),
                "attempts": train.get("attempts", "NOT_MEASURED"),
            }
            oracle = any(success_row(row) for row in official["checkpoints"])
            first = next((row["checkpoint"] for row in official["checkpoints"] if success_row(row)), None)
            lost = bool(oracle and not methods[method]["content_failure"])
            methods[method]["oracle_any_content_failure"] = oracle
            methods[method]["first_legal_content_failure"] = first
            methods[method]["success_lost_after_first_success"] = lost
            paired_rows.append({"sample_id": sample_id, "admitted": sample_id in admitted, "method": method, **{key: methods[method][key] for key in ("selector_checkpoint", "semantic_status", "content_failure", "anls", "fastv_text", "same_positions_as_noeos", "full_text_exact", "full_replay")}})
            cost_rows.append({"sample_id": sample_id, "method": method, "accepted": train.get("accepted"), "attempts": train.get("attempts"), "wall_seconds": train.get("wall_seconds"), **(train.get("counters") or {})})
        status, rate, count = noise_rate(sample_id) if sample_id in admitted else ("NOT_RUN", None, 0)
        entry["methods"] = methods
        entry["noise_status"] = status
        entry["noise_content_failure_rate"] = rate
        entry["noise_replicates"] = count
        per_sample.append(entry)

    def rate(method: str) -> tuple[int, int]:
        wins = 0
        for sample_id in admitted:
            row = next(item for item in per_sample if item["sample_id"] == sample_id)
            cell = row["methods"].get(method, {})
            if cell.get("content_failure") is True:
                wins += 1
        return wins, len(admitted)

    rates = {method: rate(method) for method in METHODS}
    noise_values = [item["noise_content_failure_rate"] for item in per_sample if item["sample_id"] in admitted and isinstance(item["noise_content_failure_rate"], float)]
    noise_mean = None if not noise_values else sum(noise_values) / len(noise_values)
    differing = []
    for sample_id in admitted:
        train = read(run / "per_run" / sample_id / "D.json") or {}
        if train.get("same_positions_as_noeos") is False:
            differing.append(sample_id)
    def wins_on(method: str, ids: list[str]) -> int:
        count = 0
        for sample_id in ids:
            row = next(item for item in per_sample if item["sample_id"] == sample_id)
            if row["methods"].get(method, {}).get("content_failure") is True:
                count += 1
        return count

    contrasts = {
        "B_gt_A": rates["B"][0] > rates["A"][0],
        "C_gt_B": rates["C"][0] > rates["B"][0],
        "D_gt_C_on_proper_subset": wins_on("D", differing) > wins_on("C", differing) if differing else False,
        "proper_subset_samples": differing,
    }
    beats_noise = False
    if noise_mean is not None and admitted:
        beats_noise = any((rates[method][0] / rates[method][1]) > noise_mean for method in METHODS)
    if not admitted:
        decision = "BLOCKED_INSUFFICIENT_ADMITTED_SAMPLES"
    elif noise_mean is not None and not beats_noise:
        decision = "CONTENT_OBJECTIVE_NOT_SUPPORTED_ABOVE_RANDOM"
    elif len(admitted) < 12:
        decision = "PILOT_SIGNAL_SUPPORTED" if any(contrasts[key] for key in ("B_gt_A", "C_gt_B", "D_gt_C_on_proper_subset")) else "PILOT_SIGNAL_NOT_SUPPORTED"
    elif contrasts["D_gt_C_on_proper_subset"]:
        decision = "CORE_CONTENT_SUPPORTED_OVER_NOEOS"
    elif contrasts["C_gt_B"] and not contrasts["D_gt_C_on_proper_subset"]:
        decision = "NOEOS_SUPPORTED_ONLY"
    elif contrasts["B_gt_A"] and not contrasts["C_gt_B"]:
        decision = "FIXED_REFERENCE_SUPPORTED_ONLY"
    else:
        decision = "CONTENT_OBJECTIVE_NOT_SUPPORTED"
    (HERE / "per_sample_summary.json").write_text(json.dumps({"samples": per_sample, "rates": {key: {"success": value[0], "denominator": value[1]} for key, value in rates.items()}, "noise_mean": noise_mean, "contrasts": contrasts}, indent=2) + "\n")
    with (HERE / "paired_ablation.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0].keys()) if paired_rows else ["sample_id"])
        writer.writeheader()
        writer.writerows(paired_rows)
    with (HERE / "cost_summary.csv").open("w", newline="") as handle:
        fields = ["sample_id", "method", "accepted", "attempts", "wall_seconds", "forward", "backward", "generate"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in cost_rows:
            writer.writerow({key: row.get(key) for key in fields})
    decision_payload = {
        "status": decision,
        "admitted": admitted,
        "rates": {key: {"success": value[0], "denominator": value[1]} for key, value in rates.items()},
        "noise_mean_content_failure": noise_mean,
        "contrasts": contrasts,
        "unseen": "NOT_OPENED",
        "generalization": "NOT_SUPPORTED",
    }
    (HERE / "final_decision.json").write_text(json.dumps(decision_payload, indent=2) + "\n")
    print(json.dumps(decision_payload, indent=2))


if __name__ == "__main__":
    main()
