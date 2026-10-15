"""Build the frozen v7d tables, figures, and decision from disk-replay scores."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project/week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation")
ROOT = HERE.parent
DEV = ["1025", "2008", "6194", "7006", "39204", "47450", "47567", "50779", "51133", "60700", "62941", "64506"]
METHODS = ("B_NATIVE", "NATIVE_FDP", "B_PLUS")
RATIOS = ("0.50", "0.35", "0.25", "0.15")
PARENTS = (
    "week2_attack_v5b_expanded_pristine40",
    "week2_attack_v6_survivor_upgrade",
    "week2_attack_v6b_soft_full_anchor",
    "week2_attack_v6c_split_calibration",
    "week2_attack_v7_pristine60",
    "week2_attack_v7_budget_conditioned",
    "week2_attack_v7a_fdp_alignment_audit",
    "week2_attack_v7b_native_cache_margin_admission",
    "week2_attack_v7c_native60_readonly_remeasurement",
)


def load_method(name: str) -> dict[str, dict]:
    folder = HERE / "scored" / name
    rows = {}
    for sample_id in DEV:
        path = folder / f"{sample_id}.json"
        rows[sample_id] = json.loads(path.read_text()) if path.exists() else {"sample_id": sample_id, "status": "NOT_RUN"}
    return rows


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def hard_ok(row: dict) -> bool:
    return all(int(row.get(key, 0)) == expected for key, expected in (
        ("full_exact", 1), ("replay", 1), ("nonzero", 1), ("linf_ok", 1), ("invalid", 0),
    ))


def train_delta(row: dict) -> float:
    return mean([row["by_ratio"]["0.25"]["delta"], row["by_ratio"]["0.35"]["delta"]])


def summarize(name: str, rows: dict[str, dict]) -> dict:
    present = [rows[sid] for sid in DEV if rows[sid].get("status") != "NOT_RUN" and "full_exact" in rows[sid]]
    if len(present) != 12:
        return {"method": name, "status": "NOT_RUN", "n": len(present)}
    return {
        "method": name,
        "status": "RUN",
        "n": 12,
        "full_exact": sum(int(row["full_exact"]) for row in present),
        "replay": sum(int(row["replay"]) for row in present),
        "nonzero_linf": sum(int(row["nonzero"] and row["linf_ok"]) for row in present),
        "invalid": sum(int(row["invalid"]) for row in present),
        "strict": sum(int(row["strict_train"]) for row in present),
        "pollution": sum(int(row["pollution"]) for row in present),
        "mean_delta": mean([train_delta(row) for row in present]),
        "hard_pass": all(hard_ok(row) for row in present),
    }


def transitions(base: dict, other: dict, field: str) -> list[dict]:
    out = []
    for sid in DEV:
        left = int(base[sid][field])
        right = int(other[sid][field])
        out.append({"sample_id": sid, "B_NATIVE": left, "other": right, "change": right - left})
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bars(path: Path, labels: list[str], values: list[int], title: str) -> None:
    width, height = 720, 360
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 16), title, fill="black")
    top, bottom, left = 50, 300, 80
    peak = max(values + [1])
    span = (width - 140) / len(values)
    for index, (label, value) in enumerate(zip(labels, values)):
        x = left + index * span
        bar_h = 0 if peak == 0 else (bottom - top) * value / peak
        draw.rectangle((x, bottom - bar_h, x + span * 0.6, bottom), fill=(40, 90, 160))
        draw.text((x, bottom + 8), label, fill="black")
        draw.text((x, bottom - bar_h - 16), str(value), fill="black")
    image.save(path)


def parent_hashes() -> dict:
    files = {}
    for name in PARENTS:
        base = ROOT / name
        for path in sorted(base.rglob("*")):
            if path.is_file():
                files[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"count": len(files), "files": files}


def paired_changes(other: str) -> dict:
    base, rows = load_method("B_NATIVE"), load_method(other)
    gains = [sid for sid in DEV if rows[sid]["strict_train"] and not base[sid]["strict_train"]]
    losses = [sid for sid in DEV if base[sid]["strict_train"] and not rows[sid]["strict_train"]]
    return {"gains": gains, "losses": losses, "net": len(gains) - len(losses)}


def bplus_eligible(summaries: dict[str, dict], changes: dict) -> bool:
    base, plus = summaries["B_NATIVE"], summaries["B_PLUS"]
    return bool(
        plus.get("hard_pass")
        and plus["pollution"] <= base["pollution"]
        and plus["strict"] >= base["strict"]
        and changes["net"] > 0
        and len(changes["gains"]) > len(changes["losses"])
        and plus["mean_delta"] >= base["mean_delta"]
    )


def better(left: dict, right: dict, left_changes: dict, right_changes: dict) -> bool:
    keys = (
        ("full_exact", True),
        ("replay", True),
        ("nonzero_linf", True),
        ("invalid", False),
        ("strict", True),
    )
    for key, higher in keys:
        if left[key] != right[key]:
            return left[key] > right[key] if higher else left[key] < right[key]
    if left_changes["net"] != right_changes["net"]:
        return left_changes["net"] > right_changes["net"]
    if left["mean_delta"] != right["mean_delta"]:
        return left["mean_delta"] > right["mean_delta"]
    if left["pollution"] != right["pollution"]:
        return left["pollution"] < right["pollution"]
    return False


def decide(summaries: dict[str, dict]) -> dict:
    base = summaries["B_NATIVE"]
    changes = {name: paired_changes(name) if name != "B_NATIVE" else {"gains": [], "losses": [], "net": 0} for name in METHODS}
    if base.get("status") != "RUN" or not base["hard_pass"]:
        return {
            "decision": "RETAIN_EXISTING_B_NO_NEW_METHOD_ADMITTED",
            "selected_method": None,
            "reason": "B_NATIVE did not clear the hard gates, so no new method is admitted.",
            "transitions_vs_b_native": changes,
        }
    pool = ["B_NATIVE"]
    for name in ("NATIVE_FDP", "B_PLUS"):
        row = summaries[name]
        if row.get("status") == "RUN" and row["hard_pass"] and row["pollution"] <= base["pollution"]:
            if name != "B_PLUS" or bplus_eligible(summaries, changes["B_PLUS"]):
                pool.append(name)
    chosen = "B_NATIVE"
    for name in pool:
        if name == "B_NATIVE":
            continue
        if better(summaries[name], summaries[chosen], changes[name], changes[chosen]):
            chosen = name
    if chosen == "B_NATIVE" and pool == ["B_NATIVE"] and (
        summaries["NATIVE_FDP"].get("status") != "RUN" or not summaries["NATIVE_FDP"]["hard_pass"]
    ) and (
        summaries["B_PLUS"].get("status") != "RUN" or not summaries["B_PLUS"]["hard_pass"]
    ):
        decision = "RETAIN_EXISTING_B_NO_NEW_METHOD_ADMITTED"
        reason = "No trained method cleared every hard gate."
    elif chosen == "B_NATIVE":
        decision = "RETAIN_B_NATIVE"
        reason = "B_NATIVE remains the selection. Alternatives did not outrank it on the frozen priority list."
    elif chosen == "NATIVE_FDP":
        decision = "SELECT_NATIVE_FDP"
        reason = "NATIVE_FDP cleared the hard gates and ranks above B_NATIVE and B_PLUS on strict count."
    else:
        decision = "SELECT_B_PLUS"
        reason = "B_PLUS cleared the hard gates, gained more strict cases than it lost, and its train-band native Delta is at least B_NATIVE."
    return {
        "decision": decision,
        "selected_method": None if decision == "RETAIN_EXISTING_B_NO_NEW_METHOD_ADMITTED" else chosen,
        "reason": reason,
        "eligible_pool": pool,
        "bplus_eligible": bplus_eligible(summaries, changes["B_PLUS"]),
        "transitions_vs_b_native": changes,
    }


def main() -> None:
    loaded = {name: load_method(name) for name in METHODS}
    summaries = {name: summarize(name, loaded[name]) for name in METHODS}
    if any(row.get("status") != "RUN" for row in summaries.values()):
        raise SystemExit("finalize requires all three methods scored")
    paired = []
    for sid in DEV:
        row = {"sample_id": sid}
        for name in METHODS:
            item = loaded[name][sid]
            row[f"{name}_strict"] = int(item["strict_train"])
            row[f"{name}_full_exact"] = int(item["full_exact"])
            row[f"{name}_replay"] = int(item["replay"])
            row[f"{name}_delta"] = train_delta(item)
            row[f"{name}_pollution"] = int(item["pollution"])
        paired.append(row)
    write_csv(HERE / "per_sample_paired.csv", paired)
    write_csv(HERE / "method_summary.csv", [summaries[name] for name in METHODS])
    margin_rows = []
    for name in METHODS:
        for ratio in RATIOS:
            deltas = [loaded[name][sid]["by_ratio"][ratio]["delta"] for sid in DEV]
            stricts = [loaded[name][sid]["by_ratio"][ratio]["strict_at_ratio"] for sid in DEV]
            margin_rows.append({
                "method": name,
                "ratio": ratio,
                "mean_native_delta": mean(deltas),
                "strict": sum(stricts),
                "used_for_selection": int(ratio in {"0.25", "0.35"}),
            })
    write_csv(HERE / "margin_by_budget.csv", margin_rows)
    for field, filename in (
        ("strict_train", "strict_transition_table.csv"),
        ("full_exact", "full_exact_transition_table.csv"),
        ("replay", "replay_transition_table.csv"),
    ):
        rows = []
        for sid in DEV:
            rows.append({
                "sample_id": sid,
                "B_NATIVE": int(loaded["B_NATIVE"][sid][field]),
                "NATIVE_FDP": int(loaded["NATIVE_FDP"][sid][field]),
                "B_PLUS": int(loaded["B_PLUS"][sid][field]),
            })
        write_csv(HERE / filename, rows)
    pollution_rows = []
    for sid in DEV:
        pollution_rows.append({
            "sample_id": sid,
            **{name: int(loaded[name][sid]["pollution"]) for name in METHODS},
            **{f"{name}_full": loaded[name][sid]["full_text"] for name in METHODS},
            "clean": loaded["B_NATIVE"][sid]["clean_text"],
        })
    write_csv(HERE / "output_pollution_audit.csv", pollution_rows)
    figs = HERE / "figs"
    figs.mkdir(exist_ok=True)
    bars(figs / "strict_counts.png", list(METHODS), [summaries[name]["strict"] for name in METHODS], "Canonical strict on the 12-question development set")
    bars(
        figs / "train_delta_milli.png",
        list(METHODS),
        [int(round(1000 * summaries[name]["mean_delta"])) for name in METHODS],
        "Mean train-band native Delta x1000",
    )
    decision = decide(summaries)
    decision.update({
        "scope": "12-question development selection only; not a generalization result",
        "canonical_strict": "valid AND nonzero AND uint8 Linf<=8/255 AND Full text exact AND FastV ANLS<0.5 at both r=0.25 and r=0.35 AND disk replay",
        "historical_b_52_of_60": "unchanged; this run does not rewrite it",
        "r015_31_vs_45": "definition drift; the two historical predicates are not interchangeable",
        "unseen_cohort_started": False,
        "summaries": summaries,
    })
    (HERE / "final_decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    print(json.dumps({"decision": decision["decision"], "summaries": summaries}, indent=2))


if __name__ == "__main__":
    main()
