"""Apply the pre-registered witness rules. Thresholds are not fitted to the run."""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project/week2_attack_v10_attack_compensation_probe")
V7D = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project/week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation")
IDS = ("1025", "2008", "47450", "60700", "7006", "39204")
RATIOS = ("0.25", "0.35")


def condition(sample: dict, ratio: str) -> dict:
    ends = sample.get("endpoints") or {}
    needed = ("CLEAN", "ATTACK_ONLY", "COMPENSATION_ONLY", "ATTACK_PLUS_COMPENSATION", "SIGN_FLIP_CONTROL", "ROLL_CONTROL")
    if any(name not in ends for name in needed):
        return {"measured": False}
    clean = ends["CLEAN"]["full_text"]
    attack = ends["ATTACK_ONLY"]
    comp = ends["COMPENSATION_ONLY"]
    both = ends["ATTACK_PLUS_COMPENSATION"]
    sign = ends["SIGN_FLIP_CONTROL"]
    roll = ends["ROLL_CONTROL"]
    attack_fast = attack["ratios"][ratio]
    both_fast = both["ratios"][ratio]
    comp_fast = comp["ratios"][ratio]
    sign_fast = sign["ratios"][ratio]
    roll_fast = roll["ratios"][ratio]

    def equivalent(block, fast) -> bool:
        return block["full_text"] == clean and block["full_replay"] and fast["endpoint_failure"] and fast["replay"] and fast["hook_match"] == 1

    flags = {
        "attack_full_break": attack["full_text"] != clean and attack["full_replay"],
        "attack_fastv_failure": bool(attack_fast["endpoint_failure"] and attack_fast["replay"] and attack_fast["hook_match"] == 1),
        "compensation_recovers_full": both["full_text"] == clean and both["full_replay"],
        "combined_fastv_failure": bool(both_fast["endpoint_failure"] and both_fast["replay"] and both_fast["hook_match"] == 1),
        "compensation_alone_benign": comp["full_text"] == clean and comp["full_replay"] and comp_fast["valid"] and comp_fast["anls"] >= 0.5 and comp_fast["replay"],
        "control_not_equivalent": not (equivalent(sign, sign_fast) and equivalent(roll, roll_fast)),
    }
    flags["strong"] = all(flags.values())
    flags["integrity"] = all(
        block["full_replay"] and all(cell["replay"] and cell["hook_match"] == 1 for cell in block["ratios"].values())
        for block in (attack, comp, both, sign, roll, ends["CLEAN"])
    )
    return {"measured": True, **flags}


def main() -> None:
    rows = []
    missing = []
    for sample_id in IDS:
        path = HERE / "per_sample" / f"{sample_id}.json"
        if not path.exists():
            missing.append(sample_id)
            continue
        sample = json.loads(path.read_text())
        historical = json.loads((V7D / "scored" / "NATIVE_FDP" / f"{sample_id}.json").read_text())
        by_ratio = {ratio: condition(sample, ratio) for ratio in RATIOS}
        strong = all(cell.get("strong") for cell in by_ratio.values())
        partial = (not strong) and any(cell.get("strong") for cell in by_ratio.values())
        integrity = all(cell.get("integrity", False) for cell in by_ratio.values() if cell.get("measured"))
        if any(cell.get("measured") and not cell.get("integrity", False) for cell in by_ratio.values()):
            integrity = False
        rows.append({
            "sample_id": sample_id,
            "status": sample.get("status"),
            "historical_native_fdp_strict": historical.get("strict_train"),
            "accepted": sample.get("comp_accepted"),
            "rejected": sample.get("comp_rejected"),
            "counters": sample.get("counters"),
            "elapsed_sec": sample.get("elapsed_sec"),
            "peak_memory_mib": sample.get("peak_memory_mib"),
            "ratios": by_ratio,
            "strong": bool(strong),
            "partial": bool(partial),
            "integrity": bool(integrity) if sample.get("endpoints") else sample.get("status") != "OK",
        })
    parents = json.loads((HERE / "PARENT_HASHES.json").read_text())
    parent_mismatch = []
    root = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
    import hashlib
    for rel, digest in parents.items():
        path = root / rel
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != digest:
            parent_mismatch.append(rel)
    baseline = json.loads((HERE / "baseline_replay.json").read_text())
    strong_n = sum(row["strong"] for row in rows)
    witness_n = sum(row["strong"] or row["partial"] for row in rows)
    witness_ids = [row["sample_id"] for row in rows if row["strong"] or row["partial"]]
    repaired = []
    comp_attack = []
    for row in rows:
        if not all(cell.get("measured") for cell in row["ratios"].values()):
            continue
        if all(cell["compensation_recovers_full"] and not cell["combined_fastv_failure"] for cell in row["ratios"].values()):
            repaired.append(row["sample_id"])
        if any(not cell["compensation_alone_benign"] for cell in row["ratios"].values()):
            comp_attack.append(row["sample_id"])
    measured = [row for row in rows if all(cell.get("measured") for cell in row["ratios"].values())]
    integrity_bad = parent_mismatch or baseline.get("status") != "PASS" or missing or any(row.get("endpoints_integrity") is False for row in rows)
    # integrity flag on rows that were measured
    if any(all(cell.get("measured") for cell in row["ratios"].values()) and not row["integrity"] for row in rows):
        integrity_bad = True
    usually_repairs = bool(measured) and len(repaired) * 2 >= len(measured)
    usually_comp_attack = bool(measured) and len(comp_attack) * 2 >= len(measured)
    only_1025 = witness_ids == ["1025"]
    if integrity_bad:
        decision = "BASELINE_OR_REPLAY_DIVERGED"
    elif only_1025 or usually_repairs or usually_comp_attack or witness_n < 2:
        decision = "NO_GO_ATTACK_COMPENSATION_STRUCTURE"
    elif strong_n >= 2 and witness_n >= 3:
        decision = "GO_ATTACK_COMPENSATION_STRUCTURE"
    elif strong_n < 2 and witness_n >= 2:
        decision = "PARTIAL_ATTACK_COMPENSATION_STRUCTURE"
    else:
        decision = "NO_GO_ATTACK_COMPENSATION_STRUCTURE"
    payload = {
        "decision": decision,
        "scope": "development constructability probe",
        "denominator": 6,
        "strong": f"{strong_n}/6",
        "strong_or_partial": f"{witness_n}/6",
        "witness_ids": witness_ids,
        "compensation_repairs_fastv_ids": repaired,
        "compensation_only_not_benign_ids": comp_attack,
        "missing_samples": missing,
        "parent_mismatch": parent_mismatch,
        "rows": rows,
        "native_fdp_method_comparison": "NOT_RUN",
        "dynamic_role_assignment": "NOT_RUN",
        "fixed_weighted_dual_loss": "NOT_RUN",
        "alternating_optimization": "NOT_RUN",
        "generic_gradient_projection": "NOT_RUN",
    }
    (HERE / "final_decision.json").write_text(json.dumps(payload, indent=2) + "\n")
    write_report(payload)
    print(decision)


def write_report(payload: dict) -> None:
    lines = [
        f"# Attack-compensation probe",
        "",
        f"Decision: `{payload['decision']}`.",
        "",
        "This is a 6-sample development constructability probe. It is not a method result, not an unseen evaluation, and not evidence that the construction beats Native-FDP.",
        "",
        "## Baseline",
        "",
        "Clean Full text, clean FastV text/K/keep hash at r=0.25 and r=0.35, and the historical Native-FDP PNG replay matched the v7d records. See `baseline_replay.json`.",
        "",
        "## Witnesses",
        "",
        f"Denominator 6. Strong {payload['strong']}. Strong or partial {payload['strong_or_partial']}. Witness ids: {payload['witness_ids'] or 'none'}.",
        "Sample 1025 is marked noise-sensitive at r=0.25 from the earlier 28/32 matched-noise diagnostic and cannot decide the result alone.",
        "",
        "| sample | Native-FDP strict | status | witness | accepted | rejected |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload["rows"]:
        kind = "strong" if row["strong"] else "partial" if row["partial"] else "none"
        lines.append(
            f"| {row['sample_id']} | {row['historical_native_fdp_strict']} | {row['status']} | {kind} | {row['accepted']} | {row['rejected']} |"
        )
    lines.extend(["", "## Ratio conditions", ""])
    for row in payload["rows"]:
        lines.append(f"### {row['sample_id']}")
        for ratio, cell in row["ratios"].items():
            if not cell.get("measured"):
                lines.append(f"- r={ratio}: not measured.")
                continue
            lines.append(
                "- r={ratio}: full_break={attack_full_break}, fastv_fail={attack_fastv_failure}, recovers_full={compensation_recovers_full}, combined_fastv_fail={combined_fastv_failure}, comp_benign={compensation_alone_benign}, control_not_equivalent={control_not_equivalent}.".format(ratio=ratio, **cell)
            )
        lines.append("")
    lines.extend([
        "## What was not run",
        "",
        "Native-FDP was not retrained. Fixed dual loss, alternating optimization, generic gradient projection, and dynamic attack-compensation role assignment are NOT_RUN.",
        "",
        "## Cost",
        "",
    ])
    for row in payload["rows"]:
        lines.append(f"- {row['sample_id']}: counters={row['counters']}, elapsed_sec={row['elapsed_sec']}, peak_mib={row['peak_memory_mib']}")
    (HERE / "final_report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
