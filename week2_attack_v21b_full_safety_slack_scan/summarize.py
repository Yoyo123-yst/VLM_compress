"""Write the slack tables after the independent replay matches."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
sys.path[:0] = [str(ROOT)]
HERE = ROOT / "week2_attack_v21b_full_safety_slack_scan"

from week2_attack_v21b_full_safety_slack_scan.decisions import direction_summary, final_status  # noqa: E402


def load_points() -> dict[str, list[dict]]:
    replay = json.loads((HERE / "independent_eval" / "comparison.json").read_text())
    if not replay["match"]:
        raise SystemExit("independent replay diverged")
    matched = {(row["direction"], row["requested_uint8_linf"]) for row in replay["rows"] if row["match"]}
    grouped = {"CORE": [], "ALL": []}
    for path in sorted((HERE / "scan").glob("*/*.json")):
        row = json.loads(path.read_text())
        exact = bool(row["full_text_exact"] and row["full_token_exact"] and row["full_replay"] and row["full_output_valid"])
        row["independent_match"] = (row["direction"], row["requested_uint8_linf"]) in matched
        row["full_legal"] = bool(row["requested_uint8_linf"] and not row["quantization_radius_mismatch"] and exact and row["independent_match"])
        grouped[row["direction"]].append(row)
    return grouped


def write_tables(grouped: dict[str, list[dict]]) -> None:
    scan_rows = []
    full_rows = []
    core_rows = []
    for direction, points in grouped.items():
        for row in points:
            scan_rows.append({
                "direction": direction,
                "requested_uint8_linf": row["requested_uint8_linf"],
                "actual_uint8_linf": row["actual_uint8_linf"],
                "radius_label": row["radius_label"],
                "full_text_exact": row["full_text_exact"],
                "full_token_exact": row["full_token_exact"],
                "full_replay": row["full_replay"],
                "full_output_valid": row["full_output_valid"],
                "full_legal": row["full_legal"],
                "full_hard_mean": row["full_hard_mean"],
                "full_hard_min": row["full_hard_min"],
                "weakest_position": row["weakest_full_token"]["position"],
                "weakest_piece": row["weakest_full_token"]["piece"],
                "core_hard_mean": row["core_hard_mean"],
                "core_hard_min": row["core_hard_min"],
                "fast_text": row["fast_text"],
                "semantic_status": row["semantic_status"],
                "k": row["k"],
                "keep_hash": row["keep_hash"],
                "keep_jaccard": row["keep_jaccard"],
                "png_sha256": row["png_sha256"],
                "forward": row["forward"],
                "backward": row["backward"],
                "generate": row["generate"],
                "wall_seconds": row["wall_seconds"],
            })
            for token in row["full_margins"]:
                full_rows.append({"direction": direction, "radius": row["requested_uint8_linf"], **token})
            for token in row["core_margins"]:
                core_rows.append({"direction": direction, "radius": row["requested_uint8_linf"], **token})
    for name, rows in (("scan_results.csv", scan_rows), ("per_token_full_margins.csv", full_rows), ("per_token_core_margins.csv", core_rows)):
        with (HERE / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def point_at(points: list[dict], radius: int) -> dict | None:
    return next((row for row in points if int(row["requested_uint8_linf"]) == radius), None)


def main() -> None:
    grouped = load_points()
    summaries = {name: direction_summary(points) for name, points in grouped.items()}
    for name, points in grouped.items():
        prefix = summaries[name]["prefix_safe_radius"]
        chosen = point_at(points, prefix) if prefix else None
        summaries[name]["minimum_full_margin_at_prefix_safe_radius"] = None if chosen is None else chosen["full_hard_min"]
        summaries[name]["FastV_content_status_at_prefix_safe_radius"] = None if chosen is None else {
            "text": chosen["fast_text"],
            "semantic_status": chosen["semantic_status"],
            "core_hard_mean": chosen["core_hard_mean"],
            "core_name_intact": chosen["core_name_intact"],
        }
        legal_points = [row for row in points if row["full_legal"]]
        summaries[name]["exact_with_small_full_margin"] = [
            {"radius": row["requested_uint8_linf"], "full_hard_min": row["full_hard_min"], "piece": row["weakest_full_token"]["piece"]}
            for row in legal_points if row["full_hard_min"] < 1.0
        ]
    decision = final_status(summaries["CORE"], summaries["ALL"])
    payload = {"directions": summaries, **decision}
    (HERE / "safety_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    (HERE / "final_decision.json").write_text(json.dumps({key: payload[key] for key in ("status", "next_step", "comparison_started")}, indent=2) + "\n")
    write_tables(grouped)
    lines = ["# Full 安全余量扫描", "", f"状态：`{decision['status']}`", "", f"下一步：`{decision['next_step']}`", ""]
    for name in ("CORE", "ALL"):
        summary = summaries[name]
        lines.append(f"## {name}")
        lines.append("")
        lines.append(f"prefix-safe radius 是 {summary['prefix_safe_radius']}。最大合法测试半径是 {summary['largest_tested_legal_radius']}。非单调是 {summary['nonmonotonic_full_behavior']}。")
        lines.append("")
        lines.append("| 半径 | 实际 L∞ | Full exact | Full hard min | 最弱 token | FastV CORE mean | FastV 文本 |")
        lines.append("|---:|---:|---|---:|---|---:|---|")
        for row in grouped[name]:
            exact = row["full_text_exact"] and row["full_token_exact"] and row["full_replay"]
            lines.append(
                f"| {row['requested_uint8_linf']} | {row['actual_uint8_linf']} | {exact} | {row['full_hard_min']:.4f} | `{row['weakest_full_token']['piece']}` | {row['core_hard_mean']:.4f} | {row['fast_text']} |"
            )
        lines.append("")
    lines.append("语义状态只作为诊断记录，没有用来选择半径。本轮没有启动 Native-FDP、safety-hinge、OPGD 或 ALMA 比较。")
    lines.append("")
    (HERE / "final_report.md").write_text("\n".join(lines))
    print(json.dumps(decision), flush=True)


if __name__ == "__main__":
    main()
