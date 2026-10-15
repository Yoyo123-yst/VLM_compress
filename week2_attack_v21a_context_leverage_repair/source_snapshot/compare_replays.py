"""Compare the five zero-perturbation processes and write the phase A decision."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
sys.path[:0] = [str(ROOT)]
HERE = ROOT / "week2_attack_v21a_context_leverage_repair"

from week2_attack_v21a_context_leverage_repair.decisions import classify_replays  # noqa: E402


def main() -> None:
    runs = [json.loads((HERE / "replay_runs" / f"run_{index}.json").read_text()) for index in range(1, 6)]
    decision = classify_replays(runs)
    rows = []
    for run in runs:
        for name, path in run["paths"].items():
            rows.append({
                "run": run["run"],
                "path": name,
                "file_sha256": path["file_sha256"],
                "rgb_sha256": path["rgb_sha256"],
                "uint8_linf": path["uint8_linf"],
                "n_diff_pixels": path["n_diff_pixels"],
                "processor_exact_hash": path["processor_exact_hash"],
                "processor_max_abs_vs_original": path["processor_max_abs_vs_original"],
                "k": path["k"],
                "keep_hash": path["keep_hash"],
                "jaccard_vs_original": path["jaccard_vs_original"],
                "natural_logits_hash": path["natural_logits_hash"],
                "frozen_logits_hash": path["frozen_logits_hash"],
                "margin_hash": path["margin_hash"],
                "generated_text": path["generated_text"],
                "v21_splice_keep_hash": path["v21_splice_keep_hash"],
            })
    payload = {"runs": rows, "decision": decision}
    (HERE / "replay_comparison.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    with (HERE / "replay_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (HERE / "phase_a_decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    original = runs[0]["paths"]["ORIGINAL"]
    projected = runs[0]["paths"]["ZERO_PROJECTED"]
    lines = [
        "# 阶段 A：零扰动 replay",
        "",
        f"状态：`{decision['status']}`",
        "",
        f"ORIGINAL keep-set `{original['keep_hash']}`，K={original['k']}。",
        f"ZERO_PROJECTED 的 uint8 L∞ 是 {projected['uint8_linf']}，processor 最大绝对差是 {projected['processor_max_abs_vs_original']}。",
        f"v21 使用的 splice keep-set 是 `{original['v21_splice_keep_hash']}`。",
        "",
        "五次进程的 ORIGINAL keep-set " + ("相同。" if decision["keep_hashes_identical"] else "不同。"),
        "阶段 B " + ("可以开始。" if decision["enter_phase_b"] else "不开始。"),
        "",
    ]
    (HERE / "phase_a_report.md").write_text("\n".join(lines))
    print(json.dumps(decision), flush=True)
    if not decision["enter_phase_b"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
