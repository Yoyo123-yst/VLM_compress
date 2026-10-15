"""Decision gate for the frozen 6-question run. Does not retune anything."""

from __future__ import annotations

import csv
import json
from pathlib import Path

V4 = Path(__file__).resolve().parent
REACH = ["50743", "4912", "56776"]
HELD = ["49242", "33923"]


def load_csv(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def main():
    health = json.loads((V4 / "results" / "health.json").read_text())
    rows = load_csv(V4 / "results" / "smoke.csv")
    by_id = {row["sample_id"]: row for row in rows}
    checkpoints = []
    curve = V4 / "results" / "checkpoints.jsonl"
    if curve.exists():
        checkpoints = [json.loads(line) for line in curve.read_text().splitlines() if line.strip()]
    missing = [sample_id for sample_id in REACH + HELD + ["5169"] if sample_id not in by_id]
    if health["status"] != "PASS":
        decision = "BLOCKED_ON_FORWARD_ALIGNMENT"
    elif missing:
        decision = "INCOMPLETE"
    else:
        new_ok = [
            sample_id for sample_id in REACH
            if by_id[sample_id]["label"] == "SUCCESS_COMP_ONLY" and by_id[sample_id]["replay_ok"] == "1"
        ]
        held_ok = [
            sample_id for sample_id in HELD
            if by_id[sample_id]["label"] == "SUCCESS_COMP_ONLY" and by_id[sample_id]["replay_ok"] == "1"
        ]
        invalid = sum(row["label"] == "INVALID_OUTPUT" for row in rows)
        full_ok = all(float(row["anls_full"]) >= 0.5 for row in rows)
        eps_ok = all(row["eps_ok"] == "1" for row in rows)
        replay_all = all(row["replay_ok"] == "1" for row in rows)
        if len(new_ok) >= 2 and len(held_ok) >= 1 and invalid == 0 and full_ok and eps_ok and replay_all and len(rows) == 6:
            decision = "GO_TO_20"
        elif len(new_ok) == 1 and invalid == 0:
            decision = "WEAK_SIGNAL"
        elif len(new_ok) == 0:
            decision = "NO_GO_HARD_ANSWER"
        else:
            decision = "NO_GO_HARD_ANSWER"
    counts = {}
    for row in rows:
        counts[row["label"]] = counts.get(row["label"], 0) + 1
    lines = [
        "# v4 answer-aligned 6 题",
        "",
        f"健康检查：{health['status']}",
        f"决定：{decision}",
        "",
        "| sample | group | best step | label | replay | full ANLS | fastv ANLS | full answer | fastv answer | margin | aux share | evidence keep |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {sample_id} | {group} | {best_step} | {label} | {replay_ok} | {anls_full} | {anls_fastv} | {pred_full} | {pred_fastv} | {margin} | {aux_share} | {evidence_keep} |".format(**row)
        )
    lines += [
        "",
        f"标签计数：{json.dumps(counts, ensure_ascii=False)}",
        f"检查点数：{len(checkpoints)}",
        "",
        "这 6 题是开发诊断集，不能当成泛化结果。",
    ]
    text = "\n".join(lines) + "\n"
    (V4 / "results" / "summary.md").write_text(text)
    print(decision)


if __name__ == "__main__":
    main()
