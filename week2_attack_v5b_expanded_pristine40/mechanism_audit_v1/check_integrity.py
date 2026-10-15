"""Stage I: hash and count v5b raw inputs. Does not rewrite them."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
V5B = ROOT / "week2_attack_v5b_expanded_pristine40"
OUT = V5B / "mechanism_audit_v1"
sys.path.insert(0, str(ROOT))
from week2_attack_v4_answer_aligned.loss import official_keep_count  # noqa: E402

EPS = 8 / 255


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = json.loads((V5B / "frozen_manifest.json").read_text())
    fixed = list(csv.DictReader((V5B / "results" / "fixed_step200.csv").open()))
    ids = [row["sample_id"] for row in manifest["samples"]]
    errors = []
    if len(ids) != 40 or len(set(ids)) != 40:
        errors.append("manifest n")
    if [row["sample_id"] for row in fixed] != ids and set(row["sample_id"] for row in fixed) != set(ids):
        errors.append("fixed csv ids")
    strict = sum(row["strict"] in ("True", "true", "1") for row in fixed)
    text_same = sum(row["full_text_same"] in ("True", "true", "1") for row in fixed)
    if strict != 33:
        errors.append(f"strict {strict}")
    if text_same != 39:
        errors.append(f"text {text_same}")
    lines = []
    for spec in manifest["samples"]:
        sid = spec["sample_id"]
        art = V5B / "attacks" / "artifacts" / sid
        final = art / "final.png"
        step = art / "checkpoints" / "step_200.png"
        needed = [art / "clean.png", final, step, art / "final_provenance.json", art / "min_margin.png"]
        for path in needed:
            if not path.exists():
                errors.append(f"missing {path.relative_to(V5B)}")
        if final.exists() and step.exists() and sha(final) != sha(step):
            errors.append(f"final!=step200 {sid}")
        expected = official_keep_count(int(spec["M"]) * 3699, 0.25)
        if expected != spec["expected_k"]:
            errors.append(f"K {sid}")
        row = next(item for item in fixed if item["sample_id"] == sid)
        if float(row["linf"]) > EPS + 1e-6:
            errors.append(f"linf {sid}")
        for path in needed:
            if path.exists() and path.suffix in {".png", ".json"}:
                lines.append(f"{sha(path)}  {path.relative_to(V5B)}")
    raw_files = [
        V5B / "FROZEN.json",
        V5B / "EXECUTION_LOCK.json",
        V5B / "frozen_manifest.json",
        V5B / "final_report.md",
        V5B / "final_decision.json",
        V5B / "results" / "fixed_step200.csv",
        V5B / "results" / "all_checkpoints.csv",
        V5B / "analysis" / "replay_summary.json",
    ]
    for path in raw_files:
        if not path.exists():
            errors.append(f"missing {path.name}")
        else:
            lines.append(f"{sha(path)}  {path.relative_to(V5B)}")
    for path in sorted((V5B / "attacks").glob("shard_s*/checkpoints.jsonl")):
        lines.append(f"{sha(path)}  {path.relative_to(V5B)}")
    decision = "PASS" if not errors else "BLOCKED_V5B_RAW_INTEGRITY"
    report = {
        "decision": decision,
        "n": len(ids),
        "strict_fixed_step200": strict,
        "full_text_unchanged": text_same,
        "errors": errors,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "input_integrity.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "raw_before.sha256").write_text("\n".join(lines) + "\n")
    (OUT / "input_integrity.md").write_text(
        "\n".join([
            "# v5b input integrity",
            "",
            f"- Decision: `{decision}`",
            f"- IDs: {len(ids)}",
            f"- Fixed-step strict: {strict}/40",
            f"- Full text unchanged: {text_same}/40",
            f"- Errors: {errors or 'none'}",
        ]) + "\n"
    )
    print(json.dumps(report))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
