"""Read-only check of frozen v5b inputs. Does not rewrite them."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
V5B = ROOT / "week2_attack_v5b_expanded_pristine40"
sys.path.insert(0, str(ROOT))
from week2_attack_v4_answer_aligned.loss import official_keep_count  # noqa: E402

TOKENS = 3699


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def payload_sha(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def main():
    frozen_manifest = json.loads((V5B / "frozen_manifest.json").read_text())
    screen = json.loads((V5B / "pool" / "screen_manifest_200.json").read_text())
    union = json.loads((V5B / "exclusions" / "historical_union.json").read_text())
    excluded = {row["sample_id"] for row in union["samples"]}
    clean_rows = list(csv.DictReader((V5B / "pool" / "clean_screen_all_200.csv").open()))
    eligible = {row["sample_id"] for row in clean_rows if row["eligible"] == "1"}
    errors = []
    frozen_payload = payload_sha(frozen_manifest)
    screen_payload = payload_sha(screen)
    if frozen_payload != frozen_manifest["manifest_sha256"]:
        errors.append("frozen payload hash mismatch")
    if frozen_payload != (V5B / "frozen_manifest.sha256").read_text().strip():
        errors.append("frozen_manifest.sha256 file does not store payload hash")
    if screen_payload != screen["manifest_sha256"]:
        errors.append("screen payload hash mismatch")
    if screen_payload != (V5B / "pool" / "screen_manifest_200.sha256").read_text().strip():
        errors.append("screen sha file mismatch")
    union_file = file_sha(V5B / "exclusions" / "historical_union.json")
    if union_file != (V5B / "exclusions" / "historical_union.sha256").read_text().strip():
        errors.append("historical union file hash mismatch")
    ids = [row["sample_id"] for row in frozen_manifest["samples"]]
    if len(ids) != 40 or len(set(ids)) != 40:
        errors.append(f"final n={len(ids)}")
    overlap = sorted(set(ids) & excluded, key=int)
    if overlap:
        errors.append(f"historical overlap {overlap}")
    missing_eligible = [sid for sid in ids if sid not in eligible]
    if missing_eligible:
        errors.append(f"not clean-eligible {missing_eligible}")
    for row in frozen_manifest["samples"]:
        if row["M"] not in (2, 3, 4):
            errors.append(f"{row['sample_id']} bad M")
        if row["evidence_idx"] == row["aux_idx"]:
            errors.append(f"{row['sample_id']} aux is evidence")
        if not (0 <= row["aux_idx"] < row["M"]):
            errors.append(f"{row['sample_id']} aux oob")
        expected = official_keep_count(row["M"] * TOKENS, 0.25)
        if expected != row["expected_k"]:
            errors.append(f"{row['sample_id']} K {row['expected_k']} != {expected}")
        if len(row["page_sha256"]) != row["M"]:
            errors.append(f"{row['sample_id']} page hash count")
        if row["clean_full_anls"] < 0.5 or row["clean_fastv_anls"] < 0.5:
            errors.append(f"{row['sample_id']} clean anls")
        if not str(row["clean_full_pred"]).strip() or not str(row["clean_fastv_pred"]).strip():
            errors.append(f"{row['sample_id']} empty clean text")
    report = {
        "frozen_payload_sha256": frozen_payload,
        "frozen_file_sha256": file_sha(V5B / "frozen_manifest.json"),
        "screen_payload_sha256": screen_payload,
        "screen_file_sha256": file_sha(V5B / "pool" / "screen_manifest_200.json"),
        "frozen_json_file_sha256": file_sha(V5B / "FROZEN.json"),
        "historical_union_file_sha256": union_file,
        "n_final": len(ids),
        "historical_overlap": len(overlap),
        "errors": errors,
        "decision": "PASS" if not errors else "BLOCKED_FROZEN_INPUT_INTEGRITY_FAIL",
    }
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
