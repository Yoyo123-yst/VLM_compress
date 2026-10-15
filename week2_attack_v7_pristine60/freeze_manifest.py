"""Draw the frozen 60 after the clean screen. One draw, seed 2026100902, no M quota."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project/week2_attack_v7_pristine60")
FINAL_SEED = 2026100902
FINAL_N = 60


def main():
    screen = json.loads((ROOT / "pool" / "screen_manifest.json").read_text())
    recorded = (ROOT / "pool" / "screen_manifest.sha256").read_text().strip()
    if screen["manifest_sha256"] != recorded:
        raise SystemExit("screen manifest hash mismatch")
    rows = []
    seen = set()
    for path in sorted((ROOT / "clean_screen").glob("shard_s*.csv")):
        for row in csv.DictReader(path.open()):
            if row["sample_id"] in seen:
                raise SystemExit(f"duplicate clean-screen row {row['sample_id']}")
            seen.add(row["sample_id"])
            rows.append(row)
    if len(rows) != 400:
        raise SystemExit(f"clean screen has {len(rows)} rows, expected 400")
    by_id = {row["sample_id"]: row for row in screen["samples"]}
    if set(by_id) != seen:
        raise SystemExit("clean screen ids differ from the screen manifest")
    eligible = [row for row in rows if row["eligible"] == "1" and row["page_sha_ok"] == "1" and row["k_ok"] == "1" and row["hook_ok"] == "1"]
    summary = {
        "screen_n": 400,
        "eligible": len(eligible),
        "final_seed": FINAL_SEED,
    }
    if len(eligible) < FINAL_N:
        summary["decision"] = "BLOCKED_INSUFFICIENT_PRISTINE60_POOL"
        (ROOT / "pool" / "clean_screen_decision.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary))
        return
    rng = random.Random(FINAL_SEED)
    chosen_ids = rng.sample([row["sample_id"] for row in eligible], FINAL_N)
    clean = {row["sample_id"]: row for row in rows}
    samples = []
    for sid in sorted(chosen_ids, key=int):
        spec = dict(by_id[sid])
        screen_row = clean[sid]
        spec.update({
            "clean_full_pred": screen_row["pred_full"],
            "clean_fastv_pred": screen_row["pred_fastv"],
            "clean_full_anls": float(screen_row["anls_full"]),
            "clean_fastv_anls": float(screen_row["anls_fastv"]),
            "clean_keep_hash": screen_row["keep_hash"],
            "clean_k": int(screen_row["k"]),
        })
        samples.append(spec)
    payload = {
        "final_seed": FINAL_SEED,
        "n": FINAL_N,
        "screen_seed": screen["screen_seed"],
        "screen_manifest_sha256": recorded,
        "historical_union_sha256": screen["historical_union_sha256"],
        "sampling": "random.Random(2026100902).sample of clean-eligible screen ids, stored in numeric order, no M quota",
        "draw_order": chosen_ids,
        "fastv_r": 0.25,
        "lambda_soft": 2.049824826423478,
        "beta": None,
        "epsilon": 8 / 255,
        "steps": 200,
        "checkpoints": [0, 10, 25, 50, 75, 100, 150, 200],
        "model_path": screen["model_path"],
        "samples": samples,
    }
    raw = json.dumps(payload, sort_keys=True).encode()
    payload["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
    (ROOT / "frozen_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    (ROOT / "frozen_manifest.sha256").write_text(payload["manifest_sha256"] + "\n")
    with (ROOT / "frozen_manifest.csv").open("w", newline="") as handle:
        fields = ["sample_id", "M", "evidence_idx", "aux_idx", "expected_k", "clean_full_pred", "clean_fastv_pred", "clean_keep_hash"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in samples:
            writer.writerow({key: row[key] for key in fields})
    summary["decision"] = "PRISTINE60_MANIFEST_FROZEN"
    summary["manifest_sha256"] = payload["manifest_sha256"]
    summary["ids"] = [row["sample_id"] for row in samples]
    (ROOT / "pool" / "clean_screen_decision.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"decision": summary["decision"], "eligible": len(eligible), "sha": payload["manifest_sha256"]}))


if __name__ == "__main__":
    main()
