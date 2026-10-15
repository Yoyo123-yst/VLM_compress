"""Merge clean screen and freeze 40 if at least 40 are eligible. Read-only on raw shards."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
V5B = ROOT / "week2_attack_v5b_expanded_pristine40"
FINAL_SEED = 2026100803
FINAL_N = 40
NOISE_SEED = 2026100804


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    screen = json.loads((V5B / "pool" / "screen_manifest_200.json").read_text())
    by_id = {row["sample_id"]: row for row in screen["samples"]}
    rows = []
    for shard in (0, 1):
        path = V5B / "clean_screen" / f"shard_s{shard}.csv"
        with path.open() as handle:
            rows.extend(list(csv.DictReader(handle)))
    if len(rows) != 200 or len({r["sample_id"] for r in rows}) != 200:
        raise SystemExit(f"clean screen incomplete: {len(rows)}")
    rows.sort(key=lambda r: int(r["sample_id"]))
    fields = list(rows[0].keys())
    with (V5B / "pool" / "clean_screen_all_200.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    eligible = [r for r in rows if r["eligible"] == "1"]
    summary = {
        "screened": len(rows),
        "eligible": len(eligible),
        "eligible_ids": [r["sample_id"] for r in eligible],
        "final_seed": FINAL_SEED,
    }
    (V5B / "pool" / "clean_screen_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if len(eligible) < FINAL_N:
        decision = {
            "decision": "BLOCKED_INSUFFICIENT_CLEAN_ELIGIBLE_POOL",
            "eligible": len(eligible),
            "screened": 200,
        }
        (V5B / "final_decision.json").write_text(json.dumps(decision, indent=2) + "\n")
        print(json.dumps(decision))
        return
    rng = random.Random(FINAL_SEED)
    chosen = rng.sample([r["sample_id"] for r in eligible], FINAL_N)
    chosen_sorted = sorted(chosen, key=int)
    samples = []
    for sid in chosen_sorted:
        spec = by_id[sid]
        clean = next(r for r in rows if r["sample_id"] == sid)
        samples.append({
            **{k: spec[k] for k in spec},
            "clean_full_pred": clean["pred_full"],
            "clean_fastv_pred": clean["pred_fastv"],
            "clean_full_anls": float(clean["anls_full"]),
            "clean_fastv_anls": float(clean["anls_fastv"]),
            "clean_keep_hash": clean["keep_hash"],
            "clean_k": int(clean["k"]),
        })
    manifest = {
        "final_sample_seed": FINAL_SEED,
        "noise_seed": NOISE_SEED,
        "n": FINAL_N,
        "eligible_pool": len(eligible),
        "screen_manifest_sha256": screen["manifest_sha256"],
        "historical_union_sha256": screen["historical_union_sha256"],
        "lambda_full": 0.25,
        "steps": 200,
        "eps": 8 / 255,
        "lr": 2 / 255,
        "optimizer": "Adam",
        "fastv_r": 0.25,
        "primary_selector": "fixed_step_200",
        "secondary_selector": "prospective_min_compressed_margin",
        "checkpoints": [0, 10, 25, 50, 75, 100, 150, 200],
        "samples": samples,
    }
    raw = json.dumps(manifest, sort_keys=True).encode()
    manifest["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
    (V5B / "frozen_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (V5B / "frozen_manifest.sha256").write_text(manifest["manifest_sha256"] + "\n")
    with (V5B / "frozen_manifest.csv").open("w", newline="") as handle:
        fields = ["sample_id", "M", "aux_idx", "evidence_idx", "expected_k", "clean_full_anls", "clean_fastv_anls"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in samples:
            writer.writerow({k: row[k] for k in fields})
    scale_path = ROOT / "week2_attack_v4_answer_aligned" / "results" / "lambda_scale.json"
    scales = json.loads(scale_path.read_text())
    frozen = {
        "stage": "v5b_expanded_pristine40",
        "verify_only_after_write": True,
        "manifest_sha256": manifest["manifest_sha256"],
        "screen_manifest_sha256": screen["manifest_sha256"],
        "historical_union_sha256": screen["historical_union_sha256"],
        "screen_seed": 2026100802,
        "final_sample_seed": FINAL_SEED,
        "noise_seed": NOISE_SEED,
        "lambda_full": 0.25,
        "steps": 200,
        "eps": 8 / 255,
        "optimizer": "Adam",
        "lr": 2 / 255,
        "scales_path": str(scale_path.relative_to(ROOT)),
        "scales_sha256": sha_file(scale_path),
        "scales": scales,
        "model": screen["model"],
        "code_hashes": screen["code_hashes"],
        "attack_code_sha256": {
            "run_smoke.py": sha_file(ROOT / "week2_attack_v4_answer_aligned" / "run_smoke.py"),
            "loss.py": sha_file(ROOT / "week2_attack_v4_answer_aligned" / "loss.py"),
            "forward.py": sha_file(ROOT / "week2_attack_v4_answer_aligned" / "forward.py"),
            "run_lambda.py": sha_file(ROOT / "week2_attack_v4_relaxed_full" / "run_lambda.py"),
        },
        "primary_selector": "fixed_step_200",
        "gpu_formal_attack_allowed": True,
    }
    (V5B / "FROZEN.json").write_text(json.dumps(frozen, indent=2) + "\n")
    print(json.dumps({
        "eligible": len(eligible),
        "selected": FINAL_N,
        "manifest_sha256": manifest["manifest_sha256"],
        "gpu_threshold": True,
    }))


if __name__ == "__main__":
    main()
