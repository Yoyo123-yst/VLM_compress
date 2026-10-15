"""Write EXECUTION_LOCK.json once, before any formal sample starts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
V5B = ROOT / "week2_attack_v5b_expanded_pristine40"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    frozen = json.loads((V5B / "frozen_manifest.json").read_text())
    screen = json.loads((V5B / "pool" / "screen_manifest_200.json").read_text())
    files = [
        V5B / "run_formal_attack.py",
        V5B / "run_random_noise.py",
        V5B / "replay_artifacts.py",
        V5B / "analyze_v5b.py",
        V5B / "load_samples.py",
        V5B / "verify_frozen_inputs.py",
        ROOT / "week2_attack_v4_answer_aligned" / "loss.py",
        ROOT / "week2_attack_v4_answer_aligned" / "forward.py",
        ROOT / "week2_attack_v4_answer_aligned" / "run_smoke.py",
        ROOT / "week2_attack_v4_relaxed_full" / "run_lambda.py",
    ]
    lock = {
        "role": "execution lock; does not modify FROZEN.json or the attack math",
        "frozen_json_file_sha256": sha(V5B / "FROZEN.json"),
        "frozen_payload_sha256": frozen["manifest_sha256"],
        "frozen_file_sha256": sha(V5B / "frozen_manifest.json"),
        "screen_payload_sha256": screen["manifest_sha256"],
        "screen_file_sha256": sha(V5B / "pool" / "screen_manifest_200.json"),
        "historical_union_file_sha256": sha(V5B / "exclusions" / "historical_union.json"),
        "sample_ids": [row["sample_id"] for row in frozen["samples"]],
        "checkpoints": [0, 10, 25, 50, 75, 100, 150, 200],
        "primary_selector": "fixed_step_200",
        "secondary_selector": "prospective_min_compressed_margin",
        "min_margin_rule": "min compressed margin among output_valid checkpoints; ties take the earlier step; no ANLS or answers",
        "diagnostic_selector": "GROUND_TRUTH_ORACLE_UPPER_BOUND",
        "strict_success": "clean both ANLS>=0.5, step200 Full ANLS>=0.5, FastV ANLS<0.5, Full text identical to clean Full, valid, replay, Linf, protected pages",
        "noise": "SHA256('2026100804:<id>')[:16] -> PCG64 SeedSequence; uint8 Rademacher ±8; one draw",
        "pass_bar": {
            "strict_at_least": "27/40",
            "wilson_lower_gt": 0.5,
            "full_error": "0/40",
            "full_text_unchanged": "40/40",
            "both_wrong": "0/40",
            "invalid": "0/40",
            "final_replay": "40/40",
            "mcnemar_p_lt": 0.05,
        },
        "lambda_full": 0.25,
        "steps": 200,
        "eps": 8 / 255,
        "lr": 2 / 255,
        "optimizer": "Adam",
        "scales_sha256": sha(ROOT / "week2_attack_v4_answer_aligned" / "results" / "lambda_scale.json"),
        "model_config_sha256": sha(Path("/home/yuesiting/autodl-tmp/models/llava-onevision-qwen2-7b-ov-hf/config.json")),
        "code_hashes": {str(path.relative_to(ROOT)): sha(path) for path in files},
        "smoke": {"smoke2_exit": 0, "smoke25_exit": 0, "smoke25_replay": "final/min_margin/noise ok", "sample": "50743"},
    }
    (V5B / "EXECUTION_LOCK.json").write_text(json.dumps(lock, indent=2) + "\n")
    (V5B / "execution_lock_verified.json").write_text(json.dumps({
        "execution_lock_file_sha256": sha(V5B / "EXECUTION_LOCK.json"),
        "runner_hash_matches_lock": lock["code_hashes"]["week2_attack_v5b_expanded_pristine40/run_formal_attack.py"] == sha(V5B / "run_formal_attack.py"),
        "formal_samples_started": False,
    }, indent=2) + "\n")
    print(json.dumps({"lock": sha(V5B / "EXECUTION_LOCK.json"), "runner_ok": True}))


if __name__ == "__main__":
    main()
