"""Write the pre-run provenance lock for the frozen v28 comparison."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
MODEL = Path("/home/yuesiting/autodl-tmp/models/llava-onevision-qwen2-7b-ov-hf")

SOURCES = [
    "week2_attack_v28_complete_front_holdout/freeze_execution.py",
    "week2_attack_v28_complete_front_holdout/run_attack.py",
    "week2_attack_v28_complete_front_holdout/smoke_attack.py",
    "week2_attack_v28_complete_front_holdout/evaluate_png.py",
    "week2_attack_v28_complete_front_holdout/load_samples.py",
    "week2_attack_v28_complete_front_holdout/sample_specs.json",
    "week2_attack_v28_complete_front_holdout/sample_selection_freeze.json",
    "week2_attack_v28_complete_front_holdout/frozen_targets.json",
    "week2_attack_v28_complete_front_holdout/protocol.md",
    "week2_attack_v25_full_answer_competition_pilot/run_iterative_compare.py",
    "week2_attack_v25_full_answer_competition_pilot/generate_clean_candidates.py",
    "week2_attack_v25_full_answer_competition_pilot/score_smoke.py",
    "week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation/run_v7d.py",
    "week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation/losses.py",
    "week2_attack_v11_objective_optimizer_diagnostic/run_v11.py",
    "week2_attack_v13_branch_reachability_probe/run_branch_probe.py",
    "week2_attack_v7_pristine60/run_v7.py",
    "week2_attack_v7_pristine60/load_samples.py",
    "week2_attack_v5b_expanded_pristine40/load_samples.py",
    "week2_attack_v4_answer_aligned/common.py",
    "week2_attack_v4_answer_aligned/forward.py",
    "week2_attack_v4_answer_aligned/loss.py",
    "week2_attack_v4_answer_aligned/run_smoke.py",
    "week2_pilot/attack.py",
    "week1_mechanism/adapters/runtime.py",
    "week1_mechanism/metrics_task.py",
    "week1_mechanism/run_eval.py",
    "gate_experiment/run_causal_v2.py",
]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    source_hashes = {}
    missing = []
    for rel in SOURCES:
        path = ROOT / rel
        if not path.is_file():
            missing.append(rel)
        else:
            source_hashes[rel] = sha(path)
    if missing:
        raise SystemExit(f"MISSING_SOURCE_FILES: {missing}")

    model_files = sorted(p for p in MODEL.iterdir() if p.is_file())
    model_hashes = {p.name: {"size": p.stat().st_size, "sha256": sha(p)} for p in model_files}
    versions = subprocess.check_output(
        ["/home/yuesiting/autodl-tmp/conda/envs/vattack/bin/python", "-c",
         "import torch, transformers, PIL, numpy; print({'torch':torch.__version__,'transformers':transformers.__version__,'pillow':PIL.__version__,'numpy':numpy.__version__})"],
        text=True,
    ).strip()
    payload = {
        "status": "PRE_RUN_EXECUTION_LOCK",
        "created_at_local": subprocess.check_output(["date", "+%Y-%m-%d %H:%M:%S %Z"], text=True).strip(),
        "python_platform": platform.platform(),
        "environment_versions": versions,
        "source_sha256": source_hashes,
        "model_dir": str(MODEL),
        "model_file_manifest": model_hashes,
        "samples_sha256": sha(OUT / "sample_specs.json"),
        "targets_sha256": sha(OUT / "frozen_targets.json"),
        "protocol_sha256": sha(OUT / "protocol.md"),
        "no_source_edits_after_lock": True,
    }
    (OUT / "freeze_execution_lock.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "source_files": len(source_hashes),
                      "model_files": len(model_hashes), "model_bytes": sum(v["size"] for v in model_hashes.values()),
                      "lock": str(OUT / "freeze_execution_lock.json")}, indent=2))


if __name__ == "__main__":
    main()
