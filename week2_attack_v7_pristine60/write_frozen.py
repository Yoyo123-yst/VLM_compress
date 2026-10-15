"""Write FROZEN.json once. Later calls only verify it."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path

from week2_attack_v7_pristine60.parents import parent_hashes
from week2_attack_v7_pristine60.run_v7 import ATTACK_SEED_SALT, LAMBDA_SOFT

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project/week2_attack_v7_pristine60")
CODE = [
    "audit_universe.py", "run_clean_screen.py", "freeze_manifest.py", "run_v7.py",
    "selector.py", "select_run.py", "replay_selected.py", "run_controls.py",
    "run_2x2.py", "load_samples.py", "parents.py", "write_frozen.py", "analyze_v7.py",
    "tests/test_cpu.py",
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    dest = ROOT / "FROZEN.json"
    manifest = json.loads((ROOT / "frozen_manifest.json").read_text())
    if dest.exists():
        frozen = json.loads(dest.read_text())
        mismatches = []
        if frozen["frozen_manifest_sha256"] != manifest["manifest_sha256"]:
            mismatches.append("manifest")
        if frozen["lambda_soft"] != LAMBDA_SOFT:
            mismatches.append("lambda")
        for rel, digest in frozen["code_hashes"].items():
            if sha(ROOT / rel) != digest:
                mismatches.append(rel)
        if mismatches:
            raise SystemExit(f"FROZEN.json verification failed: {mismatches}")
        print(json.dumps({"frozen": "VERIFIED", "sha": sha(dest)}))
        return
    code = {rel: sha(ROOT / rel) for rel in CODE}
    env = {
        "python": subprocess.check_output(["/home/yuesiting/autodl-tmp/conda/envs/vattack/bin/python", "-c", "import sys,torch; print(sys.version); print(torch.__version__); print(torch.version.cuda)"], text=True),
        "platform": platform.platform(),
        "nvidia_smi": subprocess.check_output(["nvidia-smi", "-L"], text=True),
    }
    payload = {
        "protocol": "week2_attack_v7_pristine60",
        "variant": "B",
        "lambda_soft": LAMBDA_SOFT,
        "beta": None,
        "loss": "L_comp + lambda_soft * L_soft_full",
        "fastv_r": 0.25,
        "epsilon": 8 / 255,
        "optimizer": "Adam",
        "lr": 2 / 255,
        "steps": 200,
        "checkpoints": [0, 10, 25, 50, 75, 100, 150, 200],
        "attack_seed_salt": ATTACK_SEED_SALT,
        "attack_seed": "sha256(f'{salt}:{sample_id}:main')[:8] applied to both the main run and the repeat run",
        "single_noise_salt": 2026100905,
        "trajectory_noise_salt": 2026100906,
        "trajectory_steps": [10, 25, 50, 75, 100, 150, 200],
        "repeat_draw_seed": 2026100903,
        "bootstrap_seed": 2026100907,
        "screen_seed": 2026100901,
        "final_seed": 2026100902,
        "frozen_manifest_sha256": manifest["manifest_sha256"],
        "screen_manifest_sha256": (ROOT / "pool" / "screen_manifest.sha256").read_text().strip(),
        "historical_union_sha256": (ROOT / "exclusions" / "historical_union.sha256").read_text().strip(),
        "parent_hashes": parent_hashes(),
        "code_hashes": code,
        "environment": env,
        "model_path": "/home/yuesiting/autodl-tmp/models/llava-onevision-qwen2-7b-ov-hf",
        "selector": "step>0, uint8 Linf>0, output_valid, full text exact, uint8 Linf<=8/255; min compressed margin then earlier step",
        "claim_boundary": "source-bounded compression-only; not selection-stable",
    }
    dest.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"frozen": "WRITTEN", "sha": sha(dest), "n": len(manifest["samples"])}))


if __name__ == "__main__":
    main()
