"""Frozen constants for pristine40 validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
V5 = ROOT / "week2_attack_v5_pristine40"
V4_SCALE = ROOT / "week2_attack_v4_answer_aligned" / "results" / "lambda_scale.json"
MODEL_PATH = "/home/yuesiting/autodl-tmp/models/llava-onevision-qwen2-7b-ov-hf"
BASELINE = ROOT / "gate_experiment" / "results" / "clean_baseline.csv"
RATIO = 0.25
LAMBDA_FULL = 0.25
EPS = 8 / 255
STEPS = 200
POOL_SEED = 2026100801
GLOBAL_SEED = 42
CHECKPOINTS = [0, 10, 25, 50, 75, 100, 150, 200]
WRONG = 0.5
TOKENS_PER_PAGE = 3699


def code_hashes() -> dict[str, str]:
    files = [
        V5 / "run_attack.py",
        V5 / "run_clean_gate.py",
        V5 / "run_noise.py",
        V5 / "replay.py",
        V5 / "analyze.py",
        V5 / "common.py",
        ROOT / "week2_attack_v4_answer_aligned" / "loss.py",
        ROOT / "week2_attack_v4_answer_aligned" / "forward.py",
        ROOT / "week2_attack_v4_answer_aligned" / "run_smoke.py",
    ]
    out = {}
    for path in files:
        if path.exists():
            out[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def load_scales() -> dict:
    scales = json.loads(V4_SCALE.read_text())
    scales["lambda_full"] = LAMBDA_FULL
    return scales


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
