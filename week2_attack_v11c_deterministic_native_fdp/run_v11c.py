"""Deterministic admission and isolated Native-FDP runner.

This directory is an experiment fork. The historical v7d source is not edited;
the only shared-loader change is an environment-controlled attention backend.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

# These must be set before importing torch/CUDA kernels in the v7d stack.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# Eager attention materializes the full sequence attention matrix and OOMs on
# this 7B model. SDPA is the memory-safe backend used for the deterministic
# runner; the historical v7d default remains unchanged in its own directory.
os.environ.setdefault("VATTACK_ATTN_IMPLEMENTATION", "sdpa")
os.environ.setdefault("VATTACK_DETERMINISTIC_RESIZE", "1")
os.environ.setdefault("VATTACK_RESIZE_MODE", "bilinear")

import numpy as np
import torch

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
HERE = ROOT / "week2_attack_v11c_deterministic_native_fdp"
sys.path[:0] = [str(ROOT), str(ROOT / "week1_mechanism"), str(ROOT / "gate_experiment")]

from week2_attack_v11b_h0_update_forensics import run_v11b as base  # noqa: E402

ADMISSION_SAMPLES = ("1025", "60700")
TRAIN_SAMPLES = ("1025", "2008", "47450", "60700", "7006", "39204")
SAMPLES = ADMISSION_SAMPLES
REPEATS = 3


def configure_determinism() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)


def env_snapshot() -> dict:
    device = torch.cuda.current_device() if torch.cuda.is_available() else None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": None if device is None else torch.cuda.get_device_name(device),
        "gpu_id": device,
        "attn_implementation": os.environ.get("VATTACK_ATTN_IMPLEMENTATION"),
        "deterministic_resize": os.environ.get("VATTACK_DETERMINISTIC_RESIZE"),
        "resize_mode": os.environ.get("VATTACK_RESIZE_MODE"),
        "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
    }


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def signature(path: Path) -> dict:
    rows = {row["step"]: row for row in json.loads((path / "curve.json").read_text())}
    updates = json.loads((path / "updates.json").read_text())
    normalized_updates = []
    for row in updates:
        row = json.loads(json.dumps(row))
        row.pop("process_id", None)
        row.pop("gpu_id", None)
        normalized_updates.append(row)
    return {
        "curve": {
            step: {
                "png_sha256": rows[step]["png_sha256"],
                "texts": rows[step]["texts"],
                "margins": rows[step]["margins"],
                "keeps": rows[step]["keeps"],
                "selector": rows[step]["selector"],
            }
            for step in sorted(rows)
        },
        "updates": normalized_updates,
    }


def summarize_existing() -> None:
    report = {
        "status": "RUNNING",
        "runner": "v11c_deterministic_native_fdp",
        "comparison": "two fresh processes, cross-GPU; process_id/gpu_id excluded only from update-state equality",
        "config": {
            "attn_implementation": os.environ.get("VATTACK_ATTN_IMPLEMENTATION"),
            "resize_mode": os.environ.get("VATTACK_RESIZE_MODE"),
            "deterministic_resize": os.environ.get("VATTACK_DETERMINISTIC_RESIZE"),
            "deterministic_algorithms": True,
        },
        "samples": {},
    }
    for sample in SAMPLES:
        paths = [HERE / "admission_trace" / sample / f"repeat_{i}" for i in (0, 1)]
        signatures = [signature(path) for path in paths]
        report["samples"][sample] = {
            "paths": [str(path.relative_to(ROOT)) for path in paths],
            "curve_equal": signatures[0]["curve"] == signatures[1]["curve"],
            "update_state_equal": signatures[0]["updates"] == signatures[1]["updates"],
            "repeatable": signatures[0] == signatures[1],
        }
    report["status"] = (
        "DETERMINISTIC_ADMISSION_PASSED"
        if all(value["repeatable"] for value in report["samples"].values())
        else "DETERMINISTIC_ADMISSION_FAILED"
    )
    (HERE / "admission_trace_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "samples": report["samples"]}), flush=True)


def summarize_training() -> None:
    """Collect completed/partial training artifacts without rerunning inference."""
    report = {
        "status": "RUNNING",
        "runner": "v11c_deterministic_native_fdp",
        "protocol": "200-step deterministic Native-FDP; Full exact and pixel L_inf are hard gates",
        "config": {
            "attn_implementation": os.environ.get("VATTACK_ATTN_IMPLEMENTATION", "sdpa"),
            "resize_mode": os.environ.get("VATTACK_RESIZE_MODE", "bilinear"),
            "deterministic_resize": os.environ.get("VATTACK_DETERMINISTIC_RESIZE", "1"),
            "deterministic_algorithms": True,
            "historical_v7d_unchanged": True,
        },
        "samples": {},
    }
    complete = True
    for sample in TRAIN_SAMPLES:
        out = HERE / "training_h0" / sample
        curve_path = out / "curve.json"
        selected_path = out / "selected.json"
        item = {"path": str(out.relative_to(ROOT)), "status": "NOT_RUN"}
        if curve_path.exists():
            curve = json.loads(curve_path.read_text())
            item["checkpoint_count"] = len(curve)
            item["checkpoint_steps"] = [row.get("step") for row in curve]
            item["optimization_steps"] = 200
            item["last_step"] = curve[-1].get("step") if curve else None
            if curve:
                last = curve[-1]
                item["last_full_exact"] = last.get("full_exact")
                item["last_texts"] = last.get("texts")
                item["last_margins"] = last.get("margins")
                item["last_keeps"] = last.get("keeps")
            item["status"] = "DONE" if (out / "environment.json").exists() else "PARTIAL"
        if selected_path.exists():
            selected = json.loads(selected_path.read_text())
            item["selected"] = selected.get("chosen")
            item["clean_text"] = selected.get("clean_text")
            if item.get("last_full_exact") is None and selected.get("chosen") is not None:
                item["last_full_exact"] = selected["chosen"].get("full_exact")
        if item["status"] != "DONE":
            complete = False
        report["samples"][sample] = item
    report["status"] = "TRAINING_H0_COMPLETE" if complete else "TRAINING_H0_PARTIAL"
    (HERE / "training_h0_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "samples": report["samples"]}), flush=True)


def run_one(sample: str, out: Path, steps: int) -> None:
    configure_determinism()
    out.mkdir(parents=True, exist_ok=True)
    (out / "environment.json").write_text(json.dumps(env_snapshot(), indent=2) + "\n")
    model, processor = base.load_model_frozen()
    item = base.v7d.load_dev(sample)
    # Use the forensic implementation so each update records the gradient,
    # Adam moments, pre/post projection delta, and the endpoint replay fields.
    # This is still the same Native-FDP update; the extra records are read-only.
    base.attack_trace(model, processor, item, out, steps)


def launch(sample: str, out: Path, steps: int) -> None:
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPATH"] = "."
    env["PYTHONUNBUFFERED"] = "1"
    env["VATTACK_ATTN_IMPLEMENTATION"] = os.environ.get("VATTACK_ATTN_IMPLEMENTATION", "sdpa")
    env["VATTACK_DETERMINISTIC_RESIZE"] = os.environ.get("VATTACK_DETERMINISTIC_RESIZE", "1")
    env["VATTACK_RESIZE_MODE"] = os.environ.get("VATTACK_RESIZE_MODE", "bilinear")
    command = [
        "/home/yuesiting/autodl-tmp/conda/envs/vattack/bin/python",
        str(HERE / "run_v11c.py"), "run", "--sample", sample,
        "--out", str(out), "--steps", str(steps),
    ]
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def drive() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    configure_determinism()
    report = {"status": "RUNNING", "config": env_snapshot(), "samples": {}}
    for sample in SAMPLES:
        rows = []
        for repeat in range(REPEATS):
            out = HERE / "admission" / sample / f"repeat_{repeat}"
            launch(sample, out, 10)
            rows.append(signature(out))
        report["samples"][sample] = {
            "repeatable": all(row == rows[0] for row in rows[1:]),
            "signatures": rows,
        }
    report["status"] = (
        "DETERMINISTIC_ADMISSION_PASSED"
        if all(value["repeatable"] for value in report["samples"].values())
        else "DETERMINISTIC_ADMISSION_FAILED"
    )
    (HERE / "admission_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"]}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("run", "drive", "summarize", "summarize_training"))
    parser.add_argument("--sample")
    parser.add_argument("--out")
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    if args.mode == "drive":
        drive()
        return
    if args.mode == "summarize":
        summarize_existing()
        return
    if args.mode == "summarize_training":
        summarize_training()
        return
    if args.sample not in TRAIN_SAMPLES:
        raise RuntimeError(f"sample must be one of {TRAIN_SAMPLES}")
    run_one(args.sample, Path(args.out), args.steps)


if __name__ == "__main__":
    main()
