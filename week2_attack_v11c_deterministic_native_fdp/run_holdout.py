"""Frozen holdout Native-FDP runner using the deterministic v11c runtime.

This runner deliberately uses the frozen holdout list and never reads target
FastV outputs or answer semantics during checkpoint selection.
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

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("VATTACK_ATTN_IMPLEMENTATION", "sdpa")
os.environ.setdefault("VATTACK_DETERMINISTIC_RESIZE", "1")
os.environ.setdefault("VATTACK_RESIZE_MODE", "bilinear")

import torch

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
HERE = ROOT / "week2_attack_v11c_deterministic_native_fdp"
sys.path[:0] = [str(ROOT), str(ROOT / "week1_mechanism"), str(ROOT / "gate_experiment")]

from week2_attack_v11b_h0_update_forensics import run_v11b as base  # noqa: E402


def ids() -> list[str]:
    return json.loads((HERE / "holdout_ids.json").read_text())["ids"]


def configure() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)


def environment() -> dict:
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
        "resize_mode": os.environ.get("VATTACK_RESIZE_MODE"),
        "deterministic_resize": os.environ.get("VATTACK_DETERMINISTIC_RESIZE"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
    }


def run_one(sample: str, out: Path, steps: int) -> None:
    configure()
    out.mkdir(parents=True, exist_ok=True)
    (out / "environment.json").write_text(json.dumps(environment(), indent=2) + "\n")
    model, processor = base.load_model_frozen()
    item = base.v7d.load_pristine(sample)
    base.attack_trace(model, processor, item, out, steps)


def launch(sample: str, gpu: str, steps: int) -> None:
    out = HERE / "holdout_train" / sample
    log_dir = HERE / "holdout_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPATH"] = "."
    env["PYTHONUNBUFFERED"] = "1"
    command = [
        "/home/yuesiting/autodl-tmp/conda/envs/vattack/bin/python",
        str(HERE / "run_holdout.py"), "run", "--sample", sample,
        "--steps", str(steps), "--out", str(out),
    ]
    log = log_dir / f"{sample}.log"
    with log.open("w") as handle:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
    progress = HERE / "holdout_progress.jsonl"
    with progress.open("a") as handle:
        handle.write(json.dumps({"sample": sample, "gpu": gpu, "returncode": result.returncode}) + "\n")
    if result.returncode != 0:
        raise RuntimeError(f"holdout sample {sample} failed; see {log}")


def drive(steps: int) -> None:
    samples = ids()
    gpus = ["0", "1", "4", "6"]
    report = {
        "status": "RUNNING",
        "runner": "v11c_deterministic_native_fdp_holdout",
        "frozen_ids": samples,
        "steps": steps,
        "gpu_schedule": {sample: gpus[index % len(gpus)] for index, sample in enumerate(samples)},
    }
    (HERE / "holdout_run_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    # Keep one process per free physical GPU. Each subprocess loads its own model.
    running = []
    for index, sample in enumerate(samples):
        gpu = gpus[index % len(gpus)]
        running.append(subprocess.Popen([
            "/home/yuesiting/autodl-tmp/conda/envs/vattack/bin/python",
            str(HERE / "run_holdout.py"), "run", "--sample", sample,
            "--steps", str(steps), "--out", str(HERE / "holdout_train" / sample),
        ], cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "PYTHONNOUSERSITE": "1", "PYTHONPATH": ".", "PYTHONUNBUFFERED": "1"},
        stdout=(HERE / "holdout_logs" / f"{sample}.log").open("w"), stderr=subprocess.STDOUT))
        if len(running) == len(gpus) or index == len(samples) - 1:
            for process in running:
                code = process.wait()
                if code != 0:
                    raise RuntimeError(f"holdout worker failed with code {code}")
            running = []
    report["status"] = "TRAINING_COMPLETE"
    (HERE / "holdout_run_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "samples": len(samples)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("run", "drive"))
    parser.add_argument("--sample")
    parser.add_argument("--out")
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()
    if args.mode == "drive":
        (HERE / "holdout_logs").mkdir(parents=True, exist_ok=True)
        configure()
        drive(args.steps)
        return
    if args.sample not in ids():
        raise RuntimeError(f"sample must be in frozen holdout list: {ids()}")
    run_one(args.sample, Path(args.out), args.steps)


if __name__ == "__main__":
    main()
