"""One pre-registered Rademacher ±8 noise image per formal sample."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
V5B = ROOT / "week2_attack_v5b_expanded_pristine40"
sys.path.insert(0, str(ROOT))

import week2_attack_v4_answer_aligned.run_smoke as smoke  # noqa: E402
from adapters.runtime import clear_hooks  # noqa: E402
from gate_experiment.run_causal_v2 import generate, load_model, prepare  # noqa: E402
from week1_mechanism.metrics_task import normalize  # noqa: E402
from week2_attack_v4_answer_aligned.common import GROUPS, write_json  # noqa: E402
from week2_attack_v4_answer_aligned.run_smoke import Session, evaluate_uint8  # noqa: E402
from week2_pilot.attack import page_to_float  # noqa: E402
from week2_attack_v5b_expanded_pristine40.load_samples import load_formal_items, manifest  # noqa: E402

NOISE_SEED = 2026100804


def noise_seed(sample_id: str) -> int:
    raw = hashlib.sha256(f"{NOISE_SEED}:{sample_id}".encode()).hexdigest()
    return int(raw[:16], 16)


def make_noise(clean: Image.Image, sample_id: str) -> tuple[Image.Image, dict]:
    base = np.asarray(clean.convert("RGB"), dtype=np.int16)
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(noise_seed(sample_id))))
    signs = rng.integers(0, 2, size=base.shape, dtype=np.int16) * 2 - 1
    shifted = base + signs * 8
    clipped = np.clip(shifted, 0, 255).astype(np.uint8)
    saturated = float((shifted != clipped).any(axis=-1).mean())
    image = Image.fromarray(clipped, mode="RGB")
    gap = (page_to_float(image) - page_to_float(clean)).abs()
    stats = {
        "seed": noise_seed(sample_id),
        "uint8_linf": float(gap.max()),
        "l1": float(gap.sum()),
        "l2": float(gap.float().norm()),
        "saturated_pixel_fraction": saturated,
    }
    return image, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshards", type=int, default=4)
    parser.add_argument("--dev50743", action="store_true")
    parser.add_argument("--tag", default="formal")
    args = parser.parse_args()
    smoke.LAMBDA_FULL = 0.25
    if args.dev50743:
        from week2_attack_v4_answer_aligned.common import load_frozen_samples
        items = [item for item in load_frozen_samples() if item["meta"]["sample_id"] == "50743"]
        out = V5B / "tests" / args.tag
    else:
        payload = manifest()
        specs = [row for index, row in enumerate(payload["samples"]) if index % args.nshards == args.shard]
        items = load_formal_items(specs)
        out = V5B / "random_control" / f"shard_s{args.shard}"
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "noise.csv"
    done = set()
    if csv_path.exists():
        done = {row["sample_id"] for row in csv.DictReader(csv_path.open())}
    model, processor = load_model()
    for param in model.parameters():
        param.requires_grad_(False)
    fields = ["sample_id", "pred_full", "pred_fastv", "anls_full", "anls_fastv", "keep_hash", "k", "uint8_linf", "png_sha256"]
    for item in items:
        sid = item["meta"]["sample_id"]
        if sid in done:
            continue
        GROUPS.setdefault(sid, "v5b")
        clear_hooks(model)
        inputs = prepare(processor, item["images"], item["meta"]["question"])
        with torch.inference_mode():
            raw = generate(model, inputs)
        answer = torch.tensor(raw.detach().cpu().tolist(), dtype=torch.long)
        session = Session(model, processor, item, answer)
        image, stats = make_noise(item["images"][item["aux_idx"]], sid)
        dest_dir = (V5B / "attacks" / "artifacts" / sid) if not args.dev50743 else (out / sid)
        dest_dir.mkdir(parents=True, exist_ok=True)
        path = dest_dir / "random_noise.png"
        image.save(path)
        t0 = time.time()
        measured = evaluate_uint8(session, Image.open(path).convert("RGB"), 0.0)
        row = {
            "sample_id": sid,
            "pred_full": measured["pred_full"],
            "pred_fastv": measured["pred_fastv"],
            "norm_full": normalize(measured["pred_full"]),
            "norm_fastv": normalize(measured["pred_fastv"]),
            "anls_full": measured["anls_full"],
            "anls_fastv": measured["anls_fastv"],
            "output_valid": measured["output_valid"],
            "keep_hash": measured["keep_hash"],
            "k": measured["k"],
            "uint8_linf": stats["uint8_linf"],
            "l1": stats["l1"],
            "l2": stats["l2"],
            "saturated_pixel_fraction": stats["saturated_pixel_fraction"],
            "png_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "seed": stats["seed"],
            "seconds": time.time() - t0,
        }
        write_json(dest_dir / "random_noise_provenance.json", row)
        new = not csv_path.exists()
        with csv_path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if new:
                writer.writeheader()
            writer.writerow(row)
        print(json.dumps({"noise": sid, "fastv": measured["anls_fastv"], "full": measured["anls_full"]}), flush=True)
        del session
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
