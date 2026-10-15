"""Single Rademacher noise and the 7-candidate selector-matched trajectory."""

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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "week1_mechanism"))
sys.path.insert(0, str(ROOT / "gate_experiment"))

from week2_attack_v4_answer_aligned.run_smoke import evaluate_uint8  # noqa: E402
from week2_attack_v7_pristine60.load_samples import load_items, manifest  # noqa: E402
from week2_attack_v7_pristine60.run_v7 import load_model_frozen, make_session, sample_seed  # noqa: E402
from week2_attack_v7_pristine60.selector import choose, eligible  # noqa: E402
from week2_pilot.attack import page_to_float  # noqa: E402

SINGLE_SALT = 2026100905
TRAJ_SALT = 2026100906
TRAJ_STEPS = (10, 25, 50, 75, 100, 150, 200)
EPS = 8 / 255


def rademacher(clean: Image.Image, seed: int) -> tuple[Image.Image, float]:
    base = np.asarray(clean.convert("RGB"), dtype=np.int16)
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))
    signs = rng.integers(0, 2, size=base.shape, dtype=np.int16) * 2 - 1
    clipped = np.clip(base + signs * 8, 0, 255).astype(np.uint8)
    image = Image.fromarray(clipped, mode="RGB")
    gap = float((page_to_float(image) - page_to_float(clean)).abs().max())
    return image, gap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["single", "trajectory"], required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--nshards", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    specs = [row for index, row in enumerate(manifest()["samples"]) if index % args.nshards == args.shard]
    items = {item["meta"]["sample_id"]: item for item in load_items(specs)}
    model, processor = load_model_frozen()
    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / ("single.csv" if args.mode == "single" else "trajectory.csv")
    done = set()
    if csv_path.exists():
        with csv_path.open() as handle:
            done = {row["sample_id"] for row in csv.DictReader(handle)}
    for spec in specs:
        sid = spec["sample_id"]
        if sid in done:
            continue
        item = items[sid]
        session = make_session(model, processor, item)
        art = args.out / sid
        art.mkdir(parents=True, exist_ok=True)
        clean = item["images"][item["aux_idx"]]
        t0 = time.time()
        if args.mode == "single":
            seed = sample_seed(SINGLE_SALT, sid, "single")
            image, gap = rademacher(clean, seed)
            image.save(art / "noise.png")
            measured = evaluate_uint8(session, image, 0.0)
            again = evaluate_uint8(session, Image.open(art / "noise.png").convert("RGB"), 0.0)
            replay_ok = again["pred_full"] == measured["pred_full"] and again["pred_fastv"] == measured["pred_fastv"]
            row = {
                "sample_id": sid,
                "seed": seed,
                "uint8_linf": gap,
                "pred_full": measured["pred_full"],
                "pred_fastv": measured["pred_fastv"],
                "anls_full": measured["anls_full"],
                "anls_fastv": measured["anls_fastv"],
                "output_valid": int(measured["output_valid"]),
                "margin": measured["margin"],
                "keep_hash": measured["keep_hash"],
                "k": measured["k"],
                "replay_ok": int(replay_ok),
                "png_sha256": hashlib.sha256((art / "noise.png").read_bytes()).hexdigest(),
                "seconds": f"{time.time() - t0:.1f}",
            }
        else:
            legal = []
            provenance = []
            for step in TRAJ_STEPS:
                seed = sample_seed(TRAJ_SALT, sid, str(step))
                image, gap = rademacher(clean, seed)
                dest = art / f"step_{step:03d}.png"
                image.save(dest)
                measured = evaluate_uint8(session, image, 0.0)
                record = {
                    "sample_id": sid,
                    "step": step,
                    "seed": seed,
                    "margin": float(measured["margin"]),
                    "nonzero": gap > 0,
                    "output_valid": bool(measured["output_valid"]),
                    "full_exact": measured["pred_full"] == item["baseline_full"]["pred"],
                    "linf_ok": gap <= EPS + 1e-6,
                    "pixel_linf_uint8": gap,
                    "pred_full": measured["pred_full"],
                    "pred_fastv": measured["pred_fastv"],
                    "anls_full": measured["anls_full"],
                    "anls_fastv": measured["anls_fastv"],
                    "keep_hash": measured["keep_hash"],
                    "png_sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
                }
                provenance.append(record)
                if eligible(record):
                    legal.append(record)
            pick = choose(legal)
            if pick is None:
                row = {"sample_id": sid, "status": "NO_FEASIBLE_RANDOM_CANDIDATE", "seconds": f"{time.time() - t0:.1f}"}
            else:
                source = next(item_row for item_row in provenance if item_row["step"] == pick["step"])
                image = Image.open(art / f"step_{pick['step']:03d}.png").convert("RGB")
                again = evaluate_uint8(session, image, 0.0)
                replay_ok = again["pred_full"] == source["pred_full"] and again["pred_fastv"] == source["pred_fastv"]
                row = {
                    "sample_id": sid,
                    "status": "SELECTED" if replay_ok else "SELECTED_CANDIDATE_REPLAY_FAILED",
                    "step": pick["step"],
                    "margin": pick["margin"],
                    "pred_full": source["pred_full"],
                    "pred_fastv": source["pred_fastv"],
                    "anls_full": source["anls_full"],
                    "anls_fastv": source["anls_fastv"],
                    "output_valid": int(source["output_valid"]),
                    "uint8_linf": source["pixel_linf_uint8"],
                    "keep_hash": source["keep_hash"],
                    "replay_ok": int(replay_ok),
                    "png_sha256": source["png_sha256"],
                    "seconds": f"{time.time() - t0:.1f}",
                }
            (art / "candidates.json").write_text(json.dumps(provenance, indent=2) + "\n")
        fields = [
            "sample_id", "status", "seed", "step", "margin", "pred_full", "pred_fastv",
            "anls_full", "anls_fastv", "output_valid", "uint8_linf", "keep_hash", "k",
            "replay_ok", "png_sha256", "seconds",
        ]
        new = not csv_path.exists()
        with csv_path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            if new:
                writer.writeheader()
            writer.writerow(row)
        print(json.dumps({"id": sid, "mode": args.mode, "status": row.get("status", "single")}), flush=True)
        del session
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
