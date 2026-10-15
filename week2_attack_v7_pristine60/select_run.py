"""Select one checkpoint. Ranking never receives ANLS, FastV text, or labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from week2_attack_v7_pristine60.selector import choose, eligible

EPS = 8 / 255


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pixel_linf(clean: Path, adv: Path) -> float:
    a = np.asarray(Image.open(clean).convert("RGB"), dtype=np.int16)
    b = np.asarray(Image.open(adv).convert("RGB"), dtype=np.int16)
    return float(np.abs(a - b).max()) / 255.0


def select_tree(run: Path) -> tuple[list[dict], list[dict]]:
    grouped = {}
    paths = sorted(run.glob("shard_s*/**/checkpoints.jsonl")) + sorted(run.glob("*/checkpoints.jsonl"))
    seen = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                grouped.setdefault(row["sample_id"], {})[int(row["step"])] = row
    selected = []
    table = []
    for sid, by_step in grouped.items():
        art = next(run.glob(f"**/{sid}/clean.png")).parent
        clean = art / "clean.png"
        legal = []
        for step, row in sorted(by_step.items()):
            if step <= 0:
                continue
            png = art / row["png"]
            gap = pixel_linf(clean, png)
            record = {
                "sample_id": sid,
                "step": step,
                "margin": float(row["margin"]),
                "nonzero": gap > 0,
                "output_valid": bool(row["output_valid"]),
                "full_exact": row["pred_full"] == by_step[0]["pred_full"],
                "linf_ok": gap <= EPS + 1e-6,
                "pixel_linf_uint8": gap,
            }
            table.append({**record, "png_sha256": sha(png), "clean_sha256": sha(clean)})
            if eligible(record):
                legal.append(record)
        pick = choose(legal)
        if pick is None:
            selected.append({"sample_id": sid, "status": "NO_FEASIBLE_NONZERO_CHECKPOINT"})
            continue
        source = by_step[pick["step"]]
        selected.append({
            "sample_id": sid,
            "status": "SELECTED",
            "step": pick["step"],
            "margin": pick["margin"],
            "png": source["png"],
            "artifact": str(art),
            "pred_full": source["pred_full"],
            "pred_fastv": source["pred_fastv"],
            "anls_full": source["anls_full"],
            "anls_fastv": source["anls_fastv"],
            "output_valid": source["output_valid"],
            "pixel_linf_uint8": next(row["pixel_linf_uint8"] for row in table if row["sample_id"] == sid and row["step"] == pick["step"]),
            "png_sha256": sha(art / source["png"]),
            "keep_hash": source["keep_hash"],
            "k": source["k"],
            "full_exact": True,
            "selector": "prospective_min_compressed_margin_then_earlier_step",
        })
    return selected, table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    selected, table = select_tree(args.run)
    args.out.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id", "status", "step", "margin", "png", "artifact", "pred_full", "pred_fastv",
        "anls_full", "anls_fastv", "output_valid", "pixel_linf_uint8", "png_sha256",
        "keep_hash", "k", "full_exact", "selector",
    ]
    with (args.out / "selected.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected)
    with (args.out / "candidate_table.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0].keys()))
        writer.writeheader()
        writer.writerows(table)
    print(json.dumps({"selected": sum(row["status"] == "SELECTED" for row in selected), "n": len(selected)}))


if __name__ == "__main__":
    main()
