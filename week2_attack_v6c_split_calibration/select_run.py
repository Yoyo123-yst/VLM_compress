"""Select one checkpoint per sample. Does not read ANLS or FastV text while ranking."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from week2_attack_v6c_split_calibration.selector import choose, eligible

EPS = 8 / 255


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pixel_linf(clean: Path, adv: Path) -> float:
    a = np.asarray(Image.open(clean).convert("RGB"), dtype=np.int16)
    b = np.asarray(Image.open(adv).convert("RGB"), dtype=np.int16)
    return float(np.abs(a - b).max()) / 255.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    grouped = {}
    for path in sorted(args.run.glob("shard_s*/**/checkpoints.jsonl")) + sorted(args.run.glob("**/checkpoints.jsonl")):
        if "shard_s" not in str(path) and args.run.name.startswith("smoke"):
            pass
        for line in path.read_text().splitlines():
            row = json.loads(line)
            grouped.setdefault(row["sample_id"], []).append(row)
    # Prefer shard checkpoints over a nested duplicate by last write order; unique by step.
    selected = []
    table = []
    for sid, rows in grouped.items():
        by_step = {}
        for row in rows:
            by_step[int(row["step"])] = row
        art = None
        for candidate in args.run.glob(f"**/{sid}/clean.png"):
            art = candidate.parent
            break
        if art is None:
            raise RuntimeError(f"missing artifact for {sid}")
        clean = art / "clean.png"
        manifest_full = None
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
                "full_exact": row["pred_full"] == by_step[0]["pred_full"] if 0 in by_step else row["pred_full"] == row["pred_full"],
                "linf_ok": gap <= EPS + 1e-6,
                "pixel_linf_uint8": gap,
                "png_sha256": sha(png),
                "clean_sha256": sha(clean),
            }
            if manifest_full is None and 0 in by_step:
                record["full_exact"] = row["pred_full"] == by_step[0]["pred_full"]
            table.append(record)
            if eligible(record):
                legal.append({"step": step, "margin": record["margin"]})
        pick = choose(legal)
        if pick is None:
            selected.append({"sample_id": sid, "status": "NO_FEASIBLE_NONZERO_CHECKPOINT"})
            continue
        source = by_step[pick["step"]]
        pixel = next(row for row in table if row["sample_id"] == sid and row["step"] == pick["step"])
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
            "pixel_linf_uint8": pixel["pixel_linf_uint8"],
            "nonzero": pixel["nonzero"],
            "linf_ok": pixel["linf_ok"],
            "full_exact": pixel["full_exact"],
            "png_sha256": pixel["png_sha256"],
            "keep_hash": source["keep_hash"],
            "aux_share": source["aux_share"],
            "evidence_share": source["evidence_share"],
        })
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "candidate_table.json").write_text(json.dumps(table, indent=2) + "\n")
    with (args.out / "selected.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    print(json.dumps({"samples": len(selected), "selected": sum(row["status"] == "SELECTED" for row in selected)}))


if __name__ == "__main__":
    main()
