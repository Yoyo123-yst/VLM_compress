"""Variant A selector. Ranking never sees ANLS, FastV text, or replay."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

V5B = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project/week2_attack_v5b_expanded_pristine40")
OUT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project/week2_attack_v6_survivor_upgrade/variant_a_audit")
EPS = 8 / 255
DEV = ["1025", "2008", "7006", "39204", "47450", "47567", "50779", "51133", "60700", "6194", "62941", "64506"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pixel_linf(clean: Image.Image, adv: Image.Image) -> float:
    a = np.asarray(clean.convert("RGB"), dtype=np.int16)
    b = np.asarray(adv.convert("RGB"), dtype=np.int16)
    if a.shape != b.shape:
        raise RuntimeError(f"shape {a.shape} != {b.shape}")
    return float(np.abs(a - b).max()) / 255.0


def choose(candidates: list[dict]) -> dict | None:
    """candidates already contain only selector-legal fields."""
    if not candidates:
        return None
    return sorted(candidates, key=lambda row: (row["margin"], row["step"]))[0]


def main():
    manifest = {row["sample_id"]: row for row in json.loads((V5B / "frozen_manifest.json").read_text())["samples"]}
    grouped = {}
    for path in sorted((V5B / "attacks").glob("shard_s*/checkpoints.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            grouped.setdefault(row["sample_id"], []).append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    table = []
    selected = []
    for sid, spec in manifest.items():
        art = V5B / "attacks" / "artifacts" / sid
        clean = Image.open(art / "clean.png").convert("RGB")
        clean_hash = sha(art / "clean.png")
        legal = []
        for row in grouped[sid]:
            step = int(row["step"])
            png = art / "checkpoints" / f"step_{step:03d}.png"
            if step <= 0 or not png.exists():
                continue
            gap = pixel_linf(clean, Image.open(png).convert("RGB"))
            record = {
                "sample_id": sid,
                "step": step,
                "png_sha256": sha(png),
                "clean_sha256": clean_hash,
                "pixel_linf_uint8": gap,
                "nonzero": gap > 0,
                "linf_ok": gap <= EPS + 1e-6,
                "output_valid": bool(row["output_valid"]),
                "full_exact": row["pred_full"] == spec["clean_full_pred"],
                "margin": float(row["margin"]),
            }
            table.append(record)
            if record["nonzero"] and record["linf_ok"] and record["output_valid"] and record["full_exact"]:
                legal.append({"step": step, "margin": record["margin"], "png": str(png)})
        pick = choose(legal)
        if pick is None:
            selected.append({"sample_id": sid, "status": "NO_FEASIBLE_NONZERO_CHECKPOINT", "step": "", "margin": ""})
        else:
            source = next(row for row in grouped[sid] if int(row["step"]) == pick["step"])
            selected.append({
                "sample_id": sid,
                "status": "SELECTED",
                "step": pick["step"],
                "margin": pick["margin"],
                "png": f"checkpoints/step_{pick['step']:03d}.png",
                "pred_full": source["pred_full"],
                "pred_fastv": source["pred_fastv"],
                "anls_full": source["anls_full"],
                "anls_fastv": source["anls_fastv"],
                "output_valid": source["output_valid"],
                "pixel_linf_uint8": next(row["pixel_linf_uint8"] for row in table if row["sample_id"] == sid and row["step"] == pick["step"]),
                "dev12": sid in DEV,
            })
    with (OUT / "candidate_table.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    with (OUT / "per_sample_selector.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    dev = [row for row in selected if row["sample_id"] in DEV]
    strict = 0
    for row in dev:
        if row["status"] != "SELECTED":
            continue
        if float(row["anls_full"]) >= 0.5 and float(row["anls_fastv"]) < 0.5 and str(row["output_valid"]) in ("True", "true", "1"):
            strict += 1
    (OUT / "selector_provenance.json").write_text(json.dumps({
        "rule": "pixel Linf, full exact, valid, margin then step; anls not used to rank",
        "dev12_selected": sum(row["status"] == "SELECTED" for row in dev),
        "dev12_strict_before_replay": strict,
        "all40_selected": sum(row["status"] == "SELECTED" for row in selected),
    }, indent=2) + "\n")
    print(json.dumps({"dev12_strict_before_replay": strict, "selected40": sum(row["status"] == "SELECTED" for row in selected)}))


if __name__ == "__main__":
    main()
