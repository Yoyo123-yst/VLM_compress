"""Variant A: prospective feasible selector on frozen v5b checkpoints. No new PGD."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

V5B = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project/week2_attack_v5b_expanded_pristine40")
OUT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project/week2_attack_v6_survivor_upgrade")
EPS = 8 / 255


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = {row["sample_id"]: row for row in json.loads((V5B / "frozen_manifest.json").read_text())["samples"]}
    grouped = {}
    for path in sorted((V5B / "attacks").glob("shard_s*/checkpoints.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            grouped.setdefault(row["sample_id"], []).append(row)
    chosen = []
    for sid, spec in manifest.items():
        rows = grouped[sid]
        art = V5B / "attacks" / "artifacts" / sid
        clean_hash = sha(art / "clean.png")
        feasible = []
        for row in rows:
            step = int(row["step"])
            if step <= 0:
                continue
            png = art / "checkpoints" / f"step_{step:03d}.png"
            if not png.exists():
                continue
            nonzero = sha(png) != clean_hash
            text_ok = row["pred_full"] == spec["clean_full_pred"]
            valid = bool(row["output_valid"])
            linf_ok = float(row["linf"]) <= EPS + 1e-6
            if nonzero and text_ok and valid and linf_ok:
                feasible.append(row)
        if not feasible:
            chosen.append({
                "sample_id": sid,
                "status": "NO_FEASIBLE_NONZERO_CHECKPOINT",
                "step": "",
                "margin": "",
                "pred_full": "",
                "pred_fastv": "",
            })
            continue
        best = sorted(feasible, key=lambda row: (float(row["margin"]), int(row["step"])))[0]
        chosen.append({
            "sample_id": sid,
            "status": "FEASIBLE",
            "step": int(best["step"]),
            "margin": best["margin"],
            "pred_full": best["pred_full"],
            "pred_fastv": best["pred_fastv"],
            "anls_full": best["anls_full"],
            "anls_fastv": best["anls_fastv"],
            "output_valid": best["output_valid"],
            "linf": best["linf"],
            "png": f"checkpoints/step_{int(best['step']):03d}.png",
        })
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "variant_a_selector.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(chosen[0]))
        writer.writeheader()
        writer.writerows(chosen)
    feasible_n = sum(row["status"] == "FEASIBLE" for row in chosen)
    print(json.dumps({"feasible": feasible_n, "none": 40 - feasible_n}))


if __name__ == "__main__":
    main()
