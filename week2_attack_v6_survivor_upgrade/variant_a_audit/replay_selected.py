"""Disk-replay the already chosen Variant A PNG. Does not pick a replacement."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
V5B = ROOT / "week2_attack_v5b_expanded_pristine40"
OUT = ROOT / "week2_attack_v6_survivor_upgrade" / "variant_a_audit"
sys.path.insert(0, str(ROOT))

import week2_attack_v4_answer_aligned.run_smoke as smoke  # noqa: E402
from adapters.runtime import clear_hooks  # noqa: E402
from gate_experiment.run_causal_v2 import generate, load_model, prepare  # noqa: E402
from week2_attack_v4_answer_aligned.run_smoke import Session, evaluate_uint8  # noqa: E402
from week2_attack_v5b_expanded_pristine40.load_samples import load_formal_items, manifest  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--nshards", type=int, default=4)
    args = parser.parse_args()
    smoke.LAMBDA_FULL = 0.25
    chosen = list(csv.DictReader((OUT / "per_sample_selector.csv").open()))
    chosen = [row for i, row in enumerate(chosen) if i % args.nshards == args.shard and row["status"] == "SELECTED"]
    wanted = {row["sample_id"] for row in chosen}
    items = {item["meta"]["sample_id"]: item for item in load_formal_items(
        [row for row in manifest()["samples"] if row["sample_id"] in wanted]
    )}
    model, processor = load_model()
    for param in model.parameters():
        param.requires_grad_(False)
    results = []
    for row in chosen:
        sid = row["sample_id"]
        item = items[sid]
        clear_hooks(model)
        inputs = prepare(processor, item["images"], item["meta"]["question"])
        with torch.inference_mode():
            raw = generate(model, inputs)
        answer = torch.tensor(raw.detach().cpu().tolist(), dtype=torch.long)
        session = Session(model, processor, item, answer)
        image = Image.open(V5B / "attacks" / "artifacts" / sid / row["png"]).convert("RGB")
        measured = evaluate_uint8(session, image, 0.0)
        ok = measured["pred_full"] == row["pred_full"] and measured["pred_fastv"] == row["pred_fastv"]
        results.append({
            "sample_id": sid,
            "step": row["step"],
            "ok": ok,
            "status": "REPLAY_OK" if ok else "SELECTED_CANDIDATE_REPLAY_FAILED",
            "pred_full": measured["pred_full"],
            "pred_fastv": measured["pred_fastv"],
        })
        print(json.dumps({"replay": sid, "ok": ok}), flush=True)
        del session
        torch.cuda.empty_cache()
    dest = OUT / f"replay_s{args.shard}.json"
    dest.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
