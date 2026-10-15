"""Replay the variant A checkpoint PNGs that are not the already-replayed step 200."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
V5B = ROOT / "week2_attack_v5b_expanded_pristine40"
OUT = ROOT / "week2_attack_v6_survivor_upgrade"
sys.path.insert(0, str(ROOT))

import week2_attack_v4_answer_aligned.run_smoke as smoke  # noqa: E402
from adapters.runtime import clear_hooks  # noqa: E402
from gate_experiment.run_causal_v2 import generate, load_model, prepare  # noqa: E402
from week2_attack_v4_answer_aligned.run_smoke import Session, evaluate_uint8  # noqa: E402
from week2_attack_v5b_expanded_pristine40.load_samples import load_formal_items, manifest  # noqa: E402


def main():
    smoke.LAMBDA_FULL = 0.25
    chosen = {row["sample_id"]: row for row in csv.DictReader((OUT / "variant_a_selector.csv").open())}
    pending = [row for row in chosen.values() if row["step"] not in ("", "200")]
    specs = [row for row in manifest()["samples"] if row["sample_id"] in {item["sample_id"] for item in pending}]
    items = {item["meta"]["sample_id"]: item for item in load_formal_items(specs)}
    model, processor = load_model()
    for param in model.parameters():
        param.requires_grad_(False)
    results = []
    for row in pending:
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
        results.append({"sample_id": sid, "step": row["step"], "ok": ok, "pred_full": measured["pred_full"], "pred_fastv": measured["pred_fastv"]})
        print(json.dumps(results[-1]), flush=True)
        del session
        torch.cuda.empty_cache()
    (OUT / "variant_a_replay.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
