"""Engineering smoke on frozen dev sample 50743 only (does not touch v4 FROZEN.json)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
sys.path.insert(0, str(ROOT))

import week2_attack_v4_answer_aligned.run_smoke as smoke  # noqa: E402
from week2_attack_v4_answer_aligned.common import load_frozen_samples  # noqa: E402
from week2_attack_v4_answer_aligned.run_smoke import Session, attack, checkpoint_steps  # noqa: E402
from gate_experiment.run_causal_v2 import generate, load_model, prepare  # noqa: E402
from adapters.runtime import clear_hooks  # noqa: E402
from week2_attack_v5_pristine40.common import LAMBDA_FULL, V5, load_scales, write_json  # noqa: E402

DEV_ID = "50743"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    smoke.LAMBDA_FULL = LAMBDA_FULL
    out = V5 / "tests" / f"smoke_{args.steps}"
    out.mkdir(parents=True, exist_ok=True)
    scales = load_scales()
    if abs(float(scales.get("lambda_full", 1.0)) - LAMBDA_FULL) > 1e-6:
        scales["lambda_full"] = LAMBDA_FULL
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = load_model()
    model.to(device)
    for param in model.parameters():
        param.requires_grad_(False)
    item = next(s for s in load_frozen_samples() if s["meta"]["sample_id"] == DEV_ID)
    clear_hooks(model)
    inputs = prepare(processor, item["images"], item["meta"]["question"])
    with torch.inference_mode():
        raw = generate(model, inputs)
    answer = torch.tensor(raw.detach().cpu().tolist(), dtype=torch.long)
    session = Session(model, processor, item, answer)
    curve = out / "checkpoints.jsonl"
    artifact = out / DEV_ID
    row, seconds = attack(session, args.steps, scales, None, curve, artifact)
    write_json(out / "smoke_result.json", {
        "sample_id": DEV_ID,
        "steps": args.steps,
        "device": device,
        "checkpoints": checkpoint_steps(args.steps),
        "row": row,
        "seconds": seconds,
        "scales": scales,
    })
    print(json.dumps({"smoke": DEV_ID, "steps": args.steps, "device": device, "seconds": seconds}))


if __name__ == "__main__":
    main()
