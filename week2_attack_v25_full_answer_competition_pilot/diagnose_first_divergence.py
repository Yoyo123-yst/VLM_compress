"""Read-only first-divergence diagnostic for the valid iterative_v2 PNGs."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "week2_attack_v25_full_answer_competition_pilot"
sys.path[:0] = [str(ROOT), str(ROOT / "week1_mechanism"), str(ROOT / "gate_experiment")]

from run_causal_v2 import prepare, spans_of  # noqa: E402
from run_eval import decode  # noqa: E402
from week2_attack_v11_objective_optimizer_diagnostic import run_v11 as base  # noqa: E402
from week2_attack_v13_branch_reachability_probe import run_branch_probe as probe  # noqa: E402
from week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation import run_v7d as v7d  # noqa: E402
from week2_attack_v25_full_answer_competition_pilot.score_smoke import dedupe, encode_answer  # noqa: E402

RATIO = 0.25
STEPS = (0, 1, 3, 5, 10)


def sha_image(image) -> str:
    return hashlib.sha256(np.asarray(image.convert("RGB"), dtype=np.uint8).tobytes()).hexdigest()


def score_rows(model, inputs, pixels, token_ids, kept, spans):
    keep_index = probe.keep_index_from_kept(
        kept, spans, int(inputs["input_ids"].shape[1]), pixels.device
    )
    rows = v7d.fastv_rows(model, inputs, pixels, token_ids, keep_index)
    return rows


def token_record(rows, correct, wrong):
    records = []
    for pos, row in enumerate(rows):
        corr = int(correct[pos]) if pos < len(correct) else None
        bad = int(wrong[pos]) if pos < len(wrong) else None
        logits = torch.log_softmax(row.detach().float(), dim=-1)
        excluded = {x for x in (corr, bad) if x is not None}
        mask = torch.ones_like(logits, dtype=torch.bool)
        for token in excluded:
            mask[token] = False
        competitor = int(torch.argmax(logits.masked_fill(~mask, -torch.inf)))
        records.append({
            "position": pos,
            "correct_token": corr,
            "wrong_token": bad,
            "correct_logprob": None if corr is None else float(logits[corr]),
            "wrong_logprob": None if bad is None else float(logits[bad]),
            "strongest_competitor_token": competitor,
            "strongest_competitor_logprob": float(logits[competitor]),
            "wrong_minus_correct": None if corr is None or bad is None else float(logits[bad] - logits[corr]),
            "wrong_minus_competitor": None if bad is None else float(logits[bad] - logits[competitor]),
        })
    return records


def evaluate_image(model, processor, item, state, image, correct, wrong):
    images = list(item["images"])
    images[item["aux_idx"]] = image
    inputs = prepare(processor, images, item["meta"]["question"])
    spans = spans_of(inputs, int(model.config.image_token_index), len(images))
    fast, kept = v7d.generate_tokens(model, inputs, inputs["pixel_values"], spans, RATIO)
    rows_correct = score_rows(model, inputs, inputs["pixel_values"], correct, kept, spans)
    rows_wrong = score_rows(model, inputs, inputs["pixel_values"], wrong, kept, spans)
    # Both teacher-forced sequences are scored on the same naturally selected
    # keep set. This is the relevant comparison for the PNG endpoint.
    corr_records = token_record(rows_correct, correct, wrong)
    wrong_records = token_record(rows_wrong, correct, wrong)
    first = next((i for i, (a, b) in enumerate(zip(correct, wrong)) if a != b), min(len(correct), len(wrong)))
    tail = [row for row in corr_records if row["position"] > first and row["wrong_minus_correct"] is not None]
    return {
        "png_sha256": sha_image(image),
        "fast_tokens": [int(x) for x in fast],
        "fast_text": decode(processor, torch.tensor(fast)).strip(),
        "keep_hash": probe.keep_hash(kept),
        "k": len(kept),
        "first_divergence_position": first,
        "first_divergence": corr_records[first] if first < len(corr_records) else None,
        "all_positions": corr_records,
        "wrong_teacher_forced_positions": wrong_records,
        "tail_wrong_minus_correct_mean": None if not tail else float(np.mean([x["wrong_minus_correct"] for x in tail])),
        "tail_wrong_minus_competitor_mean": None if not tail else float(np.mean([x["wrong_minus_competitor"] for x in tail])),
    }


def run(sample: str, device: str, out: Path):
    probe.configure_determinism()
    torch.cuda.set_device(int(device))
    model, processor = base.load_model_frozen()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    item = v7d.load_dev(sample)
    state = v7d.prepare_item(model, processor, item)
    frozen = json.loads((HERE / "frozen_targets.json").read_text())["samples"][sample]
    correct = dedupe([[int(x) for x in state["clean_tokens"]] + []] + [encode_answer(processor, x) for x in frozen["correct_aliases"]])
    # The first registered alias is the clean sequence for this diagnostic.
    correct_seq = [int(x) for x in state["clean_tokens"]]
    wrong = [int(x) for x in frozen["wrong_target_tokens"]]
    if decode(processor, torch.tensor(wrong)).strip() != frozen["wrong_target_text"]:
        raise RuntimeError("wrong target decode mismatch")
    original = item["images"][item["aux_idx"]].convert("RGB")
    results = {}
    root = Path(args.root) if args.root else HERE / "iterative_v2"
    methods = [
        method for method in ("native_fdp", "complete_target", "complete_front", "alias_group")
        if (root / method / sample).is_dir()
        and all((root / method / sample / f"accepted_{step:03d}.png").is_file() for step in STEPS if step)
    ]
    if not methods:
        raise RuntimeError(f"NO_COMPLETE_METHOD_TRAJECTORY: root={root} sample={sample}")
    for method in methods:
        method_dir = root / method / sample
        method_result = {}
        for step in STEPS:
            image = original if step == 0 else Image.open(method_dir / f"accepted_{step:03d}.png").convert("RGB")
            method_result[str(step)] = evaluate_image(model, processor, item, state, image, correct_seq, wrong)
        results[method] = method_result
    payload = {
        "status": "DONE",
        "sample_id": sample,
        "ratio": RATIO,
        "steps": STEPS,
        "clean_fast_text": decode(processor, torch.tensor(state["clean_tokens"])).strip(),
        "correct_tokens": correct_seq,
        "wrong_target_text": frozen["wrong_target_text"],
        "wrong_target_tokens": wrong,
        "methods": results,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"sample": sample, "status": payload["status"], "out": str(out)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True, choices=("47450", "7006", "60700"))
    parser.add_argument("--device", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--root", default=None)
    args = parser.parse_args()
    run(args.sample, args.device, Path(args.out))
