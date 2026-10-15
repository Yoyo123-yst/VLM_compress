"""Two-sample complete-answer score and one-step direction smoke.

This is not an attack run.  It validates the frozen target sequence, grouped
correct aliases, gradient sign, processor-anchored pixel mapping, and a single
quantized PNG step before any iterative optimization is permitted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "week2_attack_v25_full_answer_competition_pilot"
sys.path[:0] = [str(ROOT), str(ROOT / "week1_mechanism"), str(ROOT / "gate_experiment")]

from run_causal_v2 import prepare, spans_of
from run_eval import decode
from week2_attack_v13_branch_reachability_probe import run_branch_probe as probe
from week2_attack_v11_objective_optimizer_diagnostic import run_v11 as base
from week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation import run_v7d as v7d
from week2_attack_v25_full_answer_competition_pilot import generate_clean_candidates as candidates


ALPHA = 1.0 / 255.0


def sha_image(image) -> str:
    return hashlib.sha256(np.asarray(image.convert("RGB"), dtype=np.uint8).tobytes()).hexdigest()


def encode_answer(processor, text: str) -> list[int]:
    ids = [int(x) for x in processor.tokenizer.encode(text, add_special_tokens=False)]
    eos = int(processor.tokenizer.eos_token_id)
    if not ids or ids[-1] != eos:
        ids.append(eos)
    return ids


def dedupe(sequences: list[list[int]]) -> list[list[int]]:
    seen = set()
    out = []
    for sequence in sequences:
        key = tuple(int(x) for x in sequence)
        if key not in seen:
            seen.add(key)
            out.append(list(key))
    return out


def sequence_score_inputs(model, inputs: dict, pixels: torch.Tensor, tokens: list[int], keep_index: torch.Tensor) -> torch.Tensor:
    rows = v7d.fastv_rows(model, inputs, pixels, tokens, keep_index)
    values = [torch.log_softmax(row, dim=-1)[int(token)] for row, token in zip(rows, tokens)]
    return torch.stack(values).mean()


def sequence_score(model, state: dict, item: dict, page: torch.Tensor, tokens: list[int], keep_index: torch.Tensor) -> torch.Tensor:
    pixels = candidates.anchored_pixels_from_page(state, item, page)
    return sequence_score_inputs(model, state["inputs"], pixels, tokens, keep_index)


def score_value(model, state, item, page, tokens, keep_index) -> float:
    value = sequence_score(model, state, item, page, tokens, keep_index)
    result = float(value.detach().cpu())
    del value
    return result


def score_grad(model, state, item, tokens, keep_index) -> tuple[float, torch.Tensor]:
    leaf = state["page"].detach().clone().requires_grad_(True)
    value = sequence_score(model, state, item, leaf, tokens, keep_index)
    value.backward()
    if leaf.grad is None or not torch.isfinite(leaf.grad).all():
        raise RuntimeError("sequence score produced no finite auxiliary-page gradient")
    result = float(value.detach().cpu()), leaf.grad.detach().clone()
    del value, leaf
    torch.cuda.empty_cache()
    return result


def group_values(model, state, item, page, correct, wrong, keep_index) -> dict:
    correct_scores = [score_value(model, state, item, page, seq, keep_index) for seq in correct]
    wrong_score = score_value(model, state, item, page, wrong, keep_index)
    correct_lse = float(torch.logsumexp(torch.tensor(correct_scores, dtype=torch.float64), dim=0))
    return {
        "correct_scores": correct_scores,
        "correct_logsumexp": correct_lse,
        "wrong_score": wrong_score,
        "objective": correct_lse - wrong_score,
        "wrong_advantage_vs_best_correct": wrong_score - max(correct_scores),
    }


def group_values_inputs(model, inputs, pixels, correct, wrong, keep_index) -> dict:
    correct_scores = [
        float(sequence_score_inputs(model, inputs, pixels, seq, keep_index).detach().cpu())
        for seq in correct
    ]
    wrong_score = float(sequence_score_inputs(model, inputs, pixels, wrong, keep_index).detach().cpu())
    correct_lse = float(torch.logsumexp(torch.tensor(correct_scores, dtype=torch.float64), dim=0))
    return {
        "correct_scores": correct_scores,
        "correct_logsumexp": correct_lse,
        "wrong_score": wrong_score,
        "objective": correct_lse - wrong_score,
        "wrong_advantage_vs_best_correct": wrong_score - max(correct_scores),
    }


def objective_gradient(model, state, item, correct, wrong, keep_index) -> tuple[dict, torch.Tensor]:
    correct_results = [score_grad(model, state, item, seq, keep_index) for seq in correct]
    wrong_score, wrong_grad = score_grad(model, state, item, wrong, keep_index)
    correct_scores = [row[0] for row in correct_results]
    weights = torch.softmax(torch.tensor(correct_scores, dtype=torch.float64), dim=0).tolist()
    gradient = -wrong_grad
    for weight, (_score, grad) in zip(weights, correct_results):
        gradient = gradient + float(weight) * grad
    correct_lse = float(torch.logsumexp(torch.tensor(correct_scores, dtype=torch.float64), dim=0))
    facts = {
        "correct_scores": correct_scores,
        "correct_softmax_weights": weights,
        "correct_logsumexp": correct_lse,
        "wrong_score": wrong_score,
        "objective": correct_lse - wrong_score,
        "wrong_advantage_vs_best_correct": wrong_score - max(correct_scores),
        "gradient_l1": float(gradient.float().abs().sum()),
        "gradient_l2": float(gradient.float().norm()),
        "gradient_linf": float(gradient.float().abs().max()),
    }
    return facts, gradient


def natural_eval(model, processor, state, item, pixels, correct, wrong) -> dict:
    full, _ = v7d.generate_tokens(model, state["inputs"], pixels, state["spans"], None)
    fast, kept = v7d.generate_tokens(model, state["inputs"], pixels, state["spans"], probe.RATIO)
    keep_index = probe.keep_index_from_kept(
        kept, state["spans"], int(state["inputs"]["input_ids"].shape[1]), pixels.device
    )
    scores = group_values_inputs(model, state["inputs"], pixels, correct, wrong, keep_index)
    return {
        "full_tokens": full,
        "full_text": decode(processor, torch.tensor(full)).strip(),
        "full_exact": bool(full == state["clean_tokens"]),
        "fast_tokens": fast,
        "fast_text": decode(processor, torch.tensor(fast)).strip(),
        "keep_hash": probe.keep_hash(kept),
        "k": len(kept),
        "natural_keep_sequence_scores": scores,
    }


def png_eval(model, processor, item, image, state, correct, wrong) -> dict:
    images = list(item["images"])
    images[item["aux_idx"]] = image
    inputs = prepare(processor, images, item["meta"]["question"])
    spans = spans_of(inputs, int(model.config.image_token_index), len(images))
    full, _ = v7d.generate_tokens(model, inputs, inputs["pixel_values"], spans, None)
    fast, kept = v7d.generate_tokens(model, inputs, inputs["pixel_values"], spans, probe.RATIO)
    keep_index = probe.keep_index_from_kept(kept, spans, int(inputs["input_ids"].shape[1]), inputs["pixel_values"].device)
    scores = group_values_inputs(model, inputs, inputs["pixel_values"], correct, wrong, keep_index)
    return {
        "full_tokens": full,
        "full_text": decode(processor, torch.tensor(full)).strip(),
        "full_exact": bool(full == state["clean_tokens"]),
        "fast_tokens": fast,
        "fast_text": decode(processor, torch.tensor(fast)).strip(),
        "keep_hash": probe.keep_hash(kept),
        "k": len(kept),
        "natural_keep_sequence_scores": scores,
    }


def run(sample: str, device: str, out: Path) -> dict:
    probe.configure_determinism()
    torch.cuda.set_device(int(device))
    model, processor = base.load_model_frozen()
    model.eval()
    frozen = json.loads((HERE / "frozen_targets.json").read_text())["samples"][sample]
    item = v7d.load_dev(sample)
    state = v7d.prepare_item(model, processor, item)
    fast = probe.fast_clean_reference(model, processor, item, state)
    zero = candidates.zero_injection_audit(model, state, item, fast)
    if not all((zero["pixel_tensor_exact"], zero["fast_tokens_exact"], zero["fast_keep_exact"])):
        raise RuntimeError(f"zero-injection gate failed: {zero}")
    zero_pixels = candidates.anchored_pixels_from_page(state, item, state["page"])
    _tokens, kept = v7d.generate_tokens(model, state["inputs"], zero_pixels, state["spans"], probe.RATIO)
    keep_index = probe.keep_index_from_kept(
        kept, state["spans"], int(state["inputs"]["input_ids"].shape[1]), state["page"].device
    )
    correct = dedupe([[int(x) for x in fast["tokens"]]] + [encode_answer(processor, x) for x in frozen["correct_aliases"]])
    wrong = [int(x) for x in frozen["wrong_target_tokens"]]
    decoded_wrong = decode(processor, torch.tensor(wrong)).strip()
    if decoded_wrong != frozen["wrong_target_text"]:
        raise RuntimeError(f"frozen wrong target decode mismatch: {decoded_wrong!r}")
    before, gradient = objective_gradient(model, state, item, correct, wrong, keep_index)
    finite_nonzero = bool(torch.isfinite(gradient).all() and float(gradient.abs().sum()) > 0)
    direction = -gradient.sign()
    with torch.no_grad():
        step_page = (state["page"] + ALPHA * direction).clamp(0, 1)
    after = group_values(model, state, item, step_page, correct, wrong, keep_index)
    step_pixels = candidates.anchored_pixels_from_page(state, item, step_page).detach()
    anchored_natural = natural_eval(model, processor, state, item, step_pixels, correct, wrong)
    delta = step_page - state["page"]
    image = probe.page_uint8(state["page"], delta)
    out.parent.mkdir(parents=True, exist_ok=True)
    png_path = out.with_suffix(".png")
    image.save(png_path, format="PNG")
    replay = png_eval(model, processor, item, image, state, correct, wrong)
    original = np.asarray(item["images"][item["aux_idx"]].convert("RGB"), dtype=np.int16)
    actual = np.asarray(image.convert("RGB"), dtype=np.int16)
    uint8_linf = int(np.abs(actual - original).max())
    predicted_objective = before["objective"] - ALPHA * before["gradient_l1"]
    pass_gate = bool(
        finite_nonzero
        and after["objective"] < before["objective"]
        and after["wrong_advantage_vs_best_correct"] > before["wrong_advantage_vs_best_correct"]
        and anchored_natural["natural_keep_sequence_scores"]["objective"] < before["objective"]
        and replay["natural_keep_sequence_scores"]["objective"] < before["objective"]
        and replay["natural_keep_sequence_scores"]["wrong_advantage_vs_best_correct"] > before["wrong_advantage_vs_best_correct"]
        and anchored_natural["full_exact"] and replay["full_exact"]
        and uint8_linf <= 1
    )
    result = {
        "sample_id": sample,
        "status": "PASS_SCORE_DIRECTION_SMOKE" if pass_gate else "FAIL_SCORE_DIRECTION_SMOKE",
        "development_only": True,
        "ratio": probe.RATIO,
        "alpha": ALPHA,
        "zero_injection_audit": zero,
        "clean_fast": fast,
        "correct_sequences": [{"tokens": x, "text": decode(processor, torch.tensor(x)).strip()} for x in correct],
        "wrong_sequence": {"tokens": wrong, "text": decoded_wrong},
        "before": before,
        "predicted_objective_after_sign_step": predicted_objective,
        "after_anchored_fixed_keep": after,
        "finite_nonzero_gradient": finite_nonzero,
        "anchored_natural_after_step": anchored_natural,
        "png_replay_after_step": replay,
        "png_path": str(png_path),
        "png_rgb_sha256": sha_image(image),
        "uint8_linf": uint8_linf,
        "pass_gate": pass_gate,
    }
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", choices=("47450", "7006"), required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = run(args.sample, args.device, Path(args.out))
    print(json.dumps({
        "sample": args.sample,
        "status": result["status"],
        "before": result["before"]["objective"],
        "after": result["after_anchored_fixed_keep"]["objective"],
        "uint8_linf": result["uint8_linf"],
        "out": args.out,
    }), flush=True)


if __name__ == "__main__":
    main()
