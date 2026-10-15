"""Full-sequence differentiable FastV admission on the frozen v7b conditions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
HERE = ROOT / "week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation"
sys.path[:0] = [str(ROOT), str(ROOT / "week1_mechanism"), str(ROOT / "gate_experiment")]

from run_causal_v2 import prepare, spans_of  # noqa: E402
from run_eval import decode  # noqa: E402
from week2_attack_v4_answer_aligned.common import keep_hash, sha_image  # noqa: E402
from week2_attack_v4_answer_aligned.forward import keep_index_from_kept  # noqa: E402
from week2_attack_v7_pristine60.load_samples import load_items, manifest  # noqa: E402
from week2_attack_v7_pristine60.run_v7 import load_model_frozen  # noqa: E402
from week2_attack_v7b_native_cache_margin_admission.native_cache_margin import (  # noqa: E402
    hard_margin_row,
    soft_margin_row,
    termination_of,
)
from week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation.losses import (  # noqa: E402
    conditional_gap,
    decision_preservation,
)
from week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation.native_grad import (  # noqa: E402
    TRAIN_RATIOS,
    fastv_rows,
    full_rows,
    mean_soft,
)
from week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation.run_v7d import (  # noqa: E402
    evidence_index,
    full_token_rows,
    generate_tokens,
    pixels_from_page,
)
from week2_pilot.attack import page_to_float  # noqa: E402

RATIO = 0.25
EOS = 151645


def catalog() -> list[dict]:
    protocol = json.loads((ROOT / "week2_attack_v7b_native_cache_margin_admission/FROZEN_PROTOCOL.json").read_text())
    return protocol["conditions"]


def load_images_for(spec: dict) -> tuple[dict, list]:
    row = next(item for item in manifest()["samples"] if item["sample_id"] == spec["sample_id"])
    item = load_items([row])[0]
    images = list(item["images"])
    if spec["image_kind"] == "clean":
        if sha_image(images[item["aux_idx"]]) != spec["sha256"]:
            raise RuntimeError(f"clean aux hash mismatch {spec['sample_id']}")
    else:
        path = ROOT / spec["source"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != spec["sha256"]:
            raise RuntimeError(f"png hash mismatch {spec['sample_id']} {spec['image_kind']}")
        images[item["aux_idx"]] = Image.open(path).convert("RGB")
    return item, images


def text_of(processor, tokens: list[int]) -> str:
    return decode(processor, torch.tensor(tokens)).strip()


def backward_norm(loss: torch.Tensor, page: torch.Tensor) -> float:
    if page.grad is not None:
        page.grad = None
    loss.backward()
    if page.grad is None or not torch.isfinite(page.grad).all():
        value = float("nan")
    else:
        value = float(page.grad.float().norm())
    page.grad = None
    torch.cuda.empty_cache()
    return value


def run_one(model, processor, spec: dict, index: int) -> dict:
    item, images = load_images_for(spec)
    inputs = prepare(processor, images, item["meta"]["question"])
    spans = spans_of(inputs, int(model.config.image_token_index), len(images))
    aux = item["aux_idx"]
    leaf = inputs["pixel_values"][aux].detach().clone().requires_grad_(True)
    full_tokens, _ = generate_tokens(model, inputs, inputs["pixel_values"], spans, None)
    with torch.no_grad():
        full_now = full_rows(model, inputs, inputs["pixel_values"], full_tokens)
    fast_tokens, kept = generate_tokens(model, inputs, inputs["pixel_values"], spans, RATIO)
    keep = keep_index_from_kept(kept, spans, int(inputs["input_ids"].shape[1]), inputs["input_ids"].device)
    trace: list[dict] = []
    with torch.no_grad():
        fast_now = fastv_rows(model, inputs, inputs["pixel_values"], fast_tokens, keep, trace)
    def fresh_pixels() -> torch.Tensor:
        pages = [inputs["pixel_values"][i].detach() for i in range(inputs["pixel_values"].shape[0])]
        pages[aux] = leaf
        return torch.stack(pages)

    def checkpointed_full(tokens: list[int]) -> torch.Tensor:
        current = fresh_pixels()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
        try:
            rows = full_rows(model, inputs, current, tokens)
        finally:
            model.eval()
            model.gradient_checkpointing_disable()
        if [int(row.argmax()) for row in rows] != tokens:
            raise RuntimeError("Full differentiable argmax left the generate tokens")
        return mean_soft(rows, tokens)

    def fast_mean(ratio: float) -> torch.Tensor:
        if ratio not in TRAIN_RATIOS:
            raise RuntimeError(ratio)
        current = fresh_pixels()
        tokens, kept_now = generate_tokens(model, inputs, current.detach(), spans, ratio)
        keep_now = keep_index_from_kept(kept_now, spans, int(inputs["input_ids"].shape[1]), leaf.device)
        rows = fastv_rows(model, inputs, current, tokens, keep_now)
        if [int(row.argmax()) for row in rows] != tokens:
            raise RuntimeError("FastV argmax left generate during the gradient")
        return mean_soft(rows, tokens)

    grad_compressed = backward_norm(fast_mean(0.25), leaf)
    grad_compressed_35 = backward_norm(fast_mean(0.35), leaf)
    grad_full = backward_norm(checkpointed_full(full_tokens), leaf)
    clean_images = load_items([next(row for row in manifest()["samples"] if row["sample_id"] == spec["sample_id"])])[0]["images"]
    clean_inputs = prepare(processor, clean_images, item["meta"]["question"])
    clean_tokens, _ = generate_tokens(model, clean_inputs, clean_inputs["pixel_values"], spans, None)
    with torch.no_grad():
        clean_ref = full_rows(model, clean_inputs, clean_inputs["pixel_values"], clean_tokens)
        clean_hard = torch.stack([hard_margin_row(row, token) for row, token in zip(clean_ref, clean_tokens)]).detach()
    current = fresh_pixels()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    try:
        attack_rows = full_rows(model, inputs, current, clean_tokens)
    finally:
        model.eval()
        model.gradient_checkpointing_disable()
    hard = torch.stack([hard_margin_row(row, token) for row, token in zip(attack_rows, clean_tokens)])
    grad_fdp = backward_norm(decision_preservation(hard, clean_hard), leaf)
    gap_bucket = torch.zeros_like(leaf)
    def accumulate(loss, scale):
        if leaf.grad is not None:
            leaf.grad = None
        (loss * scale).backward()
        gap_bucket.add_(leaf.grad.detach())
        leaf.grad = None
        torch.cuda.empty_cache()
    accumulate(checkpointed_full(full_tokens), -1.0)
    accumulate(fast_mean(0.25), 0.5)
    accumulate(fast_mean(0.35), 0.5)
    grad_gap = float(gap_bucket.float().norm())
    token_norms = []
    for step, token in enumerate(fast_tokens):
        rows = fastv_rows(model, inputs, fresh_pixels(), fast_tokens, keep)
        token_norms.append(backward_norm(soft_margin_row(rows[step], token), leaf))
    evidence_pixels = fresh_pixels()
    evidence_grad = torch.autograd.grad(evidence_pixels[evidence_index(item)].float().sum(), leaf, allow_unused=True)[0]
    evidence_l2 = 0.0 if evidence_grad is None else float(evidence_grad.float().norm())
    appended_eos = len(fast_tokens) < 16 and (not fast_tokens or fast_tokens[-1] != EOS) and False
    report = {
        "index": index,
        "sample_id": spec["sample_id"],
        "image_kind": spec["image_kind"],
        "ratio": RATIO,
        "full_argmax_match": [int(row.argmax()) for row in full_now] == full_tokens,
        "full_text": text_of(processor, full_tokens),
        "fast_argmax_match": [int(row.argmax()) for row in fast_now] == fast_tokens,
        "fast_text": text_of(processor, fast_tokens),
        "k": len(kept),
        "keep_hash": keep_hash(kept),
        "termination": termination_of(fast_tokens, EOS),
        "appended_eos": appended_eos,
        "steps": [
            {
                **trace[step],
                "token_id": int(token),
                "argmax": int(fast_now[step].argmax()),
                "soft_margin": float(soft_margin_row(fast_now[step], token)),
            }
            for step, token in enumerate(fast_tokens)
        ],
        "grad_l2": {
            "full_soft": grad_full,
            "fast_0.25": grad_compressed,
            "fast_0.35": grad_compressed_35,
            "fdp": grad_fdp,
            "conditional_gap": grad_gap,
        },
        "per_token_aux_grad_l2": token_norms,
        "evidence_grad_l2": evidence_l2,
    }
    finite = all(math.isfinite(value) and value > 0 for value in report["grad_l2"].values())
    later = token_norms[1:] or token_norms
    report["pass"] = bool(
        report["full_argmax_match"]
        and report["fast_argmax_match"]
        and finite
        and evidence_l2 == 0.0
        and all(math.isfinite(value) and value > 0 for value in later)
        and not report["appended_eos"]
    )
    dest = HERE / "admission" / f"cond_{index:02d}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"index": index, "id": spec["sample_id"], "kind": spec["image_kind"], "pass": report["pass"]}), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indices", required=True)
    args = parser.parse_args()
    conditions = catalog()
    model, processor = load_model_frozen()
    model.eval()
    for raw in args.indices.split(","):
        index = int(raw)
        run_one(model, processor, conditions[index], index)


if __name__ == "__main__":
    main()
