"""Small B/C construction for the recovery-path hypothesis.

B: content non-overlap on the clean FastV prefix.
C: the same loss over a bounded replay buffer of naturally observed
   recovery prefixes.  The buffer is updated only from accepted PNG states.

This runner is deliberately one-sample/one-ratio first.  It uses real FastV
generation and a real Full exact guard after every proposed update.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
V7D = ROOT / "week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation"
OUT = ROOT / "week2_attack_v15_recovery_path_probe"
sys.path[:0] = [str(ROOT), str(ROOT / "week1_mechanism"), str(ROOT / "gate_experiment"), str(V7D)]

from adapters.runtime import clear_hooks  # noqa: E402
from metrics_task import anls_score  # noqa: E402
from run_eval import decode  # noqa: E402
from week2_attack_v4_answer_aligned.run_smoke import page_uint8, project, uint8_linf  # noqa: E402
from week2_attack_v5b_expanded_pristine40.load_samples import load_formal_items, manifest as v5b_manifest  # noqa: E402
from week2_attack_v7_pristine60.run_v7 import load_model_frozen  # noqa: E402
from week2_attack_v7b_native_cache_margin_admission.native_cache_margin import hard_margin_row  # noqa: E402
from week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation.losses import decision_floor, decision_preservation  # noqa: E402
from week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation.native_grad import fastv_rows, full_rows  # noqa: E402
from week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation.run_v7d import (  # noqa: E402
    generate_tokens,
    full_token_rows,
    pixels_from_page,
    prepare,
    spans_of,
)
from week2_pilot.attack import page_to_float  # noqa: E402

EPS = 8 / 255
RATIO = 0.25
MAX_STATES = 3
OVERLAP_THRESHOLD = 0.6
LAMBDA_FULL = 2.6371882367154997
STEPS = 25
CHECKPOINTS = (0, 5, 10, 15, 20, 25)


def load_item(sample_id: str) -> dict:
    spec = next(row for row in v5b_manifest()["samples"] if row["sample_id"] == sample_id)
    return load_formal_items([spec])[0]


def sha_image(image: Image.Image) -> str:
    return hashlib.sha256(image.tobytes()).hexdigest()


def overlap(processor, clean_tokens: list[int], generated: list[int]) -> float:
    def words(tokens):
        result = set()
        for token in processor.tokenizer.convert_ids_to_tokens(tokens):
            value = str(token).replace("▁", "").replace("Ġ", "").lower()
            if any(char.isalnum() for char in value):
                result.add(value)
        return result
    a, b = words(clean_tokens), words(generated)
    return float(len(a & b) / max(1, len(a)))


def content_positions(processor, tokens: list[int]) -> list[int]:
    values = processor.tokenizer.convert_ids_to_tokens(tokens)
    out = []
    for index, value in enumerate(values):
        value = str(value).replace("▁", "").replace("Ġ", "")
        if any(char.isalnum() for char in value) and not value.startswith("<"):
            out.append(index)
    return out


def margin(row: torch.Tensor, token: int) -> torch.Tensor:
    target = row[int(token)]
    other = row.clone()
    other[int(token)] = -torch.inf
    return target - other.max()


def escape_loss(rows, clean_tokens, positions):
    use = [i for i in positions if i < len(rows) and i < len(clean_tokens)]
    if not use:
        return rows[0].sum() * 0.0
    return torch.relu(torch.stack([margin(rows[i], clean_tokens[i]) for i in use])).mean()


def state_loss(model, state, item, page, generated, kept, prefixes):
    inputs = state["inputs"]
    pixels = pixels_from_page(inputs, item["aux_idx"], page)
    keep = __import__("week2_attack_v4_answer_aligned.forward", fromlist=["keep_index_from_kept"]).keep_index_from_kept(
        kept, state["spans"], int(inputs["input_ids"].shape[1]), page.device
    )
    losses = []
    for prefix in prefixes:
        rows = fastv_rows(model, inputs, pixels, prefix, keep)
        losses.append(escape_loss(rows, state["clean_tokens"], state["positions"]))
        del rows
        torch.cuda.empty_cache()
        pixels = pixels_from_page(inputs, item["aux_idx"], page)
    return torch.stack(losses).max(), [float(x.detach()) for x in losses]


def full_fdp_loss(model, state, item, page):
    rows = full_token_rows(model, state["inputs"], page, item["aux_idx"], state["clean_tokens"], False)
    hard = torch.stack([hard_margin_row(row, token) for row, token in zip(rows, state["clean_tokens"])])
    return decision_preservation(hard, state["clean_hard"])


def observe(model, processor, item, state, page, out_path: Path) -> dict:
    inputs = state["inputs"]
    pixels = pixels_from_page(inputs, item["aux_idx"], page)
    clear_hooks(model)
    full_tokens, _ = generate_tokens(model, inputs, pixels, state["spans"], None)
    fast_tokens, kept = generate_tokens(model, inputs, pixels, state["spans"], RATIO)
    full_text = decode(processor, torch.tensor(full_tokens, device=page.device)).strip()
    fast_text = decode(processor, torch.tensor(fast_tokens, device=page.device)).strip()
    image = page_uint8(state["base_page"], page - state["base_page"])
    image.save(out_path)
    return {
        "full_text": full_text,
        "fastv_text": fast_text,
        "full_exact": int(full_text == state["clean_text"]),
        "fastv_anls_vs_clean": float(anls_score(fast_text, [state["clean_fast_text"]])),
        "fastv_tokens": fast_tokens,
        "kept": kept,
        "content_overlap": overlap(processor, state["clean_fast_tokens"], fast_tokens),
        "png_sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
        "pixel_hash": sha_image(image),
        "uint8_linf": uint8_linf(item["images"][item["aux_idx"]], image),
    }


def run(sample_id: str, method: str, steps: int) -> dict:
    torch.manual_seed(2026101301)
    np.random.seed(2026101301)
    model, processor = load_model_frozen()
    model.eval()
    item = load_item(sample_id)
    inputs = prepare(processor, item["images"], item["meta"]["question"])
    spans = spans_of(inputs, int(model.config.image_token_index), len(item["images"]))
    base_page = page_to_float(item["images"][item["aux_idx"]]).to("cuda")
    clear_hooks(model)
    clean_tokens, _ = generate_tokens(model, inputs, inputs["pixel_values"], spans, None)
    clean_fast_tokens, _ = generate_tokens(model, inputs, inputs["pixel_values"], spans, RATIO)
    clean_text = decode(processor, torch.tensor(clean_tokens, device=base_page.device)).strip()
    clean_fast_text = decode(processor, torch.tensor(clean_fast_tokens, device=base_page.device)).strip()
    with torch.no_grad():
        rows = full_token_rows(model, inputs, base_page, item["aux_idx"], clean_tokens, False)
        clean_hard = torch.stack([hard_margin_row(row, token) for row, token in zip(rows, clean_tokens)]).detach()
    state = {
        "inputs": inputs, "spans": spans, "base_page": base_page,
        "clean_tokens": clean_tokens, "clean_fast_tokens": clean_fast_tokens,
        "positions": content_positions(processor, clean_fast_tokens),
        "clean_text": clean_text, "clean_fast_text": clean_fast_text,
        "clean_hard": clean_hard, "floor": decision_floor(clean_hard).detach(),
    }
    delta = torch.zeros_like(base_page, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=2 / 255)
    prefixes = [list(clean_fast_tokens)]
    out = OUT / "construction" / sample_id / method
    out.mkdir(parents=True, exist_ok=True)
    clean_path = out / "clean.png"
    item["images"][item["aux_idx"]].save(clean_path)
    records = []
    for step in range(steps + 1):
        if step in CHECKPOINTS or step == steps:
            path = out / "checkpoints" / f"step_{step:03d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            page_detached = (base_page + delta.detach()).clamp(0, 1)
            observed = observe(model, processor, item, state, page_detached, path)
            observed.update({"step": step, "states": len(prefixes), "method": method})
            records.append(observed)
            print(json.dumps({"sample": sample_id, "method": method, "step": step, "full_exact": observed["full_exact"], "fastv": observed["fastv_text"], "states": len(prefixes)}, ensure_ascii=False), flush=True)
            if method == "C" and observed["content_overlap"] >= OVERLAP_THRESHOLD and observed["fastv_tokens"] not in prefixes and len(prefixes) < MAX_STATES:
                prefixes.append(list(observed["fastv_tokens"]))
        if step == steps:
            break
        opt.zero_grad(set_to_none=True)
        page = (base_page + delta).clamp(0, 1)
        detached_pixels = pixels_from_page(state["inputs"], item["aux_idx"], page.detach())
        current_tokens, kept = generate_tokens(model, state["inputs"], detached_pixels, state["spans"], RATIO)
        if method == "B":
            active_prefixes = [list(clean_fast_tokens)]
        else:
            active_prefixes = list(prefixes)
            if current_tokens not in active_prefixes and overlap(processor, clean_fast_tokens, current_tokens) >= OVERLAP_THRESHOLD and len(active_prefixes) < MAX_STATES:
                active_prefixes.append(list(current_tokens))
        # Accumulate the two gradients sequentially.  Keeping both the
        # FastV and Full graphs alive exceeds the 40 GiB card limit.
        l_escape, state_values = state_loss(model, state, item, page, current_tokens, kept, active_prefixes)
        l_escape.backward()
        if delta.grad is None or not torch.isfinite(delta.grad).all():
            raise RuntimeError("non-finite escape gradient")
        escape_grad = delta.grad.detach().clone()
        delta.grad = None
        del l_escape
        torch.cuda.empty_cache()
        page_fdp = (base_page + delta).clamp(0, 1)
        l_fdp = full_fdp_loss(model, state, item, page_fdp)
        l_fdp.backward()
        if delta.grad is None or not torch.isfinite(delta.grad).all():
            raise RuntimeError("non-finite FDP gradient")
        delta.grad.add_(LAMBDA_FULL * escape_grad)
        if delta.grad is None or not torch.isfinite(delta.grad).all():
            raise RuntimeError("non-finite delta gradient")
        old_delta = delta.detach().clone()
        old_opt = copy.deepcopy(opt.state_dict())
        opt.step()
        project(delta, base_page, EPS)
        candidate_page = (base_page + delta.detach()).clamp(0, 1)
        guard_path = out / "guards" / f"step_{step + 1:03d}.png"
        guard_path.parent.mkdir(parents=True, exist_ok=True)
        guard = observe(model, processor, item, state, candidate_page, guard_path)
        if not guard["full_exact"] or not (0 < guard["uint8_linf"] <= EPS + 1e-6):
            with torch.no_grad():
                delta.copy_(old_delta)
            opt.load_state_dict(old_opt)
        del page, page_fdp, detached_pixels, current_tokens, kept, l_fdp, escape_grad, state_values
        torch.cuda.empty_cache()
    payload = {"sample_id": sample_id, "method": method, "ratio": RATIO, "steps": records, "clean_text": clean_text, "clean_fast_text": clean_fast_text, "max_states": MAX_STATES, "overlap_threshold": OVERLAP_THRESHOLD}
    (out / "trajectory.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", default="7006")
    parser.add_argument("--method", choices=("B", "C"), required=True)
    parser.add_argument("--steps", type=int, default=STEPS)
    args = parser.parse_args()
    run(args.sample, args.method, args.steps)


if __name__ == "__main__":
    main()
