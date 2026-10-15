"""Diagnose clean-prefix versus natural-prefix recovery on frozen PNGs.

This is intentionally separate from the historical runner. It never updates
pixels and never selects an attack checkpoint using ANLS.  It measures a
content-only escape proxy on the clean FastV token positions and on the
currently generated FastV prefixes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
V7D = ROOT / "week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation"
OUT = ROOT / "week2_attack_v15_recovery_path_probe"
sys.path[:0] = [str(ROOT), str(ROOT / "week1_mechanism"), str(ROOT / "gate_experiment"), str(V7D)]

from adapters.runtime import clear_hooks, install  # noqa: E402
from metrics_task import anls_score  # noqa: E402
from run_eval import decode  # noqa: E402
from week2_attack_v4_answer_aligned.common import keep_hash  # noqa: E402
from week2_attack_v4_answer_aligned.forward import keep_index_from_kept  # noqa: E402
from week2_attack_v5b_expanded_pristine40.load_samples import load_formal_items, manifest as v5b_manifest  # noqa: E402
from week2_attack_v7_pristine60.run_v7 import load_model_frozen  # noqa: E402
from week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation.native_grad import fastv_rows  # noqa: E402
from week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation.run_v7d import (  # noqa: E402
    generate_tokens,
    pixels_from_page,
    prepare,
)
from week2_pilot.attack import page_to_float  # noqa: E402

DEFAULT_RATIO = 0.25
SAMPLES = ("7006", "39204", "60700")
STEPS = (0, 10, 25, 50, 75, 100, 150, 200)
TAU = 0.0
ALNUM = re.compile(r"[A-Za-z0-9]")


def load_item(sample_id: str) -> dict:
    spec = next(row for row in v5b_manifest()["samples"] if row["sample_id"] == sample_id)
    return load_formal_items([spec])[0]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def content_positions(processor, tokens: list[int]) -> list[int]:
    values = processor.tokenizer.convert_ids_to_tokens(tokens)
    positions = []
    for index, value in enumerate(values):
        text = str(value).replace("▁", "").replace("Ġ", "")
        if ALNUM.search(text) and not text.startswith("<"):
            positions.append(index)
    return positions


def row_margin(row: torch.Tensor, token: int) -> torch.Tensor:
    target = row[int(token)]
    other = row.clone()
    other[int(token)] = -torch.inf
    return target - other.max()


def escape_loss(rows: list[torch.Tensor], reference: list[int], positions: list[int]) -> torch.Tensor:
    usable = [index for index in positions if index < len(rows) and index < len(reference)]
    if not usable:
        return rows[0].sum() * 0.0
    margins = torch.stack([row_margin(rows[index], reference[index]) for index in usable])
    return torch.relu(TAU + margins).mean()


def overlap(processor, clean_tokens: list[int], generated: list[int]) -> float:
    clean = set()
    for token in processor.tokenizer.convert_ids_to_tokens(clean_tokens):
        value = str(token).replace("▁", "").replace("Ġ", "").lower()
        if ALNUM.search(value):
            clean.add(value)
    current = set()
    for token in processor.tokenizer.convert_ids_to_tokens(generated):
        value = str(token).replace("▁", "").replace("Ġ", "").lower()
        if ALNUM.search(value):
            current.add(value)
    return float(len(clean & current) / max(1, len(clean)))


def score_state(model, processor, item, state, page: torch.Tensor, png_path: Path, ratio: float) -> dict:
    inputs = state["inputs"]
    pixels = pixels_from_page(inputs, item["aux_idx"], page)
    spans = state["spans"]
    clear_hooks(model)
    full_tokens, _ = generate_tokens(model, inputs, pixels, spans, None)
    full_text = decode(processor, torch.tensor(full_tokens, device=page.device)).strip()
    generated, kept = generate_tokens(model, inputs, pixels, spans, ratio)
    fast_text = decode(processor, torch.tensor(generated, device=page.device)).strip()
    keep = keep_index_from_kept(kept, spans, int(inputs["input_ids"].shape[1]), page.device)
    positions = state["content_positions"]
    # Never retain both long FastV graphs.  A single 7B backward graph is
    # already close to the 40 GiB card limit in this runtime.
    clean_rows = fastv_rows(model, inputs, pixels, state["clean_fast_tokens"], keep)
    clean_loss = escape_loss(clean_rows, state["clean_fast_tokens"], positions)
    clean_value = float(clean_loss.detach())
    grad_clean = torch.autograd.grad(clean_loss, page, allow_unused=True)[0]
    del clean_rows, clean_loss
    torch.cuda.empty_cache()
    # Rebuild the pixel graph after the first backward; the feature graph is
    # shared by both row computations and cannot be reused after autograd.grad.
    pixels = pixels_from_page(inputs, item["aux_idx"], page)
    natural_rows = fastv_rows(model, inputs, pixels, generated, keep)
    natural_loss = escape_loss(natural_rows, state["clean_fast_tokens"], positions)
    natural_value = float(natural_loss.detach())
    grad_natural = torch.autograd.grad(natural_loss, page, allow_unused=True)[0]
    del natural_rows, natural_loss
    torch.cuda.empty_cache()
    if grad_clean is None:
        grad_clean = torch.zeros_like(page)
    if grad_natural is None:
        grad_natural = torch.zeros_like(page)
    flat_clean = grad_clean.float().flatten()
    flat_natural = grad_natural.float().flatten()
    denom = float(flat_clean.norm() * flat_natural.norm())
    cosine = float(torch.dot(flat_clean, flat_natural) / denom) if denom > 0 else float("nan")
    return {
        "step": int(png_path.stem.split("_")[-1]),
        "png_sha256": sha(png_path),
        "full_text": full_text,
        "fastv_text": fast_text,
        "full_exact": int(full_text == state["clean_text"]),
        "fastv_anls_vs_clean": float(anls_score(fast_text, [state["clean_fast_text"]])),
        "keep_k": len(kept),
        "keep_hash": keep_hash(kept),
        "clean_prefix_escape_loss": clean_value,
        "natural_prefix_escape_loss": natural_value,
        "prefix_loss_gap_natural_minus_clean": natural_value - clean_value,
        "content_overlap_vs_clean": overlap(processor, state["clean_fast_tokens"], generated),
        "generated_token_count": len(generated),
        "clean_token_count": len(state["clean_fast_tokens"]),
        "gradient_cosine_clean_vs_natural": cosine,
        "gradient_l2_clean": float(flat_clean.norm()),
        "gradient_l2_natural": float(flat_natural.norm()),
    }


def run(sample_id: str, steps: tuple[int, ...], ratio: float) -> dict:
    model, processor = load_model_frozen()
    model.eval()
    item = load_item(sample_id)
    inputs = prepare(processor, item["images"], item["meta"]["question"])
    spans = __import__("run_causal_v2").spans_of(inputs, int(model.config.image_token_index), len(item["images"]))
    clean_page = page_to_float(item["images"][item["aux_idx"]]).to("cuda")
    clear_hooks(model)
    clean_full_tokens, _ = generate_tokens(model, inputs, inputs["pixel_values"], spans, None)
    clean_fast_tokens, _ = generate_tokens(model, inputs, inputs["pixel_values"], spans, ratio)
    clean_text = decode(processor, torch.tensor(clean_full_tokens, device=clean_page.device)).strip()
    clean_fast_text = decode(processor, torch.tensor(clean_fast_tokens, device=clean_page.device)).strip()
    state = {
        "inputs": inputs,
        "spans": spans,
        "clean_text": clean_text,
        "clean_fast_text": clean_fast_text,
        "clean_fast_tokens": clean_fast_tokens,
        "content_positions": content_positions(processor, clean_fast_tokens),
    }
    rows = []
    base = V7D / "runs" / "NATIVE_FDP" / sample_id / "checkpoints"
    for step in steps:
        path = base / f"step_{step:03d}.png"
        page = page_to_float(Image.open(path).convert("RGB")).to("cuda").detach().clone().requires_grad_(True)
        row = score_state(model, processor, item, state, page, path, ratio)
        rows.append(row)
        print(json.dumps({"sample": sample_id, **{k: row[k] for k in ("step", "full_exact", "fastv_text", "clean_prefix_escape_loss", "natural_prefix_escape_loss", "gradient_cosine_clean_vs_natural")}}, ensure_ascii=False), flush=True)
        del page
        torch.cuda.empty_cache()
    payload = {
        "sample_id": sample_id,
        "ratio": ratio,
        "clean_full_text": clean_text,
        "clean_fastv_text": clean_fast_text,
        "clean_fast_token_count": len(clean_fast_tokens),
        "content_positions": state["content_positions"],
        "steps": rows,
        "runtime": {"device": torch.cuda.get_device_name(0), "torch": torch.__version__},
    }
    out = OUT / "diagnostics" / f"{sample_id}_r{int(round(ratio * 100)):02d}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True, choices=SAMPLES)
    parser.add_argument("--steps", default=",".join(str(x) for x in STEPS))
    parser.add_argument("--ratio", type=float, default=DEFAULT_RATIO, choices=(0.25, 0.35))
    args = parser.parse_args()
    steps = tuple(int(x) for x in args.steps.split(",") if x.strip())
    if not steps or any(x not in STEPS for x in steps):
        raise SystemExit(f"steps must be a subset of {STEPS}")
    run(args.sample, steps, args.ratio)


if __name__ == "__main__":
    main()
