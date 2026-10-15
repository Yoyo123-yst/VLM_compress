"""Official prepare-path measurements shared by the scan and the fresh replay."""

from __future__ import annotations

import hashlib

import numpy as np
import torch

from week2_attack_v7b_native_cache_margin_admission.native_cache_margin import hard_margin_row, soft_margin_row


def sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def exact_hash(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return sha_bytes(array.tobytes())


def margin_hash(rows: list[dict]) -> str:
    values = []
    for row in rows:
        values.extend([row["hard_margin"], row["soft_margin"]])
    return sha_bytes(np.asarray(values, dtype=np.float64).tobytes())


def token_margins(logits, token_ids: list[int], pieces: list[str], positions: list[int] | None = None) -> list[dict]:
    chosen = range(len(token_ids)) if positions is None else positions
    packed = []
    for index in chosen:
        hard = hard_margin_row(logits[index], int(token_ids[index]))
        soft = soft_margin_row(logits[index], int(token_ids[index]))
        packed.append({
            "position": int(index),
            "token_id": int(token_ids[index]),
            "piece": pieces[index],
            "hard_margin": float(hard.detach()),
            "soft_margin": float(soft.detach()),
        })
    return packed


def image_stats(image, original) -> dict:
    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    base = np.asarray(original.convert("RGB"), dtype=np.uint8)
    gap = np.abs(array.astype(np.int16) - base.astype(np.int16))
    return {
        "rgb_sha256": sha_bytes(array.tobytes()),
        "mode": image.mode,
        "width": int(image.width),
        "height": int(image.height),
        "max_abs": int(gap.max()),
        "n_diff_pixels": int(np.any(gap > 0, axis=2).sum()),
        "actual_uint8_linf": int(gap.max()),
    }


def score_image(rt, model, processor, item, state, image, reference, pieces, core_positions, clean_kept, cost: dict) -> dict:
    attacked = list(item["images"])
    attacked[item["aux_idx"]] = image
    inputs = rt.v7d.prepare(processor, attacked, item["meta"]["question"])
    spans = rt.v7d.spans_of(inputs, int(model.config.image_token_index), len(attacked))
    pixels = inputs["pixel_values"]
    first, _ = rt.v7d.generate_tokens(model, inputs, pixels, spans, None)
    cost["generate"] += 1
    second, _ = rt.v7d.generate_tokens(model, inputs, pixels, spans, None)
    cost["generate"] += 1
    fast_tokens, kept = rt.v7d.generate_tokens(model, inputs, pixels, spans, 0.25)
    cost["generate"] += 1
    kept = [int(index) for index in kept]
    full_text = rt.v7d.decode(processor, torch.tensor(first)).strip()
    fast_text = rt.v7d.decode(processor, torch.tensor(fast_tokens)).strip()
    full_pieces = [processor.tokenizer.decode([int(token)], skip_special_tokens=False) for token in state["clean_tokens"]]
    natural_index = rt.v7d.keep_index_from_kept(kept, spans, int(inputs["input_ids"].shape[1]), pixels.device)
    with torch.no_grad():
        full_logits = rt.v7d.full_rows(model, inputs, pixels, state["clean_tokens"])
        cost["forward"] += 1
        fast_logits = rt.v7d.fastv_rows(model, inputs, pixels, reference, natural_index)
        cost["forward"] += 1
    full_rows = token_margins(full_logits, state["clean_tokens"], full_pieces)
    core_rows = token_margins(fast_logits, reference, pieces, core_positions)
    del full_logits, fast_logits
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "processor_exact_hash": exact_hash(pixels),
        "full_text": full_text,
        "full_tokens": [int(token) for token in first],
        "full_text_exact": bool(full_text == state["clean_text"] and rt.v7d.decode(processor, torch.tensor(second)).strip() == state["clean_text"]),
        "full_token_exact": bool(first == state["clean_tokens"] and second == state["clean_tokens"]),
        "full_replay": bool(first == second),
        "full_output_valid": bool(rt.output_valid(full_text)),
        "full_margins": full_rows,
        "full_margin_hash": margin_hash(full_rows),
        "fast_text": fast_text,
        "fast_tokens": [int(token) for token in fast_tokens],
        "fast_output_valid": bool(rt.output_valid(fast_text)),
        "k": len(kept),
        "keep_indices": kept,
        "keep_hash": rt.keep_hash(kept),
        "keep_jaccard": rt.jaccard_kept(kept, clean_kept),
        "core_margins": core_rows,
        "core_margin_hash": margin_hash(core_rows),
    }
