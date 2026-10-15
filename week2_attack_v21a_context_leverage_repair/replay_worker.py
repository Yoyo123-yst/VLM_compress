"""One fresh process, three zero-perturbation image paths. No hooks and no training."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
sys.path[:0] = [str(ROOT), str(ROOT / "week1_mechanism"), str(ROOT / "gate_experiment")]
HERE = ROOT / "week2_attack_v21a_context_leverage_repair"
V21 = ROOT / "week2_attack_v21_context_leverage_causal_probe"

from week2_attack_v20_fourway_content_ablation import runtime_utils as rt  # noqa: E402
from week2_attack_v20_fourway_content_ablation.protocol import TRAIN_RATIO  # noqa: E402
from week2_attack_v7b_native_cache_margin_admission.native_cache_margin import hard_margin_row, soft_margin_row  # noqa: E402


def sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def tensor_exact_hash(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous()
    return sha_bytes(array.numpy().tobytes())


def tensor_float_hash(value: torch.Tensor) -> str:
    array = value.detach().float().cpu().contiguous().numpy()
    return sha_bytes(array.tobytes())


def image_facts(image, original) -> dict:
    rgb = image.convert("RGB")
    array = np.asarray(rgb, dtype=np.uint8)
    base = np.asarray(original.convert("RGB"), dtype=np.uint8)
    gap = np.abs(array.astype(np.int16) - base.astype(np.int16))
    raw = array.tobytes()
    buffer = io.BytesIO()
    rgb.save(buffer, format="PNG")
    return {
        "rgb_sha256": sha_bytes(raw),
        "numpy_sha256": sha_bytes(array.tobytes()),
        "memory_png_sha256": sha_bytes(buffer.getvalue()),
        "mode": rgb.mode,
        "shape": list(array.shape),
        "max_abs": int(gap.max()),
        "n_diff_pixels": int(np.any(gap > 0, axis=2).sum()),
        "uint8_linf": int(gap.max()),
    }


def margin_rows(rows, reference: list[int], positions: list[int], pieces: list[str]) -> list[dict]:
    packed = []
    for index in positions:
        hard = hard_margin_row(rows[index], int(reference[index]))
        soft = soft_margin_row(rows[index], int(reference[index]))
        packed.append({
            "position": index,
            "token_id": int(reference[index]),
            "piece": pieces[index],
            "hard_margin": float(hard.detach()),
            "soft_margin": float(soft.detach()),
        })
    return packed


def margin_hash(core: list[dict], context: list[dict]) -> str:
    values = []
    for row in core + context:
        values.extend([row["hard_margin"], row["soft_margin"]])
    return sha_bytes(np.asarray(values, dtype=np.float64).tobytes())


def rows_hash(rows) -> str:
    stacked = torch.stack([row.detach().float().cpu() for row in rows])
    return sha_bytes(stacked.numpy().tobytes())


def evaluate(model, processor, item, image, original_inputs, original_spans, reference, pieces, core_pos, context_pos, frozen_kept) -> dict:
    attacked = list(item["images"])
    attacked[item["aux_idx"]] = image
    inputs = rt.v7d.prepare(processor, attacked, item["meta"]["question"])
    spans = rt.v7d.spans_of(inputs, int(model.config.image_token_index), len(attacked))
    pixels = inputs["pixel_values"]
    base = original_inputs["pixel_values"]
    gap = (pixels.float() - base.float()).abs()
    first, _ = rt.v7d.generate_tokens(model, inputs, pixels, spans, None)
    second, _ = rt.v7d.generate_tokens(model, inputs, pixels, spans, None)
    fast_tokens, kept = rt.v7d.generate_tokens(model, inputs, pixels, spans, TRAIN_RATIO)
    kept = [int(index) for index in kept]
    full_text = rt.v7d.decode(processor, torch.tensor(first)).strip()
    fast_text = rt.v7d.decode(processor, torch.tensor(fast_tokens)).strip()
    natural_keep = rt.v7d.keep_index_from_kept(kept, spans, int(inputs["input_ids"].shape[1]), pixels.device)
    with torch.no_grad():
        natural_rows = rt.v7d.fastv_rows(model, inputs, pixels, reference, natural_keep)
    core = margin_rows(natural_rows, reference, core_pos, pieces)
    context = margin_rows(natural_rows, reference, context_pos, pieces)
    del natural_rows
    page = rt.tensor_page(image)
    spliced = rt.v7d.pixels_from_page(original_inputs, item["aux_idx"], page)
    splice_tokens, splice_kept = rt.v7d.generate_tokens(model, original_inputs, spliced, original_spans, TRAIN_RATIO)
    del spliced, page
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "processor_type": type(inputs).__name__,
        "pixel_shape": list(pixels.shape),
        "pixel_dtype": str(pixels.dtype),
        "processor_exact_hash": tensor_exact_hash(pixels),
        "processor_float_hash": tensor_float_hash(pixels),
        "pixel_min": float(pixels.float().min()),
        "pixel_max": float(pixels.float().max()),
        "pixel_mean": float(pixels.float().mean()),
        "processor_max_abs_vs_original": float(gap.max()),
        "processor_nonzero_diff": int((gap > 0).sum()),
        "full_text": full_text,
        "full_tokens": [int(token) for token in first],
        "full_replay": bool(first == second),
        "generated_text": fast_text,
        "generated_tokens": [int(token) for token in fast_tokens],
        "k": len(kept),
        "keep_indices": kept,
        "keep_hash": rt.keep_hash(kept),
        "natural_logits_hash": None,
        "core_natural": core,
        "context_natural": context,
        "margin_hash": margin_hash(core, context),
        "v21_splice_keep_hash": rt.keep_hash([int(index) for index in splice_kept]),
        "v21_splice_tokens": [int(token) for token in splice_tokens],
        "frozen_kept_used": frozen_kept,
    }


def attach_logits(model, processor, item, image, reference, frozen_kept, payload) -> None:
    attacked = list(item["images"])
    attacked[item["aux_idx"]] = image
    inputs = rt.v7d.prepare(processor, attacked, item["meta"]["question"])
    spans = rt.v7d.spans_of(inputs, int(model.config.image_token_index), len(attacked))
    pixels = inputs["pixel_values"]
    natural_keep = rt.v7d.keep_index_from_kept(payload["keep_indices"], spans, int(inputs["input_ids"].shape[1]), pixels.device)
    frozen_keep = rt.v7d.keep_index_from_kept(frozen_kept, spans, int(inputs["input_ids"].shape[1]), pixels.device)
    with torch.no_grad():
        natural_rows = rt.v7d.fastv_rows(model, inputs, pixels, reference, natural_keep)
        frozen_rows = rt.v7d.fastv_rows(model, inputs, pixels, reference, frozen_keep)
    payload["natural_logits_hash"] = rows_hash(natural_rows)
    payload["frozen_logits_hash"] = rows_hash(frozen_rows)
    del natural_rows, frozen_rows
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=int, required=True)
    args = parser.parse_args()
    groups = json.loads((V21 / "frozen_token_groups.json").read_text())
    reference = [int(row["token_id"]) for row in groups["tokens"]]
    pieces = [row["decoded_piece"] for row in groups["tokens"]]
    core_pos = groups["s_core"]["positions"]
    context_pos = groups["s_context"]["positions"]
    rt.configure()
    model, processor = rt.v7d.load_model_frozen()
    item = rt.v7d.load_dev("47450")
    original = item["images"][item["aux_idx"]].convert("RGB")
    clean_page = rt.tensor_page(original)
    delta = torch.zeros_like(clean_page)
    projected = rt.project_image(clean_page, delta)
    projected_path = HERE / "replay_runs" / f"run_{args.run}_zero_projected.png"
    direct_path = HERE / "replay_runs" / f"run_{args.run}_zero_direct.png"
    rt.save_png(projected, projected_path)
    rt.save_png(original, direct_path)
    projected_image = rt.reload_png(projected_path)
    direct_image = rt.reload_png(direct_path)
    original_inputs = rt.v7d.prepare(processor, item["images"], item["meta"]["question"])
    original_spans = rt.v7d.spans_of(original_inputs, int(model.config.image_token_index), len(item["images"]))
    paths = {}
    order = (("ORIGINAL", original, None), ("ZERO_PROJECTED", projected_image, projected_path), ("ZERO_DIRECT_SAVE", direct_image, direct_path))
    frozen_kept = None
    for name, image, saved in order:
        facts = image_facts(image, original)
        if saved is None:
            facts["file_sha256"] = facts["memory_png_sha256"]
        else:
            facts["file_sha256"] = rt.sha_file(saved)
        measured = evaluate(
            model, processor, item, image, original_inputs, original_spans,
            reference, pieces, core_pos, context_pos, frozen_kept,
        )
        if name == "ORIGINAL":
            frozen_kept = measured["keep_indices"]
        measured.update(facts)
        measured["jaccard_vs_original"] = 1.0
        paths[name] = measured
        print(json.dumps({"run": args.run, "path": name, "keep": measured["keep_hash"], "k": measured["k"], "linf": measured["uint8_linf"]}), flush=True)
    for name, image, _saved in order:
        attach_logits(model, processor, item, image, reference, frozen_kept, paths[name])
        paths[name]["jaccard_vs_original"] = rt.jaccard_kept(paths[name]["keep_indices"], paths["ORIGINAL"]["keep_indices"])
    payload = {"run": args.run, "gpu": "0", "paths": paths, "frozen_keep_hash": rt.keep_hash(frozen_kept)}
    rt.write_json(HERE / "replay_runs" / f"run_{args.run}.json", payload)


if __name__ == "__main__":
    main()
