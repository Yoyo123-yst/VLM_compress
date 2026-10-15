"""Seal the selected deltas and score the six real endpoints."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from adapters.runtime import clear_hooks, install
from metrics_task import anls_score
from projection import apply_uint8, delta_norms, float_from_uint8
from run_eval import decode
from run_probe import (
    CHECKPOINTS,
    EVAL_RATIOS,
    HERE,
    TRAIN_RATIOS,
    Access,
    component_margin,
    counters,
    leaf_uint8,
    page_of,
    png_from_deltas,
    prepare_state,
    run_attack,
    run_comp,
    sha_file,
)
from week2_attack_v4_answer_aligned.common import keep_hash
from week2_attack_v4_answer_aligned.loss import output_valid
from week2_attack_v7b_native_cache_margin_admission.native_cache_margin import (
    cache_replay,
    hard_margin_row,
    soft_margin_row,
)
from week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation import run_v7d as v7d


def margins(rows, tokens) -> dict:
    hard = torch.stack([hard_margin_row(row, token) for row, token in zip(rows, tokens)])
    soft = torch.stack([soft_margin_row(row, token) for row, token in zip(rows, tokens)])
    return {"hard": float(hard.mean()), "soft": float(soft.mean())}


def measure(model, processor, item, image: Image.Image) -> dict:
    attacked = list(item["images"])
    attacked[item["aux_idx"]] = image
    inputs = v7d.prepare(processor, attacked, item["meta"]["question"])
    spans = v7d.spans_of(inputs, int(model.config.image_token_index), len(attacked))
    answers = list(item["meta"]["answers"])

    def one_full():
        tokens, _ = v7d.generate_tokens(model, inputs, inputs["pixel_values"], spans, None)
        text = decode(processor, torch.tensor(tokens)).strip()
        clear_hooks(model)
        rows, _ = cache_replay(model, dict(inputs), tokens)
        return {"tokens": tokens, "text": text, "valid": output_valid(text), **margins(rows, tokens)}

    first = one_full()
    second = one_full()
    ratios = {}
    for ratio in EVAL_RATIOS:
        passes = []
        for _ in range(2):
            tokens, kept = v7d.generate_tokens(model, inputs, inputs["pixel_values"], spans, ratio)
            text = decode(processor, torch.tensor(tokens)).strip()
            clear_hooks(model)
            state = install(model, "fastv", ratio, int(model.config.image_token_index))
            state.spans = spans
            rows, _ = cache_replay(model, dict(inputs), tokens)
            hook = list(state.kept_visual)
            clear_hooks(model)
            passes.append({
                "tokens": tokens,
                "text": text,
                "valid": output_valid(text),
                "anls": anls_score(text, answers),
                "k": len(kept),
                "keep_hash": keep_hash(kept),
                "hook_match": int(list(kept) == hook),
                **margins(rows, tokens),
            })
        ratios[f"{ratio:.2f}"] = {
            "text": passes[0]["text"],
            "tokens": passes[0]["tokens"],
            "valid": passes[0]["valid"],
            "anls": passes[0]["anls"],
            "k": passes[0]["k"],
            "keep_hash": passes[0]["keep_hash"],
            "hook_match": passes[0]["hook_match"] and passes[1]["hook_match"],
            "hard": passes[0]["hard"],
            "soft": passes[0]["soft"],
            "replay": passes[0]["tokens"] == passes[1]["tokens"] and passes[0]["text"] == passes[1]["text"],
            "endpoint_failure": bool(passes[0]["valid"] and passes[0]["anls"] < 0.5 and passes[0]["tokens"] == passes[1]["tokens"]),
        }
    clean = np.asarray(item["images"][item["aux_idx"]].convert("RGB"), dtype=np.uint8)
    got = np.asarray(image.convert("RGB"), dtype=np.uint8)
    other = []
    for index, page in enumerate(item["images"]):
        if index == item["aux_idx"]:
            continue
        other.append(sha_file_image(page))
    return {
        "full_text": first["text"],
        "full_tokens": first["tokens"],
        "full_valid": first["valid"],
        "full_hard": first["hard"],
        "full_soft": first["soft"],
        "full_replay": first["tokens"] == second["tokens"] and first["text"] == second["text"],
        "ratios": ratios,
        "aux_sha256": sha_png(image),
        "other_page_sha256": other,
        "norms": delta_norms(clean, got.astype(np.int16) - clean.astype(np.int16)),
    }


def sha_png(image: Image.Image) -> str:
    import io
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return sha_bytes(buffer.getvalue())


def sha_file_image(image: Image.Image) -> str:
    return sha_png(image)


def sha_bytes(payload: bytes) -> str:
    import hashlib
    return hashlib.sha256(payload).hexdigest()


def delta_from_png(clean: Image.Image, image: Image.Image, device) -> torch.Tensor:
    return (float_from_uint8(image) - float_from_uint8(clean)).to(device)


def single_project(clean_u8: np.ndarray, delta_u8: np.ndarray) -> np.ndarray:
    from projection import project_int
    return project_int(delta_u8, clean_u8)


def run_full(model, processor, sample_id: str) -> None:
    item = v7d.load_dev(sample_id)
    state = prepare_state(model, processor, item)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    work = HERE / "work" / sample_id
    attack = run_attack(model, processor, item, state, 100, work)
    (HERE / "curves").mkdir(parents=True, exist_ok=True)
    (HERE / "curves" / f"{sample_id}_attack.json").write_text(json.dumps(attack["curve"], indent=2) + "\n")
    result = {
        "sample_id": sample_id,
        "clean_text": state["clean_text"],
        "attack_chosen": attack["chosen"],
        "status": "OK",
    }
    if attack["chosen"] is None:
        result["status"] = "NO_VALID_ATTACK_COMPONENT"
        finish(result, started)
        return
    attack_png = Image.open(next(row["png"] for row in attack["curve"] if row["selector"]["step"] == attack["chosen"]["step"])).convert("RGB")
    attack_delta = delta_from_png(item["images"][item["aux_idx"]], attack_png, state["page"].device)
    ref = component_margin(model, state, item, page_of(state, attack_delta))
    comp = run_comp(model, processor, item, state, attack_delta, ref, 100, work)
    (HERE / "curves" / f"{sample_id}_comp.json").write_text(json.dumps(comp["curve"], indent=2) + "\n")
    result["comp_chosen"] = comp["chosen"]
    result["comp_accepted"] = comp["accepted"]
    result["comp_rejected"] = comp["rejected"]
    result["attack_ref_margin"] = ref
    if comp["chosen"] is None:
        result["status"] = "NO_VALID_COMPENSATED_CANDIDATE"
        result["endpoints"] = {
            "CLEAN": measure(model, processor, item, item["images"][item["aux_idx"]]),
            "ATTACK_ONLY": measure(model, processor, item, attack_png),
        }
        finish(result, started)
        return
    comp_png = Image.open(next(row["png"] for row in comp["curve"] if row["selector"]["step"] == comp["chosen"]["step"])).convert("RGB")
    clean_u8 = np.asarray(item["images"][item["aux_idx"]].convert("RGB"), dtype=np.uint8)
    attack_u8 = np.asarray(attack_png.convert("RGB"), dtype=np.int16) - clean_u8.astype(np.int16)
    ac_u8 = np.asarray(comp_png.convert("RGB"), dtype=np.int16) - clean_u8.astype(np.int16)
    comp_u8 = ac_u8 - attack_u8
    comp_only = apply_uint8(clean_u8, comp_u8)
    sign = apply_uint8(clean_u8, single_project(clean_u8, attack_u8 - comp_u8))
    rolled = np.roll(comp_u8, shift=(clean_u8.shape[0] // 4, clean_u8.shape[1] // 4), axis=(0, 1))
    roll = apply_uint8(clean_u8, single_project(clean_u8, attack_u8 + rolled))
    sealed = HERE / "sealed" / sample_id
    sealed.mkdir(parents=True, exist_ok=True)
    images = {
        "CLEAN": item["images"][item["aux_idx"]],
        "ATTACK_ONLY": attack_png,
        "COMPENSATION_ONLY": comp_only,
        "ATTACK_PLUS_COMPENSATION": comp_png,
        "SIGN_FLIP_CONTROL": sign,
        "ROLL_CONTROL": roll,
    }
    names = {
        "ATTACK_ONLY": "attack_only.png",
        "COMPENSATION_ONLY": "compensation_only.png",
        "ATTACK_PLUS_COMPENSATION": "attack_plus_compensation.png",
        "SIGN_FLIP_CONTROL": "sign_flip_control.png",
        "ROLL_CONTROL": "roll_control.png",
    }
    for key, name in names.items():
        images[key].save(sealed / name)
    result["endpoints"] = {key: measure(model, processor, item, image) for key, image in images.items()}
    result["control_norms_not_equalized"] = True
    finish(result, started)


def finish(result: dict, started: float) -> None:
    result["counters"] = counters()
    result["elapsed_sec"] = time.perf_counter() - started
    result["peak_memory_mib"] = torch.cuda.max_memory_allocated() / (1024 ** 2)
    dest = HERE / "per_sample" / f"{result['sample_id']}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"sample": result["sample_id"], "status": result["status"], "counters": result["counters"]}), flush=True)
