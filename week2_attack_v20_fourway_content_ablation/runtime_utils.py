"""Shared runtime helpers. No semantic labels and no checkpoint selection."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
HERE = ROOT / "week2_attack_v20_fourway_content_ablation"
sys.path[:0] = [str(ROOT), str(ROOT / "week1_mechanism"), str(ROOT / "gate_experiment"), str(HERE)]

from week2_attack_v4_answer_aligned.common import jaccard, keep_hash  # noqa: E402
from week2_attack_v4_answer_aligned.loss import output_valid  # noqa: E402
from week2_attack_v4_answer_aligned.run_smoke import page_uint8  # noqa: E402
from week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation import run_v7d as v7d  # noqa: E402
from week2_attack_v13_branch_reachability_probe.run_branch_probe import configure_determinism  # noqa: E402

from week2_attack_v20_fourway_content_ablation.protocol import EPS, EPSILON_UINT8, LAMBDA_FULL, SEED, TRAIN_RATIO  # noqa: E402


def configure() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("VATTACK_ATTN_IMPLEMENTATION", "sdpa")
    configure_determinism()
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED % (2**32))


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def hash_tensor(value: torch.Tensor) -> str:
    array = value.detach().float().cpu().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def tensor_page(image: Image.Image) -> torch.Tensor:
    return v7d.page_to_float(image.convert("RGB")).to("cuda")


def save_png(image: Image.Image, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return sha_file(path)


def reload_png(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def uint8_linf(original: Image.Image, attacked: Image.Image) -> int:
    return int(np.max(np.abs(np.asarray(attacked, dtype=np.int16) - np.asarray(original, dtype=np.int16))))


def full_guard(model, processor, item, state, image: Image.Image) -> dict:
    attacked = list(item["images"])
    attacked[item["aux_idx"]] = image
    inputs = v7d.prepare(processor, attacked, item["meta"]["question"])
    spans = v7d.spans_of(inputs, int(model.config.image_token_index), len(attacked))
    first, _ = v7d.generate_tokens(model, inputs, inputs["pixel_values"], spans, None)
    second, _ = v7d.generate_tokens(model, inputs, inputs["pixel_values"], spans, None)
    text_a = v7d.decode(processor, torch.tensor(first)).strip()
    text_b = v7d.decode(processor, torch.tensor(second)).strip()
    result = {
        "full_text": text_a,
        "full_tokens": [int(token) for token in first],
        "full_text_exact": bool(text_a == state["clean_text"] and text_b == state["clean_text"]),
        "full_token_exact": bool(first == state["clean_tokens"] and second == state["clean_tokens"]),
        "full_replay": bool(first == second and text_a == text_b),
        "output_valid": bool(output_valid(text_a)),
    }
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def fastv_generate(model, processor, item, image: Image.Image, ratio: float) -> dict:
    attacked = list(item["images"])
    attacked[item["aux_idx"]] = image
    inputs = v7d.prepare(processor, attacked, item["meta"]["question"])
    spans = v7d.spans_of(inputs, int(model.config.image_token_index), len(attacked))
    tokens, kept = v7d.generate_tokens(model, inputs, inputs["pixel_values"], spans, ratio)
    text = v7d.decode(processor, torch.tensor(tokens)).strip()
    return {
        "text": text,
        "tokens": [int(token) for token in tokens],
        "kept": [int(token) for token in (kept or [])],
        "valid": bool(output_valid(text)),
        "k": 0 if kept is None else len(kept),
        "keep_hash": keep_hash(list(kept or [])),
    }


def prepare_clean_state(model, processor, item) -> dict:
    state = v7d.prepare_item(model, processor, item)
    image = item["images"][item["aux_idx"]].convert("RGB")
    fast = fastv_generate(model, processor, item, image, TRAIN_RATIO)
    state["clean_fast_tokens"] = list(fast["tokens"])
    state["clean_fast_text"] = fast["text"]
    state["clean_fast_kept"] = list(fast["kept"])
    state["clean_fast_keep_hash"] = fast["keep_hash"]
    state["lambda_full"] = LAMBDA_FULL
    state["eps"] = EPS
    return state


def project_image(clean_page: torch.Tensor, delta: torch.Tensor) -> Image.Image:
    return page_uint8(clean_page, delta)


def jaccard_kept(left: list[int], right: list[int]) -> float:
    return float(jaccard(left, right))


def finite_tensor(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value).all())


def hard_rows(model, state, item, page, tokens: list[int]) -> torch.Tensor:
    rows = v7d.full_token_rows(model, state["inputs"], page, item["aux_idx"], tokens, False)
    return torch.stack([v7d.hard_margin_row(row, token) for row, token in zip(rows, tokens)])
