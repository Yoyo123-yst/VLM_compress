"""Shared Full-path helpers. This module does not import FastV."""

from __future__ import annotations

import ast
import hashlib
import io
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
HERE = ROOT / "week2_attack_v9b_rank_signal_oblivious_admission"
sys.path[:0] = [str(ROOT), str(ROOT / "week1_mechanism")]

from data_mpdoc import parquet_dir  # noqa: E402
from metrics_task import anls_score, normalize  # noqa: E402

MODEL_PATH = "/home/yuesiting/autodl-tmp/models/llava-onevision-qwen2-7b-ov-hf"
MANIFEST = ROOT / "week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation" / "DEV_MANIFEST_COPY.json"
ID_SOURCE = ROOT / "week2_attack_v8_graybox_admission" / "future_transfer_admission_protocol.md"
ADMISSION_IDS = ("1025", "2008", "7006", "39204", "47450")
FROZEN_1025_IDS = [8847, 13, 356, 13, 425, 4110, 13979, 151645]
FROZEN_1025_TEXT = "Dr. C. Baur MD"
PAGE_TOKENS = 3699
MAX_NEW_TOKENS = 16
EPS = 8 / 255
TOLERANCE = 1e-6
GLOBAL_SEED = 2026101115
TRAIN_STEPS = 200
CHECKPOINTS = (0, 10, 25, 50, 75, 100, 150, 200)
DIAGNOSTIC_RATIOS = (0.25, 0.35, 0.50)
ANLS_CLEAN = 0.5
XI = 1.0
ETA = 1e-3


def sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha_tensor(value: torch.Tensor) -> str:
    raw = value.detach().cpu().contiguous()
    return sha_bytes(raw.view(torch.uint8).numpy().tobytes())


def sha_image(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return sha_bytes(buffer.getvalue())


def load_build_prompt():
    """Execute the historical function body from run_eval.py. Do not retype it."""
    path = ROOT / "week1_mechanism" / "run_eval.py"
    text = path.read_text()
    tree = ast.parse(text)
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [getattr(target, "id", None) for target in node.targets]
            if "IMAGE_TOKEN" in names:
                nodes.append(node)
        if isinstance(node, ast.FunctionDef) and node.name == "build_prompt":
            nodes.append(node)
    if len(nodes) != 2:
        raise RuntimeError("historical build_prompt was not found intact")
    segment = ast.get_source_segment(text, nodes[-1])
    blob = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(blob)
    namespace: dict = {}
    exec(compile(blob, str(path), "exec"), namespace)
    return namespace["build_prompt"], sha_bytes(segment.encode()), str(path)


def manifest_samples() -> list[dict]:
    payload = json.loads(MANIFEST.read_text())
    rows = [row for row in payload["samples"] if row["sample_id"] in ADMISSION_IDS]
    if [row["sample_id"] for row in rows] != list(ADMISSION_IDS):
        found = [row["sample_id"] for row in rows]
        raise RuntimeError(f"admission ids missing from the frozen dev manifest: {found}")
    return rows


def answers_of(spec: dict) -> list[str]:
    value = spec["answers"]
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def select_pages(pages, evidence_idx, sample_id, side: int = 768):
    n = len(pages)
    if n <= 4:
        order = list(range(n))
    else:
        rng = random.Random(f"42:{sample_id}")
        others = [i for i in range(n) if i != evidence_idx]
        chosen = rng.sample(others, 3)
        order = chosen + [evidence_idx]
        rng.shuffle(order)
    images = [pages[i].resize((side, side), Image.Resampling.BICUBIC) for i in order]
    return images, order.index(evidence_idx), order


def load_images(spec: dict) -> list[Image.Image]:
    import pyarrow.parquet as pq

    frame = pq.read_table(parquet_dir() / spec["shard"]).to_pandas()
    row = frame.iloc[spec["row"]]
    pages = []
    for index in range(1, 21):
        cell = row[f"image_{index}"]
        if cell is None or not isinstance(cell, dict) or cell.get("bytes") is None:
            break
        pages.append(Image.open(io.BytesIO(cell["bytes"])).convert("RGB"))
    images, evidence_idx, order = select_pages(pages, spec["answer_page_idx"], spec["sample_id"])
    digest = [sha_image(image) for image in images]
    if digest != spec["page_sha256"] or order != spec["order"] or evidence_idx != spec["evidence_idx"]:
        raise RuntimeError(f"page restore mismatch {spec['sample_id']}")
    return images


def load_model():
    from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    model.config.image_grid_pinpoints = [[768, 768]]
    processor.image_processor.image_grid_pinpoints = [[768, 768]]
    model.to("cuda")
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, processor


def prepare(processor, images, question: str) -> dict:
    build_prompt, _, _ = load_build_prompt()
    prompt = build_prompt(question, len(images), "A")
    inputs = processor(images=images, text=prompt, return_tensors="pt")
    moved = {key: value.to("cuda") if torch.is_tensor(value) else value for key, value in inputs.items()}
    moved["prompt"] = prompt
    return moved


def generate_tokens(model, processor, inputs: dict) -> tuple[list[int], str]:
    with torch.inference_mode():
        output = model.generate(
            **{key: value for key, value in inputs.items() if key != "prompt"},
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,
            use_cache=True,
        )
    prompt_len = int(inputs["input_ids"].shape[1])
    tokens = [int(token) for token in output[0, prompt_len:].tolist()]
    text = processor.batch_decode(torch.tensor(tokens).unsqueeze(0), skip_special_tokens=True)[0].strip()
    return tokens, text


def image_spans(input_ids: torch.Tensor, image_token_id: int, n_pages: int) -> list[tuple[int, int]]:
    ids = input_ids[0].tolist()
    spans = []
    index = 0
    while index < len(ids):
        if ids[index] == image_token_id:
            start = index
            while index < len(ids) and ids[index] == image_token_id:
                index += 1
            spans.append((start, index))
        else:
            index += 1
    if len(spans) == 1 and spans[0][1] - spans[0][0] == n_pages * PAGE_TOKENS:
        start = spans[0][0]
        spans = [(start + index * PAGE_TOKENS, start + (index + 1) * PAGE_TOKENS) for index in range(n_pages)]
    if len(spans) != n_pages or any(end - start != PAGE_TOKENS for start, end in spans):
        raise RuntimeError(f"image spans {spans} do not match {n_pages} pages of {PAGE_TOKENS}")
    return spans


def output_valid(text: str) -> bool:
    return bool(text and text.strip())


def page_to_float(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def quantize_in_ball(clean_uint8: torch.Tensor, adv_float: torch.Tensor, eps_u: int = 8) -> torch.Tensor:
    raw = adv_float.detach().mul(255).round().clamp(0, 255)
    lo = (clean_uint8 - eps_u).clamp(0, 255)
    hi = (clean_uint8 + eps_u).clamp(0, 255)
    return raw.clamp(lo, hi).to(torch.uint8)


def page_uint8(clean_page: torch.Tensor, delta: torch.Tensor) -> Image.Image:
    clean_u8 = clean_page.detach().mul(255).round()
    quantized = quantize_in_ball(clean_u8, (clean_page.detach() + delta.detach()).clamp(0, 1), 8)
    array = quantized.permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array, mode="RGB")


def project(delta: torch.Tensor, clean: torch.Tensor, eps: float = EPS) -> None:
    with torch.no_grad():
        delta.clamp_(-eps, eps)
        delta.copy_((clean + delta).clamp(0, 1) - clean)


def uint8_linf(original: Image.Image, attacked: Image.Image) -> float:
    return float((page_to_float(attacked) - page_to_float(original)).abs().max())


def normalized_tiles(image: torch.Tensor) -> torch.Tensor:
    if image.shape != (3, 768, 768):
        raise ValueError(f"expected 3x768x768, got {tuple(image.shape)}")
    high = image.view(3, 2, 384, 2, 384).permute(1, 3, 0, 2, 4).reshape(4, 3, 384, 384)
    torch_base = torch.nn.functional.interpolate(
        image.unsqueeze(0), size=(384, 384), mode="bicubic", align_corners=False, antialias=True
    ).squeeze(0)
    page = image.detach().clamp(0, 1).mul(255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    small = np.asarray(Image.fromarray(page, mode="RGB").resize((384, 384), Image.Resampling.BICUBIC), dtype=np.float32)
    pil_base = torch.from_numpy(small).permute(2, 0, 1).to(device=image.device, dtype=image.dtype) / 255.0
    base = pil_base + torch_base - torch_base.detach()
    tiles = torch.cat([base.unsqueeze(0), high], dim=0)
    return (tiles - 0.5) / 0.5


def splice_aux(pixel_values: torch.Tensor, aux_idx: int, tiles: torch.Tensor) -> torch.Tensor:
    pages = [pixel_values[index].detach() for index in range(pixel_values.shape[0])]
    pages[aux_idx] = tiles.to(dtype=pixel_values.dtype)
    return torch.stack(pages)


def fastv_modules() -> list[str]:
    return sorted(name for name in sys.modules if "fastv" in name.lower())
