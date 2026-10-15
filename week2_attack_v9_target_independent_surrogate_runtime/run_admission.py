"""Clean, replay, gradient, and isolation admission for the v9 surrogates.

No pixel update and no target FastV call.
"""

from __future__ import annotations

import hashlib
import io
import json
import random
import sys
import time
from pathlib import Path

import torch
from PIL import Image

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
HERE = ROOT / "week2_attack_v9_target_independent_surrogate_runtime"
sys.path[:0] = [str(ROOT), str(ROOT / "week1_mechanism"), str(HERE)]

from data_mpdoc import parquet_dir  # noqa: E402
from metrics_task import anls_score  # noqa: E402
from surrogate_runtime import (  # noqa: E402
    BLOCK_SEED,
    PAGE_TOKENS,
    apply_plan,
    index_hash,
    local_spatial_merge,
    seeded_block_drop,
    spatial_strided_drop,
)

MODEL_PATH = "/home/yuesiting/autodl-tmp/models/llava-onevision-qwen2-7b-ov-hf"
DEV_IDS = ["1025", "2008", "7006", "39204", "47450", "47567", "50779", "51133", "60700", "6194", "62941", "64506"]
RATIOS = (0.25, 0.35)
ANLS_THRESHOLD = 0.5
DENY_MODULES = (
    "adapters.runtime",
    "adapters.fastv_select",
    "llmc.compression.token_reduction.fastv",
    "week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation.native_grad",
    "week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation.run_v7d",
)


class IsolationFinder:
    def find_spec(self, fullname, path, target=None):
        lowered = fullname.lower()
        if "fastv" in lowered or fullname in DENY_MODULES:
            raise RuntimeError(f"FAIL_TARGET_ISOLATION import {fullname}")
        return None


def install_guard() -> None:
    sys.meta_path.insert(0, IsolationFinder())


def output_valid(text: str) -> bool:
    return bool(text and text.strip())


def sha_image(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def prompt_of(question: str, n_images: int) -> str:
    body = ("<image>" * n_images) + f"\nQuestion: {question}\nAnswer the question using a single word or phrase."
    return body


def select_pages(pages, evidence_idx, sample_id, side: int):
    n = len(pages)
    if evidence_idx < 0 or evidence_idx >= n:
        raise ValueError(f"evidence {evidence_idx} outside {n}")
    if n <= 4:
        order = list(range(n))
        dropped = False
    else:
        rng = random.Random(f"42:{sample_id}")
        others = [i for i in range(n) if i != evidence_idx]
        chosen = rng.sample(others, 3)
        order = chosen + [evidence_idx]
        rng.shuffle(order)
        dropped = True
    images = [pages[i].resize((side, side), Image.Resampling.BICUBIC) for i in order]
    return images, order.index(evidence_idx), order, dropped


def load_spec_images(spec: dict) -> list[Image.Image]:
    import pyarrow.parquet as pq

    frame = pq.read_table(parquet_dir() / spec["shard"]).to_pandas()
    row = frame.iloc[spec["row"]]
    pages = []
    for index in range(1, 21):
        cell = row[f"image_{index}"]
        if cell is None or not isinstance(cell, dict) or cell.get("bytes") is None:
            break
        pages.append(Image.open(io.BytesIO(cell["bytes"])).convert("RGB"))
    images, evidence_idx, order, _ = select_pages(pages, spec["answer_page_idx"], spec["sample_id"], 768)
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
    return model, processor


def prepare_question(processor, images, question: str) -> dict:
    inputs = processor(images=images, text=prompt_of(question, len(images)), return_tensors="pt")
    return {key: value.to("cuda") if torch.is_tensor(value) else value for key, value in inputs.items()}


def decode(processor, token_ids: torch.Tensor) -> str:
    if token_ids.ndim == 1:
        token_ids = token_ids.unsqueeze(0)
    return processor.batch_decode(token_ids, skip_special_tokens=True)[0].strip()


def image_spans(input_ids: torch.Tensor, image_token_id: int, n_pages: int) -> list[tuple[int, int]]:
    positions = (input_ids[0] == image_token_id).nonzero().flatten().tolist()
    if len(positions) != n_pages * PAGE_TOKENS:
        raise RuntimeError(f"image tokens {len(positions)} != {n_pages}*{PAGE_TOKENS}")
    if positions != list(range(positions[0], positions[0] + len(positions))):
        raise RuntimeError("image tokens are not one contiguous run")
    start = positions[0]
    return [(start + page * PAGE_TOKENS, start + (page + 1) * PAGE_TOKENS) for page in range(n_pages)]


def plan_pages(name: str, ratio: float, n_pages: int):
    plans = []
    for page in range(n_pages):
        if name == "SeededBlockDrop":
            plans.append(seeded_block_drop(ratio, page, BLOCK_SEED))
        elif name == "SpatialStridedDrop":
            plans.append(spatial_strided_drop(ratio))
        elif name == "LocalSpatialMerge":
            plans.append(local_spatial_merge(ratio))
        else:
            raise RuntimeError(name)
    return plans


def packed_features(model, inputs: dict) -> list[torch.Tensor]:
    return model.model.get_image_features(
        inputs["pixel_values"],
        inputs["image_sizes"],
        batch_num_images=inputs.get("batch_num_images"),
    )


def build_compressed(model, inputs: dict, features: list[torch.Tensor], plans) -> tuple[torch.Tensor, dict]:
    if len(features) != len(plans):
        raise RuntimeError("feature pages and plans differ")
    reduced = []
    per_page = []
    for feature, plan in zip(features, plans):
        reduced.append(apply_plan(feature, plan))
        per_page.append(list(plan.source_local))
    embeds = model.get_input_embeddings()(inputs["input_ids"])
    spans = image_spans(inputs["input_ids"], int(model.config.image_token_index), len(plans))
    pieces = []
    cursor = 0
    boundaries = []
    for (start, end), visual in zip(spans, reduced):
        pieces.append(embeds[0, cursor:start])
        left = sum(int(piece.shape[0]) for piece in pieces)
        pieces.append(visual.to(dtype=embeds.dtype))
        right = left + int(visual.shape[0])
        boundaries.append({"packed_start": left, "packed_end": right, "source_count": end - start, "output_count": int(visual.shape[0])})
        cursor = end
    pieces.append(embeds[0, cursor:])
    packed = torch.cat(pieces, dim=0).unsqueeze(0)
    n_in = int(inputs["input_ids"].shape[1])
    info = {
        "n_input_tokens": n_in,
        "k_output_tokens": int(packed.shape[1]),
        "per_page_k": [int(item.shape[0]) for item in reduced],
        "per_page_source": per_page,
        "index_hash": index_hash(per_page),
        "page_boundaries": boundaries,
        "dropped": any(plan.dropped for plan in plans),
        "merged": any(plan.merged for plan in plans),
    }
    return packed, info


def generate_ids(model, processor, inputs: dict, embeds: torch.Tensor | None = None) -> list[int]:
    with torch.inference_mode():
        if embeds is None:
            output = model.generate(**inputs, do_sample=False, max_new_tokens=16, use_cache=True)
            prompt_len = int(inputs["input_ids"].shape[1])
            tokens = output[0, prompt_len:]
        else:
            mask = torch.ones(embeds.shape[:2], dtype=torch.long, device=embeds.device)
            output = model.generate(
                inputs_embeds=embeds,
                attention_mask=mask,
                do_sample=False,
                max_new_tokens=16,
                use_cache=True,
            )
            tokens = output[0]
            if tokens.shape[0] > 16:
                tokens = tokens[-16:]
    return [int(token) for token in tokens.detach().cpu().tolist()]


def soft_margin(row: torch.Tensor, token: int) -> torch.Tensor:
    chosen = row[token]
    rival = row.clone()
    rival[token] = float("-inf")
    return chosen - torch.logsumexp(rival, dim=0)


def gradient_smoke(model, processor, item: dict, name: str, ratio: float) -> dict:
    images = item["images"]
    inputs = prepare_question(processor, images, item["spec"]["question"])
    plans = plan_pages(name, ratio, len(images))
    with torch.inference_mode():
        features = packed_features(model, inputs)
        embeds, info = build_compressed(model, inputs, features, plans)
        tokens = generate_ids(model, processor, inputs, embeds)
    if not tokens:
        raise RuntimeError("gradient smoke has no generated tokens")
    pixels = inputs["pixel_values"]
    aux_idx = int(item["aux_idx"])
    leaves = []
    pieces = []
    for index in range(pixels.shape[0]):
        leaf = pixels[index].detach().clone().requires_grad_(True)
        leaves.append(leaf)
        pieces.append(leaf if index == aux_idx else leaf.detach())
    watched = torch.stack(pieces)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    try:
        grad_inputs = dict(inputs)
        grad_inputs["pixel_values"] = watched
        features = packed_features(model, grad_inputs)
        embeds, _ = build_compressed(model, grad_inputs, features, plans)
        teacher = torch.tensor(tokens, device=embeds.device).view(1, -1)
        teacher_embeds = model.get_input_embeddings()(teacher)
        full = torch.cat([embeds, teacher_embeds], dim=1)
        mask = torch.ones(full.shape[:2], dtype=torch.long, device=full.device)
        output = model(inputs_embeds=full, attention_mask=mask, use_cache=False)
        prompt_len = int(embeds.shape[1])
        rows = [output.logits[0, prompt_len - 1 + offset].float() for offset in range(len(tokens))]
        loss = -torch.stack([soft_margin(row, token) for row, token in zip(rows, tokens)]).mean()
        loss = loss + sum(0 * leaf.float().sum() for index, leaf in enumerate(leaves) if index != aux_idx)
        loss.backward()
    finally:
        model.eval()
        model.gradient_checkpointing_disable()
    aux_grad = leaves[aux_idx].grad
    others = []
    for index, leaf in enumerate(leaves):
        if index == aux_idx:
            continue
        grad = leaf.grad
        others.append(0.0 if grad is None else float(grad.detach().float().norm()))
    aux_l2 = None if aux_grad is None else float(aux_grad.detach().float().norm())
    finite = aux_grad is not None and bool(torch.isfinite(aux_grad).all())
    return {
        "sample_id": item["spec"]["sample_id"],
        "policy": name,
        "ratio": ratio,
        "aux_grad_l2": aux_l2,
        "aux_finite": finite,
        "aux_nonzero": bool(finite and aux_l2 > 0),
        "protected_grad_l2": others,
        "protected_all_zero": all(value == 0.0 for value in others),
        "index_hash": info["index_hash"],
        "pass": bool(finite and aux_l2 > 0 and all(value == 0.0 for value in others)),
    }


def access_report() -> dict:
    loaded = sorted(name for name in sys.modules if "fastv" in name.lower() or name in DENY_MODULES)
    return {"target_fastv_access_count": len(loaded), "target_fastv_modules_loaded": loaded}


def static_scan() -> dict:
    hits = []
    banned = ("fastv", "adapters.runtime", "native_grad", "lightcompress")
    for path in sorted(HERE.glob("*.py")):
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            stripped = line.strip()
            if not stripped.startswith(("import ", "from ")):
                continue
            lowered = stripped.lower()
            for name in banned:
                if name in lowered:
                    hits.append(f"{path.name}:{line_number}:{name}")
    return {"hits": hits}


def save_and_reload(images: list[Image.Image], directory: Path) -> list[Image.Image]:
    directory.mkdir(parents=True, exist_ok=True)
    restored = []
    for index, image in enumerate(images):
        dest = directory / f"page_{index}.png"
        image.save(dest)
        restored.append(Image.open(dest).convert("RGB"))
    return restored


def main() -> None:
    install_guard()
    scan = static_scan()
    if scan["hits"]:
        report = {"decision": "FAIL_TARGET_ISOLATION", "static_scan": scan, "access": access_report()}
        (HERE / "target_access_audit.json").write_text(json.dumps(report, indent=2) + "\n")
        (HERE / "final_decision.json").write_text(json.dumps({"decision": "FAIL_TARGET_ISOLATION"}, indent=2) + "\n")
        print(json.dumps(report), flush=True)
        return
    manifest = json.loads((ROOT / "week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation" / "DEV_MANIFEST_COPY.json").read_text())
    specs = {row["sample_id"]: row for row in manifest["samples"]}
    if list(manifest["dev_ids"]) != DEV_IDS:
        raise RuntimeError("development id list differs from the frozen v7d order")
    model, processor = load_model()
    clean_rows = []
    replay_rows = []
    baseline_fail = None
    items = {}
    for sample_id in DEV_IDS:
        spec = specs[sample_id]
        images = load_spec_images(spec)
        item = {"spec": spec, "images": images, "aux_idx": spec["aux_idx"], "evidence_idx": spec["evidence_idx"]}
        items[sample_id] = item
        inputs = prepare_question(processor, images, spec["question"])
        started = time.time()
        tokens = generate_ids(model, processor, inputs)
        text = decode(processor, torch.tensor(tokens))
        match = text == spec["clean_full_pred"]
        row = {
            "sample_id": sample_id,
            "policy": "Full",
            "text": text,
            "frozen_text": spec["clean_full_pred"],
            "match": match,
            "anls": anls_score(text, spec["answers"]),
            "valid": output_valid(text),
            "seconds": round(time.time() - started, 3),
        }
        clean_rows.append(row)
        print(json.dumps({"full": sample_id, "match": match}), flush=True)
        if not match:
            baseline_fail = sample_id
            break
    if baseline_fail is not None:
        payload = {"decision": "FAIL_FULL_BASELINE_REPLAY", "failed_sample": baseline_fail, "rows": clean_rows, "access": access_report()}
        (HERE / "clean_results.json").write_text(json.dumps(payload, indent=2) + "\n")
        (HERE / "final_decision.json").write_text(json.dumps({"decision": "FAIL_FULL_BASELINE_REPLAY", "failed_sample": baseline_fail}, indent=2) + "\n")
        print(json.dumps({"decision": "FAIL_FULL_BASELINE_REPLAY", "sample": baseline_fail}), flush=True)
        return
    policy_names = ["SpatialStridedDrop", "LocalSpatialMerge"]
    for name in policy_names:
        for ratio in RATIOS:
            for sample_id in DEV_IDS:
                item = items[sample_id]
                spec = item["spec"]
                inputs = prepare_question(processor, item["images"], spec["question"])
                plans = plan_pages(name, ratio, len(item["images"]))
                started = time.time()
                features = packed_features(model, inputs)
                embeds, info = build_compressed(model, inputs, features, plans)
                tokens = generate_ids(model, processor, inputs, embeds)
                text = decode(processor, torch.tensor(tokens))
                reloaded = save_and_reload(item["images"], HERE / "replay_png" / name / f"r{ratio}" / sample_id)
                inputs2 = prepare_question(processor, reloaded, spec["question"])
                features2 = packed_features(model, inputs2)
                embeds2, info2 = build_compressed(model, inputs2, features2, plans)
                tokens2 = generate_ids(model, processor, inputs2, embeds2)
                text2 = decode(processor, torch.tensor(tokens2))
                anls = anls_score(text, spec["answers"])
                valid = output_valid(text)
                replay_ok = text == text2 and info["index_hash"] == info2["index_hash"] and tokens == tokens2
                changed = info["k_output_tokens"] < info["n_input_tokens"] and info["k_output_tokens"] > 0
                row = {
                    "sample_id": sample_id,
                    "policy": name,
                    "ratio": ratio,
                    "n": info["n_input_tokens"],
                    "k": info["k_output_tokens"],
                    "per_page_k": info["per_page_k"],
                    "index_hash": info["index_hash"],
                    "page_boundaries": info["page_boundaries"],
                    "dropped": info["dropped"],
                    "merged": info["merged"],
                    "text": text,
                    "replay_text": text2,
                    "anls": anls,
                    "valid": valid,
                    "replay": replay_ok,
                    "token_count_changed": changed,
                    "clean_pass": bool(valid and anls >= ANLS_THRESHOLD and replay_ok and changed),
                    "seconds": round(time.time() - started, 3),
                    "failure": None if valid and anls >= ANLS_THRESHOLD and replay_ok and changed else "clean_or_replay",
                }
                clean_rows.append(row)
                replay_rows.append({key: row[key] for key in ("sample_id", "policy", "ratio", "replay", "index_hash", "text", "replay_text")})
                print(json.dumps({"policy": name, "ratio": ratio, "sample": sample_id, "anls": round(anls, 4), "replay": replay_ok, "k": info["k_output_tokens"]}), flush=True)
    def ran_on_path(name: str) -> bool:
        rows = [row for row in clean_rows if row.get("policy") == name]
        return len(rows) == len(DEV_IDS) * len(RATIOS) and all(row["token_count_changed"] for row in rows)

    if ran_on_path("SpatialStridedDrop") and ran_on_path("LocalSpatialMerge"):
        for ratio in RATIOS:
            for sample_id in DEV_IDS:
                item = items[sample_id]
                spec = item["spec"]
                name = "SeededBlockDrop"
                inputs = prepare_question(processor, item["images"], spec["question"])
                plans = plan_pages(name, ratio, len(item["images"]))
                started = time.time()
                features = packed_features(model, inputs)
                embeds, info = build_compressed(model, inputs, features, plans)
                tokens = generate_ids(model, processor, inputs, embeds)
                text = decode(processor, torch.tensor(tokens))
                reloaded = save_and_reload(item["images"], HERE / "replay_png" / name / f"r{ratio}" / sample_id)
                inputs2 = prepare_question(processor, reloaded, spec["question"])
                embeds2, info2 = build_compressed(model, inputs2, packed_features(model, inputs2), plans)
                tokens2 = generate_ids(model, processor, inputs2, embeds2)
                text2 = decode(processor, torch.tensor(tokens2))
                anls = anls_score(text, spec["answers"])
                valid = output_valid(text)
                replay_ok = text == text2 and info["index_hash"] == info2["index_hash"] and tokens == tokens2
                changed = info["k_output_tokens"] < info["n_input_tokens"]
                row = {
                    "sample_id": sample_id,
                    "policy": name,
                    "ratio": ratio,
                    "n": info["n_input_tokens"],
                    "k": info["k_output_tokens"],
                    "per_page_k": info["per_page_k"],
                    "index_hash": info["index_hash"],
                    "page_boundaries": info["page_boundaries"],
                    "dropped": True,
                    "merged": False,
                    "text": text,
                    "replay_text": text2,
                    "anls": anls,
                    "valid": valid,
                    "replay": replay_ok,
                    "token_count_changed": changed,
                    "clean_pass": bool(valid and anls >= ANLS_THRESHOLD and replay_ok and changed),
                    "seconds": round(time.time() - started, 3),
                    "failure": None if valid and anls >= ANLS_THRESHOLD and replay_ok and changed else "clean_or_replay",
                    "role": "structured_surrogate_augmentation",
                }
                clean_rows.append(row)
                replay_rows.append({key: row[key] for key in ("sample_id", "policy", "ratio", "replay", "index_hash", "text", "replay_text")})
                print(json.dumps({"policy": name, "ratio": ratio, "sample": sample_id, "anls": round(anls, 4), "replay": replay_ok}), flush=True)
    admitted = []
    for name in ("SpatialStridedDrop", "LocalSpatialMerge", "SeededBlockDrop"):
        rows = [row for row in clean_rows if row.get("policy") == name]
        if rows and all(row["clean_pass"] and row["replay"] for row in rows):
            admitted.append(name)
    gradients = []
    for name in admitted:
        if name == "SeededBlockDrop":
            continue
        gradients.append(gradient_smoke(model, processor, items[DEV_IDS[0]], name, 0.25))
        print(json.dumps({"gradient": name, "pass": gradients[-1]["pass"]}), flush=True)
    access = access_report()
    if access["target_fastv_access_count"] != 0:
        decision = "FAIL_TARGET_ISOLATION"
    elif len([name for name in admitted if name in ("SpatialStridedDrop", "LocalSpatialMerge")]) == 2 and all(row["pass"] for row in gradients):
        decision = "PASS_TWO_HETEROGENEOUS_SURROGATES_ADMITTED"
    elif len([name for name in admitted if name in ("SpatialStridedDrop", "LocalSpatialMerge")]) == 1:
        decision = "PARTIAL_ONE_SURROGATE_ADMITTED"
    else:
        decision = "FAIL_NO_SURROGATE_RUNTIME_ADMITTED"
    (HERE / "clean_results.json").write_text(json.dumps({"rows": clean_rows, "admitted": admitted}, indent=2) + "\n")
    (HERE / "replay_results.json").write_text(json.dumps({"rows": replay_rows}, indent=2) + "\n")
    (HERE / "gradient_smoke.json").write_text(json.dumps({"rows": gradients}, indent=2) + "\n")
    (HERE / "target_access_audit.json").write_text(json.dumps({"static_scan": scan, "access": access}, indent=2) + "\n")
    (HERE / "final_decision.json").write_text(json.dumps({
        "decision": decision,
        "admitted": admitted,
        "pixel_optimization_run": False,
        "target_fastv_called": access["target_fastv_access_count"] != 0,
        "new_attack_success_rate": None,
        "next_stage_allowed": decision == "PASS_TWO_HETEROGENEOUS_SURROGATES_ADMITTED",
    }, indent=2) + "\n")
    print(json.dumps({"decision": decision, "admitted": admitted}), flush=True)


if __name__ == "__main__":
    main()
