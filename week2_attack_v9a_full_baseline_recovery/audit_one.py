"""Compare the historical Full prompt with the v9 wrapper on sample 1025.

The historical prompt text is copied from week1_mechanism/run_eval.py build_prompt.
This file does not import that module, because that module imports the FastV adapter.
"""

from __future__ import annotations

import hashlib
import io
import json
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
HERE = ROOT / "week2_attack_v9a_full_baseline_recovery"
sys.path[:0] = [str(ROOT), str(ROOT / "week1_mechanism")]

from data_mpdoc import parquet_dir  # noqa: E402

MODEL_PATH = Path("/home/yuesiting/autodl-tmp/models/llava-onevision-qwen2-7b-ov-hf")
FROZEN_TEXT = "Dr. C. Baur MD"
PYTHON = "/home/yuesiting/autodl-tmp/conda/envs/vattack/bin/python"


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


def historical_prompt(question: str, n_images: int) -> str:
    body = ("<image>" * n_images) + f"\nQuestion: {question}\nAnswer the question using a single word or phrase."
    return "<|im_start|>user\n" + body + "<|im_end|>\n<|im_start|>assistant\n"


def v9_prompt(question: str, n_images: int) -> str:
    return ("<image>" * n_images) + f"\nQuestion: {question}\nAnswer the question using a single word or phrase."


def select_pages(pages, evidence_idx, sample_id, side: int):
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
    images, evidence_idx, order = select_pages(pages, spec["answer_page_idx"], spec["sample_id"], 768)
    digest = [sha_image(image) for image in images]
    if digest != spec["page_sha256"] or order != spec["order"] or evidence_idx != spec["evidence_idx"]:
        raise RuntimeError("page restore mismatch")
    return images


def git_state() -> dict:
    def run(args: list[str]) -> str:
        result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True)
        return (result.stdout or result.stderr).strip()

    return {
        "commit": run(["git", "rev-parse", "HEAD"]),
        "status": run(["git", "status", "--short"]),
        "diff_stat": run(["git", "diff", "--stat"]),
    }


def package_versions() -> dict:
    import transformers

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
    }


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


def jsonable(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def main() -> None:
    which = sys.argv[1]
    if which not in {"original", "v9"}:
        raise SystemExit("usage: audit_one.py original|v9")
    started = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    manifest = json.loads((ROOT / "week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation" / "DEV_MANIFEST_COPY.json").read_text())
    spec = next(row for row in manifest["samples"] if row["sample_id"] == "1025")
    images = load_images(spec)
    prompt = historical_prompt(spec["question"], len(images)) if which == "original" else v9_prompt(spec["question"], len(images))
    model, processor = load_model()
    inputs = processor(images=images, text=prompt, return_tensors="pt")
    inputs = {key: value.to("cuda") if torch.is_tensor(value) else value for key, value in inputs.items()}
    image_id = int(model.config.image_token_index)
    ids = inputs["input_ids"]
    gen = model.generation_config
    with torch.inference_mode():
        logits = model(**inputs, use_cache=False).logits[0, -1].float()
        order = torch.argsort(logits, descending=True)
        output = model.generate(**inputs, do_sample=False, max_new_tokens=16, use_cache=True)
    new_ids = [int(token) for token in output[0, ids.shape[1]:].tolist()]
    text = processor.batch_decode(torch.tensor(new_ids).unsqueeze(0), skip_special_tokens=True)[0].strip()
    dr_id = processor.tokenizer.encode("Dr", add_special_tokens=False)[0]
    eos_id = 151645
    top = []
    for index in order[:50].tolist():
        top.append({
            "id": int(index),
            "token": processor.decode([int(index)]),
            "logit": float(logits[int(index)]),
        })
    report = {
        "path": which,
        "started_utc": started,
        "seconds": round(time.time() - t0, 3),
        "command": [PYTHON, str(HERE / "audit_one.py"), which],
        "gpu": "CUDA_VISIBLE_DEVICES=0",
        "versions": package_versions(),
        "git": git_state(),
        "prompt": prompt,
        "prompt_sha256": sha_bytes(prompt.encode()),
        "input_ids_shape": list(ids.shape),
        "input_ids_sha256": sha_tensor(ids),
        "attention_mask_shape": list(inputs["attention_mask"].shape),
        "attention_mask_sha256": sha_tensor(inputs["attention_mask"]),
        "pixel_values_shape": list(inputs["pixel_values"].shape),
        "pixel_values_dtype": str(inputs["pixel_values"].dtype),
        "pixel_values_sha256": sha_tensor(inputs["pixel_values"]),
        "image_sizes": inputs["image_sizes"].detach().cpu().tolist(),
        "image_sizes_sha256": sha_tensor(inputs["image_sizes"]),
        "batch_num_images": inputs["batch_num_images"].detach().cpu().tolist(),
        "image_token_count": int((ids[0] == image_id).sum()),
        "page_sha256": [sha_image(image) for image in images],
        "special_tokens": {
            "image_token_id": image_id,
            "eos_token_id": jsonable(model.config.eos_token_id),
            "pad_token_id": jsonable(model.config.pad_token_id),
            "bos_token_id": jsonable(getattr(model.config, "bos_token_id", None)),
            "generation_eos_token_id": jsonable(gen.eos_token_id),
            "generation_pad_token_id": jsonable(gen.pad_token_id),
        },
        "generation_config": {
            "max_new_tokens_call": 16,
            "do_sample_call": False,
            "temperature": jsonable(gen.temperature),
            "top_p": jsonable(gen.top_p),
            "num_beams": jsonable(gen.num_beams),
            "use_cache_call": True,
            "cache_implementation": jsonable(getattr(gen, "cache_implementation", None)),
        },
        "model": {
            "dtype": str(next(model.parameters()).dtype),
            "device": str(next(model.parameters()).device),
            "training": bool(model.training),
        },
        "top50": top,
        "eos_logit": float(logits[eos_id]),
        "eos_rank": int((order == eos_id).nonzero()[0]),
        "dr_id": int(dr_id),
        "dr_logit": float(logits[dr_id]),
        "dr_rank": int((order == dr_id).nonzero()[0]),
        "generated_ids": new_ids,
        "generated_text": text,
        "matches_frozen": text == FROZEN_TEXT,
        "fastv_modules": sorted(name for name in sys.modules if "fastv" in name.lower()),
    }
    dest = HERE / f"path_{which}.json"
    dest.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"path": which, "text": text, "match": report["matches_frozen"], "dr_rank": report["dr_rank"], "eos_rank": report["eos_rank"]}), flush=True)


if __name__ == "__main__":
    main()
