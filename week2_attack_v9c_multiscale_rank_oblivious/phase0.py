"""Phase 0. Historical build_prompt, then a fresh process replays the saved pages."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from PIL import Image

from common import (
    ADMISSION_IDS,
    FROZEN_1025_IDS,
    FROZEN_1025_TEXT,
    HERE,
    PAGE_TOKENS,
    fastv_modules,
    generate_tokens,
    image_spans,
    load_build_prompt,
    load_images,
    load_model,
    manifest_samples,
    prepare,
    sha_bytes,
    sha_image,
    sha_tensor,
)
from guard import assert_clean, install_guard


def save_pages(spec, images) -> None:
    dest = HERE / "baseline_pages" / spec["sample_id"]
    dest.mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(images):
        image.save(dest / f"page_{index}.png")


def run_generate() -> None:
    install_guard()
    build_prompt, prompt_hash, prompt_path = load_build_prompt()
    model, processor = load_model()
    image_id = int(model.config.image_token_index)
    rows = []
    for spec in manifest_samples():
        images = load_images(spec)
        inputs = prepare(processor, images, spec["question"])
        tokens, text = generate_tokens(model, processor, inputs)
        spans = image_spans(inputs["input_ids"], image_id, len(images))
        visual = sum(end - start for start, end in spans)
        row = {
            "sample_id": spec["sample_id"],
            "frozen_text": spec["clean_full_pred"],
            "text": text,
            "tokens": tokens,
            "text_match": text == spec["clean_full_pred"],
            "prompt": inputs["prompt"],
            "prompt_sha256": sha_bytes(inputs["prompt"].encode()),
            "build_prompt_sha256": prompt_hash,
            "build_prompt_path": prompt_path,
            "input_ids_shape": list(inputs["input_ids"].shape),
            "input_ids_sha256": sha_tensor(inputs["input_ids"]),
            "pixel_values_shape": list(inputs["pixel_values"].shape),
            "pixel_values_sha256": sha_tensor(inputs["pixel_values"]),
            "image_tokens": visual,
            "tokens_per_page": PAGE_TOKENS,
            "page_token_count_ok": visual == spec["M"] * PAGE_TOKENS,
            "page_sha256": [sha_image(image) for image in images],
            "first_token": tokens[0] if tokens else None,
        }
        if spec["sample_id"] == "1025":
            row["token_match"] = tokens == FROZEN_1025_IDS
            row["text_is_frozen_1025"] = text == FROZEN_1025_TEXT
            row["image_tokens_ok"] = visual == 3 * PAGE_TOKENS
            if not (row["text_match"] and row["token_match"] and row["first_token"] == 8847 and row["image_tokens_ok"]):
                (HERE / "baseline_first.json").write_text(json.dumps({"rows": rows + [row], "pass": False}, indent=2) + "\n")
                print(json.dumps({"sample": "1025", "pass": False, "text": text, "tokens": tokens}), flush=True)
                return
        save_pages(spec, images)
        rows.append(row)
        print(json.dumps({"sample": spec["sample_id"], "text_match": row["text_match"], "tokens": tokens}), flush=True)
    passed = all(row["text_match"] for row in rows) and len(rows) == len(ADMISSION_IDS)
    (HERE / "baseline_first.json").write_text(json.dumps({"rows": rows, "pass": passed, "fastv_modules": fastv_modules()}, indent=2) + "\n")
    assert_clean("phase0-generate")


def run_replay() -> None:
    install_guard()
    first = json.loads((HERE / "baseline_first.json").read_text())
    if not first.get("pass"):
        raise SystemExit("generate pass is false")
    model, processor = load_model()
    rows = []
    for saved in first["rows"]:
        spec = next(row for row in manifest_samples() if row["sample_id"] == saved["sample_id"])
        folder = HERE / "baseline_pages" / spec["sample_id"]
        images = [Image.open(folder / f"page_{index}.png").convert("RGB") for index in range(spec["M"])]
        inputs = prepare(processor, images, spec["question"])
        tokens, text = generate_tokens(model, processor, inputs)
        rows.append({
            "sample_id": spec["sample_id"],
            "text": text,
            "tokens": tokens,
            "matches_first": text == saved["text"] and tokens == saved["tokens"],
            "matches_frozen_text": text == spec["clean_full_pred"],
            "prompt_sha256": sha_bytes(inputs["prompt"].encode()),
            "input_ids_sha256": sha_tensor(inputs["input_ids"]),
            "pixel_values_sha256": sha_tensor(inputs["pixel_values"]),
        })
        print(json.dumps(rows[-1] | {"tokens": tokens}), flush=True)
    passed = all(row["matches_first"] and row["matches_frozen_text"] for row in rows)
    payload = {
        "status": "BASELINE_RECOVERED" if passed else "FAIL_BASELINE_REPLAY",
        "generate_rows": first["rows"],
        "replay_rows": rows,
        "fastv_modules": fastv_modules(),
    }
    (HERE / "baseline_recovery.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": payload["status"]}), flush=True)
    assert_clean("phase0-replay")


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False
    if sys.argv[1] == "generate":
        run_generate()
    elif sys.argv[1] == "replay":
        run_replay()
    else:
        raise SystemExit("usage: phase0.py generate|replay")
