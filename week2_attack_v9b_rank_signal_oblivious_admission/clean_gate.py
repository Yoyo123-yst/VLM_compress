"""Clean-image gate for one admission id. Writes only that id's JSON."""

from __future__ import annotations

import json
import sys

import torch
from PIL import Image

from spatial_rule import plan_for
from common import (
    ANLS_CLEAN,
    DIAGNOSTIC_RATIOS,
    GLOBAL_SEED,
    HERE,
    answers_of,
    anls_score,
    fastv_modules,
    generate_tokens,
    load_images,
    load_model,
    manifest_samples,
    output_valid,
    prepare,
    sha_image,
)
from guard import assert_clean, install_guard
from surrogate import surrogate_generate


def main() -> None:
    install_guard()
    sample_id = sys.argv[1]
    spec = next(row for row in manifest_samples() if row["sample_id"] == sample_id)
    images = load_images(spec)
    folder = HERE / "clean_pages" / sample_id
    folder.mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(images):
        image.save(folder / f"page_{index}.png")
    reloaded = [Image.open(folder / f"page_{index}.png").convert("RGB") for index in range(len(images))]
    model, processor = load_model()
    inputs = prepare(processor, images, spec["question"])
    full_tokens, full_text = generate_tokens(model, processor, inputs)
    replay_inputs = prepare(processor, reloaded, spec["question"])
    replay_tokens, replay_text = generate_tokens(model, processor, replay_inputs)
    pixels = inputs["pixel_values"]
    rows = []
    for family in ("SpatialStridedDrop", "LocalSpatialMerge"):
        for ratio in DIAGNOSTIC_RATIOS:
            plan = plan_for(family, ratio, sample_id, 0, GLOBAL_SEED)
            first = surrogate_generate(model, processor, inputs, pixels, plan)
            second = surrogate_generate(model, processor, replay_inputs, replay_inputs["pixel_values"], plan)
            anls = anls_score(first["text"], answers_of(spec))
            row = {
                "family": family,
                "seed": GLOBAL_SEED,
                "step": 0,
                "retention_ratio": ratio,
                "offset": plan.offset,
                "page_in": plan.page_in,
                "page_out": first["page_out"],
                "page_boundaries": len(first["page_out"]),
                "rule_hash": plan.rule_hash(),
                "hidden_shape": first["hidden_shape"],
                "text": first["text"],
                "replay_text": second["text"],
                "anls": anls,
                "token_count_changed": all(count < plan.page_in for count in first["page_out"]),
                "nonempty": output_valid(first["text"]),
                "finite": first["finite"],
                "replay_match": first["text"] == second["text"] and first["tokens"] == second["tokens"],
                "full_exact": full_text == spec["clean_full_pred"] and replay_text == spec["clean_full_pred"],
                "full_replay_tokens": full_tokens == replay_tokens,
            }
            row["pass"] = bool(
                row["nonempty"] and row["finite"] and row["token_count_changed"] and row["replay_match"]
                and anls >= ANLS_CLEAN and row["full_exact"] and row["full_replay_tokens"]
            )
            rows.append(row)
            print(json.dumps({"sample": sample_id, "family": family, "ratio": ratio, "pass": row["pass"], "anls": anls, "text": first["text"]}), flush=True)
    dest = HERE / "clean_gate_parts" / f"{sample_id}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({
        "sample_id": sample_id,
        "full_text": full_text,
        "full_tokens": full_tokens,
        "page_sha256": [sha_image(image) for image in reloaded],
        "rows": rows,
        "fastv_access_count": 0,
        "fastv_modules": fastv_modules(),
    }, indent=2) + "\n")
    assert_clean("clean-gate")


if __name__ == "__main__":
    main()
