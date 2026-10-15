"""Clean admission for one id. One process writes only that id's JSON."""

from __future__ import annotations

import json
import sys

from PIL import Image

from common import (
    ANLS_CLEAN,
    FROZEN_1025_IDS,
    GLOBAL_SEED,
    HERE,
    PAGE_TOKENS,
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
from spatial_pool import PAGE_TOKENS as POOL_PAGE
from spatial_pool import RATIOS
from surrogate import plans_for, surrogate_generate


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
    full_ok = (
        full_text == spec["clean_full_pred"]
        and replay_text == spec["clean_full_pred"]
        and full_tokens == replay_tokens
        and (sample_id != "1025" or full_tokens == FROZEN_1025_IDS)
    )
    rows = []
    jobs = [("BasePreservingAdaptivePool", ratio, 0) for ratio in RATIOS]
    jobs += [("BasePreservingJitteredPool", ratio, offset) for ratio in RATIOS for offset in (0, 1, 2)]
    for family, ratio, offset_id in jobs:
        plans = plans_for(family, ratio, sample_id, 0, offset_id, len(images), GLOBAL_SEED)
        first = surrogate_generate(model, processor, inputs, inputs["pixel_values"], plans)
        second_plans = plans_for(family, ratio, sample_id, 0, offset_id, len(images), GLOBAL_SEED)
        second = surrogate_generate(model, processor, replay_inputs, replay_inputs["pixel_values"], second_plans)
        anls = anls_score(first["text"], answers_of(spec))
        actual = [count / POOL_PAGE for count in first["page_out"]]
        row = {
            "family": family,
            "not_an_official_compressor": True,
            "seed": GLOBAL_SEED,
            "step": 0,
            "offset_id": offset_id,
            "retention_ratio": ratio,
            "page_in": PAGE_TOKENS,
            "page_out": first["page_out"],
            "actual_ratio": actual,
            "page_boundaries": len(first["page_out"]),
            "shifts": [(plan.shift_row, plan.shift_col) for plan in plans],
            "rule_hash": [plan.rule_hash() for plan in plans],
            "hidden_shape": first["hidden_shape"],
            "text": first["text"],
            "tokens": first["tokens"],
            "replay_text": second["text"],
            "replay_tokens": second["tokens"],
            "anls": anls,
            "token_count_changed": all(count < PAGE_TOKENS for count in first["page_out"]),
            "nonempty": output_valid(first["text"]),
            "finite": first["finite"],
            "replay_match": first["text"] == second["text"] and first["tokens"] == second["tokens"],
            "full_ok": full_ok,
        }
        row["pass"] = bool(
            row["nonempty"] and row["finite"] and row["token_count_changed"] and row["replay_match"]
            and anls >= ANLS_CLEAN and full_ok
        )
        rows.append(row)
        print(json.dumps({"sample": sample_id, "family": family, "ratio": ratio, "offset": offset_id, "pass": row["pass"], "anls": anls, "text": first["text"]}), flush=True)
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
