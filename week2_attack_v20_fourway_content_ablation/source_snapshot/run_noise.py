"""Matched random-noise controls. No gradient steps."""

from __future__ import annotations

import sys

sys.path[:0] = ["/home/yuesiting/autodl-tmp/multimodal_attack_project"]

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from week2_attack_v20_fourway_content_ablation import runtime_utils as rt
from week2_attack_v20_fourway_content_ablation.evaluate_independent import score_image
from week2_attack_v20_fourway_content_ablation.protocol import EPSILON_UINT8, NOISE_ATTEMPTS, NOISE_REPLICATES, SEED
from week2_attack_v20_fourway_content_ablation.semantics import aliases_for

HERE = rt.HERE


def matched_noise(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    array = np.asarray(image, dtype=np.int16)
    noise = rng.integers(-EPSILON_UINT8, EPSILON_UINT8 + 1, size=array.shape, dtype=np.int16)
    # Keep the draw inside the same uint8 Linf ball and force the bound to be met.
    flat = noise.reshape(-1)
    if int(np.max(np.abs(flat))) == 0:
        flat[0] = EPSILON_UINT8
    attacked = np.clip(array + noise, 0, 255).astype(np.uint8)
    gap = np.clip(attacked.astype(np.int16) - array, -EPSILON_UINT8, EPSILON_UINT8)
    attacked = np.clip(array + gap, 0, 255).astype(np.uint8)
    return Image.fromarray(attacked, mode="RGB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    manifest = json.loads((HERE / "semantic_manifest.json").read_text())
    by_id = {row["sample_id"]: row for row in manifest["samples"]}
    rt.configure()
    model, processor = rt.v7d.load_model_frozen()
    out = Path(args.out)
    for offset, sample_id in enumerate([item.strip() for item in args.samples.split(",") if item.strip()]):
        item = rt.v7d.load_dev(sample_id)
        state = rt.prepare_clean_state(model, processor, item)
        aliases = by_id[sample_id]["normalized_aliases"]
        if aliases != aliases_for(sample_id, by_id[sample_id]["gold_answers"]):
            raise RuntimeError("semantic manifest aliases drifted")
        rng = np.random.default_rng(SEED + offset + 17)
        clean = item["images"][item["aux_idx"]].convert("RGB")
        kept = []
        attempts = []
        for attempt in range(NOISE_ATTEMPTS):
            if len(kept) >= NOISE_REPLICATES:
                break
            image = matched_noise(clean, rng)
            path = out / sample_id / f"attempt_{attempt:02d}.png"
            rt.save_png(image, path)
            guard = rt.full_guard(model, processor, item, state, rt.reload_png(path))
            linf = rt.uint8_linf(clean, rt.reload_png(path))
            passed = bool(guard["full_text_exact"] and guard["full_token_exact"] and guard["full_replay"] and linf <= EPSILON_UINT8 and linf > 0)
            attempts.append({"attempt": attempt, "passed_full_gate": passed, "uint8_linf": linf, "png_sha256": rt.sha_file(path)})
            if not passed:
                path.unlink(missing_ok=True)
                continue
            scored = score_image(model, processor, item, state, path, aliases, sample_id)
            scored["replicate"] = len(kept)
            kept.append(scored)
            print(json.dumps({"sample": sample_id, "replicate": scored["replicate"], "r25": scored["ratios"]["0.25"]["semantic_status"]}), flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        rt.write_json(
            out / sample_id / "noise.json",
            {
                "sample_id": sample_id,
                "requested_replicates": NOISE_REPLICATES,
                "found_replicates": len(kept),
                "status": "DONE" if len(kept) >= NOISE_REPLICATES else "NOT_MEASURED",
                "attempts": attempts,
                "replicates": kept,
            },
        )


if __name__ == "__main__":
    main()
