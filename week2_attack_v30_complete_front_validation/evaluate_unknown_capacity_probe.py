"""Fresh-process evaluation of the isolated 5786 unknown-capacity probe."""
from __future__ import annotations

import hashlib
import json
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "week1_mechanism"), str(ROOT / "gate_experiment")]

from metrics_task import anls_score  # noqa: E402
from run_eval import decode  # noqa: E402
from week2_attack_v11_objective_optimizer_diagnostic import run_v11 as base  # noqa: E402
from week2_attack_v13_branch_reachability_probe import run_branch_probe as probe  # noqa: E402
from week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation import run_v7d as v7d  # noqa: E402
from week2_attack_v4_answer_aligned.loss import output_valid  # noqa: E402
from week2_attack_v29_batch_compare.load_samples import load_item  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--out-tag", default="unknown_capacity_probe")
    args = parser.parse_args()
    device = args.device
    torch.cuda.set_device(device)
    item = load_item("5786")
    model, processor = base.load_model_frozen()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    state = v7d.prepare_item(model, processor, item)
    full_clean = [int(x) for x in state["clean_tokens"]]
    fast_clean, kept_clean = v7d.generate_tokens(model, state["inputs"], state["inputs"]["pixel_values"], state["spans"], 0.25)
    run_root = OUT / args.out_tag / "complete_target" / "5786"
    run = json.loads((run_root / "probe_result.json").read_text())
    accepted = int(run["accepted"])
    png_path = run_root / f"accepted_{accepted:03d}.png"
    image = Image.open(png_path).convert("RGB")

    def infer_once():
        images = list(item["images"])
        images[item["aux_idx"]] = image
        inputs = v7d.prepare(processor, images, item["meta"]["question"])
        spans = v7d.spans_of(inputs, int(model.config.image_token_index), len(images))
        full, _ = v7d.generate_tokens(model, inputs, inputs["pixel_values"], spans, None)
        fast, kept = v7d.generate_tokens(model, inputs, inputs["pixel_values"], spans, 0.25)
        return full, fast, kept

    full1, fast1, kept1 = infer_once()
    full2, fast2, kept2 = infer_once()
    full_text = decode(processor, torch.tensor(full1)).strip()
    fast_text = decode(processor, torch.tensor(fast1)).strip()
    base_array = np.asarray(item["images"][item["aux_idx"]].convert("RGB"), dtype=np.int16)
    attack_array = np.asarray(image, dtype=np.int16)
    result = {
        "status": "INDEPENDENT_PROBE_EVALUATED",
        "sample_id": "5786",
        "png": str(png_path.resolve().relative_to(ROOT)),
        "png_sha256": hashlib.sha256(png_path.read_bytes()).hexdigest(),
        "uint8_linf": int(np.abs(attack_array - base_array).max()),
        "clean_full_text": state["clean_text"],
        "full_text": full_text,
        "full_text_exact": bool(full_text == state["clean_text"]),
        "full_token_exact": bool(full1 == full_clean),
        "replay_exact": bool(full1 == full2 and fast1 == fast2 and kept1 == kept2),
        "clean_fast_text": decode(processor, torch.tensor(fast_clean)).strip(),
        "fastv_text": fast_text,
        "fastv_anls": float(anls_score(fast_text, item["meta"]["answers"])),
        "fastv_valid": bool(output_valid(fast_text)),
        "target_text": "670",
        "target_hit": bool(fast_text.strip() == "670"),
        "keep_hash": probe.keep_hash(kept1),
        "clean_keep_hash": probe.keep_hash(kept_clean),
        "k": len(kept1),
        "expected_k": int(item["spec"]["expected_k"]),
        "target_semantic_status": "VALID_COMPLETE_WRONG_NUMBER_UNKNOWN_FULL_CAPACITY",
        "not_a_method_result": True,
    }
    out = OUT / args.out_tag / "independent_eval.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
