"""Clean-only diverse generation for target admission.

This script never reads attacked PNGs and never updates pixels. It is only a
candidate-discovery aid; every candidate remains pending semantic review.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "week2_attack_v25_full_answer_competition_pilot"
sys.path[:0] = [str(ROOT), str(ROOT / "week1_mechanism"), str(ROOT / "gate_experiment")]

from adapters.runtime import clear_hooks, install
from run_eval import decode
from week2_attack_v13_branch_reachability_probe import run_branch_probe as probe
from week2_attack_v11_objective_optimizer_diagnostic import run_v11 as base
from week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation import run_v7d as v7d
from week2_attack_v7b_native_cache_margin_admission.native_cache_margin import MAX_NEW_TOKENS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = int(args.device)
    probe.configure_determinism()
    torch.cuda.set_device(device)
    model, processor = base.load_model_frozen()
    model.eval()
    item = v7d.load_dev(args.sample)
    state = v7d.prepare_item(model, processor, item)
    fast = probe.fast_clean_reference(model, processor, item, state)
    local = dict(state["inputs"])
    local["pixel_values"] = state["inputs"]["pixel_values"].detach()
    seen = {}
    prompt_len = int(local["input_ids"].shape[1])
    special = set(getattr(processor.tokenizer, "all_special_ids", []))
    for seed in range(2026101500, 2026101520):
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        clear_hooks(model)
        hs = install(model, "fastv", probe.RATIO, int(model.config.image_token_index))
        hs.spans = state["spans"]
        try:
            with torch.inference_mode():
                out = model.generate(
                    **local,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.92,
                    max_new_tokens=MAX_NEW_TOKENS,
                    use_cache=True,
                    return_dict_in_generate=True,
                    output_scores=False,
                )
            toks = [int(x) for x in out.sequences[0, prompt_len:].detach().cpu().tolist()]
            while toks and toks[-1] in special:
                break
            text = decode(processor, torch.tensor(toks)).strip()
            key = " ".join(text.lower().split())
            if key and key not in seen:
                seen[key] = {"seed": seed, "tokens": toks, "text": text, "keep_hash": probe.keep_hash(list(hs.kept_visual)), "k": len(hs.kept_visual)}
        finally:
            clear_hooks(model)
    result = {
        "sample_id": args.sample,
        "ratio": probe.RATIO,
        "clean_fastv_text": fast["text"],
        "clean_fastv_tokens": fast["tokens"],
        "clean_keep_hash": fast["keep_hash"],
        "candidate_selection_uses_labels": False,
        "attacked_png_used": False,
        "status": "DIVERSE_CLEAN_CANDIDATES_PENDING_SEMANTIC_REVIEW",
        "candidates": list(seen.values()),
        "seed_range": [2026101500, 2026101519],
    }
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"sample": args.sample, "count": len(result["candidates"]), "out": str(p)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
