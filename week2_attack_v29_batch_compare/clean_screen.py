"""Clean Full/FastV admission for the six frozen samples; no target is selected here."""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "week1_mechanism"), str(ROOT / "gate_experiment")]

from metrics_task import anls_score  # noqa: E402
from run_eval import decode  # noqa: E402
from week2_attack_v11_objective_optimizer_diagnostic import run_v11 as base  # noqa: E402
from week2_attack_v13_branch_reachability_probe import run_branch_probe as probe  # noqa: E402
from week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation import run_v7d as v7d  # noqa: E402
from week2_attack_v29_batch_compare.load_samples import load_item  # noqa: E402
from week2_attack_v4_answer_aligned.loss import output_valid  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--nshards", type=int, default=4)
    args = parser.parse_args()
    probe.configure_determinism()
    torch.cuda.set_device(args.device)
    model, processor = base.load_model_frozen()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    frozen = json.loads((OUT / "sample_specs.json").read_text())["samples_in_frozen_order"]
    rows = [row for i, row in enumerate(frozen) if i % args.nshards == args.shard]
    out_dir = OUT / "clean_screen"
    out_dir.mkdir(parents=True, exist_ok=True)
    for spec in rows:
        started = time.perf_counter()
        item = load_item(spec["sample_id"])
        state = v7d.prepare_item(model, processor, item)
        fast, kept = v7d.generate_tokens(model, state["inputs"], state["inputs"]["pixel_values"], state["spans"], 0.25)
        full2, _ = v7d.generate_tokens(model, state["inputs"], state["inputs"]["pixel_values"], state["spans"], None)
        fast2, kept2 = v7d.generate_tokens(model, state["inputs"], state["inputs"]["pixel_values"], state["spans"], 0.25)
        full_text = state["clean_text"]
        fast_text = decode(processor, torch.tensor(fast)).strip()
        full_tokens = [int(x) for x in state["clean_tokens"]]
        record = {
            "sample_id": spec["sample_id"], "question": spec["question"], "answers": spec["answers"],
            "clean_full_text": full_text, "clean_full_tokens": full_tokens,
            "clean_fastv_text": fast_text, "clean_fastv_tokens": [int(x) for x in fast],
            "clean_full_anls": float(anls_score(full_text, spec["answers"])),
            "clean_fastv_anls": float(anls_score(fast_text, spec["answers"])),
            "full_valid": bool(output_valid(full_text)),
            "fastv_valid": bool(output_valid(fast_text)),
            "full_same_process_repeat": bool(full_tokens == [int(x) for x in full2]),
            "fastv_same_process_repeat": bool(fast == fast2 and kept == kept2),
            "fastv_k": len(kept), "expected_k": spec["expected_k"],
            "keep_hash": probe.keep_hash(kept), "keep_hash_repeat": probe.keep_hash(kept2),
            "page_sha256": spec["page_sha256"], "layout_M": spec["M"],
            "aux_idx": spec["aux_idx"], "evidence_idx": spec["evidence_idx"],
            "clean_gate_eligible": bool(
                output_valid(full_text) and output_valid(fast_text)
                and anls_score(full_text, spec["answers"]) >= 0.5
                and anls_score(fast_text, spec["answers"]) >= 0.5
                and len(kept) == spec["expected_k"] and kept == kept2 and fast == fast2
                and full_tokens == [int(x) for x in full2]
            ),
            "replay_scope": "IN_PROCESS_REPEAT_ONLY; independent process reserved for final PNG evaluator",
            "seconds": round(time.perf_counter() - started, 3),
        }
        dest = out_dir / f"{spec['sample_id']}.json"
        dest.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps({"sample": spec["sample_id"], "eligible": record["clean_gate_eligible"],
                          "full_anls": record["clean_full_anls"], "fast_anls": record["clean_fastv_anls"],
                          "K": record["fastv_k"], "seconds": record["seconds"]}), flush=True)
        del state, item
        gc.collect()
        torch.cuda.empty_cache()
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
