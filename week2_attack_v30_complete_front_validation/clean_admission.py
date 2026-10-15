"""Fresh-process clean admission for the frozen v30 validation cohort."""
from __future__ import annotations

import argparse
import gc
import hashlib
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


def sha_tokens(values: list[int]) -> str:
    return hashlib.sha256(json.dumps(values).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--nshards", type=int, default=4)
    args = parser.parse_args()
    frozen = json.loads((OUT / "cohort_freeze.json").read_text())
    samples = [row for i, row in enumerate(frozen["samples"]) if i % args.nshards == args.shard]
    torch.cuda.set_device(args.device)
    probe.configure_determinism()
    model, processor = base.load_model_frozen()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    out_dir = OUT / "clean_admission"
    out_dir.mkdir(parents=True, exist_ok=True)
    for row in samples:
        started = time.perf_counter()
        item = load_item(row["sample_id"])
        state = v7d.prepare_item(model, processor, item)
        fast1, kept1 = v7d.generate_tokens(model, state["inputs"], state["inputs"]["pixel_values"], state["spans"], 0.25)
        full2, _ = v7d.generate_tokens(model, state["inputs"], state["inputs"]["pixel_values"], state["spans"], None)
        fast2, kept2 = v7d.generate_tokens(model, state["inputs"], state["inputs"]["pixel_values"], state["spans"], 0.25)
        full_tokens = [int(x) for x in state["clean_tokens"]]
        full_text = state["clean_text"]
        fast_text = decode(processor, torch.tensor(fast1)).strip()
        answers = list(item["meta"]["answers"])
        record = {
            "status": "CLEAN_ADMISSION_COMPLETE",
            "sample_id": row["sample_id"],
            "question": item["meta"]["question"],
            "dataset_answers": answers,
            "full_text": full_text,
            "full_tokens_sha256": sha_tokens(full_tokens),
            "fastv_text": fast_text,
            "fastv_tokens_sha256": sha_tokens([int(x) for x in fast1]),
            "full_anls": float(anls_score(full_text, answers)),
            "fastv_anls": float(anls_score(fast_text, answers)),
            "full_valid": bool(output_valid(full_text)),
            "fastv_valid": bool(output_valid(fast_text)),
            "full_repeat_exact": bool(full_tokens == [int(x) for x in full2]),
            "fastv_repeat_exact": bool(fast1 == fast2 and kept1 == kept2),
            "K": len(kept1),
            "expected_K": int(item["spec"]["expected_k"]),
            "keep_hash": probe.keep_hash(kept1),
            "keep_hash_repeat": probe.keep_hash(kept2),
            "clean_gate_eligible": bool(
                output_valid(full_text) and output_valid(fast_text)
                and anls_score(full_text, answers) >= 0.5 and anls_score(fast_text, answers) >= 0.5
                and full_tokens == [int(x) for x in full2]
                and fast1 == fast2 and kept1 == kept2
                and len(kept1) == int(item["spec"]["expected_k"])
            ),
            "semantic_clean_review": "PENDING_MANUAL_REVIEW",
            "seconds": round(time.perf_counter() - started, 3),
        }
        (out_dir / f"{row['sample_id']}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"sample": row["sample_id"], "eligible": record["clean_gate_eligible"], "fastv": fast_text, "seconds": record["seconds"]}), flush=True)
        del item, state
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
