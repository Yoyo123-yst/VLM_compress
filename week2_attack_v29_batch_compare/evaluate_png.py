"""Separate-process final PNG evaluator (fresh model load per invocation)."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
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


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize(text: str) -> str:
    return " ".join("".join(ch.lower() for ch in text if ch.isalnum()).split())


def run_eval(sample: str, method: str, device: int, run_root: Path) -> dict:
    torch.cuda.set_device(device)
    item = load_item(sample)
    model, processor = base.load_model_frozen()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    state = v7d.prepare_item(model, processor, item)
    baseline_fast, baseline_kept = v7d.generate_tokens(
        model, state["inputs"], state["inputs"]["pixel_values"], state["spans"], 0.25
    )
    clean_record = json.loads((OUT / "clean_screen" / f"{sample}.json").read_text())
    if ([int(x) for x in state["clean_tokens"]] != clean_record["clean_full_tokens"]
            or baseline_fast != clean_record["clean_fastv_tokens"]
            or probe.keep_hash(baseline_kept) != clean_record["keep_hash"]):
        raise RuntimeError(f"CLEAN_BASELINE_DIVERGED in independent evaluator: {sample}")

    run = json.loads((run_root / method / sample / "result.json").read_text())
    accepted = int(run["accepted"])
    if accepted == 0:
        result = {"status": "NO_VALID_SEALED_CANDIDATE", "sample_id": sample, "method": method,
                  "attack_png": None, "independent_process": True}
        return result
    png_path = run_root / method / sample / f"accepted_{accepted:03d}.png"
    raw = png_path.read_bytes()
    image = Image.open(png_path).convert("RGB")
    base_array = np.asarray(item["images"][item["aux_idx"]].convert("RGB"), dtype=np.int16)
    attack_array = np.asarray(image, dtype=np.int16)
    linf = int(np.abs(attack_array - base_array).max())

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
    clean_full_tokens = [int(x) for x in state["clean_tokens"]]
    full_text_exact = bool(full_text == state["clean_text"])
    full_token_exact = bool(full1 == clean_full_tokens)
    replay = bool(full1 == full2 and fast1 == fast2 and kept1 == kept2)
    answers = item["meta"]["answers"]
    target = None
    if method != "native_fdp":
        frozen = json.loads((OUT / "frozen_targets.json").read_text())["samples"][sample]
        target = frozen["wrong_target_text"]
    result = {
        "status": "EVALUATED_INDEPENDENT_PROCESS",
        "sample_id": sample, "method": method,
        "independent_process": True,
        "png_path": str(png_path.resolve().relative_to(ROOT)), "png_sha256": sha_bytes(raw),
        "uint8_linf": linf,
        "clean_full_text": state["clean_text"], "full_text": full_text,
        "full_text_exact": full_text_exact, "full_token_exact": full_token_exact,
        "same_evaluator_process_repeat_exact": replay,
        "fastv_text": fast_text,
        "fastv_anls": float(anls_score(fast_text, answers)),
        "fastv_valid": bool(output_valid(fast_text)),
        "k": len(kept1), "expected_k": item["spec"]["expected_k"],
        "keep_hash": probe.keep_hash(kept1),
        "dataset_answers": answers,
        "target_text": target,
        "target_hit_normalized_exact": bool(target is not None and normalize(fast_text) == normalize(target)),
        "content_status": "PENDING_QUESTION_SPECIFIC_REVIEW",
        "semantic_review_required": True,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True)
    parser.add_argument("--method", choices=("native_fdp", "complete_target", "complete_front"), required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--run-root", default=str(OUT / "attacks"), help="root containing method/sample/result.json")
    args = parser.parse_args()
    run_root = Path(args.run_root)
    result = run_eval(args.sample, args.method, args.device, run_root)
    eval_dir = "independent_eval_smoke" if "smoke" in str(run_root) else "independent_eval"
    path = OUT / eval_dir / args.method / f"{args.sample}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"sample": args.sample, "method": args.method, "status": result["status"],
                      "full_exact": result.get("full_text_exact"), "fastv": result.get("fastv_text"),
                      "target_hit": result.get("target_hit_normalized_exact")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
