"""Token/keep-set causal diagnostic for the 60700 recovery observation.

This is read-only: it loads two already sealed PNGs, replays real Full/FastV,
and evaluates teacher-forced FastV logits under four sequence/keep-set pairs.
It does not optimize pixels and cannot be reported as an attack result.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
HERE = ROOT / "week2_attack_v17_token_causal_diagnostic"
V13 = ROOT / "week2_attack_v13_branch_reachability_probe"
sys.path[:0] = [
    str(ROOT), str(ROOT / "week1_mechanism"), str(ROOT / "gate_experiment"),
    str(ROOT / "week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation"),
    str(V13),
]

import run_branch_probe as probe  # noqa: E402
from metrics_task import anls_score  # noqa: E402
from week2_attack_v4_answer_aligned.common import keep_hash  # noqa: E402
from week2_attack_v4_answer_aligned.forward import keep_index_from_kept  # noqa: E402
from week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation import run_v7d as v7d  # noqa: E402
from week2_attack_v11_objective_optimizer_diagnostic import run_v11 as base  # noqa: E402
from week2_attack_v7b_native_cache_margin_admission.native_cache_margin import (  # noqa: E402
    hard_margin_row,
    soft_margin_row,
)

SAMPLE = "60700"
RATIO = 0.25
REFERENCE = [4406, 16680, 151645]  # PPSB at the starting error PNG
IMAGES = {
    "start_error": V13 / "runs/state_retention_v1/60700/AWARE_FREEZE/step_011.png",
    "recovery_step1": HERE.parent / "week2_attack_v16_corrected_direction_audit/60700_attack_only/step_001.png",
}


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pieces(processor, ids):
    tok = processor.tokenizer
    return [tok.decode([int(x)], skip_special_tokens=False) for x in ids]


def margins(rows, ids):
    out = []
    for row, token in zip(rows, ids):
        out.append({
            "token_id": int(token),
            "hard": float(hard_margin_row(row, int(token)).detach().cpu()),
            "soft": float(soft_margin_row(row, int(token)).detach().cpu()),
        })
    return out


def evaluate_pair(model, inputs, spans, aux_idx, page, seq, kept):
    pixels = v7d.pixels_from_page(inputs, aux_idx, page)
    keep_index = keep_index_from_kept(
        kept, spans, int(inputs["input_ids"].shape[1]), page.device
    )
    with torch.no_grad():
        rows = v7d.fastv_rows(model, inputs, pixels, list(seq), keep_index)
        return margins(rows, seq)


def main():
    probe.configure_determinism()
    model, processor = base.load_model_frozen()
    model.eval()
    item = v7d.load_dev(SAMPLE)
    clean_inputs = v7d.prepare(processor, item["images"], item["meta"]["question"])
    spans = v7d.spans_of(clean_inputs, int(model.config.image_token_index), len(item["images"]))
    clean_tokens, _ = v7d.generate_tokens(
        model, clean_inputs, clean_inputs["pixel_values"], spans, None
    )
    state = {"clean_tokens": clean_tokens,
             "clean_text": v7d.decode(processor, torch.tensor(clean_tokens)).strip()}
    records = {}
    start_kept = None
    for label, path in IMAGES.items():
        image = Image.open(path).convert("RGB")
        attacked = list(item["images"])
        attacked[item["aux_idx"]] = image
        inputs = v7d.prepare(processor, attacked, item["meta"]["question"])
        spans_local = v7d.spans_of(inputs, int(model.config.image_token_index), len(attacked))
        page = v7d.page_to_float(image).to("cuda")
        full = base.full_exact_check(model, processor, item, state, image)
        with torch.no_grad():
            actual, kept = v7d.generate_tokens(model, inputs, inputs["pixel_values"], spans_local, RATIO)
        actual_text = v7d.decode(processor, torch.tensor(actual)).strip()
        if start_kept is None:
            start_kept = list(kept)
        pairs = {
            "generated_actual_keep": evaluate_pair(model, inputs, spans_local, item["aux_idx"], page, actual, kept),
            "reference_actual_keep": evaluate_pair(model, inputs, spans_local, item["aux_idx"], page, REFERENCE, kept),
            "reference_start_keep": evaluate_pair(model, inputs, spans_local, item["aux_idx"], page, REFERENCE, start_kept),
            "generated_start_keep": evaluate_pair(model, inputs, spans_local, item["aux_idx"], page, actual, start_kept),
        }
        records[label] = {
            "path": str(path.relative_to(ROOT)),
            "png_sha256": file_sha(path),
            "full": full,
            "fast_text": actual_text,
            "fast_tokens": [int(x) for x in actual],
            "fast_pieces": pieces(processor, actual),
            "fast_anls": float(anls_score(actual_text, list(item["meta"]["answers"]))),
            "keep_hash": keep_hash(kept),
            "keep_k": len(kept),
            "kept": [int(x) for x in kept],
            "pairs_teacher_forced": pairs,
        }
        print(json.dumps({"image": label, "fast_text": actual_text,
                          "tokens": actual, "keep_hash": keep_hash(kept),
                          "full": full}, ensure_ascii=False), flush=True)
    HERE.mkdir(parents=True, exist_ok=True)
    report = {
        "decision_scope": "diagnostic_only",
        "sample": SAMPLE, "ratio": RATIO, "reference": REFERENCE,
        "reference_pieces": pieces(processor, REFERENCE),
        "start_keep_hash": keep_hash(start_kept),
        "records": records,
        "note": "Teacher-forced margins under four sequence/keep-set combinations; not endpoint evidence.",
    }
    (HERE / "causal_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
