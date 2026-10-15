"""Clean admission and the semantic manifest.

This process does not train and does not read attack outputs. The manifest it
writes is hashed before any four-way run starts.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path[:0] = ["/home/yuesiting/autodl-tmp/multimodal_attack_project"]

import argparse
import json

import torch

from week2_attack_v20_fourway_content_ablation import runtime_utils as rt
from week2_attack_v20_fourway_content_ablation.protocol import PRIMARY_SAMPLES, TRAIN_RATIO
from week2_attack_v20_fourway_content_ablation.selection import (
    assert_core_not_silent_noeos,
    control_id_set,
    core_content_selection,
    fixed_all_selection,
    fixed_noeos_selection,
)
from week2_attack_v20_fourway_content_ablation.semantics import (
    AMBIGUOUS_CASES,
    CORE_PHRASES,
    FAIL_RULES,
    KEEP_RULES,
    MANUAL_NOTES,
    SPAN_TYPE,
    STATUS_AUGMENTED,
    STATUS_PRESERVED,
    aliases_for,
    classify,
)

HERE = rt.HERE


def provenance(item: dict) -> tuple[bool, dict]:
    meta = item.get("meta") or {}
    record = {
        "sample_id": meta.get("sample_id"),
        "question": meta.get("question"),
        "answers": list(meta.get("answers") or []),
        "evidence_idx": item.get("evidence_idx"),
        "aux_idx": item.get("aux_idx"),
        "page_sha256": list(item.get("page_sha") or []),
        "order": list(item.get("order") or []),
    }
    present = all(record[key] not in (None, "", []) for key in ("sample_id", "question", "answers", "page_sha256"))
    present = present and record["evidence_idx"] is not None and record["aux_idx"] is not None
    present = present and record["aux_idx"] != record["evidence_idx"]
    present = present and bool(item.get("images"))
    return present, record


def selection_payload(selection) -> dict:
    return {
        "name": selection.name,
        "positions": list(selection.positions),
        "token_ids": list(selection.token_ids),
        "pieces": list(selection.pieces),
        "reason": selection.reason,
        "same_positions_as_noeos": selection.same_positions_as_noeos,
        "available": selection.available,
    }


def admit_one(model, processor, sample_id: str, out_dir: Path) -> tuple[dict, dict]:
    item = rt.v7d.load_dev(sample_id)
    ok, prov = provenance(item)
    gold = [str(answer) for answer in item["meta"]["answers"]]
    aliases = aliases_for(sample_id, gold)
    image = item["images"][item["aux_idx"]].convert("RGB")
    source_path = out_dir / "pages" / sample_id / "source.png"
    clean_path = out_dir / "pages" / sample_id / "clean.png"
    source_sha = rt.save_png(image, source_path)
    clean_sha = rt.save_png(image, clean_path)
    checks = {
        "provenance": ok,
        "full_valid": False,
        "full_replay": False,
        "fastv_valid": False,
        "fastv_content_correct": False,
        "core_span": False,
    }
    semantic_row = {
        "sample_id": sample_id,
        "gold_answers": gold,
        "normalized_aliases": aliases,
        "clean_full_text": None,
        "clean_fastv_text": None,
        "clean_fastv_token_ids": None,
        "core_spans": CORE_PHRASES[sample_id],
        "core_token_positions": [],
        "core_token_ids": [],
        "core_pieces": [],
        "span_type": SPAN_TYPE[sample_id],
        "content_keep_rule": KEEP_RULES[sample_id],
        "content_fail_rule": FAIL_RULES[sample_id],
        "ambiguous_cases": AMBIGUOUS_CASES[sample_id],
        "manual_review_note": MANUAL_NOTES[sample_id],
        "selection_b": None,
        "selection_c": None,
        "selection_d": None,
        "source_png_sha256": source_sha,
        "clean_png_sha256": clean_sha,
        "dataset_page_sha256": list(item.get("page_sha") or []),
    }
    status = "PROVENANCE_INVALID"
    detail = {}
    if ok:
        state = rt.prepare_clean_state(model, processor, item)
        guard = rt.full_guard(model, processor, item, state, image)
        fast = rt.fastv_generate(model, processor, item, image, TRAIN_RATIO)
        reference = list(state["clean_fast_tokens"])
        if reference != fast["tokens"]:
            status = "REPLAY_DIVERGED"
        else:
            label = classify(sample_id, fast["text"], aliases)
            controls = control_id_set(processor.tokenizer)
            selected_b = fixed_all_selection(processor.tokenizer, reference)
            selected_c = fixed_noeos_selection(processor.tokenizer, reference, controls)
            selected_d = core_content_selection(processor.tokenizer, reference, CORE_PHRASES[sample_id], selected_c.positions)
            assert_core_not_silent_noeos(selected_d, selected_c)
            semantic_row.update(
                {
                    "clean_full_text": state["clean_text"],
                    "clean_full_token_ids": [int(token) for token in state["clean_tokens"]],
                    "clean_fastv_text": fast["text"],
                    "clean_fastv_token_ids": reference,
                    "clean_fastv_keep_hash": fast["keep_hash"],
                    "clean_fastv_k": fast["k"],
                    "core_token_positions": list(selected_d.positions),
                    "core_token_ids": list(selected_d.token_ids),
                    "core_pieces": list(selected_d.pieces),
                    "core_match_reason": selected_d.reason,
                    "selection_b": selection_payload(selected_b),
                    "selection_c": selection_payload(selected_c),
                    "selection_d": selection_payload(selected_d),
                    "clean_fastv_semantic_status": label["semantic_status"],
                    "clean_fastv_semantic_reason": label["reason"],
                }
            )
            checks["full_valid"] = bool(guard["output_valid"] and guard["full_text"])
            checks["full_replay"] = bool(guard["full_text_exact"] and guard["full_token_exact"] and guard["full_replay"])
            checks["fastv_valid"] = bool(fast["valid"])
            checks["fastv_content_correct"] = label["semantic_status"] in {STATUS_PRESERVED, STATUS_AUGMENTED}
            checks["core_span"] = bool(selected_d.available)
            detail = {"full_text": guard["full_text"], "fastv_text": fast["text"], "semantic": label}
            if not checks["full_valid"]:
                status = "CLEAN_FULL_INVALID"
            elif not checks["full_replay"]:
                status = "REPLAY_DIVERGED"
            elif not checks["fastv_valid"]:
                status = "CLEAN_FASTV_INVALID"
            elif not checks["fastv_content_correct"]:
                status = "CLEAN_FASTV_CONTENT_WRONG"
            elif not checks["core_span"]:
                status = "CORE_SPAN_UNAVAILABLE"
            else:
                status = "ADMITTED_CLEAN_CORRECT"
        del state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    admission = {
        "sample_id": sample_id,
        "status": status,
        "checks": checks,
        "provenance": prov,
        "detail": detail,
        "counts_in_main_denominator": status == "ADMITTED_CLEAN_CORRECT",
        "pressure_only": status == "CLEAN_FASTV_CONTENT_WRONG",
    }
    return admission, semantic_row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", default=",".join(PRIMARY_SAMPLES))
    parser.add_argument("--out", default=str(HERE))
    args = parser.parse_args()
    out = Path(args.out)
    rt.configure()
    model, processor = rt.v7d.load_model_frozen()
    admissions = []
    rows = []
    for sample_id in [item.strip() for item in args.samples.split(",") if item.strip()]:
        admission, row = admit_one(model, processor, sample_id, out)
        admissions.append(admission)
        rows.append(row)
        print(json.dumps({"sample": sample_id, "status": admission["status"]}, ensure_ascii=False), flush=True)
    admitted = [row["sample_id"] for row in admissions if row["counts_in_main_denominator"]]
    excluded = [
        {"sample_id": row["sample_id"], "status": row["status"]}
        for row in admissions
        if not row["counts_in_main_denominator"]
    ]
    rt.write_json(
        out / "admission.json",
        {
            "requested_samples": [item.strip() for item in args.samples.split(",") if item.strip()],
            "fixed_development_count": len(PRIMARY_SAMPLES),
            "admitted_count": len(admitted),
            "admitted_ids": admitted,
            "excluded": excluded,
            "samples": admissions,
        },
    )
    manifest = {
        "frozen_before_attack_results": True,
        "rule_version": "v20_semantic_rules_2026_10_14",
        "samples": rows,
    }
    rt.write_json(out / "semantic_manifest.json", manifest)
    digest = rt.sha_file(out / "semantic_manifest.json")
    (out / "sealed").mkdir(parents=True, exist_ok=True)
    (out / "sealed" / "semantic_manifest.sha256").write_text(digest + "\n")
    print(json.dumps({"admitted": admitted, "semantic_sha256": digest}), flush=True)


if __name__ == "__main__":
    main()
