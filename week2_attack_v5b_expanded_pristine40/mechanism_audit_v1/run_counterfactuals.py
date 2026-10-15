"""Frozen 2x2 on v5b final.png. Does not optimize and does not rewrite v5b raw."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
V5B = ROOT / "week2_attack_v5b_expanded_pristine40"
OUT = V5B / "mechanism_audit_v1"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "week1_mechanism"))
sys.path.insert(0, str(ROOT / "gate_experiment"))

from adapters.runtime import clear_hooks, install  # noqa: E402
from metrics_task import anls_score  # noqa: E402
from run_causal_v2 import budget, decode, generate, load_model, prepare  # noqa: E402
from week2_attack_v4_answer_aligned.common import RATIO, keep_hash, spans_for  # noqa: E402
from week2_attack_v4_answer_aligned.loss import official_keep_count  # noqa: E402
from week2_attack_v5b_expanded_pristine40.load_samples import load_formal_items, manifest  # noqa: E402

WRONG = 0.5


def fastv(model, processor, images, question, forced=None):
    clear_hooks(model)
    inputs = prepare(processor, images, question)
    spans = spans_for(model, inputs, len(images))
    state = install(model, "fastv", RATIO, int(model.config.image_token_index))
    state.spans = spans
    if forced is not None:
        target = [int(x) for x in forced]

        def override(visual_index, visual_scores, kept, target=target):
            out = torch.tensor(target, device=kept.device, dtype=torch.long)
            if int(out.numel()) != int(kept.numel()):
                raise RuntimeError(f"forced keep {int(out.numel())} != K {int(kept.numel())}")
            if len(set(target)) != len(target):
                raise RuntimeError("duplicate forced keep index")
            return out

        state.keep_override = override
    tokens = generate(model, inputs)
    pred = decode(processor, tokens)
    kept = [int(x) for x in state.kept_visual]
    first = int(tokens[0].item()) if tokens.numel() else None
    clear_hooks(model)
    visual_n = sum(end - start for start, end in spans)
    if len(kept) != official_keep_count(visual_n, RATIO):
        raise RuntimeError("K mismatch after FastV")
    return pred, kept, spans, first


def plain(model, processor, images, question):
    clear_hooks(model)
    inputs = prepare(processor, images, question)
    tokens = generate(model, inputs)
    pred = decode(processor, tokens)
    first = int(tokens[0].item()) if tokens.numel() else None
    clear_hooks(model)
    return pred, first


def cell_row(sid, condition, content, keep_name, pred, anls, kept, spans, evidence, aux, first, seconds, valid):
    stats = budget(kept, spans, evidence, aux, [])
    return {
        "sample_id": sid,
        "condition": condition,
        "content_source": content,
        "keep_source": keep_name,
        "K": len(kept),
        "keep_hash": keep_hash(kept),
        "per_page_keep": json.dumps(stats["kept_per_image"]),
        "evidence_share": stats["evidence_share"],
        "aux_share": stats["aux_share"],
        "prediction": pred,
        "anls": anls,
        "valid": int(valid),
        "first_token": first,
        "seconds": round(seconds, 2),
        "status": "ok",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--nshards", type=int, default=4)
    args = parser.parse_args()
    payload = manifest()
    specs = [row for i, row in enumerate(payload["samples"]) if i % args.nshards == args.shard]
    fixed = {row["sample_id"]: row for row in csv.DictReader((V5B / "results" / "fixed_step200.csv").open())}
    items = load_formal_items(specs)
    out = OUT / "shards"
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"cells_s{args.shard}.csv"
    done = set()
    if csv_path.exists():
        done = {row["sample_id"] for row in csv.DictReader(csv_path.open()) if row["condition"] == "AA"}
    model, processor = load_model()
    for param in model.parameters():
        param.requires_grad_(False)
    fields = None
    for item in items:
        sid = item["meta"]["sample_id"]
        if sid in done:
            continue
        spec = item["spec"]
        final = Image.open(V5B / "attacks" / "artifacts" / sid / "final.png").convert("RGB")
        attacked = list(item["images"])
        attacked[item["aux_idx"]] = final
        question = item["meta"]["question"]
        answers = item["meta"]["answers"]
        t0 = time.time()
        cc_pred, cc_keep, cc_spans, cc_first = fastv(model, processor, item["images"], question)
        aa_pred, aa_keep, aa_spans, aa_first = fastv(model, processor, attacked, question)
        ac_pred, ac_keep, _, ac_first = fastv(model, processor, attacked, question, cc_keep)
        ca_pred, ca_keep, _, ca_first = fastv(model, processor, item["images"], question, aa_keep)
        full_clean, full_clean_first = plain(model, processor, item["images"], question)
        full_final, full_final_first = plain(model, processor, attacked, question)
        cc_ok = cc_pred == spec["clean_fastv_pred"]
        aa_ok = aa_pred == fixed[sid]["pred_fastv"]
        full_clean_ok = full_clean == spec["clean_full_pred"]
        full_final_ok = full_final == fixed[sid]["pred_full"]
        rows = []
        packs = [
            ("CC", "clean", "clean", cc_pred, cc_keep, cc_spans, cc_first),
            ("CA", "clean", "adversarial", ca_pred, ca_keep, cc_spans, ca_first),
            ("AC", "adversarial", "clean", ac_pred, ac_keep, aa_spans, ac_first),
            ("AA", "adversarial", "adversarial", aa_pred, aa_keep, aa_spans, aa_first),
        ]
        for name, content, keep_name, pred, kept, spans, first in packs:
            rows.append(cell_row(
                sid, name, content, keep_name, pred, anls_score(pred, answers), kept, spans,
                item["evidence_idx"], item["aux_idx"], first, time.time() - t0, bool(pred.strip()),
            ))
        rows.append({
            "sample_id": sid, "condition": "FULL_CLEAN", "content_source": "clean", "keep_source": "none",
            "K": "", "keep_hash": "", "per_page_keep": "", "evidence_share": "", "aux_share": "",
            "prediction": full_clean, "anls": anls_score(full_clean, answers), "valid": int(bool(full_clean.strip())),
            "first_token": full_clean_first, "seconds": round(time.time() - t0, 2),
            "status": "ok" if full_clean_ok else "FULL_CLEAN_MISMATCH",
        })
        rows.append({
            "sample_id": sid, "condition": "FULL_FINAL", "content_source": "adversarial", "keep_source": "none",
            "K": "", "keep_hash": "", "per_page_keep": "", "evidence_share": "", "aux_share": "",
            "prediction": full_final, "anls": anls_score(full_final, answers), "valid": int(bool(full_final.strip())),
            "first_token": full_final_first, "seconds": round(time.time() - t0, 2),
            "status": "ok" if full_final_ok else "FULL_FINAL_MISMATCH",
        })
        rows[0]["status"] = "ok" if cc_ok else "CC_MISMATCH"
        rows[3]["status"] = "ok" if aa_ok else "AA_MISMATCH"
        fields = list(rows[0])
        new = not csv_path.exists()
        with csv_path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if new:
                writer.writeheader()
            writer.writerows(rows)
        print(json.dumps({
            "id": sid, "cc_ok": cc_ok, "aa_ok": aa_ok,
            "cc": cc_pred, "aa": aa_pred, "ac": ac_pred, "ca": ca_pred,
        }), flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
