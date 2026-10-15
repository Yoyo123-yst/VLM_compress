"""Clean Full/FastV screen of the frozen 200. Does not rewrite the screen manifest."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
V5B = ROOT / "week2_attack_v5b_expanded_pristine40"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "gate_experiment"))
sys.path.insert(0, str(ROOT / "week1_mechanism"))

from adapters.runtime import clear_hooks  # noqa: E402
from data_mpdoc import iter_meta, list_val_files  # noqa: E402
from metrics_task import anls_score  # noqa: E402
from run_causal_v2 import generate, load_model, prepare, run_fastv, run_plain  # noqa: E402
from run_step0_v2 import load_images, select_pages  # noqa: E402
from week2_attack_v4_answer_aligned.common import RATIO, keep_hash, sha_image, spans_for  # noqa: E402
from week2_attack_v4_answer_aligned.forward import hook_kept  # noqa: E402
from week2_attack_v4_answer_aligned.loss import official_keep_count, output_valid  # noqa: E402

WRONG = 0.5
TOKENS = 3699
FIELDS = [
    "sample_id", "M", "evidence_idx", "aux_idx", "expected_k", "k",
    "pred_full", "pred_fastv", "anls_full", "anls_fastv",
    "valid_full", "valid_fastv", "eligible", "keep_hash",
    "page_sha_ok", "k_ok", "hook_ok", "seconds",
]


def load_item(spec: dict, meta: dict, pages: list):
    images, evidence_idx, order, dropped = select_pages(pages, meta["answer_page_idx"], spec["sample_id"], 768)
    sha = [sha_image(im) for im in images]
    if sha != spec["page_sha256"] or order != spec["order"] or evidence_idx != spec["evidence_idx"]:
        raise RuntimeError(f"screen manifest page mismatch {spec['sample_id']}")
    return images, evidence_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--nshards", type=int, default=2)
    args = parser.parse_args()
    manifest = json.loads((V5B / "pool" / "screen_manifest_200.json").read_text())
    recorded = (V5B / "pool" / "screen_manifest_200.sha256").read_text().strip()
    if manifest.get("manifest_sha256") != recorded:
        raise SystemExit("screen manifest hash mismatch")
    metas = {row["sample_id"]: row for row in iter_meta(list_val_files())}
    specs = [row for i, row in enumerate(manifest["samples"]) if i % args.nshards == args.shard]
    out = V5B / "clean_screen"
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"shard_s{args.shard}.csv"
    jsonl_path = out / f"shard_s{args.shard}.jsonl"
    done = set()
    if csv_path.exists():
        with csv_path.open() as handle:
            done = {row["sample_id"] for row in csv.DictReader(handle)}
    model, processor = load_model()
    for param in model.parameters():
        param.requires_grad_(False)
    token_id = int(model.config.image_token_index)
    pending = [spec for spec in specs if spec["sample_id"] not in done]
    loaded = load_images([metas[spec["sample_id"]] for spec in pending])
    for spec in pending:
        sid = spec["sample_id"]
        t0 = time.time()
        images, evidence_idx = load_item(spec, metas[sid], loaded[sid])
        clear_hooks(model)
        pred_full, _, _, _ = run_plain(model, processor, images, spec["question"], token_id, False, evidence_idx)
        pred_fast, _, kept, _ = run_fastv(
            model, processor, images, spec["question"], token_id, RATIO, evidence_idx, spec["aux_idx"]
        )
        clear_hooks(model)
        inputs = prepare(processor, images, spec["question"])
        spans = spans_for(model, inputs, len(images))
        hooked = hook_kept(model, inputs, spans, RATIO)
        clear_hooks(model)
        k = len(kept)
        expected = official_keep_count(len(images) * TOKENS, RATIO)
        anls_full = anls_score(pred_full, spec["answers"])
        anls_fast = anls_score(pred_fast, spec["answers"])
        valid_full = output_valid(pred_full)
        valid_fast = output_valid(pred_fast)
        page_ok = True
        k_ok = k == expected == spec["expected_k"]
        hook_ok = list(hooked) == list(kept)
        eligible = (
            spec["M"] > 1 and valid_full and valid_fast
            and anls_full >= WRONG and anls_fast >= WRONG
            and page_ok and k_ok and hook_ok
        )
        row = {
            "sample_id": sid,
            "M": spec["M"],
            "evidence_idx": evidence_idx,
            "aux_idx": spec["aux_idx"],
            "expected_k": expected,
            "k": k,
            "pred_full": pred_full,
            "pred_fastv": pred_fast,
            "anls_full": f"{anls_full:.4f}",
            "anls_fastv": f"{anls_fast:.4f}",
            "valid_full": int(valid_full),
            "valid_fastv": int(valid_fast),
            "eligible": int(eligible),
            "keep_hash": keep_hash(list(kept)),
            "page_sha_ok": int(page_ok),
            "k_ok": int(k_ok),
            "hook_ok": int(hook_ok),
            "seconds": f"{time.time() - t0:.1f}",
        }
        new = not csv_path.exists()
        with csv_path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            if new:
                writer.writeheader()
            writer.writerow(row)
        with jsonl_path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps({"id": sid, "eligible": int(eligible), "full": row["anls_full"], "fastv": row["anls_fastv"]}), flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
