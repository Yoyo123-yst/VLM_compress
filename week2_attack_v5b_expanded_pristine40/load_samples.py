"""Restore frozen v5b pages from the full MP-DocVQA val metadata."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
V5B = ROOT / "week2_attack_v5b_expanded_pristine40"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "gate_experiment"))
sys.path.insert(0, str(ROOT / "week1_mechanism"))

from data_mpdoc import iter_meta, list_val_files  # noqa: E402
from run_step0_v2 import load_images, select_pages  # noqa: E402
from week2_attack_v4_answer_aligned.common import sha_image  # noqa: E402
from week2_attack_v4_answer_aligned.loss import official_keep_count  # noqa: E402


def manifest() -> dict:
    return json.loads((V5B / "frozen_manifest.json").read_text())


def load_formal_items(specs: list[dict]) -> list[dict]:
    catalog = {row["sample_id"]: row for row in iter_meta(list_val_files())}
    metas = []
    for spec in specs:
        meta = catalog.get(spec["sample_id"])
        if meta is None:
            raise RuntimeError(f"missing val row {spec['sample_id']}")
        if meta["question"] != spec["question"]:
            raise RuntimeError(f"question mismatch {spec['sample_id']}")
        metas.append(meta)
    pages = load_images(metas)
    items = []
    for spec, meta in zip(specs, metas):
        images, evidence_idx, order, dropped = select_pages(
            pages[spec["sample_id"]], meta["answer_page_idx"], spec["sample_id"], 768
        )
        sha = [sha_image(image) for image in images]
        if sha != spec["page_sha256"] or order != spec["order"] or evidence_idx != spec["evidence_idx"]:
            raise RuntimeError(f"page restore mismatch {spec['sample_id']}")
        if len(images) != spec["M"] or spec["aux_idx"] == evidence_idx:
            raise RuntimeError(f"layout mismatch {spec['sample_id']}")
        expected = official_keep_count(len(images) * 3699, 0.25)
        if expected != spec["expected_k"]:
            raise RuntimeError(f"K mismatch {spec['sample_id']}")
        items.append({
            "meta": meta,
            "images": images,
            "evidence_idx": evidence_idx,
            "aux_idx": spec["aux_idx"],
            "order": order,
            "dropped": dropped,
            "page_sha": sha,
            "baseline_full": {"anls": spec["clean_full_anls"], "pred": spec["clean_full_pred"]},
            "baseline_fastv": {"anls": spec["clean_fastv_anls"], "pred": spec["clean_fastv_pred"]},
            "spec": spec,
        })
    return items
