"""Load only samples in the frozen v29 cohort and verify restored page hashes."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "week1_mechanism"), str(ROOT / "gate_experiment")]

from data_mpdoc import iter_meta, list_val_files  # noqa: E402
from run_step0_v2 import load_images, select_pages  # noqa: E402
from week2_attack_v4_answer_aligned.common import sha_image  # noqa: E402


def specs() -> dict[str, dict]:
    path = OUT / "sample_specs.json"
    payload = json.loads(path.read_text())
    if payload.get("status") != "FROZEN_SAMPLE_SPECS":
        raise RuntimeError("sample specs are not frozen/admitted")
    return {row["sample_id"]: row for row in payload["samples_in_frozen_order"]}


def load_item(sample_id: str) -> dict:
    manifest = specs()
    if sample_id not in manifest:
        raise KeyError(f"sample {sample_id} is not in frozen v29 cohort")
    spec = manifest[sample_id]
    meta = None
    for row in iter_meta(list_val_files()):
        if row["sample_id"] == sample_id:
            if meta is not None:
                raise RuntimeError(f"duplicate metadata ID {sample_id}")
            meta = row
    if meta is None or meta["question"] != spec["question"]:
        raise RuntimeError(f"dataset metadata mismatch for {sample_id}")
    pages = load_images([meta]).get(sample_id, [])
    if len(pages) != len(meta["page_ids"]):
        raise RuntimeError(f"raw page count mismatch for {sample_id}")
    images, evidence_idx, order, dropped = select_pages(pages, meta["answer_page_idx"], sample_id, 768)
    hashes = [sha_image(image) for image in images]
    if hashes != spec["page_sha256"] or list(order) != spec["order"] or evidence_idx != spec["evidence_idx"]:
        raise RuntimeError(f"page restoration mismatch for {sample_id}")
    return {
        "meta": meta,
        "images": images,
        "evidence_idx": int(evidence_idx),
        "aux_idx": int(spec["aux_idx"]),
        "order": list(order),
        "dropped": bool(dropped),
        "page_sha": hashes,
        "spec": spec,
    }
