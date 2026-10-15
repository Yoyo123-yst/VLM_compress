"""Materialize metadata/layout for the already-frozen six ID-only cohort."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "gate_experiment"), str(ROOT / "week1_mechanism")]

from data_mpdoc import iter_meta, list_val_files  # noqa: E402
from run_step0_v2 import load_images, select_pages  # noqa: E402
from week2_attack_v4_answer_aligned.common import sha_image  # noqa: E402
from week2_attack_v4_answer_aligned.loss import official_keep_count  # noqa: E402


def main() -> None:
    freeze = json.loads((OUT / "sample_selection_freeze.json").read_text())
    pool_path = ROOT / freeze["candidate_pool"]["path"]
    pool = {row["sample_id"]: row for row in csv.DictReader(pool_path.open(newline=""))}
    wanted = [row["sample_id"] for row in freeze["frozen_order"]]
    metas_by_id: dict[str, dict] = {}
    for row in iter_meta(list_val_files()):
        if row["sample_id"] in wanted:
            if row["sample_id"] in metas_by_id:
                raise RuntimeError(f"duplicate metadata ID {row['sample_id']}")
            metas_by_id[row["sample_id"]] = row
    if set(metas_by_id) != set(wanted):
        raise RuntimeError(f"frozen IDs missing from dataset: {sorted(set(wanted)-set(metas_by_id))}")

    metas = [metas_by_id[sid] for sid in wanted]
    pages = load_images(metas)
    samples = []
    problems = []
    for sid, meta in zip(wanted, metas):
        source = pool[sid]
        if (meta["shard"] != source["shard"] or int(meta["row"]) != int(source["row"])
                or len(meta["page_ids"]) != int(source["M_raw"])
                or int(meta["answer_page_idx"]) != int(source["answer_page_idx"])):
            raise RuntimeError(f"ID-only pool provenance mismatch for {sid}")
        raw_pages = pages.get(sid, [])
        if len(raw_pages) != len(meta["page_ids"]):
            problems.append({"sample_id": sid, "reason": "RAW_PAGE_COUNT_MISMATCH", "got": len(raw_pages)})
            continue
        images, evidence_idx, order, dropped = select_pages(raw_pages, meta["answer_page_idx"], sid, 768)
        if len(images) <= 1 or evidence_idx < 0 or evidence_idx >= len(images):
            problems.append({"sample_id": sid, "reason": "NO_AUXILIARY_PAGE_AFTER_LAYOUT"})
            continue
        aux_idx = next(i for i in range(len(images)) if i != evidence_idx)
        samples.append({
            "sample_id": sid,
            "shard": meta["shard"],
            "row": int(meta["row"]),
            "question": meta["question"],
            "answers": list(meta["answers"]),
            "page_ids": list(meta["page_ids"]),
            "answer_page_idx": int(meta["answer_page_idx"]),
            "order": list(order),
            "dropped": bool(dropped),
            "M": len(images),
            "M_raw": len(meta["page_ids"]),
            "evidence_idx": int(evidence_idx),
            "aux_idx": int(aux_idx),
            "page_sha256": [sha_image(image) for image in images],
            "expected_k": official_keep_count(len(images) * 3699, 0.25),
        })
    payload = {
        "status": "FROZEN_SAMPLE_SPECS" if not problems and len(samples) == len(wanted) else "BLOCKED_LAYOUT_RESTORE",
        "source_selection_sha256": hashlib.sha256((OUT / "sample_selection_freeze.json").read_bytes()).hexdigest(),
        "layout_rule": "run_step0_v2.select_pages seed 42:{sample_id}, cap M=4; first non-evidence page is auxiliary",
        "samples_in_frozen_order": samples,
        "problems": problems,
    }
    out_path = OUT / "sample_specs.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    out_path.with_suffix(".sha256").write_text(hashlib.sha256(out_path.read_bytes()).hexdigest() + "\n")
    print(json.dumps({"status": payload["status"], "n": len(samples), "problems": problems,
                      "sample_specs_sha256": hashlib.sha256(out_path.read_bytes()).hexdigest()}, ensure_ascii=False), flush=True)
    if payload["status"] != "FROZEN_SAMPLE_SPECS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
