"""Freeze the v7 historical exclusion union and the 400-question screen manifest.

Reads earlier experiment trees. Writes only under week2_attack_v7_pristine60.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
V7 = ROOT / "week2_attack_v7_pristine60"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "gate_experiment"))
sys.path.insert(0, str(ROOT / "week1_mechanism"))

from data_mpdoc import iter_meta, list_val_files, parquet_dir  # noqa: E402
from run_step0_v2 import load_images, select_pages  # noqa: E402
from week2_attack_v4_answer_aligned.common import MODEL_PATH, RATIO, sha_image  # noqa: E402
from week2_attack_v4_answer_aligned.loss import official_keep_count  # noqa: E402
from week2_attack_v5_pristine40 import build_exclusions as scan  # noqa: E402

SCREEN_SEED = 2026100901
SCREEN_N = 400
TOKENS = 3699
V5_EXPECTED = 280

ATTACK_ROOTS = [
    ROOT / "week2_attack_v5_pristine40",
    ROOT / "week2_attack_v5b_expanded_pristine40",
    ROOT / "week2_attack_v6_survivor_upgrade",
    ROOT / "week2_attack_v6b_soft_full_anchor",
    ROOT / "week2_attack_v6c_split_calibration",
]
SKIP_PARTS = {
    "pool",
    "clean_screen",
    "exclusions",
    "__pycache__",
    "tests",
}
# v5 smoke tests are real attack images and must be included despite tests/.
FORCE_INCLUDE_PARTS = ("tests/smoke_",)


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def add(record: dict, sid: str, source: str):
    scan.add(record, sid, source)


def rescan_v5() -> set[str]:
    record: dict = {}
    for root in scan.SCAN_ROOTS:
        scan.scan_tree(root, record)
        if root.name == "week2_attack_v4_answer_aligned":
            for sid in scan.KNOWN_DEV:
                scan.add(record, sid, "hardcoded:v4_dev_IDS")
    holdout = ROOT / "week2_attack_v4_relaxed_full" / "holdout" / "frozen_manifest.json"
    if holdout.exists():
        scan.scan_json_manifest(holdout, record)
    return set(record)


def skip_attack_path(path: Path) -> bool:
    rel = str(path.relative_to(ROOT))
    if any(fragment in rel for fragment in FORCE_INCLUDE_PARTS):
        return False
    return any(part in SKIP_PARTS for part in path.parts)


def harvest_attack_ids(record: dict) -> dict[str, int]:
    counts = Counter()
    for root in ATTACK_ROOTS:
        if not root.exists():
            continue
        before = len(record)
        for path in root.rglob("*"):
            if skip_attack_path(path):
                continue
            if path.is_dir() and scan.SID_RE.match(path.name):
                has_png = any(path.glob("*.png")) or any(path.glob("checkpoints/*.png"))
                if has_png or (path / "provenance.json").exists() or (path / "curve.json").exists():
                    add(record, path.name, f"artifact_dir:{path.relative_to(ROOT)}")
            if not path.is_file():
                continue
            name = path.name.lower()
            if name.endswith(".csv"):
                scan.scan_csv(path, record)
            elif name.endswith(".jsonl"):
                scan.scan_jsonl(path, record)
            elif name in {
                "frozen_manifest.json",
                "provenance.json",
                "curve.json",
                "selected.csv",
            } or name.startswith("health") or "replay" in name or name.startswith("cells"):
                if name.endswith(".json"):
                    try:
                        payload = json.loads(path.read_text())
                    except Exception:
                        continue
                    if isinstance(payload, dict) and "sample_id" in payload:
                        add(record, payload["sample_id"], f"json:{path.relative_to(ROOT)}")
                    scan.scan_json_manifest(path, record)
                    if isinstance(payload, dict):
                        for key in ("dev_ids", "ids", "health_ids"):
                            values = payload.get(key)
                            if isinstance(values, list):
                                for item in values:
                                    if isinstance(item, str):
                                        add(record, item, f"json:{path.relative_to(ROOT)}:{key}")
        counts[root.name] = len(record) - before
    for sid in ("1025", "6194", "7006"):
        add(record, sid, "hardcoded:v6b_v6c_health_probe_smoke")
    return dict(counts)


def main():
    v5_payload = json.loads(
        (ROOT / "week2_attack_v5_pristine40" / "exclusions" / "historical_attack_ids.json").read_text()
    )
    v5_ids = {row["sample_id"] for row in v5_payload["samples"]}
    rescanned = rescan_v5()
    (V7 / "exclusions").mkdir(parents=True, exist_ok=True)
    (V7 / "pool").mkdir(parents=True, exist_ok=True)
    if v5_ids != rescanned or len(v5_ids) != V5_EXPECTED:
        audit = {
            "decision": "BLOCKED_UNIVERSE_AUDIT_MISMATCH",
            "v5_n": len(v5_ids),
            "rescan_n": len(rescanned),
            "only_in_v5": sorted(v5_ids - rescanned, key=int)[:30],
            "only_in_rescan": sorted(rescanned - v5_ids, key=int)[:30],
        }
        (V7 / "pool" / "universe_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        print(json.dumps(audit))
        return

    record = {row["sample_id"]: dict(row) for row in v5_payload["samples"]}
    for row in record.values():
        row["sources"] = list(row["sources"]) + ["v5_historical_union"]
    extra_counts = harvest_attack_ids(record)
    items = [record[sid] for sid in sorted(record, key=int)]
    union = {
        "n_unique": len(items),
        "v5_union_n": len(v5_ids),
        "v5_rescan_matches": True,
        "extra_new_ids_by_root": extra_counts,
        "source": "v5 historical union plus later attack/health/probe/smoke/mechanism artifacts",
        "samples": items,
    }
    raw_union = json.dumps(union, sort_keys=True).encode()
    union_hash = hashlib.sha256(raw_union).hexdigest()
    union["union_sha256"] = union_hash
    (V7 / "exclusions" / "historical_union.json").write_text(json.dumps(union, indent=2) + "\n")
    (V7 / "exclusions" / "historical_union.sha256").write_text(union_hash + "\n")
    with (V7 / "exclusions" / "historical_union.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "reason", "sources"])
        writer.writeheader()
        for row in items:
            writer.writerow({
                "sample_id": row["sample_id"],
                "reason": row["reason"],
                "sources": "|".join(row["sources"]),
            })
    excluded = set(record)

    rows = list(iter_meta(list_val_files()))
    id_counts = Counter(row["sample_id"] for row in rows)
    duplicates = sorted([sid for sid, n in id_counts.items() if n > 1], key=int)
    unique_rows = {}
    for row in rows:
        unique_rows.setdefault(row["sample_id"], row)
    bad_evidence = []
    multipage = []
    for sid, row in unique_rows.items():
        if row["answer_page_idx"] < 0 or row["answer_page_idx"] >= len(row["page_ids"]):
            bad_evidence.append(sid)
            continue
        if len(row["page_ids"]) > 1:
            multipage.append(sid)
    multipage_set = set(multipage)
    unseen = sorted(multipage_set - excluded, key=int)
    counts = {
        "val_unique": len(unique_rows),
        "multipage": len(multipage),
        "bad_evidence": len(bad_evidence),
        "duplicate_ids": len(duplicates),
        "exclusion_union": len(excluded),
        "excluded_multipage": len(multipage_set & excluded),
        "excluded_not_in_val": sorted(excluded - set(unique_rows), key=int),
        "unseen_multipage": len(unseen),
        "v5_union_n": len(v5_ids),
        "parquet_dir": str(parquet_dir()),
        "extra_new_ids_by_root": extra_counts,
    }
    if duplicates or bad_evidence or len(unseen) < SCREEN_N:
        counts["decision"] = "BLOCKED_UNIVERSE_AUDIT_MISMATCH"
        (V7 / "pool" / "universe_audit.json").write_text(json.dumps(counts, indent=2) + "\n")
        print(json.dumps({"decision": counts["decision"], "unseen": len(unseen)}))
        return
    with (V7 / "pool" / "unseen_multipage_candidates.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "M_raw", "answer_page_idx"])
        writer.writeheader()
        for sid in unseen:
            row = unique_rows[sid]
            writer.writerow({
                "sample_id": sid,
                "M_raw": len(row["page_ids"]),
                "answer_page_idx": row["answer_page_idx"],
            })

    rng = random.Random(SCREEN_SEED)
    screened_ids = rng.sample(unseen, SCREEN_N)
    metas = [dict(unique_rows[sid]) for sid in screened_ids]
    pages = load_images(metas)
    samples = []
    page_problems = []
    for meta in metas:
        sid = meta["sample_id"]
        raw_pages = pages.get(sid, [])
        if len(raw_pages) != len(meta["page_ids"]):
            page_problems.append({"sample_id": sid, "images": len(raw_pages), "page_ids": len(meta["page_ids"])})
            continue
        images, evidence_idx, order, dropped = select_pages(raw_pages, meta["answer_page_idx"], sid, 768)
        if len(images) <= 1 or evidence_idx < 0 or evidence_idx >= len(images):
            page_problems.append({"sample_id": sid, "reason": "collapsed_or_bad_evidence"})
            continue
        aux_idx = next(i for i in range(len(images)) if i != evidence_idx)
        samples.append({
            "sample_id": sid,
            "shard": meta["shard"],
            "row": meta["row"],
            "question": meta["question"],
            "answers": meta["answers"],
            "page_ids": meta["page_ids"],
            "answer_page_idx": meta["answer_page_idx"],
            "order": order,
            "dropped": dropped,
            "M": len(images),
            "evidence_idx": evidence_idx,
            "aux_idx": aux_idx,
            "page_sha256": [sha_image(im) for im in images],
            "expected_k": official_keep_count(len(images) * TOKENS, RATIO),
        })
    if page_problems or len(samples) != SCREEN_N:
        payload = {
            "decision": "BLOCKED_UNIVERSE_AUDIT_MISMATCH",
            "page_problems": page_problems,
            "built": len(samples),
            **counts,
        }
        (V7 / "pool" / "universe_audit.json").write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps({"decision": payload["decision"], "built": len(samples), "problems": len(page_problems)}))
        return
    overlap = sorted(set(row["sample_id"] for row in samples) & excluded, key=int)
    if overlap:
        print(json.dumps({"decision": "BLOCKED_UNIVERSE_AUDIT_MISMATCH", "overlap": overlap[:20]}))
        return
    samples_sorted = sorted(samples, key=lambda row: int(row["sample_id"]))
    manifest = {
        "screen_seed": SCREEN_SEED,
        "screen_n": SCREEN_N,
        "sampling": "numeric-sorted unseen multipage, then random.Random(2026100901).sample without replacement",
        "historical_union_sha256": union_hash,
        "fastv_r": RATIO,
        "page_side": 768,
        "select_pages": "run_step0_v2.select_pages seed 42:{sample_id} cap M=4",
        "model_path": MODEL_PATH,
        "data_source": "lmms-lab-encoder/MP-DocVQA val parquet via week1_mechanism.data_mpdoc",
        "sample_order_from_rng": screened_ids,
        "samples": samples_sorted,
    }
    raw = json.dumps(manifest, sort_keys=True).encode()
    manifest_hash = hashlib.sha256(raw).hexdigest()
    manifest["manifest_sha256"] = manifest_hash
    (V7 / "pool" / "screen_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (V7 / "pool" / "screen_manifest.sha256").write_text(manifest_hash + "\n")
    with (V7 / "pool" / "screen_manifest.csv").open("w", newline="") as handle:
        fields = ["sample_id", "M", "evidence_idx", "aux_idx", "expected_k", "shard", "dropped"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in samples_sorted:
            writer.writerow({key: row[key] for key in fields})
    counts["decision"] = "UNIVERSE_AUDIT_PASS"
    counts["screen_n"] = SCREEN_N
    counts["screen_manifest_sha256"] = manifest_hash
    counts["union_sha256"] = union_hash
    (V7 / "pool" / "universe_audit.json").write_text(json.dumps(counts, indent=2) + "\n")
    print(json.dumps({
        "decision": "UNIVERSE_AUDIT_PASS",
        "exclusion": len(excluded),
        "unseen_multipage": len(unseen),
        "screen_sha": manifest_hash,
        "extra": extra_counts,
    }))


if __name__ == "__main__":
    main()
