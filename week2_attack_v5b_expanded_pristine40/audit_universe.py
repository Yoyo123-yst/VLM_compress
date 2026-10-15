"""Enumerate full MP-DocVQA val, verify the v5 exclusion union, freeze 200-screen manifest."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
V5 = ROOT / "week2_attack_v5_pristine40"
V5B = ROOT / "week2_attack_v5b_expanded_pristine40"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "gate_experiment"))
sys.path.insert(0, str(ROOT / "week1_mechanism"))

from data_mpdoc import iter_meta, list_val_files, parquet_dir  # noqa: E402
from run_step0_v2 import load_images, select_pages  # noqa: E402
from week2_attack_v4_answer_aligned.common import MODEL_PATH, RATIO, sha_image  # noqa: E402
from week2_attack_v4_answer_aligned.loss import official_keep_count  # noqa: E402
from week2_attack_v5_pristine40 import build_exclusions as scan  # noqa: E402

SCREEN_SEED = 2026100802
SCREEN_N = 200
TOKENS = 3699
EXPECTED = {
    "val_unique": 5187,
    "multipage": 3554,
    "exclusion": 280,
    "unseen_multipage": 3274,
}


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_hashes() -> dict[str, str]:
    paths = [
        V5B / "audit_universe.py",
        V5B / "run_clean_screen.py",
        ROOT / "week2_attack_v4_answer_aligned" / "loss.py",
        ROOT / "week2_attack_v4_answer_aligned" / "forward.py",
        ROOT / "week2_attack_v4_answer_aligned" / "run_smoke.py",
        ROOT / "week2_attack_v4_relaxed_full" / "run_lambda.py",
        ROOT / "gate_experiment" / "run_step0_v2.py",
        ROOT / "week1_mechanism" / "data_mpdoc.py",
    ]
    return {str(p.relative_to(ROOT)): sha_file(p) for p in paths if p.exists()}


def model_identity() -> dict:
    cfg = Path(MODEL_PATH) / "config.json"
    return {
        "model_path": MODEL_PATH,
        "config_sha256": sha_file(cfg) if cfg.exists() else None,
    }


def rescan_ids() -> set[str]:
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


def main():
    v5_payload = json.loads((V5 / "exclusions" / "historical_attack_ids.json").read_text())
    v5_ids = {row["sample_id"] for row in v5_payload["samples"]}
    rescanned = rescan_ids()
    if v5_ids != rescanned:
        only_v5 = sorted(v5_ids - rescanned, key=int)[:20]
        only_new = sorted(rescanned - v5_ids, key=int)[:20]
        audit = {
            "decision": "BLOCKED_UNIVERSE_AUDIT_MISMATCH",
            "reason": "rescan ID set differs from v5 union",
            "v5_n": len(v5_ids),
            "rescan_n": len(rescanned),
            "only_in_v5_sample": only_v5,
            "only_in_rescan_sample": only_new,
        }
        (V5B / "pool").mkdir(parents=True, exist_ok=True)
        (V5B / "pool" / "universe_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        print(json.dumps(audit))
        return

    excl_dir = V5B / "exclusions"
    excl_dir.mkdir(parents=True, exist_ok=True)
    union_json = json.dumps(
        {"n_unique": len(v5_ids), "source": "week2_attack_v5_pristine40/exclusions/historical_attack_ids.json", "samples": v5_payload["samples"]},
        indent=2,
    )
    (excl_dir / "historical_union.json").write_text(union_json + "\n")
    union_hash = hashlib.sha256((excl_dir / "historical_union.json").read_bytes()).hexdigest()
    (excl_dir / "historical_union.sha256").write_text(union_hash + "\n")
    with (excl_dir / "historical_union.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "reason", "sources"])
        writer.writeheader()
        for row in v5_payload["samples"]:
            writer.writerow({"sample_id": row["sample_id"], "reason": row["reason"], "sources": "|".join(row["sources"])})

    rows = list(iter_meta(list_val_files()))
    id_counts = Counter(row["sample_id"] for row in rows)
    duplicates = sorted([sid for sid, n in id_counts.items() if n > 1], key=int)
    unique_rows = {}
    for row in rows:
        unique_rows.setdefault(row["sample_id"], row)
    bad_evidence = []
    multipage = []
    single = []
    for sid, row in unique_rows.items():
        m_raw = len(row["page_ids"])
        if row["answer_page_idx"] < 0 or row["answer_page_idx"] >= m_raw:
            bad_evidence.append(sid)
            continue
        if m_raw > 1:
            multipage.append(sid)
        else:
            single.append(sid)
    multipage_set = set(multipage)
    excluded_multipage = sorted(multipage_set & v5_ids, key=int)
    unseen = sorted(multipage_set - v5_ids, key=int)
    counts = {
        "val_rows": len(rows),
        "val_unique": len(unique_rows),
        "duplicate_ids": len(duplicates),
        "duplicate_id_examples": duplicates[:20],
        "single_page": len(single),
        "multipage": len(multipage),
        "bad_evidence": len(bad_evidence),
        "bad_evidence_examples": bad_evidence[:20],
        "exclusion_union": len(v5_ids),
        "excluded_that_are_multipage": len(excluded_multipage),
        "excluded_not_in_val": sorted(v5_ids - set(unique_rows), key=int),
        "unseen_multipage": len(unseen),
        "expected_reference_only": EXPECTED,
        "v5_union_file_sha256": v5_payload.get("union_sha256"),
        "copied_union_sha256": union_hash,
        "rescan_matches_v5": True,
        "parquet_dir": str(parquet_dir()),
        "val_files": [p.name for p in list_val_files()],
    }
    mismatches = []
    if counts["val_unique"] != EXPECTED["val_unique"]:
        mismatches.append(f"val_unique {counts['val_unique']} != reference {EXPECTED['val_unique']}")
    if counts["multipage"] != EXPECTED["multipage"]:
        mismatches.append(f"multipage {counts['multipage']} != reference {EXPECTED['multipage']}")
    if counts["exclusion_union"] != EXPECTED["exclusion"]:
        mismatches.append(f"exclusion {counts['exclusion_union']} != reference {EXPECTED['exclusion']}")
    if counts["unseen_multipage"] != EXPECTED["unseen_multipage"]:
        mismatches.append(
            f"unseen_multipage {counts['unseen_multipage']} != reference {EXPECTED['unseen_multipage']}; "
            f"multipage {counts['multipage']} minus excluded_multipage {counts['excluded_that_are_multipage']}"
        )
    if duplicates or bad_evidence:
        mismatches.append("duplicate IDs or invalid evidence_idx present")
    pool = V5B / "pool"
    pool.mkdir(parents=True, exist_ok=True)
    with (pool / "unseen_multipage_candidates.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "M_raw", "answer_page_idx", "shard", "row"])
        writer.writeheader()
        for sid in unseen:
            row = unique_rows[sid]
            writer.writerow({
                "sample_id": sid,
                "M_raw": len(row["page_ids"]),
                "answer_page_idx": row["answer_page_idx"],
                "shard": row["shard"],
                "row": row["row"],
            })
    if mismatches:
        counts["decision"] = "BLOCKED_UNIVERSE_AUDIT_MISMATCH"
        counts["mismatches"] = mismatches
        (pool / "universe_audit.json").write_text(json.dumps(counts, indent=2) + "\n")
        (pool / "universe_audit.md").write_text(
            "# Universe audit\n\n" + "\n".join(f"- {item}" for item in mismatches) + "\n"
        )
        print(json.dumps({"decision": counts["decision"], "mismatches": mismatches, **{k: counts[k] for k in ("val_unique", "multipage", "unseen_multipage")}}))
        return

    rng = random.Random(SCREEN_SEED)
    screened = rng.sample(unseen, SCREEN_N)
    metas = []
    for sid in screened:
        meta = dict(unique_rows[sid])
        metas.append(meta)
    pages = load_images(metas)
    samples = []
    page_problems = []
    for meta in metas:
        sid = meta["sample_id"]
        raw_pages = pages.get(sid, [])
        if len(raw_pages) != len(meta["page_ids"]):
            page_problems.append({"sample_id": sid, "images": len(raw_pages), "page_ids": len(meta["page_ids"])})
            continue
        if meta["answer_page_idx"] >= len(raw_pages):
            page_problems.append({"sample_id": sid, "reason": "evidence_oob"})
            continue
        images, evidence_idx, order, dropped = select_pages(raw_pages, meta["answer_page_idx"], sid, 768)
        if len(images) <= 1:
            page_problems.append({"sample_id": sid, "reason": "collapsed_M"})
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
            "page_paths": [f"{meta['shard']}#image_{i+1}" for i in order],
            "page_sha256": [sha_image(im) for im in images],
            "expected_k": official_keep_count(len(images) * TOKENS, RATIO),
        })
    if page_problems or len(samples) != SCREEN_N:
        payload = {"decision": "BLOCKED_UNIVERSE_AUDIT_MISMATCH", "page_problems": page_problems, "built": len(samples)}
        (pool / "universe_audit.json").write_text(json.dumps({**counts, **payload}, indent=2) + "\n")
        print(json.dumps(payload))
        return
    samples.sort(key=lambda row: int(row["sample_id"]))
    manifest = {
        "screen_seed": SCREEN_SEED,
        "screen_n": SCREEN_N,
        "sampling": "stable numeric ID order of unseen multipage, then random.Random(SCREEN_SEED).sample",
        "historical_union_sha256": union_hash,
        "v5_union_sha256": v5_payload.get("union_sha256"),
        "fastv_r": RATIO,
        "page_side": 768,
        "select_pages": "run_step0_v2.select_pages seed 42:{sample_id} cap M=4",
        "model": model_identity(),
        "code_hashes": code_hashes(),
        "data_source": "lmms-lab-encoder/MP-DocVQA val parquet via week1_mechanism.data_mpdoc",
        "sample_order_from_rng": screened,
        "samples": samples,
    }
    raw = json.dumps(manifest, sort_keys=True).encode()
    manifest_hash = hashlib.sha256(raw).hexdigest()
    manifest["manifest_sha256"] = manifest_hash
    (pool / "screen_manifest_200.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (pool / "screen_manifest_200.sha256").write_text(manifest_hash + "\n")
    with (pool / "screen_manifest_200.csv").open("w", newline="") as handle:
        fields = ["sample_id", "M", "evidence_idx", "aux_idx", "expected_k", "shard", "dropped"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in samples:
            writer.writerow({k: row[k] for k in fields})
    counts["decision"] = "UNIVERSE_AUDIT_PASS"
    counts["screen_n"] = SCREEN_N
    counts["screen_manifest_sha256"] = manifest_hash
    counts["screen_ids"] = [row["sample_id"] for row in samples]
    (pool / "universe_audit.json").write_text(json.dumps(counts, indent=2) + "\n")
    lines = [
        "# Universe audit",
        "",
        f"- Val unique questions: {counts['val_unique']}",
        f"- Multipage (raw page_ids > 1, valid evidence): {counts['multipage']}",
        f"- Historical exclusion union: {counts['exclusion_union']}",
        f"- Excluded IDs that are multipage: {counts['excluded_that_are_multipage']}",
        f"- Unseen multipage: {counts['unseen_multipage']}",
        f"- Duplicate IDs: {counts['duplicate_ids']}",
        f"- Invalid evidence: {counts['bad_evidence']}",
        f"- Copied union SHA-256: `{union_hash}`",
        f"- Screen manifest SHA-256: `{manifest_hash}`",
        f"- Decision: {counts['decision']}",
        "",
        "Reference counts matched. v5 exclusion ID set matches a fresh scan. Oracle-only IDs remain excluded.",
    ]
    (pool / "universe_audit.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "decision": "UNIVERSE_AUDIT_PASS",
        "val_unique": counts["val_unique"],
        "multipage": counts["multipage"],
        "exclusion": counts["exclusion_union"],
        "unseen_multipage": counts["unseen_multipage"],
        "screen_sha": manifest_hash,
    }))


if __name__ == "__main__":
    main()
