"""Freeze an ID/document-disjoint development screening queue before model outputs."""
from __future__ import annotations
import csv, hashlib, json, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "week1_mechanism"), str(ROOT / "gate_experiment")]
from data_mpdoc import iter_meta, list_val_files

POOL = ROOT / "week2_attack_v5b_expanded_pristine40/pool/unseen_multipage_candidates.csv"
V28 = ROOT / "week2_attack_v28_complete_front_holdout/sample_selection_freeze.json"

def main() -> None:
    old = json.loads(V28.read_text())
    reserved = {str(x) for x in old["prior_exposure_audit"]["reserved_ids_used_only_for_exclusion"]}
    exposed = set()
    # Reuse the audited prior ID set; do not inspect reserved-question content.
    for p in ROOT.glob("week2_attack_*/**/*"):
        if p.suffix.lower() not in {".json", ".csv"}:
            continue
        if p.resolve() == POOL.resolve():
            continue
        try:
            txt = p.read_text(errors="ignore")
        except OSError:
            continue
        # Conservative ID extraction from prior artifacts. This is only an exclusion set.
        for token in __import__("re").findall(r'"sample_id"\s*:\s*"?(\d+)"?', txt):
            exposed.add(token)
    exposed |= {str(x["sample_id"]) for x in old["frozen_order"]}
    exposed |= reserved
    rows = list(iter_meta(list_val_files()))
    by_id = {str(r["sample_id"]): r for r in rows}
    exposed_docs = {by_id[s]["doc_id"] for s in exposed if s in by_id}
    pool = list(csv.DictReader(POOL.open(newline="")))
    candidates = []
    for row in pool:
        sid = str(row["sample_id"])
        meta = by_id.get(sid)
        if sid in exposed or meta is None or meta["doc_id"] in exposed_docs:
            continue
        candidates.append({"sample_id": sid, "M_raw": int(row["M_raw"]),
                           "answer_page_idx": int(row["answer_page_idx"]),
                           "shard": row["shard"], "row": int(row["row"]),
                           "doc_id": meta["doc_id"]})
    rng = random.Random(2026101502)
    rng.shuffle(candidates)
    queue = candidates[:120]
    payload = {
        "status": "BATCH_SCREEN_QUEUE_FROZEN",
        "date": "2026-10-15", "seed": 2026101502,
        "selection_unit": "sample_id and document ID only; no question, answer, clean output, or attack result used",
        "candidate_pool": {"path": str(POOL.relative_to(ROOT)), "sha256": hashlib.sha256(POOL.read_bytes()).hexdigest(), "rows": len(pool)},
        "exclusion": {"prior_exposed_ids": len(exposed), "prior_exposed_docs": len(exposed_docs), "reserved_ids": sorted(reserved), "document_disjoint": True},
        "screen_cap": 120, "target_clean_eligible": 30,
        "frozen_order": queue,
        "stop_rule": "screen all 120 in order; attack first 30 clean-eligible only; fewer than 30 is reported, never replaced",
    }
    (OUT / "sample_selection_freeze.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"queue": len(queue), "eligible_pool": len(candidates), "prior_ids": len(exposed), "prior_docs": len(exposed_docs)}, ensure_ascii=False))

if __name__ == "__main__": main()
