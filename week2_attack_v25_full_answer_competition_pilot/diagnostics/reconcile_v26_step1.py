"""Reconcile endpoint states and cost from frozen v25 JSON trajectories."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "iterative_v5_60700"
OLD = ROOT / "iterative_v3_extension10"
TARGETS = json.loads((ROOT / "frozen_targets.json").read_text())['samples']


def classify(text: str, clean: str, target: str) -> str:
    if text == target:
        return "CONTENT_FAILURE_TARGET"
    if text == clean:
        return "ANSWER_PRESERVED"
    return "AMBIGUOUS_REQUIRES_REVIEW"


def read_result(path: Path, sample: str, method: str):
    d = json.loads(path.read_text())
    clean = d["clean_fast_text"]
    target = d["wrong_sequence"]
    rows = []
    for u in d["updates"]:
        text = u["guard"]["fast_text"]
        rows.append({
            "accepted_step": u["accepted_after"],
            "attempt": u["attempt"],
            "fast_text": text,
            "semantic_status": classify(text, clean, target),
            "wall_seconds": u["wall_seconds"],
            "accepted": bool(u["accepted"]),
            "full_exact": bool(u["guard"]["full_text_exact"] and u["guard"]["full_token_exact"]),
            "replay": bool(u["guard"]["full_replay"] and u["guard"]["fast_replay"]),
            "uint8_linf": u["uint8_linf"],
        })
    accepted = [r for r in rows if r["accepted"]]
    statuses = [r["semantic_status"] for r in accepted]
    target_hit = next((r for r in accepted if r["semantic_status"] == "CONTENT_FAILURE_TARGET"), None)
    content_failure = target_hit
    recover_after = None
    if target_hit:
        idx = accepted.index(target_hit)
        recover_after = next((r for r in accepted[idx + 1:] if r["semantic_status"] == "ANSWER_PRESERVED"), None)
    return {
        "sample": sample,
        "method": method,
        "clean_text": clean,
        "target_text": target,
        "accepted": d["accepted"],
        "attempts": d["attempts"],
        "wall_seconds": d["wall_seconds"],
        "first_content_failure": content_failure,
        "first_target_hit": target_hit,
        "recovered_to_clean_after_target": recover_after,
        "final": accepted[-1] if accepted else None,
        "status_sequence": statuses,
        "all_guards_pass": all(r["full_exact"] and r["replay"] and r["uint8_linf"] <= 8 for r in accepted),
    }


def locate(sample: str, method: str):
    if sample == "60700":
        return RUNS / method / method / sample / "result.json"
    if method == "complete_front":
        return ROOT / "iterative_v4_front_constraint" / method / sample / "result.json"
    return OLD / method / sample / "result.json"


def main():
    methods = ["native_fdp", "complete_target", "complete_front"]
    samples = ["47450", "7006", "60700"]
    out = []
    for sample in samples:
        for method in methods:
            path = locate(sample, method)
            if path.exists():
                out.append(read_result(path, sample, method))
    payload = {
        "status": "DONE",
        "protocol": "v26_step1_endpoint_reconciliation",
        "target_provenance_branch_candidate_id_60700": TARGETS["60700"]["branch_candidate_id"],
        "records": out,
    }
    (ROOT / "diagnostics" / "v26_step1_endpoint_reconciliation.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    lines = ["# v26 Step 1 endpoint reconciliation", "", "This report is recomputed from the stored trajectory JSON files.", ""]
    for r in out:
        hit = r["first_target_hit"]
        fail = r["first_content_failure"]
        rec = r["recovered_to_clean_after_target"]
        lines.append(f"- {r['sample']} / {r['method']}: accepted={r['accepted']}, attempts={r['attempts']}, wall={r['wall_seconds']:.1f}s, first_content_failure={None if fail is None else fail['accepted_step']}, first_target_hit={None if hit is None else hit['accepted_step']}, recovered={None if rec is None else rec['accepted_step']}, final={r['final']['fast_text']!r}, guards={r['all_guards_pass']}")
    (ROOT / "diagnostics" / "v26_step1_endpoint_reconciliation.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": payload["status"], "records": len(out), "out": "diagnostics/v26_step1_endpoint_reconciliation.json"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
