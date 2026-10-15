#!/usr/bin/env python3
"""Reconcile the existing v26 trajectories without loading a model.

This is deliberately a small, explicit semantic audit.  It keeps exact target
hits separate from content errors and marks ambiguous strings for review.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "week2_attack_v25_full_answer_competition_pilot"
OUT = Path(__file__).resolve().parent
DEV = ROOT / "week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation" / "DEV_MANIFEST_COPY.json"

GT = {
    "47567": {"answer": "600", "kind": "number", "status": "clean_correct"},
    "50779": {"answer": "Dr. Darby", "kind": "person", "status": "clean_correct"},
    "51133": {"answer": "two", "kind": "number", "status": "clean_correct"},
    "6194": {"answer": "0107-XX-XX", "kind": "identifier", "status": "clean_correct"},
    "62941": {"answer": "December 19, 1979", "kind": "date", "status": "clean_correct"},
    "64506": {"answer": "E", "kind": "letter", "status": "clean_correct"},
}

TARGET_TYPE = {
    "47567": "valid_numeric",
    "50779": "valid_person_phrase",
    "51133": "valid_numeric_phrase",
    "6194": "valid_identifier",
    "62941": "valid_date",
    "64506": "invalid_for_question_type",  # "Section 2" is not a letter answer.
}

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def norm(s: str) -> str:
    return "".join(ch.lower() for ch in s if ch.isalnum())

def semantic(sample: str, text: str) -> str:
    t = text.strip()
    n = norm(t)
    if sample == "47567":
        return "ANSWER_PRESERVED" if n == "600" else "CONTENT_FAILURE_NUMERIC"
    if sample == "50779":
        if n == "drdarby":
            return "ANSWER_PRESERVED"
        if "darby" in n or "donbary" in n or "donary" in n:
            return "AMBIGUOUS_PERSON_IDENTITY_OR_SPELLING_REVIEW"
        return "CONTENT_FAILURE_PERSON"
    if sample == "51133":
        return "ANSWER_PRESERVED" if n in {"two", "twoyears"} else "CONTENT_FAILURE_NUMERIC"
    if sample == "6194":
        return "ANSWER_PRESERVED" if n == "0107xxxx" else "AMBIGUOUS_IDENTIFIER_FORMAT_REVIEW"
    if sample == "62941":
        return "ANSWER_PRESERVED" if "december19" in n and "1979" in n else "CONTENT_FAILURE_DATE"
    if sample == "64506":
        return "ANSWER_PRESERVED" if n == "e" else "CONTENT_FAILURE_LETTER"
    return "UNREVIEWED"

def load_result(sample: str, method: str) -> dict:
    p = ROOT / "iterative_v6_step3_fixed" / sample / method / method / sample / "result.json"
    return json.loads(p.read_text())

def main() -> None:
    methods = ["native_fdp", "complete_target", "complete_front"]
    rows = []
    for sample in GT:
        for method in methods:
            result = load_result(sample, method)
            updates = [u for u in result["updates"] if u.get("accepted")]
            states = []
            for u in updates:
                g = u["guard"]
                states.append({
                    "accepted_after": u["accepted_after"],
                    "text": g["fast_text"],
                    "semantic": semantic(sample, g["fast_text"]),
                    "target_exact": g["fast_text"].strip() == result["wrong_sequence"].strip(),
                    "full_exact": bool(g["full_text_exact"] and g["full_token_exact"] and g["full_replay"]),
                    "png_linf": u["uint8_linf"],
                })
            first_error = next((x for x in states if x["semantic"].startswith("CONTENT_FAILURE")), None)
            first_target = next((x for x in states if x["target_exact"]), None)
            final = states[-1]
            rows.append({
                "sample_id": sample,
                "method": method,
                "clean_fast_text": result["clean_fast_text"],
                "clean_status": GT[sample]["status"],
                "target_text": result["wrong_sequence"],
                "target_type_review": TARGET_TYPE[sample],
                "accepted": result["accepted"],
                "attempts": result["attempts"],
                "wall_seconds": result["wall_seconds"],
                "final": final,
                "first_content_failure": first_error,
                "first_target_hit": first_target,
                "ever_content_failure": first_error is not None,
                "ever_target_hit": first_target is not None,
                "all_full_exact": all(x["full_exact"] for x in states),
                "script_sha256": result["script_sha256"],
                "result_path": str((ROOT / "iterative_v6_step3_fixed" / sample / method / method / sample / "result.json").relative_to(ROOT.parent)),
            })
    payload = {
        "status": "AUDITED_EXISTING_V26_RESULTS",
        "source_result_sha256": {str(p.relative_to(ROOT.parent)): sha(p) for p in ROOT.glob("iterative_v6_step3_fixed/*/*/*/*/result.json")},
        "rows": rows,
        "semantic_rule": "Explicit per-question rules; person identity and invalid target type remain review labels.",
        "unseen_opened": False,
    }
    (OUT / "existing_result_audit.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    lines = ["# Existing result audit", "", "This report separates final state, first content error, and exact target hit.", "", "|sample|method|final|first content error|target hit|Full exact|seconds|", "|---|---|---|---|---|---|---:|"]
    for r in rows:
        lines.append(f"|{r['sample_id']}|{r['method']}|{r['final']['text']} ({r['final']['semantic']})|{(r['first_content_failure'] or {}).get('text','-')}|{(r['first_target_hit'] or {}).get('text','-')}|{r['all_full_exact']}|{r['wall_seconds']:.1f}|")
    lines += ["", "## Interpretation", "", "A target miss is not counted as an attack miss when the final answer is a reviewed content error. 50779 remains ambiguous because adding a first name may or may not identify a different person. 64506's frozen target is invalid for a letter question, so its target-hit metric is not a valid method metric.", "", "Unseen samples were not opened."]
    (OUT / "existing_result_audit.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT / 'existing_result_audit.md'}")

if __name__ == "__main__":
    main()
