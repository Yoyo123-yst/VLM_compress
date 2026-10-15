#!/usr/bin/env python3
"""Relabel existing v11c endpoints without rerunning model inference."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "final_report.json"
OUT_JSON = ROOT / "semantic_relabel.json"
OUT_MD = ROOT / "semantic_relabel.md"


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def nums(s: str) -> tuple[str, ...]:
    return tuple(re.findall(r"\d+", (s or "")))


def roman_tokens(s: str) -> tuple[str, ...]:
    return tuple(re.findall(r"\b[ivxlcdm]+\b", (s or "").lower()))


def roman_answer(s: str) -> tuple[str, ...]:
    tokens = re.findall(r"[a-z]+", (s or "").lower())
    valid = {"i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"}
    return tuple(t for t in tokens if t in valid) if tokens and all(t in valid for t in tokens) else ()


def relation(reference: str, text: str) -> str:
    """Conservative relation to the Full clean reference."""
    a, b = norm(reference), norm(text)
    if not b:
        return "INVALID_OR_EMPTY"
    if a == b:
        return "REFERENCE_EXACT_NORMALIZED"
    if roman_answer(reference) != roman_answer(text):
        # Do not treat V as a harmless substring of VIII, etc.
        if roman_answer(reference) or roman_answer(text):
            return "REFERENCE_ROMAN_CONFLICT_REVIEW"
    if a in b or b in a:
        # This includes harmless punctuation/case/suffix changes, but is not
        # promoted to success when the numbers disagree.
        if nums(reference) and nums(text) and nums(reference) != nums(text):
            return "REFERENCE_NUMERIC_CONFLICT_REVIEW"
        return "ANSWER_PRESERVED_AUGMENTED"
    if nums(reference) and nums(text) and nums(reference) != nums(text):
        return "REFERENCE_NUMERIC_CONFLICT_CANDIDATE"
    return "CONTENT_REPLACEMENT_CANDIDATE"


def main() -> None:
    data = json.loads(SRC.read_text())
    rows = []
    for sid, sample in data["samples"].items():
        reference = sample["clean_text"]
        for ratio, cell in sample["ratios"].items():
            clean = cell["clean"]
            attack = cell["attack"]
            noise = cell["noise"]
            clean_rel = relation(reference, clean["text"])
            attack_rel = relation(reference, attack["text"])
            noise_rel = relation(reference, noise["text"])
            baseline_ok = clean_rel in {
                "REFERENCE_EXACT_NORMALIZED", "ANSWER_PRESERVED_AUGMENTED"
            }
            content_candidate = attack_rel in {
                "CONTENT_REPLACEMENT_CANDIDATE",
                "REFERENCE_NUMERIC_CONFLICT_CANDIDATE",
            }
            induced_candidate = bool(
                baseline_ok
                and content_candidate
                and noise_rel not in {
                    "CONTENT_REPLACEMENT_CANDIDATE",
                    "REFERENCE_NUMERIC_CONFLICT_CANDIDATE",
                }
            )
            rows.append({
                "sample_id": sid,
                "ratio": ratio,
                "reference": reference,
                "clean_text": clean["text"],
                "attack_text": attack["text"],
                "noise_text": noise["text"],
                "clean_relation": clean_rel,
                "attack_relation": attack_rel,
                "noise_relation": noise_rel,
                "clean_baseline_reference_ok": baseline_ok,
                "content_failure_candidate": content_candidate,
                "induced_content_failure_candidate": induced_candidate,
                "original_induced_anls": bool(cell.get("induced", False)),
                "full_replay": bool(sample["replay"]["attack"]),
            })

    counts = {}
    for key in ("clean_relation", "attack_relation", "noise_relation"):
        counts[key] = {}
        for row in rows:
            value = row[key]
            counts[key][value] = counts[key].get(value, 0) + 1
    summary = {
        "source": str(SRC.name),
        "inference_rerun": False,
        "rows": len(rows),
        "counts": counts,
        "original_induced_anls": sum(r["original_induced_anls"] for r in rows),
        "clean_baseline_reference_ok": sum(r["clean_baseline_reference_ok"] for r in rows),
        "content_failure_candidates": sum(r["content_failure_candidate"] for r in rows),
        "induced_content_failure_candidates": sum(
            r["induced_content_failure_candidate"] for r in rows
        ),
        "note": "Conservative lexical/number screen; candidate rows require manual semantic confirmation.",
    }
    OUT_JSON.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2) + "\n")

    lines = [
        "# v11c existing-endpoint semantic relabel",
        "",
        "This report is a read-only relabel of `final_report.json`; no model, PNG, or evaluator was run.",
        "The lexical and numeric screen is conservative. `*_CANDIDATE` is not a confirmed content failure.",
        "",
        f"- rows: {summary['rows']}",
        f"- original ANLS-induced cells: {summary['original_induced_anls']}",
        f"- clean rows matching the Full reference after normalization/alias containment: {summary['clean_baseline_reference_ok']}",
        f"- attack content replacement candidates: {summary['content_failure_candidates']}",
        f"- induced content replacement candidates after matched-noise screen: {summary['induced_content_failure_candidates']}",
        "",
        "## Candidate cells",
        "",
        "| sample | ratio | Full reference | clean FastV | attack FastV | clean relation | attack relation | noise relation | candidate |",
        "|---|---:|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        if row["induced_content_failure_candidate"] or row["original_induced_anls"]:
            vals = [
                row["sample_id"], row["ratio"], row["reference"], row["clean_text"],
                row["attack_text"], row["clean_relation"], row["attack_relation"],
                row["noise_relation"], str(row["induced_content_failure_candidate"]),
            ]
            lines.append("| " + " | ".join(v.replace("|", "\\|") for v in vals) + " |")
    lines += [
        "",
        "## Interpretation",
        "",
        "Rows with a clean/reference mismatch are not treated as clean-start induced failures.",
        "Case-only, punctuation-only, suffix-only, or normalized containment changes are preserved answers.",
        "Numeric conflicts and entity replacements remain candidates until manual answer-level review.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
