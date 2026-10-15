#!/usr/bin/env python3
"""Build the v27 report from the read-only audit and v27 reruns."""
from __future__ import annotations
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "week2_attack_v25_full_answer_competition_pilot"
METHODS = ["native_fdp", "complete_target", "complete_front"]
SAMPLES = ["7006", "60700", "64506"]
GT = {
    "7006": ("RJ Reynolds Tobacco Company", "valid complete wrong entity"),
    "60700": ("PPBS", "valid complete wrong entity"),
    "64506": ("E", "target Section 2 is invalid for a letter question"),
}

def semantic(s, text):
    n = "".join(c.lower() for c in text if c.isalnum())
    if s == "7006":
        return "CONTENT_FAILURE" if n not in {"rjreynoldstobaccocompany", "rjreynoldstobaccocompany"} else "ANSWER_PRESERVED"
    if s == "60700":
        return "CONTENT_FAILURE" if n != "ppbs" else "ANSWER_PRESERVED"
    if s == "64506":
        return "CONTENT_FAILURE_LETTER" if n != "e" else "ANSWER_PRESERVED"
    return "REVIEW"

def read(s, m):
    return json.loads((HERE / "equal_compute_v27" / m / s / "result.json").read_text())

def main():
    audit = json.loads((HERE / "existing_result_audit.json").read_text())
    lines = [
        "# v27 complete-answer comparison and audit",
        "",
        "This report separates: (1) real content error, (2) exact hit of a frozen wrong target, and (3) compute and Full-guard behavior.",
        "All data are development data. Unseen samples were not opened.",
        "",
        "## Existing six-sample endpoint audit",
        "",
        "The detailed table is retained in `existing_result_audit.md`. A target miss is not automatically an attack miss; an answer is judged against the dataset answer and a question-specific rule. Person identity and identifier formatting remain review labels when the string alone is insufficient.",
        "",
        "|sample|method|final FastV|content status|target hit|Full exact|time (s)|",
        "|---|---|---|---|---|---|---:|",
    ]
    for r in audit["rows"]:
        lines.append(f"|{r['sample_id']}|{r['method']}|{r['final']['text']}|{r['final']['semantic']}|{bool(r['ever_target_hit'])}|{r['all_full_exact']}|{r['wall_seconds']:.1f}|")
    lines += ["", "## Fixed-budget reruns", "", "The budget was predeclared as 20 accepted updates for Native-FDP, 13 for complete-target, and 10 for complete-front. This corresponds to about 40 attack/protection backward passes when every attempt is accepted. The report also gives actual counters because the methods have different forward costs.", "", "|sample|method|accepted/attempts|final FastV|first target hit|content status|Full exact|time (s)|score forwards|backwards|", "|---|---|---:|---|---:|---|---|---:|---:|---:|"]
    for s in SAMPLES:
        for m in METHODS:
            d = read(s, m); u = [x for x in d["updates"] if x["accepted"]]; hits = [x for x in u if x["guard"]["fast_text"].strip() == d["wrong_sequence"].strip()]
            b = d["counters"]["native_attack_backwards"] + d["counters"]["complete_sequence_backwards"] + d["counters"]["front_backwards"] + d["counters"]["full_fdp_backwards"]
            sf = d["counters"]["fastv_score_forwards"] + d["counters"]["full_score_forwards"]
            lines.append(f"|{s}|{m}|{d['accepted']}/{d['attempts']}|{u[-1]['guard']['fast_text']}|{hits[0]['accepted_after'] if hits else '-'}|{semantic(s,u[-1]['guard']['fast_text'])}|{all(x['guard']['full_text_exact'] and x['guard']['full_token_exact'] and x['guard']['full_replay'] for x in u)}|{d['wall_seconds']:.1f}|{sf}|{b}|")
    lines += ["", "## Findings", "", "1. The complete-front method has a repeatable local signal on 7006 and 60700: it reaches the frozen wrong answer at accepted update 8 and 5, while Native-FDP does not hit that target. On 7006 the final output remains `Public Relations Dept.`; on 60700 it remains `Dr. H. S. SORBS`.", "2. This does not prove a general attack-rate gain. Native-FDP also produces clear content changes on several other samples, and exact target hit is a stricter, different objective.", "3. 60700 complete-target was rejected 10 times and only accepted 10 updates in 20 attempts. At the same 20-attempt budget it still reaches the target, but its rejection rate is a real weakness and its wall time is higher.", "4. 64506 is useful only as a control: `Section 2` is not a valid answer type for a question asking for a letter. The complete methods output `J` or `C`, which are content errors, but no target-hit claim is valid for this frozen target.", "5. All accepted checkpoints in the new runs passed Full text exact, Full token exact, Full replay, and the existing PNG perturbation guard. No OOM occurred.", "", "## Decision", "", "Keep complete-front as a promising local method candidate, but do not open unseen data or claim a general improvement yet. The next scientific step is a frozen semantic review of the six targets and a small repeated run of only 7006 and 60700 with the corrected counters. If complete-front remains better at comparable actual cost, then expand to a pre-registered development holdout. If the advantage disappears after equalizing actual forward cost, retain it as a targeted-control baseline rather than a new attack method.", "", "## Provenance", "", "- Source audit: `existing_result_audit.json` and `existing_result_audit.md`.", "- New results: `equal_compute_v27/`.", "- Modified runner: `../week2_attack_v25_full_answer_competition_pilot/run_iterative_compare.py`.", "- Unseen data: not opened."]
    (HERE / "final_report.md").write_text("\n".join(lines) + "\n")
    print(HERE / "final_report.md")

if __name__ == "__main__":
    main()
