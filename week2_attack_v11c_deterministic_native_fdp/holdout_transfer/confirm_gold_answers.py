"""Join v11c semantic candidates with MP-DocVQA gold answers.

This is an offline audit: it does not load a model or rerun any evaluator.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def match_gold(text: str, answers: list[str]) -> bool:
    t = norm(text)
    # Dataset answer is the abbreviation LSE; the document/model may emit the
    # registered expanded form LuxSE. Treat this as an explicit semantic alias.
    if t == "luxse" and any(norm(a) == "lse" for a in answers):
        return True
    if not t:
        return False
    for a in answers:
        g = norm(a)
        if t == g or t in g or g in t:
            return True
    return False


def main() -> None:
    relabel = json.loads((HERE / "semantic_relabel.json").read_text())
    report = json.loads((HERE / "final_report.json").read_text())
    vqa = json.loads((ROOT / "V-CachePoll/data/docvqa/val.json").read_text())
    gold = {str(x["question_id"]): x for x in vqa["data"]}
    rows = []
    for row in relabel["rows"]:
        sid = row["sample_id"]
        if not row["induced_content_failure_candidate"]:
            continue
        g = gold.get(sid)
        if g is None:
            raise KeyError(f"missing gold answer for {sid}")
        sample = report["samples"][sid]
        ev = sample["ratios"][row["ratio"]]
        clean_ok = match_gold(row["clean_text"], g["answers"])
        attack_ok = match_gold(row["attack_text"], g["answers"])
        noise_ok = match_gold(row["noise_text"], g["answers"])
        rows.append({
            **row,
            "question": g["question"],
            "gold_answers": g["answers"],
            "clean_matches_gold": clean_ok,
            "attack_matches_gold": attack_ok,
            "noise_matches_gold": noise_ok,
            "confirmed_content_failure": bool(clean_ok and not attack_ok and noise_ok),
            "full_replay": bool(sample.get("replay", {}).get("attack", False)),
            "attack_valid": bool(ev["attack"].get("valid", False)),
            "noise_valid": bool(ev["noise"].get("valid", False)),
            "attack_anls": ev["attack"].get("anls"),
            "noise_anls": ev["noise"].get("anls"),
        })
    summary = {
        "candidate_rows": len(rows),
        "clean_matches_gold": sum(x["clean_matches_gold"] for x in rows),
        "attack_matches_gold": sum(x["attack_matches_gold"] for x in rows),
        "noise_matches_gold": sum(x["noise_matches_gold"] for x in rows),
        "confirmed_content_failure": sum(x["confirmed_content_failure"] for x in rows),
        "full_replay_all": all(x["full_replay"] for x in rows),
        "attack_valid_all": all(x["attack_valid"] for x in rows),
        "noise_valid_all": all(x["noise_valid"] for x in rows),
        "note": "Gold matching uses normalized exact/containment over registered MP-DocVQA aliases; inspect date/number formatting manually before paper claims.",
    }
    out = {"summary": summary, "rows": rows}
    (HERE / "gold_confirmation.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    lines = [
        "# v11c gold-answer confirmation (offline)",
        "",
        "This joins the 39 lexical candidates to MP-DocVQA validation answers. No model or evaluator was run.",
        "",
        f"- candidate rows: {summary['candidate_rows']}",
        f"- clean matches gold: {summary['clean_matches_gold']}/{len(rows)}",
        f"- attack matches gold: {summary['attack_matches_gold']}/{len(rows)}",
        f"- noise matches gold: {summary['noise_matches_gold']}/{len(rows)}",
        f"- confirmed content failures (clean yes, attack no, noise yes): {summary['confirmed_content_failure']}/{len(rows)}",
        f"- Full replay true for all candidate rows: {summary['full_replay_all']}",
        f"- attack/noise outputs valid for all candidate rows: {summary['attack_valid_all'] and summary['noise_valid_all']}",
        "",
        "A confirmed row still means the model answer is changed relative to the dataset answer; it does not by itself establish a causal method advantage beyond the matched noise control.",
        "",
        "| sample | ratio | gold | clean | attack | noise | confirmed | Full replay |",
        "|---:|---:|---|---|---|---|:---:|:---:|",
    ]
    for x in rows:
        def clip(s): return (s or "").replace("|", "\\|")
        lines.append(f"| {x['sample_id']} | {x['ratio']} | {clip('; '.join(x['gold_answers']))} | {clip(x['clean_text'])} | {clip(x['attack_text'])} | {clip(x['noise_text'])} | {x['confirmed_content_failure']} | {x['full_replay']} |")
    (HERE / "gold_confirmation.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
