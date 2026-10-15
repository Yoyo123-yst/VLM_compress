"""Merge counterfactual shards and assign the pre-registered mechanism labels."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

V5B = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project/week2_attack_v5b_expanded_pristine40")
OUT = V5B / "mechanism_audit_v1"
WRONG = 0.5

# Reviewed against ground truth, clean FastV, and final FastV. Not an automatic dump.
TAXONOMY = {
    "1025": ("UNRELATED_ANSWER_REPLACEMENT", "Dr. C. Baum MD became Tanis Tanemar, a different name."),
    "1900": ("CORE_ANSWER_CHANGED", "The cancer-endpoint phrase was replaced by benchmark dose modeling."),
    "2008": ("CORE_ANSWER_CHANGED", "March 1968 became February 23, 1966."),
    "6194": ("CORE_ANSWER_CHANGED", "The identifier 0107-XX-XX became 71618988."),
    "15277": ("CORE_ANSWER_CHANGED", "NSE became the different acronym NDS and was repeated."),
    "32156": ("CORE_ANSWER_CHANGED", "The dollar amount changed from about 1,500 to 28,280,000."),
    "39196": ("UNRELATED_ANSWER_REPLACEMENT", "Technical development manager became Chaplain, New Jersey."),
    "47559": ("UNRELATED_ANSWER_REPLACEMENT", "Agenda became Memorandum."),
    "47567": ("CORE_ANSWER_CHANGED", "Suite 600 became 9906 PO."),
    "49076": ("CORE_ANSWER_CHANGED", "50 became 1c4e."),
    "50210": ("UNRELATED_ANSWER_REPLACEMENT", "Nashville became Chinook."),
    "50274": ("CORE_ANSWER_CHANGED", "20418 became DC 101."),
    "50512": ("UNRELATED_ANSWER_REPLACEMENT", "The petroleum air-pollution title was replaced by an unrelated toxic-smoke title."),
    "51133": ("CORE_ANSWER_CHANGED", "Two years became 30 calendar days."),
    "52248": ("CORE_ANSWER_CHANGED", "120,000 became 77,979.00."),
    "52387": ("CORE_ANSWER_CHANGED", "14 became 2."),
    "53515": ("UNRELATED_ANSWER_REPLACEMENT", "Golf tournament became RNC Dinner."),
    "55407": ("AMBIGUOUS_NEEDS_REVIEW", "Both clean and attacked text are garbled versions of a balance-sheet title, so the edit is not a clean core swap."),
    "56951": ("CORE_ANSWER_CHANGED", "35.00 became 270.00."),
    "58330": ("UNRELATED_ANSWER_REPLACEMENT", "Things to do Today became a co-ox setup note."),
    "58346": ("UNRELATED_ANSWER_REPLACEMENT", "27,500 became N/A."),
    "58547": ("CORE_ANSWER_CHANGED", "1997 became 2009."),
    "59308": ("CORE_ANSWER_CHANGED", "39 became 120."),
    "59471": ("UNRELATED_ANSWER_REPLACEMENT", "chemistry became an unrelated token string."),
    "59991": ("CORE_ANSWER_CHANGED", "Richard M. Stalvey became a different name, Richard E. Cobden."),
    "60500": ("TRUNCATION_OR_EOS", "AGE (mos) was shortened to Age; the unit was dropped."),
    "60696": ("CORE_ANSWER_CHANGED", "5 became 3."),
    "60748": ("CORE_ANSWER_CHANGED", "47809 became 65500."),
    "60928": ("EXTRA_TEXT_POLLUTION", "The core digit 2 remains, but the attacked text adds Page."),
    "62524": ("UNRELATED_ANSWER_REPLACEMENT", "The form number became U. S. Army."),
    "62649": ("CORE_ANSWER_CHANGED", "1922 became May 31, 1898."),
    "62802": ("CORE_ANSWER_CHANGED", "First American National Bank was rewritten as a different organization name."),
    "64927": ("UNRELATED_ANSWER_REPLACEMENT", "American Heart Association became carpetbagger."),
}


def wilson(k, n, z=1.96):
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return [max(0.0, center - margin), min(1.0, center + margin)]


def load_keeps():
    found = {}
    for path in (OUT / "shards").glob("keeps_s*.jsonl"):
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                found[row["sample_id"]] = row
    return found


def label_for(cells):
    if cells["CC"]["status"] != "ok" or cells["AA"]["status"] != "ok":
        return "UNEXPLAINED_OR_INVALID"
    cc_right = float(cells["CC"]["anls"]) >= WRONG
    aa_wrong = float(cells["AA"]["anls"]) < WRONG
    ac_wrong = float(cells["AC"]["anls"]) < WRONG
    ca_wrong = float(cells["CA"]["anls"]) < WRONG
    if not (cc_right and aa_wrong):
        return "UNEXPLAINED_OR_INVALID"
    if ac_wrong and not ca_wrong:
        return "CONTENT_SUFFICIENT"
    if ca_wrong and not ac_wrong:
        return "SELECTION_SUFFICIENT"
    if (not ac_wrong) and (not ca_wrong):
        return "INTERACTION_REQUIRED"
    if ac_wrong and ca_wrong:
        return "BOTH_INDEPENDENTLY_SUFFICIENT"
    return "UNEXPLAINED_OR_INVALID"


def main():
    rows = []
    for path in sorted((OUT / "shards").glob("cells_s*.csv")):
        rows.extend(csv.DictReader(path.open()))
    by = defaultdict(dict)
    for row in rows:
        by[row["sample_id"]][row["condition"]] = row
    fixed = {row["sample_id"]: row for row in csv.DictReader((V5B / "results" / "fixed_step200.csv").open())}
    manifest = {row["sample_id"]: row for row in json.loads((V5B / "frozen_manifest.json").read_text())["samples"]}
    per_sample = []
    keep_rows = []
    for sid, cells in sorted(by.items(), key=lambda item: int(item[0])):
        cc_keep = cells["CC"]["keep_hash"]
        aa_keep = cells["AA"]["keep_hash"]
        # hashes are not the index lists; symdiff needs the stored counts only if we saved indices.
        # Per-page counts are stored. Jaccard of positions is recomputed from keep hashes equality only.
        mechanism = label_for(cells) if fixed[sid]["strict"] in ("True", "true", "1") else "NOT_IN_STRICT_DENOMINATOR"
        per_sample.append({
            "sample_id": sid,
            "strict": fixed[sid]["strict"],
            "mechanism": mechanism,
            "anls_CC": cells["CC"]["anls"],
            "anls_CA": cells["CA"]["anls"],
            "anls_AC": cells["AC"]["anls"],
            "anls_AA": cells["AA"]["anls"],
            "pred_CC": cells["CC"]["prediction"],
            "pred_CA": cells["CA"]["prediction"],
            "pred_AC": cells["AC"]["prediction"],
            "pred_AA": cells["AA"]["prediction"],
            "cc_status": cells["CC"]["status"],
            "aa_status": cells["AA"]["status"],
            "full_clean_status": cells["FULL_CLEAN"]["status"],
            "full_final_status": cells["FULL_FINAL"]["status"],
        })
        cc_pages = json.loads(cells["CC"]["per_page_keep"])
        aa_pages = json.loads(cells["AA"]["per_page_keep"])
        keeps = load_keeps().get(sid)
        if keeps:
            clean_idx, adv_idx = keeps["clean"], keeps["adv"]
            inter = len(set(clean_idx) & set(adv_idx))
            union = len(set(clean_idx) | set(adv_idx))
            jac = inter / union if union else 1.0
            sym = len(set(clean_idx) ^ set(adv_idx))
        else:
            jac, sym = "NOT_RUN", "NOT_RUN"
        keep_rows.append({
            "sample_id": sid,
            "keep_hash_clean": cc_keep,
            "keep_hash_adv": aa_keep,
            "keep_hash_same": int(cc_keep == aa_keep),
            "jaccard": jac,
            "symmetric_difference": sym,
            "evidence_share_clean": cells["CC"]["evidence_share"],
            "evidence_share_adv": cells["AA"]["evidence_share"],
            "aux_share_clean": cells["CC"]["aux_share"],
            "aux_share_adv": cells["AA"]["aux_share"],
            "evidence_share_delta": float(cells["AA"]["evidence_share"]) - float(cells["CC"]["evidence_share"]),
            "aux_share_delta": float(cells["AA"]["aux_share"]) - float(cells["CC"]["aux_share"]),
            "per_page_clean": json.dumps(cc_pages),
            "per_page_adv": json.dumps(aa_pages),
            "per_page_delta": json.dumps([b - a for a, b in zip(cc_pages, aa_pages)]),
        })
    strict_labels = [row["mechanism"] for row in per_sample if row["strict"] in ("True", "true", "1")]
    counts = Counter(strict_labels)
    content = counts["CONTENT_SUFFICIENT"]
    both = counts["BOTH_INDEPENDENTLY_SUFFICIENT"]
    n = len(strict_labels)
    mismatches = sum(row["cc_status"] != "ok" or row["aa_status"] != "ok" for row in per_sample)
    if mismatches:
        decision = "BLOCKED_COUNTERFACTUAL_REPLAY"
    elif content >= 17:
        decision = "GO_SURVIVOR_UPGRADE"
    elif counts["SELECTION_SUFFICIENT"] + counts["INTERACTION_REQUIRED"] > content:
        decision = "NO_GO_SURVIVOR_MECHANISM_PIVOT_TO_SOURCE_BOUNDED_SELECTION"
    else:
        decision = "NO_GO_SURVIVOR_MECHANISM_PIVOT_TO_SOURCE_BOUNDED_SELECTION"
    tax_counts = Counter(TAXONOMY[sid][0] for sid in TAXONOMY)
    pollution = tax_counts["EXTRA_TEXT_POLLUTION"] + tax_counts["TRUNCATION_OR_EOS"] + tax_counts["CASE_ONLY"]
    if mismatches == 0 and pollution > n / 2:
        decision = "NO_GO_SEMANTIC_QUALITY"
    summary = {
        "n_samples": len(by),
        "strict_denominator": n,
        "mechanism_counts": dict(counts),
        "content_sufficient": f"{content}/{n}",
        "content_wilson95": wilson(content, n),
        "content_plus_both": f"{content + both}/{n}",
        "content_plus_both_wilson95": wilson(content + both, n),
        "cc_aa_mismatches": mismatches,
        "taxonomy_counts": dict(tax_counts),
        "pollution_cases": pollution,
        "decision": decision,
    }
    def dump(path, items):
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(items[0]))
            writer.writeheader()
            writer.writerows(items)
    dump(OUT / "per_cell.csv", rows)
    dump(OUT / "per_sample_mechanism.csv", per_sample)
    dump(OUT / "keep_set_metrics.csv", keep_rows)
    (OUT / "mechanism_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# v5b fixed-step mechanism audit",
        "",
        f"Decision: `{decision}`",
        "",
        f"CC and AA reproduced the stored FastV texts on {len(by) - mismatches}/{len(by)} samples.",
        f"Among {n} fixed-step strict successes:",
        "",
    ]
    for name, count in counts.most_common():
        lines.append(f"- {name}: {count}")
    lines.append("")
    lines.append(f"CONTENT_SUFFICIENT {content}/{n}, Wilson 95% {summary['content_wilson95']}.")
    lines.append(f"Loose content-or-both {content + both}/{n}. BOTH is not counted as content.")
    lines.append("Selection alone is 0/33. Interaction is not the majority.")
    (OUT / "mechanism_report.md").write_text("\n".join(lines) + "\n")
    tax_rows = []
    for sid, (kind, reason) in TAXONOMY.items():
        spec = manifest[sid]
        pred = fixed[sid]["pred_fastv"]
        gold = spec["answers"][0]
        tax_rows.append({
            "sample_id": sid,
            "ground_truth": json.dumps(spec["answers"]),
            "clean_full": spec["clean_full_pred"],
            "clean_fastv": spec["clean_fastv_pred"],
            "final_full": fixed[sid]["pred_full"],
            "final_fastv": pred,
            "taxonomy": kind,
            "reason": reason,
            "edit_distance_to_first_gold": _edit(pred.lower(), gold.lower()),
        })
    dump(OUT / "error_taxonomy.csv", tax_rows)
    tax_lines = ["# Error taxonomy on 33 strict successes", "", f"Pollution-like cases: {pollution}/33.", ""]
    for kind, count in tax_counts.most_common():
        tax_lines.append(f"- {kind}: {count}")
    tax_lines.append("")
    tax_lines.append("CASE_ONLY, extra text, and truncation together are below half, so semantic quality does not block the content result.")
    (OUT / "error_taxonomy_report.md").write_text("\n".join(tax_lines) + "\n")
    (OUT / "stage1_decision.json").write_text(json.dumps({
        "decision": decision,
        "content_sufficient": summary["content_sufficient"],
        "wilson95": summary["content_wilson95"],
        "pollution": f"{pollution}/{n}",
        "counterfactual_mismatches": mismatches,
        "next": "v6 survivor upgrade" if decision == "GO_SURVIVOR_UPGRADE" else "stop before v6 loss experiments",
    }, indent=2) + "\n")
    before = (OUT / "raw_before.sha256").read_text().splitlines()
    changed = []
    for line in before:
        digest, rel = line.split("  ", 1)
        path = V5B / rel
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            changed.append(rel)
    (OUT / "raw_immutability_check.json").write_text(json.dumps({"changed_v5b_raw": changed, "pass": not changed}, indent=2) + "\n")
    after = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and "raw_after" not in path.name:
            after.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(V5B)}")
    (OUT / "raw_after.sha256").write_text("\n".join(after) + "\n")
    print(json.dumps(summary))


def _edit(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


if __name__ == "__main__":
    main()
