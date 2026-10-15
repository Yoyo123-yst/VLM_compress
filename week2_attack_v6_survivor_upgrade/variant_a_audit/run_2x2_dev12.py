"""CC/CA/AC/AA on the 12 A-selected PNGs. Does not change the selector."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
V5B = ROOT / "week2_attack_v5b_expanded_pristine40"
OUT = ROOT / "week2_attack_v6_survivor_upgrade" / "variant_a_audit"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "week1_mechanism"))
sys.path.insert(0, str(ROOT / "gate_experiment"))

from metrics_task import anls_score  # noqa: E402
from run_causal_v2 import budget, load_model  # noqa: E402
from week2_attack_v4_answer_aligned.common import keep_hash  # noqa: E402
from week2_attack_v5b_expanded_pristine40.load_samples import load_formal_items, manifest  # noqa: E402
from week2_attack_v5b_expanded_pristine40.mechanism_audit_v1.run_counterfactuals import fastv, plain  # noqa: E402

DEV = ["1025", "2008", "7006", "39204", "47450", "47567", "50779", "51133", "60700", "6194", "62941", "64506"]
WRONG = 0.5


def mechanism(ac_wrong, ca_wrong, aa_wrong, valid):
    if not valid:
        return "UNEXPLAINED_OR_INVALID"
    if ac_wrong and not ca_wrong:
        return "CONTENT_SUFFICIENT"
    if ca_wrong and not ac_wrong:
        return "SELECTION_SUFFICIENT"
    if (not ac_wrong) and (not ca_wrong) and aa_wrong:
        return "INTERACTION_REQUIRED"
    if ac_wrong and ca_wrong:
        return "BOTH_INDEPENDENTLY_SUFFICIENT"
    return "UNEXPLAINED_OR_INVALID"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--nshards", type=int, default=4)
    args = parser.parse_args()
    chosen = {row["sample_id"]: row for row in csv.DictReader((OUT / "per_sample_selector.csv").open())}
    ids = [sid for i, sid in enumerate(DEV) if i % args.nshards == args.shard]
    specs = [row for row in manifest()["samples"] if row["sample_id"] in ids]
    items = {item["meta"]["sample_id"]: item for item in load_formal_items(specs)}
    model, processor = load_model()
    rows = []
    keeps = []
    for sid in ids:
        item = items[sid]
        pick = chosen[sid]
        image = Image.open(V5B / "attacks" / "artifacts" / sid / pick["png"]).convert("RGB")
        attacked = list(item["images"])
        attacked[item["aux_idx"]] = image
        question = item["meta"]["question"]
        answers = item["meta"]["answers"]
        t0 = time.time()
        cc, cc_keep, cc_spans, _ = fastv(model, processor, item["images"], question)
        aa, aa_keep, aa_spans, _ = fastv(model, processor, attacked, question)
        ac, _, _, _ = fastv(model, processor, attacked, question, cc_keep)
        ca, _, _, _ = fastv(model, processor, item["images"], question, aa_keep)
        full_c, _ = plain(model, processor, item["images"], question)
        full_a, _ = plain(model, processor, attacked, question)
        scores = {name: anls_score(pred, answers) for name, pred in (("CC", cc), ("CA", ca), ("AC", ac), ("AA", aa))}
        valid = all(text.strip() for text in (cc, ca, ac, aa, full_c, full_a))
        label = mechanism(scores["AC"] < WRONG, scores["CA"] < WRONG, scores["AA"] < WRONG, valid)
        cc_stats = budget(cc_keep, cc_spans, item["evidence_idx"], item["aux_idx"], [])
        aa_stats = budget(aa_keep, aa_spans, item["evidence_idx"], item["aux_idx"], [])
        sym = len(set(cc_keep) ^ set(aa_keep))
        union = len(set(cc_keep) | set(aa_keep))
        rows.append({
            "sample_id": sid,
            "step": pick["step"],
            "pred_CC": cc, "pred_CA": ca, "pred_AC": ac, "pred_AA": aa,
            "full_clean": full_c, "full_selected": full_a,
            "anls_CC": scores["CC"], "anls_CA": scores["CA"], "anls_AC": scores["AC"], "anls_AA": scores["AA"],
            "mechanism": label,
            "output_valid": int(valid),
            "seconds": round(time.time() - t0, 2),
        })
        keeps.append({
            "sample_id": sid,
            "K": len(aa_keep),
            "keep_hash_clean": keep_hash(cc_keep),
            "keep_hash_adv": keep_hash(aa_keep),
            "jaccard": (len(set(cc_keep) & set(aa_keep)) / union) if union else 1.0,
            "symmetric_difference": sym,
            "evidence_share_clean": cc_stats["evidence_share"],
            "evidence_share_adv": aa_stats["evidence_share"],
            "aux_share_clean": cc_stats["aux_share"],
            "aux_share_adv": aa_stats["aux_share"],
            "per_page_clean": json.dumps(cc_stats["kept_per_image"]),
            "per_page_adv": json.dumps(aa_stats["kept_per_image"]),
        })
        (OUT / "keep_lists").mkdir(exist_ok=True)
        (OUT / "keep_lists" / f"{sid}.json").write_text(json.dumps({"clean": cc_keep, "adv": aa_keep}) + "\n")
        print(json.dumps({"id": sid, "mechanism": label, "AA": scores["AA"], "AC": scores["AC"], "CA": scores["CA"]}), flush=True)
    dest = OUT / f"cells_s{args.shard}.json"
    dest.write_text(json.dumps({"rows": rows, "keeps": keeps}, indent=2) + "\n")


if __name__ == "__main__":
    main()
