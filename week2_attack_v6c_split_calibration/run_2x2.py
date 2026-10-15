"""CC/CA/AC/AA on selected PNGs. ANLS is an audit label, not a selector input."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "week1_mechanism"))
sys.path.insert(0, str(ROOT / "gate_experiment"))

from metrics_task import anls_score, normalize  # noqa: E402
from run_causal_v2 import budget, load_model  # noqa: E402
from week2_attack_v4_answer_aligned.common import keep_hash  # noqa: E402
from week2_attack_v5b_expanded_pristine40.load_samples import load_formal_items, manifest  # noqa: E402
from week2_attack_v5b_expanded_pristine40.mechanism_audit_v1.run_counterfactuals import fastv, plain  # noqa: E402
from week2_attack_v6b_soft_full_anchor.soft_loss import mechanism_label  # noqa: E402

WRONG = 0.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--nshards", type=int, default=4)
    args = parser.parse_args()
    chosen = list(csv.DictReader(args.selected.open()))
    chosen = [row for i, row in enumerate(chosen) if i % args.nshards == args.shard and row["status"] == "SELECTED"]
    specs = [row for row in manifest()["samples"] if row["sample_id"] in {item["sample_id"] for item in chosen}]
    items = {item["meta"]["sample_id"]: item for item in load_formal_items(specs)}
    model, processor = load_model()
    rows = []
    keeps = []
    for pick in chosen:
        sid = pick["sample_id"]
        item = items[sid]
        image = Image.open(Path(pick["artifact"]) / pick["png"]).convert("RGB")
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
        label = mechanism_label(scores["AC"] < WRONG, scores["CA"] < WRONG, scores["AA"] < WRONG, valid)
        cc_stats = budget(cc_keep, cc_spans, item["evidence_idx"], item["aux_idx"], [])
        aa_stats = budget(aa_keep, aa_spans, item["evidence_idx"], item["aux_idx"], [])
        union = set(cc_keep) | set(aa_keep)
        rows.append({
            "sample_id": sid,
            "step": pick["step"],
            "pred_CC": cc, "pred_CA": ca, "pred_AC": ac, "pred_AA": aa,
            "full_clean": full_c, "full_selected": full_a,
            "anls_CC": scores["CC"], "anls_CA": scores["CA"], "anls_AC": scores["AC"], "anls_AA": scores["AA"],
            "mechanism": label,
            "output_valid": int(valid),
            "pollution": int(
                normalize(aa) != normalize(cc) and (normalize(aa) in normalize(cc) or normalize(cc) in normalize(aa))
            ),
            "seconds": round(time.time() - t0, 2),
        })
        keeps.append({
            "sample_id": sid,
            "K": len(aa_keep),
            "keep_hash_clean": keep_hash(cc_keep),
            "keep_hash_adv": keep_hash(aa_keep),
            "jaccard": (len(set(cc_keep) & set(aa_keep)) / len(union)) if union else 1.0,
            "evidence_share_clean": cc_stats["evidence_share"],
            "evidence_share_adv": aa_stats["evidence_share"],
            "aux_share_clean": cc_stats["aux_share"],
            "aux_share_adv": aa_stats["aux_share"],
        })
        print(json.dumps({"id": sid, "mechanism": label}), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"cells_s{args.shard}.json").write_text(json.dumps({"rows": rows, "keeps": keeps}, indent=2) + "\n")


if __name__ == "__main__":
    main()
