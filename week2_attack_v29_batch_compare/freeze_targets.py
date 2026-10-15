"""Freeze the pre-registered clean-only target review for the v29 batch."""
from __future__ import annotations
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent

# Manual semantic review is intentionally conservative: only complete, answer-type
# compatible, clearly non-equivalent targets enter the attack. No attack outcome is
# read here.
PICKS = {
    "55341": (0, "VALID_COMPLETE_WRONG_NUMBER"),
    "55192": (0, "VALID_COMPLETE_WRONG_BOOLEAN"),
    "63843": (4, "VALID_COMPLETE_WRONG_NUMBER"),
    "56601": (0, "VALID_COMPLETE_WRONG_YEAR"),
    "57069": (0, "VALID_COMPLETE_WRONG_PAGE_NUMBER"),
    "4415": (0, "VALID_COMPLETE_WRONG_OBJECTIVE"),
    "65311": (0, "VALID_COMPLETE_WRONG_LOCATION"),
    "59042": (0, "VALID_COMPLETE_WRONG_AMOUNT"),
    "58647": (1, "VALID_COMPLETE_WRONG_WARNING"),
    "5942": (5, "VALID_COMPLETE_WRONG_NUMBER"),
    "50763": (0, "VALID_COMPLETE_WRONG_BOOLEAN"),
    "4449": (2, "VALID_COMPLETE_WRONG_PRODUCT"),
    "8560": (0, "VALID_COMPLETE_WRONG_PERSON"),
    "51024": (0, "VALID_COMPLETE_WRONG_EVENT"),
    "63839": (0, "VALID_COMPLETE_WRONG_PERCENTAGE"),
}

def main() -> None:
    admission = json.loads((OUT / "batch_admission.json").read_text())
    specs = {x["sample_id"]: x for x in json.loads((OUT / "sample_specs.json").read_text())["samples_in_frozen_order"]}
    samples = {}
    rejected = []
    for row in admission["selected_order"]:
        sid = row["sample_id"]
        if sid not in PICKS:
            rejected.append({"sample_id": sid, "status": "NO_VALID_PREREGISTERED_SEMANTIC_TARGET_OR_CLEAN_CONTENT_REVIEW"})
            continue
        idx, status = PICKS[sid]
        cand = json.loads((OUT / "candidate_manifest" / f"{sid}.json").read_text())["candidates"][idx]
        if not cand.get("estimated_feasible_at_eps") or cand.get("continuation_lexically_equivalent_to_clean"):
            raise RuntimeError(f"target review precondition failed for {sid}")
        clean = json.loads((OUT / "clean_screen" / f"{sid}.json").read_text())
        samples[sid] = {
            "question": specs[sid]["question"],
            "correct_aliases": list(dict.fromkeys(specs[sid]["answers"] + [clean["clean_fastv_text"]])),
            "clean_fastv_text": clean["clean_fastv_text"],
            "wrong_target_text": cand["forced_continuation"]["text"],
            "wrong_target_tokens": cand["forced_continuation"]["tokens"],
            "candidate_index": idx,
            "candidate_manifest": f"candidate_manifest/{sid}.json",
            "semantic_status": status,
            "semantic_review": "clean-only candidate; complete answer, same answer type, clearly distinct from dataset answer and clean expression",
            "attack_outcomes_used": False,
        }
    payload = {
        "status": "FROZEN_V29_TARGETS_AFTER_CLEAN_ONLY_SEMANTIC_REVIEW",
        "development_only": True, "ratio": 0.25,
        "candidate_protocol": "same clean-only generator and max limits as v28; semantic review before attacks",
        "samples": samples, "rejected_selected_samples": rejected,
        "target_selection_does_not_change_sample_denominator": True,
    }
    (OUT / "frozen_targets.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    (OUT / "target_review.md").write_text("# v29 target review\n\n" + "\n".join(f"- {sid}: {x['wrong_target_text']} ({x['semantic_status']})" for sid, x in samples.items()) + "\n\nRejected selected samples:\n" + "\n".join(f"- {x['sample_id']}: {x['status']}" for x in rejected) + "\n")
    print(json.dumps({"frozen_targets": len(samples), "rejected": len(rejected), "ids": list(samples)}, ensure_ascii=False))

if __name__ == "__main__": main()
