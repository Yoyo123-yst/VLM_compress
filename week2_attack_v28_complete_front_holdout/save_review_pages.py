"""Save source evidence/auxiliary pages for manual target-answer review."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from week2_attack_v28_complete_front_holdout.load_samples import load_item

OUT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True)
    args = parser.parse_args()
    item = load_item(args.sample)
    dest = OUT / "source_pages" / args.sample
    dest.mkdir(parents=True, exist_ok=True)
    files = {}
    for label, index in (("evidence", item["evidence_idx"]), ("auxiliary", item["aux_idx"])):
        path = dest / f"{label}.png"
        item["images"][index].convert("RGB").save(path, format="PNG")
        files[label] = {"path": str(path.relative_to(OUT)), "page_index": int(index),
                        "sha256": item["spec"]["page_sha256"][index]}
    record = {"sample_id": args.sample, "question": item["meta"]["question"],
              "answers": item["meta"]["answers"], "pages": files,
              "role_layout": {"evidence_idx": item["evidence_idx"], "aux_idx": item["aux_idx"]}}
    (dest / "review.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
