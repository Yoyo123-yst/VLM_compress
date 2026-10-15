"""Disk replay of already selected PNGs. A mismatch does not choose another step."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "week1_mechanism"))
sys.path.insert(0, str(ROOT / "gate_experiment"))

from run_causal_v2 import load_model, prepare  # noqa: E402
from week2_attack_v4_answer_aligned.run_smoke import Session, evaluate_uint8  # noqa: E402
from week2_attack_v5b_expanded_pristine40.load_samples import load_formal_items, manifest  # noqa: E402
from week2_attack_v6c_split_calibration.run_v6c import make_session  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--nshards", type=int, default=4)
    args = parser.parse_args()
    rows = [row for i, row in enumerate(csv.DictReader(args.selected.open())) if i % args.nshards == args.shard]
    specs = [row for row in manifest()["samples"] if row["sample_id"] in {item["sample_id"] for item in rows}]
    items = {item["meta"]["sample_id"]: item for item in load_formal_items(specs)}
    model, processor = load_model()
    for param in model.parameters():
        param.requires_grad_(False)
    out_rows = []
    for row in rows:
        sid = row["sample_id"]
        if row["status"] != "SELECTED":
            out_rows.append({"sample_id": sid, "ok": False, "status": row["status"]})
            continue
        session = make_session(model, processor, items[sid])
        image = Image.open(Path(row["artifact"]) / row["png"]).convert("RGB")
        measured = evaluate_uint8(session, image, 0.0)
        ok = measured["pred_full"] == row["pred_full"] and measured["pred_fastv"] == row["pred_fastv"]
        out_rows.append({
            "sample_id": sid,
            "ok": ok,
            "status": "REPLAY_OK" if ok else "SELECTED_CANDIDATE_REPLAY_FAILED",
            "pred_full": measured["pred_full"],
            "pred_fastv": measured["pred_fastv"],
        })
        print(json.dumps(out_rows[-1]), flush=True)
        del session
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
