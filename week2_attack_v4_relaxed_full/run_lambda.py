"""Same v4 attack with one changed multiplier: --lambda-full.

Scales stay the ones frozen on 50743. Outputs never enter the v4 directory.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
sys.path.insert(0, str(ROOT))

import week2_attack_v4_answer_aligned.run_smoke as smoke  # noqa: E402
from week2_attack_v4_answer_aligned.common import load_frozen_samples, write_json  # noqa: E402
from week2_attack_v4_answer_aligned.run_smoke import (  # noqa: E402
    FIELDS,
    Session,
    attack,
    done_ids,
)
from gate_experiment.run_causal_v2 import append_csv, load_model  # noqa: E402

OUT = ROOT / "week2_attack_v4_relaxed_full"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda-full", type=float, required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--ids", default="50743,4912,56776,49242,33923,5169")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshards", type=int, default=1)
    parser.add_argument("--manifest", default="")
    args = parser.parse_args()
    torch.manual_seed(42)
    smoke.LAMBDA_FULL = float(args.lambda_full)
    if args.manifest:
        samples = load_manifest(Path(args.manifest))
    else:
        wanted = [item for item in args.ids.split(",") if item]
        samples = [item for item in load_frozen_samples() if item["meta"]["sample_id"] in wanted]
    samples = [item for index, item in enumerate(samples) if index % args.nshards == args.shard]
    scales = json.loads(smoke.SCALE_PATH.read_text())
    if abs(float(scales["lambda_full"]) - 1.0) > 1e-8:
        raise SystemExit("v4 scale file is not the frozen lambda=1 calibration")
    tag_dir = OUT / "runs" / args.tag
    tag_dir.mkdir(parents=True, exist_ok=True)
    curve = tag_dir / f"checkpoints_s{args.shard}.jsonl"
    csv_path = tag_dir / f"shard_s{args.shard}.csv"
    finished = done_ids(csv_path)
    model, processor = load_model()
    for param in model.parameters():
        param.requires_grad_(False)
    write_json(tag_dir / "config.json", {
        "lambda_full": args.lambda_full,
        "steps": args.steps,
        "scales": scales,
        "changed_from_v4": ["lambda_full"],
        "shard": args.shard,
        "nshards": args.nshards,
        "ids": [item["meta"]["sample_id"] for item in samples],
    })
    for item in samples:
        sid = item["meta"]["sample_id"]
        if sid in finished:
            continue
        smoke.GROUPS.setdefault(item["meta"]["sample_id"], "holdout")
        clear_inputs = prepare_answer(model, processor, item)
        session = Session(model, processor, item, clear_inputs)
        row, seconds = attack(
            session,
            args.steps,
            scales,
            None,
            curve,
            tag_dir / sid,
        )
        row["lambda_full"] = f"{args.lambda_full:.4f}"
        row["seconds"] = f"{seconds:.1f}"
        append_csv(csv_path, row, FIELDS + ["lambda_full", "seconds"])
        print(json.dumps({"done": sid, "seconds": seconds, "label": row["label"]}), flush=True)
        del session
        torch.cuda.empty_cache()


def load_manifest(path: Path) -> list[dict]:
    import csv
    from gate_experiment.run_step0_v2 import draw_questions, load_images, select_pages
    from week2_attack_v4_answer_aligned.common import BASELINE, RATIO, sha_image

    manifest = json.loads(path.read_text())
    questions = {row["sample_id"]: row for row in draw_questions(400, 42)}
    metas = []
    for row in manifest["samples"]:
        if row["sample_id"] not in questions:
            raise RuntimeError(f"holdout id missing from the seed-42 draw: {row['sample_id']}")
        metas.append(questions[row["sample_id"]])
    pages = load_images(metas)
    base: dict[str, dict] = {}
    with BASELINE.open() as handle:
        for row in csv.DictReader(handle):
            if row["method"] == "full":
                base.setdefault(row["sample_id"], {})["full"] = row
            if row["method"] == "fastv" and abs(float(row["r"]) - RATIO) < 1e-6:
                base.setdefault(row["sample_id"], {})["fastv"] = row
    samples = []
    for meta, spec in zip(metas, manifest["samples"]):
        images, evidence_idx, order, dropped = select_pages(
            pages[meta["sample_id"]], meta["answer_page_idx"], meta["sample_id"], 768
        )
        sha = [sha_image(image) for image in images]
        if sha != spec["page_sha"] or order != spec["order"] or evidence_idx != spec["evidence_idx"]:
            raise RuntimeError(f"holdout manifest mismatch for {meta['sample_id']}")
        aux_idx = next(i for i in range(len(images)) if i != evidence_idx)
        samples.append({
            "meta": meta,
            "images": images,
            "evidence_idx": evidence_idx,
            "aux_idx": aux_idx,
            "order": order,
            "dropped": dropped,
            "page_sha": sha,
            "baseline_full": base[meta["sample_id"]]["full"],
            "baseline_fastv": base[meta["sample_id"]]["fastv"],
        })
    return samples


def prepare_answer(model, processor, item):
    from adapters.runtime import clear_hooks
    from run_causal_v2 import generate, prepare
    clear_hooks(model)
    inputs = prepare(processor, item["images"], item["meta"]["question"])
    raw = generate(model, inputs)
    return torch.tensor(raw.detach().cpu().tolist(), dtype=torch.long)


if __name__ == "__main__":
    main()
