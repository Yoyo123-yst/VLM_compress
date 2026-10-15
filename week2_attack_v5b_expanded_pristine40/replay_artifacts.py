"""Reload saved PNGs and rerun Full/FastV generate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
V5B = ROOT / "week2_attack_v5b_expanded_pristine40"
sys.path.insert(0, str(ROOT))

import week2_attack_v4_answer_aligned.run_smoke as smoke  # noqa: E402
from adapters.runtime import clear_hooks  # noqa: E402
from gate_experiment.run_causal_v2 import generate, load_model, prepare  # noqa: E402
from week2_attack_v4_answer_aligned.common import GROUPS  # noqa: E402
from week2_attack_v4_answer_aligned.run_smoke import Session, evaluate_uint8  # noqa: E402
from week2_attack_v5b_expanded_pristine40.load_samples import load_formal_items, manifest  # noqa: E402


def replay_one(session: Session, path: Path, reference: dict) -> dict:
    image = Image.open(path).convert("RGB")
    measured = evaluate_uint8(session, image, 0.0)
    ok = (
        measured["pred_full"] == reference["pred_full"]
        and measured["pred_fastv"] == reference["pred_fastv"]
        and measured["keep_hash"] == reference["keep_hash"]
        and int(measured["k"]) == int(reference["k"])
    )
    return {"path": str(path), "ok": ok, "pred_full": measured["pred_full"], "pred_fastv": measured["pred_fastv"], "keep_hash": measured["keep_hash"], "k": measured["k"], "anls_full": measured["anls_full"], "anls_fastv": measured["anls_fastv"], "linf": measured["linf"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshards", type=int, default=4)
    parser.add_argument("--dev50743", action="store_true")
    parser.add_argument("--tag", default="formal")
    args = parser.parse_args()
    smoke.LAMBDA_FULL = 0.25
    if args.dev50743:
        from week2_attack_v4_answer_aligned.common import load_frozen_samples
        items = [item for item in load_frozen_samples() if item["meta"]["sample_id"] == "50743"]
        root = V5B / "tests" / args.tag / "50743"
        refs = {}
        for line in (V5B / "tests" / args.tag / "checkpoints.jsonl").read_text().splitlines():
            row = json.loads(line)
            refs[int(row["step"])] = row
        noise = json.loads((root / "random_noise_provenance.json").read_text())
        targets = {
            "final": (root / "final.png", refs[args_step(refs)]),
            "min_margin": (root / "min_margin.png", refs[int(json.loads((root / "min_margin_provenance.json").read_text())["step"])]),
            "random_noise": (root / "random_noise.png", noise),
        }
    else:
        payload = manifest()
        specs = [row for index, row in enumerate(payload["samples"]) if index % args.nshards == args.shard]
        items = load_formal_items(specs)
        targets = None
    model, processor = load_model()
    for param in model.parameters():
        param.requires_grad_(False)
    out_rows = []
    for item in items:
        sid = item["meta"]["sample_id"]
        GROUPS.setdefault(sid, "v5b")
        clear_hooks(model)
        inputs = prepare(processor, item["images"], item["meta"]["question"])
        with torch.inference_mode():
            raw = generate(model, inputs)
        answer = torch.tensor(raw.detach().cpu().tolist(), dtype=torch.long)
        session = Session(model, processor, item, answer)
        if args.dev50743:
            local = targets
        else:
            art = V5B / "attacks" / "artifacts" / sid
            rows = []
            for shard in range(4):
                path = V5B / "attacks" / f"shard_s{shard}" / "checkpoints.jsonl"
                if not path.exists():
                    continue
                for line in path.read_text().splitlines():
                    row = json.loads(line)
                    if row["sample_id"] == sid:
                        rows.append(row)
            by_step = {int(row["step"]): row for row in rows}
            noise = json.loads((art / "random_noise_provenance.json").read_text())
            margin_step = int(json.loads((art / "min_margin_provenance.json").read_text())["step"])
            local = {
                "final": (art / "final.png", by_step[200]),
                "min_margin": (art / "min_margin.png", by_step[margin_step]),
                "random_noise": (art / "random_noise.png", noise),
            }
        for name, (path, ref) in local.items():
            result = replay_one(session, path, ref)
            result["sample_id"] = sid
            result["kind"] = name
            out_rows.append(result)
            print(json.dumps({"replay": sid, "kind": name, "ok": result["ok"]}), flush=True)
        del session
        torch.cuda.empty_cache()
    dest = (V5B / "tests" / args.tag / "replay.json") if args.dev50743 else (V5B / "replay" / f"shard_s{args.shard}.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out_rows, indent=2) + "\n")


def args_step(refs: dict) -> int:
    return max(refs)


if __name__ == "__main__":
    main()
