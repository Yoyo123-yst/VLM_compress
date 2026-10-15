"""Frozen λ=0.25 attack for the expanded pristine40 manifest.

The optimization math is the v4 Session loop. Primary endpoint is step 200.
Min-margin selection reads only output_valid, compressed margin, and step.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import torch
from PIL import Image

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
V5B = ROOT / "week2_attack_v5b_expanded_pristine40"
sys.path.insert(0, str(ROOT))

import week2_attack_v4_answer_aligned.run_smoke as smoke  # noqa: E402
from adapters.runtime import clear_hooks  # noqa: E402
from gate_experiment.run_causal_v2 import generate, load_model, prepare  # noqa: E402
from week1_mechanism.metrics_task import normalize  # noqa: E402
from week2_attack_v4_answer_aligned.common import GROUPS, sha_image, write_json  # noqa: E402
from week2_attack_v4_answer_aligned.run_smoke import (  # noqa: E402
    Session,
    checkpoint_steps,
    evaluate_uint8,
    page_uint8,
    project,
)
from week2_attack_v5b_expanded_pristine40.load_samples import load_formal_items, manifest  # noqa: E402

CHECKPOINTS = [0, 10, 25, 50, 75, 100, 150, 200]


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pick_min_margin(rows: list[dict]) -> dict:
    """Prospective selector. Only valid flag, margin, and step are read."""
    pool = [row for row in rows if row["output_valid"]]
    if not pool:
        raise RuntimeError("no valid checkpoint for min-margin")
    return sorted(pool, key=lambda row: (float(row["margin"]), int(row["step"])))[0]


def attack_sample(session: Session, steps: int, scales: dict, artifact: Path) -> tuple[dict, list[dict]]:
    delta = torch.zeros_like(session.clean_page, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=2.0 / 255.0)
    scale_c = float(scales["scale_comp_l2"])
    scale_f = float(scales["scale_full_l2"])
    marks = set(checkpoint_steps(steps))
    rows = []
    images = {}
    margin_start = None
    artifact.mkdir(parents=True, exist_ok=True)
    session.images[session.aux_idx].save(artifact / "clean.png")
    for step in range(steps + 1):
        if step in marks or step == steps:
            image = page_uint8(session.clean_page, delta)
            measured = evaluate_uint8(session, image, 0.0 if margin_start is None else margin_start)
            if margin_start is None:
                margin_start = measured["margin"]
            name = f"step_{step:03d}.png"
            dest = artifact / "checkpoints" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            image.save(dest)
            row = {
                "sample_id": session.item["meta"]["sample_id"],
                "step": step,
                "pred_full": measured["pred_full"],
                "pred_fastv": measured["pred_fastv"],
                "norm_full": normalize(measured["pred_full"]),
                "norm_fastv": normalize(measured["pred_fastv"]),
                "anls_full": measured["anls_full"],
                "anls_fastv": measured["anls_fastv"],
                "output_valid": measured["output_valid"],
                "margin": measured["margin"],
                "full_kl": measured["full_kl"],
                "keep_hash": measured["keep_hash"],
                "k": measured["k"],
                "aux_share": measured["aux_share"],
                "evidence_share": measured["evidence_share"],
                "linf": measured["linf"],
                "png_sha256": file_sha(dest),
                "page_sha": measured["page_sha"],
            }
            rows.append(row)
            images[step] = image
            print(json.dumps({
                "id": row["sample_id"], "step": step,
                "full": measured["anls_full"], "fastv": measured["anls_fastv"],
                "margin": measured["margin"],
            }), flush=True)
        if step == steps:
            break
        opt.zero_grad(set_to_none=True)
        loss_c, _, _, _ = session.compression(session.clean_page + delta)
        (loss_c / scale_c).backward()
        loss_f = session.full_kl(session.clean_page + delta)
        (smoke.LAMBDA_FULL * loss_f / scale_f).backward()
        if delta.grad is None or not torch.isfinite(delta.grad).all():
            raise RuntimeError(f"non-finite gradient at step {step}")
        opt.step()
        project(delta, session.clean_page, smoke.EPS)
        torch.cuda.empty_cache()
    primary = next(row for row in rows if int(row["step"]) == steps)
    secondary = pick_min_margin(rows)
    final_path = artifact / "final.png"
    shutil.copyfile(artifact / "checkpoints" / f"step_{steps:03d}.png", final_path)
    images[int(secondary["step"])].save(artifact / "min_margin.png")
    if file_sha(final_path) != primary["png_sha256"]:
        raise RuntimeError("final.png is not byte-identical to step 200")
    for row in rows:
        row["selected_by_primary"] = int(row["step"]) == steps
        row["selected_by_min_margin"] = int(row["step"]) == int(secondary["step"])
    write_json(artifact / "final_provenance.json", {
        "sample_id": primary["sample_id"],
        "selector": "fixed_step_200",
        "step": steps,
        "png_sha256": file_sha(final_path),
        "uint8_linf": primary["linf"],
        "margin": primary["margin"],
    })
    write_json(artifact / "min_margin_provenance.json", {
        "sample_id": secondary["sample_id"],
        "selector": "prospective_min_compressed_margin",
        "step": int(secondary["step"]),
        "margin": secondary["margin"],
        "output_valid": True,
        "png_sha256": file_sha(artifact / "min_margin.png"),
        "reads": ["output_valid", "margin", "step"],
    })
    return primary, rows


def done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open() as handle:
        return {row["sample_id"] for row in csv.DictReader(handle) if row.get("status") == "ok"}


def append_row(path: Path, row: dict, fields: list[str]):
    new = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if new:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fields})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshards", type=int, default=4)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--tag", default="formal")
    parser.add_argument("--dev50743", action="store_true")
    args = parser.parse_args()
    if args.tag == "formal" and args.steps != 200:
        raise SystemExit("formal attack steps are frozen at 200")
    if args.tag == "formal":
        lock = json.loads((V5B / "EXECUTION_LOCK.json").read_text())
        if lock["code_hashes"]["week2_attack_v5b_expanded_pristine40/run_formal_attack.py"] != file_sha(Path(__file__)):
            raise SystemExit("formal runner hash does not match execution lock")
        if lock["frozen_json_file_sha256"] != file_sha(V5B / "FROZEN.json"):
            raise SystemExit("FROZEN.json hash does not match execution lock")
    smoke.LAMBDA_FULL = 0.25
    torch.manual_seed(42)
    scales = json.loads((ROOT / "week2_attack_v4_answer_aligned" / "results" / "lambda_scale.json").read_text())
    if args.dev50743:
        from week2_attack_v4_answer_aligned.common import load_frozen_samples
        items = [item for item in load_frozen_samples() if item["meta"]["sample_id"] == "50743"]
        out = V5B / "tests" / args.tag
        image_root = out
    else:
        payload = manifest()
        specs = [row for index, row in enumerate(payload["samples"]) if index % args.nshards == args.shard]
        items = load_formal_items(specs)
        out = V5B / "attacks" / f"shard_s{args.shard}"
        image_root = V5B / "attacks" / "artifacts"
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "samples.csv"
    jsonl_path = out / "checkpoints.jsonl"
    finished = done_ids(csv_path)
    model, processor = load_model()
    for param in model.parameters():
        param.requires_grad_(False)
    fields = ["sample_id", "status", "primary_step", "min_margin_step", "seconds", "png_sha256"]
    for item in items:
        sid = item["meta"]["sample_id"]
        if sid in finished:
            continue
        GROUPS.setdefault(sid, "v5b")
        clear_hooks(model)
        inputs = prepare(processor, item["images"], item["meta"]["question"])
        with torch.inference_mode():
            raw = generate(model, inputs)
        answer = torch.tensor(raw.detach().cpu().tolist(), dtype=torch.long)
        session = Session(model, processor, item, answer)
        t0 = time.time()
        primary, rows = attack_sample(session, args.steps, scales, image_root / sid)
        with jsonl_path.open("a") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        append_row(csv_path, {
            "sample_id": sid,
            "status": "ok",
            "primary_step": primary["step"],
            "min_margin_step": next(row["step"] for row in rows if row["selected_by_min_margin"]),
            "seconds": f"{time.time() - t0:.1f}",
            "png_sha256": primary["png_sha256"],
        }, fields)
        print(json.dumps({"done": sid, "seconds": time.time() - t0}), flush=True)
        del session
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
