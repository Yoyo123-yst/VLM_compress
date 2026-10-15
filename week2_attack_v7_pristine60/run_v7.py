"""Frozen Variant B runner for pristine60. Does not write v5b, v6, v6b, or v6c."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
V7 = ROOT / "week2_attack_v7_pristine60"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "week1_mechanism"))
sys.path.insert(0, str(ROOT / "gate_experiment"))

from adapters.runtime import clear_hooks  # noqa: E402
from run_causal_v2 import budget, generate, load_model, prepare  # noqa: E402
from week2_attack_v4_answer_aligned.common import image_mask_of  # noqa: E402
from week2_attack_v4_answer_aligned.forward import (  # noqa: E402
    SCORE_LAYER,
    _run_layer,
    build_embeds,
    full_answer_logits,
    score_prompt,
)
from week2_attack_v4_answer_aligned.loss import answer_margin, official_keep_count  # noqa: E402
from week2_attack_v4_answer_aligned.run_smoke import Session, evaluate_uint8, page_uint8, project  # noqa: E402
from week2_attack_v5b_expanded_pristine40.load_samples import load_formal_items, manifest as v5b_manifest  # noqa: E402
from week2_attack_v6b_soft_full_anchor.soft_loss import page_soft_mass, soft_full_anchor, with_eos  # noqa: E402
from week2_attack_v6c_split_calibration.run_v6c import component_grads, grad_l2  # noqa: E402
from week2_attack_v7_pristine60.load_samples import load_items, manifest  # noqa: E402
from week2_pilot.attack import last_query_scores, normalized_tiles  # noqa: E402

LAMBDA_SOFT = 2.049824826423478
EPS = 8 / 255
CHECKPOINTS = (0, 10, 25, 50, 75, 100, 150, 200)
ATTACK_SEED_SALT = 2026100904
SMOKE_ID = "1025"


def sample_seed(salt: int, sample_id: str, extra: str = "") -> int:
    raw = hashlib.sha256(f"{salt}:{sample_id}:{extra}".encode()).hexdigest()
    return int(raw[:8], 16)


class SoftSession(Session):
    def __init__(self, model, processor, item, answer_tokens):
        tokens = with_eos(answer_tokens, int(model.generation_config.eos_token_id))
        super().__init__(model, processor, item, tokens)
        _, self.prompt_mask = image_mask_of(model, self.inputs["input_ids"])
        with torch.no_grad():
            clean_logits = full_answer_logits(
                model, self.embeds_from(self.clean_page), self.prompt_len - 1, int(self.answer_tokens.numel())
            )
            self.clean_margin = answer_margin(clean_logits, self.answer_tokens).detach()
            self.clean_mass = page_soft_mass(self.prompt_scores(self.clean_page), self.spans).detach()

    def prompt_embeds(self, page):
        tiles = normalized_tiles(page).to(dtype=self.pixels.dtype).unsqueeze(0)
        return build_embeds(
            self.model, self.inputs["input_ids"], self.prompt_mask, self.pixels, self.sizes, self.aux_idx, tiles
        )

    def prompt_scores(self, page):
        language_model = self.model.model.language_model
        prompt_embeds = self.prompt_embeds(page)
        position_ids = torch.arange(prompt_embeds.shape[1], device=prompt_embeds.device).unsqueeze(0)
        position_embeddings = language_model.rotary_emb(prompt_embeds, position_ids)
        hidden = prompt_embeds
        for index, decoder in enumerate(language_model.layers):
            if index == SCORE_LAYER:
                normed = decoder.input_layernorm(hidden)
                return last_query_scores(
                    decoder.self_attn, normed, {"position_embeddings": position_embeddings, "attention_mask": None}
                )

            def run(hidden_states, decoder=decoder, position_ids=position_ids, position_embeddings=position_embeddings):
                return _run_layer(decoder, hidden_states, position_ids, position_embeddings)

            hidden = torch.utils.checkpoint.checkpoint(run, hidden, use_reentrant=False)
        raise RuntimeError("FastV score layer was not reached")

    def soft_loss(self, page):
        logits = full_answer_logits(
            self.model, self.embeds_from(page), self.prompt_len - 1, int(self.answer_tokens.numel())
        )
        return soft_full_anchor(logits, self.answer_tokens, self.clean_margin)


def make_session(model, processor, item):
    clear_hooks(model)
    inputs = prepare(processor, item["images"], item["meta"]["question"])
    with torch.inference_mode():
        raw = generate(model, inputs)
    answer = torch.tensor(raw.detach().cpu().tolist(), dtype=torch.long)
    return SoftSession(model, processor, item, answer)


def load_model_frozen():
    torch.manual_seed(42)
    model, processor = load_model()
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()
    return model, processor


def historical_1025():
    specs = [row for row in v5b_manifest()["samples"] if row["sample_id"] == SMOKE_ID]
    if len(specs) != 1:
        raise RuntimeError("1025 is not in the historical v5b manifest")
    return load_formal_items(specs)[0]


def health(session: SoftSession) -> dict:
    item = session.item
    clean = item["images"][item["aux_idx"]]
    measured = evaluate_uint8(session, clean, 0.0)
    reloaded = evaluate_uint8(session, page_uint8(session.clean_page, torch.zeros_like(session.clean_page)), 0.0)
    with torch.no_grad():
        reference = score_prompt(session.model.model.language_model, session.prompt_embeds(session.clean_page))
        ours = session.prompt_scores(session.clean_page)
    stats = component_grads(session, session.clean_page, ("comp", "soft"))
    raw_evidence = item["evidence_idx"]
    evidence_ids = [raw_evidence] if isinstance(raw_evidence, int) else list(raw_evidence)
    evidence_ids = [i for i in evidence_ids if i != item["aux_idx"]]
    pixels = session.pixels.detach().clone().requires_grad_(True)
    aux_leaf = session.clean_page.detach().clone().requires_grad_(True)
    tiles = normalized_tiles(aux_leaf).to(dtype=session.pixels.dtype).unsqueeze(0)
    embeds = build_embeds(
        session.model, session.inputs["input_ids"], session.prompt_mask, pixels, session.sizes, session.aux_idx, tiles
    )
    embeds.sum().backward()
    evidence_grad = 0.0 if pixels.grad is None else float(pixels.grad[evidence_ids].float().norm())
    aux_embed_grad = 0.0 if aux_leaf.grad is None else float(aux_leaf.grad.float().norm())
    comp = grad_l2(stats["comp"]["grad"])
    soft = grad_l2(stats["soft"]["grad"])
    expected_k = official_keep_count(item["spec"]["M"] * 3699, 0.25)
    report = {
        "sample_id": item["meta"]["sample_id"],
        "ends_with_eos": int(session.answer_tokens[-1]) == int(session.model.generation_config.eos_token_id),
        "k": int(stats["comp"]["k"]),
        "k_official": expected_k,
        "k_generate": int(measured["k"]),
        "keep_match": list(stats["comp"]["kept"]) == list(measured["kept"]) and stats["comp"]["k"] == measured["k"],
        "k_official_match": int(stats["comp"]["k"]) == expected_k == int(measured["k"]),
        "clean_full_match": measured["pred_full"] == item["baseline_full"]["pred"],
        "clean_fastv_match": measured["pred_fastv"] == item["baseline_fastv"]["pred"],
        "uint8_reload_match": reloaded["pred_full"] == measured["pred_full"] and reloaded["pred_fastv"] == measured["pred_fastv"],
        "uint8_reload_linf": reloaded["linf"],
        "fastv_score_max_abs_diff": float((reference - ours).abs().max()),
        "aux_grad_comp": comp,
        "aux_grad_soft": soft,
        "aux_embed_grad_l2": aux_embed_grad,
        "losses_finite": all(math.isfinite(stats[name]["loss"]) for name in ("comp", "soft")),
        "evidence_pixel_grad_l2": evidence_grad,
        "lambda_soft": LAMBDA_SOFT,
        "beta_used": False,
    }
    report["health_pass"] = bool(
        report["ends_with_eos"]
        and report["keep_match"]
        and report["k_official_match"]
        and report["clean_full_match"]
        and report["clean_fastv_match"]
        and report["uint8_reload_match"]
        and report["uint8_reload_linf"] <= 1e-8
        and report["fastv_score_max_abs_diff"] <= 1e-5
        and report["losses_finite"]
        and math.isfinite(comp) and comp > 1e-6
        and math.isfinite(soft) and soft > 1e-6
        and math.isfinite(aux_embed_grad) and aux_embed_grad > 1e-6
        and evidence_grad <= 1e-8
    )
    return report


def attack(session: SoftSession, steps: int, artifact: Path, seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2**32))
    delta = torch.zeros_like(session.clean_page, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=2.0 / 255.0)
    marks = {step for step in CHECKPOINTS if step <= steps}
    marks.add(steps)
    artifact.mkdir(parents=True, exist_ok=True)
    session.images[session.aux_idx].save(artifact / "clean.png")
    rows = []
    for step in range(steps + 1):
        if step in marks:
            image = page_uint8(session.clean_page, delta)
            measured = evaluate_uint8(session, image, 0.0)
            dest = artifact / "checkpoints" / f"step_{step:03d}.png"
            dest.parent.mkdir(parents=True, exist_ok=True)
            image.save(dest)
            page = session.clean_page + delta.detach()
            with torch.no_grad():
                loss_c, _, _, _ = session.compression(page)
                loss_soft = session.soft_loss(page)
            page_budget = budget(measured["kept"], session.spans, session.item["evidence_idx"], session.aux_idx, [])
            counts = page_budget["kept_per_image"]
            total = sum(counts) or 1
            row = {
                "sample_id": session.item["meta"]["sample_id"],
                "step": step,
                "seed": seed,
                "pred_full": measured["pred_full"],
                "pred_fastv": measured["pred_fastv"],
                "anls_full": measured["anls_full"],
                "anls_fastv": measured["anls_fastv"],
                "output_valid": bool(measured["output_valid"]),
                "margin": measured["margin"],
                "L_comp": float(loss_c),
                "L_soft_full": float(loss_soft),
                "keep_hash": measured["keep_hash"],
                "kept": measured["kept"],
                "k": measured["k"],
                "aux_share": measured["aux_share"],
                "evidence_share": measured["evidence_share"],
                "page_keep_counts": counts,
                "page_shares": [count / total for count in counts],
                "linf": measured["linf"],
                "png": f"checkpoints/step_{step:03d}.png",
                "png_sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
                "protected_ok": True,
            }
            rows.append(row)
            print(json.dumps({
                "id": row["sample_id"], "step": step, "margin": row["margin"],
                "full": row["pred_full"], "fastv": row["pred_fastv"],
            }), flush=True)
        if step == steps:
            break
        opt.zero_grad(set_to_none=True)
        page = session.clean_page + delta
        loss_c, _, _, _ = session.compression(page)
        loss_c.backward()
        (LAMBDA_SOFT * session.soft_loss(page)).backward()
        if delta.grad is None or not torch.isfinite(delta.grad).all():
            raise RuntimeError(f"non-finite gradient at step {step}")
        if rows and rows[-1]["step"] == step:
            rows[-1]["grad_norm"] = float(delta.grad.float().norm())
        opt.step()
        project(delta, session.clean_page, EPS)
        torch.cuda.empty_cache()
    public = []
    for row in rows:
        item = dict(row)
        item.pop("kept", None)
        public.append(item)
    (artifact / "curve.json").write_text(json.dumps(public, indent=2) + "\n")
    (artifact / "checkpoints.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (artifact / "provenance.json").write_text(json.dumps({
        "sample_id": session.item["meta"]["sample_id"],
        "variant": "B",
        "lambda_soft": LAMBDA_SOFT,
        "beta": None,
        "seed": seed,
        "steps": steps,
        "checkpoints": sorted(marks),
    }, indent=2) + "\n")
    return rows


def assert_not_pristine(sample_id: str):
    frozen = V7 / "frozen_manifest.json"
    if not frozen.exists():
        return
    ids = {row["sample_id"] for row in json.loads(frozen.read_text())["samples"]}
    if sample_id in ids:
        raise SystemExit(f"{sample_id} is inside pristine60 and cannot be used for smoke or health")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["health", "attack"], required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--sample", default=SMOKE_ID)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshards", type=int, default=1)
    parser.add_argument("--ids", default="")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeat", action="store_true")
    args = parser.parse_args()
    if (V7 / "FROZEN.json").exists():
        frozen = json.loads((V7 / "FROZEN.json").read_text())
        if frozen.get("lambda_soft") != LAMBDA_SOFT:
            raise SystemExit("FROZEN lambda_soft mismatch")
    model, processor = load_model_frozen()
    if args.mode == "health" or args.steps < 200:
        assert_not_pristine(args.sample)
        if args.sample != SMOKE_ID:
            raise SystemExit("smoke and health are only legal on historical sample 1025")
        session = make_session(model, processor, historical_1025())
        if args.mode == "health":
            report = health(session)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({"health_pass": report["health_pass"], "comp": report["aux_grad_comp"], "soft": report["aux_grad_soft"]}), flush=True)
            if not report["health_pass"]:
                raise SystemExit(3)
            return
        seed = sample_seed(ATTACK_SEED_SALT, SMOKE_ID, f"smoke{args.steps}")
        attack(session, args.steps, args.out / SMOKE_ID, seed)
        return
    specs = manifest()["samples"]
    if args.ids:
        wanted = set(args.ids.split(","))
        specs = [row for row in specs if row["sample_id"] in wanted]
    specs = [row for index, row in enumerate(specs) if index % args.nshards == args.shard]
    items = {item["meta"]["sample_id"]: item for item in load_items(specs)}
    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "samples.csv"
    done = set()
    if csv_path.exists():
        with csv_path.open() as handle:
            done = {row["sample_id"] for row in csv.DictReader(handle) if row.get("status") == "ok"}
    fields = ["sample_id", "status", "seconds", "n_checkpoints", "seed"]
    for spec in specs:
        sid = spec["sample_id"]
        if sid in done:
            print(json.dumps({"skip": sid}), flush=True)
            continue
        seed = sample_seed(ATTACK_SEED_SALT, sid, "main")
        session = make_session(model, processor, items[sid])
        t0 = time.time()
        rows = attack(session, 200, args.out / sid, seed)
        new = not csv_path.exists()
        with csv_path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if new:
                writer.writeheader()
            writer.writerow({
                "sample_id": sid, "status": "ok", "seconds": f"{time.time() - t0:.1f}",
                "n_checkpoints": len(rows), "seed": seed,
            })
        print(json.dumps({"done": sid, "seconds": round(time.time() - t0, 1)}), flush=True)
        del session
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
