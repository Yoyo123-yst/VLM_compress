"""Independent v6b trainer. Does not write v4, v5, v5b, or v6."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
V6B = ROOT / "week2_attack_v6b_soft_full_anchor"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "week1_mechanism"))
sys.path.insert(0, str(ROOT / "gate_experiment"))

from adapters.runtime import clear_hooks  # noqa: E402
from run_causal_v2 import generate, load_model, prepare  # noqa: E402
from week2_attack_v4_answer_aligned.common import image_mask_of  # noqa: E402
from week2_attack_v4_answer_aligned.forward import (  # noqa: E402
    SCORE_LAYER,
    _run_layer,
    build_embeds,
    full_answer_logits,
    score_prompt,
)
from week2_attack_v4_answer_aligned.loss import answer_margin  # noqa: E402
from week2_attack_v4_answer_aligned.run_smoke import (  # noqa: E402
    Session,
    evaluate_uint8,
    page_uint8,
    project,
)
from week2_attack_v5b_expanded_pristine40.load_samples import load_formal_items, manifest  # noqa: E402
from week2_attack_v6b_soft_full_anchor.soft_loss import (  # noqa: E402
    page_soft_mass,
    selection_l1,
    soft_full_anchor,
    with_eos,
)
from week2_pilot.attack import last_query_scores, normalized_tiles  # noqa: E402

DEV = ["1025", "2008", "7006", "39204", "47450", "47567", "50779", "51133", "60700", "6194", "62941", "64506"]
EPS = 8 / 255
FLOOR = 1e-6


class SoftSession(Session):
    def __init__(self, model, processor, item, answer_tokens):
        eos = int(model.generation_config.eos_token_id)
        tokens = with_eos(answer_tokens, eos)
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


def grad_l2(grad: torch.Tensor) -> float:
    return float(grad.float().norm())


def component_grads(session: SoftSession, page: torch.Tensor):
    page = page.detach().requires_grad_(True)

    def one(which: str):
        page.grad = None
        if which == "comp":
            loss, margin, k, kept = session.compression(page)
        elif which == "soft":
            loss = session.soft_loss(page)
            margin, k, kept = 0.0, 0, []
        else:
            mass = page_soft_mass(session.prompt_scores(page), session.spans)
            loss = selection_l1(mass, session.clean_mass)
            margin, k, kept = 0.0, 0, []
        loss.backward()
        if page.grad is None:
            grad = torch.zeros_like(page)
        else:
            grad = page.grad.detach().clone()
        value = float(loss.detach())
        del loss
        torch.cuda.empty_cache()
        return value, grad, margin, k, kept

    loss_c, g_comp, margin, k, kept = one("comp")
    loss_s, g_soft, _, _, _ = one("soft")
    loss_sel, g_sel, _, _, _ = one("sel")
    return {
        "loss_comp": loss_c,
        "loss_soft": loss_s,
        "loss_sel": loss_sel,
        "margin": margin,
        "k": k,
        "kept": kept,
        "g_comp": g_comp,
        "g_soft": g_soft,
        "g_sel": g_sel,
    }


def load_dev(ids: list[str]):
    specs = [row for row in manifest()["samples"] if row["sample_id"] in ids]
    return load_formal_items(specs)


def make_session(model, processor, item):
    clear_hooks(model)
    inputs = prepare(processor, item["images"], item["meta"]["question"])
    with torch.inference_mode():
        raw = generate(model, inputs)
    answer = torch.tensor(raw.detach().cpu().tolist(), dtype=torch.long)
    return SoftSession(model, processor, item, answer)


def usable(value: float) -> bool:
    return math.isfinite(value) and value > FLOOR


def health(session: SoftSession) -> dict:
    item = session.item
    clean_image = item["images"][item["aux_idx"]]
    measured = evaluate_uint8(session, clean_image, 0.0)
    reloaded = evaluate_uint8(session, page_uint8(session.clean_page, torch.zeros_like(session.clean_page)), 0.0)
    with torch.no_grad():
        reference = score_prompt(session.model.model.language_model, session.prompt_embeds(session.clean_page))
        ours = session.prompt_scores(session.clean_page)
    score_gap = float((reference - ours).abs().max())
    stats = component_grads(session, session.clean_page)
    raw_evidence = item["evidence_idx"]
    evidence_ids = [raw_evidence] if isinstance(raw_evidence, int) else list(raw_evidence)
    evidence_ids = [i for i in evidence_ids if i != item["aux_idx"]]
    torch.cuda.empty_cache()
    pixels = session.pixels.detach().clone().requires_grad_(True)
    aux_leaf = session.clean_page.detach().clone().requires_grad_(True)
    tiles = normalized_tiles(aux_leaf).to(dtype=session.pixels.dtype).unsqueeze(0)
    embeds = build_embeds(
        session.model, session.inputs["input_ids"], session.prompt_mask, pixels, session.sizes, session.aux_idx, tiles
    )
    embeds.sum().backward()
    evidence_grad = 0.0 if pixels.grad is None else float(pixels.grad[evidence_ids].float().norm())
    comp_l2 = grad_l2(stats["g_comp"])
    soft_l2 = grad_l2(stats["g_soft"])
    sel_l2 = grad_l2(stats["g_sel"])
    report = {
        "sample_id": item["meta"]["sample_id"],
        "answer_len_with_eos": int(session.answer_tokens.numel()),
        "ends_with_eos": int(session.answer_tokens[-1]) == int(session.model.generation_config.eos_token_id),
        "k": int(stats["k"]),
        "k_generate": int(measured["k"]),
        "keep_match": stats["k"] == measured["k"] and measured["keep_hash"] == reloaded["keep_hash"],
        "clean_full_match": measured["pred_full"] == item["baseline_full"]["pred"],
        "clean_fastv_match": measured["pred_fastv"] == item["baseline_fastv"]["pred"],
        "uint8_reload_full_match": reloaded["pred_full"] == measured["pred_full"],
        "uint8_reload_fastv_match": reloaded["pred_fastv"] == measured["pred_fastv"],
        "uint8_reload_linf": reloaded["linf"],
        "output_valid": bool(measured["output_valid"]),
        "fastv_score_max_abs_diff": score_gap,
        "aux_grad_comp": comp_l2,
        "aux_grad_soft": soft_l2,
        "aux_grad_sel": sel_l2,
        "losses_finite": all(math.isfinite(stats[name]) for name in ("loss_comp", "loss_soft", "loss_sel")),
        "grads_finite": all(torch.isfinite(stats[name]).all() for name in ("g_comp", "g_soft", "g_sel")),
        "soft_anchor_nonzero": usable(soft_l2),
        "evidence_pixel_grad_l2": evidence_grad,
        "clean_margin": [float(x) for x in session.clean_margin.detach().cpu()],
    }
    report["health_pass"] = bool(
        report["ends_with_eos"]
        and report["keep_match"]
        and report["clean_full_match"]
        and report["clean_fastv_match"]
        and report["uint8_reload_full_match"]
        and report["uint8_reload_fastv_match"]
        and report["uint8_reload_linf"] <= 1e-8
        and report["output_valid"]
        and report["fastv_score_max_abs_diff"] <= 1e-5
        and report["losses_finite"]
        and report["grads_finite"]
        and report["soft_anchor_nonzero"]
        and math.isfinite(comp_l2)
        and math.isfinite(sel_l2)
        and evidence_grad <= 1e-8
    )
    return report


def calibrate(session: SoftSession) -> dict:
    stats = component_grads(session, session.clean_page)
    comp = grad_l2(stats["g_comp"])
    soft = grad_l2(stats["g_soft"])
    sel = grad_l2(stats["g_sel"])
    ok = usable(comp) and usable(soft) and usable(sel)
    return {
        "sample_id": session.item["meta"]["sample_id"],
        "state": "delta=0",
        "probe": False,
        "ok": ok,
        "grad_l2": {"comp": comp, "soft": soft, "sel": sel},
        "lambda_soft": (comp / soft) if ok else None,
        "beta": (comp / sel) if ok else None,
        "losses": {"comp": stats["loss_comp"], "soft": stats["loss_soft"], "sel": stats["loss_sel"]},
    }


def attack(session: SoftSession, steps: int, variant: str, scales: dict, artifact: Path):
    delta = torch.zeros_like(session.clean_page, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=2.0 / 255.0)
    lam = float(scales["lambda_soft"])
    beta = float(scales["beta"]) if variant == "C" else 0.0
    marks = {step for step in (0, 10, 25, 50, 75, 100, 150, 200) if step <= steps}
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
            with torch.no_grad():
                loss_c, _, _, _ = session.compression(session.clean_page + delta.detach())
                loss_soft = session.soft_loss(session.clean_page + delta.detach())
                loss_sel = selection_l1(
                    page_soft_mass(session.prompt_scores(session.clean_page + delta.detach()), session.spans),
                    session.clean_mass,
                )
            row = {
                "sample_id": session.item["meta"]["sample_id"],
                "step": step,
                "pred_full": measured["pred_full"],
                "pred_fastv": measured["pred_fastv"],
                "anls_full": measured["anls_full"],
                "anls_fastv": measured["anls_fastv"],
                "output_valid": measured["output_valid"],
                "margin": measured["margin"],
                "L_comp": float(loss_c),
                "L_soft_full": float(loss_soft),
                "L_selection": float(loss_sel),
                "keep_hash": measured["keep_hash"],
                "kept": measured["kept"],
                "k": measured["k"],
                "aux_share": measured["aux_share"],
                "evidence_share": measured["evidence_share"],
                "linf": measured["linf"],
                "png": f"checkpoints/step_{step:03d}.png",
            }
            rows.append(row)
            print(json.dumps({
                "id": row["sample_id"], "step": step, "margin": row["margin"],
                "full": row["pred_full"], "fastv": row["pred_fastv"], "Lsoft": row["L_soft_full"],
            }), flush=True)
        if step == steps:
            break
        opt.zero_grad(set_to_none=True)
        page = session.clean_page + delta
        loss_c, _, _, _ = session.compression(page)
        loss_c.backward()
        (lam * session.soft_loss(page)).backward()
        if beta:
            mass = page_soft_mass(session.prompt_scores(page), session.spans)
            (beta * selection_l1(mass, session.clean_mass)).backward()
        if delta.grad is None or not torch.isfinite(delta.grad).all():
            raise RuntimeError(f"non-finite gradient at step {step}")
        grad_norm = float(delta.grad.float().norm())
        opt.step()
        project(delta, session.clean_page, EPS)
        if rows and rows[-1]["step"] == step:
            rows[-1]["grad_norm"] = grad_norm
        torch.cuda.empty_cache()
    public = []
    for row in rows:
        item = dict(row)
        item.pop("kept", None)
        public.append(item)
    (artifact / "curve.json").write_text(json.dumps(public, indent=2) + "\n")
    (artifact / "checkpoints.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["health", "calibrate", "attack"], required=True)
    parser.add_argument("--variant", choices=["B", "C"], default="B")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--sample", default="1025")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshards", type=int, default=1)
    parser.add_argument("--ids", default="")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(42)
    model, processor = load_model()
    for param in model.parameters():
        param.requires_grad_(False)
    if args.mode in ("health", "calibrate"):
        item = load_dev([args.sample])[0]
        session = make_session(model, processor, item)
        report = health(session) if args.mode == "health" else calibrate(session)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: report[k] for k in report if k != "clean_margin"}), flush=True)
        if args.mode == "calibrate" and not report["ok"]:
            raise SystemExit(2)
        if args.mode == "health" and not report["health_pass"]:
            raise SystemExit(3)
        return
    scales = json.loads((V6B / "FROZEN_COEFFICIENTS.json").read_text())
    if args.ids:
        pool = args.ids.split(",")
    else:
        pool = DEV
    ids = [sid for index, sid in enumerate(pool) if index % args.nshards == args.shard]
    if args.steps < 200:
        ids = [args.sample]
    items = {item["meta"]["sample_id"]: item for item in load_dev(ids)}
    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "samples.csv"
    done = set()
    if csv_path.exists():
        with csv_path.open() as handle:
            done = {row["sample_id"] for row in csv.DictReader(handle) if row.get("status") == "ok"}
    fields = ["sample_id", "status", "seconds", "n_checkpoints"]
    for sid in ids:
        if sid in done:
            print(json.dumps({"skip": sid}), flush=True)
            continue
        session = make_session(model, processor, items[sid])
        t0 = time.time()
        rows = attack(session, args.steps, args.variant, scales, args.out / sid)
        new = not csv_path.exists()
        with csv_path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if new:
                writer.writeheader()
            writer.writerow({
                "sample_id": sid,
                "status": "ok",
                "seconds": f"{time.time() - t0:.1f}",
                "n_checkpoints": len(rows),
            })
        print(json.dumps({"done": sid, "seconds": round(time.time() - t0, 1)}), flush=True)
        del session
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
