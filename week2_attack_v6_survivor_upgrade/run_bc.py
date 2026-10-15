"""Variant B/C trainer. Does not write v4, v5, or v5b."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
V6 = ROOT / "week2_attack_v6_survivor_upgrade"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "week1_mechanism"))
sys.path.insert(0, str(ROOT / "gate_experiment"))

from adapters.runtime import clear_hooks  # noqa: E402
from run_causal_v2 import generate, load_model, prepare  # noqa: E402
from week2_attack_v4_answer_aligned.common import image_mask_of, spans_for  # noqa: E402
from week2_attack_v4_answer_aligned.forward import (  # noqa: E402
    SCORE_LAYER,
    _run_layer,
    build_embeds,
    full_answer_logits,
    score_prompt,
)
from week2_attack_v4_answer_aligned.loss import answer_margin, compression_margin_loss  # noqa: E402
from week2_attack_v4_answer_aligned.run_smoke import (  # noqa: E402
    Session,
    evaluate_uint8,
    page_uint8,
    project,
)
from week2_attack_v5b_expanded_pristine40.load_samples import load_formal_items, manifest  # noqa: E402
from week2_attack_v6_survivor_upgrade.bc_math import (  # noqa: E402
    kappa_from_clean_margin,
    page_soft_mass,
    selection_l1,
    sequence_hinge,
    with_eos,
)
from week2_pilot.attack import last_query_scores, normalized_tiles  # noqa: E402

DEV = ["1025", "2008", "7006", "39204", "47450", "47567", "50779", "51133", "60700", "6194", "62941", "64506"]
EPS = 8 / 255


class BCSession(Session):
    def __init__(self, model, processor, item, answer_tokens):
        eos = int(model.generation_config.eos_token_id)
        tokens = with_eos(answer_tokens, eos)
        super().__init__(model, processor, item, tokens)
        prompt_ids = self.inputs["input_ids"]
        _, self.prompt_mask = image_mask_of(model, prompt_ids)
        with torch.no_grad():
            clean_logits = full_answer_logits(
                model, self.embeds_from(self.clean_page), self.prompt_len - 1, int(self.answer_tokens.numel())
            )
            self.kappa = kappa_from_clean_margin(answer_margin(clean_logits, self.answer_tokens))
            self.clean_mass = page_soft_mass(self.prompt_scores(self.clean_page), self.spans).detach()

    def prompt_embeds(self, page):
        tiles = normalized_tiles(page).to(dtype=self.pixels.dtype).unsqueeze(0)
        return build_embeds(
            self.model, self.inputs["input_ids"], self.prompt_mask, self.pixels, self.sizes, self.aux_idx, tiles
        )

    def prompt_scores(self, page):
        """Real FastV last-query scores, checkpointed through the score layer."""
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

    def losses(self, page):
        loss_c, margin, k, kept = self.compression(page)
        clear_hooks(self.model)
        logits = full_answer_logits(
            self.model, self.embeds_from(page), self.prompt_len - 1, int(self.answer_tokens.numel())
        )
        loss_seq = sequence_hinge(logits, self.answer_tokens, self.kappa)
        mass = page_soft_mass(self.prompt_scores(page), self.spans)
        loss_sel = selection_l1(mass, self.clean_mass)
        return loss_c, loss_seq, loss_sel, margin, k, kept, mass


def component_grads(session: BCSession, page: torch.Tensor):
    page = page.detach().requires_grad_(True)

    def one(which: str):
        page.grad = None
        if which == "comp":
            loss, margin, k, kept = session.compression(page)
            mass = session.clean_mass
        elif which == "seq":
            logits = full_answer_logits(
                session.model, session.embeds_from(page), session.prompt_len - 1, int(session.answer_tokens.numel())
            )
            loss = sequence_hinge(logits, session.answer_tokens, session.kappa)
            margin, k, kept, mass = 0.0, 0, [], session.clean_mass
        else:
            mass = page_soft_mass(session.prompt_scores(page), session.spans)
            loss = selection_l1(mass, session.clean_mass)
            margin, k, kept = 0.0, 0, []
        loss.backward()
        grad = page.grad.detach().clone()
        value = loss.detach()
        del loss
        torch.cuda.empty_cache()
        return value, grad, margin, k, kept, mass

    loss_c, g_comp, margin, k, kept, mass = one("comp")
    loss_seq, g_seq, _, _, _, _ = one("seq")
    loss_sel, g_sel, _, _, _, _ = one("sel")
    return {
        "loss_comp": float(loss_c),
        "loss_seq": float(loss_seq),
        "loss_sel": float(loss_sel),
        "margin": margin,
        "k": k,
        "kept": kept,
        "mass": [float(x) for x in mass.detach()],
        "g_comp": g_comp,
        "g_seq": g_seq,
        "g_sel": g_sel,
    }


def grad_l2(grad: torch.Tensor) -> float:
    return float(grad.float().norm())


def load_dev(ids: list[str]):
    specs = [row for row in manifest()["samples"] if row["sample_id"] in ids]
    return load_formal_items(specs)


def make_session(model, processor, item):
    clear_hooks(model)
    inputs = prepare(processor, item["images"], item["meta"]["question"])
    with torch.inference_mode():
        raw = generate(model, inputs)
    answer = torch.tensor(raw.detach().cpu().tolist(), dtype=torch.long)
    return BCSession(model, processor, item, answer)


def health(session: BCSession) -> dict:
    item = session.item
    measured = evaluate_uint8(session, item["images"][item["aux_idx"]], 0.0)
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
    aux_embed_grad = 0.0 if aux_leaf.grad is None else float(aux_leaf.grad.float().norm())
    finite = all(torch.isfinite(stats[name]).all() for name in ("g_comp", "g_seq", "g_sel"))
    report = {
        "sample_id": item["meta"]["sample_id"],
        "answer_len_with_eos": int(session.answer_tokens.numel()),
        "ends_with_eos": int(session.answer_tokens[-1]) == int(session.model.generation_config.eos_token_id),
        "k": stats["k"],
        "clean_full_match": measured["pred_full"] == item["baseline_full"]["pred"],
        "clean_fastv_match": measured["pred_fastv"] == item["baseline_fastv"]["pred"],
        "pred_full": measured["pred_full"],
        "pred_fastv": measured["pred_fastv"],
        "output_valid": measured["output_valid"],
        "keep_match_len": stats["k"] == measured["k"],
        "fastv_score_max_abs_diff": score_gap,
        "aux_grad_comp": grad_l2(stats["g_comp"]),
        "aux_grad_seq": grad_l2(stats["g_seq"]),
        "aux_grad_sel": grad_l2(stats["g_sel"]),
        "losses_finite": all(map(lambda x: x == x and abs(x) != float("inf"), (stats["loss_comp"], stats["loss_seq"], stats["loss_sel"]))),
        "grads_finite": bool(finite),
        "evidence_pixel_grad_l2": evidence_grad,
        "aux_embed_grad_l2": aux_embed_grad,
        "uint8_reload_linf": measured["linf"],
    }
    return report


def calibrate(session: BCSession) -> dict:
    base = component_grads(session, session.clean_page)
    shifted = None

    def pick(name, grad):
        value = grad_l2(grad)
        source = "delta0"
        if value < 1e-6:
            nonlocal shifted
            if shifted is None:
                shifted = component_grads(session, (session.clean_page + (1.0 / 255.0)).clamp(0, 1))
            value = grad_l2(shifted[name])
            source = "shift_1_over_255"
        return value, source

    comp, comp_source = pick("g_comp", base["g_comp"])
    seq, seq_source = pick("g_seq", base["g_seq"])
    sel, sel_source = pick("g_sel", base["g_sel"])
    if min(comp, seq, sel) < 1e-6:
        raise RuntimeError(f"calibration gradient still below 1e-6: comp={comp} seq={seq} sel={sel}")
    return {
        "sample_id": session.item["meta"]["sample_id"],
        "lambda_seq": comp / seq,
        "beta": comp / sel,
        "grad_l2": {"comp": comp, "seq": seq, "sel": sel},
        "source": {"comp": comp_source, "seq": seq_source, "sel": sel_source},
        "delta0_losses": {"comp": base["loss_comp"], "seq": base["loss_seq"], "sel": base["loss_sel"]},
        "answer_tokens": [int(x) for x in session.answer_tokens.detach().cpu()],
        "kappa": [float(x) for x in session.kappa.detach().cpu()],
    }


def attack(session: BCSession, steps: int, variant: str, scales: dict, artifact: Path):
    delta = torch.zeros_like(session.clean_page, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=2.0 / 255.0)
    lam = float(scales["lambda_seq"])
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
                loss_c, loss_seq, loss_sel, _, _, _, _ = session.losses(session.clean_page + delta.detach())
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
                "L_full_seq": float(loss_seq),
                "L_selection": float(loss_sel),
                "keep_hash": measured["keep_hash"],
                "k": measured["k"],
                "aux_share": measured["aux_share"],
                "evidence_share": measured["evidence_share"],
                "linf": measured["linf"],
                "png": str(dest),
            }
            rows.append(row)
            print(json.dumps({"id": row["sample_id"], "step": step, "margin": row["margin"], "full": row["pred_full"], "fastv": row["pred_fastv"]}), flush=True)
        if step == steps:
            break
        opt.zero_grad(set_to_none=True)
        page = session.clean_page + delta
        loss_c, _, _, _ = session.compression(page)
        (loss_c).backward()
        loss_seq = sequence_hinge(
            full_answer_logits(session.model, session.embeds_from(page), session.prompt_len - 1, int(session.answer_tokens.numel())),
            session.answer_tokens,
            session.kappa,
        )
        (lam * loss_seq).backward()
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
    (artifact / "curve.json").write_text(json.dumps(rows, indent=2) + "\n")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["health", "calibrate", "attack"], required=True)
    parser.add_argument("--variant", choices=["B", "C"], default="B")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--sample", default="1025")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshards", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(42)
    model, processor = load_model()
    for param in model.parameters():
        param.requires_grad_(False)
    if args.mode in ("health", "calibrate"):
        item = load_dev([args.sample])[0]
        session = make_session(model, processor, item)
        if args.mode == "health":
            report = health(session)
        else:
            report = calibrate(session)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
        return
    scales = json.loads((V6 / "FROZEN_BC_IMPLEMENTATION.json").read_text())["coefficients"]
    ids = [sid for index, sid in enumerate(DEV) if index % args.nshards == args.shard]
    if args.steps < 200:
        ids = [args.sample]
    items = {item["meta"]["sample_id"]: item for item in load_dev(ids)}
    args.out.mkdir(parents=True, exist_ok=True)
    done = set()
    csv_path = args.out / "samples.csv"
    if csv_path.exists():
        done = {row["sample_id"] for row in csv.DictReader(csv_path.open()) if row.get("status") == "ok"}
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
            writer.writerow({"sample_id": sid, "status": "ok", "seconds": f"{time.time() - t0:.1f}", "n_checkpoints": len(rows)})
        del session
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
