"""v6c runner. Does not write v5b, v6, or v6b."""

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
V6C = ROOT / "week2_attack_v6c_split_calibration"
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
from week2_attack_v4_answer_aligned.run_smoke import Session, evaluate_uint8, page_uint8, project  # noqa: E402
from week2_attack_v5b_expanded_pristine40.load_samples import load_formal_items, manifest  # noqa: E402
from week2_attack_v6c_split_calibration.losses import (  # noqa: E402
    LAMBDA_SOFT,
    PROBE_DELTA,
    page_soft_mass,
    selection_l1,
    soft_full_anchor,
    with_eos,
)
from week2_pilot.attack import last_query_scores, normalized_tiles  # noqa: E402

DEV = ["1025", "2008", "7006", "39204", "47450", "47567", "50779", "51133", "60700", "6194", "62941", "64506"]
EPS = 8 / 255


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


def grad_l2(grad: torch.Tensor) -> float:
    return float(grad.float().norm())


def component_grads(session: SoftSession, page: torch.Tensor, which_set=("comp", "soft", "sel")):
    page = page.detach().requires_grad_(True)
    out = {}

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
        grad = torch.zeros_like(page) if page.grad is None else page.grad.detach().clone()
        value = float(loss.detach())
        del loss
        torch.cuda.empty_cache()
        return value, grad, margin, k, kept

    for which in which_set:
        value, grad, margin, k, kept = one(which)
        out[which] = {"loss": value, "grad": grad, "margin": margin, "k": k, "kept": kept}
    return out


def load_ids(ids: list[str]):
    specs = [row for row in manifest()["samples"] if row["sample_id"] in ids]
    return load_formal_items(specs)


def make_session(model, processor, item):
    clear_hooks(model)
    inputs = prepare(processor, item["images"], item["meta"]["question"])
    with torch.inference_mode():
        raw = generate(model, inputs)
    answer = torch.tensor(raw.detach().cpu().tolist(), dtype=torch.long)
    return SoftSession(model, processor, item, answer)


def pixel_gap(clean: Image.Image, adv: Image.Image) -> float:
    a = np.asarray(clean.convert("RGB"), dtype=np.int16)
    b = np.asarray(adv.convert("RGB"), dtype=np.int16)
    return float(np.abs(a - b).max()) / 255.0


def health(session: SoftSession) -> dict:
    item = session.item
    clean = item["images"][item["aux_idx"]]
    measured = evaluate_uint8(session, clean, 0.0)
    reloaded = evaluate_uint8(session, page_uint8(session.clean_page, torch.zeros_like(session.clean_page)), 0.0)
    with torch.no_grad():
        reference = score_prompt(session.model.model.language_model, session.prompt_embeds(session.clean_page))
        ours = session.prompt_scores(session.clean_page)
    stats = component_grads(session, session.clean_page)
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
    comp = grad_l2(stats["comp"]["grad"])
    soft = grad_l2(stats["soft"]["grad"])
    sel = grad_l2(stats["sel"]["grad"])
    report = {
        "sample_id": item["meta"]["sample_id"],
        "ends_with_eos": int(session.answer_tokens[-1]) == int(session.model.generation_config.eos_token_id),
        "k": int(stats["comp"]["k"]),
        "k_generate": int(measured["k"]),
        "keep_match": stats["comp"]["k"] == measured["k"],
        "clean_full_match": measured["pred_full"] == item["baseline_full"]["pred"],
        "clean_fastv_match": measured["pred_fastv"] == item["baseline_fastv"]["pred"],
        "uint8_reload_match": reloaded["pred_full"] == measured["pred_full"] and reloaded["pred_fastv"] == measured["pred_fastv"],
        "uint8_reload_linf": reloaded["linf"],
        "fastv_score_max_abs_diff": float((reference - ours).abs().max()),
        "aux_grad_comp": comp,
        "aux_grad_soft": soft,
        "aux_grad_sel": sel,
        "clean_selection_grad_zero_expected": sel <= 1e-6,
        "losses_finite": all(math.isfinite(stats[name]["loss"]) for name in ("comp", "soft", "sel")),
        "evidence_pixel_grad_l2": evidence_grad,
        "lambda_soft": LAMBDA_SOFT,
    }
    report["b_health_pass"] = bool(
        report["ends_with_eos"]
        and report["keep_match"]
        and report["clean_full_match"]
        and report["clean_fastv_match"]
        and report["uint8_reload_match"]
        and report["uint8_reload_linf"] <= 1e-8
        and report["fastv_score_max_abs_diff"] <= 1e-5
        and report["losses_finite"]
        and math.isfinite(comp) and comp > 1e-6
        and math.isfinite(soft) and soft > 1e-6
        and math.isfinite(sel)
        and evidence_grad <= 1e-8
        and report["clean_selection_grad_zero_expected"]
    )
    return report


def probe_beta(session: SoftSession, dest: Path) -> dict:
    probe = (session.clean_page + PROBE_DELTA).clamp(0, 1)
    delta = probe - session.clean_page
    image = page_uint8(session.clean_page, delta)
    dest.parent.mkdir(parents=True, exist_ok=True)
    image.save(dest)
    clean_image = session.item["images"][session.aux_idx]
    gap = pixel_gap(clean_image, Image.open(dest).convert("RGB"))
    stats = component_grads(session, probe, ("comp", "sel"))
    comp = grad_l2(stats["comp"]["grad"])
    sel = grad_l2(stats["sel"]["grad"])
    ok = math.isfinite(comp) and comp > 1e-6 and math.isfinite(sel) and sel > 1e-6 and abs(gap - PROBE_DELTA) <= 1e-8
    beta = (comp / sel) if ok else None
    return {
        "sample_id": session.item["meta"]["sample_id"],
        "delta_ref": PROBE_DELTA,
        "probe_png": str(dest),
        "probe_sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
        "probe_uint8_linf": gap,
        "g_comp_ref": comp,
        "g_sel_ref": sel,
        "beta": beta,
        "ok": ok,
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "protected_pages_not_in_delta": True,
    }


def freeze_beta_once(beta: float):
    path = V6C / "FROZEN_COEFFICIENTS.json"
    payload = json.loads(path.read_text())
    if payload.get("beta") is not None:
        raise RuntimeError("beta is already frozen; a second probe is forbidden")
    if payload["lambda_soft"] != LAMBDA_SOFT:
        raise RuntimeError("lambda_soft was changed")
    payload["beta"] = beta
    payload["beta_status"] = "FROZEN_FROM_SINGLE_PROBE"
    payload["beta_frozen_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    path.write_text(json.dumps(payload, indent=2) + "\n")


def attack(session: SoftSession, steps: int, variant: str, beta: float, artifact: Path):
    delta = torch.zeros_like(session.clean_page, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=2.0 / 255.0)
    use_beta = float(beta) if variant == "C" else 0.0
    marks = {step for step in (0, 10, 25, 50, 75, 100, 150, 200) if step <= steps}
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
                loss_sel = selection_l1(page_soft_mass(session.prompt_scores(page), session.spans), session.clean_mass)
            row = {
                "sample_id": session.item["meta"]["sample_id"],
                "step": step,
                "pred_full": measured["pred_full"],
                "pred_fastv": measured["pred_fastv"],
                "anls_full": measured["anls_full"],
                "anls_fastv": measured["anls_fastv"],
                "output_valid": bool(measured["output_valid"]),
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
            print(json.dumps({"id": row["sample_id"], "variant_step": step, "margin": row["margin"], "full": row["pred_full"], "fastv": row["pred_fastv"]}), flush=True)
        if step == steps:
            break
        opt.zero_grad(set_to_none=True)
        page = session.clean_page + delta
        loss_c, _, _, _ = session.compression(page)
        loss_c.backward()
        (LAMBDA_SOFT * session.soft_loss(page)).backward()
        if use_beta:
            mass = page_soft_mass(session.prompt_scores(page), session.spans)
            (use_beta * selection_l1(mass, session.clean_mass)).backward()
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
    return rows


def load_model_frozen():
    torch.manual_seed(42)
    model, processor = load_model()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, processor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["health", "probe", "attack"], required=True)
    parser.add_argument("--variant", choices=["B", "C"], default="B")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--sample", default="1025")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshards", type=int, default=1)
    parser.add_argument("--ids", default="")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    model, processor = load_model_frozen()
    if args.mode == "health":
        session = make_session(model, processor, load_ids([args.sample])[0])
        report = health(session)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
        if not report["b_health_pass"]:
            raise SystemExit(3)
        return
    if args.mode == "probe":
        if args.sample != "1025":
            raise SystemExit("the only legal probe sample is 1025")
        session = make_session(model, processor, load_ids(["1025"])[0])
        report = health(session)
        probed = probe_beta(session, V6C / "probe_aux_1025.png")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"health": report, "probe": probed}, indent=2) + "\n")
        print(json.dumps({"b_health_pass": report["b_health_pass"], "probe_ok": probed["ok"], "beta": probed["beta"], "g_sel_ref": probed["g_sel_ref"]}), flush=True)
        if not report["b_health_pass"]:
            raise SystemExit(3)
        if not probed["ok"]:
            (V6C / "c_probe_decision.json").write_text(json.dumps({"decision": "C_BLOCKED_SELECTION_PROBE_CALIBRATION", "probe": probed}, indent=2) + "\n")
            raise SystemExit(4)
        freeze_beta_once(probed["beta"])
        return
    coeff = json.loads((V6C / "FROZEN_COEFFICIENTS.json").read_text())
    if coeff["lambda_soft"] != LAMBDA_SOFT:
        raise SystemExit("lambda_soft mismatch")
    beta = 0.0
    if args.variant == "C":
        if coeff.get("beta") is None:
            raise SystemExit("C has no frozen beta")
        beta = float(coeff["beta"])
    pool = args.ids.split(",") if args.ids else (DEV if args.steps >= 200 else [args.sample])
    if args.steps >= 200 and not args.ids:
        pool = DEV
    ids = [sid for index, sid in enumerate(pool) if index % args.nshards == args.shard]
    if args.steps < 200:
        ids = [args.sample]
    items = {item["meta"]["sample_id"]: item for item in load_ids(ids)}
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
        rows = attack(session, args.steps, args.variant, beta, args.out / sid)
        new = not csv_path.exists()
        with csv_path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if new:
                writer.writeheader()
            writer.writerow({"sample_id": sid, "status": "ok", "seconds": f"{time.time() - t0:.1f}", "n_checkpoints": len(rows)})
        print(json.dumps({"done": sid, "seconds": round(time.time() - t0, 1)}), flush=True)
        del session
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
