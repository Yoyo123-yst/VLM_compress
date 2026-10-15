"""Hard-selection answer-margin attack on the frozen 6 questions.

The keep set is recomputed every step and detached. Success is decided only
from real Full and FastV generate calls.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import torch
from PIL import Image

from week2_attack_v4_answer_aligned.common import (
    CALIBRATION_ID,
    CHECKPOINTS,
    EPS,
    GROUPS,
    RATIO,
    V4,
    append_jsonl,
    assert_protected,
    image_mask_of,
    jaccard,
    keep_hash,
    load_frozen_samples,
    sha_image,
    spans_for,
    write_json,
)
from week2_attack_v4_answer_aligned.forward import compressed_answer_logits, full_answer_logits
from week2_attack_v4_answer_aligned.forward import build_embeds, hook_kept, keep_index_from_kept
from week2_attack_v4_answer_aligned.loss import (
    LAMBDA_FULL,
    answer_margin,
    classify_sample,
    compression_margin_loss,
    kl_clean_adv,
    official_keep_count,
    output_valid,
    quantize_in_ball,
    select_best,
)
from week2_pilot.attack import float_to_page, normalized_tiles, page_to_float

import sys

sys.path.insert(0, str(V4.parents[1] / "week1_mechanism"))
sys.path.insert(0, str(V4.parents[1] / "gate_experiment"))

from metrics_task import anls_score  # noqa: E402
from run_causal_v2 import append_csv, budget, decode, generate, load_model, prepare, run_fastv, run_plain  # noqa: E402
from adapters.runtime import clear_hooks  # noqa: E402

SCALE_PATH = V4 / "results" / "lambda_scale.json"
FIELDS = [
    "sample_id", "group", "best_step", "label", "replay_ok", "anls_full", "anls_fastv",
    "pred_full", "pred_fastv", "margin", "margin_start", "full_kl", "aux_share",
    "evidence_share", "evidence_keep", "linf_uint8", "keep_hash", "eps_ok",
]


def project(delta, clean, eps):
    with torch.no_grad():
        delta.clamp_(-eps, eps)
        delta.copy_((clean + delta).clamp(0, 1) - clean)


def grad_stats(left: torch.Tensor, right: torch.Tensor) -> dict:
    a = left.detach().float().reshape(-1)
    b = right.detach().float().reshape(-1)
    denom = float(a.norm() * b.norm())
    cosine = float(torch.dot(a, b) / denom) if denom > 0 else None
    combined = a + b

    def pack(vec):
        return {
            "l1": float(vec.abs().sum()),
            "l2": float(vec.norm()),
            "linf": float(vec.abs().max()) if vec.numel() else 0.0,
        }

    return {"comp": pack(a), "full": pack(b), "cosine": cosine, "combined": pack(combined)}


def page_uint8(clean_page: torch.Tensor, delta: torch.Tensor) -> Image.Image:
    clean_u8 = clean_page.detach().mul(255).round()
    quantized = quantize_in_ball(clean_u8, (clean_page.detach() + delta.detach()).clamp(0, 1), 8)
    array = quantized.permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array, mode="RGB")


def uint8_linf(original: Image.Image, attacked: Image.Image) -> float:
    gap = (page_to_float(attacked) - page_to_float(original)).abs().max()
    return float(gap)


class Session:
    def __init__(self, model, processor, item, answer_tokens):
        self.model = model
        self.processor = processor
        self.item = item
        self.images = item["images"]
        self.aux_idx = item["aux_idx"]
        self.question = item["meta"]["question"]
        self.answer_tokens = answer_tokens.to("cuda")
        clear_hooks(model)
        self.inputs = prepare(processor, self.images, self.question)
        self.prompt_len = int(self.inputs["input_ids"].shape[1])
        self.spans = spans_for(model, self.inputs, len(self.images))
        full_ids = torch.cat([self.inputs["input_ids"], self.answer_tokens.view(1, -1)], dim=1)
        _, self.mask = image_mask_of(model, full_ids)
        self.full_ids = full_ids
        self.clean_page = page_to_float(self.images[self.aux_idx]).to("cuda")
        self.pixels = self.inputs["pixel_values"]
        self.sizes = self.inputs["image_sizes"]
        with torch.no_grad():
            clean_embeds = self.embeds_from(self.clean_page)
            logits = full_answer_logits(model, clean_embeds, self.prompt_len - 1, int(self.answer_tokens.numel()))
            self.log_p_clean = logits.float().log_softmax(-1).detach()

    def embeds_from(self, page):
        tiles = normalized_tiles(page).to(dtype=self.pixels.dtype).unsqueeze(0)
        return build_embeds(
            self.model, self.full_ids, self.mask, self.pixels, self.sizes, self.aux_idx, tiles
        )

    def compression(self, page):
        clear_hooks(self.model)
        tiles = normalized_tiles(page).to(dtype=self.pixels.dtype)
        batch = self.pixels.detach().clone()
        batch[self.aux_idx] = tiles.detach()
        hooked = dict(self.inputs)
        hooked["pixel_values"] = batch
        kept = hook_kept(self.model, hooked, self.spans, RATIO)
        n_visual = sum(end - start for start, end in self.spans)
        k = official_keep_count(n_visual, RATIO)
        if len(kept) != k:
            raise RuntimeError(f"hook keep {len(kept)} != official K {k}")
        keep_index = keep_index_from_kept(kept, self.spans, self.full_ids.shape[1], page.device)
        embeds = self.embeds_from(page)
        logits = compressed_answer_logits(
            self.model, embeds, self.prompt_len, keep_index, int(self.answer_tokens.numel())
        ).float()
        if not torch.isfinite(logits).all():
            raise RuntimeError("non-finite compressed logits")
        loss = compression_margin_loss(logits, self.answer_tokens)
        margin = float(answer_margin(logits, self.answer_tokens).mean().detach())
        return loss, margin, k, kept

    def full_kl(self, page):
        clear_hooks(self.model)
        embeds = self.embeds_from(page)
        logits = full_answer_logits(self.model, embeds, self.prompt_len - 1, int(self.answer_tokens.numel()))
        if not torch.isfinite(logits).all():
            raise RuntimeError("non-finite full logits")
        return kl_clean_adv(self.log_p_clean, logits)


def separate_grads(session: Session, page):
    page = page.detach().requires_grad_(True)
    loss_c, margin, k, kept = session.compression(page)
    loss_c.backward()
    g_comp = page.grad.detach().clone()
    page.grad = None
    loss_f = session.full_kl(page)
    loss_f.backward()
    g_full = page.grad.detach().clone()
    return loss_c.detach(), loss_f.detach(), margin, k, kept, g_comp, g_full


def calibrate(session: Session) -> dict:
    loss_c, loss_f, margin, _, _, g_comp, g_full = separate_grads(session, session.clean_page)
    stats = grad_stats(g_comp, g_full)
    source = "delta0"
    scale_full = stats["full"]["l2"]
    # At δ=0 the KL is exactly 0, so its pixel gradient is numerical dust
    # (here ~1e-5 against a compression gradient of ~20). Require it to be
    # at least 0.1% of the compression gradient; otherwise use one fixed +1/255 shift.
    if float(loss_f) == 0.0 or scale_full < 1e-3 * stats["comp"]["l2"]:
        probe = (session.clean_page + (1.0 / 255.0)).clamp(0, 1)
        _, loss_shift, _, _, _, _, g_shift = separate_grads(session, probe)
        scale_full = float(g_shift.float().norm())
        source = "shift_1_over_255"
        if scale_full < 1e-6:
            raise RuntimeError("L_full gradient is degenerate at δ=0 and at a 1/255 shift")
    payload = {
        "sample_id": CALIBRATION_ID,
        "lambda_full": LAMBDA_FULL,
        "scale_comp_l2": stats["comp"]["l2"],
        "scale_full_l2": scale_full,
        "scale_full_source": source,
        "delta0_comp_loss": float(loss_c),
        "delta0_full_loss": float(loss_f),
        "delta0_margin": margin,
        "delta0_grad": stats,
        "rule": (
            "Divide each term by its L2 pixel-gradient norm, then set lambda_full=1. "
            "L_full is flat at δ=0, so its norm is taken from a fixed +1/255 shift "
            "when the δ=0 norm is below 0.1% of the compression gradient. "
            "Scales are frozen from sample 50743."
        ),
    }
    if payload["scale_comp_l2"] < 1e-8:
        raise RuntimeError("L_comp gradient is zero at δ=0 on the calibration sample")
    write_json(SCALE_PATH, payload)
    return payload


def load_scales() -> dict:
    if not SCALE_PATH.exists():
        raise RuntimeError(f"missing {SCALE_PATH}; run --calibrate first")
    return json.loads(SCALE_PATH.read_text())


def checkpoint_steps(steps: int) -> list[int]:
    chosen = [step for step in CHECKPOINTS if step <= steps]
    if steps not in chosen:
        chosen.append(steps)
    return chosen


def evaluate_uint8(session: Session, image: Image.Image, margin_start: float) -> dict:
    attacked = list(session.images)
    attacked[session.aux_idx] = image
    after = [sha_image(page) for page in attacked]
    assert_protected(session.item["page_sha"], after, session.aux_idx)
    linf = uint8_linf(session.images[session.aux_idx], image)
    if linf > EPS + 1e-6:
        raise RuntimeError(f"uint8 Linf {linf} exceeds eps")
    token_id = int(session.model.config.image_token_index)
    clear_hooks(session.model)
    pred_full, _, _, _ = run_plain(
        session.model, session.processor, attacked, session.question, token_id, False, session.item["evidence_idx"]
    )
    pred_fast, _, kept, _ = run_fastv(
        session.model, session.processor, attacked, session.question, token_id, RATIO,
        session.item["evidence_idx"], session.aux_idx,
    )
    clear_hooks(session.model)
    eval_inputs = prepare(session.processor, attacked, session.question)
    spans = spans_for(session.model, eval_inputs, len(attacked))
    hook = hook_kept(session.model, eval_inputs, spans, RATIO)
    if hook != list(kept):
        raise RuntimeError(
            f"keep-set mismatch at checkpoint: hook {len(hook)} vs runtime {len(kept)}, "
            f"symdiff {len(set(hook) ^ set(kept))}"
        )
    k = len(kept)
    full_ids = torch.cat([eval_inputs["input_ids"], session.answer_tokens.view(1, -1)], dim=1)
    _, mask = image_mask_of(session.model, full_ids)
    with torch.no_grad():
        embeds = build_embeds(
            session.model, full_ids, mask, eval_inputs["pixel_values"], eval_inputs["image_sizes"],
            session.aux_idx, None,
        )
        keep_index = keep_index_from_kept(hook, spans, full_ids.shape[1], embeds.device)
        logits = compressed_answer_logits(
            session.model, embeds, session.prompt_len, keep_index, int(session.answer_tokens.numel())
        ).float()
        loss_c = compression_margin_loss(logits, session.answer_tokens)
        margin = float(answer_margin(logits, session.answer_tokens).mean())
        loss_f = kl_clean_adv(session.log_p_clean, full_answer_logits(
            session.model, embeds, session.prompt_len - 1, int(session.answer_tokens.numel())
        ))
    stats = budget(kept, session.spans, session.item["evidence_idx"], session.aux_idx, [])
    anls_full = anls_score(pred_full, session.item["meta"]["answers"])
    anls_fast = anls_score(pred_fast, session.item["meta"]["answers"])
    valid = output_valid(pred_full) and output_valid(pred_fast)
    return {
        "pred_full": pred_full,
        "pred_fastv": pred_fast,
        "anls_full": anls_full,
        "anls_fastv": anls_fast,
        "output_valid": valid,
        "margin": margin,
        "full_kl": float(loss_f.detach()),
        "comp_loss": float(loss_c.detach()),
        "aux_share": stats["aux_share"],
        "evidence_share": stats["evidence_share"],
        "evidence_keep": stats["evidence_keep"],
        "kept": list(kept),
        "keep_hash": keep_hash(list(kept)),
        "k": k,
        "linf": linf,
        "page_sha": sha_image(image),
        "label_if_chosen": classify_sample(
            float(session.item["baseline_full"]["anls"]),
            float(session.item["baseline_fastv"]["anls"]),
            anls_full,
            anls_fast,
            margin,
            margin_start,
            valid,
        ),
    }


def attack(session: Session, steps: int, scales: dict, grad_log: Path | None, curve: Path, artifact_dir: Path):
    """Same loop as run_sample, but retains the uint8 image of every checkpoint."""
    delta = torch.zeros_like(session.clean_page, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=2.0 / 255.0)
    scale_c = float(scales["scale_comp_l2"])
    scale_f = float(scales["scale_full_l2"])
    marks = set(checkpoint_steps(steps))
    rows = []
    images = {}
    previous_kept = None
    margin_start = None
    t0 = time.time()
    for step in range(steps + 1):
        if grad_log is not None and step < 5 and step < steps:
            page_now = (session.clean_page + delta).detach()
            loss_c, loss_f, margin, _, _, g_comp, g_full = separate_grads(session, page_now)
            append_jsonl(grad_log, {
                "sample_id": session.item["meta"]["sample_id"],
                "step": step,
                "L_comp": float(loss_c),
                "L_full": float(loss_f),
                "margin": margin,
                **{f"grad_{name}": value for name, value in grad_stats(g_comp, g_full).items()},
            })
        if step in marks or step == steps:
            image = page_uint8(session.clean_page, delta)
            measured = evaluate_uint8(session, image, 0.0 if margin_start is None else margin_start)
            if margin_start is None:
                margin_start = measured["margin"]
            measured["label_if_chosen"] = classify_sample(
                float(session.item["baseline_full"]["anls"]),
                float(session.item["baseline_fastv"]["anls"]),
                measured["anls_full"],
                measured["anls_fastv"],
                measured["margin"],
                margin_start,
                measured["output_valid"],
            )
            jac = 1.0 if previous_kept is None else jaccard(previous_kept, measured["kept"])
            previous_kept = measured["kept"]
            row = {
                "step": step,
                "margin": measured["margin"],
                "anls_full": measured["anls_full"],
                "anls_fastv": measured["anls_fastv"],
                "output_valid": measured["output_valid"],
                "linf": measured["linf"],
                "pred_full": measured["pred_full"],
                "pred_fastv": measured["pred_fastv"],
                "full_kl": measured["full_kl"],
                "aux_share": measured["aux_share"],
                "evidence_share": measured["evidence_share"],
                "evidence_keep": measured["evidence_keep"],
                "keep_hash": measured["keep_hash"],
                "jaccard_prev": jac,
                "page_sha": measured["page_sha"],
                "k": measured["k"],
                "label_if_chosen": measured["label_if_chosen"],
                "kept": measured["kept"],
            }
            rows.append(row)
            images[step] = image
            public = {key: value for key, value in row.items() if key != "kept"}
            public["sample_id"] = session.item["meta"]["sample_id"]
            append_jsonl(curve, public)
            print(json.dumps({
                "id": session.item["meta"]["sample_id"], "step": step,
                "fastv": measured["anls_fastv"], "full": measured["anls_full"],
                "margin": measured["margin"], "label": measured["label_if_chosen"],
                "jaccard": jac,
            }), flush=True)
        if step == steps:
            break
        opt.zero_grad(set_to_none=True)
        loss_c, _, _, _ = session.compression(session.clean_page + delta)
        (loss_c / scale_c).backward()
        loss_f = session.full_kl(session.clean_page + delta)
        (LAMBDA_FULL * loss_f / scale_f).backward()
        if delta.grad is None or not torch.isfinite(delta.grad).all():
            raise RuntimeError(f"non-finite gradient at step {step}")
        opt.step()
        project(delta, session.clean_page, EPS)
        torch.cuda.empty_cache()
    best = select_best(rows)
    best_image = images[int(best["step"])]
    artifact_dir.mkdir(parents=True, exist_ok=True)
    best_path = artifact_dir / "best.png"
    best_image.save(best_path)
    replay = evaluate_uint8(session, Image.open(best_path).convert("RGB"), margin_start)
    replay_ok = (
        replay["pred_full"] == best["pred_full"]
        and replay["pred_fastv"] == best["pred_fastv"]
        and replay["keep_hash"] == best["keep_hash"]
    )
    label = best["label_if_chosen"]
    if not replay_ok:
        label_note = "replay_mismatch"
    else:
        label_note = "replay_ok"
    float_gap = (session.clean_page.detach() - page_to_float(best_image).to(session.clean_page.device)).abs()
    provenance = {
        "sample_id": session.item["meta"]["sample_id"],
        "group": GROUPS[session.item["meta"]["sample_id"]],
        "best_step": int(best["step"]),
        "label": label,
        "replay_ok": replay_ok,
        "replay_note": label_note,
        "original_page_sha256": session.item["page_sha"][session.aux_idx],
        "best_sha256": sha_image(best_image),
        "aux_idx": session.aux_idx,
        "protected_sha256_before": session.item["page_sha"],
        "protected_sha256_after": [
            session.item["page_sha"][i] if i != session.aux_idx else sha_image(best_image)
            for i in range(len(session.images))
        ],
        "float_linf": float(float_gap.max()),
        "float_l2": float(float_gap.float().norm()),
        "uint8_linf": best["linf"],
        "pred_full": best["pred_full"],
        "pred_fastv": best["pred_fastv"],
        "margin": best["margin"],
        "margin_start": margin_start,
        "full_kl": best["full_kl"],
        "keep": best["kept"],
        "keep_hash": best["keep_hash"],
        "anls_full": best["anls_full"],
        "anls_fastv": best["anls_fastv"],
        "eps": EPS,
        "seconds": time.time() - t0,
    }
    write_json(artifact_dir / "provenance.json", provenance)
    csv_row = {
        "sample_id": session.item["meta"]["sample_id"],
        "group": GROUPS[session.item["meta"]["sample_id"]],
        "best_step": int(best["step"]),
        "label": label,
        "replay_ok": int(replay_ok),
        "anls_full": f"{best['anls_full']:.4f}",
        "anls_fastv": f"{best['anls_fastv']:.4f}",
        "pred_full": best["pred_full"],
        "pred_fastv": best["pred_fastv"],
        "margin": f"{best['margin']:.6f}",
        "margin_start": f"{margin_start:.6f}",
        "full_kl": f"{best['full_kl']:.6f}",
        "aux_share": f"{best['aux_share']:.6f}",
        "evidence_share": f"{best['evidence_share']:.6f}",
        "evidence_keep": f"{best['evidence_keep']:.6f}",
        "linf_uint8": f"{best['linf']:.8f}",
        "keep_hash": best["keep_hash"],
        "eps_ok": int(best["linf"] <= EPS + 1e-6),
    }
    return csv_row, time.time() - t0


def done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    import csv
    with path.open() as handle:
        return {row["sample_id"] for row in csv.DictReader(handle)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--ids", default=",".join(["50743", "4912", "56776", "49242", "33923", "5169"]))
    parser.add_argument("--tag", default="official")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshards", type=int, default=1)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--grad-log", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(42)
    wanted = [item for item in args.ids.split(",") if item]
    samples = [item for item in load_frozen_samples() if item["meta"]["sample_id"] in wanted]
    samples = [item for index, item in enumerate(samples) if index % args.nshards == args.shard]
    if not samples:
        raise SystemExit("no samples for this shard")
    model, processor = load_model()
    for param in model.parameters():
        param.requires_grad_(False)
    if args.calibrate:
        item = next(sample for sample in load_frozen_samples() if sample["meta"]["sample_id"] == CALIBRATION_ID)
        clear_hooks(model)
        inputs = prepare(processor, item["images"], item["meta"]["question"])
        with torch.inference_mode():
            raw = generate(model, inputs)
        answer = torch.tensor(raw.detach().cpu().tolist(), dtype=torch.long)
        scales = calibrate(Session(model, processor, item, answer))
        print(json.dumps({"calibrated": True, "scale_full_source": scales["scale_full_source"],
                          "scale_comp": scales["scale_comp_l2"], "scale_full": scales["scale_full_l2"]}), flush=True)
        if args.steps == 0:
            return
    scales = load_scales()
    tag = args.tag
    curve = V4 / "results" / ("checkpoints.jsonl" if tag == "official" else f"checkpoints_{tag}.jsonl")
    out = V4 / "results" / ("smoke.csv" if tag == "official" else f"smoke_{tag}.csv")
    grad_path = V4 / "results" / "grad_probe.jsonl" if args.grad_log else None
    finished = done_ids(out)
    for item in samples:
        if item["meta"]["sample_id"] in finished:
            continue
        clear_hooks(model)
        inputs = prepare(processor, item["images"], item["meta"]["question"])
        with torch.inference_mode():
            raw = generate(model, inputs)
        answer = torch.tensor(raw.detach().cpu().tolist(), dtype=torch.long)
        session = Session(model, processor, item, answer)
        artifact = V4 / "artifacts" / item["meta"]["sample_id"] if tag == "official" else V4 / "artifacts" / tag / item["meta"]["sample_id"]
        row, seconds = attack(session, args.steps, scales, grad_path, curve, artifact)
        append_csv(out, row, FIELDS)
        print(json.dumps({"done": row["sample_id"], "label": row["label"], "sec": round(seconds, 1)}), flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
