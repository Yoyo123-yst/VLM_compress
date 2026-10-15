"""B-Native, native-FDP, and B+ on the frozen 12-question development set."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
HERE = ROOT / "week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation"
sys.path[:0] = [str(ROOT), str(ROOT / "week1_mechanism"), str(ROOT / "gate_experiment")]

from adapters.runtime import clear_hooks, install  # noqa: E402
from metrics_task import anls_score, normalize  # noqa: E402
from run_causal_v2 import prepare, spans_of  # noqa: E402
from run_eval import decode  # noqa: E402
from week2_attack_v4_answer_aligned.common import keep_hash, sha_image  # noqa: E402
from week2_attack_v4_answer_aligned.forward import keep_index_from_kept  # noqa: E402
from week2_attack_v4_answer_aligned.loss import output_valid  # noqa: E402
from week2_attack_v4_answer_aligned.run_smoke import page_uint8, project, uint8_linf  # noqa: E402
from week2_attack_v5b_expanded_pristine40.load_samples import load_formal_items, manifest as v5b_manifest  # noqa: E402
from week2_attack_v7_pristine60.load_samples import load_items, manifest as pristine_manifest  # noqa: E402
from week2_attack_v7_pristine60.run_v7 import load_model_frozen  # noqa: E402
from week2_attack_v7b_native_cache_margin_admission.native_cache_margin import (  # noqa: E402
    cache_replay,
    hard_margin_row,
    production_generate,
    soft_margin_row,
)
from week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation.losses import (  # noqa: E402
    ETA,
    XI,
    conditional_gap,
    decision_floor,
    decision_preservation,
    method_loss,
    soft_anchor,
)
from week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation.native_grad import (  # noqa: E402
    TRAIN_RATIOS,
    fastv_rows,
    full_rows,
    mean_hard,
    mean_soft,
    splice_aux,
)
from week2_pilot.attack import normalized_tiles, page_to_float  # noqa: E402

EPS = 8 / 255
SEED = 2026101102
CHECKPOINTS = (0, 10, 25, 50, 75, 100, 150, 200)
DEV = ["1025", "2008", "7006", "39204", "47450", "47567", "50779", "51133", "60700", "6194", "62941", "64506"]
METHODS = ("B-Native", "native-FDP", "B+")
LEGAL = ("step", "nonzero", "output_valid", "full_exact", "linf_ok", "pixel_linf_uint8", "margin")


def evidence_index(item: dict) -> int:
    value = item["evidence_idx"]
    return int(value if isinstance(value, int) else value[0])


def load_dev(sample_id: str) -> dict:
    spec = next(row for row in v5b_manifest()["samples"] if row["sample_id"] == sample_id)
    return load_formal_items([spec])[0]


def load_pristine(sample_id: str) -> dict:
    spec = next(row for row in pristine_manifest()["samples"] if row["sample_id"] == sample_id)
    return load_items([spec])[0]


def pixels_from_page(inputs: dict, aux_idx: int, page: torch.Tensor) -> torch.Tensor:
    return splice_aux(inputs["pixel_values"], aux_idx, normalized_tiles(page))


def generate_tokens(model, inputs: dict, pixels: torch.Tensor, spans, ratio: float | None) -> tuple[list[int], list[int] | None]:
    local = dict(inputs)
    local["pixel_values"] = pixels.detach()
    clear_hooks(model)
    kept = None
    if ratio is not None:
        state = install(model, "fastv", ratio, int(model.config.image_token_index))
        state.spans = spans
    generated = production_generate(model, local)
    if ratio is not None:
        kept = list(state.kept_visual)
    clear_hooks(model)
    return generated["tokens"], kept


def soft_fast(model, inputs, spans, page, aux_idx, ratio: float) -> torch.Tensor:
    if ratio not in TRAIN_RATIOS:
        raise RuntimeError(f"training gradient requested at r={ratio}")
    pixels = pixels_from_page(inputs, aux_idx, page)
    tokens, kept = generate_tokens(model, inputs, pixels, spans, ratio)
    keep = keep_index_from_kept(kept, spans, int(inputs["input_ids"].shape[1]), page.device)
    rows = fastv_rows(model, inputs, pixels, tokens, keep)
    if [int(row.argmax()) for row in rows] != tokens:
        raise RuntimeError("FastV differentiable argmax left the generate tokens")
    return mean_soft(rows, tokens)


def full_token_rows(model, inputs, page, aux_idx, tokens: list[int], check_argmax: bool):
    pixels = pixels_from_page(inputs, aux_idx, page)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    try:
        rows = full_rows(model, inputs, pixels, tokens)
    finally:
        model.eval()
        model.gradient_checkpointing_disable()
    if check_argmax and [int(row.argmax()) for row in rows] != tokens:
        raise RuntimeError("Full differentiable argmax left the generate tokens")
    return rows


def release(page: torch.Tensor) -> None:
    page.grad = None
    torch.cuda.empty_cache()


def backward_loss(loss: torch.Tensor, page: torch.Tensor, bucket: torch.Tensor | None) -> None:
    if bucket is not None and page.grad is not None:
        page.grad = None
    loss.backward()
    if page.grad is None or not torch.isfinite(page.grad).all():
        raise RuntimeError("non-finite aux gradient")
    if bucket is not None:
        bucket.add_(page.grad.detach())
        page.grad = None
    del loss
    torch.cuda.empty_cache()


def term_tensors(model, inputs, spans, page, aux_idx, clean_tokens, clean_hard, method: str | None) -> dict[str, torch.Tensor]:
    fast = [soft_fast(model, inputs, spans, page, aux_idx, ratio) for ratio in TRAIN_RATIOS]
    compressed = torch.stack(fast).mean()
    need_gap = method in {None, "B+"}
    need_fdp = method in {None, "native-FDP", "B+"}
    need_anchor = method in {None, "B-Native"}
    m_full = None
    if need_gap:
        own_tokens, _ = generate_tokens(model, inputs, pixels_from_page(inputs, aux_idx, page), spans, None)
        own_rows = full_token_rows(model, inputs, page, aux_idx, own_tokens, True)
        m_full = mean_soft(own_rows, own_tokens)
    anchor = fdp = gap = None
    if need_anchor or need_fdp:
        clean_rows = full_token_rows(model, inputs, page, aux_idx, clean_tokens, False)
        soft = torch.stack([soft_margin_row(row, token) for row, token in zip(clean_rows, clean_tokens)])
        hard = torch.stack([hard_margin_row(row, token) for row, token in zip(clean_rows, clean_tokens)])
        if need_anchor:
            anchor = soft_anchor(soft, clean_hard)
        if need_fdp:
            fdp = decision_preservation(hard, clean_hard)
    if need_gap:
        gap = conditional_gap(m_full, fast)
    return {"L_compressed": compressed, "L_native_soft_anchor": anchor, "L_FDP": fdp, "L_conditional_gap": gap}


def grad_of(term: torch.Tensor, page: torch.Tensor) -> float:
    if page.grad is not None:
        page.grad = None
    term.backward(retain_graph=True)
    if page.grad is None or not torch.isfinite(page.grad).all():
        return float("nan")
    return float(page.grad.float().norm())


def prepare_item(model, processor, item) -> dict:
    inputs = prepare(processor, item["images"], item["meta"]["question"])
    spans = spans_of(inputs, int(model.config.image_token_index), len(item["images"]))
    page = page_to_float(item["images"][item["aux_idx"]]).to("cuda")
    clear_hooks(model)
    clean = production_generate(model, inputs)
    clean_text = decode(processor, torch.tensor(clean["tokens"])).strip()
    with torch.no_grad():
        rows = full_rows(model, inputs, inputs["pixel_values"], clean["tokens"])
        clean_hard = torch.stack([hard_margin_row(row, token) for row, token in zip(rows, clean["tokens"])]).detach()
    return {
        "inputs": inputs,
        "spans": spans,
        "page": page,
        "clean_tokens": clean["tokens"],
        "clean_text": clean_text,
        "clean_hard": clean_hard,
        "floor": decision_floor(clean_hard).detach(),
    }


def fast_grad(model, state, item, page, ratio: float, scale: float, bucket: torch.Tensor | None, leaf: torch.Tensor | None = None) -> None:
    backward_loss(soft_fast(model, state["inputs"], state["spans"], page, item["aux_idx"], ratio) * scale, page if leaf is None else leaf, bucket)


def anchor_grad(model, state, item, page, scale: float, bucket: torch.Tensor | None, leaf: torch.Tensor | None = None) -> None:
    rows = full_token_rows(model, state["inputs"], page, item["aux_idx"], state["clean_tokens"], False)
    soft = torch.stack([soft_margin_row(row, token) for row, token in zip(rows, state["clean_tokens"])])
    backward_loss(soft_anchor(soft, state["clean_hard"]) * scale, page if leaf is None else leaf, bucket)


def fdp_grad(model, state, item, page, scale: float, bucket: torch.Tensor | None, leaf: torch.Tensor | None = None) -> None:
    rows = full_token_rows(model, state["inputs"], page, item["aux_idx"], state["clean_tokens"], False)
    hard = torch.stack([hard_margin_row(row, token) for row, token in zip(rows, state["clean_tokens"])])
    backward_loss(decision_preservation(hard, state["clean_hard"]) * scale, page if leaf is None else leaf, bucket)


def full_margin_grad(model, state, item, page, scale: float, bucket: torch.Tensor | None, leaf: torch.Tensor | None = None) -> None:
    tokens, _ = generate_tokens(model, state["inputs"], pixels_from_page(state["inputs"], item["aux_idx"], page), state["spans"], None)
    rows = full_token_rows(model, state["inputs"], page, item["aux_idx"], tokens, True)
    backward_loss(mean_soft(rows, tokens) * scale, page if leaf is None else leaf, bucket)


def isolated_norm(model, state, item, page, kind: str) -> float:
    bucket = torch.zeros_like(page)
    if kind == "L_compressed":
        for ratio in TRAIN_RATIOS:
            fast_grad(model, state, item, page, ratio, 0.5, bucket)
    elif kind == "L_native_soft_anchor":
        anchor_grad(model, state, item, page, 1.0, bucket)
    elif kind == "L_FDP":
        fdp_grad(model, state, item, page, 1.0, bucket)
    elif kind == "L_conditional_gap":
        full_margin_grad(model, state, item, page, -1.0, bucket)
        for ratio in TRAIN_RATIOS:
            fast_grad(model, state, item, page, ratio, 0.5, bucket)
    else:
        raise RuntimeError(kind)
    return float(bucket.float().norm())


def probe(model, processor, item, delta_value: float) -> dict:
    state = prepare_item(model, processor, item)
    page = state["page"].detach().clone().requires_grad_(True)
    if delta_value:
        with torch.no_grad():
            page.add_(delta_value)
            page.clamp_(0, 1)
    norms = {
        name: isolated_norm(model, state, item, page, name)
        for name in ("L_compressed", "L_FDP", "L_conditional_gap", "L_native_soft_anchor")
    }
    finite = all(math.isfinite(value) and value > 0 for value in norms.values())
    return {"delta": delta_value, "grad_l2": norms, "finite_nonzero": finite, "clean_text": state["clean_text"]}


def backward_method(model, state, item, page, method: str, coeffs: dict, leaf: torch.Tensor | None = None) -> None:
    fast_scale = 0.5
    if method == "B+":
        fast_scale += 0.5 * float(coeffs["lambda_gap"])
    for ratio in TRAIN_RATIOS:
        fast_grad(model, state, item, page, ratio, fast_scale, None, leaf)
    if method == "B-Native":
        anchor_grad(model, state, item, page, float(coeffs["lambda_full"]), None, leaf)
    if method in {"native-FDP", "B+"}:
        fdp_grad(model, state, item, page, float(coeffs["lambda_full"]), None, leaf)
    if method == "B+":
        full_margin_grad(model, state, item, page, -float(coeffs["lambda_gap"]), None, leaf)


def usable(value: float) -> bool:
    return math.isfinite(value) and value > 0


def freeze_lambdas(plus: dict) -> dict:
    norms = plus["grad_l2"]
    blocked = []
    if not (usable(norms["L_compressed"]) and usable(norms["L_native_soft_anchor"]) and usable(norms["L_FDP"])):
        blocked.append("B-Native")
    if not (usable(norms["L_compressed"]) and usable(norms["L_FDP"])):
        blocked.append("native-FDP")
    if "native-FDP" in blocked or not usable(norms["L_conditional_gap"]):
        blocked.append("B+")
    coeffs = None
    if usable(norms["L_compressed"]) and usable(norms["L_FDP"]) and usable(norms["L_conditional_gap"]):
        coeffs = {
            "lambda_full": norms["L_compressed"] / norms["L_FDP"],
            "lambda_gap": norms["L_compressed"] / norms["L_conditional_gap"],
            "xi": XI,
            "eta": ETA,
            "source": "+1/255",
        }
    elif usable(norms["L_compressed"]) and usable(norms["L_FDP"]):
        coeffs = {
            "lambda_full": norms["L_compressed"] / norms["L_FDP"],
            "lambda_gap": None,
            "xi": XI,
            "eta": ETA,
            "source": "+1/255",
        }
    return {"blocked_methods": blocked, "coefficients": coeffs}


def score_png(model, processor, item, image: Image.Image, clean_text: str) -> dict:
    attacked = list(item["images"])
    attacked[item["aux_idx"]] = image
    linf = uint8_linf(item["images"][item["aux_idx"]], image)
    inputs = prepare(processor, attacked, item["meta"]["question"])
    spans = spans_of(inputs, int(model.config.image_token_index), len(attacked))
    full_tokens, _ = generate_tokens(model, inputs, inputs["pixel_values"], spans, None)
    full_text = decode(processor, torch.tensor(full_tokens)).strip()
    margins = {}
    keeps = {}
    texts = {"full": full_text}
    for ratio in (0.25, 0.35):
        tokens, kept = generate_tokens(model, inputs, inputs["pixel_values"], spans, ratio)
        texts[f"fastv_{ratio}"] = decode(processor, torch.tensor(tokens)).strip()
        local = dict(inputs)
        clear_hooks(model)
        state = install(model, "fastv", ratio, int(model.config.image_token_index))
        state.spans = spans
        rows, _ = cache_replay(model, local, tokens)
        clear_hooks(model)
        margins[str(ratio)] = float(torch.stack([soft_margin_row(row, token) for row, token in zip(rows, tokens)]).mean())
        keeps[str(ratio)] = {"k": len(kept), "keep_hash": keep_hash(kept)}
    valid = all(output_valid(text) for text in texts.values())
    margin = 0.5 * (margins["0.25"] + margins["0.35"])
    row = {
        "nonzero": int(not np.array_equal(np.asarray(image), np.asarray(item["images"][item["aux_idx"]]))),
        "output_valid": int(valid),
        "full_exact": int(full_text == clean_text),
        "linf_ok": int(0 < linf <= EPS + 1e-6) if linf > 0 else int(linf <= EPS + 1e-6),
        "pixel_linf_uint8": linf,
        "margin": margin,
    }
    return {"selector_fields": row, "texts": texts, "margins": margins, "keeps": keeps, "full_tokens": len(full_tokens)}


def choose(candidates: list[dict]) -> dict | None:
    for row in candidates:
        if set(row) != set(LEGAL):
            raise RuntimeError("selector saw a field outside the frozen list")
    eligible = [
        row for row in candidates
        if int(row["step"]) > 0 and row["nonzero"] and row["output_valid"] and row["full_exact"] and row["linf_ok"]
        and float(row["pixel_linf_uint8"]) <= EPS + 1e-6
    ]
    if not eligible:
        return None
    return sorted(eligible, key=lambda row: (float(row["margin"]), int(row["step"])))[0]


def attack(model, processor, item, method: str, steps: int, coeffs: dict, out: Path) -> dict:
    if method not in METHODS:
        raise RuntimeError(method)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED % (2**32))
    state = prepare_item(model, processor, item)
    delta = torch.zeros_like(state["page"], requires_grad=True)
    opt = torch.optim.Adam([delta], lr=2.0 / 255.0)
    out.mkdir(parents=True, exist_ok=True)
    item["images"][item["aux_idx"]].save(out / "clean.png")
    marks = {step for step in CHECKPOINTS if step <= steps}
    marks.add(steps)
    curve = []
    selector_rows = []
    for step in range(steps + 1):
        if step in marks:
            image = page_uint8(state["page"], delta)
            dest = out / "checkpoints" / f"step_{step:03d}.png"
            dest.parent.mkdir(parents=True, exist_ok=True)
            image.save(dest)
            reloaded = Image.open(dest).convert("RGB")
            scored = score_png(model, processor, item, reloaded, state["clean_text"])
            fields = dict(scored["selector_fields"])
            fields["step"] = step
            fields["linf_ok"] = int(step > 0 and fields["nonzero"] and 0 < fields["pixel_linf_uint8"] <= EPS + 1e-6)
            selector_rows.append({key: fields[key] for key in LEGAL})
            curve.append({
                "step": step,
                "png_sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
                "texts": scored["texts"],
                "margins": scored["margins"],
                "keeps": scored["keeps"],
                "selector": selector_rows[-1],
            })
            print(json.dumps({"sample": item["meta"]["sample_id"], "method": method, "step": step, "full_exact": fields["full_exact"]}), flush=True)
        if step == steps:
            break
        opt.zero_grad(set_to_none=True)
        page = state["page"] + delta
        backward_method(model, state, item, page, method, coeffs, delta)
        if delta.grad is None or not torch.isfinite(delta.grad).all():
            raise RuntimeError(f"non-finite gradient at step {step}")
        opt.step()
        project(delta, state["page"], EPS)
        torch.cuda.empty_cache()
    chosen = choose(selector_rows)
    (out / "curve.json").write_text(json.dumps(curve, indent=2) + "\n")
    (out / "selected.json").write_text(json.dumps({"chosen": chosen, "clean_text": state["clean_text"]}, indent=2) + "\n")
    return {"chosen": chosen, "clean_text": state["clean_text"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("calibrate", "calibrate-term", "health", "attack"))
    parser.add_argument("--sample")
    parser.add_argument("--method")
    parser.add_argument("--term")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--out")
    args = parser.parse_args()
    model, processor = load_model_frozen()
    model.eval()
    if args.mode == "calibrate-term":
        item = load_pristine(args.sample)
        measured = {}
        for label, value in (("delta0", 0.0), ("plus", 1 / 255)):
            state = prepare_item(model, processor, item)
            page = state["page"].detach().clone().requires_grad_(True)
            if value:
                with torch.no_grad():
                    page.add_(value)
                    page.clamp_(0, 1)
            measured[label] = isolated_norm(model, state, item, page, args.term)
            print(json.dumps({"term": args.term, "probe": label, "grad_l2": measured[label]}), flush=True)
        dest = HERE / "calibration_parts" / f"{args.term}.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps({"sample_id": args.sample, "term": args.term, "grad_l2": measured}, indent=2) + "\n")
        return
    if args.mode == "health":
        item = load_pristine(args.sample)
        state = prepare_item(model, processor, item)
        clear_hooks(model)
        full_tokens, _ = generate_tokens(model, state["inputs"], state["inputs"]["pixel_values"], state["spans"], None)
        with torch.no_grad():
            full_match = [int(row.argmax()) for row in full_rows(model, state["inputs"], state["inputs"]["pixel_values"], full_tokens)] == full_tokens
        fast = {}
        for ratio in (0.25, 0.35):
            tokens, kept = generate_tokens(model, state["inputs"], state["inputs"]["pixel_values"], state["spans"], ratio)
            from week2_attack_v4_answer_aligned.common import keep_hash as _keep_hash
            fast[str(ratio)] = {"k": len(kept), "keep_hash": _keep_hash(kept), "n_tokens": len(tokens)}
        page = state["page"].detach().clone().requires_grad_(True)
        norms = {name: isolated_norm(model, state, item, page, name) for name in ("L_compressed", "L_FDP", "L_conditional_gap", "L_native_soft_anchor")}
        pixels = pixels_from_page(state["inputs"], item["aux_idx"], page)
        evidence = evidence_index(item)
        evidence_grad = torch.autograd.grad(pixels[evidence].float().sum(), page, allow_unused=True)[0]
        evidence_l2 = 0.0 if evidence_grad is None else float(evidence_grad.float().norm())
        report = {
            "sample_id": args.sample,
            "full_argmax_match": full_match,
            "fastv": fast,
            "grad_l2": norms,
            "evidence_grad_l2": evidence_l2,
            "aux_grad_nonzero": all(math.isfinite(value) and value > 0 for value in norms.values()),
        }
        report["pass"] = bool(report["full_argmax_match"] and report["aux_grad_nonzero"] and evidence_l2 == 0.0)
        dest = HERE / "health" / f"{args.sample}.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"sample": args.sample, "pass": report["pass"]}), flush=True)
        return
    if args.mode == "calibrate":
        protocol = json.loads((HERE / "FROZEN_BPLUS_DEV_PROTOCOL.json").read_text())
        order = [sid for sid in protocol["calibration_candidates_in_order"] if sid not in set(DEV)]
        report = []
        chosen = None
        for sample_id in order:
            item = load_pristine(sample_id)
            try:
                zero = probe(model, processor, item, 0.0)
                plus = probe(model, processor, item, 1 / 255)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    raise
                report.append({"sample_id": sample_id, "error": str(exc)})
                torch.cuda.empty_cache()
                continue
            decision = freeze_lambdas(plus)
            report.append({"sample_id": sample_id, "delta0": zero, "plus": plus, "decision": decision})
            chosen = report[-1]
            break
        payload = {"tried": report, "chosen": chosen}
        (HERE / "calibration.json").write_text(json.dumps(payload, indent=2) + "\n")
        if chosen is not None:
            frozen = {
                "sample_id": chosen["sample_id"],
                "blocked_methods": chosen["decision"]["blocked_methods"],
                "coefficients": chosen["decision"]["coefficients"],
                "delta0_grad_l2": chosen["delta0"]["grad_l2"],
                "plus_grad_l2": chosen["plus"]["grad_l2"],
            }
            (HERE / "FROZEN_COEFFICIENTS.json").write_text(json.dumps(frozen, indent=2) + "\n")
        print(json.dumps({"chosen": None if chosen is None else chosen["sample_id"], "blocked": None if chosen is None else chosen["decision"]["blocked_methods"]}), flush=True)
        return
    coeffs = json.loads((HERE / "FROZEN_COEFFICIENTS.json").read_text())
    if args.method in coeffs["blocked_methods"]:
        raise RuntimeError(f"{args.method} is BLOCKED and cannot start PGD")
    item = load_dev(args.sample) if args.sample in DEV else load_pristine(args.sample)
    attack(model, processor, item, args.method, args.steps, coeffs["coefficients"], Path(args.out))


if __name__ == "__main__":
    main()
