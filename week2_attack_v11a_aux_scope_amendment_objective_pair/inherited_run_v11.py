"""White-box diagnostic of FastV reference choice and the update rule.

The loss form, Full-FDP reference, and lambda are the frozen Native-FDP
settings. Dynamic versus fixed changes only the FastV reference sequence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
HERE = ROOT / "week2_attack_v11_objective_optimizer_diagnostic"
import sys

sys.path[:0] = [str(ROOT), str(ROOT / "week1_mechanism"), str(ROOT / "gate_experiment"), str(HERE)]

from metrics_task import anls_score  # noqa: E402
from run_eval import decode  # noqa: E402
from week2_attack_v4_answer_aligned.common import keep_hash  # noqa: E402
from week2_attack_v4_answer_aligned.loss import output_valid  # noqa: E402
from week2_attack_v4_answer_aligned.run_smoke import page_uint8, project, uint8_linf  # noqa: E402
from week2_attack_v7_pristine60.run_v7 import load_model_frozen  # noqa: E402
from week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation import run_v7d as v7d  # noqa: E402
from week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation.losses import decision_preservation  # noqa: E402

IDS = ("1025", "2008", "47450", "60700", "7006", "39204")
TRAIN_RATIOS = (0.25, 0.35)
EVAL_RATIOS = (0.15, 0.25, 0.35, 0.50)
CHECKPOINTS = (0, 10, 25, 50, 75, 100, 150, 200)
STEPS = 200
SEED = 2026101102
LAMBDA_FULL = 2.6371882367154997
ADAM_LR = 2.0 / 255.0
PGD_STEP = 1.0 / 255.0
EPS = 8 / 255
ANLS_CLEAN = 0.5
GROUPS = {
    "H0": ("dyn", "adam"),
    "H1": ("fixed", "adam"),
    "P0": ("dyn", "pgd"),
    "P1": ("fixed", "pgd"),
    "O0": ("dyn", "opgd"),
    "O1": ("fixed", "opgd"),
}


class Access:
    forward = 0
    backward = 0
    generate = 0
    fastv = 0
    generated_tokens = 0


def install_counters(model) -> None:
    original_generate = model.generate
    original_backward = torch.Tensor.backward

    def generate(*args, **kwargs):
        Access.generate += 1
        return original_generate(*args, **kwargs)

    def count_forward(module, args, kwargs):
        if module is model:
            Access.forward += 1

    def backward(self, *args, **kwargs):
        Access.backward += 1
        return original_backward(self, *args, **kwargs)

    model.generate = generate
    model.register_forward_pre_hook(count_forward, with_kwargs=True)
    torch.Tensor.backward = backward
    original_tokens = v7d.generate_tokens

    def generate_tokens(model, inputs, pixels, spans, ratio=None):
        if ratio is not None:
            Access.fastv += 1
        tokens, kept = original_tokens(model, inputs, pixels, spans, ratio)
        Access.generated_tokens += len(tokens)
        return tokens, kept

    v7d.generate_tokens = generate_tokens


def reset_counters() -> None:
    Access.forward = Access.backward = Access.generate = Access.fastv = Access.generated_tokens = 0


def counters() -> dict:
    return {
        "forward": Access.forward,
        "backward": Access.backward,
        "generate": Access.generate,
        "fastv": Access.fastv,
        "generated_tokens": Access.generated_tokens,
    }


def sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def fast_margin(model, state, item, page, ratio: float, reference: list[int] | None):
    """Current FastV keep-set, then soft margin on the supplied reference."""
    pixels = v7d.pixels_from_page(state["inputs"], item["aux_idx"], page)
    generated, kept = v7d.generate_tokens(model, state["inputs"], pixels, state["spans"], ratio)
    target = generated if reference is None else reference
    keep = v7d.keep_index_from_kept(kept, state["spans"], int(state["inputs"]["input_ids"].shape[1]), page.device)
    rows = v7d.fastv_rows(model, state["inputs"], pixels, target, keep)
    if reference is None and [int(row.detach().argmax()) for row in rows] != generated:
        raise RuntimeError("dynamic FastV argmax left the generated tokens")
    if len(rows) != len(target):
        raise RuntimeError("teacher-forced length left the reference")
    return v7d.mean_soft(rows, target), kept, generated


def fdp_value(model, state, item, page):
    rows = v7d.full_token_rows(model, state["inputs"], page, item["aux_idx"], state["clean_tokens"], False)
    hard = torch.stack([v7d.hard_margin_row(row, token) for row, token in zip(rows, state["clean_tokens"])])
    violated = bool((hard.detach() <= 0).any())
    loss = decision_preservation(hard, state["clean_hard"])
    return loss, violated


def attack_grads(model, state, item, page, mode: str, fixed: dict, leaf: torch.Tensor):
    leaf.grad = None
    total = None
    keeps = {}
    for ratio in TRAIN_RATIOS:
        reference = None if mode == "dyn" else fixed[ratio]["tokens"]
        margin, kept, generated = fast_margin(model, state, item, page, ratio, reference)
        keeps[ratio] = {"k": len(kept), "keep_hash": keep_hash(kept), "generated": generated, "source": "current_perturbed_fastv_generate"}
        term = 0.5 * margin
        total = term if total is None else total + term
    (total).backward()
    if leaf.grad is None or not torch.isfinite(leaf.grad).all():
        raise RuntimeError("non-finite attack gradient")
    attack = leaf.grad.detach().clone()
    leaf.grad = None
    fdp, violated = fdp_value(model, state, item, page)
    (LAMBDA_FULL * fdp).backward()
    if leaf.grad is None or not torch.isfinite(leaf.grad).all():
        raise RuntimeError("non-finite Full gradient")
    full = leaf.grad.detach().clone()
    leaf.grad = None
    return attack, full, violated, keeps


def project_out(source: torch.Tensor, onto: torch.Tensor) -> torch.Tensor:
    denom = torch.dot(onto.flatten(), onto.flatten())
    if float(denom) == 0.0:
        return source
    scale = torch.dot(source.flatten(), onto.flatten()) / denom
    return source - onto * scale


def pack_update(before: torch.Tensor, after: torch.Tensor) -> dict:
    delta = (after - before).detach()
    raw = delta.mul(255).round().to(torch.int16).cpu().numpy().tobytes()
    flat = delta.detach().float()
    return {
        "l1": float(flat.abs().sum()),
        "l2": float(flat.norm()),
        "linf": float(flat.abs().max()) if flat.numel() else 0.0,
        "sha256": sha_bytes(raw),
    }


def apply_update(delta: torch.Tensor, clean: torch.Tensor, direction: torch.Tensor, optimizer) -> dict:
    before = delta.detach().clone()
    if optimizer is None:
        with torch.no_grad():
            delta.add_((-direction.sign()) * PGD_STEP)
            project(delta, clean, EPS)
    else:
        delta.grad = direction
        optimizer.step()
        project(delta, clean, EPS)
        delta.grad = None
    return pack_update(before, delta.detach())


def clean_references(model, state, item) -> dict:
    found = {}
    for ratio in TRAIN_RATIOS:
        margin, kept, generated = fast_margin(model, state, item, state["page"], ratio, None)
        found[ratio] = {"tokens": generated, "keep_hash": keep_hash(kept), "margin": float(margin.detach())}
    return found


def score_png(model, processor, item, state, image: Image.Image, fixed: dict) -> dict:
    clean = item["images"][item["aux_idx"]]
    linf_u8 = int(np.max(np.abs(np.asarray(image, dtype=np.int16) - np.asarray(clean, dtype=np.int16))))
    attacked = list(item["images"])
    attacked[item["aux_idx"]] = image
    inputs = v7d.prepare(processor, attacked, item["meta"]["question"])
    spans = v7d.spans_of(inputs, int(model.config.image_token_index), len(attacked))
    full_a, _ = v7d.generate_tokens(model, inputs, inputs["pixel_values"], spans, None)
    full_b, _ = v7d.generate_tokens(model, inputs, inputs["pixel_values"], spans, None)
    text_a = decode(processor, torch.tensor(full_a)).strip()
    text_b = decode(processor, torch.tensor(full_b)).strip()
    answers = list(item["meta"]["answers"])
    ratios = {}
    fixed_margins = []
    dynamic_margins = []
    for ratio in TRAIN_RATIOS:
        tokens, kept = v7d.generate_tokens(model, inputs, inputs["pixel_values"], spans, ratio)
        text = decode(processor, torch.tensor(tokens)).strip()
        page = v7d.page_to_float(image).to("cuda")
        fixed_value, fixed_kept, _ = fast_margin(model, state, item, page, ratio, fixed[ratio]["tokens"])
        dynamic_value, _, _ = fast_margin(model, state, item, page, ratio, None)
        fixed_margins.append(float(fixed_value.detach()))
        dynamic_margins.append(float(dynamic_value.detach()))
        anls = anls_score(text, answers)
        ratios[f"{ratio:.2f}"] = {
            "text": text,
            "tokens": tokens,
            "anls": anls,
            "valid": output_valid(text),
            "k": len(kept),
            "keep_hash": keep_hash(kept),
            "keep_source": "current_png_fastv_generate",
            "endpoint_failure": bool(output_valid(text) and anls < ANLS_CLEAN),
        }
        del fixed_kept
    return {
        "full_text": text_a,
        "full_tokens": full_a,
        "full_text_exact": int(text_a == state["clean_text"] and text_b == state["clean_text"]),
        "full_token_exact": int(full_a == state["clean_tokens"] and full_b == state["clean_tokens"]),
        "full_replay": int(full_a == full_b and text_a == text_b),
        "output_valid": int(output_valid(text_a) and all(row["valid"] for row in ratios.values())),
        "nonzero": int(not np.array_equal(np.asarray(image), np.asarray(clean))),
        "linf_uint8": linf_u8,
        "linf_ok": int(0 < linf_u8 <= 8),
        "fixed_margin": 0.5 * sum(fixed_margins),
        "dynamic_margin": 0.5 * sum(dynamic_margins),
        "ratios": ratios,
        "uint8_linf_float": uint8_linf(clean, image),
    }


def legal(row: dict) -> bool:
    return bool(row["nonzero"] and row["output_valid"] and row["full_text_exact"] and row["linf_ok"] and row["full_replay"])


def choose_fixed(rows: list[dict]) -> dict | None:
    eligible = [row for row in rows if legal(row)]
    if not eligible:
        return None
    return sorted(eligible, key=lambda row: (float(row["fixed_margin"]), int(row["step"])))[0]


def choose_historical(rows: list[dict]) -> dict | None:
    eligible = [row for row in rows if legal(row)]
    if not eligible:
        return None
    return sorted(eligible, key=lambda row: (float(row["dynamic_margin"]), int(row["step"])))[0]


def oracle_success(row: dict, clean_ok: dict) -> bool:
    if not legal(row):
        return False
    for ratio in TRAIN_RATIOS:
        key = f"{ratio:.2f}"
        if not clean_ok[key]:
            return False
        if not row["ratios"][key]["endpoint_failure"]:
            return False
    return True


def train_one(model, processor, sample_id: str, group: str, steps: int, out: Path) -> dict:
    mode, rule = GROUPS[group]
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED % (2**32))
    item = v7d.load_dev(sample_id)
    state = v7d.prepare_item(model, processor, item)
    fixed = clean_references(model, state, item)
    clean_ok = {}
    answers = list(item["meta"]["answers"])
    for ratio, row in fixed.items():
        text = decode(processor, torch.tensor(row["tokens"])).strip()
        clean_ok[f"{ratio:.2f}"] = anls_score(text, answers) >= ANLS_CLEAN
        row["text"] = text
        row["anls"] = anls_score(text, answers)
    delta = torch.zeros_like(state["page"], requires_grad=True)
    opt = torch.optim.Adam([delta], lr=ADAM_LR) if rule == "adam" else None
    out.mkdir(parents=True, exist_ok=True)
    marks = {step for step in CHECKPOINTS if step <= steps}
    marks.add(steps)
    curve = []
    updates = []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    reset_counters()
    for step in range(steps + 1):
        if step in marks:
            image = page_uint8(state["page"], delta)
            dest = out / f"step_{step:03d}.png"
            image.save(dest)
            reloaded = Image.open(dest).convert("RGB")
            scored = score_png(model, processor, item, state, reloaded, fixed)
            scored["step"] = step
            scored["png_sha256"] = hashlib.sha256(dest.read_bytes()).hexdigest()
            scored["oracle_joint_success"] = oracle_success(scored, clean_ok)
            scored["oracle_label"] = "post-hoc oracle checkpoint diagnostic, not an unlabeled selector result"
            curve.append(scored)
            print(json.dumps({"group": group, "sample": sample_id, "step": step, "full_text_exact": scored["full_text_exact"], "fixed_margin": scored["fixed_margin"]}), flush=True)
        if step == steps:
            break
        page = state["page"] + delta
        attack, full, violated, keeps = attack_grads(model, state, item, page, mode, fixed, delta)
        if rule == "adam":
            direction = attack + full
            branch = "weighted_adam"
        elif rule == "pgd":
            direction = attack + full
            branch = "weighted_pgd"
        else:
            direction = project_out(full, attack) if violated else project_out(attack, full)
            branch = "full_primary" if violated else "attack_primary"
        executed = apply_update(delta, state["page"], direction, opt if rule == "adam" else None)
        updates.append({
            "step": step,
            "attack_l2": float(attack.float().norm()),
            "full_l2": float(full.float().norm()),
            "direction_l2": float(direction.float().norm()),
            "full_hard_violated": violated,
            "branch": branch,
            "executed_update": executed,
            "keep": {f"{ratio:.2f}": {"k": keeps[ratio]["k"], "keep_hash": keeps[ratio]["keep_hash"], "source": keeps[ratio]["source"]} for ratio in TRAIN_RATIOS},
        })
        torch.cuda.empty_cache()
    sealed = choose_fixed(curve)
    historical = choose_historical(curve) if group == "H0" else None
    first = next((row["step"] for row in curve if row["oracle_joint_success"]), None)
    flips = 0
    previous = False
    for row in curve:
        now = bool(row["oracle_joint_success"])
        if previous and not now:
            flips += 1
        previous = now
    payload = {
        "sample_id": sample_id,
        "group": group,
        "mode": mode,
        "rule": rule,
        "clean_text": state["clean_text"],
        "clean_tokens": state["clean_tokens"],
        "fixed_reference": {f"{ratio:.2f}": fixed[ratio] for ratio in TRAIN_RATIOS},
        "clean_fastv_meets_threshold": clean_ok,
        "sealed_by_fixed_margin": None if sealed is None else {"step": sealed["step"], "fixed_margin": sealed["fixed_margin"], "png_sha256": sealed["png_sha256"]},
        "historical_dynamic_margin_selector": None if historical is None else {"step": historical["step"], "dynamic_margin": historical["dynamic_margin"]},
        "oracle_any_joint_success": any(row["oracle_joint_success"] for row in curve),
        "oracle_first_success_step": first,
        "observed_success_then_fail": flips,
        "curve_selector_view": [
            {key: row[key] for key in ("step", "full_text_exact", "full_token_exact", "output_valid", "linf_ok", "full_replay", "nonzero", "fixed_margin", "dynamic_margin", "oracle_joint_success", "png_sha256")}
            for row in curve
        ],
        "updates": updates,
        "counters": counters(),
        "elapsed_sec": time.perf_counter() - started,
        "peak_memory_mib": torch.cuda.max_memory_allocated() / (1024 ** 2),
        "projection_note": "投影只约束代理损失的一阶变化，不保证 Full 精确输出；Full 是否保持以真实 generate 为准。",
    }
    (out / "trajectory.json").write_text(json.dumps(payload) + "\n")
    return payload


def phase_a(model, processor) -> dict:
    checks = []
    for sample_id in ("60700", "7006"):
        item = v7d.load_dev(sample_id)
        state = v7d.prepare_item(model, processor, item)
        fixed = clean_references(model, state, item)
        dyn_again = clean_references(model, state, item)
        same_tokens = all(fixed[ratio]["tokens"] == dyn_again[ratio]["tokens"] for ratio in TRAIN_RATIOS)
        checks.append({"name": f"{sample_id}_clean_fixed_equals_dyn", "pass": same_tokens})
        for ratio in TRAIN_RATIOS:
            reference = fixed[ratio]["tokens"]
            margin, kept, generated = fast_margin(model, state, item, state["page"], ratio, reference)
            prefix_ok = generated[:-1] == reference[:-1] and len(generated) == len(reference)
            checks.append({"name": f"{sample_id}_r{ratio}_prefix_aligned_on_clean", "pass": prefix_ok, "keep_source": "current_clean_fastv_generate", "k": len(kept)})
        delta = torch.zeros_like(state["page"], requires_grad=True)
        opt = torch.optim.Adam([delta], lr=ADAM_LR)
        page = state["page"] + delta
        attack, full, violated, keeps = attack_grads(model, state, item, page, "fixed", fixed, delta)
        delta.grad = attack + full
        opt.step()
        project(delta, state["page"], EPS)
        image = page_uint8(state["page"], delta)
        dest = HERE / "phase_a" / sample_id / "one_step.png"
        dest.parent.mkdir(parents=True, exist_ok=True)
        image.save(dest)
        reloaded = Image.open(dest).convert("RGB")
        linf_u8 = int(np.max(np.abs(np.asarray(reloaded, dtype=np.int16) - np.asarray(item["images"][item["aux_idx"]], dtype=np.int16))))
        perturbed_keeps = {}
        for ratio in TRAIN_RATIOS:
            _, kept, generated = fast_margin(model, state, item, state["page"] + delta.detach(), ratio, fixed[ratio]["tokens"])
            perturbed_keeps[f"{ratio:.2f}"] = {"k": len(kept), "keep_hash": keep_hash(kept), "same_as_clean": keep_hash(kept) == fixed[ratio]["keep_hash"], "generated_len": len(generated), "reference_len": len(fixed[ratio]["tokens"])}
        aux = state["page"].detach().clone().requires_grad_(True)
        base = state["inputs"]["pixel_values"].detach()
        pages = []
        others = []
        for index in range(base.shape[0]):
            if index == item["aux_idx"]:
                pages.append(v7d.normalized_tiles(aux))
            else:
                leaf = base[index].clone().requires_grad_(True)
                others.append(leaf)
                pages.append(leaf)
        pixels = torch.stack(pages)
        generated, kept = v7d.generate_tokens(model, state["inputs"], pixels, state["spans"], 0.25)
        keep = v7d.keep_index_from_kept(kept, state["spans"], int(state["inputs"]["input_ids"].shape[1]), pixels.device)
        rows = v7d.fastv_rows(model, state["inputs"], pixels, fixed[0.25]["tokens"], keep)
        v7d.mean_soft(rows, fixed[0.25]["tokens"]).backward()
        other_l1 = [0.0 if leaf.grad is None else float(leaf.grad.abs().sum()) for leaf in others]
        checks.append({
            "name": f"{sample_id}_aux_grad_only",
            "pass": aux.grad is not None and float(aux.grad.abs().sum()) > 0 and all(value == 0.0 for value in other_l1),
            "aux_l1": None if aux.grad is None else float(aux.grad.abs().sum()),
            "other_l1": other_l1,
        })
        checks.append({"name": f"{sample_id}_png_linf", "pass": 0 < linf_u8 <= 8, "linf_uint8": linf_u8})
        checks.append({"name": f"{sample_id}_keep_from_current_input", "pass": all(row["source"] == "current_perturbed_fastv_generate" for row in keeps.values()), "perturbed": perturbed_keeps, "full_violated": violated, "attack_l2": float(attack.float().norm()), "full_l2": float(full.float().norm())})
    status = "PASS" if all(row["pass"] for row in checks) else "FAIL"
    payload = {"status": status, "checks": checks}
    (HERE / "phase_a_report.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"phase_a": status}), flush=True)
    if status != "PASS":
        raise SystemExit(1)
    return payload


def write_protocol() -> None:
    protocol = {
        "scope": "white-box development diagnostic on six questions",
        "samples": list(IDS),
        "sample_1025": "known high sensitivity; a change on 1025 alone is not evidence",
        "train_ratios": list(TRAIN_RATIOS),
        "eval_ratios": list(EVAL_RATIOS),
        "steps": STEPS,
        "checkpoints": list(CHECKPOINTS),
        "seed": SEED,
        "lambda_full": LAMBDA_FULL,
        "lambda_source": "week2_attack_v7d FROZEN_COEFFICIENTS.json native-FDP lambda_full; not retuned",
        "adam_lr": ADAM_LR,
        "pgd_step": PGD_STEP,
        "linf_uint8": 8,
        "groups": {key: {"reference": value[0], "update": value[1]} for key, value in GROUPS.items()},
        "opgd": "if any clean-token hard margin is <= 0, descend the Full gradient with its projection on the attack gradient removed; otherwise descend the attack gradient with its projection on the Full gradient removed; then sign and step 1/255",
        "selector": "legal if nonzero, valid, Full text exact, uint8 Linf<=8, Full replay; rank by fixed-reference soft margin, then earlier step",
        "historical_h0_selector": "same legal set ranked by dynamic-reference margin",
        "oracle": "post-hoc oracle checkpoint diagnostic, not an unlabeled selector result",
        "anls_failure": 0.5,
        "clean_success_threshold": 0.5,
        "projection_note": "投影只约束代理损失的一阶变化，不保证 Full 精确输出；Full 是否保持以真实 generate 为准。",
        "phase_d": "NOT_RUN",
    }
    (HERE / "FROZEN_V11_PROTOCOL.json").write_text(json.dumps(protocol, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("init", "phase_a", "cost", "train"))
    parser.add_argument("--sample")
    parser.add_argument("--group")
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()
    HERE.mkdir(parents=True, exist_ok=True)
    if args.mode == "init":
        write_protocol()
        return
    model, processor = load_model_frozen()
    install_counters(model)
    if args.mode == "phase_a":
        phase_a(model, processor)
        return
    if args.mode == "cost":
        out = HERE / "cost_probe"
        started = time.perf_counter()
        train_one(model, processor, "60700", "H0", args.steps, out)
        elapsed = time.perf_counter() - started
        (HERE / "cost_probe.json").write_text(json.dumps({"sample": "60700", "group": "H0", "steps": args.steps, "elapsed_sec": elapsed}, indent=2) + "\n")
        print(json.dumps({"cost_elapsed_sec": elapsed, "steps": args.steps}), flush=True)
        return
    if args.mode == "train":
        out = HERE / "runs" / args.group / args.sample
        train_one(model, processor, args.sample, args.group, STEPS, out)
        return


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise
