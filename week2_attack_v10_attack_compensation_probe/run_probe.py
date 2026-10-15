"""Development probe for an attack delta plus a Full-only compensation delta.

Imports the v7d native-cache and FastV margin path. Does not modify v7d.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
V7D = ROOT / "week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation"
HERE = ROOT / "week2_attack_v10_attack_compensation_probe"
sys.path[:0] = [str(ROOT), str(ROOT / "week1_mechanism"), str(ROOT / "gate_experiment"), str(HERE)]

from metrics_task import anls_score  # noqa: E402
from probe_state import Access  # noqa: E402
from projection import (  # noqa: E402
    apply_uint8,
    delta_norms,
    float_from_uint8,
    project_attack_leaf,
    project_comp_leaf,
    uint8_array,
)
from run_eval import decode  # noqa: E402
from week2_attack_v4_answer_aligned.common import keep_hash  # noqa: E402
from week2_attack_v4_answer_aligned.loss import output_valid  # noqa: E402
from week2_attack_v7_pristine60.run_v7 import load_model_frozen  # noqa: E402
from week2_attack_v7b_native_cache_margin_admission.native_cache_margin import (  # noqa: E402
    cache_replay,
    hard_margin_row,
    soft_margin_row,
)
from week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation import run_v7d as v7d  # noqa: E402
from week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation.losses import decision_preservation  # noqa: E402
from week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation.native_grad import mean_hard, mean_soft  # noqa: E402

IDS = ("1025", "2008", "47450", "60700", "7006", "39204")
SUCCESS_IDS = ("1025", "2008", "47450", "60700")
FAILURE_IDS = ("7006", "39204")
TRAIN_RATIOS = (0.25, 0.35)
EVAL_RATIOS = (0.50, 0.35, 0.25, 0.15)
CHECKPOINTS = (0, 10, 25, 50, 75, 100)
SEED = 2026101201
SCALES = (1.0, 0.5, 0.25, 0.125, 0.0625)
EPS = 8 / 255
FROZEN_1025 = [8847, 13, 356, 13, 425, 4110, 13979, 151645]
ATTACK_FIELDS = ("step", "nonzero", "output_valid", "linf_ok", "pixel_linf_uint8", "margin", "replay")
COMP_FIELDS = ("step", "nonzero", "output_valid", "linf_ok", "pixel_linf_uint8", "full_exact", "margin", "replay")


def install_counters(model) -> None:
    original_generate = model.generate
    original_backward = torch.Tensor.backward

    def generate(*args, **kwargs):
        Access.generate += 1
        return original_generate(*args, **kwargs)

    def count_forward(module, args, kwargs):
        Access.forward += 1

    def backward(self, *args, **kwargs):
        Access.backward += 1
        return original_backward(self, *args, **kwargs)

    model.generate = generate
    model.register_forward_pre_hook(count_forward, with_kwargs=True)
    torch.Tensor.backward = backward
    original = v7d.generate_tokens

    def generate_tokens(model, inputs, pixels, spans, ratio=None):
        if ratio is not None:
            Access.fastv += 1
        return original(model, inputs, pixels, spans, ratio)

    v7d.generate_tokens = generate_tokens


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha_tensor(value: torch.Tensor) -> str:
    raw = value.detach().cpu().contiguous()
    return hashlib.sha256(raw.view(torch.uint8).numpy().tobytes()).hexdigest()


def ratio_key(ratio: float) -> str:
    return f"{ratio:.2f}"


def historical_curve(sample_id: str) -> dict:
    curve = json.loads((V7D / "runs" / "NATIVE_FDP" / sample_id / "curve.json").read_text())
    return next(row for row in curve if int(row["step"]) == 0)


def write_protocol() -> None:
    parent_files = [
        V7D / "losses.py",
        V7D / "native_grad.py",
        V7D / "run_v7d.py",
        V7D / "score_selected.py",
        ROOT / "week2_attack_v7b_native_cache_margin_admission" / "native_cache_margin.py",
        ROOT / "week1_mechanism" / "run_eval.py",
    ]
    for sample_id in IDS:
        parent_files.append(V7D / "scored" / "NATIVE_FDP" / f"{sample_id}.json")
        parent_files.append(V7D / "runs" / "NATIVE_FDP" / sample_id / "selected.json")
        parent_files.append(V7D / "runs" / "NATIVE_FDP" / sample_id / "curve.json")
    parents = {str(path.relative_to(ROOT)): sha_file(path) for path in parent_files}
    (HERE / "PARENT_HASHES.json").write_text(json.dumps(parents, indent=2) + "\n")
    protocol = {
        "scope": "development constructability probe",
        "not_a_method_result": True,
        "samples": list(IDS),
        "sample_reason": {
            "historical_native_fdp_strict_success": list(SUCCESS_IDS),
            "historical_native_fdp_strict_failure": list(FAILURE_IDS),
            "selection": "known development outcomes, not a random or unseen cohort",
            "sample_1025": "r=0.25 matched random noise induced-failure rate was 28/32 in the v9c diagnostic; 1025 cannot decide the conclusion alone",
        },
        "train_ratios": list(TRAIN_RATIOS),
        "eval_ratios": list(EVAL_RATIOS),
        "linf_uint8": 8,
        "generation": {"do_sample": False, "max_new_tokens": 16, "use_cache": True},
        "seed": SEED,
        "attack_steps": 100,
        "compensation_steps": 100,
        "checkpoints": list(CHECKPOINTS),
        "adam_lr": 2 / 255,
        "constraint_tolerance": 1e-4,
        "backtracking_scales": list(SCALES),
        "attack_selector": list(ATTACK_FIELDS),
        "attack_tie_break": "smaller native L_attack, then earlier step",
        "compensation_selector": list(COMP_FIELDS),
        "compensation_tie_break": "smaller native L_attack, then earlier step",
        "selector_does_not_read": ["reference answer", "ANLS", "strict", "FastV text", "error semantics"],
        "losses": {
            "L_attack": "0.5 * (m_fastv_0.25 + m_fastv_0.35) via v7d native_grad.fastv_rows soft margins",
            "L_FDP": "v7d decision_preservation on native_rows_from_full_ids",
        },
        "projection": "integer Euclidean projection onto the component and sum L-infinity intersection; phase 2 freezes delta_attack",
        "decision_thresholds": {
            "GO_ATTACK_COMPENSATION_STRUCTURE": "at least 2/6 strong and at least 3/6 strong or partial, witnesses not all from 1025, no integrity divergence",
            "PARTIAL_ATTACK_COMPENSATION_STRUCTURE": "fewer than 2/6 strong, at least 2/6 strong or partial, no integrity divergence",
            "NO_GO_ATTACK_COMPENSATION_STRUCTURE": "fewer than 2/6 strong or partial, or compensation usually repairs FastV as well as Full, or compensation-only is the main attack, or every witness is 1025",
            "BASELINE_OR_REPLAY_DIVERGED": "any baseline, prompt, token, K, keep-set, or disk-replay mismatch",
        },
        "native_fdp_comparisons": "NOT_RUN",
        "dynamic_role_methods": "NOT_RUN",
    }
    (HERE / "FROZEN_PROTOCOL.json").write_text(json.dumps(protocol, indent=2) + "\n")


def counters() -> dict:
    return {
        "forward": Access.forward,
        "backward": Access.backward,
        "generate": Access.generate,
        "fastv": Access.fastv,
    }


def prepare_state(model, processor, item) -> dict:
    return v7d.prepare_item(model, processor, item)


def page_of(state, attack: torch.Tensor, comp: torch.Tensor | None = None) -> torch.Tensor:
    page = state["page"] + attack
    if comp is not None:
        page = page + comp
    return page


def loss_attack_value(model, state, item, page: torch.Tensor) -> float:
    values = []
    for ratio in TRAIN_RATIOS:
        values.append(v7d.soft_fast(model, state["inputs"], state["spans"], page, item["aux_idx"], ratio))
    return float((0.5 * (values[0] + values[1])).detach())


def loss_fdp_value(model, state, item, page: torch.Tensor) -> torch.Tensor:
    rows = v7d.full_token_rows(model, state["inputs"], page, item["aux_idx"], state["clean_tokens"], False)
    hard = torch.stack([v7d.hard_margin_row(row, token) for row, token in zip(rows, state["clean_tokens"])])
    return decision_preservation(hard, state["clean_hard"])


def png_from_deltas(clean_image: Image.Image, attack_u8: np.ndarray, comp_u8: np.ndarray | None = None) -> Image.Image:
    clean_u8 = np.asarray(clean_image.convert("RGB"), dtype=np.uint8)
    total = attack_u8 if comp_u8 is None else attack_u8.astype(np.int16) + comp_u8.astype(np.int16)
    return apply_uint8(clean_u8, total)


def leaf_uint8(delta: torch.Tensor) -> np.ndarray:
    array = delta.detach().mul(255).round().permute(1, 2, 0).cpu().numpy()
    return array.astype(np.int16)


def component_margin(model, state, item, page: torch.Tensor) -> float:
    total = 0.0
    for ratio in TRAIN_RATIOS:
        tokens, kept = v7d.generate_tokens(model, state["inputs"], v7d.pixels_from_page(state["inputs"], item["aux_idx"], page.detach()), state["spans"], ratio)
        keep = v7d.keep_index_from_kept(kept, state["spans"], int(state["inputs"]["input_ids"].shape[1]), page.device)
        with torch.no_grad():
            rows = v7d.fastv_rows(model, state["inputs"], v7d.pixels_from_page(state["inputs"], item["aux_idx"], page.detach()), tokens, keep)
        total += float(mean_soft(rows, tokens).detach())
    return 0.5 * total


def selector_from_png(model, processor, item, state, image: Image.Image, kind: str) -> dict:
    clean_u8 = np.asarray(item["images"][item["aux_idx"]].convert("RGB"))
    got = np.asarray(image.convert("RGB"))
    diff = got.astype(np.int16) - clean_u8.astype(np.int16)
    linf = int(np.max(np.abs(diff)))
    attacked = list(item["images"])
    attacked[item["aux_idx"]] = image
    inputs = v7d.prepare(processor, attacked, item["meta"]["question"])
    spans = v7d.spans_of(inputs, int(model.config.image_token_index), len(attacked))
    full_a, _ = v7d.generate_tokens(model, inputs, inputs["pixel_values"], spans, None)
    full_b, _ = v7d.generate_tokens(model, inputs, inputs["pixel_values"], spans, None)
    text_a = decode(processor, torch.tensor(full_a)).strip()
    text_b = decode(processor, torch.tensor(full_b)).strip()
    replay = full_a == full_b and text_a == text_b
    valid = output_valid(text_a) and output_valid(text_b)
    page = float_from_uint8(image).to("cuda")
    margin = component_margin(model, state, item, page)
    row = {
        "nonzero": int(not np.array_equal(got, clean_u8)),
        "output_valid": int(valid),
        "linf_ok": int(0 < linf <= 8),
        "pixel_linf_uint8": linf / 255.0,
        "margin": margin,
        "replay": int(replay),
    }
    if kind == "comp":
        row["full_exact"] = int(text_a == state["clean_text"] and text_b == state["clean_text"])
    return row


def choose_attack(rows: list[dict]) -> dict | None:
    eligible = [row for row in rows if row["nonzero"] and row["output_valid"] and row["linf_ok"] and row["replay"]]
    if not eligible:
        return None
    return sorted(eligible, key=lambda row: (float(row["margin"]), int(row["step"])))[0]


def choose_comp(rows: list[dict]) -> dict | None:
    eligible = [
        row for row in rows
        if row["nonzero"] and row["output_valid"] and row["linf_ok"] and row["replay"] and row["full_exact"]
    ]
    if not eligible:
        return None
    return sorted(eligible, key=lambda row: (float(row["margin"]), int(row["step"])))[0]


def save_checkpoint(model, processor, item, state, clean_image, attack_u8, comp_u8, step, kind, folder: Path) -> dict:
    image = png_from_deltas(clean_image, attack_u8, comp_u8)
    dest = folder / f"step_{step:03d}.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    image.save(dest)
    reloaded = Image.open(dest).convert("RGB")
    fields = selector_from_png(model, processor, item, state, reloaded, kind)
    fields["step"] = step
    allowed = ATTACK_FIELDS if kind == "attack" else COMP_FIELDS
    return {"selector": {key: fields[key] for key in allowed}, "png": str(dest), "png_sha256": sha_file(dest)}


def run_attack(model, processor, item, state, steps: int, folder: Path) -> dict:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED % (2**32))
    delta = torch.zeros_like(state["page"], requires_grad=True)
    opt = torch.optim.Adam([delta], lr=2.0 / 255.0)
    clean_image = item["images"][item["aux_idx"]]
    marks = {step for step in CHECKPOINTS if step <= steps}
    marks.add(steps)
    curve = []
    for step in range(steps + 1):
        if step in marks:
            attack_u8 = leaf_uint8(delta)
            curve.append(save_checkpoint(model, processor, item, state, clean_image, attack_u8, None, step, "attack", folder / "attack_checkpoints"))
            print(json.dumps({"phase": "attack", "sample": item["meta"]["sample_id"], "step": step, "margin": curve[-1]["selector"]["margin"]}), flush=True)
        if step == steps:
            break
        opt.zero_grad(set_to_none=True)
        for ratio in TRAIN_RATIOS:
            page = page_of(state, delta)
            value = v7d.soft_fast(model, state["inputs"], state["spans"], page, item["aux_idx"], ratio)
            (0.5 * value).backward()
        if delta.grad is None or not torch.isfinite(delta.grad).all() or float(delta.grad.float().norm()) == 0.0:
            raise RuntimeError(f"attack gradient failed at step {step}")
        opt.step()
        project_attack_leaf(delta, state["page"])
        torch.cuda.empty_cache()
    chosen = choose_attack([row["selector"] for row in curve])
    return {"curve": curve, "chosen": chosen, "final_delta": delta.detach()}


def run_comp(model, processor, item, state, attack: torch.Tensor, attack_ref: float, steps: int, folder: Path) -> dict:
    delta = torch.zeros_like(state["page"], requires_grad=True)
    opt = torch.optim.Adam([delta], lr=2.0 / 255.0)
    clean_image = item["images"][item["aux_idx"]]
    attack_u8 = leaf_uint8(attack)
    marks = {step for step in CHECKPOINTS if step <= steps}
    marks.add(steps)
    curve = []
    accepted = 0
    rejected = 0
    for step in range(steps + 1):
        if step in marks:
            curve.append(save_checkpoint(model, processor, item, state, clean_image, attack_u8, leaf_uint8(delta), step, "comp", folder / "comp_checkpoints"))
            print(json.dumps({"phase": "comp", "sample": item["meta"]["sample_id"], "step": step, "accepted": accepted, "rejected": rejected}), flush=True)
        if step == steps:
            break
        before = delta.detach().clone()
        current = loss_fdp_value(model, state, item, page_of(state, attack.detach(), delta))
        current_value = float(current.detach())
        opt.zero_grad(set_to_none=True)
        current.backward()
        if delta.grad is None or not torch.isfinite(delta.grad).all():
            raise RuntimeError(f"compensation gradient failed at step {step}")
        opt.step()
        proposal = delta.detach() - before
        took = False
        for scale in SCALES:
            with torch.no_grad():
                delta.copy_(before + proposal * scale)
            project_comp_leaf(delta, state["page"], attack.detach())
            new_fdp = float(loss_fdp_value(model, state, item, page_of(state, attack.detach(), delta)).detach())
            new_attack = component_margin(model, state, item, page_of(state, attack.detach(), delta.detach()))
            if new_fdp <= current_value + 1e-8 and new_attack <= attack_ref + 1e-4:
                took = True
                break
        if not took:
            with torch.no_grad():
                delta.copy_(before)
            rejected += 1
        else:
            accepted += 1
        torch.cuda.empty_cache()
    chosen = choose_comp([row["selector"] for row in curve])
    return {"curve": curve, "chosen": chosen, "accepted": accepted, "rejected": rejected, "final_delta": delta.detach()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("init", "baseline", "smoke", "full"))
    parser.add_argument("--sample")
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()
    HERE.mkdir(parents=True, exist_ok=True)
    if args.mode == "init":
        write_protocol()
        return
    if args.mode == "baseline":
        raise SystemExit("baseline entry is baseline.py")
    model, processor = load_model_frozen()
    install_counters(model)
    item = v7d.load_dev(args.sample)
    state = prepare_state(model, processor, item)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    if args.mode == "smoke":
        folder = HERE / "smoke" / f"steps_{args.steps}"
        attack = run_attack(model, processor, item, state, args.steps, folder)
        if attack["chosen"] is None and args.steps >= 2:
            attack_delta = attack["final_delta"]
        else:
            attack_delta = attack["final_delta"]
        ref = component_margin(model, state, item, page_of(state, attack_delta))
        comp = run_comp(model, processor, item, state, attack_delta, ref, args.steps, folder)
        report = {
            "sample": args.sample,
            "steps": args.steps,
            "attack_chosen": attack["chosen"],
            "comp_accepted": comp["accepted"],
            "comp_rejected": comp["rejected"],
            "counters": counters(),
            "elapsed_sec": time.perf_counter() - started,
            "peak_memory_mib": torch.cuda.max_memory_allocated() / (1024 ** 2),
            "scientific_evidence": False,
        }
        (folder / "smoke_report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"smoke": args.steps, "accepted": comp["accepted"], "rejected": comp["rejected"]}), flush=True)
        return
    if args.mode == "full":
        from endpoints import run_full
        run_full(model, processor, args.sample)
        return
    raise RuntimeError(args.mode)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
