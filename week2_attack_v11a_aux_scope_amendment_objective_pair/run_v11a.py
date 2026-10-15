"""Correct the auxiliary-page scope gate, then compare H0 and H1 only.

H0 calls the historical native-FDP attack unchanged. H1 uses that same
loop and changes only the FastV reference token sequence.
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
HERE = ROOT / "week2_attack_v11a_aux_scope_amendment_objective_pair"
PARENT = ROOT / "week2_attack_v11_objective_optimizer_diagnostic"
HIST = ROOT / "week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation" / "runs" / "NATIVE_FDP"
import sys

sys.path[:0] = [str(ROOT), str(ROOT / "week1_mechanism"), str(ROOT / "gate_experiment"), str(HERE)]

from metrics_task import anls_score  # noqa: E402
from run_eval import decode  # noqa: E402
from week2_attack_v4_answer_aligned.common import keep_hash, sha_image  # noqa: E402
from week2_attack_v7_pristine60.run_v7 import load_model_frozen  # noqa: E402
from week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation import run_v7d as v7d  # noqa: E402

IDS = ("1025", "2008", "47450", "60700", "7006", "39204")
GATE_IDS = ("60700", "7006")
CHECKPOINTS = (0, 10, 25, 50, 75, 100, 150, 200)
STEPS = 200
SEED = v7d.SEED
LAMBDA_FULL = 2.6371882367154997
ADAM_LR = 2.0 / 255.0
EPS = 8 / 255
COEFFS = {
    "lambda_full": LAMBDA_FULL,
    "lambda_gap": 0.913170951688196,
    "xi": 1.0,
    "eta": 0.001,
    "source": "+1/255",
}


class Access:
    forward = 0
    backward = 0
    generate = 0
    fastv = 0
    generated_tokens = 0
    keep_events = []


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
        if ratio is not None and kept is not None:
            Access.keep_events.append({
                "ratio": float(ratio),
                "keep_hash": keep_hash(kept),
                "pixels_sha256": tensor_sha(pixels.detach()),
            })
        return tokens, kept

    v7d.generate_tokens = generate_tokens


def reset_counters() -> None:
    Access.forward = Access.backward = Access.generate = Access.fastv = Access.generated_tokens = 0
    Access.keep_events = []


def counters() -> dict:
    return {
        "forward": Access.forward,
        "backward": Access.backward,
        "generate": Access.generate,
        "fastv": Access.fastv,
        "generated_tokens": Access.generated_tokens,
    }


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha(value: torch.Tensor) -> str:
    raw = value.detach().float().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def attacked_images(item: dict, aux_image: Image.Image) -> list:
    images = list(item["images"])
    images[item["aux_idx"]] = aux_image
    return images


def soft_fast_fixed(model, state, item, page, ratio: float, reference: list[int]) -> torch.Tensor:
    pixels = v7d.pixels_from_page(state["inputs"], item["aux_idx"], page)
    _generated, kept = v7d.generate_tokens(model, state["inputs"], pixels, state["spans"], ratio)
    keep = v7d.keep_index_from_kept(kept, state["spans"], int(state["inputs"]["input_ids"].shape[1]), page.device)
    rows = v7d.fastv_rows(model, state["inputs"], pixels, reference, keep)
    if len(rows) != len(reference):
        raise RuntimeError("fixed teacher-forced length left the saved reference")
    return v7d.mean_soft(rows, reference)


def capture_clean_fastv(model, state, item) -> dict:
    found = {}
    for ratio in v7d.TRAIN_RATIOS:
        pixels = v7d.pixels_from_page(state["inputs"], item["aux_idx"], state["page"])
        tokens, kept = v7d.generate_tokens(model, state["inputs"], pixels, state["spans"], ratio)
        found[f"{ratio:.2f}"] = {"tokens": tokens, "keep_hash": keep_hash(kept), "source": "clean_fastv_generate"}
    return found


def one_update(model, state, item, mode: str, fixed: dict):
    delta = torch.zeros_like(state["page"], requires_grad=True)
    opt = torch.optim.Adam([delta], lr=ADAM_LR)
    before = v7d.pixels_from_page(state["inputs"], item["aux_idx"], state["page"] + delta).detach().clone()
    page = state["page"] + delta
    error = None
    try:
        if mode == "H0":
            v7d.backward_method(model, state, item, page, "native-FDP", COEFFS, delta)
        else:
            for ratio in v7d.TRAIN_RATIOS:
                reference = fixed[f"{ratio:.2f}"]["tokens"]
                v7d.backward_loss(soft_fast_fixed(model, state, item, page, ratio, reference) * 0.5, delta, None)
            v7d.fdp_grad(model, state, item, page, LAMBDA_FULL, None, delta)
    except RuntimeError as exc:
        error = str(exc)
    grad_ok = error is None and delta.grad is not None and bool(torch.isfinite(delta.grad).all()) and float(delta.grad.abs().sum()) > 0.0
    if grad_ok:
        opt.step()
        v7d.project(delta, state["page"], EPS)
    after = v7d.pixels_from_page(state["inputs"], item["aux_idx"], state["page"] + delta.detach()).detach().clone()
    return delta, opt, before, after, grad_ok, error


def optimizer_scope(model, delta: torch.Tensor, opt: torch.optim.Optimizer, state: dict) -> dict:
    params = [param for group in opt.param_groups for param in group["params"]]
    model_ids = {id(param) for param in model.parameters()}
    pixel_ptr = state["inputs"]["pixel_values"].untyped_storage().data_ptr()
    return {
        "param_count": len(params),
        "only_delta": len(params) == 1 and params[0] is delta,
        "includes_model_parameter": any(id(param) in model_ids for param in params),
        "delta_shape": list(delta.shape),
        "page_shape": list(state["page"].shape),
        "pixel_values_shape": list(state["inputs"]["pixel_values"].shape),
        "shape_is_single_aux_page": list(delta.shape) == list(state["page"].shape) and delta.ndim == state["page"].ndim and int(np.prod(delta.shape)) != int(np.prod(state["inputs"]["pixel_values"].shape)),
        "delta_shares_pixel_storage": delta.untyped_storage().data_ptr() == pixel_ptr,
        "model_requires_grad": any(bool(param.requires_grad) for param in model.parameters()),
    }


def page_diffs(before: torch.Tensor, after: torch.Tensor, aux_idx: int) -> list[dict]:
    rows = []
    for index in range(before.shape[0]):
        left = before[index]
        right = after[index]
        rows.append({
            "page_index": index,
            "is_aux": index == aux_idx,
            "max_abs_diff": float((left - right).abs().max()),
            "torch_equal": bool(torch.equal(left, right)),
            "sha256": tensor_sha(right),
            "sha256_before": tensor_sha(left),
        })
    return rows


def context_gradient(model, state, item, reference: list[int]) -> dict:
    base = state["inputs"]["pixel_values"].detach()
    leaves = []
    pages = []
    for index in range(base.shape[0]):
        if index == item["aux_idx"]:
            pages.append(base[index].detach())
        else:
            leaf = base[index].detach().clone().requires_grad_(True)
            leaves.append((index, leaf))
            pages.append(leaf)
    pixels = torch.stack(pages)
    _tokens, kept = v7d.generate_tokens(model, state["inputs"], pixels, state["spans"], 0.25)
    keep = v7d.keep_index_from_kept(kept, state["spans"], int(state["inputs"]["input_ids"].shape[1]), pixels.device)
    rows = v7d.fastv_rows(model, state["inputs"], pixels, reference, keep)
    v7d.mean_soft(rows, reference).backward()
    return {
        "label": "context_gradient_observed",
        "used_for_pass_fail": False,
        "coordinate_space": "processor pixel_values; not comparable to normalized_tiles(aux)",
        "pages": [
            {"page_index": index, "l2": 0.0 if leaf.grad is None else float(leaf.grad.float().norm())}
            for index, leaf in leaves
        ],
    }


def phase_a(model, processor) -> dict:
    groups = []
    for sample_id in GATE_IDS:
        item = v7d.load_dev(sample_id)
        state = v7d.prepare_item(model, processor, item)
        fixed = capture_clean_fastv(model, state, item)
        dynamic = capture_clean_fastv(model, state, item)
        aligned = all(fixed[key]["tokens"] == dynamic[key]["tokens"] for key in fixed)
        prefix_ok = True
        for ratio in v7d.TRAIN_RATIOS:
            key = f"{ratio:.2f}"
            reference = fixed[key]["tokens"]
            pixels = v7d.pixels_from_page(state["inputs"], item["aux_idx"], state["page"])
            generated, kept = v7d.generate_tokens(model, state["inputs"], pixels, state["spans"], ratio)
            keep = v7d.keep_index_from_kept(kept, state["spans"], int(state["inputs"]["input_ids"].shape[1]), pixels.device)
            rows = v7d.fastv_rows(model, state["inputs"], pixels, reference, keep)
            prefix_ok = prefix_ok and generated == reference and len(rows) == len(reference)
        original_hashes = [sha_image(image) for image in item["images"]]
        question = item["meta"]["question"]
        prompt_ids = state["inputs"]["input_ids"].detach().cpu().tolist()
        for mode in ("H0", "H1"):
            Access.keep_events = []
            delta, opt, before_pixels, after_pixels, grad_ok, grad_error = one_update(model, state, item, mode, fixed)
            update_events = list(Access.keep_events)
            scope = optimizer_scope(model, delta, opt, state)
            diffs = page_diffs(before_pixels, after_pixels, item["aux_idx"])
            non_aux = [row for row in diffs if not row["is_aux"]]
            aux_row = next(row for row in diffs if row["is_aux"])
            image = v7d.page_uint8(state["page"], delta.detach())
            dest = HERE / "phase_a" / sample_id / f"{mode}.png"
            dest.parent.mkdir(parents=True, exist_ok=True)
            image.save(dest)
            reloaded = Image.open(dest).convert("RGB")
            linf = int(np.max(np.abs(np.asarray(reloaded, dtype=np.int16) - np.asarray(item["images"][item["aux_idx"]], dtype=np.int16))))
            attacked = attacked_images(item, reloaded)
            attacked_hashes = [sha_image(page) for page in attacked]
            replaced_only_aux = attacked_hashes[item["aux_idx"]] != original_hashes[item["aux_idx"]] and all(
                attacked_hashes[index] == original_hashes[index] for index in range(len(original_hashes)) if index != item["aux_idx"]
            )
            rebuilt = v7d.prepare(processor, attacked, question)
            perturbed_sha = tensor_sha(after_pixels)
            clean_sha = tensor_sha(before_pixels)
            Access.keep_events = []
            probe = (state["page"] + delta.detach()).detach().requires_grad_(True)
            probe_error = None
            try:
                for ratio in v7d.TRAIN_RATIOS:
                    if mode == "H0":
                        probed = v7d.soft_fast(model, state["inputs"], state["spans"], probe, item["aux_idx"], ratio)
                    else:
                        probed = soft_fast_fixed(model, state, item, probe, ratio, fixed[f"{ratio:.2f}"]["tokens"])
                    del probed
            except RuntimeError as exc:
                probe_error = str(exc)
            probe_events = list(Access.keep_events)
            torch.cuda.empty_cache()
            current_keep = {}
            keep_from_current = True
            for ratio in v7d.TRAIN_RATIOS:
                _tokens, kept = v7d.generate_tokens(model, rebuilt, after_pixels, state["spans"], ratio)
                used = [event for event in probe_events if event["ratio"] == float(ratio)]
                current_keep[f"{ratio:.2f}"] = {
                    "replay_keep_hash": keep_hash(kept),
                    "loss_keep_hash": None if not used else used[-1]["keep_hash"],
                    "loss_pixels_sha256": None if not used else used[-1]["pixels_sha256"],
                    "clean_keep_hash": fixed[f"{ratio:.2f}"]["keep_hash"],
                    "source": "current_perturbed_fastv_generate",
                    "loss_used_perturbed_pixels": bool(used) and used[-1]["pixels_sha256"] == perturbed_sha and used[-1]["pixels_sha256"] != clean_sha,
                    "loss_keep_matches_replay": bool(used) and used[-1]["keep_hash"] == keep_hash(kept),
                }
                keep_from_current = keep_from_current and current_keep[f"{ratio:.2f}"]["loss_used_perturbed_pixels"] and current_keep[f"{ratio:.2f}"]["loss_keep_matches_replay"]
            observed = context_gradient(model, state, item, fixed["0.25"]["tokens"])
            non_aux_unchanged = all(row["torch_equal"] and row["max_abs_diff"] == 0.0 and row["sha256"] == row["sha256_before"] for row in non_aux)
            checks = {
                "optimizer_only_delta": scope["only_delta"] and not scope["includes_model_parameter"] and not scope["model_requires_grad"] and not scope["delta_shares_pixel_storage"],
                "delta_shape_is_aux_page": scope["shape_is_single_aux_page"],
                "non_aux_pixel_update_zero": non_aux_unchanged,
                "aux_pixel_update_nonzero": aux_row["max_abs_diff"] > 0.0 and not aux_row["torch_equal"],
                "image_list_replaces_only_aux": replaced_only_aux,
                "delta_finite_nonzero": grad_ok,
                "only_aux_png_saved": dest.is_file(),
                "png_uint8_bound": 0 < linf <= 8,
                "clean_fixed_equals_dynamic": aligned,
                "teacher_forcing_prefix_aligned": prefix_ok,
                "keep_from_current_generate": keep_from_current,
                "prompt_and_order_unchanged": rebuilt["input_ids"].detach().cpu().tolist() == prompt_ids and len(attacked) == len(item["images"]),
            }
            groups.append({
                "sample_id": sample_id,
                "mode": mode,
                "aux_idx": item["aux_idx"],
                "page_count": len(item["images"]),
                "checks": checks,
                "pass": all(checks.values()),
                "scope": scope,
                "page_diffs": diffs,
                "original_page_sha256": original_hashes,
                "attacked_page_sha256": attacked_hashes,
                "png_sha256": file_sha(dest),
                "uint8_linf": linf,
                "question": question,
                "current_keep": current_keep,
                "gradient_error": grad_error,
                "probe_error": probe_error,
                "pre_step_keep_events": update_events,
                "context_gradient_observed": observed,
            })
            with torch.no_grad():
                delta.zero_()
    status = "PASS" if all(row["pass"] for row in groups) else "BLOCKED_AUX_SCOPE_GATE_FAILED"
    payload = {"status": status, "groups": groups, "gate": "actual aux update support; context gradients are recorded and are not pass/fail"}
    (HERE / "phase_a_amendment.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"phase_a": status}), flush=True)
    return payload


def attack_h1(model, processor, item, out: Path) -> dict:
    preview = v7d.prepare_item(model, processor, item)
    fixed = capture_clean_fastv(model, preview, item)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED % (2**32))
    state = v7d.prepare_item(model, processor, item)
    delta = torch.zeros_like(state["page"], requires_grad=True)
    opt = torch.optim.Adam([delta], lr=ADAM_LR)
    out.mkdir(parents=True, exist_ok=True)
    item["images"][item["aux_idx"]].save(out / "clean.png")
    marks = {step for step in CHECKPOINTS if step <= STEPS}
    curve = []
    selector_rows = []
    for step in range(STEPS + 1):
        if step in marks:
            image = v7d.page_uint8(state["page"], delta)
            dest = out / "checkpoints" / f"step_{step:03d}.png"
            dest.parent.mkdir(parents=True, exist_ok=True)
            image.save(dest)
            reloaded = Image.open(dest).convert("RGB")
            scored = v7d.score_png(model, processor, item, reloaded, state["clean_text"])
            fields = dict(scored["selector_fields"])
            fields["step"] = step
            fields["linf_ok"] = int(step > 0 and fields["nonzero"] and 0 < fields["pixel_linf_uint8"] <= EPS + 1e-6)
            selector_rows.append({key: fields[key] for key in v7d.LEGAL})
            curve.append({
                "step": step,
                "png_sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
                "texts": scored["texts"],
                "margins": scored["margins"],
                "keeps": scored["keeps"],
                "selector": selector_rows[-1],
            })
            print(json.dumps({"sample": item["meta"]["sample_id"], "method": "H1", "step": step, "full_exact": fields["full_exact"]}), flush=True)
        if step == STEPS:
            break
        opt.zero_grad(set_to_none=True)
        page = state["page"] + delta
        for ratio in v7d.TRAIN_RATIOS:
            reference = fixed[f"{ratio:.2f}"]["tokens"]
            v7d.backward_loss(soft_fast_fixed(model, state, item, page, ratio, reference) * 0.5, delta, None)
        v7d.fdp_grad(model, state, item, page, LAMBDA_FULL, None, delta)
        if delta.grad is None or not torch.isfinite(delta.grad).all():
            raise RuntimeError(f"non-finite gradient at step {step}")
        opt.step()
        v7d.project(delta, state["page"], EPS)
        torch.cuda.empty_cache()
    chosen = v7d.choose(selector_rows)
    (out / "curve.json").write_text(json.dumps(curve, indent=2) + "\n")
    (out / "selected.json").write_text(json.dumps({"chosen": chosen, "clean_text": state["clean_text"]}, indent=2) + "\n")
    (out / "fixed_reference.json").write_text(json.dumps(fixed) + "\n")
    return {"chosen": chosen, "clean_text": state["clean_text"]}


def train(model, processor, sample_id: str, group: str) -> None:
    item = v7d.load_dev(sample_id)
    out = HERE / "runs" / group / sample_id
    reset_counters()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    if group == "H0":
        result = v7d.attack(model, processor, item, "native-FDP", STEPS, COEFFS, out)
    elif group == "H1":
        result = attack_h1(model, processor, item, out)
    else:
        raise RuntimeError(group)
    meta = {
        "sample_id": sample_id,
        "group": group,
        "aux_idx": item["aux_idx"],
        "question": item["meta"]["question"],
        "page_sha256": [sha_image(image) for image in item["images"]],
        "counters": counters(),
        "elapsed_sec": time.perf_counter() - started,
        "peak_memory_mib": torch.cuda.max_memory_allocated() / (1024 ** 2),
        "historical_selector": result["chosen"],
        "clean_text": result["clean_text"],
    }
    (out / "train_meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def compare_h0() -> dict:
    rows = []
    blocked = False
    for sample_id in IDS:
        current = json.loads((HERE / "runs" / "H0" / sample_id / "curve.json").read_text())
        historical = json.loads((HIST / sample_id / "curve.json").read_text())
        hist_selected = json.loads((HIST / sample_id / "selected.json").read_text())["chosen"]
        cur_selected = json.loads((HERE / "runs" / "H0" / sample_id / "selected.json").read_text())["chosen"]
        mismatches = []
        by_step = {row["step"]: row for row in current}
        for row in historical:
            got = by_step.get(row["step"])
            if got is None:
                mismatches.append({"step": row["step"], "field": "missing"})
                continue
            for field, left, right in (
                ("png_sha256", row["png_sha256"], got["png_sha256"]),
                ("full", row["texts"]["full"], got["texts"]["full"]),
                ("fastv_0.25", row["texts"]["fastv_0.25"], got["texts"]["fastv_0.25"]),
                ("fastv_0.35", row["texts"]["fastv_0.35"], got["texts"]["fastv_0.35"]),
                ("keep_0.25", row["keeps"]["0.25"]["keep_hash"], got["keeps"]["0.25"]["keep_hash"]),
                ("keep_0.35", row["keeps"]["0.35"]["keep_hash"], got["keeps"]["0.35"]["keep_hash"]),
            ):
                if left != right:
                    mismatches.append({"step": row["step"], "field": field, "historical": left, "h0": right})
            if abs(float(row["selector"]["margin"]) - float(got["selector"]["margin"])) > 1e-5:
                mismatches.append({"step": row["step"], "field": "margin", "historical": row["selector"]["margin"], "h0": got["selector"]["margin"]})
        chosen_match = (hist_selected is None and cur_selected is None) or (
            hist_selected is not None and cur_selected is not None and int(hist_selected["step"]) == int(cur_selected["step"])
        )
        if not chosen_match:
            mismatches.append({"field": "historical_selector_step", "historical": None if hist_selected is None else hist_selected["step"], "h0": None if cur_selected is None else cur_selected["step"]})
        earliest = min((row["step"] for row in mismatches if "step" in row), default=None)
        blocked = blocked or bool(mismatches)
        meta = json.loads((HERE / "runs" / "H0" / sample_id / "train_meta.json").read_text())
        rows.append({
            "sample_id": sample_id,
            "match": not mismatches,
            "earliest_mismatch_step": earliest,
            "mismatches": mismatches[:12],
            "question": meta["question"],
            "page_sha256": meta["page_sha256"],
        })
    payload = {"status": "BLOCKED_H0_REPLAY_DIVERGED" if blocked else "PASS", "samples": rows}
    (HERE / "h0_replay.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"h0_replay": payload["status"]}), flush=True)
    return payload


def fixed_margin_on_png(model, processor, item, image: Image.Image, reference: dict) -> dict:
    attacked = attacked_images(item, image)
    inputs = v7d.prepare(processor, attacked, item["meta"]["question"])
    spans = v7d.spans_of(inputs, int(model.config.image_token_index), len(attacked))
    page = v7d.page_to_float(image).to("cuda")
    state = {"inputs": inputs, "spans": spans, "page": page}
    margins = {}
    keeps = {}
    for ratio in v7d.TRAIN_RATIOS:
        key = f"{ratio:.2f}"
        pixels = v7d.pixels_from_page(inputs, item["aux_idx"], page)
        _tokens, kept = v7d.generate_tokens(model, inputs, pixels, spans, ratio)
        keep = v7d.keep_index_from_kept(kept, spans, int(inputs["input_ids"].shape[1]), page.device)
        rows = v7d.fastv_rows(model, inputs, pixels, reference[key]["tokens"], keep)
        margins[key] = float(v7d.mean_soft(rows, reference[key]["tokens"]).detach())
        keeps[key] = keep_hash(kept)
    full_a, _ = v7d.generate_tokens(model, inputs, inputs["pixel_values"], spans, None)
    full_b, _ = v7d.generate_tokens(model, inputs, inputs["pixel_values"], spans, None)
    return {
        "fixed_margin": 0.5 * (margins["0.25"] + margins["0.35"]),
        "fixed_margins": margins,
        "current_keep_hash": keeps,
        "full_tokens_a": full_a,
        "full_token_replay": full_a == full_b,
        "full_text_replay": decode(processor, torch.tensor(full_a)).strip() == decode(processor, torch.tensor(full_b)).strip(),
    }


def reference_for(model, processor, sample_id: str) -> dict:
    saved = HERE / "runs" / "H1" / sample_id / "fixed_reference.json"
    if saved.exists():
        return json.loads(saved.read_text())
    item = v7d.load_dev(sample_id)
    state = v7d.prepare_item(model, processor, item)
    return capture_clean_fastv(model, state, item)


def score_group(model, processor, group: str) -> None:
    for sample_id in IDS:
        item = v7d.load_dev(sample_id)
        reference = reference_for(model, processor, sample_id)
        curve = json.loads((HERE / "runs" / group / sample_id / "curve.json").read_text())
        answers = list(item["meta"]["answers"])
        clean_anls = {}
        step0 = next(row for row in curve if row["step"] == 0)
        for key in ("fastv_0.25", "fastv_0.35"):
            clean_anls[key] = anls_score(step0["texts"][key], answers)
        rows = []
        clean_tokens = None
        for row in curve:
            image = Image.open(HERE / "runs" / group / sample_id / "checkpoints" / f"step_{row['step']:03d}.png").convert("RGB")
            measured = fixed_margin_on_png(model, processor, item, image, reference)
            if clean_tokens is None:
                clean_tokens = measured["full_tokens_a"]
            clean = item["images"][item["aux_idx"]]
            linf = int(np.max(np.abs(np.asarray(image, dtype=np.int16) - np.asarray(clean, dtype=np.int16))))
            anls = {
                "0.25": anls_score(row["texts"]["fastv_0.25"], answers),
                "0.35": anls_score(row["texts"]["fastv_0.35"], answers),
            }
            legal = bool(
                row["step"] > 0
                and row["selector"]["nonzero"]
                and row["selector"]["output_valid"]
                and row["texts"]["full"] == step0["texts"]["full"]
                and 0 < linf <= 8
                and measured["full_token_replay"]
                and measured["full_text_replay"]
                and measured["full_text_replay"]
            )
            strict = bool(legal and anls["0.25"] < 0.5 and anls["0.35"] < 0.5)
            rows.append({
                "step": row["step"],
                "png_sha256": row["png_sha256"],
                "full_text": row["texts"]["full"],
                "full_text_exact": int(row["texts"]["full"] == step0["texts"]["full"]),
                "full_token_exact": int(measured["full_tokens_a"] == clean_tokens and measured["full_token_replay"]),
                "fastv_0.25": row["texts"]["fastv_0.25"],
                "fastv_0.35": row["texts"]["fastv_0.35"],
                "anls_0.25": anls["0.25"],
                "anls_0.35": anls["0.35"],
                "keep_0.25": row["keeps"]["0.25"]["keep_hash"],
                "keep_0.35": row["keeps"]["0.35"]["keep_hash"],
                "dynamic_margin": row["selector"]["margin"],
                "fixed_margin": measured["fixed_margin"],
                "uint8_linf": linf,
                "output_valid": row["selector"]["output_valid"],
                "legal": legal,
                "strict": strict,
                "oracle_label": "post-hoc oracle checkpoint diagnostic, not an unlabeled selector result",
            })
            print(json.dumps({"score": group, "sample": sample_id, "step": row["step"], "strict": strict}), flush=True)
        legal_rows = [row for row in rows if row["legal"]]
        sealed = None if not legal_rows else sorted(legal_rows, key=lambda row: (float(row["fixed_margin"]), int(row["step"])))[0]
        historical = json.loads((HERE / "runs" / group / sample_id / "selected.json").read_text())["chosen"]
        first = next((row["step"] for row in rows if row["strict"]), None)
        flips = 0
        previous = False
        for row in rows:
            now = bool(row["strict"])
            if previous and not now:
                flips += 1
            previous = now
        meta = json.loads((HERE / "runs" / group / sample_id / "train_meta.json").read_text())
        payload = {
            "sample_id": sample_id,
            "group": group,
            "clean_fastv": {"0.25": step0["texts"]["fastv_0.25"], "0.35": step0["texts"]["fastv_0.35"], "anls": clean_anls},
            "sealed_selector_result": None if sealed is None else {key: sealed[key] for key in ("step", "png_sha256", "fixed_margin", "full_text", "fastv_0.25", "fastv_0.35", "anls_0.25", "anls_0.35", "uint8_linf", "strict")},
            "historical_dynamic_selector": None if historical is None else {"step": historical["step"], "margin": historical["margin"]},
            "trajectory_oracle_result": {"any_strict": any(row["strict"] for row in rows), "first_strict_step": first, "observed_success_then_fail": flips},
            "selector_oracle_agree": bool((sealed is not None and sealed["strict"]) == any(row["strict"] for row in rows)),
            "checkpoints": rows,
            "train_counters": meta["counters"],
            "elapsed_sec": meta["elapsed_sec"],
            "peak_memory_mib": meta["peak_memory_mib"],
            "ratios_0.15_and_0.50": "NOT_RUN",
            "noise": "NOT_RUN",
            "pgd": "NOT_RUN",
            "opgd": "NOT_RUN",
        }
        dest = HERE / "per_sample" / f"{sample_id}_{group}.json"
        dest.write_text(json.dumps(payload, indent=2) + "\n")


def decide() -> dict:
    paired = []
    for sample_id in IDS:
        left = json.loads((HERE / "per_sample" / f"{sample_id}_H0.json").read_text())
        right = json.loads((HERE / "per_sample" / f"{sample_id}_H1.json").read_text())
        h0_strict = bool(left["sealed_selector_result"] and left["sealed_selector_result"]["strict"])
        h1_strict = bool(right["sealed_selector_result"] and right["sealed_selector_result"]["strict"])
        h0_full = sum(row["full_text_exact"] for row in left["checkpoints"])
        h1_full = sum(row["full_text_exact"] for row in right["checkpoints"])
        paired.append({
            "sample_id": sample_id,
            "h0_sealed_strict": h0_strict,
            "h1_sealed_strict": h1_strict,
            "h0_full_exact_checkpoints": h0_full,
            "h1_full_exact_checkpoints": h1_full,
            "h0_oracle_any": left["trajectory_oracle_result"]["any_strict"],
            "h1_oracle_any": right["trajectory_oracle_result"]["any_strict"],
            "h0_sealed_step": None if left["sealed_selector_result"] is None else left["sealed_selector_result"]["step"],
            "h1_sealed_step": None if right["sealed_selector_result"] is None else right["sealed_selector_result"]["step"],
            "selector_oracle_agree_h0": left["selector_oracle_agree"],
            "selector_oracle_agree_h1": right["selector_oracle_agree"],
        })
    def strict_ids(key):
        return [row["sample_id"] for row in paired if row[key]]
    h0_ids, h1_ids = strict_ids("h0_sealed_strict"), strict_ids("h1_sealed_strict")
    gains = [sample_id for sample_id in h1_ids if sample_id not in h0_ids]
    losses = [sample_id for sample_id in h0_ids if sample_id not in h1_ids]
    full_drop = sum(row["h1_full_exact_checkpoints"] for row in paired) < sum(row["h0_full_exact_checkpoints"] for row in paired)
    only_1025 = gains == ["1025"] and not losses
    better = len(h1_ids) > len(h0_ids) and not only_1025 and any(sample_id != "1025" for sample_id in gains) and not full_drop
    if better:
        decision = "H1_OBJECTIVE_SIGNAL_SUPPORTED_ON_DEVELOPMENT"
    else:
        decision = "H1_NO_ADDITIONAL_SIGNAL_ON_DEVELOPMENT"
    payload = {
        "decision": decision,
        "label": "development objective signal" if better else "no additional development objective signal",
        "h0_sealed_strict": h0_ids,
        "h1_sealed_strict": h1_ids,
        "gains": gains,
        "losses": losses,
        "excluding_1025": {
            "h0": [sample_id for sample_id in h0_ids if sample_id != "1025"],
            "h1": [sample_id for sample_id in h1_ids if sample_id != "1025"],
            "gains": [sample_id for sample_id in gains if sample_id != "1025"],
            "losses": [sample_id for sample_id in losses if sample_id != "1025"],
        },
        "full_exact_checkpoint_drop": full_drop,
        "paired": paired,
        "pgd": "NOT_RUN",
        "opgd": "NOT_RUN",
        "noise": "NOT_RUN",
        "unseen": "NOT_RUN",
        "stage_c": "NOT_RUN",
    }
    (HERE / "paired_h0_h1.json").write_text(json.dumps(payload, indent=2) + "\n")
    (HERE / "final_decision.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def write_protocol() -> None:
    parent_protocol = PARENT / "FROZEN_V11_PROTOCOL.json"
    parent_decision = PARENT / "final_decision.json"
    protocol = {
        "parent_directory": str(PARENT),
        "parent_protocol_sha256": file_sha(parent_protocol),
        "parent_decision_sha256": file_sha(parent_decision),
        "retired_gate": "The old aux_grad_only check required the loss to be independent of pages that are not optimization variables. That requirement is retired before any H0 or H1 result.",
        "new_gate": "PASS iff clean fixed and dynamic references match, teacher forcing matches clean generate, the current perturbed generate supplies the keep-set, optimizer parameters are exactly {delta}, delta matches the aux page shape, reconstructed non-aux pixel_values are unchanged, the aux update is nonzero, the image list replaces only aux_idx, and the reloaded aux PNG has 0 < uint8 Linf <= 8. A counterfactual non-aux gradient is context_gradient_observed and is not pass/fail.",
        "gate_samples": list(GATE_IDS),
        "one_step_groups": ["H0", "H1"],
        "development_samples": list(IDS),
        "unchanged": ["model", "checkpoint schedule", "processor", "prompt", "page order", "aux index", "pixel budget", "train ratios", "seed", "lambda_full", "Adam lr"],
        "seed": SEED,
        "lambda_full": LAMBDA_FULL,
        "adam_lr": ADAM_LR,
        "steps": STEPS,
        "checkpoints": list(CHECKPOINTS),
        "train_ratios": [0.25, 0.35],
        "h0": "dynamic FastV reference, current keep-set, native-FDP, historical Adam loop",
        "h1": "same loop; FastV mean soft margin uses the clean token sequence saved at r=0.25 and r=0.35",
        "h0_replay": "png sha256, full text, both FastV texts, both keep hashes, and selector margin within 1e-5 must match historical NATIVE_FDP at every checkpoint; the historical selector step must match",
        "common_selector": "legal if nonzero, valid, full text exact, 0 < uint8 Linf <= 8, and full disk replay; rank by fixed-reference soft margin, then earlier step",
        "oracle_label": "post-hoc oracle checkpoint diagnostic, not an unlabeled selector result",
        "not_run": ["P0", "P1", "O0", "O1", "ALMA", "noise", "unseen", "stage C"],
    }
    (HERE / "FROZEN_V11A_PROTOCOL.json").write_text(json.dumps(protocol, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("init", "phase_a", "train", "compare_h0", "score", "decide"))
    parser.add_argument("--sample")
    parser.add_argument("--group")
    args = parser.parse_args()
    HERE.mkdir(parents=True, exist_ok=True)
    if args.mode == "init":
        write_protocol()
        return
    if args.mode == "compare_h0":
        result = compare_h0()
        if result["status"] != "PASS":
            raise SystemExit(2)
        return
    if args.mode == "decide":
        decide()
        return
    model, processor = load_model_frozen()
    install_counters(model)
    if args.mode == "phase_a":
        result = phase_a(model, processor)
        if result["status"] != "PASS":
            raise SystemExit(2)
        return
    if args.mode == "train":
        train(model, processor, args.sample, args.group)
        return
    if args.mode == "score":
        score_group(model, processor, args.group)
        return


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise
