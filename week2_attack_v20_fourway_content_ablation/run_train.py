"""One method queue. Loads the model once and walks its sample list."""

from __future__ import annotations

import sys

sys.path[:0] = ["/home/yuesiting/autodl-tmp/multimodal_attack_project"]

import argparse
import json
import time
import traceback
from pathlib import Path

import torch

from week2_attack_v20_fourway_content_ablation import runtime_utils as rt
from week2_attack_v20_fourway_content_ablation.losses import (
    core_content_loss,
    dynamic_sequence_loss,
    fixed_all_loss,
    fixed_noeos_loss,
    full_fdp_loss,
    selected_hard_margins,
)
from week2_attack_v20_fourway_content_ablation.optim import sign_step, state_hash
from week2_attack_v20_fourway_content_ablation.protocol import (
    CHECKPOINTS,
    EPS,
    EPSILON_UINT8,
    LAMBDA_FULL,
    MAX_ACCEPTED,
    MAX_ATTEMPTS,
    METHODS,
    STEP_ALPHA,
    TRAIN_RATIO,
)
from week2_attack_v20_fourway_content_ablation.selection import (
    CoreSpanUnavailable,
    assert_core_not_silent_noeos,
    dynamic_sequence_selection,
)
from week2_attack_v20_fourway_content_ablation.selector import choose_fixed

HERE = rt.HERE


class Counters:
    def __init__(self) -> None:
        self.forward = 0
        self.backward = 0
        self.generate = 0

    def as_dict(self) -> dict:
        return {"forward": self.forward, "backward": self.backward, "generate": self.generate}


def selection_from_manifest(row: dict, name: str) -> dict:
    payload = row[f"selection_{name}"]
    if payload is None:
        raise RuntimeError(f"manifest is missing selection {name}")
    return payload


def log_line(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False)
    with path.open("a") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def attack_forward(model, processor, item, state, page, method: str, frozen: dict, counters: Counters):
    pixels = rt.v7d.pixels_from_page(state["inputs"], item["aux_idx"], page)
    counters.generate += 1
    counters.forward += 1
    natural_tokens, natural_kept = rt.v7d.generate_tokens(model, state["inputs"], pixels, state["spans"], TRAIN_RATIO)
    natural_tokens = [int(token) for token in natural_tokens]
    natural_kept = [int(index) for index in natural_kept]
    keep_index = rt.v7d.keep_index_from_kept(
        natural_kept, state["spans"], int(state["inputs"]["input_ids"].shape[1]), page.device
    )
    counters.forward += 1
    if method == "A":
        rows = rt.v7d.fastv_rows(model, state["inputs"], pixels, natural_tokens, keep_index)
        if [int(row.detach().argmax()) for row in rows] != natural_tokens:
            raise RuntimeError("dynamic FastV argmax left the generated tokens")
        attack = dynamic_sequence_loss(rows, natural_tokens)
        positions = list(range(len(natural_tokens)))
        token_ids = natural_tokens
        reference_ids = natural_tokens
        selection = dynamic_sequence_selection(processor.tokenizer, natural_tokens)
    else:
        reference_ids = list(frozen["reference_ids"])
        rows = rt.v7d.fastv_rows(model, state["inputs"], pixels, reference_ids, keep_index)
        if method == "B":
            positions = list(frozen["B"]["positions"])
            attack = fixed_all_loss(rows, reference_ids)
        elif method == "C":
            positions = list(frozen["C"]["positions"])
            attack = fixed_noeos_loss(rows, reference_ids, positions)
        elif method == "D":
            positions = list(frozen["D"]["positions"])
            attack = core_content_loss(rows, reference_ids, positions)
        else:
            raise RuntimeError(method)
        token_ids = reference_ids
        selection = None
    margins = selected_hard_margins(rows, token_ids, positions)
    info = {
        "natural_tokens": natural_tokens,
        "natural_text": rt.v7d.decode(processor, torch.tensor(natural_tokens)).strip(),
        "k": len(natural_kept),
        "keep_hash": rt.keep_hash(natural_kept),
        "keep_jaccard_vs_clean": rt.jaccard_kept(natural_kept, state["clean_fast_kept"]),
        "positions": positions,
        "selected_token_ids": [int(token_ids[index]) for index in positions],
        "selected_margins": margins,
        "selection_reason": None if selection is None else selection.reason,
        "dynamic_selection": None
        if selection is None
        else {
            "positions": list(selection.positions),
            "token_ids": list(selection.token_ids),
            "pieces": list(selection.pieces),
            "reason": selection.reason,
        },
    }
    return attack, rows, info


def fdp_forward(model, state, item, page, counters: Counters):
    counters.forward += 1
    hard = rt.hard_rows(model, state, item, page, state["clean_tokens"])
    loss = full_fdp_loss(hard, state["clean_hard"])
    return loss, [float(value) for value in hard.detach()]


def measure(model, processor, item, state, page, method: str, frozen: dict, counters: Counters) -> dict:
    with torch.no_grad():
        attack, rows, info = attack_forward(model, processor, item, state, page, method, frozen, counters)
        attack_value = float(attack.detach())
        del attack, rows
        fdp, hard_margins = fdp_forward(model, state, item, page, counters)
        fdp_value = float(fdp.detach())
        del fdp
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "attack_loss": attack_value,
        "fdp_loss": fdp_value,
        "total_loss": float(attack_value + LAMBDA_FULL * fdp_value),
        "full_hard_margins": hard_margins,
        **info,
    }


def train_sample(model, processor, sample_id: str, method: str, manifest: dict, out_root: Path, max_accepted: int, max_attempts: int, checkpoints: tuple[int, ...]) -> dict:
    if method not in METHODS:
        raise RuntimeError(method)
    row = next(item for item in manifest["samples"] if item["sample_id"] == sample_id)
    item = rt.v7d.load_dev(sample_id)
    state = rt.prepare_clean_state(model, processor, item)
    if state["clean_fast_tokens"] != row["clean_fastv_token_ids"]:
        raise RuntimeError("clean FastV reference drifted away from the frozen semantic manifest")
    if state["clean_text"] != row["clean_full_text"]:
        raise RuntimeError("clean Full text drifted away from the frozen semantic manifest")
    frozen = {
        "reference_ids": list(row["clean_fastv_token_ids"]),
        "B": selection_from_manifest(row, "b"),
        "C": selection_from_manifest(row, "c"),
        "D": selection_from_manifest(row, "d"),
    }
    noeos_positions = tuple(frozen["C"]["positions"])
    core_positions = tuple(frozen["D"]["positions"])
    if method == "D" and not frozen["D"]["available"]:
        if core_positions:
            raise RuntimeError("CORE_SPAN_UNAVAILABLE still had positions")
        if core_positions == noeos_positions and noeos_positions:
            raise RuntimeError("core span silently reused noEOS positions")
        payload = {
            "status": "CORE_SPAN_UNAVAILABLE",
            "sample_id": sample_id,
            "method": method,
            "method_name": METHODS[method],
            "counts_in_method_denominator": False,
        }
        rt.write_json(out_root / "per_run" / sample_id / f"{method}.json", payload)
        return payload
    assert_core_not_silent_noeos(
        type("S", (), {"available": frozen["D"]["available"], "positions": core_positions})(),
        type("S", (), {"positions": noeos_positions})(),
    )
    dest = out_root / "per_run" / sample_id / method
    dest.mkdir(parents=True, exist_ok=True)
    log_path = dest / "logs.txt"
    counters = Counters()
    delta = torch.zeros_like(state["page"], requires_grad=True)
    records = []
    checkpoint_rows = []
    accepted = 0
    attempts = 0
    started = time.perf_counter()

    def emit_checkpoint(accepted_count: int) -> None:
        image = rt.project_image(state["page"], delta)
        path = dest / "checkpoints" / f"accepted_{accepted_count:03d}.png"
        file_sha = rt.save_png(image, path)
        reloaded = rt.reload_png(path)
        reloadable = list(reloaded.getdata()) == list(image.getdata())
        guard = rt.full_guard(model, processor, item, state, reloaded)
        counters.generate += 2
        counters.forward += 2
        page = rt.tensor_page(reloaded).detach().requires_grad_(True)
        measured = measure(model, processor, item, state, page, method, frozen, counters)
        fast = rt.fastv_generate(model, processor, item, reloaded, TRAIN_RATIO)
        counters.generate += 1
        counters.forward += 1
        linf = rt.uint8_linf(item["images"][item["aux_idx"]], reloaded)
        selector_row = {
            "checkpoint": accepted_count,
            "nonzero": linf > 0,
            "output_valid": bool(fast["valid"] and guard["output_valid"]),
            "full_text_exact": guard["full_text_exact"],
            "full_token_exact": guard["full_token_exact"],
            "full_replay": guard["full_replay"],
            "linf_ok": linf <= EPSILON_UINT8,
            "png_reloadable": reloadable,
            "attack_surrogate": measured["attack_loss"],
        }
        stored = {
            **selector_row,
            "png": str(path),
            "png_sha256": file_sha,
            "uint8_linf": linf,
            "fastv_text": fast["text"],
            "fastv_tokens": fast["tokens"],
            "full_text": guard["full_text"],
            "full_tokens": guard["full_tokens"],
            "k": fast["k"],
            "keep_hash": fast["keep_hash"],
            "keep_jaccard_vs_clean": rt.jaccard_kept(fast["kept"], state["clean_fast_kept"]),
            "attack_loss": measured["attack_loss"],
            "fdp_loss": measured["fdp_loss"],
            "total_loss": measured["total_loss"],
            "selected_margins": measured["selected_margins"],
            "positions": measured["positions"],
            "selected_token_ids": measured["selected_token_ids"],
            "optimizer_state_hash": state_hash(None),
        }
        checkpoint_rows.append(stored)
        log_line(log_path, {"event": "checkpoint", "sample": sample_id, "method": method, **selector_row, "png_sha256": file_sha, "fastv_text": fast["text"]})

    emit_checkpoint(0)
    while accepted < max_accepted and attempts < max_attempts:
        attempts += 1
        step_started = time.perf_counter()
        previous = delta.detach().clone()
        optimizer_before = state_hash(None)
        delta.grad = None
        try:
            attack, rows, info = attack_forward(model, processor, item, state, state["page"] + delta, method, frozen, counters)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise
            log_line(log_path, {"event": "attempt_error", "attempt": attempts, "error": str(exc)})
            raise
        if not rt.finite_tensor(attack):
            raise RuntimeError("non-finite attack loss")
        counters.backward += 1
        attack.backward()
        if delta.grad is None or not rt.finite_tensor(delta.grad):
            raise RuntimeError("non-finite attack gradient")
        grad_attack = delta.grad.detach().clone()
        attack_value = float(attack.detach())
        del attack, rows
        delta.grad = None
        fdp, full_margins = fdp_forward(model, state, item, state["page"] + delta, counters)
        if not rt.finite_tensor(fdp):
            raise RuntimeError("non-finite FDP loss")
        counters.backward += 1
        (LAMBDA_FULL * fdp).backward()
        if delta.grad is None or not rt.finite_tensor(delta.grad):
            raise RuntimeError("non-finite FDP gradient")
        grad_fdp = delta.grad.detach().clone()
        grad = grad_attack + grad_fdp
        fdp_value = float(fdp.detach())
        del fdp
        delta.grad = None
        sign_step(delta, state["page"], grad, STEP_ALPHA, EPS)
        candidate_delta_hash = rt.hash_tensor(delta)
        candidate = rt.project_image(state["page"], delta)
        candidate_path = dest / "candidates" / f"attempt_{attempts:03d}.png"
        candidate_sha = rt.save_png(candidate, candidate_path)
        reloaded = rt.reload_png(candidate_path)
        float_page = (state["page"] + delta).detach()
        png_page = rt.tensor_page(reloaded)
        guard = rt.full_guard(model, processor, item, state, reloaded)
        counters.generate += 2
        counters.forward += 2
        linf = rt.uint8_linf(item["images"][item["aux_idx"]], reloaded)
        passed = bool(guard["full_text_exact"] and guard["full_token_exact"] and guard["full_replay"] and linf <= EPSILON_UINT8)
        changed = candidate_sha != rt.sha_file(dest / "checkpoints" / f"accepted_{accepted:03d}.png")
        accept = passed and changed
        if not accept:
            delta.data.copy_(previous)
            if state_hash(None) != optimizer_before:
                raise RuntimeError("sign-PGD optimizer state changed across a rejected update")
            candidate_path.unlink(missing_ok=True)
        else:
            accepted += 1
            sealed = dest / "checkpoints" / f"accepted_{accepted:03d}.png"
            sealed.write_bytes(candidate_path.read_bytes())
            candidate_path.unlink(missing_ok=True)
        record = {
            "attempt": attempts,
            "accepted_after": accepted,
            "accepted": accept,
            "rejection_reason": None if accept else ("FULL_GUARD" if not passed else "NO_UINT8_CHANGE"),
            "total_loss": attack_value + LAMBDA_FULL * fdp_value,
            "attack_loss": attack_value,
            "fdp_loss": fdp_value,
            "full_token_hard_margins": full_margins,
            "selected_token_margins": info["selected_margins"],
            "grad_hash": rt.hash_tensor(grad),
            "grad_l2": float(grad.float().norm()),
            "grad_linf": float(grad.float().abs().max()),
            "candidate_delta_hash": candidate_delta_hash,
            "delta_hash": rt.hash_tensor(delta),
            "float_projection_hash": rt.hash_tensor(float_page),
            "png_page_hash": rt.hash_tensor(png_page),
            "candidate_png_sha256": candidate_sha,
            "fastv_text": info["natural_text"],
            "fastv_tokens": info["natural_tokens"],
            "k": info["k"],
            "keep_hash": info["keep_hash"],
            "keep_jaccard_vs_clean": info["keep_jaccard_vs_clean"],
            "positions": info["positions"],
            "selected_token_ids": info["selected_token_ids"],
            "full_text": guard["full_text"],
            "full_tokens": guard["full_tokens"],
            "full_text_exact": guard["full_text_exact"],
            "full_token_exact": guard["full_token_exact"],
            "full_replay": guard["full_replay"],
            "uint8_linf": linf,
            "optimizer_state_hash": state_hash(None),
            "wall_seconds": time.perf_counter() - step_started,
            "counters": counters.as_dict(),
            "dynamic_selection": info["dynamic_selection"],
        }
        records.append(record)
        log_line(
            log_path,
            {
                "event": "update",
                "sample": sample_id,
                "method": method,
                "attempt": attempts,
                "accepted": accept,
                "accepted_after": accepted,
                "attack_loss": attack_value,
                "fdp_loss": fdp_value,
                "grad_hash": record["grad_hash"],
                "keep_hash": info["keep_hash"],
                "fastv_text": info["natural_text"],
            },
        )
        if accept and accepted in checkpoints:
            emit_checkpoint(accepted)
        del grad, grad_attack, grad_fdp
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    selector_input = [
        {
            "checkpoint": row["checkpoint"],
            "nonzero": row["nonzero"],
            "output_valid": row["output_valid"],
            "full_text_exact": row["full_text_exact"],
            "full_token_exact": row["full_token_exact"],
            "full_replay": row["full_replay"],
            "linf_ok": row["linf_ok"],
            "png_reloadable": row["png_reloadable"],
            "attack_surrogate": row["attack_surrogate"],
        }
        for row in checkpoint_rows
    ]
    chosen = choose_fixed(selector_input)
    payload = {
        "status": "DONE",
        "sample_id": sample_id,
        "method": method,
        "method_name": METHODS[method],
        "optimizer": "sign_pgd",
        "step_alpha": STEP_ALPHA,
        "lambda_full": LAMBDA_FULL,
        "train_ratio": TRAIN_RATIO,
        "max_accepted": max_accepted,
        "max_attempts": max_attempts,
        "accepted": accepted,
        "attempts": attempts,
        "frozen_reference_ids": frozen["reference_ids"],
        "selection_b": frozen["B"],
        "selection_c": frozen["C"],
        "selection_d": frozen["D"],
        "same_positions_as_noeos": bool(frozen["D"]["same_positions_as_noeos"]),
        "updates": records,
        "checkpoints": checkpoint_rows,
        "fixed_selector": None
        if chosen is None
        else {"checkpoint": chosen["checkpoint"], "attack_surrogate": chosen["attack_surrogate"]},
        "counters": counters.as_dict(),
        "wall_seconds": time.perf_counter() - started,
        "semantic_manifest_sha256": rt.sha_file(HERE / "semantic_manifest.json"),
    }
    rt.write_json(out_root / "per_run" / sample_id / f"{method}.json", payload)
    log_line(log_path, {"event": "done", "sample": sample_id, "method": method, "accepted": accepted, "selector": payload["fixed_selector"]})
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=tuple(METHODS))
    parser.add_argument("--samples", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-accepted", type=int, default=MAX_ACCEPTED)
    parser.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS)
    parser.add_argument("--checkpoints", default=",".join(str(item) for item in CHECKPOINTS))
    args = parser.parse_args()
    out = Path(args.out)
    manifest = json.loads((HERE / "semantic_manifest.json").read_text())
    checkpoints = tuple(int(item) for item in args.checkpoints.split(",") if item.strip())
    rt.configure()
    model, processor = rt.v7d.load_model_frozen()
    model.eval()
    for sample_id in [item.strip() for item in args.samples.split(",") if item.strip()]:
        dest = out / "per_run" / sample_id / f"{args.method}.json"
        if dest.exists():
            existing = json.loads(dest.read_text())
            if existing.get("status") in {"DONE", "CORE_SPAN_UNAVAILABLE"}:
                print(json.dumps({"skip": sample_id, "status": existing["status"]}), flush=True)
                continue
        try:
            result = train_sample(
                model,
                processor,
                sample_id,
                args.method,
                manifest,
                out,
                args.max_accepted,
                args.max_attempts,
                checkpoints,
            )
            print(json.dumps({"sample": sample_id, "method": args.method, "status": result["status"]}), flush=True)
        except torch.cuda.OutOfMemoryError:
            traceback.print_exc()
            raise SystemExit(42)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                traceback.print_exc()
                raise SystemExit(42)
            traceback.print_exc()
            rt.write_json(
                dest,
                {"status": "FAILED_RUNTIME", "sample_id": sample_id, "method": args.method, "error": str(exc)},
            )
            raise SystemExit(1)


if __name__ == "__main__":
    main()
