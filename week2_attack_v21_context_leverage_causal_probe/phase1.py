"""One-step local causal probe from the clean page. No multi-step optimization."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
sys.path[:0] = [str(ROOT), str(ROOT / "week1_mechanism"), str(ROOT / "gate_experiment")]
HERE = ROOT / "week2_attack_v21_context_leverage_causal_probe"

from week2_attack_v20_fourway_content_ablation import runtime_utils as rt  # noqa: E402
from week2_attack_v20_fourway_content_ablation.losses import fixed_noeos_loss, full_fdp_loss  # noqa: E402
from week2_attack_v20_fourway_content_ablation.optim import sign_step  # noqa: E402
from week2_attack_v20_fourway_content_ablation.protocol import EPS, EPSILON_UINT8, LAMBDA_FULL, STEP_ALPHA, TRAIN_RATIO  # noqa: E402
from week2_attack_v7b_native_cache_margin_admission.native_cache_margin import hard_margin_row, soft_margin_row  # noqa: E402
from week2_attack_v21_context_leverage_causal_probe.groups import SCALES, supporting_pair  # noqa: E402


def pack(grad: torch.Tensor) -> dict:
    flat = grad.detach().float()
    return {
        "hash": rt.hash_tensor(grad),
        "l1": float(flat.abs().sum()),
        "l2": float(flat.norm()),
        "linf": float(flat.abs().max()),
    }


def cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    denom = float(left.float().norm() * right.float().norm())
    if denom == 0.0:
        return None
    return float(torch.dot(left.float().flatten(), right.float().flatten()) / denom)


def margins(rows, reference: list[int], positions: list[int]) -> dict:
    hard = [float(hard_margin_row(rows[index], int(reference[index])).detach()) for index in positions]
    soft = [float(soft_margin_row(rows[index], int(reference[index])).detach()) for index in positions]
    return {
        "positions": positions,
        "hard": hard,
        "soft": soft,
        "hard_mean": sum(hard) / len(hard),
        "soft_mean": sum(soft) / len(soft),
    }


def teacher_rows(model, state, item, page, reference, kept):
    pixels = rt.v7d.pixels_from_page(state["inputs"], item["aux_idx"], page)
    keep_index = rt.v7d.keep_index_from_kept(kept, state["spans"], int(state["inputs"]["input_ids"].shape[1]), page.device)
    return rt.v7d.fastv_rows(model, state["inputs"], pixels, reference, keep_index)


def natural_kept(model, state, item, page):
    pixels = rt.v7d.pixels_from_page(state["inputs"], item["aux_idx"], page)
    tokens, kept = rt.v7d.generate_tokens(model, state["inputs"], pixels, state["spans"], TRAIN_RATIO)
    return [int(token) for token in tokens], [int(index) for index in kept]


def compute_grads(model, processor, state, item, reference, positions, clean_kept):
    delta = torch.zeros_like(state["page"], requires_grad=True)
    if not positions:
        zero = torch.zeros_like(delta)
        text = rt.fastv_generate(model, processor, item, item["images"][item["aux_idx"]], TRAIN_RATIO)
        return {
            "attack_loss": None,
            "fdp_loss": None,
            "attack_grad": zero,
            "full_grad": zero,
            "update_grad": zero,
            "natural_text": text["text"],
            "natural_tokens": text["tokens"],
            "k": text["k"],
            "keep_hash": text["keep_hash"],
        }
    page = state["page"] + delta
    natural_tokens, kept = natural_kept(model, state, item, page)
    rows = teacher_rows(model, state, item, page, reference, kept)
    attack = fixed_noeos_loss(rows, reference, positions)
    if not rt.finite_tensor(attack):
        raise RuntimeError("non-finite attack loss")
    attack.backward()
    attack_grad = delta.grad.detach().clone()
    attack_value = float(attack.detach())
    core_rows = margins(rows, reference, state["core_positions"])
    context_rows = margins(rows, reference, state["context_positions"])
    del attack, rows
    delta.grad = None
    hard = rt.hard_rows(model, state, item, state["page"] + delta, state["clean_tokens"])
    fdp = full_fdp_loss(hard, state["clean_hard"])
    if not rt.finite_tensor(fdp):
        raise RuntimeError("non-finite FDP loss")
    (LAMBDA_FULL * fdp).backward()
    full_grad = delta.grad.detach().clone()
    fdp_value = float(fdp.detach())
    del fdp, hard
    delta.grad = None
    return {
        "attack_loss": attack_value,
        "fdp_loss": fdp_value,
        "attack_grad": attack_grad,
        "full_grad": full_grad,
        "update_grad": attack_grad + full_grad,
        "natural_text": rt.v7d.decode(processor, torch.tensor(natural_tokens)).strip(),
        "natural_tokens": natural_tokens,
        "k": len(kept),
        "keep_hash": rt.keep_hash(kept),
        "core_at_clean_graph": core_rows,
        "context_at_clean_graph": context_rows,
    }


def intervene(model, processor, state, item, reference, grad, scale, clean_kept, tag):
    delta = torch.zeros_like(state["page"])
    if float(grad.abs().sum()) != 0.0:
        sign_step(delta, state["page"], grad, STEP_ALPHA * scale, EPS)
    image = rt.project_image(state["page"], delta)
    path = HERE / "phase1_png" / tag / f"scale_{scale:.2f}.png"
    rt.save_png(image, path)
    reloaded = rt.reload_png(path)
    guard = rt.full_guard(model, processor, item, state, reloaded)
    page = rt.tensor_page(reloaded)
    natural_tokens, kept = natural_kept(model, state, item, page)
    with torch.no_grad():
        natural_teacher = teacher_rows(model, state, item, page, reference, kept)
        frozen_teacher = teacher_rows(model, state, item, page, reference, clean_kept)
    core_natural = margins(natural_teacher, reference, state["core_positions"])
    core_frozen = margins(frozen_teacher, reference, state["core_positions"])
    context_natural = margins(natural_teacher, reference, state["context_positions"])
    del natural_teacher, frozen_teacher
    linf = rt.uint8_linf(item["images"][item["aux_idx"]], reloaded)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "scale": scale,
        "png": str(path),
        "png_sha256": rt.sha_file(path),
        "uint8_linf": linf,
        "linf_ok": linf <= EPSILON_UINT8,
        "png_reloaded": True,
        "full_exact": bool(guard["full_text_exact"] and guard["full_token_exact"] and guard["full_replay"]),
        "full_text": guard["full_text"],
        "natural_text": rt.v7d.decode(processor, torch.tensor(natural_tokens)).strip(),
        "natural_tokens": natural_tokens,
        "k": len(kept),
        "keep_hash": rt.keep_hash(kept),
        "keep_hash_equals_clean": rt.keep_hash(kept) == rt.keep_hash(clean_kept),
        "core_margin_natural": core_natural["hard_mean"],
        "core_margin_frozen_keep": core_frozen["hard_mean"],
        "core_soft_natural": core_natural["soft_mean"],
        "context_margin_natural": context_natural["hard_mean"],
        "core_hard_natural": core_natural["hard"],
        "core_hard_frozen_keep": core_frozen["hard"],
    }


def main() -> None:
    groups = json.loads((HERE / "frozen_token_groups.json").read_text())
    controls = json.loads((HERE / "random_controls.json").read_text())
    if groups["status"] != "PASS":
        raise SystemExit("phase 0 did not pass")
    reference = [row["token_id"] for row in groups["tokens"]]
    named = {
        "CORE": groups["s_core"]["positions"],
        "CONTEXT": groups["s_context"]["positions"],
        "ALL": groups["s_all"]["positions"],
        "NOOP": [],
        "SHAM": groups["s_all"]["positions"],
    }
    for row in controls["controls"]:
        named[row["name"]] = row["positions"]
    rt.configure()
    model, processor = rt.v7d.load_model_frozen()
    item = rt.v7d.load_dev("47450")
    state = rt.prepare_clean_state(model, processor, item)
    state["core_positions"] = groups["s_core"]["positions"]
    state["context_positions"] = groups["s_context"]["positions"]
    clean_kept = state["clean_fast_kept"]
    clean_image = item["images"][item["aux_idx"]].convert("RGB")
    clean_page = rt.tensor_page(clean_image)
    with torch.no_grad():
        clean_tokens, clean_runtime_kept = natural_kept(model, state, item, clean_page)
        clean_rows = teacher_rows(model, state, item, clean_page, reference, clean_runtime_kept)
        clean_core = margins(clean_rows, reference, state["core_positions"])
        del clean_rows
    diagnostics = []
    local_steps = []
    grads = {}
    for name, positions in named.items():
        used = [] if name in {"NOOP", "SHAM"} and name == "NOOP" else positions
        if name == "SHAM":
            used = groups["s_all"]["positions"]
        computed = compute_grads(model, processor, state, item, reference, [] if name == "NOOP" else used, clean_kept)
        update = torch.zeros_like(computed["update_grad"]) if name in {"NOOP", "SHAM"} else computed["update_grad"]
        grads[name] = update.detach()
        attack_stats = pack(computed["attack_grad"])
        full_stats = pack(computed["full_grad"])
        diagnostics.append({
            "group": name,
            "positions": [] if name == "NOOP" else used,
            "attack_loss": computed["attack_loss"],
            "fdp_loss": computed["fdp_loss"],
            "attack_gradient": attack_stats,
            "full_gradient": full_stats,
            "cosine_attack_full": cosine(computed["attack_grad"], computed["full_grad"]),
            "update_applied": name not in {"NOOP", "SHAM"},
            "update_gradient": pack(update),
            "natural_text": computed["natural_text"],
            "natural_tokens": computed["natural_tokens"],
            "k": computed["k"],
            "keep_hash": computed["keep_hash"],
            "core_margin_hard_mean": None if "core_at_clean_graph" not in computed else computed["core_at_clean_graph"]["hard_mean"],
            "core_margin_hard": None if "core_at_clean_graph" not in computed else computed["core_at_clean_graph"]["hard"],
            "core_margin_soft": None if "core_at_clean_graph" not in computed else computed["core_at_clean_graph"]["soft"],
            "context_margin_hard": None if "context_at_clean_graph" not in computed else computed["context_at_clean_graph"]["hard"],
            "context_margin_soft": None if "context_at_clean_graph" not in computed else computed["context_at_clean_graph"]["soft"],
            "full_text": state["clean_text"],
            "full_tokens": state["clean_tokens"],
        })
        for scale in SCALES:
            step = intervene(model, processor, state, item, reference, update, scale, clean_kept, name)
            step["group"] = name
            step["clean_core_margin_natural"] = clean_core["hard_mean"]
            local_steps.append(step)
            print(json.dumps({"group": name, "scale": scale, "core_natural": step["core_margin_natural"], "core_frozen": step["core_margin_frozen_keep"], "full_exact": step["full_exact"], "text": step["natural_text"]}, ensure_ascii=False), flush=True)
        del computed, update
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    context = {row["scale"]: row for row in local_steps if row["group"] == "CONTEXT"}
    random_context = {}
    for row in controls["controls"]:
        if row["kind"] != "R_context":
            continue
        random_context[row["seed"]] = {item["scale"]: item for item in local_steps if item["group"] == row["name"]}
    if controls["status"] != "PASS":
        decision = {
            "status": "BLOCKED_RANDOM_POSITION_CONTROL_UNDERDETERMINED",
            "pair": {"supported": False, "scales": [], "random_status": "NOT_CONSTRUCTED"},
        }
    else:
        pair = supporting_pair(context, random_context, clean_core["hard_mean"])
        decision = {
            "status": "PHASE1_LOCAL_EFFECT_PRESENT" if pair["supported"] else "NO_GO_CONTEXT_LEVERAGE_LOCAL_CAUSAL_EFFECT",
            "pair": pair,
            "clean_core_margin_natural": clean_core["hard_mean"],
            "decision_margin": "mean hard margin of S_core on the PNG path with the current natural keep-set",
        }
    rt.write_json(HERE / "gradient_diagnostics.json", {"groups": diagnostics})
    rt.write_json(HERE / "local_step_results.json", {"clean_core": clean_core, "steps": local_steps})
    rt.write_json(HERE / "phase1_decision.json", decision)
    with (HERE / "cross_effect_matrix.csv").open("w", newline="") as handle:
        fields = ["group", "scale", "core_margin_natural", "core_margin_frozen_keep", "margin_drop_natural", "full_exact", "uint8_linf", "keep_hash_equals_clean", "natural_text", "png_sha256"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in local_steps:
            writer.writerow({
                "group": row["group"],
                "scale": row["scale"],
                "core_margin_natural": row["core_margin_natural"],
                "core_margin_frozen_keep": row["core_margin_frozen_keep"],
                "margin_drop_natural": clean_core["hard_mean"] - row["core_margin_natural"],
                "full_exact": row["full_exact"],
                "uint8_linf": row["uint8_linf"],
                "keep_hash_equals_clean": row["keep_hash_equals_clean"],
                "natural_text": row["natural_text"],
                "png_sha256": row["png_sha256"],
            })
    report = [
        "# Phase 1 local causal probe",
        "",
        f"Status: `{decision['status']}`",
        "",
        f"Clean S_core hard-margin mean on the natural keep-set: {clean_core['hard_mean']}.",
        "",
        "A supporting pair needs two adjacent step scales, after PNG reload, with Full exact, a lower natural-keep S_core margin, and a larger drop than every R_context control.",
        "",
        f"Pair result: `{json.dumps(decision['pair'], ensure_ascii=False)}`",
        "",
    ]
    (HERE / "phase1_report.md").write_text("\n".join(report) + "\n")
    print(json.dumps(decision, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
