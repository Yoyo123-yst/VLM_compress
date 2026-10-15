"""One-shot radius scan along the frozen v21a attack-only signs. No training."""

from __future__ import annotations

import hashlib
import json
import time
import sys
from pathlib import Path

import torch

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
sys.path[:0] = [str(ROOT), str(ROOT / "week1_mechanism"), str(ROOT / "gate_experiment")]
HERE = ROOT / "week2_attack_v21b_full_safety_slack_scan"
V20 = ROOT / "week2_attack_v20_fourway_content_ablation"
V21 = ROOT / "week2_attack_v21_context_leverage_causal_probe"
V21A = ROOT / "week2_attack_v21a_context_leverage_repair"

from week2_attack_v20_fourway_content_ablation import runtime_utils as rt  # noqa: E402
from week2_attack_v20_fourway_content_ablation.losses import fixed_noeos_loss  # noqa: E402
from week2_attack_v20_fourway_content_ablation.optim import sign_step  # noqa: E402
from week2_attack_v20_fourway_content_ablation.protocol import TRAIN_RATIO  # noqa: E402
from week2_attack_v20_fourway_content_ablation.semantics import classify  # noqa: E402
from week2_attack_v21b_full_safety_slack_scan.measure import image_stats, score_image  # noqa: E402

RADII = (0, 1, 2, 4, 6, 8)
EXPECTED_KEEP = "4cdba19ca679fa03b6713894463fa303ce07866e41bab92265aaeae62995bfe2"
CORE_TEXT = "William J. Darby"


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_block(status: str, detail: dict) -> None:
    payload = {"status": status, "next_step": "NO_MULTISTEP_COMPARISON_AUTHORIZED", "comparison_started": False, "detail": detail}
    rt.write_json(HERE / "final_decision.json", payload)
    (HERE / "final_report.md").write_text(f"# Full 安全余量扫描\n\n状态：`{status}`\n\n扫描没有进入安全余量解释。\n")
    print(json.dumps(payload), flush=True)
    raise SystemExit(2)


def heartbeat(row: dict) -> None:
    path = HERE / "runtime" / "heartbeat.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def check_provenance() -> None:
    sealed = json.loads((V21A / "sealed_hashes_after.json").read_text())
    mismatched = []
    for rel, digest in sealed["files"].items():
        if sha_file(ROOT / rel) != digest:
            mismatched.append(rel)
    groups = V21 / "frozen_token_groups.json"
    geometry = V21A / "gradient_geometry.json"
    current = {
        "frozen_token_groups": sha_file(groups),
        "gradient_geometry": sha_file(geometry),
        "sealed_matches": not mismatched,
    }
    rt.write_json(HERE / "provenance.json", {"sealed_reference": "week2_attack_v21a_context_leverage_repair/sealed_hashes_after.json", "current": current, "mismatched": mismatched})
    if mismatched:
        write_block("BLOCKED_PROVENANCE_HASH_DRIFT", {"files": mismatched})


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.dot(left.float().flatten(), right.float().flatten()) / (left.float().norm() * right.float().norm()))


def sign_agreement(left: torch.Tensor, right: torch.Tensor) -> float:
    left_sign = left.detach().float().sign()
    right_sign = right.detach().float().sign()
    active = (left_sign != 0) & (right_sign != 0)
    return float((left_sign[active] == right_sign[active]).float().mean())


def attack_grads(model, state, item, reference, core_pos, context_pos, all_pos, cost: dict) -> dict[str, torch.Tensor]:
    """Match the v21a backward order so the saved gradient hashes stay comparable."""
    page = state["page"].detach()
    with torch.no_grad():
        _tokens, kept = rt.v7d.generate_tokens(
            model, state["inputs"], rt.v7d.pixels_from_page(state["inputs"], item["aux_idx"], page), state["spans"], TRAIN_RATIO
        )
    cost["generate"] += 1
    kept = [int(index) for index in kept]
    delta = torch.zeros_like(page, requires_grad=True)
    pixels = rt.v7d.pixels_from_page(state["inputs"], item["aux_idx"], page + delta)
    keep_index = rt.v7d.keep_index_from_kept(kept, state["spans"], int(state["inputs"]["input_ids"].shape[1]), page.device)
    rows = rt.v7d.fastv_rows(model, state["inputs"], pixels, reference, keep_index)
    cost["forward"] += 1
    ordered = (
        ("CORE", fixed_noeos_loss(rows, reference, core_pos)),
        ("CONTEXT_UNUSED", fixed_noeos_loss(rows, reference, context_pos)),
        ("ALL", fixed_noeos_loss(rows, reference, all_pos)),
    )
    grads = {}
    for name, loss in ordered:
        if delta.grad is not None:
            delta.grad = None
        loss.backward(retain_graph=True)
        cost["backward"] += 1
        if name != "CONTEXT_UNUSED":
            grads[name] = delta.grad.detach().clone()
            if not torch.isfinite(grads[name]).all():
                write_block("BLOCKED_NONFINITE_GRADIENT", {"direction": name})
    del rows, ordered, delta
    return grads


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "runtime").mkdir(exist_ok=True)
    check_provenance()
    config = json.loads((HERE / "config.json").read_text())
    groups = json.loads((V21 / "frozen_token_groups.json").read_text())
    geometry = json.loads((V21A / "gradient_geometry.json").read_text())
    reference = [int(row["token_id"]) for row in groups["tokens"]]
    pieces = [row["decoded_piece"] for row in groups["tokens"]]
    core_pos = groups["s_core"]["positions"]
    all_pos = groups["s_all"]["positions"]
    if core_pos != config["s_core"] or all_pos != config["s_all"]:
        write_block("BLOCKED_CLEAN_REPLAY_DRIFT", {"reason": "token groups differ from config"})
    rt.configure()
    model, processor = rt.v7d.load_model_frozen()
    item = rt.v7d.load_dev("47450")
    state = rt.prepare_clean_state(model, processor, item)
    original = item["images"][item["aux_idx"]].convert("RGB")
    cost = {"forward": 0, "backward": 0, "generate": 0}
    clean = score_image(rt, model, processor, item, state, original, reference, pieces, core_pos, [], cost)
    clean["keep_jaccard"] = 1.0
    problems = []
    if state["clean_text"] != "Dr. or Mrs. William J. Darby" or clean["full_text"] != state["clean_text"]:
        problems.append("full_text")
    if clean["fast_text"] != "Dr. or Mrs. William J. Darby" or clean["fast_tokens"] != reference:
        problems.append("fast_tokens")
    if clean["k"] != 3699 or clean["keep_hash"] != EXPECTED_KEEP:
        problems.append("keep")
    if not (clean["full_text_exact"] and clean["full_token_exact"] and clean["full_replay"]):
        problems.append("full_replay")
    if problems:
        write_block("BLOCKED_CLEAN_REPLAY_DRIFT", {"problems": problems, "keep": clean["keep_hash"], "text": clean["fast_text"]})
    clean_kept = clean["keep_indices"]
    try:
        grads = attack_grads(model, state, item, reference, core_pos, groups["s_context"]["positions"], all_pos, cost)
    except torch.cuda.OutOfMemoryError:
        write_block("BLOCKED_OOM", {"stage": "gradient"})
    names = {"CORE": "g_core", "ALL": "g_all"}
    provenance = {"grads": {}, "cosine": cosine(grads["CORE"], grads["ALL"]), "sign_agreement": sign_agreement(grads["CORE"], grads["ALL"])}
    drift = []
    for direction, stored_name in names.items():
        packed = {
            "hash": rt.hash_tensor(grads[direction]),
            "l1": float(grads[direction].float().abs().sum()),
            "l2": float(grads[direction].float().norm()),
            "linf": float(grads[direction].float().abs().max()),
        }
        provenance["grads"][direction] = packed
        stored = geometry["grads"][stored_name]
        if packed["hash"] != stored["hash"]:
            drift.append({"direction": direction, "got": packed["hash"], "expected": stored["hash"]})
    stored_pair = geometry["pairs"]["g_core__g_all"]
    provenance["v21a_cosine"] = stored_pair["cosine"]
    provenance["v21a_sign_agreement"] = stored_pair["element_agreement"]
    rt.write_json(HERE / "gradient_provenance.json", provenance)
    if drift:
        write_block("BLOCKED_GRADIENT_PROVENANCE_DRIFT", {"drift": drift})
    clean_page = state["page"].detach()
    manifest = []
    for direction, grad in grads.items():
        for radius in RADII:
            started = time.perf_counter()
            point_cost = {"forward": 0, "backward": 0, "generate": 0}
            delta = torch.zeros_like(clean_page)
            if radius:
                sign_step(delta, clean_page, grad, radius / 255.0, radius / 255.0)
            image = rt.project_image(clean_page, delta)
            path = HERE / "scan" / direction / f"radius_{radius:02d}.png"
            png_sha = rt.save_png(image, path)
            reloaded = rt.reload_png(path)
            stats = image_stats(reloaded, original)
            try:
                scored = score_image(rt, model, processor, item, state, reloaded, reference, pieces, core_pos, clean_kept, point_cost)
            except torch.cuda.OutOfMemoryError:
                write_block("BLOCKED_OOM", {"direction": direction, "radius": radius})
            mismatch = bool(radius and stats["actual_uint8_linf"] != radius)
            core_mean = sum(row["hard_margin"] for row in scored["core_margins"]) / len(scored["core_margins"])
            clean_core_mean = sum(row["hard_margin"] for row in clean["core_margins"]) / len(clean["core_margins"])
            full_hard = [row["hard_margin"] for row in scored["full_margins"]]
            weakest = min(scored["full_margins"], key=lambda row: row["hard_margin"])
            record = {
                "direction": direction,
                "requested_uint8_linf": radius,
                "actual_uint8_linf": stats["actual_uint8_linf"],
                "quantization_radius_mismatch": mismatch,
                "radius_label": "QUANTIZATION_RADIUS_MISMATCH" if mismatch else "RADIUS_MATCH",
                "delta_hash": rt.hash_tensor(delta),
                "float_image_hash": rt.hash_tensor(clean_page + delta),
                "png_sha256": png_sha,
                **stats,
                **{key: value for key, value in scored.items() if key != "keep_indices"},
                "keep_indices": scored["keep_indices"],
                "full_hard_mean": sum(full_hard) / len(full_hard),
                "full_hard_min": min(full_hard),
                "full_soft_mean": sum(row["soft_margin"] for row in scored["full_margins"]) / len(scored["full_margins"]),
                "weakest_full_token": weakest,
                "core_hard_mean": core_mean,
                "core_hard_min": min(row["hard_margin"] for row in scored["core_margins"]),
                "core_name_intact": CORE_TEXT in scored["fast_text"],
                "semantic_status": classify("47450", scored["fast_text"], [])["semantic_status"],
                "core_margin_decreased": core_mean < clean_core_mean,
                "fastv_text_changed": scored["fast_text"] != clean["fast_text"],
                "clean_core_hard_mean": clean_core_mean,
                "forward": point_cost["forward"],
                "backward": point_cost["backward"],
                "generate": point_cost["generate"],
                "wall_seconds": time.perf_counter() - started,
            }
            for row, base in zip(record["full_margins"], clean["full_margins"]):
                row["hard_delta"] = row["hard_margin"] - base["hard_margin"]
                row["soft_delta"] = row["soft_margin"] - base["soft_margin"]
            for row, base in zip(record["core_margins"], clean["core_margins"]):
                row["hard_delta"] = row["hard_margin"] - base["hard_margin"]
                row["soft_delta"] = row["soft_margin"] - base["soft_margin"]
            rt.write_json(path.with_suffix(".json"), record)
            manifest.append({"direction": direction, "radius": radius, "png": str(path)})
            heartbeat({
                "direction": direction,
                "radius": radius,
                "png_sha256": png_sha,
                "actual_uint8_linf": stats["actual_uint8_linf"],
                "full_exact": bool(record["full_text_exact"] and record["full_token_exact"] and record["full_replay"]),
                "fast_text": record["fast_text"],
                "core_hard_mean": core_mean,
            })
            print(json.dumps(heartbeat_row(record)), flush=True)
            for key in cost:
                cost[key] += point_cost[key]
    rt.write_json(HERE / "scan" / "manifest.json", {"points": manifest, "cost": cost, "clean_keep_hash": EXPECTED_KEEP})
    print(json.dumps({"status": "SCAN_COMPLETE", "cost": cost}), flush=True)


def heartbeat_row(record: dict) -> dict:
    return {
        "direction": record["direction"],
        "radius": record["requested_uint8_linf"],
        "actual": record["actual_uint8_linf"],
        "full_exact": bool(record["full_text_exact"] and record["full_token_exact"] and record["full_replay"]),
        "text": record["fast_text"],
        "core_mean": record["core_hard_mean"],
    }


if __name__ == "__main__":
    main()
