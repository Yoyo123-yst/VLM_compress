"""Reproduce the clean page and the v21b CORE L∞=8 PNG before any training."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
sys.path[:0] = [str(ROOT), str(ROOT / "week1_mechanism"), str(ROOT / "gate_experiment")]
HERE = ROOT / "week2_attack_v22_safety_slack_constraint_pilot"

from week2_attack_v20_fourway_content_ablation import runtime_utils as rt  # noqa: E402
from week2_attack_v22_safety_slack_constraint_pilot.objectives import safety_tau  # noqa: E402
from week2_attack_v22_safety_slack_constraint_pilot.runtime import load_groups, read_expected, reference_of, score  # noqa: E402


def fail(detail: dict) -> None:
    rt.write_json(HERE / "preflight.json", {"status": "BLOCKED_REPLAY_DIVERGED", "detail": detail})
    rt.write_json(HERE / "final_decision.json", {
        "status": "BLOCKED_REPLAY_DIVERGED",
        "second_sample_authorized": False,
    })
    (HERE / "final_report.md").write_text("# 约束比较试点\n\n状态：`BLOCKED_REPLAY_DIVERGED`\n\n预训练检查没有通过，因此没有开始轨迹。\n")
    raise SystemExit(2)


def main() -> None:
    groups = load_groups()
    reference, pieces, core_pos = reference_of(groups)
    rt.configure()
    model, processor = rt.v7d.load_model_frozen()
    item = rt.v7d.load_dev("47450")
    state = rt.prepare_clean_state(model, processor, item)
    original = item["images"][item["aux_idx"]].convert("RGB")
    cost = {"forward": 0, "backward": 0, "generate": 0}
    clean = score(model, processor, item, state, original, reference, pieces, core_pos, [], cost)
    expected_clean = read_expected("radius_00.json")
    clean_problems = []
    for field in ("full_text", "full_tokens", "fast_text", "fast_tokens", "k", "keep_hash", "processor_exact_hash", "core_margin_hash"):
        if clean[field] != expected_clean[field]:
            clean_problems.append(field)
    image = rt.reload_png(ROOT / "week2_attack_v21b_full_safety_slack_scan" / "scan" / "CORE" / "radius_08.png")
    attacked = score(model, processor, item, state, image, reference, pieces, core_pos, clean["keep_indices"], cost)
    expected = read_expected("radius_08.json")
    problems = []
    if rt.uint8_linf(original, image) != 8:
        problems.append("linf")
    for field in ("full_text", "full_tokens", "keep_hash", "core_margin_hash", "processor_exact_hash"):
        if attacked[field] != expected[field]:
            problems.append(field)
    if not (attacked["full_text_exact"] and attacked["full_token_exact"] and attacked["full_replay"]):
        problems.append("full_exact")
    if clean_problems or problems:
        fail({"clean": clean_problems, "radius8": problems})
    clean_hard = torch.tensor([row["hard_margin"] for row in clean["full_margins"]], dtype=torch.float32)
    tau = safety_tau(clean_hard)
    payload = {
        "status": "PASS",
        "clean_full_text": clean["full_text"],
        "clean_fast_text": clean["fast_text"],
        "k": clean["k"],
        "keep_hash": clean["keep_hash"],
        "processor_exact_hash": clean["processor_exact_hash"],
        "core_positions": core_pos,
        "core_token_ids": [reference[index] for index in core_pos],
        "clean_full_hard_margins": [row["hard_margin"] for row in clean["full_margins"]],
        "tau": [float(value) for value in tau],
        "radius8_keep_hash": attacked["keep_hash"],
        "radius8_core_margin_hash": attacked["core_margin_hash"],
        "radius8_core_hard_mean": sum(row["hard_margin"] for row in attacked["core_margins"]) / len(core_pos),
    }
    rt.write_json(HERE / "preflight.json", payload)
    print(json.dumps({"status": "PASS", "tau0": payload["tau"][0], "core_mean": payload["radius8_core_hard_mean"]}), flush=True)


if __name__ == "__main__":
    import torch
    main()
