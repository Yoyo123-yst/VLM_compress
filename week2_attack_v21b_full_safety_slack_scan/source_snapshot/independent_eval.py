"""Fresh process replay of every sealed PNG. No gradient and no radius selection."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
sys.path[:0] = [str(ROOT), str(ROOT / "week1_mechanism"), str(ROOT / "gate_experiment")]
HERE = ROOT / "week2_attack_v21b_full_safety_slack_scan"
V21 = ROOT / "week2_attack_v21_context_leverage_causal_probe"

from week2_attack_v20_fourway_content_ablation import runtime_utils as rt  # noqa: E402
from week2_attack_v21b_full_safety_slack_scan.measure import score_image  # noqa: E402

FIELDS = ("full_text", "full_tokens", "k", "keep_hash", "fast_text", "fast_tokens", "full_margin_hash", "core_margin_hash", "processor_exact_hash")


def main() -> None:
    groups = json.loads((V21 / "frozen_token_groups.json").read_text())
    reference = [int(row["token_id"]) for row in groups["tokens"]]
    pieces = [row["decoded_piece"] for row in groups["tokens"]]
    core_pos = groups["s_core"]["positions"]
    rt.configure()
    model, processor = rt.v7d.load_model_frozen()
    item = rt.v7d.load_dev("47450")
    state = rt.prepare_clean_state(model, processor, item)
    original = item["images"][item["aux_idx"]].convert("RGB")
    clean_cost = {"forward": 0, "backward": 0, "generate": 0}
    clean = score_image(rt, model, processor, item, state, original, reference, pieces, core_pos, [], clean_cost)
    diverged = []
    rows = []
    for path in sorted((HERE / "scan").glob("*/*.json")):
        saved = json.loads(path.read_text())
        image = rt.reload_png(Path(saved["png"]) if "png" in saved else path.with_suffix(".png"))
        if "png_sha256" in saved:
            image = rt.reload_png(path.with_suffix(".png"))
        cost = {"forward": 0, "backward": 0, "generate": 0}
        scored = score_image(rt, model, processor, item, state, image, reference, pieces, core_pos, clean["keep_indices"], cost)
        mismatch = {field: {"scan": saved.get(field), "replay": scored.get(field)} for field in FIELDS if saved.get(field) != scored.get(field)}
        row = {
            "direction": saved["direction"],
            "requested_uint8_linf": saved["requested_uint8_linf"],
            "match": not mismatch,
            "mismatch": mismatch,
            "replay_full_text": scored["full_text"],
            "replay_fast_text": scored["fast_text"],
            "replay_keep_hash": scored["keep_hash"],
            "replay_full_margin_hash": scored["full_margin_hash"],
            "replay_core_margin_hash": scored["core_margin_hash"],
        }
        rows.append(row)
        rt.write_json(HERE / "independent_eval" / f"{saved['direction']}_radius_{saved['requested_uint8_linf']:02d}.json", row)
        print(json.dumps({"direction": saved["direction"], "radius": saved["requested_uint8_linf"], "match": row["match"]}), flush=True)
        if mismatch:
            diverged.append(row)
    rt.write_json(HERE / "independent_eval" / "comparison.json", {"match": not diverged, "rows": rows})
    if diverged:
        rt.write_json(HERE / "final_decision.json", {
            "status": "BLOCKED_SCAN_REPLAY_DIVERGED",
            "next_step": "NO_MULTISTEP_COMPARISON_AUTHORIZED",
            "comparison_started": False,
            "n_diverged": len(diverged),
        })
        (HERE / "final_report.md").write_text("# Full 安全余量扫描\n\n状态：`BLOCKED_SCAN_REPLAY_DIVERGED`\n\n独立复评与扫描记录不一致，因此不解释安全余量。\n")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
