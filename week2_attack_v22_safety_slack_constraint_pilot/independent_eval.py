"""Fresh process evaluation of the selector PNGs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
sys.path[:0] = [str(ROOT), str(ROOT / "week1_mechanism"), str(ROOT / "gate_experiment")]
HERE = ROOT / "week2_attack_v22_safety_slack_constraint_pilot"

from metrics_task import anls_score  # noqa: E402
from week2_attack_v20_fourway_content_ablation import runtime_utils as rt  # noqa: E402
from week2_attack_v20_fourway_content_ablation.semantics import classify  # noqa: E402
from week2_attack_v22_safety_slack_constraint_pilot.runtime import load_groups, reference_of, score  # noqa: E402

RATIOS = (0.15, 0.25, 0.35, 0.50)


def main() -> None:
    groups = load_groups()
    reference, pieces, core_pos = reference_of(groups)
    rt.configure()
    model, processor = rt.v7d.load_model_frozen()
    item = rt.v7d.load_dev("47450")
    state = rt.prepare_clean_state(model, processor, item)
    original = item["images"][item["aux_idx"]].convert("RGB")
    clean_cost = {"forward": 0, "backward": 0, "generate": 0}
    clean = score(model, processor, item, state, original, reference, pieces, core_pos, [], clean_cost)
    results = {}
    diverged = []
    for name in ("T0", "T1", "T2"):
        selector = json.loads((HERE / "trajectories" / name / "selector.json").read_text())
        step = int(selector["chosen"]["step"])
        saved = json.loads((HERE / "trajectories" / name / f"step_{step:02d}.json").read_text())
        image = rt.reload_png(HERE / "sealed_png" / f"{name}_step_{step:02d}.png")
        ratios = {}
        for ratio in RATIOS:
            cost = {"forward": 0, "backward": 0, "generate": 0}
            if ratio == 0.25:
                scored = score(model, processor, item, state, image, reference, pieces, core_pos, clean["keep_indices"], cost)
                fast_text, fast_tokens, k, keep_hash = scored["fast_text"], scored["fast_tokens"], scored["k"], scored["keep_hash"]
                full = scored
            else:
                attacked = list(item["images"])
                attacked[item["aux_idx"]] = image
                inputs = rt.v7d.prepare(processor, attacked, item["meta"]["question"])
                spans = rt.v7d.spans_of(inputs, int(model.config.image_token_index), len(attacked))
                tokens, kept = rt.v7d.generate_tokens(model, inputs, inputs["pixel_values"], spans, ratio)
                fast_tokens = [int(token) for token in tokens]
                fast_text = rt.v7d.decode(processor, torch.tensor(fast_tokens)).strip()
                kept = [int(index) for index in kept]
                k = len(kept)
                keep_hash = rt.keep_hash(kept)
                full = None
            semantic = classify("47450", fast_text, [])
            ratios[f"{ratio:.2f}"] = {
                "fast_text": fast_text,
                "fast_tokens": fast_tokens,
                "anls": float(anls_score(fast_text, list(item["meta"]["answers"]))),
                "semantic_status": semantic["semantic_status"],
                "k": k,
                "keep_hash": keep_hash,
                "valid": bool(fast_text.strip()),
            }
            if full is not None:
                ratios[f"{ratio:.2f}"].update({
                    "full_text": full["full_text"],
                    "full_tokens": full["full_tokens"],
                    "full_text_exact": full["full_text_exact"],
                    "full_token_exact": full["full_token_exact"],
                    "full_replay": full["full_replay"],
                    "full_output_valid": full["full_output_valid"],
                    "core_hard_mean": sum(row["hard_margin"] for row in full["core_margins"]) / len(full["core_margins"]),
                    "core_margins": full["core_margins"],
                    "full_hard_min": min(row["hard_margin"] for row in full["full_margins"]),
                    "uint8_linf": rt.uint8_linf(original, image),
                    "png_sha256": rt.sha_file(HERE / "sealed_png" / f"{name}_step_{step:02d}.png"),
                })
                if full["keep_hash"] != saved["keep_hash"] or full["fast_text"] != saved["fast_text"] or full["full_tokens"] != saved["full_tokens"]:
                    diverged.append(name)
        results[name] = {"selector": selector["chosen"], "ratios": ratios, "saved_step": step}
        print(json.dumps({"trajectory": name, "step": step, "r025": ratios["0.25"]["semantic_status"], "text": ratios["0.25"]["fast_text"]}), flush=True)
    payload = {"match": not diverged, "diverged": diverged, "results": results, "t3": "NOT_RUN_UNVERIFIED_CONSTRAINED_BASELINE"}
    rt.write_json(HERE / "independent_eval.json", payload)
    if diverged:
        rt.write_json(HERE / "final_decision.json", {"status": "BLOCKED_REPLAY_DIVERGED", "diverged": diverged, "second_sample_authorized": False})
        raise SystemExit(2)


if __name__ == "__main__":
    main()
