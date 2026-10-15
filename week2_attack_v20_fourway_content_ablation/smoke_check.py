"""Smoke-gate audit. Any failed check blocks the formal run."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
HERE = ROOT / "week2_attack_v20_fourway_content_ablation"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    smoke = HERE / "smoke"
    problems = []
    runs = []
    for sample in ("60700", "47450"):
        for method in ("A", "B", "C", "D"):
            path = smoke / "per_run" / sample / f"{method}.json"
            if not path.exists():
                problems.append(f"missing {sample} {method}")
                continue
            payload = load(path)
            runs.append(payload)
            if payload.get("status") not in {"DONE", "CORE_SPAN_UNAVAILABLE"}:
                problems.append(f"{sample} {method} status {payload.get('status')}")
                continue
            if payload.get("status") == "CORE_SPAN_UNAVAILABLE":
                if payload.get("selection_d", {}).get("positions"):
                    problems.append(f"{sample} D unavailable but has positions")
                continue
            text = json.dumps(payload)
            if "NaN" in text or "Infinity" in text:
                problems.append(f"{sample} {method} non-finite")
            for update in payload.get("updates", []):
                if not (update["full_text_exact"] and update["full_token_exact"] and update["full_replay"]):
                    if update["accepted"]:
                        problems.append(f"{sample} {method} accepted a Full failure")
                if update["uint8_linf"] > 8:
                    problems.append(f"{sample} {method} linf {update['uint8_linf']}")
            selected_b = payload["selection_b"]
            selected_c = payload["selection_c"]
            selected_d = payload["selection_d"]
            if not any(piece.strip() in {"<|im_end|>", "</s>", "<eos>"} or "im_end" in piece for piece in selected_b["pieces"]):
                # EOS may decode to a special string. Fall back to the id gap between B and C.
                if set(selected_c["positions"]) >= set(selected_b["positions"]):
                    problems.append(f"{sample} B did not contain a token excluded by C")
            if set(selected_c["token_ids"]) & set(selected_b["token_ids"]) == set(selected_b["token_ids"]):
                problems.append(f"{sample} C did not remove any B token")
            removed = [token for token in selected_b["token_ids"] if token not in selected_c["token_ids"]]
            if not removed:
                problems.append(f"{sample} C excluded nothing")
            if selected_d["available"] and selected_d["positions"] == selected_c["positions"] and not selected_d["same_positions_as_noeos"]:
                problems.append(f"{sample} D matches C without the equality flag")
            if not selected_d["available"] and selected_d["positions"]:
                problems.append(f"{sample} D fell back to a non-empty set")
            if method == "A":
                for update in payload["updates"]:
                    dynamic = update["dynamic_selection"]
                    if dynamic["reason"] != "current_perturbed_natural_fastv_sequence":
                        problems.append(f"{sample} A reason drifted")
                    if dynamic["token_ids"] != update["fastv_tokens"]:
                        problems.append(f"{sample} A did not train on the natural tokens")
            if method in {"B", "C", "D"}:
                for update in payload["updates"]:
                    if update["selected_token_ids"] != [payload["frozen_reference_ids"][index] for index in update["positions"]]:
                        problems.append(f"{sample} {method} selected ids left the frozen reference")
    repeat = smoke / "repeat" / "per_run" / "60700" / "A.json"
    original = smoke / "per_run" / "60700" / "A.json"
    if not repeat.exists() or not original.exists():
        problems.append("repeat trajectory missing")
    else:
        left = load(original)
        right = load(repeat)
        if left.get("status") == "DONE" and right.get("status") == "DONE":
            if [row["grad_hash"] for row in left["updates"]] != [row["grad_hash"] for row in right["updates"]]:
                problems.append("repeat gradient hash mismatch")
            if [row["png_sha256"] for row in left["checkpoints"]] != [row["png_sha256"] for row in right["checkpoints"]]:
                problems.append("repeat png hash mismatch")
            if [row["keep_hash"] for row in left["checkpoints"]] != [row["keep_hash"] for row in right["checkpoints"]]:
                problems.append("repeat keep-set hash mismatch")
            if left.get("fixed_selector") != right.get("fixed_selector"):
                problems.append("repeat selector mismatch")
            left_text = [row["fastv_text"] for row in left["checkpoints"]]
            right_text = [row["fastv_text"] for row in right["checkpoints"]]
            if left_text != right_text:
                problems.append("repeat text mismatch")
    eval_path = HERE / "smoke" / "independent_eval" / "60700" / "A.json"
    if not eval_path.exists():
        problems.append("independent evaluator missing")
    elif original.exists():
        official = load(eval_path)
        trained = load(original)
        trained_by = {row["checkpoint"]: row for row in trained.get("checkpoints", [])}
        official_by = {row["checkpoint"]: row for row in official.get("checkpoints", [])}
        for checkpoint, other in trained_by.items():
            row = official_by.get(checkpoint)
            if row is None:
                problems.append(f"evaluator missed training checkpoint {checkpoint}")
                continue
            if row["png_sha256"] != other["png_sha256"]:
                problems.append("evaluator read a different png")
            if row["ratios"]["0.25"]["text"] != other["fastv_text"]:
                problems.append("evaluator text disagrees with the sealed png replay")
            if row["full_text_exact"] != other["full_text_exact"] or row["full_replay"] != other["full_replay"]:
                problems.append("evaluator Full gate disagrees")
    status = "PASS" if not problems and len(runs) == 8 else "BLOCKED_SMOKE_GATE"
    report = {"status": status, "runs": len(runs), "problems": problems}
    (HERE / "smoke" / "smoke_gate.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
