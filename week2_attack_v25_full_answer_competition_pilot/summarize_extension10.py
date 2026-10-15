"""Aggregate the fixed-selector semantic endpoint comparison without model execution."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE / "iterative_v3_extension10"
NOISE = Path(__file__).resolve().parents[1] / "week2_attack_v20_fourway_content_ablation" / "noise_controls"
SAMPLES = ("47450", "7006")
METHODS = ("native_fdp", "complete_target")


def endpoint_status(text: str, clean: str, wrong: str) -> str:
    if text == wrong:
        return "CONTENT_FAILURE"
    if text == clean:
        return "ANSWER_PRESERVED"
    return "OTHER_OUTPUT_REQUIRES_REVIEW"


def main() -> None:
    rows = []
    for sample in SAMPLES:
        for method in METHODS:
            path = ROOT / method / sample / "result.json"
            data = json.loads(path.read_text())
            assert data["status"] == "DONE" and data["accepted"] == 10
            clean = data["clean_fast_text"]
            wrong = data["wrong_sequence"]
            bad = []
            for update in data["updates"]:
                guard = update["guard"]
                if not (
                    update["accepted"]
                    and guard["full_text_exact"]
                    and guard["full_token_exact"]
                    and guard["full_replay"]
                    and guard["fast_replay"]
                    and update["uint8_linf"] <= 8
                ):
                    bad.append(update["attempt"])
            assert not bad, (sample, method, bad)
            end = data["updates"][-1]
            text = end["guard"]["fast_text"]
            rows.append({
                "sample_id": sample,
                "method": method,
                "clean_fast_text": clean,
                "wrong_target_text": wrong,
                "endpoint_fast_text": text,
                "semantic_status": endpoint_status(text, clean, wrong),
                "full_text_exact": end["guard"]["full_text_exact"],
                "full_token_exact": end["guard"]["full_token_exact"],
                "full_replay": end["guard"]["full_replay"],
                "fast_replay": end["guard"]["fast_replay"],
                "uint8_linf": end["uint8_linf"],
                "png_sha256": end["png_sha256"],
                "all_updates_guarded": True,
            })
        noise = json.loads((NOISE / sample / "noise.json").read_text())
        for rep in noise["replicates"]:
            r25 = rep["ratios"]["0.25"]
            rows.append({
                "sample_id": sample,
                "method": "matched_random_noise",
                "replicate": rep["replicate"],
                "endpoint_fast_text": r25["text"],
                "semantic_status": r25["semantic_status"],
                "full_text_exact": rep["full_text_exact"],
                "full_token_exact": rep["full_token_exact"],
                "full_replay": rep["full_replay"],
                "uint8_linf": rep["uint8_linf"],
                "png_sha256": rep["png_sha256"],
            })
    payload = {
        "status": "DONE",
        "scope": "fixed_selector_equal_budget_endpoint_comparison",
        "ratio": 0.25,
        "accepted_updates": 10,
        "rows": rows,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    out = HERE / "diagnostics" / "extension10_semantic_comparison.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": payload["status"], "rows": len(rows), "out": str(out)}))


if __name__ == "__main__":
    main()
