"""Refuse FastV imports inside baseline, surrogate, and training processes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project/week2_attack_v9b_rank_signal_oblivious_admission")
DENY = {
    "adapters.runtime",
    "adapters.fastv_select",
    "week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation.native_grad",
    "week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation.run_v7d",
}
HITS = {"imports": 0, "names": []}


class IsolationFinder:
    def find_spec(self, fullname, path, target=None):
        lowered = fullname.lower()
        if "fastv" in lowered or fullname in DENY:
            HITS["imports"] += 1
            HITS["names"].append(fullname)
            write_violation(f"import blocked: {fullname}")
            raise ImportError(fullname)
        return None


def write_violation(reason: str) -> None:
    payload = {
        "decision": "THREAT_MODEL_VIOLATION",
        "reason": reason,
        "hits": HITS,
        "loaded": sorted(name for name in sys.modules if "fastv" in name.lower()),
    }
    (HERE / "final_decision.json").write_text(json.dumps(payload, indent=2) + "\n")
    (HERE / "threat_violation.json").write_text(json.dumps(payload, indent=2) + "\n")


def install_guard() -> None:
    if not any(isinstance(finder, IsolationFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, IsolationFinder())


def assert_clean(stage: str) -> None:
    loaded = sorted(name for name in sys.modules if "fastv" in name.lower())
    if HITS["imports"] or loaded:
        write_violation(f"{stage} loaded FastV: {loaded or HITS['names']}")
        raise SystemExit(2)
