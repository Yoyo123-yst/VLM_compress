"""Aggregate frozen holdout FastV shards and compute predeclared counts."""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project/week2_attack_v11c_deterministic_native_fdp")


def main() -> None:
    frozen = json.loads((HERE / "holdout_ids.json").read_text())
    merged = {"status": "TRANSFER_EVALUATED", "protocol": "frozen holdout aggregate", "samples": {}}
    for path in sorted((HERE / "holdout_transfer" / "shards").glob("*.json")):
        shard = json.loads(path.read_text())
        merged["samples"].update(shard.get("samples", {}))
    rows = [item for item in merged["samples"].values() if item.get("status") == "OK"]
    cells = [cell for item in rows for cell in item["ratios"].values()]
    merged["summary"] = {
        "frozen_samples": len(frozen["ids"]),
        "samples_measured": len(rows),
        "cells": len(cells),
        "all_replay_pass": all(all(item["replay"].values()) for item in rows),
        "induced_cells": sum(int(cell["induced"]) for cell in cells),
        "attack_strict_cells": sum(int(cell["attack_strict"]) for cell in cells),
        "noise_strict_cells": sum(int(cell["noise_strict"]) for cell in cells),
        "induced_beats_noise_cells": sum(int(cell["induced"] and cell["beats_noise"]) for cell in cells),
        "attack_beats_noise_cells": sum(int(cell["beats_noise"]) for cell in cells),
    }
    out = HERE / "holdout_transfer" / "final_report.json"
    out.write_text(json.dumps(merged, indent=2) + "\n")
    print(json.dumps(merged["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
