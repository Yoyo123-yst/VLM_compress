"""Hash the read-only experiments before and after this pilot."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
HERE = ROOT / "week2_attack_v22_safety_slack_constraint_pilot"
V21A = ROOT / "week2_attack_v21a_context_leverage_repair"
V21B = ROOT / "week2_attack_v21b_full_safety_slack_scan"


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def targets() -> list[Path]:
    sealed = json.loads((V21A / "sealed_hashes_after.json").read_text())
    files = [ROOT / rel for rel in sealed["files"]]
    files += sorted(V21A.glob("*.py"))
    files += [V21A / name for name in ("config.json", "gradient_geometry.json", "final_decision.json", "final_report.md", "phase_a_decision.json")]
    files += sorted(V21B.glob("*.py"))
    files += [V21B / name for name in ("config.json", "final_decision.json", "final_report.md", "safety_summary.json")]
    files += [
        V21B / "scan" / "CORE" / "radius_00.json",
        V21B / "scan" / "CORE" / "radius_08.json",
        V21B / "scan" / "CORE" / "radius_08.png",
    ]
    return files


def collect() -> dict:
    return {str(path.relative_to(ROOT)): sha_file(path) for path in targets()}


def main() -> None:
    label = sys.argv[1]
    current = collect()
    if label == "before":
        import torch
        import transformers
        gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.used", "--format=csv,noheader"], text=True)
        manifest = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "cudnn": int(torch.backends.cudnn.version()),
            "gpu": gpu.strip().splitlines(),
            "seed": 2026101102,
            "determinism_env": "CUBLAS_WORKSPACE_CONFIG=:4096:8",
            "config": json.loads((HERE / "config.json").read_text()),
            "t3": "NOT_RUN_UNVERIFIED_CONSTRAINED_BASELINE",
            "reason_t3": "No checked-in Orthogonal PGD or ALMA implementation has a verified projection and rollback test.",
        }
        (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (HERE / "sealed_hashes_before.json").write_text(json.dumps(current, indent=2) + "\n")
        return
    before = json.loads((HERE / "sealed_hashes_before.json").read_text())
    mismatch = [rel for rel, digest in before.items() if current.get(rel) != digest]
    (HERE / "sealed_hashes_after.json").write_text(json.dumps({"matches_before": not mismatch, "mismatch": mismatch}, indent=2) + "\n")
    if mismatch:
        (HERE / "final_decision.json").write_text(json.dumps({
            "status": "PROVENANCE_MUTATED",
            "files": mismatch,
            "second_sample_authorized": False,
        }, indent=2) + "\n")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
