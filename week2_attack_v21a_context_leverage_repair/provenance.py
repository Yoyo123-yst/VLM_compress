"""Hash v20 and v21 before and after this repair. Both trees stay read-only."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
HERE = ROOT / "week2_attack_v21a_context_leverage_repair"
V20 = ROOT / "week2_attack_v20_fourway_content_ablation"
V21 = ROOT / "week2_attack_v21_context_leverage_causal_probe"


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def v20_files() -> list[Path]:
    files = sorted(V20.glob("*.py"))
    files += [
        V20 / "config.json",
        V20 / "semantic_manifest.json",
        V20 / "admission.json",
        V20 / "full_dev_fourway_v1" / "per_run" / "47450" / "C.json",
        V20 / "full_dev_fourway_v1" / "per_run" / "47450" / "D.json",
        V20 / "independent_eval" / "47450" / "C.json",
        V20 / "independent_eval" / "47450" / "D.json",
    ]
    return files


def v21_files() -> list[Path]:
    names = [
        "config.json",
        "frozen_token_groups.json",
        "random_controls.json",
        "gradient_diagnostics.json",
        "cross_effect_matrix.csv",
        "local_step_results.json",
        "phase1_decision.json",
        "phase1_report.md",
        "final_decision.json",
        "final_report.md",
    ]
    return sorted(V21.glob("*.py")) + [V21 / name for name in names]


def collect() -> dict:
    files = {}
    for path in v20_files() + v21_files():
        files[str(path.relative_to(ROOT))] = sha_file(path)
    return files


def write(label: str) -> None:
    payload = {"label": label, "files": collect()}
    if label == "after":
        before = json.loads((HERE / "sealed_hashes_before.json").read_text())
        payload["matches_before"] = payload["files"] == before["files"]
        if not payload["matches_before"]:
            raise SystemExit("v20 or v21 hash changed")
    (HERE / f"sealed_hashes_{label}.json").write_text(json.dumps(payload, indent=2) + "\n")


def environment() -> dict:
    import torch
    import transformers
    from week1_mechanism.data_mpdoc import parquet_dir
    from gate_experiment.run_causal_v2 import LLAVA
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,driver_version", "--format=csv,noheader"],
        text=True,
    )
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu_table": gpu.strip().splitlines(),
        "determinism": {
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
        "model_checkpoint": LLAVA,
        "parquet_dir": str(parquet_dir()),
    }


if __name__ == "__main__":
    import sys
    sys.path[:0] = [str(ROOT), str(ROOT / "week1_mechanism"), str(ROOT / "gate_experiment")]
    label = sys.argv[1]
    if label == "before":
        env = environment()
        from week2_attack_v20_fourway_content_ablation import runtime_utils as rt
        item = rt.v7d.load_dev("47450")
        image = item["images"][item["aux_idx"]].convert("RGB")
        env["sample_47450"] = {
            "shard": item["meta"]["shard"],
            "row": item["meta"]["row"],
            "doc_id": item["meta"]["doc_id"],
            "page_ids": item["meta"]["page_ids"],
            "order": item["order"],
            "aux_idx": item["aux_idx"],
            "evidence_idx": item["evidence_idx"],
            "mode": image.mode,
            "width": image.width,
            "height": image.height,
            "note": "The auxiliary page is a 768 RGB resize of a parquet image cell. It has no standalone file path.",
        }
        (HERE / "provenance.json").write_text(json.dumps({"environment": env, "hashes": "sealed_hashes_before.json"}, indent=2) + "\n")
    write(label)
