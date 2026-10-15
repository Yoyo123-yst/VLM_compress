"""Write the frozen config and development manifest before any attack."""

from __future__ import annotations

import sys

sys.path[:0] = ["/home/yuesiting/autodl-tmp/multimodal_attack_project"]

import hashlib
import json
from pathlib import Path

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
HERE = ROOT / "week2_attack_v20_fourway_content_ablation"

from week2_attack_v20_fourway_content_ablation.protocol import (  # noqa: E402
    CHECKPOINTS,
    DEV12,
    EPS,
    EPSILON_UINT8,
    EVAL_RATIOS,
    GPU_BY_METHOD,
    LAMBDA_FULL,
    MAX_ACCEPTED,
    MAX_ATTEMPTS,
    METHODS,
    NOISE_REPLICATES,
    OPTIMIZER,
    PRIMARY_SAMPLES,
    SEED,
    SMOKE_MAX_ACCEPTED,
    SMOKE_SAMPLES,
    STEP_ALPHA,
    TRAIN_RATIO,
)

CONFIG = {
    "tag_full": "full_dev_fourway_v1",
    "optimizer": OPTIMIZER,
    "step_alpha": STEP_ALPHA,
    "step_alpha_note": "Frozen before smoke. Smoke does not retune alpha from content outcomes.",
    "epsilon": EPS,
    "epsilon_uint8": EPSILON_UINT8,
    "lambda_full": LAMBDA_FULL,
    "train_ratio": TRAIN_RATIO,
    "eval_ratios": list(EVAL_RATIOS),
    "eval_ratio_note": "Only 0.25 trains. Other ratios are transfer diagnostics and cannot select a checkpoint.",
    "max_accepted": MAX_ACCEPTED,
    "max_attempts": MAX_ATTEMPTS,
    "checkpoints": list(CHECKPOINTS),
    "seed": SEED,
    "methods": METHODS,
    "gpu_by_method": GPU_BY_METHOD,
    "selector": "lowest attack surrogate among legal PNG checkpoints; earlier checkpoint wins ties",
    "selector_forbidden_inputs": ["ANLS", "CONTENT_FAILURE", "gold semantic label", "attack success"],
    "smoke_samples": list(SMOKE_SAMPLES),
    "smoke_max_accepted": SMOKE_MAX_ACCEPTED,
    "noise_replicates": NOISE_REPLICATES,
    "unseen": "NOT_OPENED",
}


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "config.json").write_text(json.dumps(CONFIG, indent=2) + "\n")
    source = ROOT / "week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation" / "DEV_MANIFEST_COPY.json"
    manifest = {
        "source": str(source.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "dev12_frozen_order": list(DEV12),
        "primary6_in_frozen_order": list(PRIMARY_SAMPLES),
        "extension_to_12": "NOT_RUN",
        "unseen": "NOT_OPENED",
    }
    (HERE / "development_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    digest = hashlib.sha256((HERE / "config.json").read_bytes()).hexdigest()
    sealed = HERE / "sealed"
    sealed.mkdir(exist_ok=True)
    (sealed / "config.sha256").write_text(digest + "\n")
    print(json.dumps({"config_sha256": digest, "primary": list(PRIMARY_SAMPLES)}))


if __name__ == "__main__":
    main()
