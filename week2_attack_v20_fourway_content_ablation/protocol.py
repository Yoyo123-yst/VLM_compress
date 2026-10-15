"""Frozen v20 development-ablation constants.

This file is the pre-registered protocol. Smoke may stop the run. It may not
retune epsilon, the step size, the selector, or the semantic rules from
content-failure outcomes.
"""

from __future__ import annotations

EPS = 8 / 255
EPSILON_UINT8 = 8
TRAIN_RATIO = 0.25
EVAL_RATIOS = (0.15, 0.25, 0.35, 0.50)
LAMBDA_FULL = 2.6371882367154997
SEED = 2026101102
STEP_ALPHA = 1.0 / 255.0
OPTIMIZER = "sign_pgd"
MAX_ACCEPTED = 25
MAX_ATTEMPTS = 40
CHECKPOINTS = (0, 1, 5, 10, 15, 20, 25)
SMOKE_MAX_ACCEPTED = 4
SMOKE_MAX_ATTEMPTS = 8
SMOKE_SAMPLES = ("60700", "47450")
PRIMARY_SAMPLES = ("1025", "2008", "7006", "39204", "47450", "60700")
DEV12 = (
    "1025",
    "2008",
    "7006",
    "39204",
    "47450",
    "47567",
    "50779",
    "51133",
    "60700",
    "6194",
    "62941",
    "64506",
)
METHODS = {
    "A": "dynamic_sequence",
    "B": "fixed_all",
    "C": "fixed_noeos",
    "D": "core_content",
}
GPU_BY_METHOD = {"A": "0", "B": "1", "C": "4", "D": "6"}
ALLOWED_GPUS = ("0", "1", "4", "6")
NOISE_REPLICATES = 3
NOISE_ATTEMPTS = 30
TAG_FULL = "full_dev_fourway_v1"
