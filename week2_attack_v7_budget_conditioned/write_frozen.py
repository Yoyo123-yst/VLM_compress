"""Write FROZEN_V7_PROTOCOL.json once. Later calls only verify it."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path

from week2_attack_v7_budget_conditioned.constants import (
    ADAM_LR,
    ATTACK_SEED_SALT,
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    CANDIDATE_R,
    DEV12,
    DEV_CHECKPOINTS,
    EPS,
    EPSILON_MARGIN,
    FINAL_SEED,
    GAMMA,
    GIT_COMMIT,
    HEALTH_ID,
    KAPPA_COMPRESSED,
    LAMBDA_SOFT_B,
    MAX_NEW_TOKENS,
    MODEL_PATH,
    REPEAT_SEED,
    SCREEN_SEED,
    STEPS,
    TAU_R,
    TAU_VOCAB,
    XI,
)

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
HERE = ROOT / "week2_attack_v7_budget_conditioned"
DEST = HERE / "FROZEN_V7_PROTOCOL.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    before = json.loads((HERE / "parent_raw_sha256_before.json").read_text())
    manifests = {
        "week2_attack_v7_pristine60/frozen_manifest.json": sha(ROOT / "week2_attack_v7_pristine60/frozen_manifest.json"),
        "week2_attack_v7_pristine60/FROZEN.json": sha(ROOT / "week2_attack_v7_pristine60/FROZEN.json"),
        "week2_attack_v5b_expanded_pristine40/frozen_manifest.json": sha(ROOT / "week2_attack_v5b_expanded_pristine40/frozen_manifest.json"),
        "parent_raw_sha256_before.json": sha(HERE / "parent_raw_sha256_before.json"),
    }
    env = subprocess.check_output(
        ["/home/yuesiting/autodl-tmp/conda/envs/vattack/bin/python", "-c",
         "import sys,torch; print(sys.version); print(torch.__version__); print(torch.version.cuda)"],
        text=True,
    )
    payload = {
        "protocol": "week2_attack_v7_budget_conditioned",
        "parent_protocols": [
            "week2_attack_v7_pristine60/FROZEN.json",
            "week2_attack_v6c_split_calibration/FROZEN_PROTOCOL.json",
            "week2_attack_v6b_soft_full_anchor/FROZEN_PROTOCOL.json",
        ],
        "parent_raw_sha256": before["sha256"],
        "parent_raw_n": before["n"],
        "manifest_sha256": manifests,
        "model_path": MODEL_PATH,
        "model_checkpoint": "llava-onevision-qwen2-7b-ov-hf weights as stored; no finetune",
        "data_source": "MP-DocVQA val, same page loader as week2_attack_v7_pristine60",
        "environment": {"python_torch": env, "platform": platform.platform()},
        "git_commit": GIT_COMMIT,
        "candidate_R": list(CANDIDATE_R),
        "active_R_rule": (
            "Health on sample 1025 tries every candidate r in the real FastV hook. "
            "An r that raises or returns K outside 1..n_visual is removed from every later stage. "
            "No replacement r is added. The removed set is written to ratio_legality.json."
        ),
        "full_path_name": "infinity",
        "decoding": {
            "greedy": True,
            "do_sample": False,
            "temperature": 0,
            "max_new_tokens": MAX_NEW_TOKENS,
            "use_cache": True,
            "eos": "model.generation_config.eos_token_id, appended when generate does not already end with it",
        },
        "epsilon": EPS,
        "steps": STEPS,
        "adam_lr": ADAM_LR,
        "dev_checkpoints": list(DEV_CHECKPOINTS),
        "selector": {
            "reads": ["step", "nonzero", "uint8_linf", "valid", "source_full_exact", "source_replay_valid", "m_r", "m_infinity", "Delta_r"],
            "never_reads": ["gold", "anls", "strict", "fastv_text", "manual_judgment"],
            "rule": "feasible candidates minimize max_r m_r, then earlier step; replay failure drops that candidate",
        },
        "losses": {
            "a_hard": "z[y] - max_{v != y} z[v]",
            "a_soft": "z[y] - tau_vocab * logsumexp_{v != y}(z[v] / tau_vocab)",
            "bar_a_soft": "min(a_soft_clean, max(a_soft_clean - xi, epsilon_margin))",
            "L_FDP": "mean_t softplus(bar_a_soft_t - a_soft_t(x+delta))",
            "m_r": "mean soft token margin of the clean Full sequence including EOS, teacher-forced under real FastV at keep ratio r",
            "m_infinity": "same mean on the unpruned Full path",
            "D_r": "m_r(clean) - m_r(attack)",
            "D_infinity": "m_infinity(clean) - m_infinity(attack)",
            "Delta_r": "D_r - D_infinity",
            "L_B": "L_comp + lambda_soft_B * L_soft_full, lambda_soft_B frozen from Variant B",
            "L_BFDP": "L_comp + lambda_soft_B * L_FDP",
            "L_robust_comp": "tau_r * logsumexp_r (m_r(x+delta) + kappa_compressed) / tau_r",
            "L_diff": "tau_r * logsumexp_r (gamma - Delta_r(delta)) / tau_r",
            "L_Bplus": "L_robust_comp + lambda_full * L_FDP + lambda_diff * L_diff",
            "curve_regularizer": "absent unless Stage I reports a monotone budget trend and a later addendum freezes it",
        },
        "coefficients": {
            "tau_vocab": TAU_VOCAB,
            "xi": XI,
            "epsilon_margin": EPSILON_MARGIN,
            "tau_r": TAU_R,
            "kappa_compressed": KAPPA_COMPRESSED,
            "gamma": GAMMA,
            "lambda_soft_B": LAMBDA_SOFT_B,
            "lambda_full": "NOT_HAND_SET",
            "lambda_diff": "NOT_HAND_SET",
            "calibration_rule": (
                "On the frozen 12-sample development set only, at delta=0, take the median L2 "
                "pixel gradient of L_comp, L_FDP, and L_diff. "
                "lambda_full = median_L2(L_comp) / median_L2(L_FDP). "
                "lambda_diff = median_L2(L_comp) / median_L2(L_diff). "
                "One shot. A zero or non-finite required gradient blocks B+."
            ),
            "xi_reason": "Equals the pre-existing MARGIN_DROP of 1. Not fit on Stage I.",
            "gamma_reason": "Zero asks only for a positive compression-conditional gap. Not fit on Stage I.",
            "kappa_reason": "Zero adds no extra margin offset. Not fit on Stage I.",
        },
        "seeds": {
            "bootstrap": BOOTSTRAP_SEED,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "screen": SCREEN_SEED,
            "final_pristine": FINAL_SEED,
            "repeat10": REPEAT_SEED,
            "attack_salt": ATTACK_SEED_SALT,
            "attack_seed": "sha256(f'{salt}:{sample_id}:{arm}')[:8]",
            "health_sample": HEALTH_ID,
            "dev12": list(DEV12),
        },
        "matched_noise": "week2_attack_v7_pristine60/controls/single/*/noise.png, one uint8 image per sample",
        "stage1_gates": ["A_paired_delta_025", "B_sixty_percent_positive", "C_median_order", "D_margin_text_consistent"],
        "stop_decisions": [
            "NO_GO_NO_COMPRESSION_CONDITIONAL_EFFECT",
            "BLOCKED_FDP_IMPLEMENTATION",
            "BLOCKED_BPLUS_CALIBRATION",
            "KEEP_EXISTING_B_OR_BFDP_DEV_ONLY",
            "BLOCKED_INSUFFICIENT_NEW_PRISTINE_POOL",
            "BPLUS_PRISTINE60_NOT_SUPPORTED",
            "RAW_IMMUTABILITY_FAIL",
        ],
        "go_decision": "BPLUS_PRISTINE60_SUPPORTED_SOURCE_BOUNDED",
    }
    raw = json.dumps(payload, sort_keys=True).encode()
    payload["protocol_sha256"] = hashlib.sha256(raw).hexdigest()
    if DEST.exists():
        current = json.loads(DEST.read_text())
        if current != payload:
            raise SystemExit("FROZEN_V7_PROTOCOL.json verification failed")
        print(json.dumps({"frozen": "VERIFIED", "sha": payload["protocol_sha256"]}))
        return
    DEST.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"frozen": "WRITTEN", "sha": payload["protocol_sha256"]}))


if __name__ == "__main__":
    main()
