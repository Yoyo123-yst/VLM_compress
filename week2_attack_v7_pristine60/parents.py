"""Parent files that v7 may hash but must not modify."""

from pathlib import Path

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")

PARENTS = [
    "week2_attack_v5b_expanded_pristine40/final_decision.json",
    "week2_attack_v5b_expanded_pristine40/FROZEN.json",
    "week2_attack_v5b_expanded_pristine40/frozen_manifest.json",
    "week2_attack_v5b_expanded_pristine40/results/random_noise.csv",
    "week2_attack_v5b_expanded_pristine40/mechanism_audit_v1/raw_before.sha256",
    "week2_attack_v6_survivor_upgrade/FROZEN_DEV_PROTOCOL.json",
    "week2_attack_v6_survivor_upgrade/DEV_PROTOCOL_ADDENDUM.json",
    "week2_attack_v6_survivor_upgrade/final_decision.json",
    "week2_attack_v6_survivor_upgrade/final_v6_report.md",
    "week2_attack_v6_survivor_upgrade/variant_a_audit/audit_decision.json",
    "week2_attack_v6b_soft_full_anchor/FROZEN_PROTOCOL.json",
    "week2_attack_v6b_soft_full_anchor/FROZEN_COEFFICIENTS.json",
    "week2_attack_v6b_soft_full_anchor/final_decision.json",
    "week2_attack_v6b_soft_full_anchor/final_report.md",
    "week2_attack_v6b_soft_full_anchor/logs/health_1025.json",
    "week2_attack_v6b_soft_full_anchor/logs/health_6194.json",
    "week2_attack_v6b_soft_full_anchor/logs/health_7006.json",
    "week2_attack_v6c_split_calibration/FROZEN_PROTOCOL.json",
    "week2_attack_v6c_split_calibration/FROZEN_COEFFICIENTS.json",
    "week2_attack_v6c_split_calibration/final_dev_decision.json",
    "week2_attack_v6c_split_calibration/final_dev_report.md",
    "week2_attack_v5_pristine40/exclusions/historical_attack_ids.json",
]


def parent_hashes() -> dict[str, str]:
    import hashlib
    out = {}
    for rel in PARENTS:
        path = ROOT / rel
        out[rel] = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    return out
