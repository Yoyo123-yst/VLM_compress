"""Apply the pre-registered gate to the independent endpoints."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
sys.path[:0] = [str(ROOT)]
HERE = ROOT / "week2_attack_v22_safety_slack_constraint_pilot"

from week2_attack_v22_safety_slack_constraint_pilot.decisions import decide  # noqa: E402


def checkpoints(name: str) -> list[dict]:
    rows = []
    for path in sorted((HERE / "trajectories" / name).glob("step_*.json")):
        row = json.loads(path.read_text())
        rows.append({"step": row["checkpoint_step"], "core_hard_mean": row["core_hard_mean"], "l_core": row["l_core"], "full_hard_min": row["full_hard_min"], "fast_text": row["fast_text"], "full_exact": bool(row["full_text_exact"] and row["full_token_exact"] and row["full_replay"])})
    return rows


def main() -> None:
    evaluation = json.loads((HERE / "independent_eval.json").read_text())
    if not evaluation["match"]:
        raise SystemExit("replay diverged")
    clean = checkpoints("T1")[0]["core_hard_mean"]
    packed = {}
    for name in ("T0", "T1", "T2"):
        endpoint = evaluation["results"][name]["ratios"]["0.25"]
        packed[name] = {
            "clean_core_hard_mean": clean,
            "checkpoints": checkpoints(name),
            "selector_core_hard_mean": endpoint["core_hard_mean"],
            "semantic_status": endpoint["semantic_status"],
            "full_exact": bool(endpoint["full_text_exact"] and endpoint["full_token_exact"] and endpoint["full_replay"]),
            "text": endpoint["fast_text"],
            "step": evaluation["results"][name]["saved_step"],
        }
    decision = decide(packed["T1"], packed["T2"], packed["T0"]["full_exact"])
    decision["t3"] = "NOT_RUN_UNVERIFIED_CONSTRAINED_BASELINE"
    (HERE / "final_decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    lines = [
        "# 安全余量约束试点",
        "",
        f"状态：`{decision['status']}`",
        "",
        "研究问题：Native-FDP 的 softplus(m_clean - m_adv) 在 Full 输出仍安全时是否持续占用攻击容量。本轮只看 47450，r=0.25。",
        "",
        "冻结协议：CORE token 5–9；sign-PGD；λ 与 v20 相同；τ_t = max(0.125, 0.5 m_clean,t)；最多 25 次接受、40 次尝试；selector 只看合法 checkpoint 的 L_core。图像前向是 processor.prepare。T3 没有已验证的 Orthogonal PGD 或 ALMA 实现，因此没有运行。",
        "",
        "第二个样本：不允许。",
        "",
    ]
    for name in ("T0", "T1", "T2"):
        lines.append(f"## {name}")
        lines.append("")
        lines.append("| step | L_core | CORE mean | Full min | Full exact | FastV |")
        lines.append("|---:|---:|---:|---:|---|---|")
        for row in packed[name]["checkpoints"]:
            lines.append(f"| {row['step']} | {row['l_core']:.4f} | {row['core_hard_mean']:.4f} | {row['full_hard_min']:.4f} | {row['full_exact']} | {row['fast_text']} |")
        endpoint = evaluation["results"][name]["ratios"]["0.25"]
        lines.append("")
        lines.append(f"selector step {packed[name]['step']}，独立评估 r=0.25：`{endpoint['semantic_status']}`，ANLS {endpoint['anls']:.4f}，文本 `{endpoint['fast_text']}`。")
        lines.append("")
    (HERE / "final_report.md").write_text("\n".join(lines))
    print(json.dumps(decision), flush=True)


if __name__ == "__main__":
    main()
