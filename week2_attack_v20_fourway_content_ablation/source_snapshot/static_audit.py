"""Static gate. A failure here blocks the GPU experiment."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
HERE = ROOT / "week2_attack_v20_fourway_content_ablation"
sys.path[:0] = [str(ROOT)]

from week2_attack_v20_fourway_content_ablation.protocol import (  # noqa: E402
    EPS,
    EPSILON_UINT8,
    EVAL_RATIOS,
    LAMBDA_FULL,
    MAX_ACCEPTED,
    METHODS,
    OPTIMIZER,
    PRIMARY_SAMPLES,
    STEP_ALPHA,
    TRAIN_RATIO,
)

REQUIRED = [
    "dynamic_sequence_loss",
    "fixed_all_loss",
    "fixed_noeos_loss",
    "core_content_loss",
    "full_fdp_loss",
    "CORE_SPAN_UNAVAILABLE",
    "choose_fixed",
]
FORBIDDEN = [
    "true_unseen",
    "content_only",
    "V19_OBJECTIVE",
    "anls_score",
    "semantic_status",
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    problems = []
    losses = (HERE / "losses.py").read_text()
    selection = (HERE / "selection.py").read_text()
    train = (HERE / "run_train.py").read_text()
    selector = (HERE / "selector.py").read_text()
    for name in REQUIRED[:5]:
        if f"def {name}" not in losses:
            problems.append(f"missing {name}")
    if "CORE_SPAN_UNAVAILABLE" not in selection:
        problems.append("core fallback marker missing")
    if "content_only" in train or "V19_OBJECTIVE" in train:
        problems.append("training still uses a boolean objective switch")
    for banned in ("true_unseen", "anls_score"):
        if banned in train or banned in selector:
            problems.append(f"{banned} is visible to training or the selector")
    if "row[\"semantic_status\"]" in selector or "row['anls']" in selector:
        problems.append("selector reads a forbidden outcome field")
    if "semantic_status" in train or "anls" in train:
        problems.append("training records or reads a semantic outcome")
    if "dynamic_sequence_loss(" not in train or "fixed_all_loss(" not in train:
        problems.append("A/B losses are not called directly")
    if "fixed_noeos_loss(" not in train or "core_content_loss(" not in train:
        problems.append("C/D losses are not called directly")
    if OPTIMIZER != "sign_pgd":
        problems.append("optimizer drifted")
    if abs(EPS - 8 / 255) > 1e-12 or EPSILON_UINT8 != 8:
        problems.append("epsilon drifted")
    if TRAIN_RATIO != 0.25 or EVAL_RATIOS != (0.15, 0.25, 0.35, 0.50):
        problems.append("ratio set drifted")
    if MAX_ACCEPTED != 25 or abs(STEP_ALPHA - 1 / 255) > 1e-12:
        problems.append("step protocol drifted")
    if abs(LAMBDA_FULL - 2.6371882367154997) > 1e-12:
        problems.append("lambda drifted")
    if PRIMARY_SAMPLES != ("1025", "2008", "7006", "39204", "47450", "60700"):
        problems.append("primary sample order drifted")
    if set(METHODS) != {"A", "B", "C", "D"}:
        problems.append("method set drifted")
    if problems:
        print(json.dumps({"status": "FAIL", "problems": problems}, indent=2))
        raise SystemExit(1)
    sources = sorted(path for path in HERE.glob("*.py"))
    snapshot = HERE / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    hashes = {}
    for path in sources:
        target = snapshot / path.name
        target.write_bytes(path.read_bytes())
        hashes[path.name] = sha(target)
    git = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--is-inside-work-tree"], capture_output=True, text=True)
    git_diff = "NOT_A_GIT_REPOSITORY"
    if git.returncode == 0 and git.stdout.strip() == "true":
        diff = subprocess.run(["git", "-C", str(ROOT), "diff", "--", str(HERE)], capture_output=True, text=True)
        git_diff = diff.stdout
    env = {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
    }
    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda"] = torch.version.cuda
    except Exception as exc:  # pragma: no cover
        env["torch_error"] = str(exc)
    report = {"status": "PASS", "source_sha256": hashes, "environment": env, "git_diff": git_diff}
    (HERE / "source_snapshot" / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "files": hashes}, indent=2))


if __name__ == "__main__":
    main()
