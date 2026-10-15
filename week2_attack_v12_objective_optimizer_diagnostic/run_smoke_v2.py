"""Run the frozen 3-sample x 6-group smoke matrix on GPUs 0,1,4,6."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import argparse
from pathlib import Path

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
HERE = ROOT / "week2_attack_v12_objective_optimizer_diagnostic"
PYTHON = "/home/yuesiting/autodl-tmp/conda/envs/vattack/bin/python"
SAMPLES = ("1025", "60700", "47450")
GROUPS = ("H0", "H1", "P0", "P1", "O0", "O1")
GPUS = ("0", "1", "4", "6")
DEFAULT_TAG = "smoke_frozen_v2"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default=DEFAULT_TAG)
    args = parser.parse_args()
    tag = args.tag
    log_dir = HERE / "smoke_v2_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(sample, group) for sample in SAMPLES for group in GROUPS]
    pending = list(jobs)
    running: dict[int, tuple[str, str, subprocess.Popen, object, str]] = {}
    report_path = HERE / f"{tag}_report.json"
    report = {"status": "RUNNING", "tag": tag, "samples": SAMPLES,
              "groups": GROUPS, "gpus": GPUS, "jobs": {f"{s}/{g}": "PENDING" for s, g in jobs}}
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    while pending or running:
        for gpu in GPUS:
            if any(slot == gpu for _, _, _, _, slot in running.values()) or not pending:
                continue
            sample, group = pending.pop(0)
            key = f"{sample}/{group}"
            log = log_dir / f"{sample}_{group}_gpu{gpu}.log"
            env = os.environ.copy()
            env.update({"CUDA_VISIBLE_DEVICES": gpu, "PYTHONUNBUFFERED": "1",
                        "PYTHONPATH": str(ROOT), "TOKENIZERS_PARALLELISM": "false",
                        "PYTHONNOUSERSITE": "1"})
            cmd = [PYTHON, str(HERE / "run_deterministic.py"), "--sample", sample,
                   "--group", group, "--steps", "3", "--tag", tag,
                   "--checkpoint-mode", "dense"]
            fh = log.open("w")
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
            running[proc.pid] = (sample, group, proc, fh, gpu)
            report["jobs"][key] = f"RUNNING_GPU_{gpu}"
        for pid, (sample, group, proc, fh, gpu) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            fh.close()
            key = f"{sample}/{group}"
            report["jobs"][key] = "DONE" if rc == 0 else f"FAILED_RC_{rc}"
            del running[pid]
        report["pending"] = len(pending)
        report["running"] = len(running)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        if pending or running:
            time.sleep(2)
    failed = [key for key, value in report["jobs"].items() if value != "DONE"]
    report["status"] = "PASS_PROCESS_LEVEL" if not failed else "FAILED"
    report["failed"] = failed
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if not failed else 1)


if __name__ == "__main__":
    main()
