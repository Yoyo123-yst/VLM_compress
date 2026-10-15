"""Append a heartbeat while the four method queues are alive."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

HERE = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project/week2_attack_v20_fourway_content_ablation")
RUNTIME = HERE / "runtime"
INTERVAL = 300


def gpu_rows() -> list[dict]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    rows = []
    for line in result.stdout.splitlines():
        index, used, free, util = [item.strip() for item in line.split(",")]
        if index in {"0", "1", "4", "6"}:
            rows.append({"index": index, "memory_used_mib": int(used), "memory_free_mib": int(free), "utilization": int(util)})
    return rows


def pid_alive(pid: int) -> bool:
    try:
        Path(f"/proc/{pid}").stat()
        return True
    except FileNotFoundError:
        return False


def latest_log(path: Path) -> str | None:
    if not path.exists():
        return None
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(path.stat().st_mtime))


def main() -> None:
    RUNTIME.mkdir(parents=True, exist_ok=True)
    sink = RUNTIME / "heartbeat.jsonl"
    while True:
        manifest = json.loads((RUNTIME / "process_manifest.json").read_text()) if (RUNTIME / "process_manifest.json").exists() else {}
        processes = []
        alive = 0
        for row in manifest.get("processes", []):
            ok = pid_alive(int(row["pid"]))
            alive += int(ok)
            processes.append({**row, "alive": ok})
        logs = sorted(RUNTIME.glob("*.stdout"))
        results = list((HERE / "smoke").rglob("*.json")) + list((HERE / "full_dev_fourway_v1").rglob("*.json"))
        payload = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "gpus": gpu_rows(),
            "processes": processes,
            "alive_processes": alive,
            "log_mtime": {path.name: latest_log(path) for path in logs},
            "result_files": len(results),
            "result_bytes": sum(path.stat().st_size for path in results),
            "phase": manifest.get("phase"),
        }
        with sink.open("a") as handle:
            handle.write(json.dumps(payload) + "\n")
        print(json.dumps({"heartbeat": payload["time"], "alive": alive, "phase": payload["phase"]}), flush=True)
        if manifest.get("phase") in {"COMPLETE", "BLOCKED_SMOKE_GATE", "BLOCKED_RUNTIME"} and alive == 0:
            break
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
