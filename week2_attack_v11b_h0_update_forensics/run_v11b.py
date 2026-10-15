"""Ten-step forensic comparison of a plain Native-FDP run and the v11a wrappers.

Plain and trace runs do not import the v11a module. Instrumented runs load
the v11a counter implementation only after the case asks for it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
HERE = ROOT / "week2_attack_v11b_h0_update_forensics"
PARENT_V11A = ROOT / "week2_attack_v11a_aux_scope_amendment_objective_pair"
HIST = ROOT / "week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation" / "runs" / "NATIVE_FDP"
sys.path[:0] = [str(ROOT), str(ROOT / "week1_mechanism"), str(ROOT / "gate_experiment")]

from week2_attack_v4_answer_aligned.common import sha_image  # noqa: E402
from week2_attack_v7_pristine60.run_v7 import load_model_frozen  # noqa: E402
from week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation import run_v7d as v7d  # noqa: E402

STEPS = 10
SAMPLES = ("1025", "60700")
COEFFS = {
    "lambda_full": 2.6371882367154997,
    "lambda_gap": 0.913170951688196,
    "xi": 1.0,
    "eta": 0.001,
    "source": "+1/255",
}
SOURCES = (
    "week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation/run_v7d.py",
    "week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation/native_grad.py",
    "week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation/losses.py",
    "week2_attack_v4_answer_aligned/forward.py",
    "week2_attack_v4_answer_aligned/run_smoke.py",
    "week2_attack_v7b_native_cache_margin_admission/native_cache_margin.py",
    "week2_attack_v7_pristine60/run_v7.py",
    "gate_experiment/run_causal_v2.py",
    "week1_mechanism/adapters/runtime.py",
    "week1_mechanism/adapters/fastv_select.py",
    "week2_attack_v11a_aux_scope_amendment_objective_pair/run_v11a.py",
)


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha(value: torch.Tensor) -> str:
    raw = value.detach().float().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def tensor_stats(value: torch.Tensor) -> dict:
    flat = value.detach().float()
    return {
        "sha256": tensor_sha(value),
        "l1": float(flat.abs().sum()),
        "l2": float(flat.norm()),
        "linf": float(flat.abs().max()) if flat.numel() else 0.0,
        "finite": bool(torch.isfinite(value).all()),
    }


def environment() -> dict:
    import transformers

    device = torch.cuda.current_device() if torch.cuda.is_available() else None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "transformers": transformers.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": None if device is None else torch.cuda.get_device_name(device),
        "gpu_id": device,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "process_id": os.getpid(),
    }


def provenance() -> dict:
    hashes = {rel: file_sha(ROOT / rel) for rel in SOURCES}
    historical_hash_files = list((ROOT / "week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation").glob("*HASH*"))
    payload = {
        "status": "HISTORICAL_SOURCE_PROVENANCE_INCOMPLETE",
        "reason": "The historical Native-FDP run did not save source, package, or GPU provenance hashes. The current command is not claimed to be the historical command.",
        "historical_hash_files": [str(path.relative_to(ROOT)) for path in historical_hash_files],
        "source_sha256": hashes,
        "v11a_h0_process": "The v11a H0 train process called install_counters and then v7d.attack. Phase A was a separate process and was not executed inside that H0 process.",
        "model_loader": "week2_attack_v7_pristine60/run_v7.py:load_model_frozen -> gate_experiment/run_causal_v2.py:load_model",
        "processor": "AutoProcessor.from_pretrained(LLAVA) with image_grid_pinpoints [[768, 768]]",
        "environment": environment(),
    }
    (HERE / "provenance.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"provenance": payload["status"]}), flush=True)
    return payload


def case_dir(case: str, sample: str) -> Path:
    path = HERE / "cases" / case / sample
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_meta(out: Path, case: str, sample: str) -> None:
    (out / "environment.json").write_text(json.dumps({"case": case, "sample": sample, **environment()}, indent=2) + "\n")


def run_plain(sample: str, case: str) -> None:
    out = case_dir(case, sample)
    write_meta(out, case, sample)
    model, processor = load_model_frozen()
    item = v7d.load_dev(sample)
    v7d.attack(model, processor, item, "native-FDP", STEPS, COEFFS, out)
    print(json.dumps({"done": case, "sample": sample}), flush=True)


def attack_trace(model, processor, item, out: Path, steps: int = STEPS) -> None:
    torch.manual_seed(v7d.SEED)
    torch.cuda.manual_seed_all(v7d.SEED)
    np.random.seed(v7d.SEED % (2**32))
    state = v7d.prepare_item(model, processor, item)
    delta = torch.zeros_like(state["page"], requires_grad=True)
    opt = torch.optim.Adam([delta], lr=2.0 / 255.0)
    out.mkdir(parents=True, exist_ok=True)
    item["images"][item["aux_idx"]].save(out / "clean.png")
    marks = {step for step in v7d.CHECKPOINTS if step <= steps}
    marks.add(steps)
    curve = []
    selector_rows = []
    updates = []
    clean = {
        "page_sha256": [sha_image(image) for image in item["images"]],
        "aux_idx": item["aux_idx"],
        "question": item["meta"]["question"],
        "prompt_sha256": hashlib.sha256(item["meta"]["question"].encode()).hexdigest(),
        "input_ids_sha256": tensor_sha(state["inputs"]["input_ids"]),
        "pixel_values_sha256": tensor_sha(state["inputs"]["pixel_values"]),
        "clean_full_tokens": state["clean_tokens"],
        "clean_full_text": state["clean_text"],
        "clean_fastv": "NOT_CAPTURED_INSIDE_THE_TRAINING_LOOP",
    }
    (out / "clean_provenance.json").write_text(json.dumps(clean) + "\n")
    for step in range(steps + 1):
        if step in marks:
            image = v7d.page_uint8(state["page"], delta)
            dest = out / "checkpoints" / f"step_{step:03d}.png"
            dest.parent.mkdir(parents=True, exist_ok=True)
            image.save(dest)
            reloaded = Image_open(dest)
            scored = v7d.score_png(model, processor, item, reloaded, state["clean_text"])
            fields = dict(scored["selector_fields"])
            fields["step"] = step
            fields["linf_ok"] = int(step > 0 and fields["nonzero"] and 0 < fields["pixel_linf_uint8"] <= v7d.EPS + 1e-6)
            selector_rows.append({key: fields[key] for key in v7d.LEGAL})
            curve.append({
                "step": step,
                "png_sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
                "texts": scored["texts"],
                "margins": scored["margins"],
                "keeps": scored["keeps"],
                "selector": selector_rows[-1],
            })
            print(json.dumps({"sample": item["meta"]["sample_id"], "method": "trace", "step": step, "full_exact": fields["full_exact"]}), flush=True)
        if step == steps:
            break
        opt.zero_grad(set_to_none=True)
        page = state["page"] + delta
        v7d.backward_method(model, state, item, page, "native-FDP", COEFFS, delta)
        if delta.grad is None or not torch.isfinite(delta.grad).all():
            raise RuntimeError(f"non-finite gradient at step {step}")
        gradient = tensor_stats(delta.grad)
        opt.step()
        before = tensor_stats(delta)
        moment = opt.state[delta]
        average = tensor_stats(moment["exp_avg"])
        square = tensor_stats(moment["exp_avg_sq"])
        v7d.project(delta, state["page"], v7d.EPS)
        after = tensor_stats(delta)
        updates.append({
            "step": step,
            "seed": v7d.SEED,
            "process_id": os.getpid(),
            "gpu_id": torch.cuda.current_device(),
            "before_update": {
                "fastv_tokens_and_margins": "INSIDE_BACKWARD_NOT_RETURNED",
                "full_fdp": "INSIDE_BACKWARD_NOT_RETURNED",
                "total_gradient": gradient,
            },
            "after_adam_before_project": {
                "delta": before,
                "exp_avg_sha256": average["sha256"],
                "exp_avg_sq_sha256": square["sha256"],
            },
            "after_project": {"delta": after},
        })
        torch.cuda.empty_cache()
    chosen = v7d.choose(selector_rows)
    (out / "curve.json").write_text(json.dumps(curve, indent=2) + "\n")
    (out / "selected.json").write_text(json.dumps({"chosen": chosen, "clean_text": state["clean_text"]}, indent=2) + "\n")
    (out / "updates.json").write_text(json.dumps(updates) + "\n")


def Image_open(path: Path):
    from PIL import Image
    return Image.open(path).convert("RGB")


def v11a_module():
    import importlib.util
    name = "v11a_exact_forensics"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, PARENT_V11A / "run_v11a.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def install_generate_only(model) -> None:
    v11a = v11a_module()
    original_generate = model.generate

    def generate(*args, **kwargs):
        v11a.Access.generate += 1
        return original_generate(*args, **kwargs)

    model.generate = generate


def install_backward() -> None:
    v11a = v11a_module()
    original_backward = torch.Tensor.backward

    def backward(self, *args, **kwargs):
        v11a.Access.backward += 1
        return original_backward(self, *args, **kwargs)

    torch.Tensor.backward = backward


def install_forward_hook(model) -> None:
    v11a = v11a_module()

    def count_forward(module, args, kwargs):
        if module is model:
            v11a.Access.forward += 1

    model.register_forward_pre_hook(count_forward, with_kwargs=True)


def install_v11a_counters(model) -> None:
    v11a_module().install_counters(model)


def phase_a_prerun(model, processor, out: Path) -> None:
    module = v11a_module()
    module.HERE = out / "phase_a_prerun"
    module.phase_a(model, processor)


def run_trace_case(sample: str, case: str) -> None:
    out = case_dir(case, sample)
    write_meta(out, case, sample)
    model, processor = load_model_frozen()
    if case == "C1_model_generate":
        install_generate_only(model)
    elif case == "C2_plus_tensor_backward":
        install_generate_only(model)
        install_backward()
    elif case == "C3_plus_forward_hook":
        install_generate_only(model)
        install_backward()
        install_forward_hook(model)
    elif case == "C4_plus_generate_tokens_wrapper":
        install_v11a_counters(model)
    elif case in {"C5_plus_phase_a_prerun", "C_all_v11a"}:
        install_v11a_counters(model)
        phase_a_prerun(model, processor, out)
    elif case != "C0_trace":
        raise RuntimeError(case)
    item = v7d.load_dev(sample)
    attack_trace(model, processor, item, out)
    print(json.dumps({"done": case, "sample": sample}), flush=True)


def curve_signature(path: Path) -> dict:
    rows = {row["step"]: row for row in json.loads((path / "curve.json").read_text())}
    found = {}
    for step in (0, 10):
        row = rows[step]
        found[step] = {
            "png_sha256": row["png_sha256"],
            "full": row["texts"]["full"],
            "fastv_0.25": row["texts"]["fastv_0.25"],
            "fastv_0.35": row["texts"]["fastv_0.35"],
            "keep_0.25": row["keeps"]["0.25"]["keep_hash"],
            "keep_0.35": row["keeps"]["0.35"]["keep_hash"],
            "margin": row["selector"]["margin"],
        }
    return found


def same_signature(left: dict, right: dict) -> bool:
    return left == right


def historical_signature(sample: str) -> dict:
    rows = {row["step"]: row for row in json.loads((HIST / sample / "curve.json").read_text())}
    found = {}
    for step in (0, 10):
        row = rows[step]
        found[step] = {
            "png_sha256": row["png_sha256"],
            "full": row["texts"]["full"],
            "fastv_0.25": row["texts"]["fastv_0.25"],
            "fastv_0.35": row["texts"]["fastv_0.35"],
            "keep_0.25": row["keeps"]["0.25"]["keep_hash"],
            "keep_0.35": row["keeps"]["0.35"]["keep_hash"],
            "margin": row["selector"]["margin"],
        }
    return found


def first_update_diff(left: str, right: str, sample: str) -> int | None:
    a = json.loads((case_dir(left, sample) / "updates.json").read_text())
    b = json.loads((case_dir(right, sample) / "updates.json").read_text())
    for row_a, row_b in zip(a, b):
        if row_a["before_update"]["total_gradient"]["sha256"] != row_b["before_update"]["total_gradient"]["sha256"]:
            return int(row_a["step"])
        if row_a["after_project"]["delta"]["sha256"] != row_b["after_project"]["delta"]["sha256"]:
            return int(row_a["step"])
    return None


def launch(mode: str, case: str, sample: str) -> None:
    import subprocess
    log_dir = HERE / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPATH"] = "."
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    command = [
        "/home/yuesiting/autodl-tmp/conda/envs/vattack/bin/python",
        str(HERE / "run_v11b.py"),
        mode,
        "--case",
        case,
        "--sample",
        sample,
    ]
    log = log_dir / f"{case}_{sample}.txt"
    with log.open("w") as handle:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
    progress = HERE / "drive_progress.jsonl"
    with progress.open("a") as handle:
        handle.write(json.dumps({"case": case, "sample": sample, "returncode": result.returncode}) + "\n")
    if result.returncode != 0:
        raise RuntimeError(f"{case} {sample} failed; see {log}")


def drive() -> None:
    report = {"sample_primary": "1025", "gpu": "0", "cases": {}}
    launch("plain", "C0_plain_a", "1025")
    launch("plain", "C0_plain_b", "1025")
    plain_a = curve_signature(case_dir("C0_plain_a", "1025"))
    plain_b = curve_signature(case_dir("C0_plain_b", "1025"))
    report["plain_repeatable"] = plain_a == plain_b
    report["plain_vs_historical"] = plain_a == historical_signature("1025")
    report["cases"]["C0_plain_a"] = plain_a
    if plain_a != plain_b:
        report["status"] = "RUNTIME_NONREPEATABLE"
        (HERE / "forensic_decision.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"status": report["status"]}), flush=True)
        return
    launch("trace", "C0_trace", "1025")
    trace = curve_signature(case_dir("C0_trace", "1025"))
    report["trace_matches_plain"] = trace == plain_a
    if trace != plain_a:
        report["status"] = "TRACE_IMPLEMENTATION_INTRUSIVE"
        report["trace"] = trace
        (HERE / "forensic_decision.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"status": report["status"]}), flush=True)
        return
    launch("trace", "C_all_v11a", "1025")
    all_sig = curve_signature(case_dir("C_all_v11a", "1025"))
    report["all_matches_plain"] = all_sig == plain_a
    report["all_first_diff_step"] = first_update_diff("C0_trace", "C_all_v11a", "1025")
    if all_sig == plain_a:
        report["status"] = "INSTRUMENTATION_NOT_SUFFICIENT"
        (HERE / "forensic_decision.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"status": report["status"]}), flush=True)
        return
    sequence = (
        "C1_model_generate",
        "C2_plus_tensor_backward",
        "C3_plus_forward_hook",
        "C4_plus_generate_tokens_wrapper",
        "C5_plus_phase_a_prerun",
    )
    previous = "C0_trace"
    first = None
    for case in sequence:
        launch("trace", case, "1025")
        matched = curve_signature(case_dir(case, "1025")) == plain_a
        report["cases"][case] = {"matches_plain": matched, "first_diff_step": first_update_diff("C0_trace", case, "1025")}
        if not matched:
            first = case
            break
        previous = case
    report["first_diverging_case"] = first
    report["last_matching_case"] = previous
    if first is not None:
        for case in (previous, first):
            mode = "plain" if case == "C0_plain_a" else "trace"
            if case == "C0_trace":
                mode = "trace"
            launch(mode, case, "60700")
        report["confirmation_60700"] = {
            "previous_vs_1025_pattern": curve_signature(case_dir(previous, "60700")) != historical_signature("60700"),
            "pair_differs": curve_signature(case_dir(previous, "60700")) != curve_signature(case_dir(first, "60700")),
            "first_diff_step": first_update_diff(previous, first, "60700") if previous != "C0_plain_a" else None,
        }
    report["status"] = "FORK_LOCALIZED" if first else "CUMULATIVE_CASES_MATCHED_PLAIN"
    (HERE / "forensic_decision.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "first": first, "previous": previous}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("provenance", "plain", "trace", "drive"))
    parser.add_argument("--case")
    parser.add_argument("--sample")
    args = parser.parse_args()
    HERE.mkdir(parents=True, exist_ok=True)
    if args.mode == "provenance":
        provenance()
        return
    if args.mode == "drive":
        drive()
        return
    if args.mode == "plain":
        run_plain(args.sample, args.case)
        return
    run_trace_case(args.sample, args.case)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise
