"""No-model tests for the v24 signed-step execution contract."""
from pathlib import Path
import sys
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from week2_attack_v24_signed_feasible_step_pilot.pilot import solve_signed


def check_descent_and_bound():
    core = [torch.tensor([1.0, 0.0])]
    full = [torch.tensor([0.0, 1.0])]
    delta, meta = solve_signed(core, full, [1.0], 2, 0.1, 1e-3, 1.0)
    assert meta["candidate_valid"] and delta is not None
    assert float(torch.dot(core[0], delta)) < 0.0
    assert float(delta.abs().max()) <= 2.0 / 255.0 + 1e-8
    assert meta["linearized_full_changes"][0] >= -0.9


def check_conflicting_constraint_is_backtracked():
    core = [torch.tensor([1.0, 0.0])]
    full = [torch.tensor([-1.0, 0.0])]
    delta, meta = solve_signed(core, full, [0.2], 2, 0.1, 1e-3, 1.0)
    assert meta["candidate_valid"] and delta is not None
    assert float(delta.abs().max()) <= 2.0 / 255.0 + 1e-8
    assert meta["linearized_full_changes"][0] >= -0.1 - 1e-7


def check_infeasible_not_zero_success():
    core = [torch.tensor([1.0, 0.0])]
    full = [torch.tensor([0.0, 0.0])]
    delta, meta = solve_signed(core, full, [-1.0], 2, 0.1, 1e-3, 1.0)
    assert delta is None
    assert meta["candidate_valid"] is False


def check_png_quantization():
    base = np.full((8, 8, 3), 127, dtype=np.uint8)
    candidate = np.clip(base.astype(np.int16) - 2, 0, 255).astype(np.uint8)
    path = Path(__file__).with_name("_math_tmp.png")
    Image.fromarray(candidate).save(path)
    reloaded = np.asarray(Image.open(path).convert("RGB"), dtype=np.int16)
    assert int(np.max(np.abs(reloaded - base.astype(np.int16)))) == 2
    path.unlink()


if __name__ == "__main__":
    check_descent_and_bound()
    check_conflicting_constraint_is_backtracked()
    check_infeasible_not_zero_success()
    check_png_quantization()
    print("MATH_AND_PIXEL_TESTS_PASS")
