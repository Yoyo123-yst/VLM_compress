"""Sign-PGD is the frozen optimizer. Adam rollback is implemented and tested."""

from __future__ import annotations

import copy
import hashlib

import torch

from week2_attack_v4_answer_aligned.run_smoke import project


def state_hash(optimizer: torch.optim.Optimizer | None) -> str:
    if optimizer is None:
        return "SIGN_PGD_NO_MOMENTUM"
    payload = []
    for key, value in optimizer.state_dict().items():
        payload.append(str(key).encode())
        if isinstance(value, dict):
            for inner_key, inner in value.items():
                payload.append(str(inner_key).encode())
                if torch.is_tensor(inner):
                    payload.append(inner.detach().cpu().numpy().tobytes())
                else:
                    payload.append(repr(inner).encode())
        else:
            payload.append(repr(value).encode())
    return hashlib.sha256(b"".join(payload)).hexdigest()


def snapshot_adam(optimizer: torch.optim.Optimizer) -> dict:
    return copy.deepcopy(optimizer.state_dict())


def restore_adam(optimizer: torch.optim.Optimizer, snapshot: dict, delta: torch.Tensor, previous: torch.Tensor) -> None:
    delta.data.copy_(previous)
    optimizer.load_state_dict(snapshot)
    step_ok = True
    for state in optimizer.state.values():
        if "exp_avg" not in state or "exp_avg_sq" not in state or "step" not in state:
            step_ok = False
    if not step_ok:
        raise RuntimeError("Adam rollback did not restore exp_avg, exp_avg_sq, and step")


def sign_step(delta: torch.Tensor, clean: torch.Tensor, grad: torch.Tensor, alpha: float, eps: float) -> None:
    with torch.no_grad():
        delta.add_((-grad.sign()) * float(alpha))
        project(delta, clean, eps)
