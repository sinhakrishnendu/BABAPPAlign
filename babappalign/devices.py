"""Device selection helpers for BABAPPAlign."""

from __future__ import annotations

import warnings
from functools import lru_cache
from typing import Optional, Union

import torch


DEVICE_CHOICES = ("auto", "cpu", "cuda", "mps")


def cuda_is_available() -> bool:
    return torch.cuda.is_available()


def mps_is_available() -> bool:
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is None:
        return False
    is_built = getattr(mps_backend, "is_built", lambda: True)
    is_available = getattr(mps_backend, "is_available", lambda: False)
    return bool(is_built() and is_available())


@lru_cache(maxsize=None)
def device_is_usable(device_type: str) -> bool:
    try:
        device = torch.device(device_type)
        x = torch.ones((2, 2), device=device)
        y = torch.matmul(x, x)
        z = torch.nn.functional.gelu(y).sum()
        z.detach().cpu().item()
        return True
    except Exception as exc:
        warnings.warn(
            f"{device_type!r} is available but failed a runtime probe: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return False


def best_available_device() -> torch.device:
    if cuda_is_available() and device_is_usable("cuda"):
        return torch.device("cuda")
    if mps_is_available() and device_is_usable("mps"):
        return torch.device("mps")
    return torch.device("cpu")


def resolve_device(
    user_choice: Optional[Union[str, torch.device]] = "auto",
) -> torch.device:
    if isinstance(user_choice, torch.device):
        return user_choice

    choice = (user_choice or "auto").lower()

    if choice == "auto":
        return best_available_device()
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if cuda_is_available() and device_is_usable("cuda"):
            return torch.device("cuda")
        if cuda_is_available():
            warnings.warn(
                "CUDA was requested but failed a runtime probe; falling back to CPU.",
                RuntimeWarning,
                stacklevel=2,
            )
            return torch.device("cpu")
        warnings.warn(
            "CUDA was requested but is not available; falling back to CPU.",
            RuntimeWarning,
            stacklevel=2,
        )
        return torch.device("cpu")
    if choice == "mps":
        if mps_is_available() and device_is_usable("mps"):
            return torch.device("mps")
        if mps_is_available():
            warnings.warn(
                "Apple Metal/MPS was requested but failed a runtime probe; "
                "falling back to CPU.",
                RuntimeWarning,
                stacklevel=2,
            )
            return torch.device("cpu")
        warnings.warn(
            "Apple Metal/MPS was requested but is not available; falling back to CPU.",
            RuntimeWarning,
            stacklevel=2,
        )
        return torch.device("cpu")

    valid = ", ".join(DEVICE_CHOICES)
    raise ValueError(
        f"Unknown device choice '{user_choice}'. Expected one of: {valid}."
    )
