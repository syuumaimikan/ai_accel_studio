from __future__ import annotations
import os, random, time
import numpy as np
import torch


def seed_all(seed: int = 1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available(): return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def sync(device: torch.device):
    if device.type == "cuda": torch.cuda.synchronize(device)
    elif device.type == "mps": torch.mps.synchronize()


def dtype_from_name(name: str, device: torch.device):
    name = name.lower()
    if name == "fp32": return torch.float32
    if name == "fp16": return torch.float16
    if name == "bf16": return torch.bfloat16
    raise ValueError(name)


def supports_dtype(device: torch.device, dtype: torch.dtype) -> bool:
    if device.type == "cpu" and dtype == torch.float16:
        # Some CPU ops are supported, but fp16 is not a useful baseline on most CPUs.
        return False
    return True


def human_bytes(n: float) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    n = float(n)
    for u in units:
        if n < 1024 or u == units[-1]: return f"{n:.2f} {u}"
        n /= 1024
