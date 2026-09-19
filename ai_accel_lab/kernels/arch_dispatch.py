from __future__ import annotations

from dataclasses import dataclass
import torch

from ..hardware import detect_hardware
from .groupwise_int2 import triton_groupwise_int2_linear
from .persistent_int2 import persistent_groupwise_int2_linear


@dataclass
class KernelChoice:
    backend: str
    reason: str


def choose_int2_backend(x: torch.Tensor, m: int, n: int, k: int) -> KernelChoice:
    info = detect_hardware(x.device.index or 0 if x.is_cuda else 0)
    if not x.is_cuda:
        return KernelChoice("reference", "CUDA is unavailable")
    # Persistent scheduling helps decode/small-M shapes most often.
    if info.tma and m <= 32:
        return KernelChoice("persistent-int2", "SM90+ and decode/small-M workload")
    if info.triton:
        return KernelChoice("triton-int2", "Triton CUDA path")
    return KernelChoice("reference", "Triton unavailable")


def run_groupwise_int2(x, pw, bias=None, backend="auto"):
    if backend == "auto":
        backend = choose_int2_backend(x, x.shape[0], pw.n, x.shape[1]).backend
    if backend == "persistent-int2":
        return persistent_groupwise_int2_linear(x,pw,bias)
    if backend == "triton-int2":
        return triton_groupwise_int2_linear(x,pw,bias)
    from .groupwise_int2 import reference_linear
    return reference_linear(x,pw,bias)
