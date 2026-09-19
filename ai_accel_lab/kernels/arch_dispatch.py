from __future__ import annotations
from dataclasses import dataclass
import torch
from ..hardware import detect_hardware
from .groupwise_int2 import triton_groupwise_int2_linear
from .persistent_int2 import persistent_groupwise_int2_linear

@dataclass
class KernelChoice:
    backend:str
    reason:str

def choose_int2_backend(x,m,n,k):
    if not x.is_cuda:return KernelChoice('reference','CUDA unavailable')
    info=detect_hardware(x.device.index or 0)
    # v3 selected persistent automatically on Hopper small-M. Real measurements
    # showed that heuristic can be catastrophically wrong. v4 reserves the
    # persistent path for explicit benchmarking and uses the specialized M<=4
    # GEMV or generic Triton path by default.
    if m<=4 and info.triton:return KernelChoice('decode-int2','small-M autoregressive decode specialization')
    if info.triton:return KernelChoice('triton-int2','generic Triton packed INT2 path')
    return KernelChoice('reference','Triton unavailable')

def run_groupwise_int2(x,pw,bias=None,backend='auto'):
    if backend=='auto':backend=choose_int2_backend(x,x.shape[0],pw.n,x.shape[1]).backend
    if backend=='decode-int2':
        from .decode_int2 import decode_int2_linear
        return decode_int2_linear(x,pw,bias)
    if backend=='persistent-int2':return persistent_groupwise_int2_linear(x,pw,bias)
    if backend=='triton-int2':return triton_groupwise_int2_linear(x,pw,bias)
    from .groupwise_int2 import reference_linear
    return reference_linear(x,pw,bias)
