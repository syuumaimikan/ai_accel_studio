from __future__ import annotations
import torch
import torch.nn.functional as F
from .packing import prune_24


def supported() -> tuple[bool, str]:
    if not torch.cuda.is_available():
        return False, 'CUDA is unavailable'
    major, minor = torch.cuda.get_device_capability()
    if major < 8:
        return False, f'2:4 Sparse Tensor Core requires compute capability >= 8.0; got {major}.{minor}'
    try:
        from torch.sparse import to_sparse_semi_structured
    except Exception as e:
        return False, f'to_sparse_semi_structured unavailable: {e}'
    return True, 'ok'


def make_sparse_weight(weight: torch.Tensor):
    """Prune to 2:4 then compress using PyTorch CUTLASS/cuSPARSELt backend."""
    ok, reason = supported()
    if not ok:
        raise RuntimeError(reason)
    from torch.sparse import to_sparse_semi_structured
    if weight.dtype not in (torch.float16, torch.bfloat16):
        weight = weight.half()
    w24 = prune_24(weight).contiguous()
    # Current CUDA backend has shape constraints (usually multiples of 64 for fp16/bf16).
    return to_sparse_semi_structured(w24)


def linear(x: torch.Tensor, sparse_weight, bias=None):
    # F.linear dispatches to sparse semi-structured addmm/mm kernels.
    return F.linear(x, sparse_weight, bias)
