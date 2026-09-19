from __future__ import annotations

from pathlib import Path
import os
import torch

_EXT = None
_EXT_ERROR = None


def load_cuda_extension(verbose: bool = False):
    """Lazy-build the raw CUDA fused kernels using torch.utils.cpp_extension."""
    global _EXT, _EXT_ERROR
    if _EXT is not None:
        return _EXT
    if _EXT_ERROR is not None:
        raise RuntimeError(_EXT_ERROR)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available')
    try:
        from torch.utils.cpp_extension import load
        root = Path(__file__).resolve().parents[2]
        sources = [str(root/'csrc'/'packed_kernels.cpp'), str(root/'csrc'/'packed_kernels_cuda.cu')]
        _EXT = load(
            name='ai_accel_lab_packed_cuda_v2',
            sources=sources,
            extra_cuda_cflags=['-O3', '--use_fast_math', '-lineinfo'],
            extra_cflags=['-O3'],
            verbose=verbose,
        )
        return _EXT
    except Exception as e:
        _EXT_ERROR = f'CUDA extension build failed: {e}'
        raise RuntimeError(_EXT_ERROR) from e


def _bias(x, bias):
    if bias is None:
        return torch.empty(0, device=x.device, dtype=x.dtype)
    return bias.to(device=x.device, dtype=x.dtype).contiguous()


def _act(name: str):
    return {'none':0, 'identity':0, 'relu':1, 'silu':2, 'swish':2}[name.lower()]


def int2_linear(x, pw, bias=None, activation='none', verbose_build=False):
    ext = load_cuda_extension(verbose_build)
    return ext.int2_linear(x.contiguous(), pw.packed.contiguous(), pw.scale.contiguous(), _bias(x,bias), _act(activation))


def ternary_linear(x, pw, bias=None, activation='none', verbose_build=False):
    ext = load_cuda_extension(verbose_build)
    return ext.ternary_linear(x.contiguous(), pw.packed.contiguous(), pw.scale.contiguous(), _bias(x,bias), _act(activation))


def int2_sparse24_linear(x, pw, bias=None, activation='none', verbose_build=False):
    ext = load_cuda_extension(verbose_build)
    return ext.int2_sparse24_linear(x.contiguous(), pw.packed.contiguous(), pw.scale.contiguous(), _bias(x,bias), _act(activation))
