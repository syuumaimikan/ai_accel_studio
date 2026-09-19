from __future__ import annotations
import time
import pandas as pd
import torch
import torch.nn.functional as F

from .packing import (
    pack_int2, pack_ternary, pack_int2_sparse24,
    unpack_int2, unpack_ternary, unpack_int2_sparse24, packed_bytes, prune_24,
)


def _sync(): torch.cuda.synchronize()

def _time(fn, warmup=20, repeats=100):
    for _ in range(warmup): fn()
    _sync()
    start=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats): fn()
    end.record(); _sync()
    return start.elapsed_time(end)/repeats


def _metrics(ref,y):
    a=ref.float(); b=y.float(); d=b-a
    return {
        'mse':float((d*d).mean()),
        'mae':float(d.abs().mean()),
        'cosine':float(F.cosine_similarity(a.flatten()[None],b.flatten()[None]).item()),
    }


def run_gpu_kernel_benchmark(m=128,n=4096,k=4096,repeats=100,activation='none',include_cuda_ext=False):
    if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU required')
    if k%4: raise ValueError('K must be divisible by 4')
    torch.manual_seed(123)
    x=torch.randn(m,k,device='cuda',dtype=torch.float16)
    w=torch.randn(n,k,device='cuda',dtype=torch.float16)/(k**0.5)
    bias=torch.randn(n,device='cuda',dtype=torch.float16)*0.01

    def epi(y):
        if activation=='relu': return F.relu(y)
        if activation in ('silu','swish'): return F.silu(y)
        return y
    ref=epi(F.linear(x,w,bias))
    dense_ms=_time(lambda: epi(F.linear(x,w,bias)),repeats=repeats)
    dense_bytes=w.numel()*2
    dense_flops=2.0*m*n*k
    rows=[dict(method='torch_dense_fp16',latency_ms=dense_ms,speedup=1.0,weight_bytes=dense_bytes,compression_vs_fp16=1.0,effective_tflops=dense_flops/(dense_ms*1e-3)/1e12,**_metrics(ref,ref))]

    packed_cases=[('int2',pack_int2(w)),('ternary',pack_ternary(w)),('int2_2to4',pack_int2_sparse24(w))]

    try:
        from . import triton_fused as tf
        for name,pw in packed_cases:
            if name=='int2': fn=lambda pw=pw: tf.int2_linear(x,pw,bias,activation)
            elif name=='ternary': fn=lambda pw=pw: tf.ternary_linear(x,pw,bias,activation)
            else: fn=lambda pw=pw: tf.int2_sparse24_linear(x,pw,bias,activation)
            y=fn(); ms=_time(fn,repeats=repeats)
            wb=packed_bytes(pw); rows.append(dict(method='triton_'+name,latency_ms=ms,speedup=dense_ms/ms,weight_bytes=wb,compression_vs_fp16=dense_bytes/wb,effective_tflops=dense_flops/(ms*1e-3)/1e12,**_metrics(ref,y)))
    except Exception as e:
        rows.append(dict(method='triton_unavailable',latency_ms=float('nan'),speedup=float('nan'),weight_bytes=0,compression_vs_fp16=float('nan'),effective_tflops=float('nan'),mse=float('nan'),mae=float('nan'),cosine=float('nan'),note=str(e)))

    if include_cuda_ext:
        try:
            from . import cuda_ext as ce
            for name,pw in packed_cases:
                if name=='int2': fn=lambda pw=pw: ce.int2_linear(x,pw,bias,activation)
                elif name=='ternary': fn=lambda pw=pw: ce.ternary_linear(x,pw,bias,activation)
                else: fn=lambda pw=pw: ce.int2_sparse24_linear(x,pw,bias,activation)
                y=fn(); ms=_time(fn,repeats=repeats)
                wb=packed_bytes(pw); rows.append(dict(method='cuda_'+name,latency_ms=ms,speedup=dense_ms/ms,weight_bytes=wb,compression_vs_fp16=dense_bytes/wb,effective_tflops=dense_flops/(ms*1e-3)/1e12,**_metrics(ref,y)))
        except Exception as e:
            rows.append(dict(method='cuda_ext_unavailable',latency_ms=float('nan'),speedup=float('nan'),weight_bytes=0,compression_vs_fp16=float('nan'),effective_tflops=float('nan'),mse=float('nan'),mae=float('nan'),cosine=float('nan'),note=str(e)))

    # True hardware 2:4 path: PyTorch dispatches to CUTLASS/cuSPARSELt sparse kernels.
    try:
        from .sparse24_native import make_sparse_weight
        w24_dense=prune_24(w).contiguous()
        w24=make_sparse_weight(w)
        ref24=F.linear(x,w24_dense,bias)
        fn=lambda: F.linear(x,w24,bias)
        y=fn(); ms=_time(fn,repeats=repeats)
        # FP16 semi-structured representation uses values + metadata; PyTorch docs list 9/16 compression.
        wb=int(dense_bytes*9/16)
        rows.append(dict(method='native_sparse24_tensorcore',latency_ms=ms,speedup=dense_ms/ms,
                         weight_bytes=wb,compression_vs_fp16=dense_bytes/wb,effective_tflops=dense_flops/(ms*1e-3)/1e12,
                         **_metrics(ref24,y)))
    except Exception as e:
        rows.append(dict(method='native_sparse24_unavailable',latency_ms=float('nan'),speedup=float('nan'),weight_bytes=0,compression_vs_fp16=float('nan'),effective_tflops=float('nan'),mse=float('nan'),mae=float('nan'),cosine=float('nan'),note=str(e)))

    df=pd.DataFrame(rows)
    return df
