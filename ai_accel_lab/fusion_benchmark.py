from __future__ import annotations
import time
import pandas as pd
import torch
from .power import NvmlPowerSampler
from .metrics import output_metrics
from .kernels.groupwise_int2 import pack_groupwise_int2
from .kernels.fused_decode_attention import fused_decode_attention_int2,reference_decode_attention_int2,supported_giant_fusion


def _bench(fn,repeats):
    for _ in range(5): y=fn()
    torch.cuda.synchronize(); s=NvmlPowerSampler(0); s.start(); t=time.perf_counter()
    for _ in range(repeats): y=fn()
    torch.cuda.synchronize(); sec=time.perf_counter()-t; w=s.stop()
    return y,sec*1000/repeats,w


def run_fusion_benchmark(hidden=4096,heads=32,seq=128,group_size=128,repeats=50):
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    d=hidden//heads; ok,reason=supported_giant_fusion(hidden,heads,heads,d)
    if not ok: raise RuntimeError(reason)
    torch.manual_seed(9); dev='cuda'; pos=seq-1
    x=torch.randn(1,hidden,device=dev,dtype=torch.float16)
    ws=[pack_groupwise_int2(torch.randn(hidden,hidden,device=dev,dtype=torch.float16)/(hidden**.5),group_size) for _ in range(3)]
    ang=torch.randn(seq,d//2,device=dev); cos=ang.cos().half(); sin=ang.sin().half()
    kc=torch.randn(1,heads,seq,d,device=dev,dtype=torch.float16); vc=torch.randn_like(kc)
    def ref(): return reference_decode_attention_int2(x,*ws,cos,sin,kc,vc,pos,heads)
    def fused():
        # cache clones are excluded from the measured kernel path in a real engine;
        # here we reuse buffers because writing the current token is deterministic.
        return fused_decode_attention_int2(x,*ws,cos,sin,kc,vc,pos,heads)
    yr,mr,wr=_bench(ref,max(3,repeats//5)); yf,mf,wf=_bench(fused,repeats)
    q=output_metrics(yr,yf)
    df=pd.DataFrame([
        {'method':'reference-int2-pipeline','latency_ms':mr,'speedup':1.0,'avg_power_w':wr,'energy_mj':None if wr is None else wr*mr,**{k:0.0 if k!='cosine_vs_dense' else 1.0 for k in q}},
        {'method':'giant-fused-qkv-rope-attn','latency_ms':mf,'speedup':mr/mf,'avg_power_w':wf,'energy_mj':None if wf is None else wf*mf,**q},
    ])
    return df
