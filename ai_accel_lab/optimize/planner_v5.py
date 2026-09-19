from __future__ import annotations
from dataclasses import dataclass,asdict
import torch
from torch import nn
import torch.nn.functional as F
from .planner_v4 import decide_linear_v4,quality
from .hybrid_int2 import HybridInt2Linear
from .hybrid_int4 import HybridInt4Linear

@dataclass
class LayerDecisionV5:
    name:str
    method:str
    bits:int
    group_size:int
    outlier_columns:int
    residual_rank:int
    nmse:float
    cosine:float
    compression:float
    decode_speedup:float|None
    prefill_speedup:float|None
    projected_speedup:float|None
    reason:str
    def to_dict(self):return asdict(self)


def _int4_bytes(layer,cand):
    orig=layer.weight.numel()*2+(0 if layer.bias is None else layer.bias.numel()*2)
    opt=cand.packed.numel()*cand.packed.element_size()+cand.scales.numel()*cand.scales.element_size()
    opt+=cand.outlier_idx.numel()*cand.outlier_idx.element_size()+cand.outlier_weight.numel()*cand.outlier_weight.element_size()
    if cand.bias is not None:opt+=cand.bias.numel()*cand.bias.element_size()
    return orig/max(1,opt)


def _cuda_ms(fn,repeats):
    for _ in range(3):fn()
    torch.cuda.synchronize();st=torch.cuda.Event(enable_timing=True);en=torch.cuda.Event(enable_timing=True);st.record()
    for _ in range(repeats):fn()
    en.record();en.synchronize();return float(st.elapsed_time(en))/repeats

def _runtime_speedups(layer,candidate,x,expected_decode_tokens=64):
    """Measure M=1 decode and a representative small prefill, then project total latency."""
    if layer.weight.device.type!='cuda':return None,None,None
    bias=None if layer.bias is None else layer.bias.to(x.dtype)
    x1=x[:1].contiguous();mp=min(32,x.shape[0]);xp=x[:mp].contiguous()
    dense1=lambda:F.linear(x1,layer.weight.to(x1.dtype),bias); quant1=lambda:candidate(x1)
    densep=lambda:F.linear(xp,layer.weight.to(xp.dtype),bias); quantp=lambda:candidate(xp)
    d1=_cuda_ms(dense1,24);q1=_cuda_ms(quant1,24);dp=_cuda_ms(densep,12);qp=_cuda_ms(quantp,12)
    decode=d1/max(q1,1e-9);prefill=dp/max(qp,1e-9)
    projected=(dp+d1*expected_decode_tokens)/max(qp+q1*expected_decode_tokens,1e-9)
    return decode,prefill,projected


def decide_linear_v5(name,layer,sample_x,group_sizes=(128,64,32),outliers=(0,8,16,32),residual_ranks=(0,4,8),
                     nmse_limit=.06,cosine_limit=.96,min_compression=1.45,min_decode_speedup=1.0,min_projected_speedup=1.02,expected_decode_tokens=64):
    dev=layer.weight.device;dt=torch.float16 if dev.type=='cuda' else torch.float32
    x=sample_x.to(dev,dtype=dt);rms=sample_x.float().square().mean(0).sqrt().to(dev)

    # Aggressive first: v4 INT2 accuracy search, then a real M=1 performance gate.
    d2=decide_linear_v4(name,layer,sample_x,group_sizes,outliers,residual_ranks,nmse_limit,cosine_limit,min_compression)
    int2_rejected_for_speed=False
    if d2.method!='fp16':
        try:
            c2=HybridInt2Linear(layer,d2.group_size,rms,d2.outlier_columns,d2.residual_rank,'auto').to(dev)
            sp,ps,proj=_runtime_speedups(layer,c2,x,expected_decode_tokens);del c2
        except Exception:sp=ps=proj=None
        if sp is None or (sp>=min_decode_speedup and proj>=min_projected_speedup):
            return LayerDecisionV5(name,d2.method,2,d2.group_size,d2.outlier_columns,d2.residual_rank,d2.nmse,d2.cosine,d2.compression,sp,ps,proj,'INT2 passes quality + measured decode/projected-runtime guard')
        int2_rejected_for_speed=True

    # If INT2 is inaccurate OR measured slower, promote only this layer to INT4.
    with torch.no_grad():ref=F.linear(x,layer.weight.to(dt),None if layer.bias is None else layer.bias.to(dt))
    tried=[]
    for g in group_sizes:
        if layer.in_features<g:continue
        for oc in outliers:
            if oc>=layer.in_features:continue
            try:
                cand=HybridInt4Linear(layer,g,rms,oc).to(dev)
                with torch.no_grad():y=cand(x)
                nm,co=quality(ref,y);comp=_int4_bytes(layer,cand);tried.append((nm,co,comp,g,oc));del cand
            except Exception:continue
    passing=[t for t in tried if t[0]<=nmse_limit and t[1]>=cosine_limit and t[2]>=min_compression]
    # Try candidates in quality/compression order until one also wins the decode timing.
    for b in sorted(passing,key=lambda t:(t[2],-t[0],t[1]),reverse=True):
        try:
            c4=HybridInt4Linear(layer,b[3],rms,b[4]).to(dev);sp,ps,proj=_runtime_speedups(layer,c4,x,expected_decode_tokens);del c4
        except Exception:sp=ps=proj=None
        if sp is None or (sp>=min_decode_speedup and proj>=min_projected_speedup):
            reason='INT4 promoted fallback passes quality and measured decode-speed guard'
            if int2_rejected_for_speed:reason='INT2 was slower than FP16; INT4 passes measured decode-speed guard'
            return LayerDecisionV5(name,'int4-outlier' if b[4] else 'int4',4,b[3],b[4],0,b[0],b[1],b[2],sp,ps,proj,reason)
    if tried:
        b=min(tried,key=lambda t:t[0]);reason='kept FP16: no packed candidate passed quality + measured decode speed'
        return LayerDecisionV5(name,'fp16',16,b[3],b[4],0,b[0],b[1],1.0,1.0,1.0,1.0,reason)
    return LayerDecisionV5(name,'fp16',16,128,0,0,float('inf'),0.0,1.0,1.0,1.0,1.0,'no viable packed candidate')


def build_plan_v5(model,captures,group_sizes=(128,64,32),outliers=(0,8,16,32),residual_ranks=(0,4,8),
                  nmse_limit=.06,cosine_limit=.96,min_compression=1.45,min_decode_speedup=1.0,min_projected_speedup=1.02,expected_decode_tokens=64):
    mods=dict(model.named_modules());out=[]
    for name,x in captures.items():
        layer=mods.get(name)
        if isinstance(layer,nn.Linear):
            out.append(decide_linear_v5(name,layer,x,group_sizes,outliers,residual_ranks,nmse_limit,cosine_limit,min_compression,min_decode_speedup,min_projected_speedup,expected_decode_tokens))
    return out
