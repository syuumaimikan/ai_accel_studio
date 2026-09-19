from __future__ import annotations
from dataclasses import dataclass,asdict
import torch
from torch import nn
import torch.nn.functional as F
from .planner import collect_linear_inputs, DEFAULT_TARGET_TOKENS
from .hybrid_int2 import HybridInt2Linear

@dataclass
class LayerDecisionV4:
    name:str
    method:str
    group_size:int
    outlier_columns:int
    residual_rank:int
    nmse:float
    cosine:float
    compression:float
    reason:str
    def to_dict(self): return asdict(self)

def quality(ref,cand):
    d=cand.float()-ref.float(); mse=d.square().mean().item(); denom=ref.float().square().mean().item()+1e-12
    a=ref.float().reshape(ref.shape[0],-1); b=cand.float().reshape(cand.shape[0],-1)
    return mse/denom,F.cosine_similarity(a,b,dim=-1).mean().item()

def decide_linear_v4(name,layer,sample_x,group_sizes=(128,64,32),outliers=(0,8,16,32),residual_ranks=(0,4,8),
                     nmse_limit=.06,cosine_limit=.96,min_compression=1.5):
    dev=layer.weight.device; dt=torch.float16 if dev.type=='cuda' else torch.float32
    x=sample_x.to(dev,dtype=dt); rms=sample_x.float().square().mean(0).sqrt().to(dev)
    with torch.no_grad(): ref=F.linear(x,layer.weight.to(dt),None if layer.bias is None else layer.bias.to(dt))
    tried=[]
    # First search zero-residual paths. Prefer the highest compression that passes.
    for g in group_sizes:
        if layer.in_features < g: continue
        for oc in outliers:
            if oc>=layer.in_features: continue
            try:
                cand=HybridInt2Linear(layer,g,rms,oc,0,'auto').to(dev)
                with torch.no_grad(): y=cand(x)
                nm,co=quality(ref,y); inf=cand.optimization_info()
                tried.append((nm,co,inf.compression,g,oc,0))
                del cand
            except Exception:
                continue
    passing=[t for t in tried if t[0]<=nmse_limit and t[1]>=cosine_limit and t[2]>=min_compression]
    if passing:
        # accuracy first enough to pass, then compression; modest preference to fewer outliers
        best=max(passing,key=lambda t:(t[2],-t[0],t[1]))
        return LayerDecisionV4(name,'int2-outlier' if best[4] else 'int2',best[3],best[4],0,best[0],best[1],best[2],'passes local guard')
    # Only run SVD for the best few zero-rank candidates, avoiding combinatorial SVD cost.
    seeds=sorted(tried,key=lambda t:(t[0],-t[1]))[:2]
    for seed in seeds:
        for rank in [r for r in residual_ranks if r>0]:
            try:
                cand=HybridInt2Linear(layer,seed[3],rms,seed[4],rank,'auto').to(dev)
                with torch.no_grad(): y=cand(x)
                nm,co=quality(ref,y); inf=cand.optimization_info()
                t=(nm,co,inf.compression,seed[3],seed[4],rank)
                tried.append(t); del cand
                if nm<=nmse_limit and co>=cosine_limit and inf.compression>=min_compression:
                    return LayerDecisionV4(name,'int2-outlier-residual',seed[3],seed[4],rank,nm,co,inf.compression,'passes local guard with residual')
            except Exception:
                continue
    if not tried:
        return LayerDecisionV4(name,'fp16',128,0,0,float('inf'),0.0,1.0,'no viable quantized candidate')
    b=min(tried,key=lambda t:t[0])
    return LayerDecisionV4(name,'fp16',b[3],b[4],b[5],b[0],b[1],1.0,'kept dense: local guard failed')

def build_plan_v4(model,captures,group_sizes=(128,64,32),outliers=(0,8,16,32),residual_ranks=(0,4,8),
                  nmse_limit=.06,cosine_limit=.96,min_compression=1.5):
    mods=dict(model.named_modules()); out=[]
    for name,x in captures.items():
        layer=mods.get(name)
        if isinstance(layer,nn.Linear):
            out.append(decide_linear_v4(name,layer,x,group_sizes,outliers,residual_ranks,nmse_limit,cosine_limit,min_compression))
    return out
