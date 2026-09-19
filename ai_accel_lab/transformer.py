from __future__ import annotations
from dataclasses import dataclass, replace
import math,time
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from .layers import DenseLinear,TernaryResidualLinear,PreQuantLinear,NMPrunedLinear,BlockSparseLinear
from .metrics import output_metrics
from .utils import resolve_device,sync,seed_all
from .power import NvmlPowerSampler


class SelfAttention(nn.Module):
    def __init__(self,d_model,n_heads):
        super().__init__(); assert d_model%n_heads==0
        self.d_model=d_model; self.n_heads=n_heads; self.head_dim=d_model//n_heads
        self.qkv=nn.Linear(d_model,3*d_model,bias=False); self.out=nn.Linear(d_model,d_model,bias=False)
    def forward(self,x):
        b,t,d=x.shape; qkv=self.qkv(x).view(b,t,3,self.n_heads,self.head_dim).permute(2,0,3,1,4)
        q,k,v=qkv[0],qkv[1],qkv[2]
        y=F.scaled_dot_product_attention(q,k,v,is_causal=False)
        y=y.transpose(1,2).contiguous().view(b,t,d)
        return self.out(y)


class TransformerBlock(nn.Module):
    def __init__(self,d_model=256,n_heads=8,ff_mult=4):
        super().__init__(); h=d_model*ff_mult
        self.norm1=nn.LayerNorm(d_model); self.attn=SelfAttention(d_model,n_heads); self.norm2=nn.LayerNorm(d_model)
        self.fc1=nn.Linear(d_model,h); self.fc2=nn.Linear(h,d_model)
    def forward(self,x):
        x=x+self.attn(self.norm1(x)); return x+self.fc2(F.gelu(self.fc1(self.norm2(x))))


class ApproxTransformerBlock(nn.Module):
    def __init__(self,base:TransformerBlock,method='ternary',rank=8):
        super().__init__(); self.norm1=base.norm1; self.norm2=base.norm2; self.attn=base.attn
        if method=='ternary': maker=lambda l:TernaryResidualLinear.from_linear(l,rank=rank)
        elif method=='int4': maker=lambda l:PreQuantLinear.from_linear(l,bits=4)
        elif method=='nm2_4': maker=lambda l:NMPrunedLinear(l.weight,l.bias,2,4)
        elif method=='block50': maker=lambda l:BlockSparseLinear(l.weight,l.bias,block_size=32,keep_ratio=0.5)
        else: raise ValueError(method)
        self.fc1=maker(base.fc1); self.fc2=maker(base.fc2); self.method=method
    def forward(self,x):
        x=x+self.attn(self.norm1(x)); return x+self.fc2(F.gelu(self.fc1(self.norm2(x))))
    def estimated_weight_bytes(self): return self.fc1.estimated_weight_bytes()+self.fc2.estimated_weight_bytes()


@dataclass(frozen=True)
class TransformerPreset:
    batch:int; seq:int; d_model:int; heads:int; ff_mult:int; warmup:int; repeats:int

PRESETS={
    'tiny':TransformerPreset(2,64,128,4,4,5,20),
    'small':TransformerPreset(4,128,256,8,4,5,20),
    'base':TransformerPreset(4,256,512,8,4,5,15),
}


def _time(mod,x,p,device,power):
    with torch.no_grad():
        for _ in range(p.warmup): _=mod(x)
        sync(device); sampler=NvmlPowerSampler() if power and device.type=='cuda' else None
        if sampler:sampler.start()
        t=time.perf_counter(); y=None
        for _ in range(p.repeats): y=mod(x)
        sync(device); elapsed=time.perf_counter()-t; avgp=sampler.stop() if sampler else None
    ms=elapsed*1000/p.repeats; energy=None if avgp is None else avgp*ms
    return y,ms,avgp,energy


def run_transformer(device_arg='auto',preset_name='tiny',repeats=None,power=False,csv=None):
    seed_all(); device=resolve_device(device_arg); p=PRESETS[preset_name]
    if repeats:p=replace(p,repeats=repeats)
    base=TransformerBlock(p.d_model,p.heads,p.ff_mult).to(device).eval(); x=torch.randn(p.batch,p.seq,p.d_model,device=device)
    with torch.no_grad(): ref=base(x)
    methods={'dense':base}
    for n in ('ternary','int4','nm2_4','block50'):
        # Deepcopy ensures each approximation gets the same baseline state.
        import copy
        bcopy=copy.deepcopy(base)
        methods[n]=ApproxTransformerBlock(bcopy,n,rank=8).to(device).eval()
    rows=[]; base_mlp_bytes=(base.fc1.weight.numel()+base.fc2.weight.numel())*base.fc1.weight.element_size()
    for name,mod in methods.items():
        y,ms,avgp,energy=_time(mod,x,p,device,power)
        m=output_metrics(ref,y)
        wb=base_mlp_bytes if name=='dense' else mod.estimated_weight_bytes()
        rows.append({'method':name,'latency_ms':ms,'estimated_mlp_weight_bytes':wb,'mlp_weight_reduction':1-wb/base_mlp_bytes,
                     'avg_power_w':avgp,'energy_mj_per_inference':energy,'edp_mj_ms':None if energy is None else energy*ms,**m})
    df=pd.DataFrame(rows); base_ms=float(df.loc[df.method=='dense','latency_ms'].iloc[0]); df['speedup_vs_dense']=base_ms/df.latency_ms
    df=df.sort_values('latency_ms').reset_index(drop=True)
    if csv: df.to_csv(csv,index=False)
    return df,device,p
