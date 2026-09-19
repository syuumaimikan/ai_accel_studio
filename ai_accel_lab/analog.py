from __future__ import annotations
import time
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from .quant import quantize_per_row,dequantize_per_row,bitpack_estimated_bytes
from .calibration import svd_residual_factors
from .benchmark import make_weight,LinearPreset
from .layers import DenseLinear
from .metrics import output_metrics
from .utils import resolve_device,sync,seed_all

class AnalogResidualLinear(nn.Module):
    def __init__(self,weight,bias=None,bits=4,noise_std=0.01,rank=8):
        super().__init__(); self.bits=bits; self.noise_std=noise_std; self.rank=rank
        q,s=quantize_per_row(weight,bits); self.register_buffer('q',q); self.register_buffer('s',s)
        approx=dequantize_per_row(q,s,torch.float32); U,V=svd_residual_factors(weight.float()-approx,rank)
        self.register_buffer('u',U.half() if U is not None else torch.empty(0)); self.register_buffer('v',V.half() if V is not None else torch.empty(0))
        self.bias=None if bias is None else nn.Parameter(bias.detach().float().clone(),requires_grad=False)
    def forward(self,x):
        w=dequantize_per_row(self.q,self.s,x.dtype)
        if self.noise_std: w=w+torch.randn_like(w)*(w.abs().mean().clamp_min(1e-8)*self.noise_std)
        y=F.linear(x,w)
        if self.rank>0:y=y+F.linear(F.linear(x,self.v.to(x.dtype)),self.u.to(x.dtype))
        if self.bias is not None:y=y+self.bias.to(x.dtype)
        return y
    def estimated_weight_bytes(self):
        n=bitpack_estimated_bytes(self.q.numel(),self.bits)+self.s.numel()*4+self.u.numel()*2+self.v.numel()*2
        if self.bias is not None:n+=self.bias.numel()*4
        return int(n)

def run_analog(device_arg='auto',features=1024,batch=16,bits=4,noise=0.01,rank=8,repeats=50):
    seed_all(); device=resolve_device(device_arg); p=LinearPreset(batch,features,features,5,repeats); w,b=make_weight(p,device); x=torch.randn(batch,features,device=device)
    dense=DenseLinear(w,b); ana=AnalogResidualLinear(w,b,bits,noise,rank).to(device); ref=dense(x)
    for _ in range(5):ana(x)
    sync(device); t=time.perf_counter(); y=None
    for _ in range(repeats):y=ana(x)
    sync(device); ms=(time.perf_counter()-t)*1000/repeats
    return pd.Series({'device':str(device),'software_sim_latency_ms':ms,'bits':bits,'noise_std':noise,'rank':rank,'estimated_weight_bytes':ana.estimated_weight_bytes(),**output_metrics(ref,y)})
