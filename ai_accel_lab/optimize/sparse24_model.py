from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from ..kernels.sparse24_native import make_sparse_weight

class Sparse24Linear(nn.Module):
    def __init__(self,src:nn.Linear):
        super().__init__()
        self.out_features=src.out_features; self.in_features=src.in_features
        sw=make_sparse_weight(src.weight.detach())
        self.weight=nn.Parameter(sw,requires_grad=False)
        if src.bias is None: self.bias=None
        else: self.register_buffer('bias',src.bias.detach().to(src.weight.dtype))
    def forward(self,x): return F.linear(x,self.weight,self.bias)

def apply_sparse24(model:nn.Module,target_tokens=("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj","fc1","fc2")):
    replacements=[]
    for name,m in model.named_modules():
        if isinstance(m,nn.Linear) and any(t in name for t in target_tokens): replacements.append(name)
    done=0
    for fqn in replacements:
        parts=fqn.split('.'); parent=model
        for p in parts[:-1]: parent=getattr(parent,p)
        src=getattr(parent,parts[-1])
        try:
            setattr(parent,parts[-1],Sparse24Linear(src)); done+=1
        except Exception:
            pass
    return done
