from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F
from .quant import (
    bitpack_estimated_bytes, quantize_per_row, dequantize_per_row,
    fake_quant_activation, ternarize_weight,
)
from .calibration import svd_residual_factors, activation_aware_factors


class ExperimentalLinear(nn.Module):
    def theoretical_macs(self, batch_items: int) -> int: raise NotImplementedError
    def estimated_weight_bytes(self) -> int: raise NotImplementedError


class DenseLinear(ExperimentalLinear):
    def __init__(self, weight, bias=None):
        super().__init__()
        self.weight = nn.Parameter(weight.detach().clone(), requires_grad=False)
        self.bias = None if bias is None else nn.Parameter(bias.detach().clone(), requires_grad=False)

    @classmethod
    def from_linear(cls, mod: nn.Linear): return cls(mod.weight, mod.bias)
    def forward(self, x): return F.linear(x, self.weight, self.bias)
    def theoretical_macs(self, batch_items): return batch_items * self.weight.numel()
    def estimated_weight_bytes(self):
        n = self.weight.numel()*self.weight.element_size()
        if self.bias is not None: n += self.bias.numel()*self.bias.element_size()
        return int(n)


class PreQuantLinear(ExperimentalLinear):
    """Weight is quantized once at construction. Activation fake-quant is optional."""
    def __init__(self, weight, bias=None, bits=4, quantize_activation=True):
        super().__init__(); self.bits=int(bits); self.quantize_activation=quantize_activation
        q, s = quantize_per_row(weight, self.bits)
        self.register_buffer('qweight', q)
        self.register_buffer('scale', s)
        self.bias = None if bias is None else nn.Parameter(bias.detach().float().clone(), requires_grad=False)

    @classmethod
    def from_linear(cls, mod, **kw): return cls(mod.weight, mod.bias, **kw)
    def forward(self, x):
        qx = fake_quant_activation(x, self.bits) if self.quantize_activation else x
        w = dequantize_per_row(self.qweight, self.scale, x.dtype)
        b = None if self.bias is None else self.bias.to(x.dtype)
        return F.linear(qx, w, b)
    def theoretical_macs(self, batch_items): return batch_items*self.qweight.numel()
    def estimated_weight_bytes(self):
        b = bitpack_estimated_bytes(self.qweight.numel(), self.bits) + self.scale.numel()*4
        if self.bias is not None: b += self.bias.numel()*4
        return int(b)


class TernaryResidualLinear(ExperimentalLinear):
    def __init__(self, weight, bias=None, rank=8, threshold_factor=0.7,
                 calibration_x=None, activation_aware=False, residual_dtype=torch.float16):
        super().__init__(); self.rank=int(rank)
        t, scale, approx = ternarize_weight(weight, threshold_factor)
        self.register_buffer('ternary', t)
        self.register_buffer('scale', scale)
        residual = weight.detach().float() - approx.float()
        if self.rank > 0:
            if activation_aware and calibration_x is not None:
                U, V = activation_aware_factors(residual, calibration_x, self.rank)
            else:
                U, V = svd_residual_factors(residual, self.rank)
            self.register_buffer('res_u', U.to(residual_dtype))
            self.register_buffer('res_v', V.to(residual_dtype))
        else:
            self.res_u=None; self.res_v=None
        self.bias = None if bias is None else nn.Parameter(bias.detach().float().clone(), requires_grad=False)

    @classmethod
    def from_linear(cls, mod, **kw): return cls(mod.weight, mod.bias, **kw)
    def forward(self, x):
        w = self.ternary.to(x.dtype) * self.scale.to(x.dtype).unsqueeze(1)
        y = F.linear(x, w, None)
        if self.rank > 0:
            y = y + F.linear(F.linear(x, self.res_v.to(x.dtype)), self.res_u.to(x.dtype))
        if self.bias is not None: y = y + self.bias.to(x.dtype)
        return y
    def theoretical_macs(self, batch_items):
        nz = int(self.ternary.count_nonzero().item())
        main = batch_items*nz
        if self.rank <= 0: return main
        o,i = self.ternary.shape
        return main + batch_items*self.rank*(i+o)
    def estimated_weight_bytes(self):
        n = bitpack_estimated_bytes(self.ternary.numel(),2) + self.scale.numel()*4
        if self.rank>0: n += self.res_u.numel()*self.res_u.element_size()+self.res_v.numel()*self.res_v.element_size()
        if self.bias is not None: n += self.bias.numel()*4
        return int(n)


class DynamicTopKLinear(ExperimentalLinear):
    """Per-sample top-k activation sparsity. Good for algorithm tests; gather can be expensive."""
    def __init__(self, weight, bias=None, keep_ratio=0.25):
        super().__init__(); self.keep_ratio=float(keep_ratio)
        self.weight=nn.Parameter(weight.detach().clone(),requires_grad=False)
        self.bias=None if bias is None else nn.Parameter(bias.detach().clone(),requires_grad=False)
    def forward(self,x):
        orig=x.shape; flat=x.reshape(-1,orig[-1]); k=max(1,int(round(orig[-1]*self.keep_ratio)))
        idx=torch.topk(flat.abs(),k,dim=-1,sorted=False).indices
        xs=torch.gather(flat,1,idx)
        ws=self.weight[:,idx].permute(1,0,2)
        y=torch.bmm(ws,xs.unsqueeze(-1)).squeeze(-1)
        if self.bias is not None: y=y+self.bias
        return y.reshape(*orig[:-1],self.weight.shape[0])
    def theoretical_macs(self,batch_items):
        o,i=self.weight.shape; k=max(1,int(round(i*self.keep_ratio))); return batch_items*o*k
    def estimated_weight_bytes(self): return DenseLinear(self.weight,self.bias).estimated_weight_bytes()


class BlockSparseLinear(ExperimentalLinear):
    """Selects input blocks using batch-average activation. One shared block set reduces gather overhead."""
    def __init__(self, weight, bias=None, block_size=32, keep_ratio=0.25):
        super().__init__(); self.block_size=int(block_size); self.keep_ratio=float(keep_ratio)
        self.weight=nn.Parameter(weight.detach().clone(),requires_grad=False)
        self.bias=None if bias is None else nn.Parameter(bias.detach().clone(),requires_grad=False)
        self.last_kept=weight.shape[1]
    def forward(self,x):
        i=x.shape[-1]; pad=(-i)%self.block_size
        xp=F.pad(x,(0,pad)) if pad else x
        nblocks=xp.shape[-1]//self.block_size
        score=xp.reshape(-1,nblocks,self.block_size).abs().mean(dim=(0,2))
        kb=max(1,int(math.ceil(nblocks*self.keep_ratio)))
        blocks=torch.topk(score,kb,sorted=False).indices
        idx=(blocks[:,None]*self.block_size+torch.arange(self.block_size,device=x.device)[None,:]).reshape(-1)
        idx=idx[idx<i]; self.last_kept=int(idx.numel())
        xs=x.index_select(-1,idx); ws=self.weight.index_select(1,idx)
        return F.linear(xs,ws,self.bias)
    def theoretical_macs(self,batch_items): return batch_items*self.weight.shape[0]*self.last_kept
    def estimated_weight_bytes(self): return DenseLinear(self.weight,self.bias).estimated_weight_bytes()


class NMPrunedLinear(ExperimentalLinear):
    """Static structured N:M pruning (e.g. 2:4). Dense fallback forward, sparse MAC accounting."""
    def __init__(self, weight, bias=None, n=2, m=4):
        super().__init__(); self.n=int(n); self.m=int(m)
        if not (0 < n <= m): raise ValueError('require 0<n<=m')
        w=weight.detach().clone(); o,i=w.shape; pad=(-i)%m
        wp=F.pad(w,(0,pad)) if pad else w
        groups=wp.reshape(o,-1,m)
        idx=torch.topk(groups.abs(),k=n,dim=-1).indices
        mask=torch.zeros_like(groups,dtype=torch.bool); mask.scatter_(-1,idx,True)
        pruned=(groups*mask).reshape(o,-1)[:,:i]
        self.register_buffer('weight',pruned)
        self.bias=None if bias is None else nn.Parameter(bias.detach().clone(),requires_grad=False)
        self.nonzero=int(pruned.count_nonzero().item())
    def forward(self,x): return F.linear(x,self.weight,self.bias)
    def theoretical_macs(self,batch_items): return batch_items*self.nonzero
    def estimated_weight_bytes(self):
        # value bits are assumed fp16 + mask metadata approximation
        val=self.nonzero*2; groups=math.ceil(self.weight.shape[1]/self.m)*self.weight.shape[0]
        metadata=bitpack_estimated_bytes(groups*self.m,1)
        b=val+metadata+(0 if self.bias is None else self.bias.numel()*4)
        return int(b)


class DeltaLinear(ExperimentalLinear):
    def __init__(self, weight, bias=None, threshold=0.0):
        super().__init__(); self.threshold=float(threshold)
        self.weight=nn.Parameter(weight.detach().clone(),requires_grad=False)
        self.bias=None if bias is None else nn.Parameter(bias.detach().clone(),requires_grad=False)
        self.prev_x=None; self.prev_y=None; self.last_change_ratio=1.0
    def reset_state(self): self.prev_x=None; self.prev_y=None; self.last_change_ratio=1.0
    def forward(self,x):
        if self.prev_x is None or self.prev_x.shape!=x.shape:
            y=F.linear(x,self.weight,self.bias); self.prev_x=x.detach().clone(); self.prev_y=y.detach().clone(); return y
        d=x-self.prev_x; mask=d.abs()>self.threshold; self.last_change_ratio=mask.float().mean().item()
        # Exact sparse update per row. Intended for small-change streaming workloads.
        flat=d.reshape(-1,d.shape[-1]); mf=mask.reshape_as(flat); updates=[]
        for row,m in zip(flat,mf):
            idx=torch.nonzero(m,as_tuple=False).flatten()
            if idx.numel()==0: updates.append(torch.zeros(self.weight.shape[0],device=x.device,dtype=x.dtype))
            else: updates.append(torch.mv(self.weight[:,idx],row[idx]))
        upd=torch.stack(updates).reshape(*x.shape[:-1],self.weight.shape[0])
        y=self.prev_y+upd; self.prev_x=x.detach().clone(); self.prev_y=y.detach().clone(); return y
    def theoretical_macs(self,batch_items): return int(batch_items*self.weight.numel()*self.last_change_ratio)
    def estimated_weight_bytes(self): return DenseLinear(self.weight,self.bias).estimated_weight_bytes()


class HybridRouterLinear(ExperimentalLinear):
    """Routes each sample between INT2/INT4/ternary/dense based on mean |activation|."""
    def __init__(self, weight, bias=None, ternary_rank=8, thresholds=(0.35,0.7,1.1)):
        super().__init__(); self.thresholds=tuple(float(x) for x in thresholds)
        self.int2=PreQuantLinear(weight,bias,2); self.int4=PreQuantLinear(weight,bias,4)
        self.ternary=TernaryResidualLinear(weight,bias,rank=ternary_rank); self.dense=DenseLinear(weight,bias)
        self.last_counts={'int2':0,'int4':0,'ternary':0,'dense':0}
    def forward(self,x):
        orig=x.shape; flat=x.reshape(-1,orig[-1]); score=flat.float().abs().mean(-1)
        t0,t1,t2=self.thresholds
        masks={'int2':score<t0,'int4':(score>=t0)&(score<t1),'ternary':(score>=t1)&(score<t2),'dense':score>=t2}
        y=torch.empty(flat.shape[0],self.dense.weight.shape[0],device=x.device,dtype=x.dtype)
        self.last_counts={}
        for name,m in masks.items():
            self.last_counts[name]=int(m.sum().item())
            if not m.any(): continue
            mod=getattr(self,name); y[m]=mod(flat[m])
        return y.reshape(*orig[:-1],y.shape[-1])
    def theoretical_macs(self,batch_items):
        return sum(getattr(self,k).theoretical_macs(v) for k,v in self.last_counts.items())
    def estimated_weight_bytes(self):
        # Deployment estimate assumes shared representations are packed and dense fallback is retained.
        return self.int2.estimated_weight_bytes()+self.int4.estimated_weight_bytes()+self.ternary.estimated_weight_bytes()+self.dense.estimated_weight_bytes()
