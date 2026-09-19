from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn.functional as F

from .packing import pack_int2, pack_ternary, pack_int2_sparse24, unpack_int2, unpack_ternary, unpack_int2_sparse24


@dataclass
class DispatchInfo:
    backend: str
    format: str
    reason: str


class FusedPackedLinear(torch.nn.Module):
    """
    Inference-only linear with packed low-bit weights and automatic GPU backend.

    backend='auto' priority:
      Triton -> raw CUDA extension -> PyTorch dequantized reference.
    """
    def __init__(self, weight, bias=None, mode='int2', backend='auto', activation='none'):
        super().__init__()
        self.mode=mode
        self.backend=backend
        self.activation=activation
        self.original_k=weight.shape[1]
        if mode=='int2': pw=pack_int2(weight)
        elif mode=='ternary': pw=pack_ternary(weight)
        elif mode in ('int2_2to4','sparse24_int2'): pw=pack_int2_sparse24(weight)
        else: raise ValueError(mode)
        self.register_buffer('packed',pw.packed)
        self.register_buffer('scale',pw.scale)
        self.padded_k=pw.padded_k
        self.format=pw.format
        self.register_buffer('bias_buf', torch.empty(0) if bias is None else bias.detach().to(torch.float16))
        self.last_dispatch=None

    def _pw(self):
        from .packing import PackedWeight, PackedSparse24
        if self.mode in ('int2_2to4','sparse24_int2'):
            return PackedSparse24(self.packed,self.scale,self.original_k,self.padded_k)
        return PackedWeight(self.packed,self.scale,self.original_k,self.padded_k,self.format)

    def _pad_x(self,x):
        if x.shape[-1]==self.padded_k: return x
        if x.shape[-1]!=self.original_k: raise ValueError('unexpected input K')
        return F.pad(x,(0,self.padded_k-self.original_k))

    def _bias(self, x):
        return None if self.bias_buf.numel()==0 else self.bias_buf.to(x.device,x.dtype)

    def forward(self,x):
        x2=x.reshape(-1,x.shape[-1])
        x2=self._pad_x(x2)
        pw=self._pw()
        candidates=[self.backend] if self.backend!='auto' else ['triton','cuda','reference']
        errors=[]
        for be in candidates:
            try:
                if be=='triton':
                    from . import triton_fused as tf
                    if self.mode=='int2': y=tf.int2_linear(x2,pw,self._bias(x2),self.activation)
                    elif self.mode=='ternary': y=tf.ternary_linear(x2,pw,self._bias(x2),self.activation)
                    else: y=tf.int2_sparse24_linear(x2,pw,self._bias(x2),self.activation)
                elif be=='cuda':
                    from . import cuda_ext as ce
                    if self.mode=='int2': y=ce.int2_linear(x2,pw,self._bias(x2),self.activation)
                    elif self.mode=='ternary': y=ce.ternary_linear(x2,pw,self._bias(x2),self.activation)
                    else: y=ce.int2_sparse24_linear(x2,pw,self._bias(x2),self.activation)
                elif be=='reference':
                    if self.mode=='int2': w=unpack_int2(pw,dtype=x2.dtype)
                    elif self.mode=='ternary': w=unpack_ternary(pw,dtype=x2.dtype)
                    else: w=unpack_int2_sparse24(pw,dtype=x2.dtype)
                    y=F.linear(x2[:,:self.original_k],w.to(x2.device),self._bias(x2))
                    if self.activation=='relu': y=F.relu(y)
                    elif self.activation in ('silu','swish'): y=F.silu(y)
                else: raise ValueError(be)
                self.last_dispatch=DispatchInfo(be,self.format,'ok')
                return y.view(*x.shape[:-1],y.shape[-1])
            except Exception as e:
                errors.append(f'{be}: {e}')
                if self.backend!='auto': raise
        raise RuntimeError('all backends failed: '+' | '.join(errors))
