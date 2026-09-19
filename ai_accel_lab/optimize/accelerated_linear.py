from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F

from ..kernels.groupwise_int2 import GroupwiseInt2Weight, pack_groupwise_int2, unpack_groupwise_int2
from ..kernels.arch_dispatch import run_groupwise_int2


@dataclass
class LinearOptimizationInfo:
    method: str
    group_size: int
    residual_rank: int
    original_bytes: int
    optimized_bytes: int
    estimated_compression: float
    activation_aware: bool


class GroupwiseInt2Linear(nn.Module):
    """Drop-in nn.Linear replacement with packed groupwise INT2 + optional low-rank residual."""
    def __init__(self, source: nn.Linear, group_size=128, activation_rms=None,
                 residual_rank: int = 0, backend: str = "auto"):
        super().__init__()
        w = source.weight.detach()
        pw = pack_groupwise_int2(w, group_size=group_size, activation_rms=activation_rms)
        self.register_buffer("packed", pw.packed)
        self.register_buffer("scales", pw.scales)
        self.original_k = pw.original_k
        self.padded_k = pw.padded_k
        self.group_size = pw.group_size
        self.out_features = source.out_features
        self.in_features = source.in_features
        self.backend = backend
        self.activation_aware = pw.activation_aware

        if source.bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", source.bias.detach().to(torch.float16))

        rank = max(0, min(int(residual_rank), min(w.shape)))
        self.residual_rank = rank
        if rank:
            q = unpack_groupwise_int2(pw, dtype=torch.float32).to(w.device)
            err = w.detach().float() - q
            U,S,Vh = torch.linalg.svd(err, full_matrices=False)
            ur = (U[:,:rank] * S[:rank]).to(torch.float16)
            vr = Vh[:rank,:].to(torch.float16)
            self.register_buffer("res_u", ur)
            self.register_buffer("res_v", vr)
        else:
            self.res_u = None
            self.res_v = None

    def packed_weight(self) -> GroupwiseInt2Weight:
        return GroupwiseInt2Weight(
            packed=self.packed,
            scales=self.scales,
            original_k=self.original_k,
            padded_k=self.padded_k,
            group_size=self.group_size,
            activation_aware=self.activation_aware,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape=x.shape
        x2=x.reshape(-1,shape[-1])
        if self.padded_k != self.original_k:
            x2=F.pad(x2,(0,self.padded_k-self.original_k))
        bias = None if self.bias is None else self.bias.to(x2.dtype)
        try:
            y=run_groupwise_int2(x2,self.packed_weight(),bias,backend=self.backend)
        except Exception:
            w=unpack_groupwise_int2(self.packed_weight(),dtype=x2.dtype)
            y=F.linear(x2[...,:self.original_k],w,bias)

        if self.residual_rank:
            xr=x.reshape(-1,self.original_k).to(self.res_v.dtype)
            corr=F.linear(F.linear(xr,self.res_v),self.res_u).to(y.dtype)
            y=y+corr
        return y.reshape(*shape[:-1],self.out_features)

    def optimization_info(self) -> LinearOptimizationInfo:
        orig = self.out_features*self.in_features*2 + (0 if self.bias is None else self.out_features*2)
        packed = self.packed.numel()*self.packed.element_size()+self.scales.numel()*self.scales.element_size()
        if self.bias is not None: packed += self.bias.numel()*self.bias.element_size()
        if self.residual_rank:
            packed += self.res_u.numel()*self.res_u.element_size()+self.res_v.numel()*self.res_v.element_size()
        return LinearOptimizationInfo(
            method="groupwise-int2-residual" if self.residual_rank else "groupwise-int2",
            group_size=self.group_size,residual_rank=self.residual_rank,
            original_bytes=orig,optimized_bytes=packed,
            estimated_compression=orig/max(1,packed),activation_aware=self.activation_aware,
        )
