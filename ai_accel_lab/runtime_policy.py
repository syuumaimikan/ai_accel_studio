from __future__ import annotations
from dataclasses import dataclass
import torch

@dataclass
class RuntimePolicy:
    architecture:str
    decode_backend:str
    prefill_backend:str
    attention_backend:str
    reason:str

def choose_runtime_policy(device=None,m=1):
    if not torch.cuda.is_available():
        return RuntimePolicy('cpu','reference','torch','sdpa','CUDA unavailable')
    idx=torch.cuda.current_device() if device is None else int(device)
    major,minor=torch.cuda.get_device_capability(idx)
    if major>=10:
        return RuntimePolicy('blackwell','decode-int2-gemv' if m<=4 else 'tcgen05/tma','tcgen05/tma','flash-attn3','SM100+: Blackwell low-latency + tensor-memory path')
    if major==9:
        return RuntimePolicy('hopper','decode-int2-gemv' if m<=4 else 'wgmma/tma','wgmma/tma','flash-attn3','SM90: Hopper WGMMA/TMA path')
    if major>=8:
        return RuntimePolicy('ampere-ada','decode-int2-gemv' if m<=4 else 'triton-int2','tensorcore','flash-attn2','SM80+: specialized decode + tensorcore baseline')
    return RuntimePolicy(f'sm{major}{minor}','triton-int2','torch','sdpa','legacy CUDA fallback')
