from __future__ import annotations
from dataclasses import dataclass,asdict

@dataclass
class ModelArchitecture:
    family:str
    hidden_size:int|None
    num_layers:int|None
    num_heads:int|None
    num_kv_heads:int|None
    head_dim:int|None
    gqa_ratio:int|None
    fused_decode_support:str
    def to_dict(self): return asdict(self)

def inspect_model_architecture(model):
    c=model.config
    mt=str(getattr(c,'model_type','unknown')).lower()
    family=('qwen' if 'qwen' in mt else 'llama' if 'llama' in mt else 'mistral' if 'mistral' in mt else 'gemma' if 'gemma' in mt else 'smollm' if 'smol' in mt else mt)
    h=getattr(c,'hidden_size',None)
    nl=getattr(c,'num_hidden_layers',getattr(c,'n_layer',None))
    nh=getattr(c,'num_attention_heads',None)
    nkh=getattr(c,'num_key_value_heads',nh)
    hd=getattr(c,'head_dim',None) or ((h//nh) if h and nh else None)
    ratio=(nh//nkh) if nh and nkh and nh%nkh==0 else None
    support='mha-fused' if nh==nkh and hd in (64,128) else ('gqa-adapter' if ratio and hd in (64,128) else 'fallback')
    return ModelArchitecture(family,h,nl,nh,nkh,hd,ratio,support)
