from __future__ import annotations
import json
from pathlib import Path
import torch
from torch import nn

from .planner import LayerDecision
from .planner_v4 import LayerDecisionV4
from .planner_v5 import LayerDecisionV5
from .accelerated_linear import GroupwiseInt2Linear
from .hybrid_int2 import HybridInt2Linear
from .hybrid_int4 import HybridInt4Linear


def save_bundle(model,tokenizer,source_model_id,decisions,out_dir,extra=None):
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    version=int((extra or {}).get('format_version',4)); manifest={'format':f'ai-accel-studio-bundle-v{version}','source_model':source_model_id,
              'layers':[d.to_dict() for d in decisions],'extra':extra or {}}
    (out/'accel_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    # state_dict contains packed INT2/scales/outliers/residuals, not original dense
    # weights for replaced layers.
    torch.save(model.state_dict(),out/'optimized_state.pt')
    try:tokenizer.save_pretrained(out/'tokenizer')
    except Exception:pass
    return out


def _parent(root,fqn):
    ps=fqn.split('.');o=root
    for p in ps[:-1]:o=getattr(o,p)
    return o,ps[-1]


def _prepare_modules(model,decisions):
    for d in decisions:
        if d.method=='fp16':continue
        p,leaf=_parent(model,d.name);src=getattr(p,leaf)
        if not isinstance(src,nn.Linear):continue
        if isinstance(d,LayerDecisionV5):
            if d.bits==2: repl=HybridInt2Linear(src,d.group_size,None,d.outlier_columns,d.residual_rank,'auto')
            elif d.bits==4: repl=HybridInt4Linear(src,d.group_size,None,d.outlier_columns)
            else: continue
        elif isinstance(d,LayerDecisionV4):
            repl=HybridInt2Linear(src,d.group_size,None,d.outlier_columns,d.residual_rank,'auto')
        else:
            repl=GroupwiseInt2Linear(src,d.group_size,None,d.residual_rank,'auto')
        setattr(p,leaf,repl.to(src.weight.device))


def load_bundle(out_dir,device='cuda',dtype=torch.float16,trust_remote_code=False):
    from transformers import AutoModelForCausalLM,AutoTokenizer
    out=Path(out_dir);manifest=json.loads((out/'accel_manifest.json').read_text(encoding='utf-8'));source=manifest['source_model']
    kwargs={'dtype':dtype,'trust_remote_code':trust_remote_code}
    if device=='cuda':kwargs['device_map']='cuda'
    model=AutoModelForCausalLM.from_pretrained(source,**kwargs)
    if device!='cuda':model=model.to(device)
    fmt=manifest.get('format','')
    if fmt.endswith('v5'):
        decisions=[LayerDecisionV5(**x) for x in manifest['layers']]
    elif fmt.endswith('v4'):
        decisions=[LayerDecisionV4(**x) for x in manifest['layers']]
    else:
        decisions=[LayerDecision(**x) for x in manifest['layers']]
    _prepare_modules(model,decisions)
    state=torch.load(out/'optimized_state.pt',map_location=device,weights_only=True)
    missing,unexpected=model.load_state_dict(state,strict=False)
    tok_path=out/'tokenizer';tokenizer=AutoTokenizer.from_pretrained(tok_path if tok_path.exists() else source,trust_remote_code=trust_remote_code)
    model.eval();return model,tokenizer,manifest,missing,unexpected
