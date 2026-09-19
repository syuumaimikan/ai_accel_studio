from __future__ import annotations

import json
from pathlib import Path
import torch

from .model_optimizer import apply_plan
from .planner import LayerDecision


def save_bundle(model,tokenizer,source_model_id:str,decisions:list[LayerDecision],out_dir:str|Path,extra=None):
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    manifest={
        'format':'ai-accel-studio-bundle-v1',
        'source_model':source_model_id,
        'layers':[d.to_dict() for d in decisions],
        'extra':extra or {},
    }
    (out/'accel_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    torch.save(model.state_dict(),out/'optimized_state.pt')
    try: tokenizer.save_pretrained(out/'tokenizer')
    except Exception: pass
    return out


def load_bundle(out_dir:str|Path,device='cuda',dtype=torch.float16,trust_remote_code=False):
    from transformers import AutoModelForCausalLM,AutoTokenizer
    out=Path(out_dir); manifest=json.loads((out/'accel_manifest.json').read_text(encoding='utf-8'))
    source=manifest['source_model']
    kwargs={'torch_dtype':dtype,'trust_remote_code':trust_remote_code}
    if device=='cuda': kwargs['device_map']='cuda'
    model=AutoModelForCausalLM.from_pretrained(source,**kwargs)
    if device!='cuda': model=model.to(device)
    decisions=[LayerDecision(**x) for x in manifest['layers']]
    apply_plan(model,decisions,captures={})
    state=torch.load(out/'optimized_state.pt',map_location=device,weights_only=True)
    missing,unexpected=model.load_state_dict(state,strict=False)
    tok_path=out/'tokenizer'
    tokenizer=AutoTokenizer.from_pretrained(tok_path if tok_path.exists() else source,trust_remote_code=trust_remote_code)
    model.eval()
    return model,tokenizer,manifest,missing,unexpected
