from __future__ import annotations
import importlib.metadata as im
import re
import pandas as pd

PACKAGES=('torch','triton','transformers','kernels','torchao','mslk','accelerate','gradio','flash-attn')

def _version(name):
    try:return im.version(name)
    except Exception:return None

def _requires(name):
    try:return im.requires(name) or []
    except Exception:return []

def _find_req(parent,needle):
    for r in _requires(parent):
        # Strip environment marker/extra and return the package specifier part.
        head=r.split(';',1)[0].strip()
        if re.match(rf'^{re.escape(needle)}(?:\b|\[|[<>=!~])',head,re.I):return head
    return None

def dependency_report():
    rows=[]
    for p in PACKAGES:
        v=_version(p);rows.append({'package':p,'version':v or 'not installed','status':'ok' if v else 'missing','required_by':''})
    kreq=_find_req('transformers','kernels');mreq=_find_req('torchao','mslk')
    for row in rows:
        if row['package']=='kernels' and kreq:row['required_by']=f'transformers -> {kreq}'
        if row['package']=='mslk' and mreq:row['required_by']=f'torchao -> {mreq}'
    return pd.DataFrame(rows)

def recommendations():
    cmds=[];notes=[]
    kreq=_find_req('transformers','kernels')
    if _version('transformers') and not _version('kernels'):
        cmds.append(f'pip install "{kreq}"' if kreq else 'pip install kernels')
    elif kreq and _version('kernels'):
        try:
            from packaging.requirements import Requirement
            req=Requirement(kreq)
            if _version('kernels') not in req.specifier:cmds.append(f'pip install -U "{kreq}"')
        except Exception:pass
    mreq=_find_req('torchao','mslk')
    if _version('torchao') and not _version('mslk') and mreq:cmds.append(f'pip install "{mreq}"')
    if not _version('triton'):notes.append('Triton missing: custom packed INT2/INT4 CUDA kernels will use fallback paths.')
    if not _version('torch'):notes.append('Install the CUDA-enabled PyTorch build matching your NVIDIA driver before full setup.')
    return {'commands':cmds,'notes':notes,'transformers_kernels_requirement':kreq,'torchao_mslk_requirement':mreq}
