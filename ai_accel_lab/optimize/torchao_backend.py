from __future__ import annotations

"""TorchAO comparison backends with compatibility fallbacks.

TorchAO changes quickly. v4 prefers the current v2 weight-only configs, then
falls back to IntxWeightOnlyConfig rather than failing the whole benchmark when
an optional kernel dependency (for example MSLK on one packaging path) is absent.
"""

def apply_torchao_int4(model, group_size=128):
    import torch
    from torchao.quantization import quantize_
    errors=[]
    # Current stable API: version 2, groupwise, TinyGEMM/HQQ qparam algorithms.
    try:
        from torchao.quantization import Int4WeightOnlyConfig
        cfg=Int4WeightOnlyConfig(group_size=group_size,version=2,int4_choose_qparams_algorithm='tinygemm')
        quantize_(model,cfg); model._ai_accel_torchao_backend='Int4WeightOnlyConfig-v2-tinygemm'; return model
    except Exception as e: errors.append(f'int4-v2-tinygemm: {e!r}')
    try:
        from torchao.quantization import Int4WeightOnlyConfig
        cfg=Int4WeightOnlyConfig(group_size=group_size,version=2,int4_choose_qparams_algorithm='hqq')
        quantize_(model,cfg); model._ai_accel_torchao_backend='Int4WeightOnlyConfig-v2-hqq'; return model
    except Exception as e: errors.append(f'int4-v2-hqq: {e!r}')
    # Generic intx path is useful when a specialized optional runtime is missing.
    try:
        from torchao.quantization import IntxWeightOnlyConfig
        try:
            from torchao.quantization.granularity import PerGroup
        except Exception:
            from torchao.quantization import PerGroup
        cfg=IntxWeightOnlyConfig(weight_dtype=torch.int4,granularity=PerGroup(group_size))
        quantize_(model,cfg); model._ai_accel_torchao_backend='IntxWeightOnlyConfig-int4'; return model
    except Exception as e: errors.append(f'intx-int4: {e!r}')
    raise RuntimeError('TorchAO INT4 backends all failed: '+' | '.join(errors))


def apply_torchao_int8(model):
    from torchao.quantization import Int8WeightOnlyConfig, quantize_
    quantize_(model,Int8WeightOnlyConfig())
    model._ai_accel_torchao_backend='Int8WeightOnlyConfig'
    return model


def apply_torchao_int2(model, group_size=128):
    import torch
    from torchao.quantization import IntxWeightOnlyConfig, quantize_
    try:
        from torchao.quantization.granularity import PerGroup
    except Exception:
        try: from torchao.quantization import PerGroup
        except Exception: PerGroup=None
    errors=[]
    if PerGroup is not None:
        try:
            cfg=IntxWeightOnlyConfig(weight_dtype=torch.int2,granularity=PerGroup(group_size),version=2)
            quantize_(model,cfg); model._ai_accel_torchao_backend='IntxWeightOnlyConfig-int2-groupwise'; return model
        except Exception as e: errors.append(repr(e))
    try:
        cfg=IntxWeightOnlyConfig(weight_dtype=torch.int2)
        quantize_(model,cfg); model._ai_accel_torchao_backend='IntxWeightOnlyConfig-int2-default'; return model
    except Exception as e: errors.append(repr(e))
    raise RuntimeError('TorchAO INT2 failed: '+' | '.join(errors))
