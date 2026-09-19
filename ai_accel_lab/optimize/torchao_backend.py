from __future__ import annotations


def apply_torchao_int4(model, group_size=128):
    from torchao.quantization import Int4WeightOnlyConfig, quantize_
    quantize_(model, Int4WeightOnlyConfig(group_size=group_size))
    return model


def apply_torchao_int8(model):
    from torchao.quantization import Int8WeightOnlyConfig, quantize_
    quantize_(model, Int8WeightOnlyConfig())
    return model


def apply_torchao_int2(model, group_size=128):
    # torchao 0.17+ exposes IntxWeightOnlyConfig and torch.int2. Kernel support
    # depends on platform/packing format; this is included as a comparison path.
    import torch
    from torchao.quantization import IntxWeightOnlyConfig, quantize_
    try:
        from torchao.quantization.granularity import PerGroup
        cfg=IntxWeightOnlyConfig(weight_dtype=torch.int2,granularity=PerGroup(group_size))
    except Exception:
        cfg=IntxWeightOnlyConfig(weight_dtype=torch.int2)
    quantize_(model,cfg)
    return model
