import torch
from ai_accel_lab.quant import quantize_per_row,dequantize_per_row,ternarize_weight,fake_quant_activation

def test_quant_shapes_and_finite():
    w=torch.randn(8,16)
    for bits in (2,4,8):
        q,s=quantize_per_row(w,bits); d=dequantize_per_row(q,s,torch.float32)
        assert q.shape==w.shape and s.shape==(8,) and torch.isfinite(d).all()
        assert torch.isfinite(fake_quant_activation(w,bits)).all()

def test_ternary_values():
    t,s,a=ternarize_weight(torch.randn(8,16)); assert set(torch.unique(t).tolist()).issubset({-1,0,1}); assert torch.isfinite(a).all()
