import torch
from ai_accel_lab.quant import ternarize_weight
from ai_accel_lab.calibration import svd_residual_factors,activation_aware_factors

def test_factors():
    torch.manual_seed(1); w=torch.randn(16,12); _,_,a=ternarize_weight(w); r=w-a; x=torch.randn(32,12)
    for fn in (lambda:svd_residual_factors(r,4),lambda:activation_aware_factors(r,x,4)):
        u,v=fn(); assert u.shape==(16,4) and v.shape==(4,12); assert torch.isfinite(u).all() and torch.isfinite(v).all()
