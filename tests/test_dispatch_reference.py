import torch
from ai_accel_lab.kernels.dispatch import FusedPackedLinear


def test_reference_dispatch_all_modes():
    torch.manual_seed(3)
    w=torch.randn(12,18)
    b=torch.randn(12)
    x=torch.randn(4,18)
    for mode in ('int2','ternary','int2_2to4'):
        m=FusedPackedLinear(w,b,mode=mode,backend='reference',activation='silu')
        y=m(x)
        assert y.shape==(4,12)
        assert torch.isfinite(y).all()
        assert m.last_dispatch.backend=='reference'
