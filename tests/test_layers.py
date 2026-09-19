import torch
from ai_accel_lab.layers import *

def data():
    torch.manual_seed(0); return torch.randn(16,32),torch.randn(16),torch.randn(4,32)

def test_shapes():
    w,b,x=data(); mods=[DenseLinear(w,b),PreQuantLinear(w,b,4),TernaryResidualLinear(w,b,4),DynamicTopKLinear(w,b,.25),BlockSparseLinear(w,b,8,.5),NMPrunedLinear(w,b,2,4),HybridRouterLinear(w,b,2)]
    for m in mods: assert m(x).shape==(4,16)

def test_delta_exact_zero_threshold():
    w,b,x=data(); d=DenseLinear(w,b); z=DeltaLinear(w,b,0); assert torch.allclose(z(x),d(x),atol=1e-5,rtol=1e-5)
    x2=x.clone(); x2[:,:3]+=.2; assert torch.allclose(z(x2),d(x2),atol=2e-5,rtol=2e-5)

def test_nm_nonzero_ratio():
    w,b,x=data(); m=NMPrunedLinear(w,b,2,4); ratio=m.weight.count_nonzero().item()/m.weight.numel(); assert abs(ratio-.5)<1e-6
