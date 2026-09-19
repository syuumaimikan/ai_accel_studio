import torch
from ai_accel_lab.kernels.groupwise_int2 import pack_groupwise_int2,unpack_groupwise_int2
from ai_accel_lab.optimize.accelerated_linear import GroupwiseInt2Linear


def test_groupwise_pack_shape_and_storage():
    torch.manual_seed(0)
    w=torch.randn(17,130)
    pw=pack_groupwise_int2(w,group_size=64)
    assert pw.padded_k==192
    assert pw.packed.shape==(17,48)
    assert pw.scales.shape==(17,3)
    u=unpack_groupwise_int2(pw)
    assert u.shape==w.shape
    assert torch.isfinite(u).all()


def test_activation_aware_pack():
    torch.manual_seed(1)
    w=torch.randn(8,128)
    rms=torch.linspace(.1,2.0,128)
    pw=pack_groupwise_int2(w,64,activation_rms=rms)
    assert pw.activation_aware
    assert torch.isfinite(pw.scales).all()


def test_accelerated_linear_cpu_fallback_shape():
    torch.manual_seed(2)
    src=torch.nn.Linear(128,32)
    mod=GroupwiseInt2Linear(src,64,residual_rank=0)
    x=torch.randn(4,7,128)
    y=mod(x)
    assert y.shape==(4,7,32)
    assert torch.isfinite(y).all()
