import torch
from ai_accel_lab.kernels.packing import (
    pack_int2, unpack_int2, pack_ternary, unpack_ternary,
    pack_int2_sparse24, unpack_int2_sparse24, prune_24,
)


def test_int2_pack_shape_and_error():
    torch.manual_seed(0)
    w=torch.randn(7,19)
    p=pack_int2(w)
    assert p.packed.shape==(7,5)
    d=unpack_int2(p)
    assert d.shape==w.shape
    assert torch.isfinite(d).all()
    assert (d-w).pow(2).mean() < w.pow(2).mean()


def test_ternary_pack_values():
    torch.manual_seed(1)
    w=torch.randn(5,17)
    p=pack_ternary(w)
    d=unpack_ternary(p)
    assert d.shape==w.shape
    assert torch.isfinite(d).all()


def test_sparse24_exact_pattern_after_unpack():
    torch.manual_seed(2)
    w=torch.randn(8,20)
    p=pack_int2_sparse24(w)
    d=unpack_int2_sparse24(p)
    g=d.view(8,5,4)
    # Codebook has no zero, therefore exactly 2 nonzeros/group.
    assert torch.equal((g!=0).sum(-1),torch.full((8,5),2))


def test_prune24_keeps_two():
    w=torch.randn(4,16)
    p=prune_24(w).view(4,4,4)
    assert torch.equal((p!=0).sum(-1),torch.full((4,4),2))
