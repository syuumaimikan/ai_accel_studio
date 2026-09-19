import torch
from ai_accel_lab.kernels.groupwise_int4 import pack_groupwise_int4,unpack_groupwise_int4
from ai_accel_lab.optimize.hybrid_int4 import HybridInt4Linear
from ai_accel_lab.optimize.planner_v5 import decide_linear_v5
from ai_accel_lab.optimize.robust_guard import split_calibration_validation,robust_summary


def test_groupwise_int4_roundtrip_shape_and_quality():
    torch.manual_seed(10);w=torch.randn(24,64)
    pw=pack_groupwise_int4(w,32);q=unpack_groupwise_int4(pw)
    assert q.shape==w.shape
    assert torch.isfinite(q).all()
    assert (q-w).square().mean() < (w.square().mean()*0.1)


def test_hybrid_int4_outliers_do_not_hurt_dominant_columns():
    torch.manual_seed(11);src=torch.nn.Linear(64,32,bias=False);src.weight.data[:,:8]*=8
    rms=torch.ones(64);rms[:8]=4;x=torch.randn(32,64)*rms
    plain=HybridInt4Linear(src,32,rms,0);out=HybridInt4Linear(src,32,rms,8);ref=src(x)
    assert (out(x)-ref).square().mean() <= (plain(x)-ref).square().mean()+1e-8


def test_v5_planner_returns_2_4_or_16_bits():
    torch.manual_seed(12);src=torch.nn.Linear(64,32,bias=False);x=torch.randn(20,64)
    d=decide_linear_v5('q_proj',src,x,group_sizes=(32,),outliers=(0,8),residual_ranks=(0,),nmse_limit=.2,cosine_limit=.8)
    assert d.bits in (2,4,16)
    assert d.method


def test_robust_split_and_summary():
    plan,val=split_calibration_validation(['a','b','c','d'])
    assert plan==['a','c'] and val==['b','d']
    s=robust_summary([2,4],[2.02,4.08])
    assert s['samples']==2 and 0 <= s['mean_rel'] < .03
