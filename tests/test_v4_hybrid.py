import torch
from ai_accel_lab.optimize.hybrid_int2 import HybridInt2Linear
from ai_accel_lab.optimize.planner_v4 import decide_linear_v4


def test_hybrid_shape_storage_and_no_dense_parameter():
    torch.manual_seed(2)
    src=torch.nn.Linear(64,32,bias=True)
    rms=torch.rand(64)+.1
    q=HybridInt2Linear(src,group_size=32,activation_rms=rms,outlier_columns=8,residual_rank=0)
    x=torch.randn(5,64)
    y=q(x)
    assert y.shape==(5,32)
    assert q.outlier_idx.numel()==8
    # Replaced module must not retain an [N,K] FP16/FP32 dense base weight.
    for _,p in q.named_parameters():
        assert p.numel()!=32*64
    info=q.optimization_info()
    assert info.optimized_bytes < info.original_bytes


def test_outliers_improve_or_match_quantization_error():
    torch.manual_seed(3)
    src=torch.nn.Linear(64,48,bias=False)
    # make a few columns dominant and activation-sensitive
    src.weight.data[:, :8] *= 8
    rms=torch.ones(64); rms[:8]=5
    x=torch.randn(32,64)*rms
    plain=HybridInt2Linear(src,32,rms,0,0)
    out=HybridInt2Linear(src,32,rms,8,0)
    ref=src(x)
    e0=(plain(x)-ref).float().square().mean()
    e1=(out(x)-ref).float().square().mean()
    assert e1 <= e0


def test_planner_v4_returns_valid_decision():
    torch.manual_seed(4)
    src=torch.nn.Linear(64,32,bias=False)
    x=torch.randn(24,64)
    d=decide_linear_v4('q_proj',src,x,group_sizes=(32,),outliers=(0,8),residual_ranks=(0,),nmse_limit=1.0,cosine_limit=0.0)
    assert d.group_size==32
    assert d.method in ('int2','int2-outlier','fp16')
