import torch
from ai_accel_lab.optimize.planner import decide_linear


def test_planner_returns_valid_decision():
    torch.manual_seed(0)
    layer=torch.nn.Linear(64,32,bias=True)
    x=torch.randn(24,64)
    d=decide_linear('x',layer,x,group_size=32,nmse_limit=1.0,cosine_limit=-1.0,residual_ranks=(0,))
    assert d.method in ('int2','fp16')
    assert d.compression>0
