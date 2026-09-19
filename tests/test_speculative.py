import torch
from ai_accel_lab.speculative import TinyCausalLM,greedy,speculative

def test_speculative_preserves_target_greedy():
    torch.manual_seed(0); target=TinyCausalLM(vocab=32,d_model=32,nhead=4,layers=1,max_len=64,seed=2).eval(); draft=TinyCausalLM(vocab=32,d_model=16,nhead=4,layers=1,max_len=64,seed=3).eval(); p=torch.tensor([[1,2]])
    g,_=greedy(target,p,6); s,stats=speculative(target,draft,p,6,3); assert torch.equal(g,s); assert stats['target_calls']<=6
