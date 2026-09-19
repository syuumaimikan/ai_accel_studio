import copy,torch
from ai_accel_lab.transformer import TransformerBlock,ApproxTransformerBlock
from ai_accel_lab.memory import full_memory_attention,hierarchical_memory_attention

def test_transformer_shapes():
    b=TransformerBlock(32,4,2); x=torch.randn(2,8,32); y=b(x); assert y.shape==x.shape
    for m in ('ternary','int4','nm2_4','block50'): assert ApproxTransformerBlock(copy.deepcopy(b),m,2)(x).shape==x.shape

def test_memory_shapes():
    q=torch.randn(3,16); m=torch.randn(128,16); assert full_memory_attention(q,m).shape==(3,16); assert hierarchical_memory_attention(q,m,16,2).shape==(3,16)
