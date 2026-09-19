import pytest
import torch

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")

def test_triton_packed_kernels_match_dequant_reference():
    pytest.importorskip("triton")
    import torch.nn.functional as F
    from ai_accel_lab.kernels.packing import pack_int2,unpack_int2,pack_ternary,unpack_ternary,pack_int2_sparse24,unpack_int2_sparse24
    from ai_accel_lab.kernels import triton_fused as tf
    torch.manual_seed(4)
    m,n,k=8,64,64
    x=torch.randn(m,k,device="cuda",dtype=torch.float16)
    w=torch.randn(n,k,device="cuda",dtype=torch.float16)
    b=torch.randn(n,device="cuda",dtype=torch.float16)
    for pack,unpack,fn in [(pack_int2,unpack_int2,tf.int2_linear),(pack_ternary,unpack_ternary,tf.ternary_linear),(pack_int2_sparse24,unpack_int2_sparse24,tf.int2_sparse24_linear)]:
        pw=pack(w); y=fn(x,pw,b,"none")
        ref=F.linear(x,unpack(pw,dtype=torch.float16),b)
        assert torch.allclose(y,ref,atol=5e-2,rtol=5e-2)
