from ai_accel_lab.runtime_policy import choose_runtime_policy

def test_cpu_policy_is_safe():
    p=choose_runtime_policy(m=1)
    assert p.decode_backend
    assert p.prefill_backend
