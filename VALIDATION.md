# Validation

The distribution is validated in two tiers.

## CPU / packaging validation

Run:

```bash
pytest
python -m ai_accel_lab --help
python -m ai_accel_lab doctor
python -m compileall -q ai_accel_lab
```

CPU tests cover INT2/INT4 packing, hybrid outlier correction, mixed-bit planner decisions, robust guard statistics, legacy benchmark code, transformer/memory helpers and Studio helpers.

## NVIDIA validation

Run on the target GPU:

```bash
python -m ai_accel_lab selftest --microbench --json results/selftest.json
```

The self-test includes:

- CUDA matmul
- CUDA Graph replay
- groupwise INT2 Triton
- M<=4 INT2 decode
- M<=4 INT4 decode
- TMA where supported
- Hopper/Blackwell experimental backends where supported
- giant-fusion correctness probe

A backend failing self-test must not be treated as accelerated merely because the GPU architecture nominally supports the instruction family.

## Performance validation

Use at least:

```text
M=1,2,4,8,16,32,128,512
```

for representative weight shapes. End-to-end model measurements should use fixed generated-token counts and multiple held-out quality texts.

The CPU environment used to package this project cannot validate Hopper/Blackwell throughput; GPU performance numbers must be produced on the target machine.

## Packaging environment result

At package-build time in the CPU-only validation environment:

```text
30 passed, 1 skipped
CUDA skip reason: CUDA required
editable install: ai-accel-studio 5.0.0
GUI construction: Gradio Blocks
```

This does **not** substitute for target-GPU self-test/benchmarking.
