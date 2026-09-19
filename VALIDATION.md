# Validation — v3.0.0

Packaging environment:

- Python: 3.13.5
- PyTorch: 2.10.0+cpu
- CUDA: unavailable in packaging environment
- GPU: unavailable in packaging environment

Validated here:

- `python -m compileall`: passed
- `pytest`: **22 passed, 1 skipped** (the skipped test is CUDA-only)
- Gradio app construction: passed (`Blocks`)
- Hardware detector / JSON report: passed
- `python -m ai_accel_lab selftest`: passed and correctly reports CUDA unavailable
- Offline editable install using `--no-build-isolation`: passed
- CPU/reference packing, groupwise INT2, planner/Accuracy Guard logic, dispatch fallback, transformer/memory/speculative tests: passed

The initial isolated editable-install attempt tried to download build dependencies from PyPI and failed because the packaging environment has no network access. Re-running with the already installed build backend using `--no-build-isolation` passed. This is an environment/network limitation, not a project import failure.

## GPU-specific validation rule

Hopper/Blackwell CUDA/Triton/Gluon kernels cannot be truthfully claimed as measured in this CPU-only build environment. The project therefore gates them with architecture checks and runtime correctness self-tests. On the target GPU run:

```bash
python -m ai_accel_lab selftest --microbench --json results/selftest.json
pytest -q
python -m ai_accel_lab kernel-suite --m 32 --n 4096 --k 4096 --group-size 128 --repeats 200 --power
```

Use `tools/verify_codegen.py` on emitted PTX/SASS/cubin to verify that expected WGMMA, MMA.SP, TMA or TCGEN05 instructions are actually present.

## Performance interpretation

No optimization is assumed to win on every shape or model. A packed format can save bandwidth while losing to a vendor Tensor Core kernel at large batch/prefill. Accuracy Guard can preserve a configured output-error bound by leaving sensitive layers dense, but it does not guarantee an improvement in model quality relative to the original checkpoint. The GUI records wall-clock speed, power, energy, VRAM and perplexity so these tradeoffs are visible.
