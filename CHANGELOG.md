# Changelog

## 3.0.0 — AI Acceleration Studio

- Added full Gradio GUI for hardware inspection, kernel microbenchmarks, multi-model comparison, giant-fusion experiments, Accuracy Guard and saved runs.
- Added hardware-aware dispatch for Ampere/Ada, Hopper and Blackwell.
- Added packed groupwise INT2 with per-group FP16 scales and activation-aware scale refinement.
- Added persistent INT2 Triton kernel and multi-stage pipeline.
- Added TMA tensor-descriptor path for SM90+.
- Added optional Hopper WGMMA and Blackwell tcgen05/Tensor Memory Gluon paths with correctness self-tests.
- Added optional Blackwell warp-specialized load/MMA/store partitioning.
- Added single-token QKV + RoPE + KV-cache append + causal attention giant-fusion Triton experiment.
- Added reusable Accuracy Guard for per-layer INT2/residual/FP16 decisions.
- Added optimized model bundle save/load.
- Added Hugging Face model matrix with TTFT, decode tok/s, total tok/s, NVML W, J/token, peak VRAM and perplexity.
- Added measured per-model auto-selection under a perplexity quality guard, including baseline-relative speed/energy/quality columns.
- Added existing-framework baselines: Transformers Hub kernels, SDPA, FlashAttention-2/3, paged FlashAttention-3, torch.compile and TorchAO.
- Added advanced GPU self-test command and PTX/SASS code-generation scanner for WGMMA, MMA.SP, TMA and TCGEN05 signatures.
- Retained v2 packed INT2, ternary, custom INT2+2:4, raw CUDA, native semi-structured sparse, Transformer, delta, memory and speculative benchmarks.

## 2.0.0

- Added real packed INT2/ternary Triton kernels and fused INT2 + 2:4 path.
- Added raw CUDA/NVCC kernels and native 2:4 Sparse Tensor Core comparison.

## 1.0.0

- Initial research benchmark suite.
