# Changelog

## 5.0.0

- Added packed groupwise INT4 format and M<=4 decode-specialized Triton kernel.
- Added mixed-bit planner that searches INT2 first, promotes sensitive layers to INT4, and keeps FP16 only when required.
- Added held-out robust model-level guard using mean, P95, and worst relative NLL degradation.
- Split calibration text into planning and validation subsets when possible to reduce calibration overfit.
- Added diverse built-in calibration/evaluation suite for the GUI.
- Added `baseline-static-compile` and `custom-mixedbit-v5-graph`.
- Added Static KV Cache + `torch.compile(mode="reduce-overhead")` generation benchmark path.
- Added graph-safe custom INT2/INT4 module dispatch mode to reduce graph breaks/exception-driven runtime dispatch.
- Added exact baseline reload when the robust guard leaves zero quantized layers.
- Added multi-text token-weighted perplexity evaluation.
- Added Dependency Doctor that reads installed package metadata for Transformers/Kernels and TorchAO/MSLK requirements.
- Added INT4 and CUDA Graph probes to GPU self-test.
- Added v5 mixed-bit bundle format/load support.
- Kept persistent INT2 out of automatic dispatch; it remains an explicit benchmark only.

## 4.0.0

- Added M<=4 decode-specialized packed groupwise INT2 kernel.
- Added activation-aware exact FP16 outlier correction and model-level calibration guard.
- Switched to fixed-token CUDA-event benchmarking and removed persistent INT2 from auto dispatch.
