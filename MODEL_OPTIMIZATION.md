# Existing-model optimization flow

1. Load original HF model in FP16/BF16.
2. Run several representative calibration prompts.
3. Record a bounded number of Linear inputs.
4. Compute activation RMS per input channel.
5. Optimize groupwise INT2 scale using activation-weighted error.
6. Evaluate output NMSE/cosine against original Linear.
7. If needed, add low-rank residual correction.
8. If the quality guard still fails, keep the layer dense.
9. Benchmark end-to-end generation, perplexity, power and VRAM.
10. Save the accepted mixed plan as an acceleration bundle.

This is intentionally conservative. A smaller quantization error is not the same thing as better downstream task accuracy, so end-to-end evaluation remains necessary.
