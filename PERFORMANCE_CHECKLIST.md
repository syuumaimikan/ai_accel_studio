# Performance checklist v5

Before claiming a speedup:

- compare against FP16/BF16 cuBLAS/Tensor Core baseline;
- compare against StaticCache + torch.compile baseline;
- keep generated token count identical;
- warm compiled/static-cache paths before timing;
- do not include quantization/calibration in inference latency;
- measure TTFT and decode separately;
- measure J/generated-token, not only instantaneous watts;
- record live model storage separately from allocator-reserved VRAM;
- use held-out perplexity/NLL, not only layer cosine;
- test M=1/2/4 decode shapes separately from prefill;
- treat persistent/WGMMA/tcgen05 paths as measured candidates, not guaranteed winners;
- if `optimized_layers == 0`, compare using the exact reloaded baseline path;
- run `python -m ai_accel_lab doctor` when HF Kernels or TorchAO optional backends fail.
