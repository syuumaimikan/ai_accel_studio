# v5 Runtime Design

## Why v4 end-to-end was slower than its microkernel

A single packed INT2 GEMV can be reasonably close to FP16 while a full model is much slower when every decoder layer repeatedly performs Python module dispatch, backend selection, allocation and kernel launch setup.

v5 attacks this separately from quantization quality.

## Graph path

`baseline-static-compile` and `custom-mixedbit-v5-graph` configure a static KV cache and compile the model forward with `mode="reduce-overhead"`.

This is deliberately separate from eager measurements. The GUI reports which runtime path actually ran.

## Graph-safe quantized modules

The INT2/INT4 modules have a `graph_safe` mode. In that mode small-M CUDA dispatch no longer relies on exception-driven backend probing inside the compiled region.

## Fixed decode length

All Studio comparisons use fixed new-token counts by default. Static-cache generation also sets the minimum and maximum generated token counts to the same value.

## Prefill caveat

Packed INT2 is optimized primarily for memory-bound decode. A large-M prefill can still favor FP16/BF16 Tensor Core or established INT4 kernels. Studio therefore reports TTFT separately and retains optimized baseline paths instead of claiming one representation is universally optimal.
