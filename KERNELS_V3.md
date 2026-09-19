# Kernel map

| Backend | Storage | Compute path | Best target | Accuracy |
|---|---|---|---|---|
| torch-fp16 | FP16 | cuBLAS/Torch | universal CUDA | baseline |
| groupwise-int2 | 2-bit + FP16 group scale | Triton decode/dequant+dot | bandwidth bound | quantized |
| persistent-int2 | 2-bit + scale | persistent Triton tile loop | low-M decode | quantized |
| native-2to4 | compressed semi-structured | vendor sparse Tensor Core | SM80+ | pruned |
| TMA matmul | FP16 | descriptor/TMA + Tensor Core | SM90+ | baseline-like |
| HB Gluon | FP16 | TMA + WGMMA/tcgen05 + multi-buffer | SM90/SM100 | baseline-like |
| Blackwell WS | FP16 | load/MMA/store warp partitions | SM100+ | baseline-like |
| giant fusion | INT2 QKV + FP16 cache | projection+RoPE+attention | single-token MHA | quantized |

## Why INT2 may lose to FP16

A packed format reduces memory traffic, but unpack/dequant and non-native instruction throughput can dominate for large-M GEMMs. Therefore the Studio separates decode-like low-M workloads from prefill-like large-M workloads and reports measured latency.

## Why native 2:4 is separate from INT2+2:4

NVIDIA's hardware sparse paths have specific data formats/instruction constraints. Combining arbitrary 2-bit codebooks and 2:4 metadata in one custom format is not automatically a native Sparse Tensor Core operation. The v2 custom `INT2+2:4` kernel remains available for experiments, while `native-2to4` is the path used when the goal is actual vendor sparse acceleration.
