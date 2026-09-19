# GPU kernel implementation

This version adds three real packed GPU execution paths.

## 1. Packed INT2 Triton/CUDA

Weights are stored at **2 bits/weight** (four weights per byte). The kernel performs:

`packed load -> bit unpack -> INT2 codebook dequant -> matrix accumulation -> bias -> activation`

inside the same GPU kernel. There is no materialized FP16 weight matrix.

The current NVIDIA Tensor Core ISA does not expose native INT2 MMA in the same way it exposes FP16/BF16/INT8 and some INT4 paths, so this project treats INT2 as a storage/bandwidth format and decodes inside the fused kernel. This distinction is important when interpreting throughput.

## 2. Packed ternary Triton/CUDA

Ternary weights use two-bit codes:

- `00 = 0`
- `01 = +1`
- `10 = -1`

Four ternary weights are stored per byte. The raw CUDA kernel skips the FMA when the code is zero. The Triton kernel decodes a tile and uses Tensor-Core-friendly FP16 `tl.dot` for throughput.

## 3. Fused INT2 + 2:4 sparse

Each original four-weight group keeps exactly two weights. One byte stores the complete group:

```text
bits 0..1  INT2 value 0
bits 2..3  INT2 value 1
bits 4..5  position 0 in [0,3]
bits 6..7  position 1 in [0,3]
```

The custom sparse kernel executes **two multiplications per original four-weight group**. It never reconstructs the pruned values.

This format is 2 bits/original-weight including sparse metadata, before per-channel scales.

## 4. Native NVIDIA 2:4 Sparse Tensor Core path

For FP16/BF16 2:4, the benchmark also uses `torch.sparse.to_sparse_semi_structured`. PyTorch dispatches this compressed representation to CUTLASS/cuSPARSELt-backed sparse kernels where supported. This is the production path to compare against for actual hardware Sparse Tensor Core acceleration.

The custom INT2+2:4 format is separate because NVIDIA's standard semi-structured sparse PyTorch path does not accept packed INT2 weights directly.

## Commands

```bash
pip install -r requirements.txt
pip install -r requirements-gpu.txt

python -m ai_accel_lab gpu-kernels \
  --m 128 --n 4096 --k 4096 \
  --repeats 100 --activation silu
```

Build and include the raw CUDA extension:

```bash
python -m ai_accel_lab gpu-kernels \
  --m 128 --n 4096 --k 4096 \
  --cuda-ext
```

The first CUDA-extension run compiles `csrc/packed_kernels.cpp` and `csrc/packed_kernels_cuda.cu` with NVCC and caches the extension.

## Recommended benchmark shapes

LLM token decode (memory dominated):

```bash
python -m ai_accel_lab gpu-kernels --m 1 --n 4096 --k 4096 --repeats 500
python -m ai_accel_lab gpu-kernels --m 8 --n 4096 --k 4096 --repeats 300
```

Prefill / larger batch:

```bash
python -m ai_accel_lab gpu-kernels --m 128 --n 4096 --k 4096 --repeats 100
python -m ai_accel_lab gpu-kernels --m 512 --n 4096 --k 11008 --repeats 50
```

For native FP16/BF16 2:4, dimensions must satisfy the backend's shape restrictions; multiples of 64 are the safest starting point.

## What is actually fused?

For the Triton and raw CUDA packed kernels, the following are one launch:

1. weight bit unpack
2. low-bit decoding / dequantization
3. matrix multiplication / sparse accumulation
4. bias
5. ReLU or SiLU epilogue

This removes both an intermediate dequantized weight tensor and extra epilogue kernel launches.
