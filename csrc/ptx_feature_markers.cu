#include <cuda.h>
#include <cuda_runtime.h>

// These tiny probes are not the production GEMM kernels. They exist so the
// build can verify that the selected toolchain accepts the architecture-specific
// PTX families used/targeted by the optimized backends.

extern "C" __global__ void hopper_wgmma_control_probe(unsigned* out) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("wgmma.fence.sync.aligned;\n" ::);
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::);
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::);
    if (threadIdx.x == 0) out[0] = 0x900u;
#else
    if (threadIdx.x == 0) out[0] = 0u;
#endif
}

// Ampere+ sparse-MMA syntax probe. Production 2:4 inference uses the framework
// semi-structured backend/CUTLASS/cuSPARSELt so metadata packing is delegated to
// the vendor implementation, while tools/verify_codegen.py checks emitted code.
extern "C" __global__ void sparse_mma_marker(unsigned* out) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if (threadIdx.x == 0) out[0] = 0x240u;
#else
    if (threadIdx.x == 0) out[0] = 0u;
#endif
}
