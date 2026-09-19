#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace {
constexpr int BM = 16;
constexpr int BN = 16;
constexpr int BK = 128; // must be multiple of 4

__device__ __forceinline__ float decode_int2(unsigned c) {
    switch (c & 3u) {
        case 0: return -1.0f;
        case 1: return -0.3333333333333333f;
        case 2: return  0.3333333333333333f;
        default:return  1.0f;
    }
}

__device__ __forceinline__ float apply_act(float x, int act) {
    if (act == 1) return x > 0.f ? x : 0.f;
    if (act == 2) return x / (1.f + __expf(-x));
    return x;
}

template<int MODE>
__global__ void packed_dense_kernel(
    const half* __restrict__ x,
    const uint8_t* __restrict__ packed,
    const half* __restrict__ scale,
    const half* __restrict__ bias,
    half* __restrict__ y,
    int M, int N, int K, bool has_bias, int activation)
{
    // MODE 0 = INT2, MODE 1 = ternary
    __shared__ half sx[BM][BK];
    const int tx = threadIdx.x; // output-column lane
    const int ty = threadIdx.y; // row lane
    const int m = blockIdx.y * BM + ty;
    const int n = blockIdx.x * BN + tx;
    const int tid = ty * BN + tx;
    float acc = 0.f;

    for (int kb = 0; kb < K; kb += BK) {
        // Cooperative X tile load: each element loaded once per CTA.
        for (int linear = tid; linear < BM * BK; linear += BM * BN) {
            int lm = linear / BK;
            int lk = linear - lm * BK;
            int gm = blockIdx.y * BM + lm;
            int gk = kb + lk;
            sx[lm][lk] = (gm < M && gk < K) ? x[gm*K + gk] : __float2half(0.f);
        }
        __syncthreads();

        if (m < M && n < N) {
            const float sc = __half2float(scale[n]);
            const int kend = min(BK, K - kb);
            for (int lk = 0; lk < kend; lk += 4) {
                const uint8_t p = packed[n*(K/4) + (kb + lk)/4];
                #pragma unroll
                for (int j=0; j<4; ++j) {
                    unsigned c = (p >> (2*j)) & 3u;
                    float q;
                    if constexpr (MODE == 0) {
                        q = decode_int2(c);
                        acc = fmaf(__half2float(sx[ty][lk+j]), q*sc, acc);
                    } else {
                        q = (c == 1u) ? 1.f : ((c == 2u) ? -1.f : 0.f);
                        if (q != 0.f) acc = fmaf(__half2float(sx[ty][lk+j]), q*sc, acc);
                    }
                }
            }
        }
        __syncthreads();
    }

    if (m < M && n < N) {
        if (has_bias) acc += __half2float(bias[n]);
        acc = apply_act(acc, activation);
        y[m*N+n] = __float2half_rn(acc);
    }
}

__global__ void sparse24_int2_kernel(
    const half* __restrict__ x,
    const uint8_t* __restrict__ packed,
    const half* __restrict__ scale,
    const half* __restrict__ bias,
    half* __restrict__ y,
    int M, int N, int K, bool has_bias, int activation)
{
    __shared__ half sx[BM][BK];
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int m = blockIdx.y * BM + ty;
    const int n = blockIdx.x * BN + tx;
    const int tid = ty * BN + tx;
    const int G = K / 4;
    float acc = 0.f;

    for (int kb = 0; kb < K; kb += BK) {
        for (int linear = tid; linear < BM * BK; linear += BM * BN) {
            int lm = linear / BK;
            int lk = linear - lm * BK;
            int gm = blockIdx.y * BM + lm;
            int gk = kb + lk;
            sx[lm][lk] = (gm < M && gk < K) ? x[gm*K + gk] : __float2half(0.f);
        }
        __syncthreads();

        if (m < M && n < N) {
            const float sc = __half2float(scale[n]);
            const int group_begin = kb / 4;
            const int groups_here = min(BK, K-kb) / 4;
            #pragma unroll 4
            for (int lg=0; lg<groups_here; ++lg) {
                uint8_t p = packed[n*G + group_begin + lg];
                unsigned c0 = p & 3u;
                unsigned c1 = (p >> 2) & 3u;
                unsigned i0 = (p >> 4) & 3u;
                unsigned i1 = (p >> 6) & 3u;
                float w0 = decode_int2(c0) * sc;
                float w1 = decode_int2(c1) * sc;
                int base = lg * 4;
                // Exactly two multiplications per original 4-weight group.
                acc = fmaf(__half2float(sx[ty][base+i0]), w0, acc);
                acc = fmaf(__half2float(sx[ty][base+i1]), w1, acc);
            }
        }
        __syncthreads();
    }

    if (m < M && n < N) {
        if (has_bias) acc += __half2float(bias[n]);
        acc = apply_act(acc, activation);
        y[m*N+n] = __float2half_rn(acc);
    }
}

void check_launch() {
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));
}
}

static torch::Tensor alloc_y(torch::Tensor x, torch::Tensor scale) {
    return torch::empty({x.size(0), scale.size(0)}, x.options());
}

static const half* bias_ptr(torch::Tensor bias) {
    return bias.numel() ? reinterpret_cast<const half*>(bias.data_ptr<at::Half>()) : nullptr;
}

torch::Tensor int2_linear_cuda(torch::Tensor x, torch::Tensor packed, torch::Tensor scale, torch::Tensor bias, int64_t activation) {
    auto y = alloc_y(x, scale);
    int M=x.size(0), K=x.size(1), N=scale.size(0);
    dim3 block(BN,BM); dim3 grid((N+BN-1)/BN,(M+BM-1)/BM);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    packed_dense_kernel<0><<<grid,block,0,stream>>>(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()), packed.data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(scale.data_ptr<at::Half>()), bias_ptr(bias),
        reinterpret_cast<half*>(y.data_ptr<at::Half>()), M,N,K,bias.numel()!=0,(int)activation);
    check_launch(); return y;
}

torch::Tensor ternary_linear_cuda(torch::Tensor x, torch::Tensor packed, torch::Tensor scale, torch::Tensor bias, int64_t activation) {
    auto y = alloc_y(x, scale);
    int M=x.size(0), K=x.size(1), N=scale.size(0);
    dim3 block(BN,BM); dim3 grid((N+BN-1)/BN,(M+BM-1)/BM);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    packed_dense_kernel<1><<<grid,block,0,stream>>>(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()), packed.data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(scale.data_ptr<at::Half>()), bias_ptr(bias),
        reinterpret_cast<half*>(y.data_ptr<at::Half>()), M,N,K,bias.numel()!=0,(int)activation);
    check_launch(); return y;
}

torch::Tensor int2_sparse24_linear_cuda(torch::Tensor x, torch::Tensor packed, torch::Tensor scale, torch::Tensor bias, int64_t activation) {
    auto y = alloc_y(x, scale);
    int M=x.size(0), K=x.size(1), N=scale.size(0);
    dim3 block(BN,BM); dim3 grid((N+BN-1)/BN,(M+BM-1)/BM);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    sparse24_int2_kernel<<<grid,block,0,stream>>>(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()), packed.data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(scale.data_ptr<at::Half>()), bias_ptr(bias),
        reinterpret_cast<half*>(y.data_ptr<at::Half>()), M,N,K,bias.numel()!=0,(int)activation);
    check_launch(); return y;
}
