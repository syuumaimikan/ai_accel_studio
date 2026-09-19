#include <torch/extension.h>

// CUDA implementations
torch::Tensor int2_linear_cuda(torch::Tensor x, torch::Tensor packed, torch::Tensor scale, torch::Tensor bias, int64_t activation);
torch::Tensor ternary_linear_cuda(torch::Tensor x, torch::Tensor packed, torch::Tensor scale, torch::Tensor bias, int64_t activation);
torch::Tensor int2_sparse24_linear_cuda(torch::Tensor x, torch::Tensor packed, torch::Tensor scale, torch::Tensor bias, int64_t activation);

static void check_common(const torch::Tensor& x, const torch::Tensor& packed, const torch::Tensor& scale) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(packed.is_cuda(), "packed must be CUDA");
    TORCH_CHECK(scale.is_cuda(), "scale must be CUDA");
    TORCH_CHECK(x.scalar_type() == at::kHalf, "x must be float16");
    TORCH_CHECK(packed.scalar_type() == at::kByte, "packed must be uint8");
    TORCH_CHECK(scale.scalar_type() == at::kHalf, "scale must be float16");
    TORCH_CHECK(x.dim() == 2 && packed.dim() == 2 && scale.dim() == 1, "invalid tensor rank");
    TORCH_CHECK(x.is_contiguous() && packed.is_contiguous() && scale.is_contiguous(), "inputs must be contiguous");
}

torch::Tensor int2_linear(torch::Tensor x, torch::Tensor packed, torch::Tensor scale, torch::Tensor bias, int64_t activation) {
    check_common(x, packed, scale);
    TORCH_CHECK(x.size(1) == packed.size(1) * 4, "K must equal packed_bytes*4");
    return int2_linear_cuda(x, packed, scale, bias, activation);
}

torch::Tensor ternary_linear(torch::Tensor x, torch::Tensor packed, torch::Tensor scale, torch::Tensor bias, int64_t activation) {
    check_common(x, packed, scale);
    TORCH_CHECK(x.size(1) == packed.size(1) * 4, "K must equal packed_bytes*4");
    return ternary_linear_cuda(x, packed, scale, bias, activation);
}

torch::Tensor int2_sparse24_linear(torch::Tensor x, torch::Tensor packed, torch::Tensor scale, torch::Tensor bias, int64_t activation) {
    check_common(x, packed, scale);
    TORCH_CHECK(x.size(1) == packed.size(1) * 4, "K must equal sparse groups*4");
    return int2_sparse24_linear_cuda(x, packed, scale, bias, activation);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("int2_linear", &int2_linear, "Fused packed INT2 linear (CUDA)");
    m.def("ternary_linear", &ternary_linear, "Fused packed ternary linear (CUDA)");
    m.def("int2_sparse24_linear", &int2_sparse24_linear, "Fused packed INT2 + 2:4 sparse linear (CUDA)");
}
