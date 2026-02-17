#include <torch/extension.h>
#include <ATen/ATen.h>

// CUDA implementations (we only support CUDA targets in this build)
// Forward declare the CUDA launch wrappers implemented in gemm.cu
void gemm_fp32_accum_cuda(const at::Tensor& A, const at::Tensor& B, at::Tensor& C);
void gemm_fp16_accum_cuda(const at::Tensor& A, const at::Tensor& B, at::Tensor& C);
void gemm_fp16_grouped_accum_cuda(const at::Tensor& A, const at::Tensor& B, at::Tensor& C, int group_k);
void gemm_blockquant_mxfp8_cuda(const at::Tensor& A, const at::Tensor& B, at::Tensor& C, int group_k, int group_n);
void gemm_blockquant_mxint8_cuda(const at::Tensor& A, const at::Tensor& B, at::Tensor& C, int group_k, int group_n);

static void check_inputs(const at::Tensor& A, const at::Tensor& B) {
    TORCH_CHECK(A.scalar_type() == at::kHalf, "A must be fp16");
    TORCH_CHECK(B.scalar_type() == at::kHalf, "B must be fp16");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A and B must be CUDA tensors");
    TORCH_CHECK(A.device() == B.device(), "A and B must be on the same CUDA device");
    TORCH_CHECK(A.size(1) == B.size(0), "A's K dimension must match B's K dimension");
}

at::Tensor gemm_fp32_accum(const at::Tensor& A, const at::Tensor& B) {
    check_inputs(A, B);
    TORCH_CHECK(A.is_cuda(), "gemm_fp32_accum: only CUDA tensors supported in this build");
    auto M = A.size(0);
    auto N = B.size(1);
    auto C = at::empty({M, N}, A.options());
    gemm_fp32_accum_cuda(A, B, C);
    return C;
}

at::Tensor gemm_fp16_accum(const at::Tensor& A, const at::Tensor& B) {
    check_inputs(A, B);
    TORCH_CHECK(A.is_cuda(), "gemm_fp16_accum: only CUDA tensors supported in this build");
    auto M = A.size(0);
    auto N = B.size(1);
    auto C = at::empty({M, N}, A.options());
    gemm_fp16_accum_cuda(A, B, C);
    return C;
}

at::Tensor gemm_fp16_grouped_accum(const at::Tensor& A, const at::Tensor& B, int group_k) {
    check_inputs(A, B);
    TORCH_CHECK(A.is_cuda(), "gemm_fp16_grouped_accum: only CUDA tensors supported in this build");
    auto M = A.size(0);
    auto N = B.size(1);
    auto C = at::empty({M, N}, A.options());
    gemm_fp16_grouped_accum_cuda(A, B, C, group_k);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_fp32_accum", &gemm_fp32_accum, "GEMM with fp32 accumulator (inputs fp16)");
    m.def("gemm_fp16_accum", &gemm_fp16_accum, "GEMM with fp16 accumulator (inputs fp16)");
    m.def("gemm_fp16_grouped_accum", &gemm_fp16_grouped_accum, "GEMM with grouped FP32-within, FP16-across accumulate (inputs fp16)");
    m.def("gemm_blockquant_mxfp8", [](const at::Tensor& A, const at::Tensor& B, int group_k, int group_n){
        check_inputs(A, B);
        auto M = A.size(0);
        auto N = B.size(1);
        auto C = at::empty({M, N}, A.options());
        gemm_blockquant_mxfp8_cuda(A, B, C, group_k, group_n);
        return C;
    }, "GEMM with block-quantized FP8 (E3M4) MX style within K-groups (inputs fp16, output fp16)");

    m.def("gemm_blockquant_mxint8", [](const at::Tensor& A, const at::Tensor& B, int group_k, int group_n){
        check_inputs(A, B);
        auto M = A.size(0);
        auto N = B.size(1);
        auto C = at::empty({M, N}, A.options());
        gemm_blockquant_mxint8_cuda(A, B, C, group_k, group_n);
        return C;
    }, "GEMM with block-quantized INT8 (power-of-two scaled) within K-groups (inputs fp16, output fp16)");
}
