#include "cuda/ops/matmul.hpp"

#include <cstddef>
#include <mutex>
#include <stdexcept>

#include "cuda/precision.hpp"
#include "spdlog/spdlog.h"

#include "../device_buffer.hpp"
#include "../runtime_internal.hpp"

namespace easy_llm {
namespace ops {

Tensor matmul_3d_cuda(const Tensor& input, const Tensor& weights) {
    using Traits = ::easy_llm::cuda::CudaPrecisionTraits<data_type>;

    auto& ctx = ::easy_llm::cuda::get_context();
    if (!ctx.available()) {
        throw std::runtime_error("CUDA is not available");
    }

    const int height = weights.shape()[0];
    const int width = weights.shape()[1];
    if (input.shape().back() != width) {
        spdlog::error("matmul_3d CUDA input.shape().back() == {}, width == {}",
                      input.shape().back(), width);
        throw std::invalid_argument("The last dimension of input must be equal to width.");
    }
    const int batch = input.shape()[0];
    const int seq_len = input.shape()[1];
    const int m = batch * seq_len;
    const int k = width;
    const int n = height;

    if (input.size() != m * k) {
        throw std::invalid_argument("matmul_3d CUDA input size does not match shape.");
    }
    if (weights.size() != n * k) {
        throw std::invalid_argument("matmul_3d CUDA weights size does not match shape.");
    }

    const size_t input_bytes = static_cast<size_t>(m) * k * Traits::kElementBytes;
    const size_t output_bytes = static_cast<size_t>(m) * n * Traits::kElementBytes;

    Tensor output(m * n);

    std::lock_guard<std::mutex> lock(ctx.mutex());

    ::easy_llm::cuda::DeviceBuffer d_input(input_bytes);
    ::easy_llm::cuda::DeviceBuffer d_output(output_bytes);

    ::easy_llm::cuda::cuda_check(cudaMemcpyAsync(d_input.data(), input.data().data(), input_bytes,
                                                 cudaMemcpyHostToDevice, ctx.stream()),
                                 "cudaMemcpyAsync input");

    ::easy_llm::cuda::WeightEntry& weights_entry = ctx.cache().get_or_upload(weights, ctx.stream());

    const float alpha = 1.0f;
    const float beta = 0.0f;

    auto gemm_status =
        cublasGemmEx(ctx.handle(),
                     CUBLAS_OP_T, CUBLAS_OP_N,
                     n, m, k,
                     &alpha,
                     weights_entry.device_ptr, Traits::kCudaType, k,
                     d_input.data(), Traits::kCudaType, k,
                     &beta,
                     d_output.data(), Traits::kCudaType, n,
                     Traits::kComputeType,
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (gemm_status == CUBLAS_STATUS_NOT_SUPPORTED) {
        ctx.disable("cublasGemmEx not supported on this device");
        throw std::runtime_error("cublasGemmEx not supported");
    }
    ::easy_llm::cuda::cublas_check(gemm_status, "cublasGemmEx");

    ::easy_llm::cuda::cuda_check(cudaMemcpyAsync(output.data().data(), d_output.data(), output_bytes,
                                                 cudaMemcpyDeviceToHost, ctx.stream()),
                                 "cudaMemcpyAsync output");
    ::easy_llm::cuda::cuda_check(cudaStreamSynchronize(ctx.stream()), "cudaStreamSynchronize");

    output.reshape({batch, seq_len, height});
    return output;
}

} // namespace ops
} // namespace easy_llm
