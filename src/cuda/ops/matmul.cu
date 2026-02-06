#include "cuda/ops/matmul.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <stdexcept>

#include "cuda/precision.hpp"
#include "spdlog/spdlog.h"

#include "../device_buffer.hpp"
#include "../runtime_internal.hpp"

namespace easy_llm {
namespace ops {

namespace {

struct TensorUploadCache {
    ::easy_llm::cuda::DeviceBuffer device;
    const void* host_ptr{nullptr};
    size_t bytes{0};
};

struct MatmulCudaState {
    ::easy_llm::cuda::DeviceBuffer input;
    ::easy_llm::cuda::DeviceBuffer output;
    TensorUploadCache bias;
};

MatmulCudaState& matmul_cuda_state() {
    static MatmulCudaState state;
    return state;
}

size_t next_capacity_bytes(size_t current_capacity, size_t required_bytes) {
    if (required_bytes == 0) {
        return 0;
    }
    size_t capacity = current_capacity;
    if (capacity == 0) {
        capacity = std::max(required_bytes, static_cast<size_t>(4096));
    }
    while (capacity < required_bytes) {
        if (capacity > std::numeric_limits<size_t>::max() / 2) {
            capacity = required_bytes;
            break;
        }
        capacity *= 2;
    }
    return capacity;
}

void ensure_device_buffer(::easy_llm::cuda::DeviceBuffer& buffer, size_t required_bytes) {
    if (required_bytes == 0 || buffer.bytes() >= required_bytes) {
        return;
    }
    buffer.reset(next_capacity_bytes(buffer.bytes(), required_bytes));
}

template <typename T>
__device__ __forceinline__ float to_float(T value);

#if defined(USE_FP32)
template <>
__device__ __forceinline__ float to_float<float>(float value) {
    return value;
}
#endif

#if defined(USE_BF16)
template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 value) {
    return __bfloat162float(value);
}
#endif

template <typename T>
__device__ __forceinline__ T from_float(float value);

#if defined(USE_FP32)
template <>
__device__ __forceinline__ float from_float<float>(float value) {
    return value;
}
#endif

#if defined(USE_BF16)
template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float value) {
    return __float2bfloat16(value);
}
#endif

template <typename T>
__global__ void add_bias_kernel(T* data, const T* bias, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx >= total) {
        return;
    }
    int col = idx % cols;
    data[idx] = from_float<T>(to_float(data[idx]) + to_float(bias[col]));
}

template <typename Traits>
void maybe_add_bias(cudaStream_t stream,
                    void* output_ptr,
                    const Tensor* bias,
                    TensorUploadCache& cache,
                    int rows,
                    int cols) {
    if (bias == nullptr || cols <= 0 || bias->size() != cols) {
        return;
    }
    const size_t bias_bytes = static_cast<size_t>(cols) * Traits::kElementBytes;
    const void* host_ptr = bias->data().data();
    bool needs_upload = false;
    if (cache.device.bytes() < bias_bytes) {
        cache.device.reset(next_capacity_bytes(cache.device.bytes(), bias_bytes));
        needs_upload = true;
    }
    if (cache.host_ptr != host_ptr || cache.bytes != bias_bytes) {
        needs_upload = true;
    }
    if (needs_upload) {
        ::easy_llm::cuda::cuda_check(cudaMemcpyAsync(cache.device.data(), host_ptr, bias_bytes,
                                                     cudaMemcpyHostToDevice, stream),
                                     "cudaMemcpyAsync bias");
        cache.host_ptr = host_ptr;
        cache.bytes = bias_bytes;
    }

    constexpr int kThreads = 256;
    int total = rows * cols;
    int blocks = (total + kThreads - 1) / kThreads;
    add_bias_kernel<typename Traits::DeviceType><<<blocks, kThreads, 0, stream>>>(
        static_cast<typename Traits::DeviceType*>(output_ptr),
        static_cast<const typename Traits::DeviceType*>(cache.device.data()),
        rows, cols);
    ::easy_llm::cuda::cuda_check(cudaGetLastError(), "add_bias_kernel matmul_3d");
}

Tensor matmul_3d_cuda_impl(const Tensor& input, const Tensor& weights, const Tensor* bias) {
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

    MatmulCudaState& state = matmul_cuda_state();
    ensure_device_buffer(state.input, input_bytes);
    ensure_device_buffer(state.output, output_bytes);

    ::easy_llm::cuda::cuda_check(cudaMemcpyAsync(state.input.data(), input.data().data(), input_bytes,
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
                     state.input.data(), Traits::kCudaType, k,
                     &beta,
                     state.output.data(), Traits::kCudaType, n,
                     Traits::kComputeType,
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (gemm_status == CUBLAS_STATUS_NOT_SUPPORTED) {
        ctx.disable("cublasGemmEx not supported on this device");
        throw std::runtime_error("cublasGemmEx not supported");
    }
    ::easy_llm::cuda::cublas_check(gemm_status, "cublasGemmEx");

    maybe_add_bias<Traits>(ctx.stream(), state.output.data(), bias, state.bias, m, n);

    ::easy_llm::cuda::cuda_check(cudaMemcpyAsync(output.data().data(), state.output.data(), output_bytes,
                                                 cudaMemcpyDeviceToHost, ctx.stream()),
                                 "cudaMemcpyAsync output");
    ::easy_llm::cuda::cuda_check(cudaStreamSynchronize(ctx.stream()), "cudaStreamSynchronize");

    output.reshape({batch, seq_len, height});
    return output;
}

} // namespace

Tensor matmul_3d_cuda(const Tensor& input, const Tensor& weights) {
    return matmul_3d_cuda_impl(input, weights, nullptr);
}

Tensor matmul_3d_cuda(const Tensor& input, const Tensor& weights, const Tensor& bias) {
    return matmul_3d_cuda_impl(input, weights, &bias);
}

} // namespace ops
} // namespace easy_llm
