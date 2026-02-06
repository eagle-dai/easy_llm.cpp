#include "cuda/ops/mlp.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>

#include <cuda_bf16.h>

#include "cuda/precision.hpp"

#include "../device_buffer.hpp"
#include "../runtime_internal.hpp"

namespace easy_llm {
namespace cuda {
namespace ops {

namespace {

struct TensorUploadCache {
    DeviceBuffer device;
    const void* host_ptr{nullptr};
    size_t bytes{0};
};

struct MlpCudaState {
    DeviceBuffer input;
    DeviceBuffer norm_out;
    DeviceBuffer up;
    DeviceBuffer gate;
    DeviceBuffer output;
    TensorUploadCache norm_weight_cache;
    TensorUploadCache up_bias_cache;
    TensorUploadCache gate_bias_cache;
    TensorUploadCache down_bias_cache;
};

MlpCudaState& mlp_cuda_state() {
    static MlpCudaState state;
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

void ensure_device_buffer(DeviceBuffer& buffer, size_t required_bytes) {
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
__global__ void rms_norm_kernel(const T* input,
                                const T* weight,
                                T* output,
                                int rows,
                                int hidden_dim,
                                float epsilon) {
    extern __shared__ float shared[];
    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    int tid = threadIdx.x;
    float local_sum = 0.0f;
    int base = row * hidden_dim;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float v = to_float(input[base + i]);
        local_sum += v * v;
    }
    shared[tid] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    float inv_rms = rsqrtf(shared[0] / static_cast<float>(hidden_dim) + epsilon);
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float x = to_float(input[base + i]);
        float w = to_float(weight[i]);
        output[base + i] = from_float<T>(x * w * inv_rms);
    }
}

template <typename T>
__global__ void add_bias_kernel(T* data, const T* bias, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx >= total) {
        return;
    }
    int col = idx % cols;
    float v = to_float(data[idx]) + to_float(bias[col]);
    data[idx] = from_float<T>(v);
}

template <typename T>
__global__ void silu_mul_inplace_kernel(T* up, const T* gate, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    float u = to_float(up[idx]);
    float g = to_float(gate[idx]);
    float silu = g / (1.0f + expf(-g));
    up[idx] = from_float<T>(u * silu);
}

void validate_rank(const Tensor& tensor, int rank, const char* name) {
    if (static_cast<int>(tensor.shape().size()) != rank) {
        throw std::invalid_argument(std::string(name) + " rank mismatch.");
    }
}

void validate_vector_like(const Tensor& tensor, const char* name) {
    const auto shape = tensor.shape();
    if (shape.size() == 1) {
        if (tensor.size() != shape[0]) {
            throw std::invalid_argument(std::string(name) + " size mismatch.");
        }
        return;
    }
    if (shape.size() == 2 && (shape[0] == 1 || shape[1] == 1)) {
        if (tensor.size() != shape[0] * shape[1]) {
            throw std::invalid_argument(std::string(name) + " size mismatch.");
        }
        return;
    }
    throw std::invalid_argument(std::string(name) + " rank mismatch.");
}

void validate_linear_weight(const Tensor& weight, const char* name) {
    validate_rank(weight, 2, name);
    int expected = weight.shape()[0] * weight.shape()[1];
    if (weight.size() != expected) {
        throw std::invalid_argument(std::string(name) + " size mismatch.");
    }
}

bool bias_matches(const Tensor& bias, int expected) {
    if (bias.size() != expected) {
        return false;
    }
    const auto shape = bias.shape();
    if (shape.size() == 1 && shape[0] == expected) {
        return true;
    }
    if (shape.size() == 2 && shape[0] * shape[1] == expected &&
        (shape[0] == 1 || shape[1] == 1)) {
        return true;
    }
    return false;
}

template <typename Traits>
const void* get_or_upload_tensor(const Tensor& tensor,
                                 TensorUploadCache& cache,
                                 cudaStream_t stream,
                                 const char* error_msg) {
    const size_t bytes = static_cast<size_t>(tensor.size()) * Traits::kElementBytes;
    if (bytes == 0) {
        return nullptr;
    }
    const void* host_ptr = tensor.data().data();
    bool needs_upload = false;
    if (cache.device.bytes() < bytes) {
        cache.device.reset(next_capacity_bytes(cache.device.bytes(), bytes));
        needs_upload = true;
    }
    if (cache.host_ptr != host_ptr || cache.bytes != bytes) {
        needs_upload = true;
    }
    if (needs_upload) {
        cuda_check(cudaMemcpyAsync(cache.device.data(), host_ptr, bytes,
                                   cudaMemcpyHostToDevice, stream),
                   error_msg);
        cache.host_ptr = host_ptr;
        cache.bytes = bytes;
    }
    return cache.device.data();
}

template <typename Traits>
void launch_linear(cublasHandle_t handle,
                   const void* weight_ptr,
                   const void* input_ptr,
                   void* output_ptr,
                   int rows,
                   int in_dim,
                   int out_dim) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublas_check(
        cublasGemmEx(handle,
                     CUBLAS_OP_T, CUBLAS_OP_N,
                     out_dim, rows, in_dim,
                     &alpha,
                     weight_ptr, Traits::kCudaType, in_dim,
                     input_ptr, Traits::kCudaType, in_dim,
                     &beta,
                     output_ptr, Traits::kCudaType, out_dim,
                     Traits::kComputeType,
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP),
        "cublasGemmEx mlp linear");
}

template <typename Traits>
void maybe_add_bias(cudaStream_t stream,
                    void* output_ptr,
                    const void* bias_ptr,
                    int rows,
                    int cols) {
    if (bias_ptr == nullptr || cols <= 0) {
        return;
    }
    constexpr int kThreads = 256;
    int total = rows * cols;
    int blocks = (total + kThreads - 1) / kThreads;
    add_bias_kernel<typename Traits::DeviceType><<<blocks, kThreads, 0, stream>>>(
        static_cast<typename Traits::DeviceType*>(output_ptr),
        static_cast<const typename Traits::DeviceType*>(bias_ptr),
        rows, cols);
    cuda_check(cudaGetLastError(), "add_bias_kernel mlp");
}

} // namespace

Tensor mlp_forward_cuda(const Tensor& input,
                        const Tensor& norm_weight,
                        const Tensor& up_weight, const Tensor& up_bias,
                        const Tensor& gate_weight, const Tensor& gate_bias,
                        const Tensor& down_weight, const Tensor& down_bias) {
    using Traits = CudaPrecisionTraits<data_type>;
    using DeviceType = typename Traits::DeviceType;

    validate_rank(input, 3, "input");
    if (input.shape()[0] <= 0 || input.shape()[1] <= 0 || input.shape()[2] <= 0) {
        throw std::invalid_argument("mlp_forward_cuda: input dimensions must be positive.");
    }
    const int batch = input.shape()[0];
    const int seq_len = input.shape()[1];
    const int hidden_dim = input.shape()[2];
    const int rows = batch * seq_len;
    if (input.size() != rows * hidden_dim) {
        throw std::invalid_argument("mlp_forward_cuda: input size mismatch.");
    }
    validate_vector_like(norm_weight, "norm_weight");
    if (norm_weight.size() != hidden_dim) {
        throw std::invalid_argument("mlp_forward_cuda: norm_weight size mismatch.");
    }
    validate_linear_weight(up_weight, "up_weight");
    validate_linear_weight(gate_weight, "gate_weight");
    validate_linear_weight(down_weight, "down_weight");

    if (up_weight.shape()[1] != hidden_dim || gate_weight.shape()[1] != hidden_dim) {
        throw std::invalid_argument("mlp_forward_cuda: up/gate weight input dim mismatch.");
    }
    const int inter_dim = up_weight.shape()[0];
    if (inter_dim <= 0 || gate_weight.shape()[0] != inter_dim) {
        throw std::invalid_argument("mlp_forward_cuda: up/gate weight output dim mismatch.");
    }
    if (down_weight.shape()[1] != inter_dim || down_weight.shape()[0] != hidden_dim) {
        throw std::invalid_argument("mlp_forward_cuda: down weight shape mismatch.");
    }
    const bool use_up_bias = bias_matches(up_bias, inter_dim);
    const bool use_gate_bias = bias_matches(gate_bias, inter_dim);
    const bool use_down_bias = bias_matches(down_bias, hidden_dim);
    auto& ctx = get_context();
    if (!ctx.available()) {
        throw std::runtime_error("mlp_forward_cuda: CUDA runtime unavailable.");
    }

    const size_t input_bytes = static_cast<size_t>(rows) * hidden_dim * Traits::kElementBytes;
    const size_t inter_bytes = static_cast<size_t>(rows) * inter_dim * Traits::kElementBytes;
    const size_t output_bytes = input_bytes;
    Tensor output(rows * hidden_dim);
    std::lock_guard<std::mutex> lock(ctx.mutex());
    cudaStream_t stream = ctx.stream();
    cublasHandle_t handle = ctx.handle();

    auto& state = mlp_cuda_state();
    ensure_device_buffer(state.input, input_bytes);
    ensure_device_buffer(state.norm_out, input_bytes);
    ensure_device_buffer(state.up, inter_bytes);
    ensure_device_buffer(state.gate, inter_bytes);
    ensure_device_buffer(state.output, output_bytes);
    cuda_check(cudaMemcpyAsync(state.input.data(), input.data().data(), input_bytes,
                               cudaMemcpyHostToDevice, stream),
               "cudaMemcpyAsync mlp input");

    const void* d_norm_weight = get_or_upload_tensor<Traits>(
        norm_weight, state.norm_weight_cache, stream, "cudaMemcpyAsync mlp norm_weight");
    constexpr int kThreads = 256;
    rms_norm_kernel<DeviceType><<<rows, kThreads, kThreads * sizeof(float), stream>>>(
        static_cast<const DeviceType*>(state.input.data()),
        static_cast<const DeviceType*>(d_norm_weight),
        static_cast<DeviceType*>(state.norm_out.data()),
        rows, hidden_dim, 1e-6f);
    cuda_check(cudaGetLastError(), "rms_norm_kernel mlp");

    WeightEntry& up_weight_entry = ctx.cache().get_or_upload(up_weight, stream);
    WeightEntry& gate_weight_entry = ctx.cache().get_or_upload(gate_weight, stream);
    WeightEntry& down_weight_entry = ctx.cache().get_or_upload(down_weight, stream);
    launch_linear<Traits>(handle, up_weight_entry.device_ptr, state.norm_out.data(), state.up.data(),
                          rows, hidden_dim, inter_dim);
    launch_linear<Traits>(handle, gate_weight_entry.device_ptr, state.norm_out.data(), state.gate.data(),
                          rows, hidden_dim, inter_dim);

    const void* d_up_bias = nullptr;
    const void* d_gate_bias = nullptr;
    const void* d_down_bias = nullptr;
    if (use_up_bias) {
        d_up_bias = get_or_upload_tensor<Traits>(
            up_bias, state.up_bias_cache, stream, "cudaMemcpyAsync mlp up_bias");
    }
    if (use_gate_bias) {
        d_gate_bias = get_or_upload_tensor<Traits>(
            gate_bias, state.gate_bias_cache, stream, "cudaMemcpyAsync mlp gate_bias");
    }
    if (use_down_bias) {
        d_down_bias = get_or_upload_tensor<Traits>(
            down_bias, state.down_bias_cache, stream, "cudaMemcpyAsync mlp down_bias");
    }
    maybe_add_bias<Traits>(stream, state.up.data(), d_up_bias, rows, inter_dim);
    maybe_add_bias<Traits>(stream, state.gate.data(), d_gate_bias, rows, inter_dim);

    int silu_total = rows * inter_dim;
    int silu_blocks = (silu_total + kThreads - 1) / kThreads;
    silu_mul_inplace_kernel<DeviceType><<<silu_blocks, kThreads, 0, stream>>>(
        static_cast<DeviceType*>(state.up.data()),
        static_cast<const DeviceType*>(state.gate.data()),
        silu_total);
    cuda_check(cudaGetLastError(), "silu_mul_inplace_kernel");

    launch_linear<Traits>(handle, down_weight_entry.device_ptr, state.up.data(), state.output.data(),
                          rows, inter_dim, hidden_dim);
    maybe_add_bias<Traits>(stream, state.output.data(), d_down_bias, rows, hidden_dim);

    cuda_check(cudaMemcpyAsync(output.data().data(), state.output.data(), output_bytes,
                               cudaMemcpyDeviceToHost, stream),
               "cudaMemcpyAsync mlp output");
    cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize mlp_forward_cuda");
    output.reshape({batch, seq_len, hidden_dim});
    return output;
}

} // namespace ops
} // namespace cuda
} // namespace easy_llm
