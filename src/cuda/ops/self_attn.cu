#include "cuda/ops/self_attn.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cfloat>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

#include <cuda_bf16.h>

#include "cuda/precision.hpp"
#include "spdlog/spdlog.h"

#include "../device_buffer.hpp"
#include "../runtime_internal.hpp"

namespace easy_llm {
namespace cuda {
namespace ops {

struct SampleCache {
    DeviceBuffer k;
    DeviceBuffer v;
    int len{0};
    int capacity_len{0};
};

struct TensorUploadCache {
    DeviceBuffer device;
    const void* host_ptr{nullptr};
    size_t bytes{0};
};

struct IntVectorUploadCache {
    DeviceBuffer device;
    const int* host_ptr{nullptr};
    size_t count{0};
    std::uint64_t checksum{0};
};

struct ForwardScratchBuffers {
    DeviceBuffer input;
    DeviceBuffer norm_out;
    DeviceBuffer q_proj;
    DeviceBuffer k_proj;
    DeviceBuffer v_proj;
    DeviceBuffer q;
    DeviceBuffer k;
    DeviceBuffer v;
    DeviceBuffer offsets;
    DeviceBuffer cache_k_batch;
    DeviceBuffer cache_v_batch;
    DeviceBuffer scores;
    DeviceBuffer sample_ids;
    DeviceBuffer context;
    DeviceBuffer merged;
    DeviceBuffer output;
    DeviceBuffer decode_active_len;
};

struct DecodeGraphCache {
    cudaGraph_t graph{nullptr};
    cudaGraphExec_t exec{nullptr};
    int batch{0};
    int sample_id{-1};
    int num_heads{0};
    int head_dim{0};
    int hidden_dim{0};
    int q_out_dim{0};
    int k_out_dim{0};
    int v_out_dim{0};
    int o_out_dim{0};
    int repeat_factor{0};
    int score_capacity{0};
    int pad_size{0};
    const void* cache_k_ptr{nullptr};
    const void* cache_v_ptr{nullptr};
    const int* d_pad_ptr{nullptr};
    const void* d_norm_weight_ptr{nullptr};
    const void* q_weight_ptr{nullptr};
    const void* k_weight_ptr{nullptr};
    const void* v_weight_ptr{nullptr};
    const void* o_weight_ptr{nullptr};
    const void* d_q_bias_ptr{nullptr};
    const void* d_k_bias_ptr{nullptr};
    const void* d_v_bias_ptr{nullptr};
    const void* d_o_bias_ptr{nullptr};

    void reset() {
        if (exec != nullptr) {
            cudaGraphExecDestroy(exec);
            exec = nullptr;
        }
        if (graph != nullptr) {
            cudaGraphDestroy(graph);
            graph = nullptr;
        }
        batch = 0;
        sample_id = -1;
        num_heads = 0;
        head_dim = 0;
        hidden_dim = 0;
        q_out_dim = 0;
        k_out_dim = 0;
        v_out_dim = 0;
        o_out_dim = 0;
        repeat_factor = 0;
        score_capacity = 0;
        pad_size = 0;
        cache_k_ptr = nullptr;
        cache_v_ptr = nullptr;
        d_pad_ptr = nullptr;
        d_norm_weight_ptr = nullptr;
        q_weight_ptr = nullptr;
        k_weight_ptr = nullptr;
        v_weight_ptr = nullptr;
        o_weight_ptr = nullptr;
        d_q_bias_ptr = nullptr;
        d_k_bias_ptr = nullptr;
        d_v_bias_ptr = nullptr;
        d_o_bias_ptr = nullptr;
    }

    ~DecodeGraphCache() {
        reset();
    }
};

struct SelfAttnCudaState::Impl {
    std::vector<SampleCache> samples;
    int num_heads{0};
    int head_dim{0};
    ForwardScratchBuffers scratch;
    TensorUploadCache norm_weight_cache;
    TensorUploadCache q_bias_cache;
    TensorUploadCache k_bias_cache;
    TensorUploadCache v_bias_cache;
    TensorUploadCache o_bias_cache;
    IntVectorUploadCache pad_lens_cache;
    DecodeGraphCache decode_graph;
    SelfAttnCudaStats stats;
};

namespace {

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
                                int num_vectors,
                                int hidden_dim,
                                float epsilon) {
    extern __shared__ float shared[];
    int vec = blockIdx.x;
    if (vec >= num_vectors) {
        return;
    }
    int tid = threadIdx.x;
    float local_sum = 0.0f;
    int base = vec * hidden_dim;
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
__global__ void split_transpose_kernel(const T* input,
                                       T* output,
                                       int batch,
                                       int seq_len,
                                       int num_heads,
                                       int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * num_heads * seq_len * head_dim;
    if (idx >= total) {
        return;
    }
    int d = idx % head_dim;
    int s = (idx / head_dim) % seq_len;
    int h = (idx / (head_dim * seq_len)) % num_heads;
    int b = idx / (head_dim * seq_len * num_heads);
    int hidden_dim = num_heads * head_dim;
    int in_idx = ((b * seq_len + s) * hidden_dim) + (h * head_dim + d);
    output[idx] = input[in_idx];
}

template <typename T>
__global__ void split_transpose_seq1_kernel(const T* input,
                                            T* output,
                                            int batch,
                                            int num_heads,
                                            int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * num_heads * head_dim;
    if (idx >= total) {
        return;
    }
    int d = idx % head_dim;
    int h = (idx / head_dim) % num_heads;
    int b = idx / (head_dim * num_heads);
    int hidden_dim = num_heads * head_dim;
    int in_idx = b * hidden_dim + h * head_dim + d;
    output[idx] = input[in_idx];
}

template <typename T>
__global__ void append_kv_interleaved_kernel(const T* src_heads_major,
                                             T* dst_interleaved,
                                             int old_len,
                                             int seq_len,
                                             int num_heads_kv,
                                             int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * num_heads_kv * head_dim;
    if (idx >= total) {
        return;
    }
    int d = idx % head_dim;
    int s = (idx / head_dim) % seq_len;
    int h = idx / (head_dim * seq_len);
    int src_idx = ((h * seq_len + s) * head_dim) + d;
    int dst_idx = (((old_len + s) * num_heads_kv + h) * head_dim) + d;
    dst_interleaved[dst_idx] = src_heads_major[src_idx];
}

template <typename T>
__global__ void pack_cache_interleaved_to_heads_kernel(const T* src_interleaved,
                                                       T* dst_heads_major,
                                                       int total_len,
                                                       int num_heads_kv,
                                                       int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = total_len * num_heads_kv * head_dim;
    if (idx >= total) {
        return;
    }
    int d = idx % head_dim;
    int t = (idx / head_dim) % total_len;
    int h = idx / (head_dim * total_len);
    int src_idx = ((t * num_heads_kv + h) * head_dim) + d;
    int dst_idx = ((h * total_len + t) * head_dim) + d;
    dst_heads_major[dst_idx] = src_interleaved[src_idx];
}

template <typename T>
__global__ void rope_kernel(T* input,
                            const int* offsets,
                            int batch,
                            int num_heads,
                            int seq_len,
                            int head_dim,
                            float rope_theta) {
    int half_dim = head_dim / 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * num_heads * seq_len * half_dim;
    if (idx >= total) {
        return;
    }
    int k = idx % half_dim;
    int s = (idx / half_dim) % seq_len;
    int h = (idx / (half_dim * seq_len)) % num_heads;
    int b = idx / (half_dim * seq_len * num_heads);

    int base = ((b * num_heads + h) * seq_len + s) * head_dim;
    int idx1 = base + k;
    int idx2 = idx1 + half_dim;
    float x1 = to_float(input[idx1]);
    float x2 = to_float(input[idx2]);
    float freq = powf(rope_theta, (2.0f * static_cast<float>(k)) / static_cast<float>(head_dim));
    float angle = static_cast<float>(s + offsets[b]) / freq;
    float cos_v = cosf(angle);
    float sin_v = sinf(angle);
    input[idx1] = from_float<T>(x1 * cos_v - x2 * sin_v);
    input[idx2] = from_float<T>(x1 * sin_v + x2 * cos_v);
}

template <typename T>
__global__ void rope_seq1_kernel(T* input,
                                 const int* offsets,
                                 int batch,
                                 int num_heads,
                                 int head_dim,
                                 float rope_theta) {
    int half_dim = head_dim / 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * num_heads * half_dim;
    if (idx >= total) {
        return;
    }
    int k = idx % half_dim;
    int h = (idx / half_dim) % num_heads;
    int b = idx / (half_dim * num_heads);

    int base = (b * num_heads + h) * head_dim;
    int idx1 = base + k;
    int idx2 = idx1 + half_dim;
    float x1 = to_float(input[idx1]);
    float x2 = to_float(input[idx2]);
    float freq = powf(rope_theta, (2.0f * static_cast<float>(k)) / static_cast<float>(head_dim));
    float angle = static_cast<float>(offsets[b]) / freq;
    float cos_v = cosf(angle);
    float sin_v = sinf(angle);
    input[idx1] = from_float<T>(x1 * cos_v - x2 * sin_v);
    input[idx2] = from_float<T>(x1 * sin_v + x2 * cos_v);
}

template <typename T>
__global__ void mask_scale_softmax_kernel(T* scores,
                                          const int* sample_ids,
                                          const int* pad_lens,
                                          int pad_lens_size,
                                          int batch,
                                          int heads,
                                          int seq_len,
                                          int total_len,
                                          float scale) {
    extern __shared__ float shared[];
    int row = blockIdx.x;
    int row_count = batch * heads * seq_len;
    if (row >= row_count) {
        return;
    }
    int tid = threadIdx.x;

    int b = row / (heads * seq_len);
    int rem = row % (heads * seq_len);
    int s = rem % seq_len;
    int row_base = row * total_len;
    int mask_start = (total_len - seq_len) + s + 1;
    bool use_causal = seq_len > 1;

    int pad_len = 0;
    if (pad_lens != nullptr && pad_lens_size > 0) {
        int sample_id = sample_ids[b];
        if (sample_id >= 0 && sample_id < pad_lens_size) {
            pad_len = pad_lens[sample_id];
            if (pad_len < 0) {
                pad_len = 0;
            }
            if (pad_len > total_len) {
                pad_len = total_len;
            }
        }
    }

    float local_max = -INFINITY;
    for (int k = tid; k < total_len; k += blockDim.x) {
        float v = to_float(scores[row_base + k]);
        if (use_causal && k >= mask_start) {
            v = -INFINITY;
        }
        if (pad_len > 0 && k < pad_len) {
            v = -INFINITY;
        }
        v *= scale;
        local_max = fmaxf(local_max, v);
    }
    shared[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
        __syncthreads();
    }
    float row_max = shared[0];

    float local_sum = 0.0f;
    for (int k = tid; k < total_len; k += blockDim.x) {
        float v = to_float(scores[row_base + k]);
        if (use_causal && k >= mask_start) {
            v = -INFINITY;
        }
        if (pad_len > 0 && k < pad_len) {
            v = -INFINITY;
        }
        v *= scale;
        float exp_v = 0.0f;
        if (isfinite(row_max)) {
            exp_v = expf(v - row_max);
        }
        scores[row_base + k] = from_float<T>(exp_v);
        local_sum += exp_v;
    }
    shared[tid] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    float denom = fmaxf(shared[0], FLT_EPSILON * 10.0f);
    for (int k = tid; k < total_len; k += blockDim.x) {
        float p = to_float(scores[row_base + k]) / denom;
        scores[row_base + k] = from_float<T>(p);
    }
}

template <typename T>
__global__ void mask_scale_softmax_decode_kernel(T* scores,
                                                 const int* sample_ids,
                                                 const int* pad_lens,
                                                 int pad_lens_size,
                                                 int batch,
                                                 int heads,
                                                 int score_len,
                                                 const int* active_len_ptr,
                                                 float scale) {
    extern __shared__ float shared[];
    int row = blockIdx.x;
    int row_count = batch * heads;
    if (row >= row_count) {
        return;
    }
    int tid = threadIdx.x;

    int b = row / heads;
    int row_base = row * score_len;
    int active_len = score_len;
    if (active_len_ptr != nullptr) {
        active_len = active_len_ptr[0];
    }
    if (active_len < 0) {
        active_len = 0;
    }
    if (active_len > score_len) {
        active_len = score_len;
    }

    int pad_len = 0;
    if (pad_lens != nullptr && pad_lens_size > 0) {
        int sample_id = sample_ids[b];
        if (sample_id >= 0 && sample_id < pad_lens_size) {
            pad_len = pad_lens[sample_id];
            if (pad_len < 0) {
                pad_len = 0;
            }
            if (pad_len > active_len) {
                pad_len = active_len;
            }
        }
    }

    float local_max = -INFINITY;
    for (int k = tid; k < score_len; k += blockDim.x) {
        float v = to_float(scores[row_base + k]);
        if (k >= active_len || (pad_len > 0 && k < pad_len)) {
            v = -INFINITY;
        }
        v *= scale;
        local_max = fmaxf(local_max, v);
    }
    shared[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
        __syncthreads();
    }
    float row_max = shared[0];

    float local_sum = 0.0f;
    for (int k = tid; k < score_len; k += blockDim.x) {
        float v = to_float(scores[row_base + k]);
        if (k >= active_len || (pad_len > 0 && k < pad_len)) {
            v = -INFINITY;
        }
        v *= scale;
        float exp_v = 0.0f;
        if (isfinite(row_max)) {
            exp_v = expf(v - row_max);
        }
        scores[row_base + k] = from_float<T>(exp_v);
        local_sum += exp_v;
    }
    shared[tid] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    float denom = fmaxf(shared[0], FLT_EPSILON * 10.0f);
    for (int k = tid; k < score_len; k += blockDim.x) {
        float p = to_float(scores[row_base + k]) / denom;
        scores[row_base + k] = from_float<T>(p);
    }
}

template <typename T>
__global__ void merge_heads_kernel(const T* input,
                                   T* output,
                                   int batch,
                                   int seq_len,
                                   int num_heads,
                                   int head_dim) {
    int hidden_dim = num_heads * head_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * seq_len * hidden_dim;
    if (idx >= total) {
        return;
    }
    int hidden_idx = idx % hidden_dim;
    int d = hidden_idx % head_dim;
    int h = hidden_idx / head_dim;
    int s = (idx / hidden_dim) % seq_len;
    int b = idx / (hidden_dim * seq_len);
    int in_idx = ((b * num_heads + h) * seq_len + s) * head_dim + d;
    output[idx] = input[in_idx];
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

void validate_bias(const Tensor& bias, const char* name) {
    validate_rank(bias, 1, name);
    if (bias.size() != bias.shape()[0]) {
        throw std::invalid_argument(std::string(name) + " size mismatch.");
    }
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
        "cublasGemmEx linear");
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
    int total = rows * cols;
    constexpr int kThreads = 256;
    int blocks = (total + kThreads - 1) / kThreads;
    add_bias_kernel<typename Traits::DeviceType><<<blocks, kThreads, 0, stream>>>(
        static_cast<typename Traits::DeviceType*>(output_ptr),
        static_cast<const typename Traits::DeviceType*>(bias_ptr),
        rows, cols);
    cuda_check(cudaGetLastError(), "add_bias_kernel");
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

void ensure_device_buffer(DeviceBuffer& buffer,
                          size_t required_bytes,
                          std::uint64_t* realloc_counter = nullptr) {
    if (required_bytes == 0 || buffer.bytes() >= required_bytes) {
        return;
    }
    const size_t new_capacity = next_capacity_bytes(buffer.bytes(), required_bytes);
    buffer.reset(new_capacity);
    if (realloc_counter != nullptr) {
        *realloc_counter += 1;
    }
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

const int* get_or_upload_int_vector(const std::vector<int>& input,
                                    IntVectorUploadCache& cache,
                                    cudaStream_t stream,
                                    const char* error_msg,
                                    std::uint64_t* upload_counter = nullptr) {
    if (input.empty()) {
        return nullptr;
    }
    const size_t bytes = static_cast<size_t>(input.size()) * sizeof(int);
    bool needs_upload = false;
    std::uint64_t checksum = 1469598103934665603ull;
    for (int value : input) {
        checksum ^= static_cast<std::uint64_t>(static_cast<std::uint32_t>(value));
        checksum *= 1099511628211ull;
    }
    if (cache.device.bytes() < bytes) {
        cache.device.reset(next_capacity_bytes(cache.device.bytes(), bytes));
        needs_upload = true;
    }
    if (cache.host_ptr != input.data() || cache.count != input.size() || cache.checksum != checksum) {
        needs_upload = true;
    }
    if (needs_upload) {
        cuda_check(cudaMemcpyAsync(cache.device.data(), input.data(), bytes,
                                   cudaMemcpyHostToDevice, stream),
                   error_msg);
        cache.host_ptr = input.data();
        cache.count = input.size();
        cache.checksum = checksum;
        if (upload_counter != nullptr) {
            *upload_counter += 1;
        }
    }
    return static_cast<const int*>(cache.device.data());
}

int next_capacity_len(int current_capacity, int required_len) {
    if (required_len <= 0) {
        return 0;
    }
    int capacity = std::max(current_capacity, 0);
    if (capacity == 0) {
        capacity = std::max(required_len, 8);
    }
    while (capacity < required_len) {
        if (capacity > std::numeric_limits<int>::max() / 2) {
            capacity = required_len;
            break;
        }
        capacity *= 2;
    }
    return capacity;
}

void ensure_sample_capacity(SelfAttnCudaState::Impl& impl, int min_size) {
    if (min_size <= 0) {
        return;
    }
    if (static_cast<int>(impl.samples.size()) < min_size) {
        impl.samples.resize(min_size);
    }
}

template <typename Traits>
void append_sample_cache(SampleCache& sample,
                         const void* k_slice,
                         const void* v_slice,
                         int num_heads_kv,
                         int seq_len,
                         int head_dim,
                         cudaStream_t stream) {
    using DeviceType = typename Traits::DeviceType;
    constexpr int kThreads = 256;
    const size_t elem_bytes = Traits::kElementBytes;
    const int old_len = sample.len;
    const int new_len = old_len + seq_len;
    const size_t row_elems = static_cast<size_t>(num_heads_kv) * head_dim;
    if (new_len > sample.capacity_len) {
        const int new_capacity = next_capacity_len(sample.capacity_len, new_len);
        const size_t total_new_bytes = static_cast<size_t>(new_capacity) * row_elems * elem_bytes;

        DeviceBuffer new_k(total_new_bytes);
        DeviceBuffer new_v(total_new_bytes);
        if (old_len > 0) {
            const size_t old_copy_bytes = static_cast<size_t>(old_len) * row_elems * elem_bytes;
            cuda_check(cudaMemcpyAsync(new_k.data(), sample.k.data(), old_copy_bytes, cudaMemcpyDeviceToDevice, stream),
                       "cudaMemcpyAsync cache k old");
            cuda_check(cudaMemcpyAsync(new_v.data(), sample.v.data(), old_copy_bytes, cudaMemcpyDeviceToDevice, stream),
                       "cudaMemcpyAsync cache v old");
        }
        sample.k = std::move(new_k);
        sample.v = std::move(new_v);
        sample.capacity_len = new_capacity;
    }

    const int append_total = seq_len * num_heads_kv * head_dim;
    const int append_blocks = (append_total + kThreads - 1) / kThreads;
    append_kv_interleaved_kernel<DeviceType><<<append_blocks, kThreads, 0, stream>>>(
        static_cast<const DeviceType*>(k_slice),
        static_cast<DeviceType*>(sample.k.data()),
        old_len, seq_len, num_heads_kv, head_dim);
    append_kv_interleaved_kernel<DeviceType><<<append_blocks, kThreads, 0, stream>>>(
        static_cast<const DeviceType*>(v_slice),
        static_cast<DeviceType*>(sample.v.data()),
        old_len, seq_len, num_heads_kv, head_dim);
    cuda_check(cudaGetLastError(), "append_kv_interleaved_kernel");
    sample.len = new_len;
}

template <typename Traits>
int append_kv_cache(SelfAttnCudaState::Impl& impl,
                    const std::vector<int>& sample_ids,
                    int batch,
                    int seq_len,
                    int num_heads_kv,
                    int head_dim,
                    const DeviceBuffer& d_k,
                    const DeviceBuffer& d_v,
                    cudaStream_t stream) {
    const size_t elem_bytes = Traits::kElementBytes;
    const size_t sample_bytes = static_cast<size_t>(num_heads_kv) * seq_len * head_dim * elem_bytes;
    for (int i = 0; i < batch; ++i) {
        int sample_id = sample_ids[i];
        if (sample_id < 0) {
            throw std::invalid_argument("sample_id must be non-negative.");
        }
        ensure_sample_capacity(impl, sample_id + 1);
        SampleCache& sample = impl.samples[sample_id];
        const char* k_slice = static_cast<const char*>(d_k.data()) + static_cast<size_t>(i) * sample_bytes;
        const char* v_slice = static_cast<const char*>(d_v.data()) + static_cast<size_t>(i) * sample_bytes;
        append_sample_cache<Traits>(sample, k_slice, v_slice, num_heads_kv, seq_len, head_dim, stream);
    }
    int total_len = 0;
    if (!sample_ids.empty()) {
        total_len = impl.samples[sample_ids[0]].len;
        for (int sample_id : sample_ids) {
            if (sample_id < 0 || sample_id >= static_cast<int>(impl.samples.size())) {
                throw std::out_of_range("sample_id out of cache range.");
            }
            if (impl.samples[sample_id].len != total_len) {
                throw std::invalid_argument("Active sample cache lengths must match.");
            }
        }
    }
    return total_len;
}

template <typename Traits>
void build_cache_batch(const SelfAttnCudaState::Impl& impl,
                       const std::vector<int>& sample_ids,
                       int num_heads_kv,
                       int total_len,
                       int head_dim,
                       DeviceBuffer& d_cache_k_batch,
                       DeviceBuffer& d_cache_v_batch,
                       cudaStream_t stream) {
    using DeviceType = typename Traits::DeviceType;
    constexpr int kThreads = 256;
    const size_t elem_bytes = Traits::kElementBytes;
    const int batch = static_cast<int>(sample_ids.size());
    const size_t sample_bytes = static_cast<size_t>(num_heads_kv) * total_len * head_dim * elem_bytes;
    ensure_device_buffer(d_cache_k_batch, sample_bytes * batch);
    ensure_device_buffer(d_cache_v_batch, sample_bytes * batch);
    const int pack_total = total_len * num_heads_kv * head_dim;
    const int pack_blocks = (pack_total + kThreads - 1) / kThreads;
    for (int i = 0; i < batch; ++i) {
        int sample_id = sample_ids[i];
        const SampleCache& sample = impl.samples[sample_id];
        if (sample.len != total_len) {
            throw std::invalid_argument("Cache length mismatch while building batch cache.");
        }
        char* dst_k = static_cast<char*>(d_cache_k_batch.data()) + static_cast<size_t>(i) * sample_bytes;
        char* dst_v = static_cast<char*>(d_cache_v_batch.data()) + static_cast<size_t>(i) * sample_bytes;
        pack_cache_interleaved_to_heads_kernel<DeviceType><<<pack_blocks, kThreads, 0, stream>>>(
            static_cast<const DeviceType*>(sample.k.data()),
            reinterpret_cast<DeviceType*>(dst_k),
            total_len, num_heads_kv, head_dim);
        pack_cache_interleaved_to_heads_kernel<DeviceType><<<pack_blocks, kThreads, 0, stream>>>(
            static_cast<const DeviceType*>(sample.v.data()),
            reinterpret_cast<DeviceType*>(dst_v),
            total_len, num_heads_kv, head_dim);
    }
    cuda_check(cudaGetLastError(), "pack_cache_interleaved_to_heads_kernel");
}

template <typename Traits>
void launch_qk_grouped_batched_gemm(cublasHandle_t handle,
                                    const void* d_cache_k,
                                    const void* d_q,
                                    void* d_scores,
                                    int batch,
                                    int num_heads_kv,
                                    int repeat_factor,
                                    int seq_len,
                                    int total_len,
                                    int head_dim,
                                    int cache_leading_dim,
                                    long long cache_instance_stride_elems) {
    const int grouped_seq = seq_len * repeat_factor;
    const int batch_count = batch * num_heads_kv;
    const long long stride_a = cache_instance_stride_elems;
    const long long stride_b = static_cast<long long>(grouped_seq) * head_dim;
    const long long stride_c = static_cast<long long>(grouped_seq) * total_len;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublas_check(
        cublasGemmStridedBatchedEx(handle,
                                   CUBLAS_OP_T, CUBLAS_OP_N,
                                   total_len, grouped_seq, head_dim,
                                   &alpha,
                                   d_cache_k, Traits::kCudaType, cache_leading_dim, stride_a,
                                   d_q, Traits::kCudaType, head_dim, stride_b,
                                   &beta,
                                   d_scores, Traits::kCudaType, total_len, stride_c,
                                   batch_count, Traits::kComputeType,
                                   CUBLAS_GEMM_DEFAULT_TENSOR_OP),
        "cublasGemmStridedBatchedEx qk grouped");
}

template <typename Traits>
void launch_av_grouped_batched_gemm(cublasHandle_t handle,
                                    const void* d_scores,
                                    const void* d_cache_v,
                                    void* d_context,
                                    int batch,
                                    int num_heads_kv,
                                    int repeat_factor,
                                    int seq_len,
                                    int total_len,
                                    int head_dim,
                                    int cache_leading_dim,
                                    long long cache_instance_stride_elems) {
    const int grouped_seq = seq_len * repeat_factor;
    const int batch_count = batch * num_heads_kv;
    const long long stride_a = cache_instance_stride_elems;
    const long long stride_b = static_cast<long long>(grouped_seq) * total_len;
    const long long stride_c = static_cast<long long>(grouped_seq) * head_dim;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublas_check(
        cublasGemmStridedBatchedEx(handle,
                                   CUBLAS_OP_N, CUBLAS_OP_N,
                                   head_dim, grouped_seq, total_len,
                                   &alpha,
                                   d_cache_v, Traits::kCudaType, cache_leading_dim, stride_a,
                                   d_scores, Traits::kCudaType, total_len, stride_b,
                                   &beta,
                                   d_context, Traits::kCudaType, head_dim, stride_c,
                                   batch_count, Traits::kComputeType,
                                   CUBLAS_GEMM_DEFAULT_TENSOR_OP),
        "cublasGemmStridedBatchedEx av grouped");
}

bool can_reuse_decode_graph(const DecodeGraphCache& graph,
                            int batch,
                            int sample_id,
                            int num_heads,
                            int head_dim,
                            int hidden_dim,
                            int q_out_dim,
                            int k_out_dim,
                            int v_out_dim,
                            int o_out_dim,
                            int repeat_factor,
                            int score_capacity,
                            int pad_size,
                            const void* cache_k_ptr,
                            const void* cache_v_ptr,
                            const int* d_pad_ptr,
                            const void* d_norm_weight,
                            const void* q_weight_ptr,
                            const void* k_weight_ptr,
                            const void* v_weight_ptr,
                            const void* o_weight_ptr,
                            const void* d_q_bias,
                            const void* d_k_bias,
                            const void* d_v_bias,
                            const void* d_o_bias) {
    return graph.exec != nullptr &&
           graph.batch == batch &&
           graph.sample_id == sample_id &&
           graph.num_heads == num_heads &&
           graph.head_dim == head_dim &&
           graph.hidden_dim == hidden_dim &&
           graph.q_out_dim == q_out_dim &&
           graph.k_out_dim == k_out_dim &&
           graph.v_out_dim == v_out_dim &&
           graph.o_out_dim == o_out_dim &&
           graph.repeat_factor == repeat_factor &&
           graph.score_capacity == score_capacity &&
           graph.pad_size == pad_size &&
           graph.cache_k_ptr == cache_k_ptr &&
           graph.cache_v_ptr == cache_v_ptr &&
           graph.d_pad_ptr == d_pad_ptr &&
           graph.d_norm_weight_ptr == d_norm_weight &&
           graph.q_weight_ptr == q_weight_ptr &&
           graph.k_weight_ptr == k_weight_ptr &&
           graph.v_weight_ptr == v_weight_ptr &&
           graph.o_weight_ptr == o_weight_ptr &&
           graph.d_q_bias_ptr == d_q_bias &&
           graph.d_k_bias_ptr == d_k_bias &&
           graph.d_v_bias_ptr == d_v_bias &&
           graph.d_o_bias_ptr == d_o_bias;
}

void fill_decode_graph_signature(DecodeGraphCache& graph,
                                 int batch,
                                 int sample_id,
                                 int num_heads,
                                 int head_dim,
                                 int hidden_dim,
                                 int q_out_dim,
                                 int k_out_dim,
                                 int v_out_dim,
                                 int o_out_dim,
                                 int repeat_factor,
                                 int score_capacity,
                                 int pad_size,
                                 const void* cache_k_ptr,
                                 const void* cache_v_ptr,
                                 const int* d_pad_ptr,
                                 const void* d_norm_weight,
                                 const void* q_weight_ptr,
                                 const void* k_weight_ptr,
                                 const void* v_weight_ptr,
                                 const void* o_weight_ptr,
                                 const void* d_q_bias,
                                 const void* d_k_bias,
                                 const void* d_v_bias,
                                 const void* d_o_bias) {
    graph.batch = batch;
    graph.sample_id = sample_id;
    graph.num_heads = num_heads;
    graph.head_dim = head_dim;
    graph.hidden_dim = hidden_dim;
    graph.q_out_dim = q_out_dim;
    graph.k_out_dim = k_out_dim;
    graph.v_out_dim = v_out_dim;
    graph.o_out_dim = o_out_dim;
    graph.repeat_factor = repeat_factor;
    graph.score_capacity = score_capacity;
    graph.pad_size = pad_size;
    graph.cache_k_ptr = cache_k_ptr;
    graph.cache_v_ptr = cache_v_ptr;
    graph.d_pad_ptr = d_pad_ptr;
    graph.d_norm_weight_ptr = d_norm_weight;
    graph.q_weight_ptr = q_weight_ptr;
    graph.k_weight_ptr = k_weight_ptr;
    graph.v_weight_ptr = v_weight_ptr;
    graph.o_weight_ptr = o_weight_ptr;
    graph.d_q_bias_ptr = d_q_bias;
    graph.d_k_bias_ptr = d_k_bias;
    graph.d_v_bias_ptr = d_v_bias;
    graph.d_o_bias_ptr = d_o_bias;
}

} // namespace

SelfAttnCudaState::SelfAttnCudaState()
    : impl_(std::make_unique<Impl>()) {}

SelfAttnCudaState::~SelfAttnCudaState() = default;

SelfAttnCudaState::SelfAttnCudaState(SelfAttnCudaState&& other) noexcept = default;

SelfAttnCudaState& SelfAttnCudaState::operator=(SelfAttnCudaState&& other) noexcept = default;

void SelfAttnCudaState::init_kv_cache(int batch_size) {
    if (batch_size < 0) {
        throw std::invalid_argument("init_kv_cache: batch_size must be non-negative.");
    }
    impl_->samples.clear();
    impl_->samples.resize(batch_size);
    impl_->decode_graph.reset();
}

void SelfAttnCudaState::clear_kv_cache(int sample_id) {
    if (sample_id < 0 || sample_id >= static_cast<int>(impl_->samples.size())) {
        return;
    }
    SampleCache& sample = impl_->samples[sample_id];
    sample.k = DeviceBuffer();
    sample.v = DeviceBuffer();
    sample.len = 0;
    sample.capacity_len = 0;
    impl_->decode_graph.reset();
}

void SelfAttnCudaState::reset_kv_cache() {
    impl_->samples.clear();
    impl_->decode_graph.reset();
}

int SelfAttnCudaState::cache_len(int sample_id) const {
    if (sample_id < 0 || sample_id >= static_cast<int>(impl_->samples.size())) {
        return 0;
    }
    return impl_->samples[sample_id].len;
}

SelfAttnCudaStats SelfAttnCudaState::stats() const {
    return impl_->stats;
}

void SelfAttnCudaState::reset_stats() {
    impl_->stats = SelfAttnCudaStats{};
}

Tensor self_attn_forward_cuda(const Tensor& input,
                              const std::vector<int>& sample_ids,
                              const std::vector<int>& offsets,
                              const std::vector<int>& pad_lens_by_sample,
                              const SelfAttnCudaParams& params,
                              const Tensor& norm_weight,
                              const Tensor& q_weight, const Tensor& q_bias,
                              const Tensor& k_weight, const Tensor& k_bias,
                              const Tensor& v_weight, const Tensor& v_bias,
                              const Tensor& o_weight, const Tensor& o_bias,
                              SelfAttnCudaState& state) {
    using Traits = CudaPrecisionTraits<data_type>;
    using DeviceType = typename Traits::DeviceType;

    if (params.hidden_dim <= 0 || params.num_heads <= 0 || params.num_heads_kv <= 0 ||
        params.head_dim <= 0 || params.rope_theta <= 0.0f) {
        throw std::invalid_argument("self_attn_forward_cuda: invalid params.");
    }
    if (params.num_heads % params.num_heads_kv != 0) {
        throw std::invalid_argument("self_attn_forward_cuda: num_heads must be divisible by num_heads_kv.");
    }
    if (params.num_heads * params.head_dim != params.hidden_dim) {
        throw std::invalid_argument("self_attn_forward_cuda: hidden_dim mismatch with num_heads * head_dim.");
    }
    if (input.shape().size() != 3) {
        throw std::invalid_argument("self_attn_forward_cuda: input must be [batch, seq, hidden].");
    }
    int batch = input.shape()[0];
    int seq_len = input.shape()[1];
    int hidden_dim = input.shape()[2];
    if (hidden_dim != params.hidden_dim) {
        throw std::invalid_argument("self_attn_forward_cuda: input hidden_dim mismatch.");
    }
    if (input.size() != batch * seq_len * hidden_dim) {
        throw std::invalid_argument("self_attn_forward_cuda: input size mismatch.");
    }
    if (sample_ids.size() != static_cast<size_t>(batch)) {
        throw std::invalid_argument("self_attn_forward_cuda: sample_ids size must match batch.");
    }
    if (offsets.size() != static_cast<size_t>(batch)) {
        throw std::invalid_argument("self_attn_forward_cuda: offsets size must match batch.");
    }

    validate_vector_like(norm_weight, "norm_weight");
    if (norm_weight.size() != hidden_dim) {
        throw std::invalid_argument("self_attn_forward_cuda: norm_weight size mismatch.");
    }
    validate_linear_weight(q_weight, "q_weight");
    validate_linear_weight(k_weight, "k_weight");
    validate_linear_weight(v_weight, "v_weight");
    validate_linear_weight(o_weight, "o_weight");
    validate_bias(q_bias, "q_bias");
    validate_bias(k_bias, "k_bias");
    validate_bias(v_bias, "v_bias");
    validate_bias(o_bias, "o_bias");

    const int q_out_dim = q_weight.shape()[0];
    const int k_out_dim = k_weight.shape()[0];
    const int v_out_dim = v_weight.shape()[0];
    const int o_out_dim = o_weight.shape()[0];
    if (q_weight.shape()[1] != hidden_dim || k_weight.shape()[1] != hidden_dim || v_weight.shape()[1] != hidden_dim) {
        throw std::invalid_argument("self_attn_forward_cuda: q/k/v weight input dim mismatch.");
    }
    if (q_out_dim != params.hidden_dim) {
        throw std::invalid_argument("self_attn_forward_cuda: q_weight output dim mismatch.");
    }
    if (k_out_dim != params.num_heads_kv * params.head_dim ||
        v_out_dim != params.num_heads_kv * params.head_dim) {
        throw std::invalid_argument("self_attn_forward_cuda: k/v weight output dim mismatch.");
    }
    if (o_weight.shape()[1] != params.hidden_dim || o_out_dim != params.hidden_dim) {
        throw std::invalid_argument("self_attn_forward_cuda: o_weight shape mismatch.");
    }
    if (params.head_dim % 2 != 0) {
        throw std::invalid_argument("self_attn_forward_cuda: head_dim must be even for RoPE.");
    }

    if (state.impl_->num_heads == 0) {
        state.impl_->num_heads = params.num_heads;
        state.impl_->head_dim = params.head_dim;
    } else if (state.impl_->num_heads != params.num_heads || state.impl_->head_dim != params.head_dim) {
        throw std::invalid_argument("self_attn_forward_cuda: state head config mismatch.");
    }

    int max_sample_id = -1;
    for (int sample_id : sample_ids) {
        if (sample_id < 0) {
            throw std::invalid_argument("self_attn_forward_cuda: sample_id must be non-negative.");
        }
        max_sample_id = std::max(max_sample_id, sample_id);
    }
    ensure_sample_capacity(*state.impl_, max_sample_id + 1);

    auto& ctx = get_context();
    if (!ctx.available()) {
        throw std::runtime_error("self_attn_forward_cuda: CUDA runtime unavailable.");
    }

    std::lock_guard<std::mutex> lock(ctx.mutex());
    cudaStream_t stream = ctx.stream();
    cublasHandle_t handle = ctx.handle();

    const int rows = batch * seq_len;
    const int repeat_factor = params.num_heads / params.num_heads_kv;
    const size_t input_bytes = static_cast<size_t>(rows) * hidden_dim * Traits::kElementBytes;
    auto& scratch = state.impl_->scratch;
    std::uint64_t* scratch_realloc_counter = &state.impl_->stats.scratch_reallocations;

    ensure_device_buffer(scratch.input, input_bytes, scratch_realloc_counter);
    cuda_check(cudaMemcpyAsync(scratch.input.data(), input.data().data(), input_bytes,
                               cudaMemcpyHostToDevice, stream),
               "cudaMemcpyAsync input");

    const void* d_norm_weight = get_or_upload_tensor<Traits>(
        norm_weight, state.impl_->norm_weight_cache, stream, "cudaMemcpyAsync norm_weight");

    ensure_device_buffer(scratch.norm_out, input_bytes, scratch_realloc_counter);
    constexpr int kThreads = 256;
    rms_norm_kernel<DeviceType><<<rows, kThreads, kThreads * sizeof(float), stream>>>(
        static_cast<const DeviceType*>(scratch.input.data()),
        static_cast<const DeviceType*>(d_norm_weight),
        static_cast<DeviceType*>(scratch.norm_out.data()),
        rows, hidden_dim, 1e-6f);
    cuda_check(cudaGetLastError(), "rms_norm_kernel");

    WeightEntry& q_weight_entry = ctx.cache().get_or_upload(q_weight, stream);
    WeightEntry& k_weight_entry = ctx.cache().get_or_upload(k_weight, stream);
    WeightEntry& v_weight_entry = ctx.cache().get_or_upload(v_weight, stream);
    WeightEntry& o_weight_entry = ctx.cache().get_or_upload(o_weight, stream);

    ensure_device_buffer(scratch.q_proj, static_cast<size_t>(rows) * q_out_dim * Traits::kElementBytes, scratch_realloc_counter);
    ensure_device_buffer(scratch.k_proj, static_cast<size_t>(rows) * k_out_dim * Traits::kElementBytes, scratch_realloc_counter);
    ensure_device_buffer(scratch.v_proj, static_cast<size_t>(rows) * v_out_dim * Traits::kElementBytes, scratch_realloc_counter);
    launch_linear<Traits>(handle, q_weight_entry.device_ptr, scratch.norm_out.data(), scratch.q_proj.data(), rows, hidden_dim, q_out_dim);
    launch_linear<Traits>(handle, k_weight_entry.device_ptr, scratch.norm_out.data(), scratch.k_proj.data(), rows, hidden_dim, k_out_dim);
    launch_linear<Traits>(handle, v_weight_entry.device_ptr, scratch.norm_out.data(), scratch.v_proj.data(), rows, hidden_dim, v_out_dim);

    const void* d_q_bias = nullptr;
    const void* d_k_bias = nullptr;
    const void* d_v_bias = nullptr;
    const void* d_o_bias = nullptr;
    if (q_bias.size() == q_out_dim) {
        d_q_bias = get_or_upload_tensor<Traits>(q_bias, state.impl_->q_bias_cache, stream, "cudaMemcpyAsync q_bias");
    }
    if (k_bias.size() == k_out_dim) {
        d_k_bias = get_or_upload_tensor<Traits>(k_bias, state.impl_->k_bias_cache, stream, "cudaMemcpyAsync k_bias");
    }
    if (v_bias.size() == v_out_dim) {
        d_v_bias = get_or_upload_tensor<Traits>(v_bias, state.impl_->v_bias_cache, stream, "cudaMemcpyAsync v_bias");
    }
    if (o_bias.size() == o_out_dim) {
        d_o_bias = get_or_upload_tensor<Traits>(o_bias, state.impl_->o_bias_cache, stream, "cudaMemcpyAsync o_bias");
    }

    if (seq_len == 1) {
        state.impl_->stats.decode_seq1_path_hits += 1;

        maybe_add_bias<Traits>(stream, scratch.q_proj.data(), d_q_bias, rows, q_out_dim);
        maybe_add_bias<Traits>(stream, scratch.k_proj.data(), d_k_bias, rows, k_out_dim);
        maybe_add_bias<Traits>(stream, scratch.v_proj.data(), d_v_bias, rows, v_out_dim);

        ensure_device_buffer(scratch.q, static_cast<size_t>(batch) * params.num_heads * params.head_dim * Traits::kElementBytes, scratch_realloc_counter);
        ensure_device_buffer(scratch.k, static_cast<size_t>(batch) * params.num_heads_kv * params.head_dim * Traits::kElementBytes, scratch_realloc_counter);
        ensure_device_buffer(scratch.v, static_cast<size_t>(batch) * params.num_heads_kv * params.head_dim * Traits::kElementBytes, scratch_realloc_counter);

        const int q_total = batch * params.num_heads * params.head_dim;
        const int kv_total = batch * params.num_heads_kv * params.head_dim;
        const int q_blocks = (q_total + kThreads - 1) / kThreads;
        const int kv_blocks = (kv_total + kThreads - 1) / kThreads;
        split_transpose_seq1_kernel<DeviceType><<<q_blocks, kThreads, 0, stream>>>(
            static_cast<const DeviceType*>(scratch.q_proj.data()),
            static_cast<DeviceType*>(scratch.q.data()),
            batch, params.num_heads, params.head_dim);
        split_transpose_seq1_kernel<DeviceType><<<kv_blocks, kThreads, 0, stream>>>(
            static_cast<const DeviceType*>(scratch.k_proj.data()),
            static_cast<DeviceType*>(scratch.k.data()),
            batch, params.num_heads_kv, params.head_dim);
        split_transpose_seq1_kernel<DeviceType><<<kv_blocks, kThreads, 0, stream>>>(
            static_cast<const DeviceType*>(scratch.v_proj.data()),
            static_cast<DeviceType*>(scratch.v.data()),
            batch, params.num_heads_kv, params.head_dim);
        cuda_check(cudaGetLastError(), "split_transpose_seq1_kernel");

        ensure_device_buffer(scratch.offsets, static_cast<size_t>(batch) * sizeof(int), scratch_realloc_counter);
        cuda_check(cudaMemcpyAsync(scratch.offsets.data(), offsets.data(), static_cast<size_t>(batch) * sizeof(int),
                                   cudaMemcpyHostToDevice, stream),
                   "cudaMemcpyAsync offsets");

        const int rope_q_total = batch * params.num_heads * (params.head_dim / 2);
        const int rope_k_total = batch * params.num_heads_kv * (params.head_dim / 2);
        const int rope_q_blocks = (rope_q_total + kThreads - 1) / kThreads;
        const int rope_k_blocks = (rope_k_total + kThreads - 1) / kThreads;
        rope_seq1_kernel<DeviceType><<<rope_q_blocks, kThreads, 0, stream>>>(
            static_cast<DeviceType*>(scratch.q.data()),
            static_cast<const int*>(scratch.offsets.data()),
            batch, params.num_heads, params.head_dim, params.rope_theta);
        rope_seq1_kernel<DeviceType><<<rope_k_blocks, kThreads, 0, stream>>>(
            static_cast<DeviceType*>(scratch.k.data()),
            static_cast<const int*>(scratch.offsets.data()),
            batch, params.num_heads_kv, params.head_dim, params.rope_theta);
        cuda_check(cudaGetLastError(), "rope_seq1_kernel");

        int total_len = append_kv_cache<Traits>(*state.impl_, sample_ids, batch, seq_len, params.num_heads_kv,
                                                params.head_dim, scratch.k, scratch.v, stream);
        if (total_len <= 0) {
            throw std::runtime_error("self_attn_forward_cuda: cache length is zero after append.");
        }

        const void* cache_k_ptr = nullptr;
        const void* cache_v_ptr = nullptr;
        int score_capacity = total_len;
        int cache_leading_dim = params.head_dim;
        long long cache_instance_stride_elems = static_cast<long long>(total_len) * params.head_dim;
        if (batch == 1) {
            const SampleCache& single = state.impl_->samples[sample_ids[0]];
            if (single.len != total_len) {
                throw std::invalid_argument("self_attn_forward_cuda: cache length mismatch for batch=1.");
            }
            cache_k_ptr = single.k.data();
            cache_v_ptr = single.v.data();
            score_capacity = std::max(single.capacity_len, total_len);
            cache_leading_dim = params.num_heads_kv * params.head_dim;
            cache_instance_stride_elems = params.head_dim;
        } else {
            build_cache_batch<Traits>(*state.impl_, sample_ids, params.num_heads_kv, total_len, params.head_dim,
                                      scratch.cache_k_batch, scratch.cache_v_batch, stream);
            cache_k_ptr = scratch.cache_k_batch.data();
            cache_v_ptr = scratch.cache_v_batch.data();
        }

        ensure_device_buffer(scratch.scores, static_cast<size_t>(batch) * params.num_heads * score_capacity * Traits::kElementBytes, scratch_realloc_counter);
        ensure_device_buffer(scratch.context, static_cast<size_t>(batch) * params.num_heads * params.head_dim * Traits::kElementBytes, scratch_realloc_counter);
        ensure_device_buffer(scratch.merged, static_cast<size_t>(rows) * hidden_dim * Traits::kElementBytes, scratch_realloc_counter);
        ensure_device_buffer(scratch.output, static_cast<size_t>(rows) * o_out_dim * Traits::kElementBytes, scratch_realloc_counter);
        ensure_device_buffer(scratch.decode_active_len, sizeof(int), scratch_realloc_counter);
        cuda_check(cudaMemcpyAsync(scratch.decode_active_len.data(), &total_len, sizeof(int),
                                   cudaMemcpyHostToDevice, stream),
                   "cudaMemcpyAsync decode_active_len");
        const int* d_active_len = static_cast<const int*>(scratch.decode_active_len.data());

        const int* d_sample_ids = nullptr;
        const int* d_pad_ptr = nullptr;
        int pad_size = static_cast<int>(pad_lens_by_sample.size());
        if (pad_size > 0) {
            ensure_device_buffer(scratch.sample_ids, static_cast<size_t>(batch) * sizeof(int), scratch_realloc_counter);
            cuda_check(cudaMemcpyAsync(scratch.sample_ids.data(), sample_ids.data(), static_cast<size_t>(batch) * sizeof(int),
                                       cudaMemcpyHostToDevice, stream),
                       "cudaMemcpyAsync sample_ids");
            d_sample_ids = static_cast<const int*>(scratch.sample_ids.data());
            d_pad_ptr = get_or_upload_int_vector(
                pad_lens_by_sample,
                state.impl_->pad_lens_cache,
                stream,
                "cudaMemcpyAsync pad_lens",
                &state.impl_->stats.pad_lens_uploads);
        }

        auto launch_decode_seq1_pipeline = [&]() {
            launch_qk_grouped_batched_gemm<Traits>(handle, cache_k_ptr, scratch.q.data(), scratch.scores.data(),
                                                   batch, params.num_heads_kv, repeat_factor, 1, score_capacity,
                                                   params.head_dim, cache_leading_dim, cache_instance_stride_elems);

            const int decode_rows = batch * params.num_heads;
            const float score_scale = 1.0f / std::sqrt(static_cast<float>(params.head_dim));
            mask_scale_softmax_decode_kernel<DeviceType><<<decode_rows, kThreads, kThreads * sizeof(float), stream>>>(
                static_cast<DeviceType*>(scratch.scores.data()),
                d_sample_ids,
                d_pad_ptr,
                pad_size,
                batch, params.num_heads, score_capacity, d_active_len, score_scale);
            cuda_check(cudaGetLastError(), "mask_scale_softmax_decode_kernel");

            launch_av_grouped_batched_gemm<Traits>(handle, scratch.scores.data(), cache_v_ptr, scratch.context.data(),
                                                   batch, params.num_heads_kv, repeat_factor, 1, score_capacity,
                                                   params.head_dim, cache_leading_dim, cache_instance_stride_elems);

            const int merge_total = rows * hidden_dim;
            const int merge_blocks = (merge_total + kThreads - 1) / kThreads;
            merge_heads_kernel<DeviceType><<<merge_blocks, kThreads, 0, stream>>>(
                static_cast<const DeviceType*>(scratch.context.data()),
                static_cast<DeviceType*>(scratch.merged.data()),
                batch, 1, params.num_heads, params.head_dim);
            cuda_check(cudaGetLastError(), "merge_heads_kernel decode_seq1");

            launch_linear<Traits>(handle, o_weight_entry.device_ptr, scratch.merged.data(), scratch.output.data(), rows, hidden_dim, o_out_dim);
            maybe_add_bias<Traits>(stream, scratch.output.data(), d_o_bias, rows, o_out_dim);
        };

        bool launched = false;
        if (batch == 1) {
            DecodeGraphCache& decode_graph = state.impl_->decode_graph;
            const int sample_id = sample_ids[0];
            const bool can_reuse = can_reuse_decode_graph(
                decode_graph,
                batch,
                sample_id,
                params.num_heads,
                params.head_dim,
                hidden_dim,
                q_out_dim,
                k_out_dim,
                v_out_dim,
                o_out_dim,
                repeat_factor,
                score_capacity,
                pad_size,
                cache_k_ptr,
                cache_v_ptr,
                d_pad_ptr,
                d_norm_weight,
                q_weight_entry.device_ptr,
                k_weight_entry.device_ptr,
                v_weight_entry.device_ptr,
                o_weight_entry.device_ptr,
                d_q_bias,
                d_k_bias,
                d_v_bias,
                d_o_bias);
            if (!can_reuse) {
                decode_graph.reset();
                cudaGraph_t graph = nullptr;
                cudaGraphExec_t exec = nullptr;
                cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
                           "cudaStreamBeginCapture decode_seq1");
                launch_decode_seq1_pipeline();
                cuda_check(cudaStreamEndCapture(stream, &graph), "cudaStreamEndCapture decode_seq1");
                cuda_check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0),
                           "cudaGraphInstantiate decode_seq1");
                decode_graph.graph = graph;
                decode_graph.exec = exec;
                fill_decode_graph_signature(
                    decode_graph,
                    batch,
                    sample_id,
                    params.num_heads,
                    params.head_dim,
                    hidden_dim,
                    q_out_dim,
                    k_out_dim,
                    v_out_dim,
                    o_out_dim,
                    repeat_factor,
                    score_capacity,
                    pad_size,
                    cache_k_ptr,
                    cache_v_ptr,
                    d_pad_ptr,
                    d_norm_weight,
                    q_weight_entry.device_ptr,
                    k_weight_entry.device_ptr,
                    v_weight_entry.device_ptr,
                    o_weight_entry.device_ptr,
                    d_q_bias,
                    d_k_bias,
                    d_v_bias,
                    d_o_bias);
                state.impl_->stats.decode_graph_captures += 1;
            }
            cuda_check(cudaGraphLaunch(decode_graph.exec, stream), "cudaGraphLaunch decode_seq1");
            state.impl_->stats.decode_graph_launches += 1;
            launched = true;
        }
        if (!launched) {
            launch_decode_seq1_pipeline();
        }

        Tensor output(rows * o_out_dim);
        cuda_check(cudaMemcpyAsync(output.data().data(), scratch.output.data(),
                                   static_cast<size_t>(rows) * o_out_dim * Traits::kElementBytes,
                                   cudaMemcpyDeviceToHost, stream),
                   "cudaMemcpyAsync output decode_seq1");
        cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize self_attn_forward_cuda decode_seq1");

        output.reshape({batch, 1, o_out_dim});
        return output;
    }

    maybe_add_bias<Traits>(stream, scratch.q_proj.data(), d_q_bias, rows, q_out_dim);
    maybe_add_bias<Traits>(stream, scratch.k_proj.data(), d_k_bias, rows, k_out_dim);
    maybe_add_bias<Traits>(stream, scratch.v_proj.data(), d_v_bias, rows, v_out_dim);

    ensure_device_buffer(scratch.q, static_cast<size_t>(batch) * params.num_heads * seq_len * params.head_dim * Traits::kElementBytes, scratch_realloc_counter);
    ensure_device_buffer(scratch.k, static_cast<size_t>(batch) * params.num_heads_kv * seq_len * params.head_dim * Traits::kElementBytes, scratch_realloc_counter);
    ensure_device_buffer(scratch.v, static_cast<size_t>(batch) * params.num_heads_kv * seq_len * params.head_dim * Traits::kElementBytes, scratch_realloc_counter);

    int q_total = batch * params.num_heads * seq_len * params.head_dim;
    int kv_total = batch * params.num_heads_kv * seq_len * params.head_dim;
    int q_blocks = (q_total + kThreads - 1) / kThreads;
    int kv_blocks = (kv_total + kThreads - 1) / kThreads;
    split_transpose_kernel<DeviceType><<<q_blocks, kThreads, 0, stream>>>(
        static_cast<const DeviceType*>(scratch.q_proj.data()),
        static_cast<DeviceType*>(scratch.q.data()),
        batch, seq_len, params.num_heads, params.head_dim);
    split_transpose_kernel<DeviceType><<<kv_blocks, kThreads, 0, stream>>>(
        static_cast<const DeviceType*>(scratch.k_proj.data()),
        static_cast<DeviceType*>(scratch.k.data()),
        batch, seq_len, params.num_heads_kv, params.head_dim);
    split_transpose_kernel<DeviceType><<<kv_blocks, kThreads, 0, stream>>>(
        static_cast<const DeviceType*>(scratch.v_proj.data()),
        static_cast<DeviceType*>(scratch.v.data()),
        batch, seq_len, params.num_heads_kv, params.head_dim);
    cuda_check(cudaGetLastError(), "split_transpose_kernel");

    ensure_device_buffer(scratch.offsets, static_cast<size_t>(batch) * sizeof(int), scratch_realloc_counter);
    cuda_check(cudaMemcpyAsync(scratch.offsets.data(), offsets.data(), static_cast<size_t>(batch) * sizeof(int),
                               cudaMemcpyHostToDevice, stream),
               "cudaMemcpyAsync offsets");
    int rope_q_total = batch * params.num_heads * seq_len * (params.head_dim / 2);
    int rope_k_total = batch * params.num_heads_kv * seq_len * (params.head_dim / 2);
    int rope_q_blocks = (rope_q_total + kThreads - 1) / kThreads;
    int rope_k_blocks = (rope_k_total + kThreads - 1) / kThreads;
    rope_kernel<DeviceType><<<rope_q_blocks, kThreads, 0, stream>>>(
        static_cast<DeviceType*>(scratch.q.data()),
        static_cast<const int*>(scratch.offsets.data()),
        batch, params.num_heads, seq_len, params.head_dim, params.rope_theta);
    rope_kernel<DeviceType><<<rope_k_blocks, kThreads, 0, stream>>>(
        static_cast<DeviceType*>(scratch.k.data()),
        static_cast<const int*>(scratch.offsets.data()),
        batch, params.num_heads_kv, seq_len, params.head_dim, params.rope_theta);
    cuda_check(cudaGetLastError(), "rope_kernel");

    int total_len = append_kv_cache<Traits>(*state.impl_, sample_ids, batch, seq_len, params.num_heads_kv,
                                            params.head_dim, scratch.k, scratch.v, stream);
    if (total_len <= 0) {
        throw std::runtime_error("self_attn_forward_cuda: cache length is zero after append.");
    }

    const void* cache_k_ptr = nullptr;
    const void* cache_v_ptr = nullptr;
    int cache_leading_dim = params.head_dim;
    long long cache_instance_stride_elems = static_cast<long long>(total_len) * params.head_dim;
    if (batch == 1) {
        const SampleCache& single = state.impl_->samples[sample_ids[0]];
        if (single.len != total_len) {
            throw std::invalid_argument("self_attn_forward_cuda: cache length mismatch for batch=1.");
        }
        cache_k_ptr = single.k.data();
        cache_v_ptr = single.v.data();
        cache_leading_dim = params.num_heads_kv * params.head_dim;
        cache_instance_stride_elems = params.head_dim;
    } else {
        build_cache_batch<Traits>(*state.impl_, sample_ids, params.num_heads_kv, total_len, params.head_dim,
                                  scratch.cache_k_batch, scratch.cache_v_batch, stream);
        cache_k_ptr = scratch.cache_k_batch.data();
        cache_v_ptr = scratch.cache_v_batch.data();
    }

    ensure_device_buffer(scratch.scores, static_cast<size_t>(batch) * params.num_heads * seq_len * total_len * Traits::kElementBytes, scratch_realloc_counter);
    launch_qk_grouped_batched_gemm<Traits>(handle, cache_k_ptr, scratch.q.data(), scratch.scores.data(),
                                           batch, params.num_heads_kv, repeat_factor, seq_len, total_len, params.head_dim,
                                           cache_leading_dim, cache_instance_stride_elems);

    const int* d_sample_ids = nullptr;
    const int* d_pad_ptr = nullptr;
    int pad_size = static_cast<int>(pad_lens_by_sample.size());
    if (pad_size > 0) {
        ensure_device_buffer(scratch.sample_ids, static_cast<size_t>(batch) * sizeof(int), scratch_realloc_counter);
        cuda_check(cudaMemcpyAsync(scratch.sample_ids.data(), sample_ids.data(), static_cast<size_t>(batch) * sizeof(int),
                                   cudaMemcpyHostToDevice, stream),
                   "cudaMemcpyAsync sample_ids");
        d_sample_ids = static_cast<const int*>(scratch.sample_ids.data());
        d_pad_ptr = get_or_upload_int_vector(
            pad_lens_by_sample,
            state.impl_->pad_lens_cache,
            stream,
            "cudaMemcpyAsync pad_lens",
            &state.impl_->stats.pad_lens_uploads);
    }

    int row_count = batch * params.num_heads * seq_len;
    float score_scale = 1.0f / std::sqrt(static_cast<float>(params.head_dim));
    mask_scale_softmax_kernel<DeviceType><<<row_count, kThreads, kThreads * sizeof(float), stream>>>(
        static_cast<DeviceType*>(scratch.scores.data()),
        d_sample_ids,
        d_pad_ptr,
        pad_size,
        batch, params.num_heads, seq_len, total_len, score_scale);
    cuda_check(cudaGetLastError(), "mask_scale_softmax_kernel");

    ensure_device_buffer(scratch.context, static_cast<size_t>(batch) * params.num_heads * seq_len * params.head_dim * Traits::kElementBytes, scratch_realloc_counter);
    launch_av_grouped_batched_gemm<Traits>(handle, scratch.scores.data(), cache_v_ptr, scratch.context.data(),
                                           batch, params.num_heads_kv, repeat_factor, seq_len, total_len, params.head_dim,
                                           cache_leading_dim, cache_instance_stride_elems);

    ensure_device_buffer(scratch.merged, static_cast<size_t>(rows) * hidden_dim * Traits::kElementBytes, scratch_realloc_counter);
    int merge_total = rows * hidden_dim;
    int merge_blocks = (merge_total + kThreads - 1) / kThreads;
    merge_heads_kernel<DeviceType><<<merge_blocks, kThreads, 0, stream>>>(
        static_cast<const DeviceType*>(scratch.context.data()),
        static_cast<DeviceType*>(scratch.merged.data()),
        batch, seq_len, params.num_heads, params.head_dim);
    cuda_check(cudaGetLastError(), "merge_heads_kernel");

    ensure_device_buffer(scratch.output, static_cast<size_t>(rows) * o_out_dim * Traits::kElementBytes, scratch_realloc_counter);
    launch_linear<Traits>(handle, o_weight_entry.device_ptr, scratch.merged.data(), scratch.output.data(), rows, hidden_dim, o_out_dim);
    maybe_add_bias<Traits>(stream, scratch.output.data(), d_o_bias, rows, o_out_dim);

    Tensor output(rows * o_out_dim);
    cuda_check(cudaMemcpyAsync(output.data().data(), scratch.output.data(),
                               static_cast<size_t>(rows) * o_out_dim * Traits::kElementBytes,
                               cudaMemcpyDeviceToHost, stream),
               "cudaMemcpyAsync output");
    cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize self_attn_forward_cuda");

    output.reshape({batch, seq_len, o_out_dim});
    return output;
}

} // namespace ops
} // namespace cuda
} // namespace easy_llm
