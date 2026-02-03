#include "cuda/runtime.hpp"

#include <sstream>
#include <stdexcept>

#include "cuda/precision.hpp"
#include "runtime_internal.hpp"
#include "spdlog/spdlog.h"

namespace easy_llm {
namespace cuda {
namespace {

const char* cublas_status_to_string(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
        default: return "CUBLAS_STATUS_UNKNOWN";
    }
}

} // namespace

void cuda_check(cudaError_t status, const char* msg) {
    if (status == cudaSuccess) {
        return;
    }
    std::ostringstream oss;
    oss << msg << ": " << cudaGetErrorString(status);
    throw std::runtime_error(oss.str());
}

void cublas_check(cublasStatus_t status, const char* msg) {
    if (status == CUBLAS_STATUS_SUCCESS) {
        return;
    }
    std::ostringstream oss;
    oss << msg << ": " << cublas_status_to_string(status);
    throw std::runtime_error(oss.str());
}

WeightEntry& WeightCache::get_or_upload(const Tensor& weights, cudaStream_t stream) {
    using Traits = CudaPrecisionTraits<data_type>;
    const size_t bytes = static_cast<size_t>(weights.size()) * Traits::kElementBytes;
    WeightKey key{weights.data().data(), bytes, weights.shape()[0], weights.shape()[1]};
    auto it = entries_.find(key);
    if (it != entries_.end()) {
        return it->second;
    }

    void* device_ptr = nullptr;
    cuda_check(cudaMalloc(&device_ptr, bytes), "cudaMalloc weights");
    cuda_check(cudaMemcpyAsync(device_ptr, weights.data().data(), bytes,
                               cudaMemcpyHostToDevice, stream),
               "cudaMemcpyAsync weights");

    WeightEntry entry;
    entry.device_ptr = device_ptr;
    entry.bytes = bytes;
    entry.height = key.height;
    entry.width = key.width;
    auto inserted = entries_.emplace(key, entry);
    return inserted.first->second;
}

void WeightCache::clear() {
    for (auto& kv : entries_) {
        if (kv.second.device_ptr) {
            cudaFree(kv.second.device_ptr);
            kv.second.device_ptr = nullptr;
        }
    }
    entries_.clear();
}

WeightCache::~WeightCache() {
    clear();
}

bool CudaContext::available() {
    std::call_once(init_once_, [this]() { init(); });
    return available_;
}

void CudaContext::clear_cache() {
    cache_.clear();
}

void CudaContext::disable(const char* reason) {
    spdlog::error("CUDA disabled: {}", reason);
    available_ = false;
}

CudaContext::~CudaContext() {
    cache_.clear();
    if (handle_) {
        cublasDestroy(handle_);
        handle_ = nullptr;
    }
    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
}

void CudaContext::init() {
    try {
        using Traits = CudaPrecisionTraits<data_type>;
        int device_count = 0;
        auto device_status = cudaGetDeviceCount(&device_count);
        if (device_status != cudaSuccess || device_count == 0) {
            spdlog::warn("CUDA unavailable: no device detected");
            available_ = false;
            return;
        }

        int device = 0;
        if (cudaGetDevice(&device) != cudaSuccess) {
            device = 0;
        }

        if (!Traits::device_supported(device)) {
            spdlog::error("CUDA device does not support {}", Traits::kName);
            available_ = false;
            return;
        }

        cuda_check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
                   "cudaStreamCreateWithFlags");
        cublas_check(cublasCreate(&handle_), "cublasCreate");
        cublas_check(cublasSetStream(handle_, stream_), "cublasSetStream");
        cublasSetMathMode(handle_, CUBLAS_TENSOR_OP_MATH);

        available_ = true;
        spdlog::info("CUDA runtime enabled (precision: {})", Traits::kName);
    } catch (const std::exception& e) {
        spdlog::error("CUDA init failed: {}", e.what());
        available_ = false;
    }
}

bool initialize() {
    return get_context().available();
}

CudaContext& get_context() {
    static CudaContext context;
    return context;
}

bool available() {
    return get_context().available();
}

void clear_weight_cache() {
    auto& ctx = get_context();
    if (!ctx.available()) {
        return;
    }
    std::lock_guard<std::mutex> lock(ctx.mutex());
    ctx.clear_cache();
}

} // namespace cuda
} // namespace easy_llm
