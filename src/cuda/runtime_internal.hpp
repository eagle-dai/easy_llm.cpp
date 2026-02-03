#ifndef EASY_LLM_CUDA_RUNTIME_INTERNAL_HPP
#define EASY_LLM_CUDA_RUNTIME_INTERNAL_HPP

#include <cstddef>
#include <functional>
#include <mutex>
#include <unordered_map>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "tensor.hpp"

namespace easy_llm {
namespace cuda {

void cuda_check(cudaError_t status, const char* msg);
void cublas_check(cublasStatus_t status, const char* msg);

struct WeightKey {
    const void* ptr;
    size_t bytes;
    int height;
    int width;

    bool operator==(const WeightKey& other) const {
        return ptr == other.ptr && bytes == other.bytes &&
               height == other.height && width == other.width;
    }
};

struct WeightKeyHash {
    size_t operator()(const WeightKey& key) const {
        size_t h = std::hash<const void*>()(key.ptr);
        h ^= std::hash<size_t>()(key.bytes + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2));
        h ^= std::hash<int>()(key.height + 0x9e3779b9 + (h << 6) + (h >> 2));
        h ^= std::hash<int>()(key.width + 0x9e3779b9 + (h << 6) + (h >> 2));
        return h;
    }
};

struct WeightEntry {
    void* device_ptr = nullptr;
    size_t bytes = 0;
    int height = 0;
    int width = 0;
};

class WeightCache {
public:
    WeightEntry& get_or_upload(const Tensor& weights, cudaStream_t stream);
    void clear();
    ~WeightCache();

private:
    std::unordered_map<WeightKey, WeightEntry, WeightKeyHash> entries_;
};

class CudaContext {
public:
    bool available();
    cudaStream_t stream() const { return stream_; }
    cublasHandle_t handle() const { return handle_; }
    std::mutex& mutex() { return mutex_; }
    WeightCache& cache() { return cache_; }
    void clear_cache();
    void disable(const char* reason);
    ~CudaContext();

private:
    void init();

    std::once_flag init_once_;
    bool available_ = false;
    cudaStream_t stream_ = nullptr;
    cublasHandle_t handle_ = nullptr;
    std::mutex mutex_;
    WeightCache cache_;
};

CudaContext& get_context();

} // namespace cuda
} // namespace easy_llm

#endif // EASY_LLM_CUDA_RUNTIME_INTERNAL_HPP
