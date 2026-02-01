#ifndef EASY_GPT_CUDA_DEVICE_BUFFER_HPP
#define EASY_GPT_CUDA_DEVICE_BUFFER_HPP

#include <cstddef>

#include <cuda_runtime.h>

namespace easy_gpt {
namespace cuda {

void cuda_check(cudaError_t status, const char* msg);

class DeviceBuffer {
public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(size_t size_bytes) {
        allocate(size_bytes);
    }

    ~DeviceBuffer() {
        release();
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept { move_from(other); }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            release();
            move_from(other);
        }
        return *this;
    }

    void* data() { return ptr_; }
    const void* data() const { return ptr_; }
    size_t bytes() const { return bytes_; }
    bool empty() const { return ptr_ == nullptr; }

    void reset(size_t size_bytes) {
        release();
        allocate(size_bytes);
    }

private:
    void allocate(size_t size_bytes) {
        if (size_bytes == 0) {
            ptr_ = nullptr;
            bytes_ = 0;
            return;
        }
        void* new_ptr = nullptr;
        cuda_check(cudaMalloc(&new_ptr, size_bytes), "cudaMalloc");
        ptr_ = new_ptr;
        bytes_ = size_bytes;
    }

    void release() {
        if (ptr_) {
            cudaFree(ptr_);
            ptr_ = nullptr;
        }
        bytes_ = 0;
    }

    void move_from(DeviceBuffer& other) {
        ptr_ = other.ptr_;
        bytes_ = other.bytes_;
        other.ptr_ = nullptr;
        other.bytes_ = 0;
    }

    void* ptr_ = nullptr;
    size_t bytes_ = 0;
};

} // namespace cuda
} // namespace easy_gpt

#endif // EASY_GPT_CUDA_DEVICE_BUFFER_HPP
