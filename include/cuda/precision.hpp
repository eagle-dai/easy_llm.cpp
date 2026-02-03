#ifndef EASY_LLM_CUDA_PRECISION_HPP
#define EASY_LLM_CUDA_PRECISION_HPP

#include <cstddef>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "precision.hpp"
#include "tensor.hpp"

namespace easy_llm {
namespace cuda {

template <typename T>
struct CudaPrecisionTraits {
    static_assert(sizeof(T) == 0,
                  "CUDA precision traits not defined for this data_type.");
};

template <>
struct CudaPrecisionTraits<float> {
    using DeviceType = float;
    static constexpr size_t kElementBytes = sizeof(DeviceType);
    static constexpr cudaDataType_t kCudaType = CUDA_R_32F;
    static constexpr cublasComputeType_t kComputeType = CUBLAS_COMPUTE_32F;
    static constexpr const char* kName = "fp32";

    static bool device_supported(int /*device*/) {
        return true;
    }
};

template <>
struct CudaPrecisionTraits<Bf16> {
    using DeviceType = __nv_bfloat16;
    static constexpr size_t kElementBytes = sizeof(DeviceType);
    static constexpr cudaDataType_t kCudaType = CUDA_R_16BF;
    static constexpr cublasComputeType_t kComputeType = CUBLAS_COMPUTE_32F;
    static constexpr const char* kName = "bf16";

    static_assert(sizeof(Bf16) == sizeof(__nv_bfloat16), "BF16 size mismatch");

    static bool device_supported(int device) {
#if defined(cudaDevAttrCudaBf16Supported)
        int bf16_supported = 0;
        if (cudaDeviceGetAttribute(&bf16_supported, cudaDevAttrCudaBf16Supported, device) == cudaSuccess) {
            return bf16_supported != 0;
        }
#endif
        return true;
    }
};

} // namespace cuda
} // namespace easy_llm

#endif // EASY_LLM_CUDA_PRECISION_HPP
