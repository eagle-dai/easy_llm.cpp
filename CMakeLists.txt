cmake_minimum_required(VERSION 3.10)

project(easy_llm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(EASY_LLM_ENABLE_CUDA "Enable CUDA backend" OFF)
if(EASY_LLM_ENABLE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    find_package(CUDAToolkit REQUIRED)
endif()

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build." FORCE)
endif()

message(STATUS "Configuring easy_llm (${CMAKE_BUILD_TYPE})")

option(EASY_LLM_ENABLE_OPENMP "Enable OpenMP parallelism" ON)
if(EASY_LLM_ENABLE_OPENMP)
    find_package(OpenMP QUIET COMPONENTS CXX)
endif()

set(EASY_LLM_WARNING_FLAGS -Wall -Wextra -Wpedantic)

file(GLOB EASY_LLM_CORE_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/models/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/third_party/spdlog/*.cpp"
)
list(REMOVE_ITEM EASY_LLM_CORE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")
if(EASY_LLM_ENABLE_CUDA)
    list(APPEND EASY_LLM_CORE_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/runtime.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/ops/matmul.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/ops/self_attn.cu"
    )
endif()

add_library(easy_llm_core STATIC ${EASY_LLM_CORE_SOURCES})
target_compile_features(easy_llm_core PUBLIC cxx_std_17)
target_compile_options(easy_llm_core PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${EASY_LLM_WARNING_FLAGS}>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:-O0 -fno-inline -fno-omit-frame-pointer -ggdb3 -fno-elide-constructors>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:-O3 -march=native>
)


target_include_directories(easy_llm_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/third_party>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/models>
)

if(EASY_LLM_ENABLE_OPENMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(easy_llm_core PUBLIC OpenMP::OpenMP_CXX)
        target_compile_definitions(easy_llm_core PUBLIC USE_OPENMP)
    else()
        message(WARNING
            "OpenMP CXX not found. Build will continue without OpenMP. "
            "To enable it, install the OpenMP runtime for your compiler "
            "(e.g., libomp-dev for clang, libgomp for GCC), or set OpenMP_ROOT. "
            "You can also disable via -DEASY_LLM_ENABLE_OPENMP=OFF."
        )
    endif()
endif()

target_compile_definitions(easy_llm_core PUBLIC SPDLOG_COMPILED_LIB USE_BF16)
if(EASY_LLM_ENABLE_CUDA)
    target_compile_definitions(easy_llm_core PUBLIC USE_CUDA)
    target_compile_features(easy_llm_core PUBLIC cuda_std_17)
    target_link_libraries(easy_llm_core PUBLIC CUDA::cudart CUDA::cublas)
    if(TARGET CUDA::cublasLt)
        target_link_libraries(easy_llm_core PUBLIC CUDA::cublasLt)
    endif()
endif()

add_executable(easy_llm src/main.cpp)
target_link_libraries(easy_llm PRIVATE easy_llm_core)

enable_testing()
add_executable(easy_llm_matmul_test test/matmul_3d_test.cpp)
target_link_libraries(easy_llm_matmul_test PRIVATE easy_llm_core)
add_test(NAME easy_llm_matmul_test COMMAND easy_llm_matmul_test)

if(EASY_LLM_ENABLE_CUDA)
    add_executable(easy_llm_self_attn_cuda_test test/self_attn_cuda_test.cpp)
    target_link_libraries(easy_llm_self_attn_cuda_test PRIVATE easy_llm_core)
    add_test(NAME easy_llm_self_attn_cuda_test COMMAND easy_llm_self_attn_cuda_test)
endif()
