#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="build"
JOBS="${JOBS:-8}"
WITH_CUDA=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        --jobs)
            JOBS="$2"
            shift 2
            ;;
        --with-cuda)
            WITH_CUDA=1
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: $0 [--build-dir <dir>] [--jobs <n>] [--with-cuda]" >&2
            exit 1
            ;;
    esac
done

CMAKE_ARGS=(
    -S .
    -B "$BUILD_DIR"
    -DCMAKE_BUILD_TYPE=RelWithDebInfo
    -DCMAKE_CXX_COMPILER="${CMAKE_CXX_COMPILER:-g++}"
)
if [[ "$WITH_CUDA" -eq 1 ]]; then
    CMAKE_ARGS+=(-DEASY_LLM_ENABLE_CUDA=ON)
fi

cmake "${CMAKE_ARGS[@]}"
cmake --build "$BUILD_DIR" --target easy_llm_regression_gates -j"$JOBS"

if [[ "$WITH_CUDA" -eq 1 ]]; then
    cmake --build "$BUILD_DIR" \
        --target easy_llm_self_attn_cuda_test easy_llm_self_attn_cuda_varlen_test \
        -j"$JOBS"
    ctest --test-dir "$BUILD_DIR" --output-on-failure -L "^invariant_gate_cuda$"
fi
