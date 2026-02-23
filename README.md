English | [简体中文](README.zh-CN.md)

# easy_llm

A minimal C++ framework for learning and understanding the LLM inference pipeline. The project keeps the full inference path readable and editable: config loading, weight loading, tokenizer, prefill/decode, sampling, and decoding. The default setup targets **Qwen2.5-0.5B**, so you can run end-to-end inference quickly on a single machine.

This project focuses on **correctness and architecture clarity**, not peak serving performance. Third-party dependencies are intentionally minimal (`spdlog` and `nlohmann/json`, both vendored); the rest is implemented in C++ and kept friendly for coursework, research prototypes, and self-study.

## Status (as of 2026-02-23)
- `release` now includes a **continuous batching service mode** (`--serve`) in addition to one-shot CLI inference.
- **Regression/invariant tests** were expanded, with CTest labels and a dedicated target `easy_llm_regression_gates`.
- Model loading now has stronger **architecture/key/shape validation** (including `LayerKeyPrefix` dispatch and parameter checks).
- CUDA path has grown (matmul/MLP/self-attn kernels), while the project remains **CPU-first by default**.

---

## Quick Start

### Dependencies
- A C++17 compiler
- CMake (>= 3.10)
- Optional: OpenMP (`EASY_LLM_ENABLE_OPENMP=ON` by default)
- Optional: CUDA toolkit (only when building CUDA path)

### Build (recommended commands)
CPU build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_COMPILER=g++
cmake --build build --target easy_llm -j8
```

CUDA build (requires local CUDA env):

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DEASY_LLM_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=<your_arch> \
  -DCMAKE_CXX_COMPILER=g++
cmake --build build --target easy_llm -j8
```

`build.sh` is kept for local experiments and may contain machine-specific flags. Prefer the CMake commands above for portable usage.

### Prepare Model Files
The default target is [**Qwen2.5-0.5B**](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct). Put required files under (`data/` is git-ignored):

```
data/model/
├─ config.json
├─ model.safetensors
├─ tokenizer.json
└─ tokenizer_config.json
```

To customize paths, edit defaults in `include/config.hpp`.

### Run Examples
```bash
./build/easy_llm --help
./build/easy_llm --max-steps 128 --temperature 0.7 --top-p 0.9 --top-k 40 "Hello"
./build/easy_llm -f test/data/test_batch.txt --max-steps 256 --temperature 0.1
./build/easy_llm --serve
```

Key arguments:
- `-f/--prompt-file`: read multiple prompts from a file
- `-m/--max-steps`: generation length limit per request
- `--temperature` / `--top-p` / `--top-k`: sampling controls
- `--seed`: RNG seed
- `--greedy`: greedy decoding
- `--serve`: run long-lived continuous batching service
- `--serve-max-active`: max concurrent active requests in service mode
- `--serve-prefill-batch`: max requests admitted per prefill round
- `--serve-idle-ms`: idle sleep interval for the service loop
- `--serve-stats-ms`: periodic stats log interval (0 disables periodic logs)

### Continuous Batching Service Mode
When running `./build/easy_llm --serve`:
- Submit one prompt per input line in stdin.
- Type `/quit` (or `:quit`) to stop accepting new input and drain existing requests.
- Accepted requests print `[accepted <id>]`; completed requests print `[request <id>] <decoded_text>`.

---

## Project Structure & Inference Flow

### Code Layout
```
include/                   # Public headers
include/models/            # GPT component interfaces
include/continuous_batch_server.hpp
src/                       # Core implementations
src/models/                # Model component implementations
src/continuous_batch_server.cpp
src/cuda/                  # CUDA runtime and CUDA operators
test/                      # Unit/invariant tests and test data
scripts/run_regression_gates.sh
data/                      # Model assets (git-ignored)
```

### Inference Flow (entry point)
`src/main.cpp` orchestrates two runtime modes:
1) Parse CLI arguments (prompt/sampling/service options)
2) Load config + model weights + tokenizer
3) Apply chat template to user prompts
4) One-shot mode: `GptEngine::run` with `DataManager`
5) Service mode: `ContinuousBatchServer::run` with prefill/decode rounds over active requests

### GptModel Core Logic (shared by both modes)
- `DataManager` tokenizes and applies **left padding**, tracking `seq_len` and `pad_len`.
- `GptModel::forward` (or continuous sampling APIs) uses **prefill -> decode** staging with per-layer KV caches.
- Prefill runs full prompt forward once; decode feeds one token per step with KV cache append/reuse.
- EOS filtering and active-sample bookkeeping keep sample IDs, generated tokens, and position lengths aligned.

---

## Key Features

- End-to-end inference path from config/weights/tokenizer to sampled output
- Clear GPT decomposition: Embedding -> Blocks(Self-Attn + MLP) x N -> Norm -> output projection
- Greedy / Top-K / Top-P sampling with explicit CLI controls
- Continuous batching service mode for multi-request decode scheduling
- Regression/invariant test gates for critical generation/cache/data-manager behaviors
- CPU-first baseline with optional CUDA acceleration path

---

## Configuration & Extension Points

- **Model/tokenizer paths**: `include/config.hpp` (defaults under `data/model/`)
- **Precision macro**: default `USE_BF16`; can be adjusted at compile time
- **OpenMP**: controlled by `EASY_LLM_ENABLE_OPENMP`
- **Model adaptation**: `create_layer_key_prefix` dispatches by `architecture/model_type` (currently Qwen2 family)
- **Model param validation**: pre-load key/shape checks reduce silent mismatch risk

---

## Tests & Reproducibility

Build and run regression gates:

```bash
cmake --build build --target easy_llm_regression_gates -j8
ctest --test-dir build --output-on-failure -L "^invariant_gate$"
```

CUDA-specific invariant tests (when CUDA build is enabled):

```bash
ctest --test-dir build --output-on-failure -L "^invariant_gate_cuda$"
```

Helper script:

```bash
bash scripts/run_regression_gates.sh
bash scripts/run_regression_gates.sh --with-cuda
```

---

## FAQ

**Q: Where is the output saved when using `-f/--prompt-file`?**  
A: Output is written to a sibling file named `*_output*` with the same extension (for example, `test_batch.txt` -> `test_batch_output.txt`).

**Q: Why isn't this heavily optimized?**  
A: It is a learning-oriented implementation prioritizing readable inference logic and debuggability.

---

## Dependencies

- `spdlog`: logging (vendored in `include/third_party/spdlog` and `src/third_party/spdlog`)
- `nlohmann/json`: JSON parsing (vendored in `include/third_party/json.hpp`)

Everything else is implemented in C++.
