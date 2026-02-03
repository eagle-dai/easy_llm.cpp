English | [简体中文](README.zh-CN.md)

# easy_llm

A minimal C++ framework for learning and understanding the LLM inference pipeline. The project aims to be readable, easy to learn, and easy to modify, while preserving the key steps of a full inference flow (config loading, weight loading, tokenizer, prefill/decode, sampling and decoding). The default setup targets **Qwen2.5-0.5B**, so you can quickly run end-to-end inference on a single machine.

This project focuses on **architecture and the inference process**, not peak performance: operators are implemented with correctness and readability as the priority. Third-party dependencies are kept minimal (`spdlog` and `nlohmann/json`, both vendored in the repo); everything else is implemented in C++, making it a good baseline for coursework, research prototypes, or personal learning.

## Updates
- `dev/cuda` (CUDA operator development) has been merged into `release` and is in an early, just-started stage.
- `release` remains CPU-first by default: `build.sh` does **not** enable CUDA. To build with CUDA, pass `-DEASY_LLM_ENABLE_CUDA=ON` (and CUDA arch) in CMake.

---

## Quick Start

### Dependencies & Build
- A C++17 compiler
- CMake (≥ 3.10)
- Optional: OpenMP (the project can still build and run without it; enabling it can speed up some operators)

```bash
bash build.sh
```

### Prepare Model Files 
The default target is [**Qwen2.5-0.5B**](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct). Put the required files under the following path (`data/` is git-ignored):

```
data/model/
├─ config.json
├─ model.safetensors
├─ tokenizer.json
└─ tokenizer_config.json
```

To customize paths, edit the default configuration in `include/config.hpp`.

### Run Examples
```bash
./build/easy_llm --help
./build/easy_llm --max-steps 128 --temperature 0.7 --top-p 0.9 --top-k 40 "Hello"
./build/easy_llm --max-steps 256 --temperature 0.1 -f test/data/test_batch.txt
```

Key arguments:
- `-f/--prompt-file`: read multiple prompts from a file
- `-m/--max-steps`: generation length limit
- `--temperature`/`--top-p`/`--top-k`: sampling controls
- `--greedy`: greedy decoding

---

## Project Structure & Inference Flow

### Code Layout
```
include/             # Public headers (core interfaces and data structures)
include/models/      # GPT component definitions (Embedding/Attention/MLP/Block, etc.)
include/third_party/ # Lightweight vendored headers (json.hpp, etc.)
src/                 # Core implementations
src/models/          # Model component implementations
src/third_party/     # Vendored implementations (spdlog)
test/                # Test scripts and data
data/                # Model weights and tokenizer assets (git-ignored)
```

### Inference Flow (from the entry point)
`src/main.cpp` orchestrates the full pipeline:
1) Parse CLI arguments (prompt, sampling parameters, random seed, etc.)
2) Read the prompt(s) or a prompt file and apply the chat template
3) Load model config and weights (`config.json` + `model.safetensors`)
4) Initialize `Tokenizer`, `DataManager`, and `GptModel`
5) Build `GptEngine` and run `run` to generate outputs

### GptModel Inference Logic (core path)
- `DataManager` tokenizes the input and applies **left padding**, tracking each sample's `seq_len` and `pad_len`.
- `GptModel::forward` creates a `GenerationContext`, initializes per-layer KV caches, and enters a **prefill → decode** two-stage process.
- **Prefill**: run a full forward pass over the entire prompt in parallel: `Embedding → Block(Self-Attn+MLP)×N → RMSNorm → output projection`, then sample the first generated token from the logits at the last position.
- **Decode**: iteratively feed only the token generated in the previous step, append attention context via the KV cache, sample the next token, update the position index, and check EOS. On EOS, the sample's KV cache is cleared and the sample is removed from the active batch.
- Generated tokens are recorded by `DataManager` and finally decoded into text.

---

## Key Features (learning-oriented)

- **Complete inference pipeline**: from config loading, weight parsing, tokenizer, prefill/decode, to sampling and output.
- **Clear GPT component decomposition**: Embedding → Transformer Blocks (Self-Attn + MLP) × N → Norm → output projection, easy to map to papers and common implementations.
- **Configurable sampling**: built-in Greedy / Top-K / Top-P sampling, controlled via CLI flags.
- **KV cache and staged inference**: separate Prefill/Decode with cache reuse during Decode, reflecting real inference frameworks.
- **Minimal dependencies, pure C++ core**: aside from logging and JSON parsing, the core logic is implemented in C++.

---

## Configuration & Extension Points

- **Model/tokenizer paths**: in `include/config.hpp`, defaulting to files under `data/model/`.
- **Precision**: default build uses BF16 (`USE_BF16`). Switch to FP16/FP32 by adjusting compile-time macro definitions.
- **OpenMP**: enabled by default (`EASY_LLM_ENABLE_OPENMP=ON`); you can disable it via CMake options or install an OpenMP runtime.
- **Model adaptation**: currently adapted to Qwen2.5-0.5B (weight key naming and config). To switch models, ensure `config.json`, weight keys, and tokenizer assets are compatible.

---

## Tests & Reproducibility

- Test data lives under `test/`.

---

## FAQ

**Q: Where is the output saved when using `-f/--prompt-file`?**  
A: An output file named `*_output*` is created in the same directory as the input file, preserving the extension (e.g., `test_batch.txt` → `test_batch_output.txt`).

**Q: Why isn't this heavily optimized?**  
A: This project is a teaching/learning implementation focused on readability and a clear structure.

---

## Dependencies

- `spdlog`: logging (vendored in `include/third_party/spdlog` and `src/third_party/spdlog`)
- `nlohmann/json`: JSON parsing (vendored as `include/third_party/json.hpp`)

Everything else is implemented in C++.
