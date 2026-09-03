[English](README.md) | 简体中文

# easy_llm

一个用于学习与理解大模型推理流程的微型 C++ 框架。项目保持“可读、可学、可改”，覆盖配置加载、权重加载、Tokenizer、Prefill/Decode、采样与解码等完整链路。默认适配 **Qwen2.5-0.5B**，可在单机快速跑通端到端推理。

项目重点是**正确性与架构清晰度**，不是极致吞吐。第三方依赖保持极简（`spdlog` 与 `nlohmann/json`，均已 vendored），其余核心逻辑由 C++ 实现，适合课程、原型和个人学习。

## 推荐学习路径

如果第一次接触 LLM inference，不建议直接从目录或 Attention/CUDA 源码开始。

1. 先读 [**第一次阅读：一条 Prompt 怎样生成下一个 Token**](my/docs/easy_llm_first_read.zh-CN.md)，只建立 `text → token IDs → Tensor → logits → next token → KV Cache → Decode` 的最小心智模型。
2. 再读 [**教学式源码教程**](my/docs/easy_llm_guide.zh-CN.md)，沿真实 Golden Path 把 Prefill/Decode、KV Cache、stable identity、padding/position、Continuous Batching 和 CPU/CUDA state boundary 映射到关键代码。
3. 需要继续深入时，再读 [**深入实现版（Deep Dive）**](my/docs/easy_llm_deep_dive.zh-CN.md)，深入 Tokenizer/BPE、GQA、RoPE、三类 Attention mask、Sampling 精确语义、loader、可靠性、Troubleshooting 与扩展边界。
4. 最后按教程末尾的源码阅读顺序进入具体实现和 tests。

这样可以先理解“LLM 到底怎样逐 token 工作”，再把核心机制钉到源码上，最后深入数学、实现细节和工程边界。

## 当前状态（截至 2026-02-23）
- `release` 已支持 **连续批处理服务模式**（`--serve`），同时保留单次 CLI 推理。
- 新增并扩展了 **回归/不变量测试**，可通过 CTest label 和 `easy_llm_regression_gates` 目标执行。
- 模型加载链路补强了 **架构分发 + 参数 key/shape 校验**（`LayerKeyPrefix` 与参数校验模块）。
- CUDA 路径持续扩展（matmul/MLP/self-attn），整体仍保持 **CPU-first** 基线。

---

## 快速开始

### 依赖
- C++17 编译器
- CMake（>= 3.10）
- 可选：OpenMP（默认 `EASY_LLM_ENABLE_OPENMP=ON`）
- 可选：CUDA Toolkit（仅 CUDA 构建需要）

### 构建（推荐命令）
CPU 构建：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_COMPILER=g++
cmake --build build --target easy_llm -j8
```

CUDA 构建（需本机 CUDA 环境）：

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DEASY_LLM_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=<your_arch> \
  -DCMAKE_CXX_COMPILER=g++
cmake --build build --target easy_llm -j8
```

`build.sh` 保留用于本地实验，可能带有机器相关参数；通用场景建议使用上面的 CMake 命令。

### 准备模型文件
默认适配 [**Qwen2.5-0.5B**](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)。将文件放到（`data/` 已 gitignore）：

```
data/model/
├─ config.json
├─ model.safetensors
├─ tokenizer.json
└─ tokenizer_config.json
```

如需自定义路径，可修改 `include/config.hpp` 默认配置。

### 运行示例
```bash
./build/easy_llm --help
./build/easy_llm --max-steps 128 --temperature 0.7 --top-p 0.9 --top-k 40 "Hello"
./build/easy_llm -f test/data/test_batch.txt --max-steps 256 --temperature 0.1
./build/easy_llm --serve
```

主要参数：
- `-f/--prompt-file`：从文件读取多条 prompt
- `-m/--max-steps`：每条请求的最大生成步数
- `--temperature` / `--top-p` / `--top-k`：采样控制
- `--seed`：采样随机种子
- `--greedy`：贪心解码
- `--serve`：启动连续批处理服务
- `--serve-max-active`：服务模式下最大活跃请求数
- `--serve-prefill-batch`：每轮 prefill 最多接纳请求数
- `--serve-idle-ms`：服务空闲轮询间隔
- `--serve-stats-ms`：周期统计日志间隔（0 表示关闭周期日志）

### 连续批处理服务模式
运行 `./build/easy_llm --serve` 后：
- 从标准输入逐行提交 prompt；
- 输入 `/quit`（或 `:quit`）停止接收新请求并等待已接入请求完成；
- 接纳成功会输出 `[accepted <id>]`，完成时输出 `[request <id>] <decoded_text>`。

---

## 项目结构与推理流程

### 代码结构
```
include/                           # 公共头文件
include/models/                    # GPT 组件接口
include/continuous_batch_server.hpp
src/                               # 核心实现
src/models/                        # 模型组件实现
src/continuous_batch_server.cpp
src/cuda/                          # CUDA runtime 与 CUDA 算子
test/                              # 单测/不变量测试与测试数据
scripts/run_regression_gates.sh
data/                              # 模型资源（git 忽略）
```

### 推理流程（入口视角）
`src/main.cpp` 统一编排两种运行模式：
1) 解析 CLI 参数（prompt/采样/服务参数）
2) 加载配置、权重、Tokenizer
3) 对输入应用 chat template
4) 单次模式：`GptEngine::run` + `DataManager`
5) 服务模式：`ContinuousBatchServer::run`，循环执行 prefill/decode

### GptModel 核心逻辑（两种模式共用）
- `DataManager` 做分词与**左侧 padding**，记录 `seq_len` / `pad_len`；
- `GptModel::forward`（或连续模式采样接口）按 **prefill -> decode** 阶段运行并维护层级 KV cache；
- Prefill 整段前向一次，Decode 每步只输入 1 token 并复用/追加 KV；
- EOS 过滤与活跃样本维护会同步更新 sample id、输出 token 与位置长度。

---

## 核心特性

- 端到端推理链路完整，便于沿调用路径学习
- GPT 组件拆分清晰：Embedding -> Blocks(Self-Attn + MLP) x N -> Norm -> 输出投影
- Greedy / Top-K / Top-P 采样可直接通过 CLI 控制
- 连续批处理服务模式支持多请求调度解码
- 关键路径具备回归/不变量测试门禁
- CPU-first 基线 + 可选 CUDA 加速路径

---

## 配置与扩展点

- **模型与 tokenizer 路径**：`include/config.hpp`（默认 `data/model/`）
- **精度宏**：默认 `USE_BF16`，可在编译时调整
- **OpenMP**：由 `EASY_LLM_ENABLE_OPENMP` 控制
- **模型适配分发**：`create_layer_key_prefix` 基于 `architecture/model_type` 分发（当前 Qwen2 family）
- **参数校验**：模型加载前执行 key/shape 校验，减少静默错配风险

---

## 测试与复现

构建并运行核心回归门：

```bash
cmake --build build --target easy_llm_regression_gates -j8
ctest --test-dir build --output-on-failure -L "^invariant_gate$"
```

CUDA 不变量测试（仅 CUDA 构建时）：

```bash
ctest --test-dir build --output-on-failure -L "^invariant_gate_cuda$"
```

辅助脚本：

```bash
bash scripts/run_regression_gates.sh
bash scripts/run_regression_gates.sh --with-cuda
```

---

## 常见问题

**Q: 使用 `-f/--prompt-file` 后输出保存在哪里？**  
A: 会在输入文件同目录生成 `*_output*` 文件，并保留原扩展名（例如 `test_batch.txt` -> `test_batch_output.txt`）。

**Q: 为什么没有做重度性能优化？**  
A: 项目定位是学习型实现，优先保证推理链路可读、可验证、易于调试。

---

## 依赖说明

- `spdlog`：日志（仓库内置，`include/third_party/spdlog` 与 `src/third_party/spdlog`）
- `nlohmann/json`：JSON 解析（仓库内置，`include/third_party/json.hpp`）

其余均为 C++ 原生实现。
