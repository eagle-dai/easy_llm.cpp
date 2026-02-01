[English](README.md) | 简体中文

# easy_gpt

一个用于学习与理解大模型推理流程的微型 C++ 框架。项目以“可读、可学、可改”为目标，尽量保留推理全链路的关键步骤（配置加载、权重加载、Tokenizer、Prefill/Decode、采样与解码），便于读者按模块逐层拆解与理解。当前默认适配 **Qwen2.5-0.5B**，可在单机环境中快速跑通端到端推理流程。

本项目重点关注**架构设计与推理流程**，而非极致性能：算子实现以正确性与可读性优先。第三方依赖保持极简（`spdlog` 与 `nlohmann/json`，仓库内已内置），其余逻辑均由 C++ 实现，适合作为课程作业、研究原型或个人学习的参考基线。

---

## 快速开始

### 依赖与构建
- C++17 编译器
- CMake（≥ 3.10）
- 可选：OpenMP（未安装也可编译运行；开启后可加速部分算子）

```bash
bash build.sh
```

### 准备模型文件
默认适配 [**Qwen2.5-0.5B**](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)；请将相关文件放在以下路径（`data/` 已被 git 忽略）：

```
data/model/
├─ config.json
├─ model.safetensors
├─ tokenizer.json
└─ tokenizer_config.json
```

如需自定义路径，可修改 `include/config.hpp` 中的默认配置。

### 运行示例
```bash
./build/easy_gpt --help
./build/easy_gpt --max-steps 128 --temperature 0.7 --top-p 0.9 --top-k 40 "Hello"
./build/easy_gpt --max-steps 256 --temperature 0.1 -f test/data/test_batch.txt
```

主要参数说明：  
- `-f/--prompt-file` 从文件读取多条 prompt；
- `-m/--max-steps` 控制生成长度；
- `--temperature`/`--top-p`/`--top-k` 控制采样；
- `--greedy` 启用贪心解码；

---

## 项目结构与推理流程

### 代码结构
```
include/             # 公共头文件（核心接口与数据结构）
include/models/      # GPT 组件定义（Embedding/Attention/MLP/Block 等）
include/third_party/ # 轻量三方头文件（json.hpp 等）
src/                 # 核心实现
src/models/          # 模型组件实现
src/third_party/     # 三方实现（spdlog）
test/                # 测试脚本与数据
data/                # 模型权重与 tokenizer 资源（git 忽略）
```

### 推理流程（入口视角）
`src/main.cpp` 负责整体流程编排：  
1) 解析命令行参数（prompt、采样参数、随机种子等）  
2) 读取 prompt 或文件，并套入对话模板  
3) 加载模型配置与权重（`config.json` + `model.safetensors`）  
4) 初始化 `Tokenizer`、`DataManager`、`GptModel`  
5) 构建 `GptEngine` 并执行 `run` 完成生成

### GptModel 推理逻辑（核心链路）
- `DataManager` 先进行分词与**左侧 padding**，记录每条样本的 `seq_len` 与 `pad_len`。  
- `GptModel::forward` 生成 `GenerationContext`，初始化每层的 KV cache，并进入 **prefill → decode** 两阶段。  
- **Prefill**：整段 prompt 并行前向，`Embedding → Block(Self-Attn+MLP)×N → RMSNorm → 输出投影`，基于最后一个位置的 logits 采样得到首个生成 token。  
- **Decode**：逐步只喂入“上一步生成的 token”，通过 KV cache 追加注意力上下文，采样新 token；更新位置索引并检测 EOS，若命中则清理该样本的 KV cache 并从活跃 batch 中移除。  
- 生成 token 会交由 `DataManager` 记录并最终解码为文本输出。

---

## 核心特性（面向学习与理解）

- **完整推理链路**：从配置加载、权重解析、Tokenizer、Prefill/Decode，到采样输出，覆盖端到端流程，便于“按路径学习”。
- **清晰的 GPT 组件划分**：Embedding → Transformer Block（Self-Attn + MLP）×N → Norm → 输出投影，结构清晰，便于对照论文与主流实现。
- **可控采样策略**：内置 Greedy / Top-K / Top-P 采样，参数由命令行配置，适合理解采样对输出的影响。
- **KV Cache 与分阶段推理**：实现 Prefill/Decode 分离，Decode 复用缓存，体现真实推理框架的关键思路。
- **最小依赖、纯 C++ 实现**：除日志与 JSON 解析外，核心逻辑完全由 C++ 实现，适合作为学习型工程参考。

---

## 配置与可扩展点

- **模型与 tokenizer 路径**：位于 `include/config.hpp`，默认指向 `data/model/` 下的文件。
- **精度选择**：默认编译为 BF16（`USE_BF16`），如需切换 FP16/FP32 可在编译选项中调整宏定义。
- **OpenMP**：默认开启（`EASY_GPT_ENABLE_OPENMP=ON`），可通过 CMake 选项关闭或自行安装运行时。
- **模型适配**：当前针对 Qwen2.5-0.5B 的权重命名与配置进行适配，切换其他模型需确保 `config.json`、权重 key 与 tokenizer 兼容。

---

## 测试与复现

- 测试数据位于 `test/`。

---

## 常见问题

**Q: 使用 `-f/--prompt-file` 后输出保存在哪里？**  
A: 会在输入文件同目录生成 `*_output*` 的输出文件，并保留原扩展名（例如 `test_batch.txt` 会生成 `test_batch_output.txt`）。

**Q: 为什么没有进行重度性能优化？**  
A: 本项目定位为学习与理解推理流程的“教学型实现”，重视可读性与清晰结构。

---

## 依赖说明

- `spdlog`：日志输出（仓库内置，位于 `include/third_party/spdlog` 与 `src/third_party/spdlog`）
- `nlohmann/json`：JSON 解析（仓库内置，`include/third_party/json.hpp`）

其余全部为 C++ 原生实现。  
