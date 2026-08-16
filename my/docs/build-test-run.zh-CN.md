# 编译、测试、运行 指南（CPU 环境）

本指南记录在一台**没有 GPU / 没有 CUDA** 的 Linux 机器上，如何从零把
`easy_llm.cpp` 编译、跑测试、并跑通一次端到端推理。所有步骤都经过实际验证。

> 适用环境示例：Ubuntu 24.04 + g++ 13.3 + CMake 3.28，无 nvcc / nvidia-smi。

---

## 0. 先判断有没有 CUDA

仓库自带的 `build.sh` **默认开启了 CUDA**，在没有 GPU 的机器上直接跑会在 cmake 配置阶段失败。
先确认环境：

```bash
nvcc --version      # 没有输出 → 无 CUDA 工具链
nvidia-smi -L       # 没有输出 → 无 GPU
```

- 有 CUDA：可以直接 `bash build.sh`。
- 无 CUDA：按下面的 **CPU-only** 方式来，不要用默认的 `build.sh`。

---

## 1. 编译（CPU-only）

关键是把 CUDA 关掉：`-DEASY_LLM_ENABLE_CUDA=OFF`。

```bash
# 在仓库根目录
cmake -S . -B build \
  -DCMAKE_SYSTEM_NAME=Linux \
  -DEASY_LLM_ENABLE_CUDA=OFF \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CXX_COMPILER=g++

cmake --build build -j 8
```

成功后会在 `build/` 下生成主程序 `easy_llm` 和一批测试可执行文件。

> 说明：`CMakeLists.txt` 里 `EASY_LLM_ENABLE_CUDA` 默认就是 `OFF`，
> 只是 `build.sh` 脚本里手动传了 `ON`。所以用上面的 cmake 命令即可，
> 或者把 `build.sh` 里那行 cmake 换成脚本里已注释的 no-cuda 版本。

---

## 2. 跑测试

用 CTest 跑全部单元测试（CPU 环境下 CUDA 相关测试不会被编译，自然也不会跑）：

```bash
cd build
ctest --output-on-failure
```

预期：**9/9 全部通过**，其中包含 4 个带 `invariant_gate` 标签的回归门测试。

只跑回归门（更快）：

```bash
cd build
ctest --output-on-failure -L "^invariant_gate$"
# 或者用自定义 target：
cmake --build build --target easy_llm_regression_gates
```

---

## 3. 准备模型权重

推理需要模型文件，仓库里**不包含**（`data/` 已被 `.gitignore` 忽略）。
默认适配 **Qwen2.5-0.5B-Instruct**，需要 4 个文件放到 `data/model/`：

| 文件 | 说明 | 大小 |
|------|------|------|
| `config.json` | 模型结构配置 | ~1 KB |
| `model.safetensors` | 权重 | ~988 MB |
| `tokenizer.json` | 分词器 | ~7 MB |
| `tokenizer_config.json` | 分词器配置 | ~7 KB |

从 HuggingFace 下载：

```bash
cd <repo_root>
mkdir -p data/model && cd data/model
BASE="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main"
curl -sSL -o config.json           "$BASE/config.json"
curl -sSL -o tokenizer_config.json "$BASE/tokenizer_config.json"
curl -sSL -o tokenizer.json        "$BASE/tokenizer.json"
curl -sSL -o model.safetensors     "$BASE/model.safetensors"   # 约 1GB，稍慢
cd ../..
```

> 这几个路径由 `include/config.hpp` 定义（默认 `data/model/`），
> 想换模型或路径可以改那里。

---

## 4. 运行推理

### 单条 prompt（贪心解码，结果确定）

```bash
printf 'Convert 37 Celsius to Fahrenheit, output as F=xx.x with one decimal.\n' > /tmp/one.txt
./build/easy_llm -f /tmp/one.txt --max-steps 120 --greedy
```

模型会正常生成回答，并在生成 `<|im_end|>`（EOS）时自然停止。

### 批量 prompt（仓库自带示例，对应 `test.sh`）

```bash
./build/easy_llm -f test/data/test_batch.txt --max-steps 256 --temperature 0.1
```

这会并行处理 `test/data/test_batch.txt` 里的 13 条 prompt。
CPU 上较慢（每步约 0.6~0.7 秒，整批可能十几分钟），属于正常现象。

---

## 5. ⚠️ 关于 `--max-steps` 的坑（重要）

**`--max-steps` 是"总序列长度上限"，不是"新生成 token 数量"。**

也就是说，它 = prompt 长度 + 生成长度 的上限。举例：

- prompt 编码后约 43 个 token，若设 `--max-steps 40`，
  则 prefill 完就已超限，**只会生成 1 个 token 就停**，看起来像"没输出"。
- 想真正看到生成效果，`--max-steps` 要留足空间（这就是示例用 `256` 的原因）。

如果发现"模型只吐一个 token 就结束"，先检查是不是 max-steps 太小，而不是以为程序坏了。

---

## 6. 常用参数速查

```
-f, --prompt-file <path>   从文件读 prompt（每行一条，忽略空行）
-m, --max-steps <n>        总序列长度上限（含 prompt），默认 100
    --temperature <float>  采样温度，默认 0.8
    --top-p <float>        nucleus 采样，默认 0.95
    --top-k <int>          top-k 采样，0 关闭，默认 20
    --greedy               贪心解码（覆盖上面的采样参数，结果确定）
    --seed <int>           随机种子，默认 42
    --serve                以常驻连续批处理服务模式运行
```

完整帮助：`./build/easy_llm --help`

---

## 7. 验证结果参考

用 `test.sh` 的命令跑完 13 条 batch 后，短回答会和参考文件
`test/data/test_batch_output.txt` 逐字符一致；较长的回答可能因为
`--max-steps 256` 的步数上限被截断（前缀仍然一致）——这是参数限制，
不是引擎错误。想复现完整的长回答，把 `--max-steps` 调大（如 512）即可。
