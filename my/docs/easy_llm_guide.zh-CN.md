# easy_llm.cpp 源码导读：从一条 Prompt 到下一个 Token

> **本文生成要求（整理版）**
>
> 通读 `eagle-dai/easy_llm.cpp` 仓库的代码、README、构建脚本和测试，在 `my/docs/` 下编写一份简体中文教程。教程需要从读者视角组织知识，先建立最少必要背景，再沿真实代码路径逐层深入；解释每个主要模块“做什么、为什么需要、代码如何实现、容易误解什么”。术语保留英文，并在文末提供词汇表。对于无法从仓库确认的行为，优先核对官方文档或规范；仍不能确认时明确说明，不把推测写成事实。适合时使用 Mermaid，尤其是 sequence diagram，并控制图的宽度。内容必须与当前仓库代码一致。生成后应重新检查结构、事实、代码路径和图表。
>
> 参考写作链接由用户提供：`https://mp.weixin.qq.com/s/1hyhKlbni06xi2q1xIarZQ`。生成环境未能可靠读取该页面，因此本文没有虚构或转述其中的具体内容，只遵循上面的明确写作要求。

---

## 这份教程怎么读

这不是一份要求从第一页背到最后一页的 API 手册，而是一条**从 Prompt 到 next token 的源码学习路线**。

为了让第一次阅读不会被 4000 行内容压住，正文按学习阶段拆成 5 个章节文件。章节编号仍然保留原来的 `0～109`，因此引用源码讨论时仍然方便定位。

| 章节 | 主要内容 | 第一次阅读建议 |
|---|---|---|
| [01：先跑起来，并跟完一条 Prompt](easy_llm_guide/01-request-path.md) | 第 0～17 节：运行、请求链路、Tokenizer/BPE、Batch、Left Padding | **必读** |
| [02：模型加载、Tensor 与 Transformer 外壳](easy_llm_guide/02-loading-tensor-model.md) | 第 18～31 节：Safetensors、参数契约、Tensor/Ops、GptModel、Block | 第二遍 |
| [03：Attention、KV Cache 与生成](easy_llm_guide/03-attention-generation.md) | 第 32～56 节：RMSNorm、Q/K/V、GQA、RoPE、KV Cache、Prefill/Decode、Sampling | **核心必读** |
| [04：Serving、CUDA、测试与兼容性](easy_llm_guide/04-serving-cuda-tests.md) | 第 57～85 节：Continuous Batching、CUDA、安全 fallback、测试、常见坑 | 第三遍 |
| [05：调试、扩展与参考资料](easy_llm_guide/05-debugging-extension-reference.md) | 第 86 节以后：调试方法、扩展点、总图、术语表、练习、外部资料 | 按需查阅 |

### 三遍阅读法

**第一遍：只建立主线。**

重点回答：

1. 文本在哪里变成 token IDs？
2. token IDs 在哪里变成 hidden states？
3. 一个 Transformer Block 对 hidden states 做了什么？
4. Prefill 为什么输入整段 Prompt，而 Decode 每次只输入 1 个 token？
5. 历史 token 为什么不需要每轮重新计算 K/V？
6. logits 最后怎样变成一个 next token？

如果这 6 个问题还没有答案，不要急着进入 CUDA。

**第二遍：看懂模型内部。**

重点追 Tensor shape：

```text
[B, S]
  ↓ Embedding
[B, S, H]
  ↓ Q/K/V projection
Q: [B, Nh,  S, D]
K: [B, Nkv, S, D]
V: [B, Nkv, S, D]
  ↓ Attention + MLP
[B, S, H]
  ↓ tied LM head
[B, S, Vocab]
```

**第三遍：理解工程化。**

再看：

```text
Continuous Batching
CUDA backend
KV slot lifecycle
CPU / CUDA fallback
invariant tests
compatibility boundaries
```

---

## 先只记住这张图

```text
Prompt
  ↓
Chat Template
  ↓
Tokenizer / BPE
  ↓
Token IDs + Left Padding
  ↓
Embedding
  ↓
Transformer Blocks × N
  ├─ RMSNorm
  ├─ Self-Attention
  │  ├─ Q/K/V Projection
  │  ├─ RoPE
  │  ├─ GQA
  │  ├─ Mask
  │  └─ KV Cache
  ├─ Residual
  ├─ RMSNorm
  ├─ Gated MLP
  └─ Residual
  ↓
Final RMSNorm
  ↓
Tied LM Head → Logits
  ↓
Softmax / Sampling
  ↓
Next Token
  ↓
Decode，直到 EOS 或达到步数限制
```

第一次看到缩写，只需要知道它们负责什么：

- **BPE（Byte Pair Encoding）**：把文本切成模型词表里的 token。
- **RMSNorm**：对 hidden state 做归一化。
- **RoPE（Rotary Position Embedding）**：把位置信息作用到 Q/K。
- **GQA（Grouped Query Attention）**：多个 Query heads 共享更少的 Key/Value heads。
- **KV Cache**：缓存历史 token 已经算好的 K/V，让 Decode 不必从头重算。
- **EOS（End of Sequence）**：表示生成结束的 token。

这些概念在正文里都会回到**真实代码**逐个展开。

---

## 两个特别容易学错的点

### 1. `tokenizer_config.json` 里有 chat template，不等于当前 C++ 会读取它

当前实现的 Qwen chat template 来自：

```text
src/cli_options.cpp::apply_chat_template()
```

也就是**代码中硬编码**。

当前 C++ 从 `tokenizer_config.json` 主要读取 special token / pad token。后面讲兼容性时会再次说明这一点。

### 2. GQA 不等于 `repeat()` K/V

Qwen2.5-0.5B-Instruct 的模型结构本身已经是：

```text
Query heads = 14
KV heads    = 2
```

这就定义了 GQA：每个 KV head 被一组 Query heads 共享。

当前 CPU Attention 为了后续矩阵运算方便，会把 K/V 用 `repeat()` 扩展到 14 heads。**这是本项目 CPU 路径的实现选择，不是 GQA 的定义要求。**

因此也不能直接拿“理论上 Nkv=2”推断当前 CPU KV Cache 一定只占 2-head 的内存；正文会结合实际 cache shape 解释。

---

## 推荐第一次动手

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CXX_COMPILER=g++

cmake --build build --target easy_llm -j8

./build/easy_llm --greedy "Hello"
```

然后打开：

```text
src/main.cpp
```

顺着这一条链走：

```text
main
→ DataManager
→ GptModel
→ Block
→ SelfAttn / MLP
→ Sampler
→ decode text
```

现在进入第一章：

**[开始阅读：01 — 先跑起来，并跟完一条 Prompt →](easy_llm_guide/01-request-path.md)**

---

## 阅读时的总原则

遇到代码先问四个问题：

```text
1. 输入 shape 是什么？
2. 输出 shape 是什么？
3. 这一步为什么必须存在？
4. 它属于模型定义，还是当前项目的实现选择？
```

第四个问题尤其重要。比如：

- `Nh=14, Nkv=2` → **模型结构 / GQA 定义**
- CPU `repeat()` K/V → **当前实现选择**
- Qwen 官方 chat template → **模型/tokenizer 约定**
- `apply_chat_template()` 硬编码 → **当前项目实现**
- `rms_norm_eps=1e-6` → 官方配置与当前默认模型一致
- C++ 里直接写死 `1e-6` → **当前实现边界**

这样读源码，才不会把“这个仓库现在这样写”误认为“所有 LLM 都必须这样写”。
