---
layout: modern-article
title: How WhalePod Raise KVCache Hit Rate - My Design of Agent Harness
tags: Agent LLM
comments: true
---

本文主要讨论 KVCache 在 Agent Harness 设计上如何做到更好的利用，在保证正确率的情况下，尽可能减少 Agent 使用开销。

所有实验基于我的开源项目 [WhalePod](https://github.com/Timothyhay/whale-pod)，项目中也包含了 benchmark 代码和本文中的结果。

> 注意 - WhalePod 的上下文管理实验复现有一条隐形要求：**确保你的模型（DeepSeek-V4 比如）的自动前缀缓存能在长对话中持续生效（因为梁圣的 KVCache 保存时间目前是最久的，似乎长达 2h）**，在此基础上让请求的字节前缀保持稳定。
 

本文的设计则是在应用层，在其他层间各种设计的基础上，如何真正在使用 Agent 的过程中利用好 KVCache 带来的增益。

接下来我们从架构原理、评测系统设计和完整实验数据三个层面，系统性地验证让前缀字节流稳定产生的增益。

涵盖两条互补轨道的完整流程：离线预测（零网络/API key 得到的理论结果）和在线实测（请求 DeepSeek 看看花了多少钱），以及两个维度的评价指标——可复用前缀份额（reusable prefix ratio）和命中率（cache hit rate）。

全文用 60 个独立 session、三种对话长度（6/12/18 轮）和三种推理回放策略（never/tool/always）的数据来论证设计的稳健性。在`bench`目录可以找到素有实验的复现入口。
如果你很急，也可以直接看 [评测目标](#1-评测目标)、我们的[Agent 架构设计：为何前缀缓存能命中](#3-agent-架构设计为何前缀缓存能命中)，或者[实验结果](#6-实验结果)。

---

# 0. 稍等，我刚来，你们在聊的 KVCache 是什么？

## 0.1 理论

KV Cache 是基于 Transformer 架构的自回归 LLM 在**推理解码（Decode）阶段**使用的一种以空间换时间的显存优化机制。

首先我们知道，标准的多头自注意力机制（Self-Attention）是：
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

在自回归文本生成过程中，模型按是按步（时间） $t$ 逐个生成 Token 的：
* **Prefill 阶段**：模型对输入的 长度为 $N$ Prompt 序列并行计算所有 Token 的投影矩阵，得到 $Q_{1:N}, K_{1:N}, V_{1:N}$，完成第一次注意力计算，并将生成的 $K_{1:N}$ 和 $V_{1:N}$ 暂存在显存中。
* **Decode 阶段**：当生成第 $t+1$ 个 Token 时，KVCache 的设计使模型**只需针对新 Token 生成单行查询向量 $q_{t+1}$ 以及对应的 $k_{t+1}, v_{t+1}$**。接下来将新计算的键值向量追加至缓存中：
  $$K_{1:t+1} = [K_{1:t} \,;\, k_{t+1}], \quad V_{1:t+1} = [V_{1:t} \,;\, v_{t+1}]$$
  然后仅通过 $q_{t+1}$ 与完整的 $K_{1:t+1}, V_{1:t+1}$ 计算单向量对矩阵的注意力：
  $$\text{Attention}(q_{t+1}, K_{1:t+1}, V_{1:t+1}) = \text{softmax}\left(\frac{q_{t+1} K_{1:t+1}^T}{\sqrt{d_k}}\right) V_{1:t+1}$$


本质上，KVCache 做的事情是一个 **消除冗余计算** 的任务。也就是避免了每生成一个新 Token 都对历史所有 Token 重新算 $K$ 和 $V$ 向量（重新执行投影线性变换，Linear Projections），使每一步的计算复杂度由全局重算的 $O(t^2 \cdot d)$ 降低为增量计算的 $O(t \cdot d)$。
> 这里的 $d$ 指 hidden layers 的总维度，因为每生成一个 Token，需要与历史 $t$ 个 Token 的向量分别做内积计算，每个向量的长度是 $d$。

由于这个**空间换时间**的做法，KV Cache 让解码阶段系统吞吐瓶颈从算力变成了 GPU 显存容量与 HBM 读写带宽。


### 0.2 在我们动手之前，还发生了什么

由于长上下文与并发请求会导致 KV Cache 显存爆炸：

$O(\text{Batch} \times \text{Length} \times \text{Layers} \times \text{Heads} \times d)$

工业界和学术界围绕 KV Cache 衍生出了一系列关键技术。其中与提高缓存命中率相关的：

#### 1. 模型架构级优化
主要在减少 KV Head 数量与维度上。
* **MQA (Multi-Query Attention)**：所有 Query Head 共享单一组 Key/Value Head，KV Cache 显存占用直接降低到原本的 $1/H$（$H$ 为注意力头数）。
* **GQA (Grouped-Query Attention)**：折中方案（如 Llama 2/3、Mistral），将 Query Head 分组，每组共享一对 Key/Value Head，在保持模型精度的同时大幅削减 KV 显存。
* **MLA (Multi-Head Latent Attention)**：DeepSeek-V2/V3 提出的创新架构。将 Key 和 Value 联合投影压缩为一个低维的**隐向量（Latent Vector）**，仅缓存该隐向量，在推理时通过矩阵吸收（Matrix Absorption）还原计算，将 KV 显存开销降低至原本的 10%~20% 左右。

#### 2. 显存管理与系统级优化
* **PagedAttention (vLLM)**：借鉴操作系统虚拟内存分页的思想，将连续的 KV Cache 离散存储在固定大小的物理显存块（Block）中，彻底解决了显存内外部碎片问题，将显存浪费率从 >60% 降至 <4%。
分页机制消除了内部与外部显存碎片，相当于释放出来的有效显存可以容纳更多并发与前缀块，其实是间接提升了命中率的上限。
* **RadixAttention / 前缀树缓存 (SGLang 等)**：采用基数树（Radix Tree）管理 KV Cache，实现跨请求、跨轮次、树形搜索（Tree-of-Thought）分支场景下的前缀精确复用与自适应 LRU 淘汰。
主要是因为 Radix Tree 在内存中维护所有历史请求的 KV Cache，这样对多轮对话（显然）直接把上轮的上下文全复用了；同一 Prompt 分叉生成多个候选回答时，分叉前的公共前缀只需计算一次；对不同用户的不同请求只要开头一致，树结构也可以精确匹配并直接命中前缀。

#### 3. 缓存压缩与稀疏化 / Compression & Eviction
* **KV 极低比特量化**：
  * 将 KV Cache 从 FP16/BF16 量化为 **FP8、INT8 或 INT4**（如 KIVI、Q-Serve），直接让显存占用减半甚至降低至 $1/4$，成倍扩大并发容量。
  比如将每个 KV 浮点数从 16 位压缩到 8 位甚至 4 位， Token 的显存占用变小了，内存中缓存的条目就翻倍了。
* **Token 动态淘汰与稀疏注意力**：
  * **StreamingLLM**：保留最开头的几个“注意力汇聚点（Attention Sinks）”和最新的局部滑动窗口 Token，丢弃中间大部分 KV Cache，以固定显存支持无限长上下文流式输出。
  * **H2O (Heavy Hitter Oracle) / Scissorhands**：基于历史注意力权重，动态识别并保留贡献最大的“Heavy Hitter Token”，丢弃不重要的 Token 缓存。

#### 4. 分布式与分层存储 / Disaggregated & Tiered Caching
* **分层缓存卸载（Hierarchical KV Caching，如 Mooncake）**：构建 `GPU HBM -> Host DRAM -> 本地 NVMe SSD -> 分布式存储` 的多级缓存系统，在超长文本及海量 Prompt 共享场景下，把不活跃的 KVCache 换到成本低容量大的地方存。
即使很久没被调用，也能在 CPU 内存中命中并快速拉回 GPU，避免冷启动重新计算。


背景到这里结束。站在其他设计的肩膀上，我们来看看 Agent 本身是否存在设计范式来提升 KVCache 利用率。

---

## 1. 评测目标

KV cache 的命中率是一个成本指标，不是一个正确性指标。我们追求的不是"命中率高"，而是"在保证功能完整的前提下命中率高的**同时** prompt token 总量**低**"——命中率一样但灌入了过多 token 的设计同样不省钱。

两条轨道：

| 轨道 | 环境 | 测量对象 | 输出 |
|:---|:---|:---|:---|
| **离线 (`validate.py`)** | 零网络、零 API key | 请求之间的字节前缀可复用比（理论上限） | 四种上下文设计的对比 |
| **在线 (`live_acceptance.py`)** | 真实服务器 | 服务器报告的 `prompt_cache_hit_tokens`（实际命中） | 命中率验证 + 预测一致性 |

离线说实验的意义在于说明*理论上*最多应该有多少token复用；在线实验则实际查看服务器确实有多少复用。
两者之间的差距（MAE）衡量我们的理解与服务器行为的一致程度。

---

## 2. 代码文件

评测系统全部驻留在 `bench/` 目录下，外加 `whalepod/core/tokenizer.py` 和 `tests/test_bench_encoding.py` 两个支撑文件。每个文件职责如下：

### `bench/validate.py` — 离线验证引擎

完全不依赖网络。其工作机制：

1. **`ScriptedEndpoint`**（继承 `VLLMEndpoint`）：替代真实 HTTP 连接，在内存中按预设脚本复读应答。因为走的是真实的 `_payload()` 构建消息、真实的 Agent 循环，所以测量的是**生产代码路径**，不是手工模拟。
2. **`wire_text(payload)`**：将发出的 OpenAI 格式 JSON 转为模型**真正接收**的输入字符串。调用的是 `bench/dsv4_encoding.py` 里的 `encode_messages()`，该函数来自 DeepSeek V4-Flash-0731 的官方仓库，处理 BOS、角色分隔符（`<|User|>` / `<|Assistant|>`）、tool schema 挂载到 system message、tool result 合并到 user message 等所有细节。
3. **`common_prefix_len(a, b)`**：对两个连续请求的输入字符串做二分查找，找最长公共前缀（字节级）。
4. **`estimate_tokens(shared_prefix)`**：用 DeepSeek V4 的官方 BPE tokenizer（`whalepod/core/tokenizer.py` → `deepseek-v4 (tokenizer.json)`）数 token，然后按 64-token 块边界截断（block-aligned）。
5. **`measure(payloads)`**：对全部请求对执行 `wire_text → common_prefix_len → estimate_tokens → blocks`，输出每个请求的 `RequestMetrics`。

五个独立实验：

| 实验 | CLI | 测量什么 |
|:---|:---|:---|
| `variants` | 默认 | as-built / no-ledger / rolling-summary / three-zone 四种设计跑同一个 session |
| `variants-small` | 自动触发 | 同上，但窗口缩小到 ~45k，迫使所有设计都 prune |
| `prune` | 默认 | as-built 在 45k 窗口下观察 prune 后前缀恢复曲线 |
| `repomap` | 默认 | repo map 在不同 token budget 下的实际渲染 token 数 |
| `deny` | 默认 | sandbox=`--yes` 模式下危险命令的拒绝链 |

三个子实验：

| 选项 | 测量什么 |
|:---|:---|
| `--only main` | 12 轮 session，`--repeat N` 跑 N 次取分布 |
| `--only prune` | 同上，但窗口缩小到 ~45k（强制 prune） |
| `--only pin-check` | Provider affinity A/B：同一 prefix，pin 到 provider vs 自由路由 |

### `bench/dsv4_encoding.py` - from DeepSeek V4 官方消息编码器

从 HuggingFace `deepseek-ai/DeepSeek-V4-Flash-0731` vendor 来的，MIT 许可。独立于任何框架（无 `transformers` 依赖），完全自包含。对外接口：

- **`encode_messages(messages, thinking_mode, drop_thinking, reasoning_effort)`**：把 OpenAI 格式的消息列表转为模型需要的输入字符串（BOS + 角色分隔符 + tool schema 渲染 + tool-result 合并）。
- **`merge_tool_messages(messages)`**：把 `role="tool"` 的消息合并到前一个 user message 中（V4 不支持独立的 tool 角色）。
- **`parse_message_from_completion_text(text)`**：从模型输出的字符串中还原 `reasoning_content` / `content` / `tool_calls`。

`wire_text` 调用 `encode_messages` 而非自造的 JSON dump，使得离线测量的字节流和服务器实际 tokenize 的完全一致。

### `tests/test_bench_encoding.py` — 回归测试

八个测试用例覆盖：
- tokenizer 优先级：无 dsv4 tokenizer 时回退到 tiktoken/heuristic
- `wire_text` 确定性：相同 payload 产生相同字节
- `wire_text` encoder 验证：输出包含 BOS、`<|User|>`、`<|Assistant|>`，不包含 `"role":"system"`
- tool result 被合并进 user message（V4 无独立 tool 角色）
- 不同首消息打破前缀

---

## 3. Agent 架构设计：为何前缀缓存能命中

KV cache 的命中率不取决于调参，取决于**代码结构**。以下七个设计决策是 95% 命中率的基石。

### 3.1 两区消息存储 - 不可变前缀 + 只追加历史

WhalePod 的消息管理器（`whalepod/core/messages.py`）把消息分成两个区：

```
Zone 1 — 稳定前缀（不可变）
├── tool definitions     ← 本轮启用的工具，session 开始时确定，全 session 不变
├── system prompt        ← 静态系统提示词（`whalepod/core/prompt.py`），无 cwd/时间/模式
└── repo-map summary     ← 仓库符号表（`whalepod/context/repo_map.py`），token-budget 约束

Zone 2 — append-only 历史                          ⟵ 从未被就地改写，只在末尾追加
├── turn 1: user → assistant → tool results
├── turn 2: user → assistant → tool results
└── ...
```

**分区原则：** Zone 1 完全不变、Zone 2 只新增不修改。在此基础上越容易改变的的内容放在越后面。

**为什么重要：** KV cache 以最长公共前缀为 key。根据我们的设计原则所以每个新请求的前缀都等于"Zone 1 + Zone 2 的已发送部分"，和上一个请求的前缀完全重叠。server 不会因为"中间某个消息被改了内容"而触发冷 miss。

prompt 放在 repo-map **前面**不是偶然。prompt 是 session 直不变的，repo-map 在 `/refresh` 后可能更新。如果 prompt 在 repo-map 之后，一次 `/refresh` 会破坏整个系统提示词的缓存——这意味着剩下的 byte 全部被重新计费。prompt 在前意味着 `/refresh` 只破坏 map 的那一小段尾缀。

### 3.2 Context Ledger — 拦截重复读

`whalepod/core/ledger.py` 记录了每个已送入上下文的文件范围（path + start + end）+ 文件身份（mtime_ns + size）。当模型请求某个文件时，Agent 走两条路径（`whalepod/core/agent.py:_run_read`）：

```
模型请求 read_file("messages.py", start=1, end=50)
  ↓
ledger.hit(path="messages.py", start=1, end=50)?
  ├─ YES → 返回指针，不发送文件:
  │         "[ledger] messages.py:1-50 is already in the window (from turn 1)"
  │         零额外 token
  └─ NO  → 发送完整文件内容
             记录 entry: (path, start, end, mtime_ns, size, turn)
```

文件被编辑后（WhalePod 自己的 edit 或者外部修改），`invalidate()` 删掉对应的 ledger 条目，让后续请求重新读到真实内容。

**为什么重要：** 在 turn 4、7、9、11 中，模型重复请求了已经读完的文件。没有 ledger，每次重复读都会往上千 token。12 轮 session 中，ledger 省下了 ~17k token（离线测量）。在更长 session 中效果更大。

### 3.3 Provider Affinity — 前缀缓存是单机状态

`whalepod/core/base.py:extra_body` 在 OpenRouter 上把请求 pin 到一个 `provider: {order: ["DeepInfra"], allow_fallbacks: false}`。前缀缓存在某**一台**服务器的 KV cache 中，聚合器的正常行为是把每个请求路由到不同 provider，每个请求都是冷 miss。

实测数据：同一 11.2k 前缀，不 pin **0.4%** 命中率，pin 到指定 provider **98.4%**。

直连 DeepSeek 官方 API 时不需要额外配置（本身就是单 endpoint），这个设计主要面向 OpenRouter。

### 3.4 Reasoning 按需回传

DeepSeek 明确要求没有工具调用时中介 assistant 的 reasoning_content 无需回传，传了也会被忽略；
而发生了工具调用时，它必须参与上下文凭借并在后续所有请求中原样回传，否则 DeepSeek 的 API 会返回 400（官方文档说的，但实际目前并不会）

因此 chain-of-thought 不能无条件回放到后续请求中。我准备了几种设计：

- **never**：一律不写入 payload，只保留在本地 UI 中显示；
- **tool**（默认）：仅当该 assistant 回合调用了工具（`m.tool_calls`）时回传思考，纯回答回合照旧剥离；
- **always**：每个回合都回传，适用于 Kimi K3 等模型。

**为什么重要：** 一次思考动辄几百上千 token。如果不加约束地回放，每个 assistant 回复都带着一大段思考回到前缀里——第一第二次能缓存，但几个 turn 之后整个区域都是几十 K 的推理文本，prompt 总量被成倍放大。默认策略只在"必须解释工具决策"的回合保留思考，其余剥离。§6.4 用三种策略各 10 次 session 量化了这个旋钮的成本与收益。

同时，我注意到一个上古时期 Agent 和推理框架可能存在的问题。就是如果不通过 reasoning_content，wire-level 的 CoT 带上 `<think>` 等 special token 时，
服务端对新请求做 tokenize 会把这些文本按普通 BPE token 重新切分，导致服务端生成的控制 token 与客户端回发时切好的普通文本 token 在 BPE 结果上不一致。
这是一种现代推理服务 tokenize 之前的安全机制，不然人人都可以随意注入 special token 了。

### 3.5 每个工具自带使用规则，不通用的不会留在前缀里

`whalepod/tools/registry.py` — 每个 tool-definition 带 `guidelines`（使用指南）。`build_system_prompt` 从当前会话中实际激活的工具中收集指南，"Using the tools" 段落是动态生成的。但生成后 **session 全程不变**（工具集是 session 开始时锁定的）。

在 readonly 沙盒中写工具不被提供，它们的用法指南也不会留在 prompt 中——所以前缀里不浪费 token 给不可用的工具。

### 3.6 Compaction 取代盲 Prune

`whalepod/compaction.py`：prune 在窗口超限时删除旧 turn 并留一个 marker。但 prune 掉的东西往往是 session 开头用户说的目标——永久丢失，浪费前缀也是重新计算的。

Compaction 用同一切线和同样的前缀失效成本，只是用一个小模型调用把要删除的 turn 总结成一份包含文件清单和行号范围的摘要。Compaction 失败会回退到 prune，所以用户的 turn 永远不会失败。

**为什么重要：** 窗口超限时只有一刀（一次全前缀失效），不管是 prune 还是 compaction 都一样。Compaction 多了一次小型 API 调用，换回了丢失的信息，让前缀在恢复后依然可以工作。

### 3.7 行结束符归一化 — 保证 BPE 稳定

`whalepod/tools/textfile.py` — 读取时文件内容归一化为 LF，写入时恢复为文件的原始行结束符。没有这个处理，在 CRLF checkout 上做编辑，`old` 文本不匹配实际文件，编辑失败；重新读取不帮忙，因为 read_file 的输出也是 LF。

**为什么重要：** BPE tokenizer 对 `\r\n` vs `\n` 产生不同的 token 序列。如果行结束符在编辑过程中改变，文件内容的编码就变了——对于包含该文件的请求，前缀中的对应部分不再复用。

---

## 4. 评测架构

### 共享的 Session 脚本

离线四个变体和在线实测共用同一组 12 轮对话（定义在 `bench/validate.py:SESSION`）：

```text
turn  1: "Where is prefix caching handled?" → grep + read_file messages.py
turn  2: "How does the ledger stop duplicates?" → read_file ledger.py
turn  3: "Walk me through the agent loop." → read_file agent.py
turn  4: "Remind me the prune thresholds." → read_file messages.py (re-read!)
turn  5: "What tools does the registry expose?" → read_file registry.py
turn  6: "Show me the endpoint abstraction." → read_file endpoints/base.py
turn  7: "In agent.py, where is the ledger consulted?" → read_file agent.py (re-read!)
turn  8: "Compare the two provider implementations." → read_file vllm.py + anthropic.py
turn  9: "Does the ledger handle file edits?" → read_file ledger.py (re-read!)
turn 10: "How is the repo map budgeted?" → read_file repo_map.py
turn 11: "Which registry function plans a write?" → read_file registry.py (re-read!)
turn 12: "How is config resolved?" → read_file config.py + tree_view whalepod/
```

Turn 4, 7, 9, 11 是重复读（same file 已经在前面的 turn 中读过），ContextLedger 应该拦截这些请求，用指针替代文件内容。

### 双重验证流程

```
                     SESSION (12 turns)
                    /                \
       Offline (validate.py)     Live (live_acceptance.py)
       ScriptedEndpoint           RecordingEndpoint
              |                   DeepSeek API
         4 variants            5 repetitions
              |                       |
  reusable prefix per request    cached tokens per request
  (byte-level prediction)       (server measurement)
              |                       |
              +---------- MAE ---------+
                    agreement check
```

离线提供上限（字节层面能复用多少），在线提供实测（服务器实际给了多少缓存），两者的 MAE 衡量预测精度。MAE 小（<3 点）说明离线模型贴切；MAE 大（>10 点）说明要么 tokenizer 不对，要么字符编码模型和服务器不一致，要么 provider 端的 KV cache 被驱逐。

### 离线计算链路（一张图的完整路径）

以 as-built 变体第 2 个请求为例：

```
VLLMEndpoint._payload()
  → OpenAI 格式 JSON:
    {model: ..., messages: [{role:"system", content:"..."}, {role:"user", content:"..."}], tools: [...]}
  ↓ wire_text()
dsv4_encoding.encode_messages()
  → 模型输入字符串:
    <|begin▁of▁sentence|>Reasoning Effort:...<|User|>where is prefix caching...<|Assistant|><think>
  ↓ common_prefix_len()
  和上一个请求的字符串比: 15,232 bytes 共享
  ↓ estimate_tokens(prefix)
  V4 BPE → 238 tokens
  ↓ blocks()
  64-token 对齐 → 192 reusable tokens (floor to block boundary)
  ↓ RequestMetrics
  reusable_frac = 192 / total_prompt_tokens
```

在线则是服务器直接告诉你 `prompt_cache_hit_tokens`（也就是它内部分配 KV cache 块时命中的 token 数）。

---

## 5. 实验流程

### 5.1 环境准备

```bash
# 创建 venv + 安装依赖
python -m venv .venv
pip install -e ".[treesitter,tokenizer,dev]"

# 一次性下载 DeepSeek V4 的 tokenizer（公开 HF 仓库，无需 API key）
python bench/fetch_tokenizer.py
# → 下载到 ~/.whalepod/tokenizers/deepseek-v4-flash/tokenizer.json (~6.4 MB)
```

### 5.2 离线验证

不需要网络，不需要 API key。

```bash
# 全量跑（五种实验 + 全部 SVG 图表）
python bench/validate.py

# 只看 variants 对比（快，适合 CI）
python bench/validate.py --only variants --no-charts

# 输出:
#   bench/results/validation.json    ← 机器可读
#   bench/results/validation.txt     ← 人类可读
#   bench/results/*.svg              ← SVG 图表
```

生成的 SVG 图表：

- **[`reusable_prefix.svg`](/images/illustration/2026-08-01/reusable_prefix.svg)** — 四种设计的 per-request 可复用前缀折线（1M 窗口）
- **[`reusable_prefix_small_window.svg`](/images/illustration/2026-08-01/reusable_prefix_small_window.svg)** — 同上，45k 窗口（强迫 prune）
- **[`prompt_tokens_split.svg`](/images/illustration/2026-08-01/prompt_tokens_split.svg)** — 四种设计的 prompt token 堆叠图（绿色 = 可复用，灰色 = 新鲜）
- **[`prompt_cost.svg`](/images/illustration/2026-08-01/prompt_cost.svg)** — 四种设计的提示词成本对比
- **[`prompt_cost_small_window.svg`](/images/illustration/2026-08-01/prompt_cost_small_window.svg)** — 同上，45k 窗口
- **[`prune_recovery.svg`](/images/illustration/2026-08-01/prune_recovery.svg)** — as-built 在 45k 窗口下，prune 后前缀恢复曲线（▼ 标记 prune 点）
- **[`repo_map_budget.svg`](/images/illustration/2026-08-01/repo_map_budget.svg)** — repo map 在不同 token budget 下的渲染大小

### 5.3 在线实测

需要 API key（通过环境变量，不会落地到文件）。

```bash
# PowerShell
$env:WHALEPOD_API_KEY = "sk-..."

# 主 session + prune session + provider affinity A/B (+ pin-check)
python bench/live_acceptance.py

# 大规模重复跑（用 --repeat N 拿到多 session 统计分布）
python bench/live_acceptance.py --repeat 10

```

生成的 SVG 图表：

- **[`live_hit_rate.svg`](/images/illustration/2026-08-01/live_hit_rate.svg)** — 实测命中率 vs 离线预测（run 1）；如果 `--only prune` 也会展示 prune 恢复曲线
- **[`live_tokens_split.svg`](/images/illustration/2026-08-01/live_tokens_split.svg)** — 单个 session 的 cached vs fresh 堆叠图
- **[`live_prune.svg`](/images/illustration/2026-08-01/live_prune.svg)** — prune 窗口下的命中率恢复曲线
- **[`live_provider_affinity.svg`](/images/illustration/2026-08-01/live_provider_affinity.svg)** — Pin vs Unpin A/B 测试（仅 `OpenRouter`，需运行 `--only pin-check` 才会生成）

`--reasoning-strip` 三种策略（never/tool/always）各跑一次后，用 `bench/reasoning_compare.py` 生成横向对比图：

- **[`reasoning_modes_hit_rate.svg`](/images/illustration/2026-08-01/reasoning_modes_hit_rate.svg)** — 三种策略的 session 命中率中位数
- **[`reasoning_modes_per_request.svg`](/images/illustration/2026-08-01/reasoning_modes_per_request.svg)** — 三种策略的 per-request 命中率折线
- **[`reasoning_modes_tokens.svg`](/images/illustration/2026-08-01/reasoning_modes_tokens.svg)** — 三种策略的 cached vs fresh 堆叠图

跨 run 的比较图（多 session 分布、不同对话长度、推理回放成本）由 `bench/article_charts.py` 从已提交的 JSON 生成（[§6.3](#63-不同对话长度的对比)、[§6.4](#64-推理回放策略-ab2026-08-11deepinfra-pin)、[§6.5](#65-单-session-内的累积成本结构)）：

- **[`live_hit_rate_band.svg`](/images/illustration/2026-08-01/live_hit_rate_band.svg)** — 22 session 的 per-request median + P10/P90 带宽
- **[`turns_comparison.svg`](/images/illustration/2026-08-01/turns_comparison.svg)** — 三种对话长度的命中率对比
- **[`mae_by_length.svg`](/images/illustration/2026-08-01/mae_by_length.svg)** — 离线预测误差随对话长度的变化
- **[`prompt_tokens_by_length.svg`](/images/illustration/2026-08-01/prompt_tokens_by_length.svg)** — 三种长度的 cached vs fresh 堆叠图
- **[`reasoning_tokens_by_mode.svg`](/images/illustration/2026-08-01/reasoning_tokens_by_mode.svg)** — 三种策略每 session 的 reasoning token 量
- **[`cumulative_cache_growth.svg`](/images/illustration/2026-08-01/cumulative_cache_growth.svg)** — 单 session 内缓存/新鲜 token 的累积曲线

### 5.4 大规模统计测试

单次 session 是点估计。服务器侧状态（缓存分布、负载）每次不同，需要重复跑拿到分布：

```bash
# 5 次重复，输出 per-request median/P10/P90
python bench/live_acceptance.py --only main --repeat 5 --no-pin `
    --base-url "https://api.deepseek.com" --model "deepseek-chat"
```

然后在 `live_acceptance.json` 中会多出 `main_aggregate`，包含：

```json
{
  "session_hit_rate": {"p10": 0.937, "median": 0.947, "p90": 0.953},
  "session_prompt_cost_usd": {"p10": ..., "median": ..., "p90": ...},
  "per_request": [
    {"request": 1, "n": 22, "median": 1.0, "p10": 0.86, "p90": 1.0},
    ...
  ]
}
```

### 5.5 不同轮数对比实验

验证缓存在不同对话长度下的表现：

```bash
# 短对话 (6 turns)
python bench/live_acceptance.py --only main --repeat 5 --turns 6 --no-pin `
    --base-url "https://api.deepseek.com" --model "deepseek-chat"

# 标准长度 (12 turns) — 推荐 20+ repetitions
python bench/live_acceptance.py --only main --repeat 20 --turns 12 --no-pin `
    --base-url "https://api.deepseek.com" --model "deepseek-chat"

# 长对话 (18 turns)
python bench/live_acceptance.py --only main --repeat 3 --turns 18 --no-pin `
    --base-url "https://api.deepseek.com" --model "deepseek-chat"
```

三种长度的结果在 [§6.3](#63-不同对话长度的对比) 中汇总比较。

**推理回放策略对比**（`--reasoning-strip`）复用同一套 12 轮 session，只切换"模型思考是否回放到下一条请求"的策略：

```bash
python bench/live_acceptance.py --repeat 10 --reasoning-strip never   # live_acceptance.json
python bench/live_acceptance.py --repeat 10 --reasoning-strip tool    # live_acceptance_rs_tool.json
python bench/live_acceptance.py --repeat 10 --reasoning-strip always  # live_acceptance_rs_always.json
python bench/reasoning_compare.py   # 生成三种策略的对比图
python bench/article_charts.py      # 生成跨 run 的全部比较图（含本表）
```

结果在 [§6.4](#64-推理回放策略-ab2026-08-11deepinfra-pin) 汇总。

### 5.6 规模建议

| 场景 | 推荐 N | 时间 | 预算 |
|:---|:---|:---|:---|
| CI 冒烟测试 | 1 | ~3 min | ~$0.02 |
| 内部验证 | 5 | ~15 min | ~$0.10 |
| 统计显著性 | 20+ | ~70 min | ~$0.40 |

P10–P90 跨度小于 3 点说明 provider 稳定、结果可信；跨度 >10 点说明 KV cache 不稳定，换 endpoint 或缩小上下文。

---

## 6. 实验结果

### 6.1 离线验证（2026-08-05，V4 BPE tokenizer）

| Variant | Prompt tokens | Reusable | Reuse % | Prompt cost |
|:---| ---:| ---:| ---:| ---:|
| **as-built** | 911,083 | 858,240 | **94.2%** | $0.020 |
| no-ledger | 1,099,805 | 1,030,528 | 93.7% | $0.025 |
| rolling-summary | 518,504 | 465,216 | 89.7% | $0.013 |
| three-zone | 913,223 | 405,312 | 44.4% | $0.053 |

![Reusable prefix per request, by context design](/images/illustration/2026-08-01/reusable_prefix.svg)

- **as-built** 在每个 turn 之间保持 99.6% 的可复用前缀，只有新文件内容进入时才跌到 76–89%。
- **no-ledger** 的可复用前缀率和 as-built 差不多（93.7% vs 94.2%），但灌入了 **20% 更多的 token**（~1.1M vs ~911k）。前缀可复用只是一个比率，token 总量少才是省钱的根本。
- **rolling-summary** 每次总结时丢掉一大段历史，前缀无可复用基础，只能靠压缩后体积小来减成本——但在实践中，每次总结会产生一个全新的 prompt（没有前缀复用），而 as-built 靠缓存大量 token 以 5x 折扣价（缓存 token 的 20% 计费）。
- **three-zone** 把"当前文件内容"放在历史后面，每个请求都有尾缀变动，前缀复用持续下降到 44.4%——这个设计在看过图表后就没写出来。

![Session prompt tokens: reusable prefix vs freshly billed](/images/illustration/2026-08-01/prompt_tokens_split.svg)

堆叠图清晰地看到：as-built 的柱子最矮（总 token 最少），且绿色（可复用）比例最高。no-ledger 柱子更高（更贵），three-zone 灰色（新鲜）部分最多（最贵）。

![Session prompt cost](/images/illustration/2026-08-01/prompt_cost.svg)

成本对比：as-built **$0.020**，no-ledger **$0.025**（+23%），three-zone **$0.053**（+162%）。rolling-summary 的柱子矮是因为它压缩了上下文所以 token 总量少，但这是以牺牲信息为代价换来的。

![Prune recovery: 45k window forced](/images/illustration/2026-08-01/prune_recovery.svg)

在小窗口下，prune 后的前缀恢复是瞬时的：一次 prune 掉到 ~0%（红色虚线标出），下一个请求立即恢复到 ~90%，session 整体仍保持 88.8% 的可复用比。这就是 prune 的全部代价。

### 6.2 在线实测 — 12 轮对话（2026-08-06，DeepSeek 官方 API）

配置：`https://api.deepseek.com`，`deepseek-chat`（V4-Flash），22 次重复，12 turns。

| Metric | Result |
|:---|---|
| **Sessions** | 22 × 12 turns |
| **Requests per session** | 37（max observed） |
| **Session 命中率** | **median 94.7%** (P10=93.7%, P90=95.3%) |
| **Session prompt tokens** | 1,732,599（1,640,960 cached, 91,639 fresh） |
| **预热曲线** | 前 3 请求综合平均 87.1% → 后 3 请求综合平均 98.2% |
| **离线 vs 实测 MAE** | **5.5 点**（改前 7.0） |
| **ledger 省量** | 2,015 tokens（1 次去重） |

22 次独立 session，命中率中位数 **94.7%**。P10=93.7%，P90=95.3%，跨度仅 1.6 点——评测结果具有统计学意义上的一致性。预热曲线显示：前 3 个请求平均 87.1%（第一次请求在 22 次重复中的命中率中位数即达 99.8%，因为服务端 KV cache 已被前面的 session 预热），随后进入稳态；22 次重复中 32/37 个请求达到 89% 以上。离线预测 MAE 为 5.5 点（+tokenizer/encoder 改进前为 7.0）。

![Per-request cache hit rate: median with P10/P90 band](/images/illustration/2026-08-01/live_hit_rate_band.svg)

上图把 22 次 session 的 per-request 命中率归并成一条 median 线并带上 P10/P90 带宽：第 1–4 个请求（冷启动）median 从 73% 快速爬升；**请求 13 之后 median 稳定在 95–100%，带宽收窄到 14 点以内，到请求 21 之后进一步收窄到 8 点以内**。前 12 个请求带宽较宽（最高 45 点）不是 provider 抖动——那是 22 次重复里"轮到新 turn 读入新文件"的请求在相同索引上错位对齐造成的字节前缀变化，正是离线模型能预测、设计应负责的部分。这张图是 [§6.3](#63-不同对话长度的对比) 里"12 turns"那条的细粒度版本。

### 6.3 不同对话长度的对比

为了验证前缀缓存在不同长度对话中的表现，用同一套 session 脚本分别跑 6、12、18 轮，每个长度重复多次。

| 对话长度 | 重复次数 | 请求数/次 | 命中率中位数 | P10 | P90 | MAE |
|:---|:---|:---|:---|:---|:---|:---|
| **6 turns** | 5 | 20 | **91.7%** | 90.8% | 92.2% | 9.1 |
| **12 turns** | 22 | 37 | **94.7%** | 93.7% | 95.3% | 5.5 |
| **18 turns** | 3 | 33 | **95.0%** | 94.7% | 95.3% | 5.9 |

![Session cache hit rate by conversation length](/images/illustration/2026-08-01/turns_comparison.svg)

![Live prompt tokens by conversation length](/images/illustration/2026-08-01/prompt_tokens_by_length.svg)

![Offline prediction vs server: mean absolute error](/images/illustration/2026-08-01/mae_by_length.svg)

**趋势分析：**

- **短对话（6 turns）命中率最低（91.7%）**：前缀积累不足。每个新 turn 的文件内容占前缀总体的比例更大，所以冷内容频率更高。MAE 也最高（9.1），因为 tokenizer 偏差越大在短上下文里越明显。
- **12 turns 到 18 turns 命中率增长趋缓**：+0.3 个百分点——前缀缓存效果在 12 轮时已经接近最大值，再拉长对话对缓存的帮助不大。这正是设计预期的结果。
- **P10–P90 跨度始终很窄（<3 点）**：不管是短对话还是长对话，分布在 2 个点以内，说明是设计本身的可持续行为，不是偶然运气的单点估计。
- **新鲜计费 token 几乎不随长度增长**：三种长度下每 session 的 billed-fresh 都只有 ~60–90k，prompt 总量的增长全部来自缓存命中的部分——对话越长，缓存分摊得越多，边际成本趋近于零。

结论：**12 turns 是实验甜点**。足够长，让缓存充分生效；足够短，能在小时内跑 22 次取分布。

### 6.4 推理回放策略 A/B（2026-08-11，DeepInfra pin）

[§3.4](#34-reasoning-控制回放-默认只保留工具回合的思考) 把 reasoning 回放定性为"命中率 vs 总字节"的旋钮。为了量化"回放"到底会付出多少、换回多少，我们对 `--reasoning-strip` 的三种策略各跑 10 次 12 轮 session（同一套脚本、同一 provider）：

| 策略 | Sessions | 命中率中位数 | P10 | P90 | MAE | Reasoning tokens | Prompt tokens | Fresh |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| **never（一律剥离）** | 10 | 94.3% | 93.5% | 95.5% | 6.4 | 3,018 | 1,416,947 | 85,363 |
| **tool（生产默认，仅工具轮回放）** | 10 | 96.4% | 96.1% | 96.7% | 3.7 | 4,408 | 1,604,825 | 63,577 |
| **always（每轮回放）** | 10 | 97.1% | 96.8% | 97.3% | 3.7 | 9,830 | 2,202,948 | 61,764 |

![Session cache hit rate by reasoning echo policy](/images/illustration/2026-08-01/reasoning_modes_hit_rate.svg)

![Per-request cache hit rate, by reasoning echo policy](/images/illustration/2026-08-01/reasoning_modes_per_request.svg)

![Prompt tokens by reasoning echo policy: cached vs billed fresh](/images/illustration/2026-08-01/reasoning_modes_tokens.svg)

![Reasoning tokens echoed back per session, by policy](/images/illustration/2026-08-01/reasoning_tokens_by_mode.svg)

**读图结论：**

- **回放确实提高命中率**：always 把 median 从 94.3% 抬到 97.1%（+2.8 点），P10/P90 跨度反而收窄到 0.5 点。reasoning 进入前缀后，前缀更大、更连续，每个请求的可复用部分更多。
- **但代价是 prompt 体积暴涨**：always 的 prompt 总量是 never 的 **1.55 倍**（2.2M vs 1.4M），reasoning token 从 3,018 涨到 9,830（**3.3 倍**）。这些 token 虽然大部分能缓存（缓存价 ~20%），但**首次进入和每次新回合的思考都会造成一次冷写入**，且缓存写入本身按 miss 计费。
- **fresh 反而下降**（85k → 62k）：reasoning 占满了本该"浪费"的冷区间，把缓存未命中填满了——这是命中率数字上升的直接原因，但它买的是"把更多的字节变成缓存"，不是"省钱"。
- **tool 策略是生产默认，也是折中**：命中率 +2.1 点、reasoning 只多 +46%，prompt 只 +13%，MAE 还从 6.4 压到 3.7。它只把"决定要调工具的回合"的思考保留下来，其余照旧剥离——这是设计上的最小成本换取最大收益的默认值。

结论：**回放策略是一个"命中率 vs 总字节"的旋钮**：命中率好看不代表成本低。never 的 prompt 总量最小、fresh 最高；always 命中率最高但把 prompt 放大 1.55 倍；生产默认的 tool 落在中间，用 13% 的额外字节换 2.1 点的命中率。三个策略的 MAE 都能压到 3.7–6.4 点，说明离线模型对三种布局的预测都成立——离线预测用的是"剥离后"的字节（`drop_thinking=True`），它对 tool 布局依然贴切，因为纯回答回合的思考本就不该在前缀里。

### 6.5 单 session 内的累积成本结构

![Cumulative prompt tokens over a 12-turn session](/images/illustration/2026-08-01/cumulative_cache_growth.svg)

把 [§6.2](#62-在线实测-12-轮对话2026-08-06deepseek-官方-api) 那个 12 轮 session（22 次重复批次中的主 run，37 个请求）的 prompt token 逐请求累积：**"served from cache" 线几乎以 45° 角爬满整张图（最终 1.64M），"billed fresh" 线则基本保持水平（最终 ~92k）**。这两条线的形状就是整个评测最想展示的一句话——在 two-zone + append-only 布局下，上下文增长的成本几乎全部被缓存吸收，真正按全价计费的新鲜字节只占 5.3%。

---

## 7. 解读指标与图表

### 7.1 命中率对照图：离线预测 vs 服务器实测

![Measured prefix-cache hit rate, live](/images/illustration/2026-08-01/live_hit_rate.svg)

两条线：
- **predicted reusable prefix (offline)** = 离线预测的每个请求的前缀可复用比
- **cached_tokens / prompt_tokens (server)** = 服务器实测的命中率

两条线应高度重合。离线预测追踪服务器实测的 MAE 在 5.5 点以内（改前 7.0）。如果服务器线显著低于预测线（且 idle 不长），说明 tokenizer 或编码器有偏差；如果时高时低，说明 provider 不稳定。

### 7.2 Prompt Token 成分图：缓存命中的数据 vs 新鲜计费

![Live session prompt tokens: served from cache vs billed fresh](/images/illustration/2026-08-01/live_tokens_split.svg)

- **绿色** = 已缓存（按缓存价计费，约为原价 20%）
- **灰色** = 新鲜 token（按全价计费）

**柱子越矮越好**（总 token 少），**绿色比例越高越好**（缓存省成本）。一个 12 轮 session 总计 ~1.7M prompt token，其中 1.64M 来自缓存，仅有 0.09M 新鲜计费。

### 7.3 上下文设计对比图：四种变体的可复用前缀

![Reusable prefix per request, by context design](/images/illustration/2026-08-01/reusable_prefix.svg)

四条线叠加在同一图上。**as-built 线最后画**（z-order 最高），如果交叉点被 as-built 覆盖住，说明 as-built 是 winner。

### 7.4 成本对比图

![Session prompt cost](/images/illustration/2026-08-01/prompt_cost.svg)

每个 context design 的总额以美元为单位。同一组 session 跑的，柱子之间的差异完全是 context design 的结构差异造成的。as-built 只需 $0.020，no-ledger $0.025（+23%），three-zone $0.053（+162%）。

### 7.5 多 session 分布

`live_acceptance.json` 中 `--repeat N` 后在 `main_aggregate` 块中：

```json
{"session_hit_rate": {"p10": 0.937, "median": 0.947, "p90": 0.953}}
```

- **P10 和 P90 跨度小**（<3 点）→ provider 稳定，设计验证可信
- **P10 和 P90 跨度大**（>10 点）→ provider 的 KV cache 不稳定，换 endpoint 或缩小上下文

aggregate 的细粒度版本是 **[`live_hit_rate_band.svg`](/images/illustration/2026-08-01/live_hit_rate_band.svg)**（[§6.2](#62-在线实测-12-轮对话2026-08-06deepseek-官方-api)）：逐请求画出 median 线并带 P10/P90 带宽。读法：

- **带宽窄且 median 高**（本报告：请求 21 之后带宽 <8 点、median 96–100%）→ 命中率是设计行为，不是运气。
- **带宽宽的请求**集中在 session 前 1/3（请求 1–12，最高 45 点）→ 冷启动 + 新 turn 文件内容在不同重复间的错位对齐，是"轮到新内容"的请求，不是 provider 稳定性问题。

### 7.6 一致性检验（`accounting_consistent`）

`live_acceptance.json` 中每个 per-request 有 `accounting_consistent` 字段。如果为 `false`，说明服务器报告的 `cached_tokens + miss_tokens ≠ prompt_tokens`。这通常在报表最后作为 warning 提示。

### 7.7 时间戳与驱逐证据（`idle_s`）

`live_acceptance.txt` 报告的表格里每行有一个 `idle` 列。当一个低命中率请求出现时，看它的 idle 值：

- **idle 长（>60s）** → 大概率是 provider 驱逐了缓存
- **idle 短（<5s）** → 不是驱逐，应该回溯到 `wire_text` 模型或 server 端 bug

### 7.8 对话长度对比图

![Session cache hit rate by conversation length](/images/illustration/2026-08-01/turns_comparison.svg)

三条曲线（P10 / median / P90）在 6 → 12 → 18 turns 上整体上移，12 → 18 趋平。**看三点**：

1. **6 turns 落后**：前缀还没攒够就结束了，median 低且 P10 明显掉队。
2. **12→18 turns 趋平**：P10/median/P90 在 12 轮和 18 轮之间只差不到 0.4 点，缓存效果在 12 轮已饱和。
3. **18 turns 的 P10–P90 跨度收窄到 0.6 点**：对话越长，冷启动占比越小，provider 抖动被进一步稀释——分布宽度本身就是"缓存稳定性随长度改善"的证据。

### 7.9 推理回放策略对比图

![Session cache hit rate by reasoning echo policy](/images/illustration/2026-08-01/reasoning_modes_hit_rate.svg)

柱子是三种 `--reasoning-strip` 策略的命中率中位数。**回放越多命中率越高，但柱子的"高度"和它的成本不成正比**——高度只反映命中率，成本要结合 [`reasoning_modes_tokens.svg`](/images/illustration/2026-08-01/reasoning_modes_tokens.svg) 里的柱高（总 prompt 量）和 [`reasoning_tokens_by_mode.svg`](/images/illustration/2026-08-01/reasoning_tokens_by_mode.svg)（每 session 回放的思考 token）一起看。命中率最高（always）的方案把 prompt 总量放大了 1.55 倍，正是 [§3.4](#34-reasoning-控制回放-默认只保留工具回合的思考) 说"不值得"的做法。

### 7.10 累积成本结构图

![Cumulative prompt tokens over a 12-turn session](/images/illustration/2026-08-01/cumulative_cache_growth.svg)

一条近似 45° 的缓存线和一条近乎水平的 fresh 线。读法：

- **缓存线斜率 = 上下文增长速度**，对话越长越陡，但对应的是缓存价（~20%）。
- **fresh 线几乎不动** → 新鲜计费字节不随对话增长，成本被前缀稳定性"锁死"在低位。
- **两条线之间的垂直距离**就是该设计省下的钱，随轮数单调扩大。

---

## 8. 结论

WhalePod 的 two-zone + append-only + context-ledger 设计在 DeepSeek V4 官方 API 上通过了统计学意义上的验证：

1. **12 轮对话命中率中位数 94.7%**（22 次独立 session，P10=93.7%，P90=95.3%），P10–P90 跨度仅 1.6 点，性能一致可复现。
2. **不同长度对话均稳健**：6 轮 91.7%，12 轮 94.7%，18 轮 95.0%——对话越长缓存效果越好，且新鲜计费 token 不随长度增长（始终 ~60–90k），12 轮是性价比甜点。
3. **离线预测精度的提升证明改进有效**：用官方 V4 BPE tokenizer 替代 cl100k_base/heuristic，用官方 `encode_messages` 替代手搓 `json.dumps`，MAE 从 7.0 降到 5.5 点。
4. **推理回放策略验证了生产默认值**：never/tool/always 三种策略的命中率分别为 94.3%/96.4%/97.1%，prompt 总量 1.42M/1.60M/2.20M——回放是个"命中率 vs 总字节"的旋钮，生产默认的 tool（仅工具轮回放）用 13% 的额外字节换 2.1 点命中率，always 则把 prompt 放大 1.55 倍只多换 0.7 点，已无性价比。
5. **架构设计经得起检验**：两区存储 + ledger + reasoning 控制回放 + compaction 七项决策的叠加效果在真实场景中持续兑现 90%+ 缓存命中率，不靠运气，不靠调参，靠的是字节级前缀稳定性的系统性保证。
