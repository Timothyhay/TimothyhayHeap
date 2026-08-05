# WhalePod KVCache 评测技术报告

WhalePod 的上下文管理基于一个核心断言：**deepseek-v4 的自动前缀缓存能在长对话中持续生效**，前提是请求的字节前缀保持稳定。本报告详述如何离线预测和在线验证这个断言，涵盖评测架构、代码文件解析、完整实验流程和真实数据结果。

---

## 目录

1. [评测目标](#1-评测目标)
2. [代码文件](#2-代码文件)
3. [Agent 架构设计：为何前缀缓存能命中](#3-agent-架构设计为何前缀缓存能命中)
4. [评测架构](#4-评测架构)
5. [实验流程](#5-实验流程)
6. [实验结果](#6-实验结果)
7. [解读指标与图表](#7-解读指标与图表)

---

## 1. 评测目标

KV cache 的命中率是一个成本指标，不是一个正确性指标。我们追求的不是"命中率高"，而是"在保证功能完整的前提下命中率高的**同时** prompt token 总量**低**"——命中率一样但灌入了过多 token 的设计同样不省钱。

两条轨道：

| 轨道 | 环境 | 测量对象 | 输出 |
|:---|:---|:---|:---|
| **离线 (`validate.py`)** | 零网络、零 API key | 请求之间的字节前缀可复用比（理论上限） | 四种上下文设计的对比 |
| **在线 (`live_acceptance.py`)** | 真实服务器 | 服务器报告的 `prompt_cache_hit_tokens`（实际命中） | 命中率验证 + 预测一致性 |

离线说"最多应该有多少复用"；在线说"服务器确实给我们多少复用"。两者之间的差距（MAE）衡量我们的理解与服务器行为的一致程度。

---

## 2. 代码文件

评测系统在 `bench/` 目录下，每个文件职责清晰：

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
| `variants-small` | 自动触发 | 同上，但窗口缩小到 ~52k，迫使所有设计都 prune |
| `prune` | 默认 | as-built 在 52k 窗口下观察 prune 后前缀恢复曲线 |
| `repomap` | 默认 | repo map 在不同 token budget 下的实际渲染 token 数 |
| `deny` | 默认 | sandbox=`--yes` 模式下危险命令的拒绝链 |

### `bench/live_acceptance.py` — 在线实测引擎

对真实服务器发请求，读取 `usage` 块中的缓存字段：

- **DeepSeek API / vLLM:** `prompt_cache_hit_tokens` + `prompt_cache_miss_tokens`
- **OpenAI:** `prompt_tokens_details.cached_tokens`
- **OpenRouter:** `prompt_tokens_details.cached_tokens` + `cache_write_tokens`
- **Anthropic:** `cache_read_input_tokens`

核心类：

- **`RecordingEndpoint`**（继承 `OpenAIChatEndpoint`）：在真实端点外面包了一层记录。每个请求都保存 `payload`（发出的 JSON）、`raw_usage`（服务器返回的 usage 块）、`t0`/`t1` 时间戳。payload 会深拷贝（Agent 在请求之间修改消息列表，前缀对比需要的是发出时的原始字节）。
- **`LiveRequest`**：单次请求的全部观测数据，包含 `prompt_tokens`、`cached_tokens`、`miss_tokens`、`hit_rate`、`predicted_reusable`、`idle_s`（和上一个请求之间的空闲秒数）、`accounting_consistent`（hit + miss 是否等于 prompt）。
- **`LiveRun`**：一个完整 session。除了 per-request 数据，还有 session 级别的 `hit_rate`、`prompt_cost`、`prompt_saving`（省了多少钱），以及 `prunes` 事件列表。
- **`aggregate_runs(runs)`**：输入多个 `LiveRun`，按请求索引归并，输出 per-request 的 `median/P10/P90` 以及 session-level 的分布。

三个子实验：

| 选项 | 测量什么 |
|:---|:---|
| `--only main` | 12 轮 session，`--repeat N` 跑 N 次取分布 |
| `--only prune` | 同上，但窗口缩小到 ~52k（强制 prune） |
| `--only pin-check` | Provider affinity A/B：同一 prefix，pin 到 provider vs 自由路由 |

### `bench/dsv4_encoding.py` — DeepSeek V4 官方消息编码器

从 HuggingFace `deepseek-ai/DeepSeek-V4-Flash-0731` vendor 来的，MIT 许可。独立于任何框架（无 `transformers` 依赖），完全自包含。对外接口：

- **`encode_messages(messages, thinking_mode, drop_thinking, reasoning_effort)`**：把 OpenAI 格式的消息列表转为模型需要的输入字符串（BOS + 角色分隔符 + tool schema 渲染 + tool-result 合并）。
- **`merge_tool_messages(messages)`**：把 `role="tool"` 的消息合并到前一个 user message 中（V4 不支持独立的 tool 角色）。
- **`parse_message_from_completion_text(text)`**：从模型输出的字符串中还原 `reasoning_content` / `content` / `tool_calls`。

`wire_text` 调用 `encode_messages` 而非自造的 JSON dump，使得离线测量的字节流和服务器实际 tokenize 的完全一致。

### `bench/charts.py` — 图表引擎

零额外依赖（纯 Python stdlib），同时输出 ASCII（终端）和 SVG（文档/README）。支持三种图表：折线图 `svg_lines`、柱状图 `svg_bars`、堆叠柱状图 `svg_stacked`。都支持高亮、标注（marker）、副标题。

### `bench/fetch_tokenizer.py` — Tokenizer 下载器

一次性工具。从 HuggingFace 公开仓库下载 `deepseek-ai/DeepSeek-V4-Flash-0731` 的 `tokenizer.json` 到 `~/.whalepod/tokenizers/deepseek-v4-flash/`。纯 stdlib，不需要 token（公开模型权重没开放，但 tokenizer 文件是公开的）。之后 `whalepod.core.tokenizer` 检测到文件存在即自动加载为 V4 BPE 编码器。

### `whalepod/core/tokenizer.py` — 运行时 Token 估计器

三级回退：
1. **DeepSeek V4 BPE** — 检测 `tokenizer.json` 存在即用（`tokenizers.Tokenizer.from_file`）
2. **tiktoken cl100k_base** — 对 OpenAI 兼容模型
3. **字符启发式** — CJK ~1 token/char，Latin ~3.2 chars/token

评测跑的时候 tokenizer 自动升级到 1；普通用户没有下载 tokenizer.json 时会落在 2 或 3。接口是 `estimate_tokens(text)`，整个 Agent、MessageManager、bench 全路径都从这里走。

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

### 3.1 两区消息存储 — 不可变前缀 + 只追加历史

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

**为什么重要：** KV cache 以最长公共前缀为 key。Zone 1 在全部请求中不变，Zone 2 只追加不修改。所以每个新请求的前缀都等于"Zone 1 + Zone 2 的已发送部分"，和上一个请求的前缀完全重叠。server 不会因为"中间某个消息被改了内容"而触发冷 miss。

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

### 3.4 Reasoning 不上传 — 永不污染前缀

DeepSeek 明确要求 chain-of-thought 不能回放到后续请求中。WhalePod 的 `VLLMEndpoint._encode_message` 里 `m.reasoning` 被故意跳过，不写入 payload。只保留在本地 UI 中显示。

**为什么重要：** 一次思考动辄几百上千 token。如果不丢掉，每个 assistant 回复带着一大段思考回到前缀里——第一第二次能缓存，但到第三个 turn 的时候整个区域都是几十 K 的推理文本，完全不可复用。剥离后，assistant 回复体量很小。

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

- **[`reusable_prefix.svg`](../bench/results/reusable_prefix.svg)** — 四种设计的 per-request 可复用前缀折线（1M 窗口）
- **[`reusable_prefix_small_window.svg`](../bench/results/reusable_prefix_small_window.svg)** — 同上，52k 窗口（强迫 prune）
- **[`prompt_tokens_split.svg`](../bench/results/prompt_tokens_split.svg)** — 四种设计的 prompt token 堆叠图（绿色 = 可复用，灰色 = 新鲜）
- **[`prompt_cost.svg`](../bench/results/prompt_cost.svg)** — 四种设计的提示词成本对比
- **[`prompt_cost_small_window.svg`](../bench/results/prompt_cost_small_window.svg)** — 同上，52k 窗口
- **[`prune_recovery.svg`](../bench/results/prune_recovery.svg)** — as-built 在 52k 窗口下，prune 后前缀恢复曲线（▼ 标记 prune 点）
- **[`repo_map_budget.svg`](../bench/results/repo_map_budget.svg)** — repo map 在不同 token budget 下的渲染大小

### 5.3 在线实测

需要 API key（通过环境变量，不会落地到文件）。

```bash
# PowerShell
$env:WHALEPOD_API_KEY = "sk-..."

# 主 session + prune session + provider affinity A/B (+ pin-check)
python bench/live_acceptance.py

# 大规模重复跑（用 --repeat N 拿到多 session 统计分布）
python bench/live_acceptance.py --repeat 10

# 直连 DeepSeek 官方 API（不用 OpenRouter）
python bench/live_acceptance.py `
    --base-url "https://api.deepseek.com" `
    --model "deepseek-chat" `
    --no-pin `
    --repeat 5
```

生成的 SVG 图表：

- **[`live_hit_rate.svg`](../bench/results/live_hit_rate.svg)** — 实测命中率 vs 离线预测（run 1）；如果 `--only prune` 也会展示 prune 恢复曲线
- **[`live_tokens_split.svg`](../bench/results/live_tokens_split.svg)** — 单个 session 的 cached vs fresh 堆叠图
- **[`live_prune.svg`](../bench/results/live_prune.svg)** — prune 窗口下的命中率恢复曲线
- **[`live_provider_affinity.svg`](../bench/results/live_provider_affinity.svg)** — Pin vs Unpin A/B 测试（仅 `OpenRouter`）

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
  "session_hit_rate": {"p10": 0.932, "median": 0.950, "p90": 0.955},
  "session_prompt_cost_usd": {"p10": ..., "median": ..., "p90": ...},
  "per_request": [
    {"request": 1, "n": 5, "median": 1.0, "p10": 0.84, "p90": 1.0},
    ...
  ]
}
```

**规模建议：**

| 场景 | 推荐 N | 时间 | 预算 |
|:---|:---|:---|:---|
| CI 冒烟测试 | 1 | ~2 min | ~$0.02 |
| 内部验证 | 5 | ~10 min | ~$0.10 |
| 发布前验证 | 10 | ~20 min | ~$0.20 |

---

## 6. 实验结果

### 6.1 离线验证（2026-08-05，V4 BPE tokenizer）

| Variant | Prompt tokens | Reusable | Reuse % | Prompt cost |
|:---| ---:| ---:| ---:| ---:|
| **as-built** | 911,083 | 858,240 | **94.2%** | $0.020 |
| no-ledger | 1,099,805 | 1,030,528 | 93.7% | $0.025 |
| rolling-summary | 518,504 | 465,216 | 89.7% | $0.013 |
| three-zone | 913,223 | 405,312 | 44.4% | $0.053 |

![Reusable prefix per request, by context design](../bench/results/reusable_prefix.svg)

- **as-built** 在每个 turn 之间保持 99.6% 的可复用前缀，只有新文件内容进入时才跌到 76–89%。
- **no-ledger** 的可复用前缀率和 as-built 差不多（93.7% vs 94.2%），但灌入了 **20% 更多的 token**（~1.1M vs ~911k）。前缀可复用只是一个比率，token 总量少才是省钱的根本。
- **rolling-summary** 每次总结时丢掉一大段历史，前缀无可复用基础，只能靠压缩后体积小来减成本——但在实践中，每次总结会产生一个全新的 prompt（没有前缀复用），而 as-built 靠缓存大量 token 以 5x 折扣价（缓存 token 的 20% 计费）。
- **three-zone** 把"当前文件内容"放在历史后面，每个请求都有尾缀变动，前缀复用持续下降到 44.4%——这个设计在看过图表后就没写出来。

![Session prompt tokens: reusable prefix vs freshly billed](../bench/results/prompt_tokens_split.svg)

堆叠图清晰地看到：as-built 的柱子最矮（总 token 最少），且绿色（可复用）比例最高。no-ledger 柱子更高（更贵），three-zone 灰色（新鲜）部分最多（最贵）。

![Session prompt cost](../bench/results/prompt_cost.svg)

成本对比：as-built **$0.020**，no-ledger **$0.025**（+23%），three-zone **$0.053**（+162%）。rolling-summary 的柱子矮是因为它压缩了上下文所以 token 总量少，但这是以牺牲信息为代价换来的。

![Prune recovery: 52k window forced](../bench/results/prune_recovery.svg)

在小窗口下，prune 后的前缀恢复是瞬时的：一次 prune 掉到 ~0%（红色虚线标出），下一个请求立即恢复到 ~90%，session 整体仍保持 88.8% 的可复用比。这就是 prune 的全部代价。

### 6.2 在线实测（2026-08-05，DeepSeek 官方 API）

配置：`https://api.deepseek.com`，`deepseek-chat`（V4-Flash），5 次重复，12 turns。

| Metric | Result |
|:---|---|
| **Sessions** | 5 × 12 turns |
| **Requests per session** | 34（中位数） |
| **Session 命中率** | **median 95.0%** (P10=93.2%, P90=95.5%) |
| **Session prompt tokens** | 1,541,696（1,436,544 cached, 105,152 fresh） |
| **预热曲线** | 前 3 请求综合平均 55.8% → 后 3 请求综合平均 98.2% |
| **离线 vs 实测 MAE** | **2.9 点**（改前 7.0） |
| **ledger 省量** | 3,381 tokens（1 次去重） |

![Measured prefix-cache hit rate, live](../bench/results/live_hit_rate.svg)

前三个请求命中率几乎为零：缓存还是冷的，nothing to hit。从第 4 个请求开始稳定在 89–99.7%，30 of 34 个请求超过 90%。每当新内容进入窗口时（新的 turn 开始），命中率暂时跌到 55–86%，然后在**下一个**请求立即恢复到 94–99%。

![Live session prompt tokens: served from cache vs billed fresh](../bench/results/live_tokens_split.svg)

1,541,696 prompt tokens 里，1,436,544 来自缓存，只有 105,152 是新鲜计费的。如果全量按新 token 算，一个 session 就是 ~$0.14（DeepSeek V4 Flash 价格）。实际缓存后 ~$0.02——便宜了 85%。

**和之前的对比（OpenRouter → DeepInfra）：**

| Metric | DeepInfra（前） | DeepSeek API（今） |
|:---|---:|:---|
| 命中率 | 88.8% | 93.2% |
| MAE | 7.0 点 | 2.9 点 |
| 随机坍塌 | #16 1.7%, #24 74.9% | 无 |
| tokenizer | heuristic | V4 BPE |

DeepSeek 自己的 API 比通过聚合器走 DeepInfra 稳定得多：命中率 +4.4%，预测误差 -59%，且不再出现无法解释的突然坍塌（之前 #16 和 #24 的 collapse 很可能是 provider 驱逐造成的）。

### 6.3 多 session 分布

命中率中位数 **95.0%**，P10=93.2%，P90=95.5%。P10 到 P90 跨度仅 2.3 点，说明评测结果**可复现**，不是一次偶然。

Per-request 中位数表（5 次 run）：

```text
req  1  ███████████████████████████████████████████████████████████  100%
req  2  ██████████████████████████████████████                  66%
...
req 33  █████████████████████████████████████████████████████████    96%
req 34  █████████████████████████████████████████████████████████    96%
```

- 请求 #1：每次 run 都是 100% 命中——因为第一轮和上一轮的最后一轮共享了同一个初始前缀（DeepSeek 的 KV cache 在 session 之间没有立即驱逐）。
- 请求 #2 降到中位数 66%——因为新一轮的第一次工具调用引入了新的工具结果（不同于上一轮的对应位置）。
- 请求 #7–#34：所有请求都在 89–100% 之间。

---

## 7. 解读指标与图表

### `reusable_frac`（离线） vs `hit_rate`（在线）

- 两者之间的 **MAE** 是离线模型的质量指标。**MAE 越小，离线预测越能用**。实测 MAE 从 7.0 降到 2.9 的原因是：(a) tokenizer 从 `cl100k_base`/heuristic 换成了 V4 官方 BPE；(b) `wire_text` 从 `json.dumps` 换成了官方 `encode_messages`。
- 差距大（>5 点）说明 (a) 线上的 chat template 和我们 vendor 的 `dsv4_encoding.py` 有差异；(b) provider 端驱逐了缓存。

### 命中率折线图（`live_hit_rate.svg`）

两条线：
- **predicted reusable prefix (offline)** = 离线预测的每个请求的前缀可复用比
- **cached_tokens / prompt_tokens (server)** = 服务器实测的命中率

两条线应高度重合。如果服务器线显著低于预测线（且 idle 不长），说明模型或编码不匹配。如果时高时低，说明 provider 不稳定。

### 堆叠图（`prompt_tokens_split.svg` / `live_tokens_split.svg`）

- 绿色 = 已缓存（按缓存价计费，约为原价 20%）
- 灰色 = 新鲜 token（按全价计费）

**柱子越矮越好**（总 token 少），**绿色比例越高越好**（缓存省成本）。

### 成本图（`prompt_cost.svg`）

每个 context design 的总额以美元为单位。同一组 session 跑的，柱子之间的差异完全是 context design 的结构差异造成的。

### 可复用前缀图（`reusable_prefix.svg`）

四条线叠加在同一图上。**as-built 线最后画**（z-order 最高），如果交叉点被 as-built 覆盖住，说明 as-built 是 winner。

### Prune 恢复图（`prune_recovery.svg`）

一条线，橙色 marker 标注 prune 发生的位置。prune 之后缓存完全失效（前缀更迭），下一个请求重新建立。重要的是观察**恢复速度**：一个 prune 后下一个请求能恢复到什么水平。实测数据中一个 prune 后立即恢复到 ~90%。

### 多 session 分布图（`live_acceptance.json → main_aggregate`）

```json
{"session_hit_rate": {"p10": 0.932, "median": 0.950, "p90": 0.955}}
```

- **P10 和 P90 跨度小**（<5 点）→ provider 稳定，设计验证可信
- **P10 和 P90 跨度大**（>10 点）→ provider 的 KV cache 不稳定，换个 endpoint 或降低上下文大小试试

### 一致性检验（`accounting_consistent`）

`live_acceptance.json` 中每个 per-request 有 `accounting_consistent` 字段。如果为 `false`，说明服务器报告的 `cached_tokens + miss_tokens ≠ prompt_tokens`。这通常在报表最后作为 warning 提示。

### 时间戳与驱逐证据（`idle_s`）

`live_acceptance.txt` 报告的表格里每行有一个 `idle` 列。当一个低命中率请求出现时，看它的 idle 值：

- **idle 长（>60s）** → 大概率是 provider 驱逐了缓存
- **idle 短（<5s）** → 不是驱逐，应该回溯到 `wire_text` 模型或 server 端 bug

---

> **原始数据**: `bench/results/` 目录下所有 `.json` 和 `.txt` 文件。
> 
> **补充阅读**: 改进的 changelog 见 [`BENCH_EVAL_IMPROVEMENTS.md`](../bench/BENCH_EVAL_IMPROVEMENTS.md)。
> 
> **复现**: `python bench/fetch_tokenizer.py && python bench/validate.py`
