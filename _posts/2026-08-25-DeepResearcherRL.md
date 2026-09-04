---
layout: modern-article
title: LLM Agentic RL w/ hand-made vanilla Deep Researcher Agent
tags: LLM
comments: true
---

本文记录了在 8 卡 A100 节点上，基于 **veRL (Ray + vLLM + FSDP)** 框架，对大语言模型进行 Multi-turn Agent 强化学习训练的实践，
以及后续放缩到 910B 集群的迁移方案。

# 1. 我们为何出发

[书接上回](/2026/08/21/DeepResearcher.html)，我们从第一性原理出发，手搓了一个架构和工具都保持简单的 ReAct DeepResearch Agent。

我们的最终目标是训练其中的 Researcher Agent，让他具备在复杂、模糊、需要多步推理的问题中自主、高效使用本地 search 工具的能力。
同时能过滤掉冗余信息、自动纠正检索方向，并能最终获得引用正式的高准确度回答。

接下来我们谈从问题建模、训练环境搭建开始，如何从单节点到多卡集群构建自定义 Agent 的训练。

## 1.1 MDP 形式化建模

在 Agentic RL 中，我们将多轮 ReAct 交互建模为一个步长有限的 MDP：

* **状态空间 $S$**：当前轮次之前的完整对话历史，包括初始 $Prompt$、历史思考过程 $Thought_t$、历史动作 $Action_t$（搜索） 和环境反馈 $Observation_t$。
* **动作空间 $A$**：模型在当前步骤生成的 Token 序列，think + tool_call(search)：
  1. **内部动作（思考）**：`<think>...</think>` 内的推理文本。
  2. **外部动作（工具调用）**：触发环境反馈的标签，如 `<call:search>query</call:search>`。
* **状态转移 $P(S_{t+1}|S_t, a_t)$**：当模型生成外部动作时，生成被暂停、检索服务执行(Action)，将返回的 $Observation_t$ `<observation>...</observation>`拼接到 $S_t$ 后，形成下一轮的状态 $S_{t+1}$。
* **奖励 $R(S, A)$**：环境在模型输出最终答案 `<answer>...</answer>` 或达到最大步数时，对整条轨迹进行综合打分。

> Anchor: 这跟单轮 RLHF 有何不同？
> 状态在转移中被环境注入了非模型生成的 token（observation），因此必须做 loss masking；且奖励是轨迹级稀疏信号，credit assignment 更难。

# 2. Enviroment

Use veRL (Ray + vLLM + FSDP) to run Agentic RL training (GRPO/DAPO) on Qwen2.5-Coder-14B-Instruct to improve multi-hop QA performance on HotpotQA.

**Testing Hardware**: 8× NVIDIA A100 80GB PCIe, 503GB RAM, 44TB disk

然后放缩到 64卡 910B集群。

**Key Software Versions**

- Python: 3.10.9 (Anaconda)
- PyTorch: 2.6.0
- vLLM: 0.8.5.post1
- Ray: 2.49.1
- Transformers: 4.51.3
- FAISS: 1.7.4 (CPU)

## Architecture Diagram

```
┌────────────────────────────────────────────────────────────┐
│                     veRL Training Loop                     │
│                                                            │
│  ┌──────────┐   ┌──────────┐   ┌────────────┐              │
│  │  Actor   │   │   vLLM   │   │  Reference │              │
│  │ (FSDP)   │   │ Rollout  │   │   Model    │              │
│  │  GPUs    │   │  GPUs    │   │  (FSDP)    │              │
│  └────┬─────┘   └────┬─────┘   └─────┬──────┘              │
│       │              │               │                     │
│       │     ┌────────▼────────┐      │                     │
│       │     │  Agent Loop     │      │                     │
│       │     │  (verltool_     │      │                     │
│       │     │   agent_loop)   │      │                     │
│       │     └────────┬────────┘      │                     │
│       │              │               │                     │
│       │     ┌────────▼────────┐      │                     │
│       │     │  Tool Server    │      │                     │
│       │     │  wiki_search    │      │                     │
│       │     │  (Ray Workers)  │      │                     │
│       │     └────────┬────────┘      │                     │
│       │              │               │                     │
│       │     ┌────────▼────────┐      │                     │
│       │     │  Wikipedia API  │      │                     │
│       │     │  (Internet)     │      │                     │
│       │     └─────────────────┘      │                     │
│       │                              │                     │
│  ┌────▼──────────────────────────────▼──────┐              │
│  │          Reward Manager                  │              │
│  │       search_r1_qa_em (EM)               │              │
│  │       extraction + normalize             │              │
│  └──────────────────────┬───────────────────┘              │
│                         │                                  │
│                    ┌────▼─────┐                            │
│                    │  GRPO /  │                            │
│                    │  DAPO    │                            │
│                    │  Update  │                            │
│                    └──────────┘                            │
└────────────────────────────────────────────────────────────┘
```

## 工具与环境交互

我们使用 `verl-tool` (TIGER-AI-Lab/verl-tool with veRL submodule 来实现自定义 Agent 和工具与 veRL 的无缝接入。

**Tool architecture**:

```
┌─────────────────┐     no proxy     ┌──────────────┐     proxy + verify=False     ┌─────────────┐
│  veRL Training  │ ───────────────→ │  Tool Server │ ──────────────────────────→ │  Wikipedia  │
│  Process        │  (127.0.0.1:xxx) │  (localhost) │   (hkgpqwg00206:8080)        │  API        │
└─────────────────┘                  └──────────────┘                              └─────────────┘
```

### 为什么需要 verl-tool？

veRL 本身是通用 RL 训练框架（PPO/GRPO/DAPO + FSDP + vLLM），
但它**不支持 Agent 的多轮工具调用**。原生 veRL 的 rollout 是一次性生成文本，
不会中途暂停去调用外部工具再继续生成。

verl-tool 在 veRL 基础上增加了三层关键能力：

| 能力 | veRL 原生 | verl-tool |
|------|----------|-----------|
| 多轮 Agent Loop | ❌ 一次性生成 | ✅ 检测 action token → 调用工具 → 拼接 observation → 继续生成 |
| 工具注册与管理 | ❌ 无 | ✅ 插件式工具系统，注册即用 |
| 工具服务器 | ❌ 无 | ✅ HTTP 服务器，解耦推理与工具调用 |

**核心架构**：

```
veRL Training Loop
    └── vLLM Rollout (生成)
         ├── 检测到  → 暂停生成
         ├── POST /get_observation → Tool Server → Wikipedia API
         ├── 收到 ...
         └── 继续生成 → 检测到  → Episode 结束
```

### 我们如何实现自己的 Agent 和工具

verl-tool 的工具系统通过 `BaseTool` 基类和 `@register_tool` 装饰器实现插件式注册。

#### 注册一个新工具只需 3 步：

**第 1 步**：在 `verl_tool/servers/tools/` 下创建 Python 文件

```python
from .base import BaseTool, register_tool

@register_tool
class WikiSearchTool(BaseTool):
    tool_type = "wiki_search"  # 工具的唯一标识符
```

**第 2 步**：实现 3 个核心方法：

```python
def get_usage_inst(self) -> str:
    """告诉模型如何使用这个工具。会被注入到 system prompt 附近。"""
    return "Search Wikipedia with your query"

def parse_action(self, action: str) -> Tuple[str, bool]:
    """从模型输出中解析动作。
    输入: 模型的原始输出文本
    输出: (解析后的查询, 是否有效)"""
    if "" in action:
        query = extract_between_tags(action, "", "")
        return query, True
    if "" in action:
        answer = extract_between_tags(action, "", "")
        return answer, True
    return "", False

def conduct_action(self, trajectory_id, action, extra_field):
    """执行工具调用，返回 observation。
    这是外部 API 调用的地方。"""
    query, is_valid = self.parse_action(action)
    if "" in action:
        return "", done=True, valid=True   # 结束 episode
    result = search_wikipedia(query)       # 调用 Wikipedia API
    return f"{result}", done=False, valid=True
```

**第 3 步**：启动工具服务器时指定工具类型

```bash
python -m verl_tool.servers.serve --tool_type "wiki_search"
```

框架自动扫描 `tools/` 目录下的所有 `.py` 文件，
发现 `@register_tool` 装饰的类，实例化并注册到路由表中。

#### 现有工具一览

verl-tool 预置了 15+ 工具，涵盖搜索、代码执行、SQL、文件浏览等：

| 工具 | 用途 |
|------|------|
| `google_search` | Google 搜索（需 API key） |
| `bing_search` | Bing 搜索（需 API key） |
| `search_retrieval` | 本地向量库检索（需 FAISS） |
| **`wiki_search`** | **Wikipedia 搜索（免费，我们实现的）** |
| `python_code` | Python 代码执行 |
| `ipython_code` | IPython 交互式执行 |
| `bash_terminal` | Bash 终端 |
| `sql` | SQL 查询 |
| `finish` | 标记任务完成 |

### Agent Loop 的多轮交互机制

verltool_agent_loop.py`：

```
Step 1: 初始化
  prompt = chat_template(system_prompt + question)
  conversation = [prompt]

Step 2: 生成循环 (最多 max_turns 轮)
  while turn < max_turns:
      response = vllm.generate(conversation)
      # 检测 action_stop_tokens: "</search>" 或 "</answer>"

      if "</search>" in response:
          # 模型想搜索
          tool_result = POST tool_server(get_observation, action=response)
          conversation += response + tool_result  # 拼接 observation
          turn += 1
          continue

      elif "</answer>" in response:
          # 模型给出最终答案 → episode 结束
          conversation += response
          break

      else:
          # 模型没有使用任何 action token → 直接结束
          break

Step 3: 计算 reward
  reward = reward_manager(conversation, ground_truth)
```

**关键设计细节**：

1. **Action Stop Tokens**：`</search>` 和 `</answer>` 是 vLLM 的 stop tokens。
   模型生成到这些 token 时会**立即停止**，Agent Loop 接管控制权。
2. **Observation 拼接**：工具返回的 `<information>...</information>` 会被拼接到
   对话历史中，作为下一轮 vLLM 生成的输入。模型看到搜索结果后可以继续推理。
3. **Loss Masking**：`mask_observations=True` 确保 observation token（工具返回的内容）
   不参与 loss 计算。模型只需要学习"何时搜索"和"搜索后的答案"，不需要学习
   复述搜索结果的文本。
4. **多轨迹并发**：Agent Loop 支持异步并发处理多条轨迹，
   每条轨迹有独立的 `trajectory_id` 和对话历史。

### Tool Server 架构

Tool Server 是多进程 HTTP 服务，负责接收 Agent Loop 的工具调用请求：

```
┌─ Backend Worker 0 (wiki_search) ─┐
Agent Loop ──POST──→│  Router (FastAPI + uvicorn)       │──→ Wikipedia API
  (veRL)    ←──resp─│  /get_observation                 │←── <information>
                    └─ Backend Worker 1 (wiki_search) ──┘
```

- **Router**：负载均衡，基于 `trajectory_id` 的一致性哈希路由
- **Backend Workers**：每个 Worker 是独立子进程，运行实际工具逻辑
- **健康检查**：Router 在启动时等待所有 Worker 就绪
- **Ray 模式**：可选使用 Ray actors 替代线程池（但我们的环境用线程池）

### 我们的 wiki_search 工具的特殊处理

因为公司网络环境的特殊性，我们的工具做了以下适配：

1. **HTTPS 强制**：Wikipedia 库默认用 HTTP，但代理只转发 HTTPS → 强制改写 URL
2. **SSL 绕过**：代理有自签名证书 → `verify=False`
3. **代理保留**：需要代理才能访问外网 → `trust_env=True`
4. **限流防护**：Wikipedia 每 ~3 请求返回 429 `Retry-After: 13s` → 尊重该头并等待
5. **API 调用优化**：每个问题最多 2 次 API 调用（之前 4-6 次）

### 与 veRL 原生训练的关键区别

如果不用 verl-tool，直接用 veRL 做 Agent RL：

1. **无法实现多轮交互**：veRL 的 rollout 是一次性的，不能中途插入 tool observation
2. **需要自己管理对话状态**：每条轨迹的 history 需要手动拼接
3. **需要自己实现 tool serving**：工具调用逻辑直接写在 reward 函数中（不灵活）
4. **无法并发处理工具调用**：每个工具调用会阻塞 rollout

verl-tool 将这些复杂性封装为：

- `AgentLoopManager`：管理多轨迹并发生成 + 工具调用
- `ToolServer`：HTTP 微服务，工具逻辑独立部署
- `BaseTool`：统一的工具接口，注册即用

---

### Tool: `wiki_search`

**File**: `verl-tool-main/verl_tool/servers/tools/wiki_search.py`

A custom Wikipedia search tool registered in verl-tool's tool system:

- **Free**: No API keys required, uses public Wikipedia API
- **Search-R1 compatible**: Parses `<search>` tags, returns results in `<information>` tags
- **Answer extraction**: Parses `<answer>` tags for final answer
- **Caching**: In-memory cache with 10,000 entry limit

**Action format**:

```
<search> Albert Einstein </search>        # Search Wikipedia
<answer> 1879 </answer>                   # Final answer
```

**Observation format**:

```
<information>**Doc 1 (Title: Albert Einstein)**
URL: https://en.wikipedia.org/wiki/Albert_Einstein
Summary text...</information>
```

### Why Wikipedia API over other options

| Option | Pros | Cons |
|--------|------|------|
| `search_retrieval` (FAISS) | Realistic, matches Search-R1 | Requires 20GB+ Wikipedia corpus + FAISS index |
| `google_search` | Best results | Requires SERPER_API_KEY |
| `bing_search` | Good results | Requires BRIGHTDATA_API_KEY |
| **`wiki_search`** ✅ | Free, no API key, works for HotpotQA | Rate-limited, fewer results |

### RL Reward Shaping

Simple exact match (EM) reward:

- Extracts text from `<answer>` tags
- Normalizes (lowercase, remove articles, punctuation, whitespace)
- Compares with ground truth
- Returns 1.0 for match, 0.0 otherwise
- Penalty: `/4` if too many `<answer>` or `</answer>` tags (>10)

## 4. Dataset: 数据来源、格式与 Prompt 设计

### 4.1 数据来源

我们使用 **Search-R1 标准数据集** `PeterJinGo/nq_hotpotqa_train`，来自 Search-R1 论文
(https://github.com/PeterGriffinJin/Search-R1)。
同时包含基于内部构造的少量多跳问答语料。

**Search-R1** 的数据集将两个 QA 基准混合在一起：

| 数据源 | 数量 | 特点 |
|--------|------|------|
| **NQ** (Natural Questions) | 79,168 | 单跳事实型问答，答案通常是实体/数字 |
| **HotpotQA** | 90,447 | **多跳问答**，需要结合多个 Wikipedia 页面才能回答 |

**总计**：169,615 条原始数据 → 按 90/10 分割为 152,653 train / 16,962 val。

RAG_ProGuide 数据集（13,289 条，已存在于 `dataset/`）未使用，因为它缺少多跳推理需求。

### 4.2 数据格式

每条数据包含以下字段：

```python
{
    'id': 'train_0',
    'question': 'total number of death row inmates in the us?',
    'golden_answers': ['2,718'],           # 可接受的答案列表
    'data_source': 'nq',                    # 'nq' 或 'hotpotqa'
    'ability': 'fact-reasoning',
    'prompt': [                             # messages 格式
        {
            'role': 'user',
            'content': '<完整指令文本>...Question: <问题>'
        }
    ],
    'reward_model': {
        'ground_truth': {
            'target': ['2,718']             # 用于 EM 比对的正确答案
        },
        'style': 'rule'
    },
    'extra_info': {
        'index': 0,
        'split': 'train'
    }
}
```

**关键点**：此数据集**已经预格式化为 Search-R1 标准格式**，由 verl-tool 的
`examples/data_preprocess/search_r1.py` 脚本从原始 NQ/HotpotQA 转换而来。
我们的 `prepare_data.py` 仅做了 train/val 分割。

### 4.3 Prompt 设计（使用 Search-R1 标准）

我们参考了 Search-R1 论文的标准指令格式。
这是一个嵌入在**单个 user message** 中的指令（无独立 system message），
对 DMI 的 DAG 依赖回答，还需要在 user prompt 中增加 fact-list，对 final answer 要求输出 new topic / sources。

Search-R1：

```
Answer the given question. You must conduct reasoning inside <think> and </think>
first every time you get new information. After reasoning, if you find you lack
some knowledge, you can call a search engine by <search> query </search> and it
will return the top searched results between <information> and </information>.
You can search as many times as your want. If you find no further external
knowledge needed, you can directly provide the answer inside <answer> and </answer>,
without detailed illustrations. For example, <answer> Beijing </answer>.
Question: {question}
```

**指令拆解**：

| 标签 | 用途 | Agent 行为 |
|------|------|-----------|
| ` ... ` | 推理过程 | 模型在每次获取新信息后进行思考 |
| ` query ` | 搜索动作 | 触发 Wikipedia 搜索工具调用 |
| ` ... ` | 搜索结果 | 工具服务器返回的观察（observation） |
| ` ... ` | 最终答案 | 触发 episode 结束，提取答案进行 EM 打分 |

### 4.4 HotpotQA vs NQ 问题对比

**NQ（单跳）**：答案可直接从 Wikipedia 找到，无需组合多篇文档。

```
Q: total number of death row inmates in the us?
A: ['2,718']
```

**HotpotQA（多跳）**：必须组合多个 Wikipedia 页面的信息才能回答。

```
Q: Musician and satirist Allie Goertz wrote a song about the "The Simpsons"
   character Milhouse, who Matt Groening named after who?
A: ['President Richard Nixon']
   → 需要先查 "Allie Goertz" → "Milhouse" → "Matt Groening" → Nixon

Q: The Oberoi family is part of a hotel company that has a head office in what city?
A: ['Delhi']
   → 需要查 "Oberoi family" → 关联的酒店公司 → 总部城市
```

这也是为什么 Agent 需要多轮搜索（`max_turns=3`）：多跳问题单次搜索往往不够。

### 4.7 多轮交互的数据流

在一次训练 rollout 中，数据在 Agent 和 Tool Server 之间的流转：

```
初始 prompt (Search-R1 指令 + 问题)
    │
    ▼
┌─ vLLM 生成 ─────────────────────────────────────────┐
│   我需要搜索这个问题...                │
│   death row inmates United States   │
└──────────────────────────────────────────────────────┘
    │ 检测到  → 调用 tool_server
    ▼
┌─ Tool Server (wiki_search) ─────────────────────────┐
│  返回:                                   │
│  Doc 1 (Title: Capital punishment...)                │
│  As of 2020, there were 2,718 death row inmates...   │
│                                        │
└──────────────────────────────────────────────────────┘
    │ 将 observation 拼接到对话中
    ▼
┌─ vLLM 继续生成 ─────────────────────────────────────┐
│   根据搜索结果，答案是 2,718           │
│   2,718                             │
└──────────────────────────────────────────────────────┘
    │ 检测到  → episode 结束
    ▼
┌─ Reward Manager ────────────────────────────────────┐
│  提取 "2,718" → 标准化 → 比对 ground_truth           │
│  匹配! → reward = 1.0                                │
└──────────────────────────────────────────────────────┘
```

# Training Configuration

### Algorithm: DAPO/GRPO

DAPO = GRPO + Dynamic Sampling Filter

Key DAPO additions (from `train_7b_dapo.sh`):

```
+algorithm.filter_groups.enable=True
+algorithm.filter_groups.metric='seq_final_reward'
+algorithm.filter_groups.max_num_gen_batches=0
actor_rollout_ref.actor.clip_ratio_high=0.3
actor_rollout_ref.actor.clip_ratio_low=0.2
actor_rollout_ref.actor.loss_agg_mode='token-mean'
```

### Hyperparameters for 14B Model

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| n (samples/prompt) | 8 | Group size for GRPO advantage normalization |
| batch_size | 64 | Conservative for 14B memory |
| ppo_mini_batch_size | 32 | Inner update batch |
| max_prompt_length | 4096 | Covers most HotpotQA questions |
| max_response_length | 6144 | Room for think + search + answer |
| max_turns | 3 | Multi-hop needs multiple searches |
| lr | 1e-6 | Conservative LR for 14B |
| gpu_memory_utilization | 0.55 | Conservative for 14B + vLLM |
| do_offload | True | Offload optimizer states to CPU |
| tensor_model_parallel_size | 1 | vLLM TP size; no tensor parallelism |

### GPU Memory Budget (per GPU)

```
vLLM (rollout):    ~16GB (55% of 80GB, with TP=1)
Actor FSDP shard:  ~14GB (28GB model / 8 GPUs × 4 for activations)
Optimizer states:  ~0GB (offloaded to CPU)
Reference model:   ~14GB (shared with actor via FSDP)
-----------------------------------
Total:             ~44GB / 80GB ✅
```

## Reward 设计与 GRPO 比较机制

### Reward

#### 第一层：提取答案（`extract_solution`）

#### 第二层：Exact Match 比对（`em_check`）

**normalize 实例**：`"The 2003 University of Oxford election"` → `"2003 university of oxford election"`

1. 小写化: "The University" → "the university"
2. 去冠词: 移除 a, an, the
3. 去标点: 移除所有 punctuation
4. 合并空格: "hello   world" → "hello world"

#### 第三层：最终打分（`compute_score`）

**reward的三个等级**：

| 模型输出 | 得分 | 含义 |
|----------|------|------|
| 无 `` 标签 | **0.0** | 完全不符合格式 |
| `错误答案` | **0.1** | 格式正确，内容错误（引导奖励） |
| `正确答案` | **1.0** | 完全正确 |
| 正确但标签 >10 次 | **0.25** | 正确但输出冗余（惩罚） |

**为什么设置 `format_score=0.1`？**

搜索式 RL 的经典"先有鸡还是先有蛋"问题：模型不知道用 ``格式 → 得不到奖励 → 学不会格式。 `format_score=0.1` 的意图是给一个**软引导**：只要用了`` 标签（即使答案错），也给 0.1 分。
期望模型先学会输出格式，再学会输出正确内容。

**实际效果**：模型确实学会了用 `<answer>...</answer>` 包装，但内容始终是 "..." 或乱码，
从不使用 `<search>` 进行工具调用。这导致所有回复分数相同（都是 0.1），
GRPO 无法区分好坏。

### GRPO 如何比较 n 个回复

GRPO（Group Relative Policy Optimization）的核心思想是：
**同一 prompt 的多个回复在组内互相比较，好的加强，差的抑制。**

#### 第一步：每个回复得到一个标量 reward

```python
# token_level_rewards: shape (batch, response_length)
# 只在最后一个有效 token 位置有非零值（= reward score）
scores = token_level_rewards.sum(dim=-1)
# 结果: [0.1, 0.1, 1.0, 0.1,  0.1, 0.1, 0.1, 0.1, ...]
#       |--- prompt_1 ---|    |--- prompt_2 ---|
```

#### 第二步：按 prompt 分组（`index` = uid）

```python
id2score = defaultdict(list)
for i in range(bsz):
    id2score[index[i]].append(scores[i])

# 例:
# id2score = {
#     "q1": [0.1, 0.1, 1.0, 0.1],    # n=4 条回复
#     "q2": [0.1, 0.1, 0.1, 0.1],    # n=4 条回复
#     ...
# }
```

#### 第三步：组内标准化（GRPO 的核心）

```python
# 计算每组均值和标准差
for prompt_id, score_list in id2score.items():
    if len(score_list) == 1:
        id2mean[prompt_id] = 0.0     # 只有 1 条时硬编码兜底
        id2std[prompt_id] = 1.0
    else:
        scores_tensor = torch.stack(score_list)
        id2mean[prompt_id] = torch.mean(scores_tensor)
        id2std[prompt_id] = torch.std(scores_tensor)

# 逐条计算 advantage = (reward - 组内均值) / 组内标准差
for i in range(bsz):
    scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + 1e-6)
```

#### 第四步：Advantage 赋值到 Token 级别

```python
advantages = torch.zeros_like(token_level_rewards)
for i in range(bsz):
    # 只在最后一个有效 token 上放进归一化后的 score
    advantages[i, valid_response_length[i] - 1] = scores[i]
```

### 具体例子

#### 场景 A：有区分度（理想情况，有学习信号）

```
Prompt: "total number of death row inmates in the us?"
Ground truth: ["2,718"]
4 个回复:
  ┌────────────────────────────────┬────────┬───────────┐
  │ 回复                          │ Reward │ Advantage │
  ├────────────────────────────────┼────────┼───────────┤
  │ A:  Paris    │  0.1   │  -0.48    │ ← 抑制
  │ B:  2,718    │  1.0   │  +1.67    │ ← 强化!
  │ C: (无标签乱码)               │  0.0   │  -0.71    │ ← 抑制
  │ D:  Beijing  │  0.1   │  -0.48    │ ← 抑制
  └────────────────────────────────┴────────┴───────────┘

组内: mean = (0.1+1.0+0.0+0.1)/4 = 0.30
      std  = 0.42
      A_adv = (0.1-0.30)/0.42 = -0.48
      B_adv = (1.0-0.30)/0.42 = +1.67  ← 回复B的行为被强化!
      C_adv = (0.0-0.30)/0.42 = -0.71
      D_adv = (0.1-0.30)/0.42 = -0.48

PG loss = -Σ(advantage × log_prob)
  → 回复B的行为（搜索→正确回答）概率上升
  → 回复A/C/D 的行为概率下降
  ✅ 模型学到了"搜索后给出正确答案"更好
```

#### 场景 B：全部相同（我们的现状，无学习信号）

```
Prompt: "total number of death row inmates in the us?"
Ground truth: ["2,718"]
4 个回复:
  ┌──────────────────────────────┬────────┬───────────┐
  │ 回复                        │ Reward │ Advantage │
  ├──────────────────────────────┼────────┼───────────┤
  │ A:  ...    │  0.1   │    0.0    │
  │ B:  ...    │  0.1   │    0.0    │
  │ C:  ...    │  0.1   │    0.0    │
  │ D:  ...    │  0.1   │    0.0    │
  └──────────────────────────────┴────────┴───────────┘

组内: mean = 0.10
      std  = 0.00               ← 标准差为零!
      A_adv = (0.1-0.1)/(0.0+1e-6) = 0.0
      B_adv = C_adv = D_adv = 0.0

PG loss = 0 × log_prob = 0    ← 梯度为零
  ❌ 模型权重完全不更新，训练在"空转"
```

#### 场景 C：恰好有一条偶然正确（突破僵局）

```
3 个回复得 0.1（格式分），1 个回复碰巧得 1.0（正确答案）
  mean = 0.325, std = 0.45
  正确回复 advantage = +1.5，其他 ≈ -0.5
  → 正确行为被放大 1.5 倍
  → 下一轮该行为概率增加
  → 可能产生更多正确回复 → std 增大 → 更强的学习信号
  ✅ 一旦某条回复"运气好"答对了，GRPO 就能开始正循环
```

### `norm_adv_by_std_in_grpo` 的作用

GRPO 有一个开关 `norm_adv_by_std_in_grpo`（默认为 True）：

```python
if norm_adv_by_std_in_grpo:
    scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
else:
    scores[i] = scores[i] - id2mean[index[i]]  # 只减均值，不除标准差
```

| 设置 | 效果 |
|------|------|
| `True`（默认） | 除以标准差 → 组内方差大时缩小 advantage，方差小时放大 |
| `False`（Dr.GRPO） | 不除标准差 → 方差大时 advantage 也大，更新更激进 |

### Policy Gradient 最终计算

有了 advantage 之后，PG loss 的计算：

```python
# 在 dp_actor.py 中
pg_loss = - advantages * log_probs  # 对每个 token
pg_loss = pg_loss * response_mask   # 只看生成 token，忽略 prompt 和 observation
pg_loss = pg_loss.sum() / response_mask.sum()  # 平均
```

### 6.6 pg_loss 的三种情况详解

#### pg_loss = 0：无学习（冷启动困境）

```
所有 n=4 回复 reward 相同 → advantage 全为 0 → pg_loss = 0
→ grad_norm = 0 → 模型权重不更新
```

GRPO 算法的核心机制：

1. 对每个 prompt 生成 n=4 个回复
2. 计算每个回复的 reward
3. 在组内标准化 reward（减去均值，除以标准差）→ 得到 advantage
4. 用 advantage 加权 policy gradient

当所有 4 个回复的 reward **相同**时（都是 0.1 格式分）：

- 组内均值 = 0.1
- 组内标准差 = 0
- advantage = (0.1 - 0.1) / 0 = **0**（除以零时设为 0）
- PG loss = 0 × log_prob = **0**

模型无法区分好坏回复，因此无法学习。

这就是我们 15 步测试和宿主机 99 步训练中 `actor/pg_loss:0.0` 的原因。

#### pg_loss < 0：正向学习 ✅

```
假设某 prompt 的 4 个回复:
  A: reward=1.0 (正确答案) → advantage=+1.5 → 模型生成了 A，log_prob > 0
  B: reward=0.1 (格式对)   → advantage=-0.5 → 模型没生成 B，log_prob < 0
  ...                      
→ advantage × log_prob > 0（同号的乘积为正）
→ pg_loss = -正数 = 负数
→ 梯度方向：增加 A 的概率，降低 B 的概率 → 模型变好！
```

**负得越多，学习越强。** Step 9 的 `pg_loss=-0.028` 是好的。

#### pg_loss > 0：反向学习 ❌

```
假设模型碰巧生成了一个高 reward 回复，但 log_prob 是负的（模型不倾向生成它）
→ advantage > 0, log_prob < 0 → 乘积 < 0
→ pg_loss = -负数 = 正数
→ 梯度方向错误，模型学到"坏行为"
```

**pg_loss > 0 意味着模型在惩罚好行为——这是信号反了，需要检查 reward 设计。**

#### 波动是正常的

```
Step 9:  score=0.184, pg_loss=-0.028  ← 学到了一批好回复
Step 10: score=0.100, pg_loss=0.0      ← 这 batch 恰好没答对
Step 11: score=0.100, pg_loss=0.0      ← 同上
```

RL 训练每步的 batch 不同，模型在探索中时而进步时而退步。
关键在于**长期趋势**：score 从 0.100 → 0.184 已经证明了学习在发生。
更多步数后，正确的搜索-回答行为会越来越频繁。

#### 为什么我们的 pg_loss 的绝对值这么小（0.01-0.03）？

1. **小 batch**：8 prompts × 4 回复 = 32 rollout，方差大
2. **response_length 平均 500+ tokens**：loss 在 token 维度平均后变小
3. **只有 ~40% 轨迹有 ``**：大部分轨迹的 advantage 为 0
4. **GRPO 的保守性**：除以组内 std 会缩小 advantage

随着训练进行，更多轨迹学会正确答案，pg_loss 的绝对值通常会增大。

### 6.6 `no_loss_on_traj` 的含义

在 verl_tool 的指标中有一个 `no_loss_on_traj`：

- **值为 1.0**：该轨迹被 mask 掉，不参与 loss 计算
- **值为 0.0**：该轨迹正常参与 loss
- 我们训练中 `no_loss_on_traj/mean ≈ 0.88-1.0`：88-100% 的轨迹被跳过
- 被 mask 的原因包括：轨迹超长、observation 被 mask（`mask_observations=True`）、轨迹格式无效等

## 训练运行记录

### Prompt 迭代过程（3 轮优化）

**第 1 轮 — Search-R1 原始格式（失败）**：

```
Role: user (单一消息)
Content: "Answer the given question. You must conduct reasoning inside ...
          For example,  Beijing . Question: ..."
```

问题：模型直接输出 "Beijing"，其余全是 `.0000...` 乱码。

**第 2 轮 — System+User 格式（失败）**：

```
Role: system
Content: "...Follow this format...e.g. Paris"
Role: user
Content: "Question: ..."
```

问题：模型复制 "Paris"，15 步中所有回答都是 Paris。

**第 3 轮 — 无示例 System+User 格式（当前使用）**：

```
Role: system
Content: "You are a helpful research assistant...To answer:
  1. Think inside ...
  2. Search Wikipedia with query
  3. Results in ...
  4. Search again if needed
  5. Final answer inside ..."
Role: user
Content: "Question: ..."
```

效果：模型不再复制示例，自己生成内容（但只是 "..." 或无关文本）。

一开始我觉得是Qwen2.5-Coder-14B-Instruct 无法理解 Search-R1 的 XML 标签格式。
后来发现乱码是 vLLM 0.8.5 V1 引擎的 bug，不是模型问题。

总之看起来Coder模型能生成 `...` 结构（获得 0.1 格式分），但从不在 ``中填入有意义的内容， 也从不使用`` 标签进行工具调用。

### 工具服务器表现

wiki_search 工具服务器在整个测试中运行正常：

- 健康检查始终通过（HTTP 200）
- `tool_call_success=1.0` 表示工具服务器可达
- 但 `tool_calls` 耗时 = 0.0s，说明模型从未发送 `` 请求
- 代理配置正确：Wikipedia API 可通过公司代理访问
- SSL 证书问题已通过 `verify=False` + `trust_env=True` 解决

---

# 关键发现与经验教训

### 模型选择的决定性影响

这是本次实验最重要的发现：**RL 训练的成功与否，80% 取决于基座模型是否理解任务格式**。

在 Search-R1 格式的示例数据上做 SFT 预热，再进行 RL 训练。

### 网络环境约束

公司代理环境带来的挑战：

1. PyPI 镜像限制（最高 torch 2.6.0）→ 无法升级 vLLM
2. SSL 证书检查（代理做 SSL inspection）→ 需要 `verify=False`
3. 代理：需要它访问外网（Wikipedia），但必须绕过它访问本地服务（127.0.0.1）

### 训练效率

- 14B 模型在 8×A100 上：~95 秒/步（32 rollouts）
- 扩展到 200 步需要 ~5.3 小时
- 更大的 batch（如 batch=64, n=8）预计每步 ~6-8 分钟，200 步 ~20 小时
- GPU 内存使用 74-89GB，刚好在 80GB 边界内（感谢 FSDP offload）

### 奖励设计的重要性

原始 Search-R1 使用 `format_score=0.0`（无格式奖励）。
我们改成了 `format_score=0.1`，希望给模型一个"软引导"。
但结果是模型学会了输出 `<answer>` 标签（拿 0.1 分），却没有动力去搜索正确答案。

**更好的奖励设计**：

- `format_score` 应随时间衰减（前期引导格式，后期只奖励正确性）
- 或者使用 **DAPO** 的 filtering 机制：过滤掉低分轨迹，只在高分轨迹上学习
- 一开始想或者增加 `search_bonus`：使用 `<search>` 标签就给额外奖励。 但是HotpotQA 的多跳要求自然体现在 EM 难度上——不额外加 search_bonus。
  模型通过 GRPO 自己发现 "多搜索 → 高 reward" 的规律。

---

## 13. SFT 预热数据准备

### 13.1 动机

15步测试表明 Qwen2.5-Coder-14B-Instruct 完全不理解XML 标签格式，
GRPO 的 advantage 始终为 0。
SFT 预热的目的是在 RL 之前教会模型基本格式：

### 13.2 方法

使用 Wikipedia API 直接搜索，构造 Search-R1 格式的示范轨迹。

对每个 HotpotQA 问题：

1. 用 golden answer 本身作为搜索词（最优策略）
2. 用清理后的问题关键词作为补充搜索
3. 对每个搜索词调用 Wikipedia API 获取真实内容
4. 如果搜索结果包含 golden answer 文本 → 标记为 "有答案上下文"
5. 构造标准的多轮轨迹

直接用 requests + Wikipedia API

- 结果：500 条测试中 **65% (325条) 答案出现在 Wikipedia 搜索结果中**
- 改进：HTTPS 直连、用 golden answer 作为搜索词、限流重试
- 最终：生成 1000 条 SFT 数据（含 60% 高质量轨迹）
- 100% 格式完整（`<think>` + `<search>` + `<information>` + `<answer>`）
- 100% API 成功率，41 分钟生成

### 13.4 SFT 数据格式

```json
{
  "messages": [
    {"role": "system", "content": "<系统提示>"},
    {"role": "user", "content": "Question: <问题>"}
  ],
  "chosen": "...\nquery\n...\n...\n答案",
  "question": "原始问题",
  "golden_answers": ["答案"],
  "found_answer_context": true/false
}
```

SFT 训练时，input = `messages`，output = `chosen`。模型学习在给定系统提示和问题后，
生成正确的多轮搜索→回答轨迹。

一开始 98% 的数据条目搜索失败：

1. **超时不是问题**：成功的 Wikipedia API 调用在 0.4-1.0s 内完成。10s 超时已足够。
2. **延迟不足是根本原因**：Wikipedia 每 ~3 个请求后返回 HTTP 429（Too Many Requests），
   `Retry-After` 头指示等待 13 秒。v1 使用 0.3s 延迟，第 4 个请求就开始被限流。
3. **每个问题触发多个 API 调用**：wikipedia 库的 `search()` + `page()` + `summary()`
   每个都是一次独立 API 调用。v1 每个问题可能触发 4-6 次调用，在 0.3s 延迟下迅速耗尽配额。

### 训练现象 (2026-07-29, 43 步验证)

**Step 1 — 多轮搜索首次成功**：

- `num_turns/mean: 2.28` → 模型主动多轮搜索
- `tool_call_success: 68.75%` → Wikipedia 搜索成功执行
- `valid_traj: 100%` → 首次全部轨迹格式正确
- `no_loss_on_traj: 0%` → 全部轨迹参与 loss

**Step 6-7 — 学习信号出现**：

- `score: 0.100 → 0.128 → 0.156` → Reward 首次上升
- `pg_loss: 0.0 → -0.017 → -0.010` → 首次非零

**Step 14-43 — 持续学习** ��：

- **前 10 步 Score 均值: 0.148**
- **后 10 步 Score 均值: 0.435** (↑ 3 倍!)
- **峰值: 0.719 (Step 17)** → 3/4 回复给出了正确答案
- **Step 43: 0.606** → 学习稳定在高位
- `num_turns` 从 2.28 降至 1.0 → 模型学会直接回答，不再盲目多轮搜索

Score 趋势可视化：

```
Step  1-11: ████░░░░░░  0.10-0.27  (探索阶段)
Step 12-25: ██████████  0.18-0.52  (开始学习)
Step 26-43: ████████████████  0.35-0.72  (稳定上升!)
```

### 关键发现：数据分布不均

**数据排列**：训练集 152,653 条**完全未混合**——
前 79,168 条 100% NQ（单跳），后 73,485 条 100% HotpotQA（多跳）。
只有 2 个连续块，中间没有任何穿插。

**对训练的影响**：

- 200 步 × batch_size=8 = 1,600 条，全部落在 NQ 单跳区域
- 模型**从未见过 HotpotQA 多跳问题**
- Score 提升全部来自 NQ 事实型问题，多跳能力完全未训练

**这解释了全部观察现象**：

| 现象 | 根因 |
|------|------|
| num_turns 从 2.28 降到 1.0 | NQ 不需要多轮搜索 |
| Score 先升后稳（0.42→0.33） | NQ 太简单，天花板低 |
| Score 峰值 0.719 无法复现 | 只有少数步骤运气好 |
| 早期 steps 1-10 Score 最低 | model 还在探索搜索策略 |

### Dr.GRPO 适用性分析

**GRPO vs Dr.GRPO**：

| 维度 | GRPO | Dr.GRPO |
|------|------|---------|
| 公式 | `adv = (R-mean)/std` | `adv = R-mean` |
| 开销 | 相同 | 相同（只跳过除法） |
| 稀有正确回复 | **放大**（std 小时 advantage 大）| 等比例 |
| 小 batch | 不稳定（std 估计不可靠）| **更稳定** |
| 长轨迹（>10K tokens） | 容易梯度爆炸 | **更适合** |

**选择建议**：

- 当前 HotpotQA 短轨迹 + batch=8 → GRPO 足够
- 引入 80K token 长轨迹 + batch=1-2 → **必须用 Dr.GRPO**
- 原因：长轨迹 reward 方差大 + 小 batch 的 std 估计噪声 → GRPO 会将噪声放大

**训练改进清单**（基于全部发现）：

1. ➕ 对数据做 `shuffle`，确保单跳/多跳混合
2. ➕ 设置 `save_freq=50` 保存 checkpoint
3. ➕ 增加步数至 1000+ 或按需
4. ➕ 长轨迹场景设置 `norm_adv_by_std_in_grpo=False`（Dr.GRPO）
5. ➕ 长轨迹场景降低 `batch_size=1-2`，保持 `n=4`

## 实验 2：Shuffle 数据 + 正确 Epoch 设计 (2026-07-30 启动)

### 改进点

| 改进 | 实验 1 | 实验 2 |
|------|--------|--------|
| 数据 | 152K 未 shuffle（全在 NQ 区域） | **1,526 条随机混合**（50% NQ + 50% HotpotQA） |
| Epoch | 200 步 ≈ 0.01 epoch | **382 步 ≈ 2 epochs** |
| Checkpoint | save_freq=-1（丢失） | **save_freq=50** |
| 算法 | GRPO | GRPO |
| 日志 | 单文件 | 时间戳独立保存 |

### Epoch 设计逻辑

RL ≠ SFT。RL 不需要多 epoch 来"记住"数据，关键是每个问题给几次尝试：

- 1 epoch (191步): NQ 够了，HotpotQA 刚热身
- **2 epochs (382步)**: 两类都正好，~18 小时
- 3+ epochs: NQ 有背诵风险

选择 2 epochs：每个问题被看到 2 次，NQ 第一次学格式、第二次学搜索，
HotpotQA 第一次发现需要多跳、第二次学会搜索链。

---

# 技术问题

## Q1: 为什么用 Dr.GRPO 思路修复 DAPO？

初次我们尝试了 GRPO，这里说一下一开始的思路 以及和 PPO 的区别：

**核心区别**：PPO 需要 Critic（Value Network）来估计 Advantage，而 GRPO 用**组内相对比较**。

- PPO：`Advantage = Reward - V(s)`，需要额外训练一个 Critic 模型（参数量 ≈ Actor）
- GRPO：同一 prompt 生成 n 个回复，组内标准化 `Advantage = (R - mean_group) / std_group`
- GRPO 省掉 Critic → 节省 ~50% GPU 内存和训练时间，更适合 14B 大模型

> PPO 的 Critic 是一份与 Actor 同量级的模型，极易 OOM；GRPO 省下这份显存全给 Actor。其优势估计为组内相对：
> 
> $$
> 
> $$

A_i=\frac{r_i-\operatorname{mean}(\mathbf{r})}{\operatorname{std}(\mathbf{r})}

$$
> 
> 相比 PPO，组内相对归一化给训练提供比单一 reward + 不可靠 critic 更清晰的信号

**为什么选择 GRPO**：

- 14B 模型已经很大，再训练一个 14B Critic 会超出 GPU 内存
- GRPO 天然支持 outcome-level reward（只有最终答案才有 reward），适合 QA 任务

### 改良实践

vanilla GRPO 的两个已知偏置：

1. **难度偏置**：除以 $\operatorname{std}(\mathbf{r})$ 会放大"极易/极难"题的权重。Dr. GRPO 取消这个缩放，平等对待所有题目。→ veRL 配置 `algorithm.norm_adv_by_std_in_grpo: False`。
2. **长度偏置**：**按序列长度平均**会让"更长的错误答案"被低估惩罚。GRPO 按序列长度归一化会导致更长的错误回答被惩罚不足。 
对我们的 DMI 场景来说，长序列的错误答案是很常见的。因此使用 Dr.GRPO 改用全局常数归一化以消除长度偏置。

我们尝试使用 Dr.GRPO 改良。

### GRPO vs Dr.GRPO 有什么区别？什么场景该用哪个？

**GRPO**: `advantage = (reward - group_mean) / group_std`
**Dr.GRPO**: `advantage = reward - group_mean`

**关键区别**：GRPO 除以 std → 当 std 很小时放大 advantage（稀有正确回复被大力强化），当 std 很大时缩小 advantage（分散的 reward 被压平）。Dr.GRPO 跳过了除法，保持原始 reward 差距。

| 场景  | 推荐  | 原因  |
| --- | --- | --- |
| 短轨迹、n 大（≥4） | **GRPO** | std 估计可靠，放大稀有正确信号有助于冷启动 |
| 长轨迹（>10K tokens） | **Dr.GRPO** | reward 方差大，GRPO 易梯度爆炸 |
| 小 batch（1-2） | **Dr.GRPO** | std 在小样本上极不可靠 |
| 冷启动（reward=0 为主） | **GRPO** | 少数高 reward 回复需要被放大 |

**开销**：完全一样——Dr.GRPO 只跳过除法运算，不增加任何计算量。

**切换**：一行参数 `algorithm.norm_adv_by_std_in_grpo=False`。

### 那么 DAPO 呢？

Completed Version: DAPO's Four Components (+ Two "Implicit" Components)


DAPO 的全称是 **D**ecoupled Clip and **D**ynamic s**A**mpling **P**olicy **O**ptimization（ByteDance Seed × 清华 AIR，arXiv 2503.14476）。它的四大组件不是四个独立 trick，而是同时体现在**一个目标函数**里的四处修改，所以最清晰的补全方式是先看完整式子，再逐项对应：

$$\mathcal{J}_{\text{DAPO}}(\theta)=\mathbb{E}_{(q,a)\sim\mathcal{D},\,\{o_i\}_{i=1}^{G}\sim\pi_{\theta_{\text{old}}}(\cdot\mid q)}\left[\underbrace{\frac{1}{\textstyle\sum_{i=1}^{G}|o_i|}\sum_{i=1}^{G}\sum_{t=1}^{|o_i|}}_{\text{③ token-level}}\min\Big(r_{i,t}(\theta)\hat{A}_{i,t},\ \operatorname{clip}\big(r_{i,t}(\theta),\,1-\underbrace{\epsilon_{\text{low}}}_{\text{① }0.2},\,1+\underbrace{\epsilon_{\text{high}}}_{\text{① }0.28}\big)\hat{A}_{i,t}\Big)\right]
$$

$$
\text{s.t.}\quad \underbrace{0<\big|\{o_i\mid \texttt{is\_equivalent}(a,o_i)\}\big|<G}_{\text{② dynamic sampling 约束 / constraint}},\qquad \hat{A}_{i,t}=\frac{R_i-\operatorname{mean}(\{R_j\}_{j=1}^{G})}{\operatorname{std}(\{R_j\}_{j=1}^{G})}
$$

其中 $r_{i,t}(\theta)=\dfrac{\pi_\theta(o_{i,t}\mid q,o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}\mid q,o_{i,<t})}$，而 $R_i$ 中包含了组件 ④ 的长度整形项。**注意：式子里没有 KL 项**——这是 DAPO 相对 GRPO 的另一个关键删除。

Here $r_{i,t}(\theta)=\dfrac{\pi_\theta(o_{i,t}\mid q,o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}\mid q,o_{i,<t})}$, and $R_i$ absorbs the length-shaping term from component ④. **Note there is no KL term** — that is another key deletion of DAPO relative to GRPO.

---

### 1. 补全后的四大组件表 / The Completed Four-Component Table

| 技巧 / Technique | 做法 / Method | 关键机制细节 / Key Mechanism | 解决的问题 / Problem Solved |
|---|---|---|---|
| **① Clip-Higher**（解耦裁剪 / decoupled clipping） | 把 PPO 单一 $\epsilon$ 拆成非对称区间 $[1-\epsilon_{\text{low}},\,1+\epsilon_{\text{high}}]$，论文取 $\epsilon_{\text{low}}=0.2,\ \epsilon_{\text{high}}=0.28$ / Split PPO's single $\epsilon$ into an asymmetric range $[1-\epsilon_{\text{low}},\,1+\epsilon_{\text{high}}]$; the paper uses $\epsilon_{\text{low}}=0.2,\ \epsilon_{\text{high}}=0.28$ | 上界 $\epsilon_{\text{high}}$ 放宽以给低概率 token "上升空间"；下界 $\epsilon_{\text{low}}$ **故意不放宽**，否则会把低概率 token 直接压到 0、缩小采样空间 / Raise $\epsilon_{\text{high}}$ to give low-probability tokens headroom to grow; **deliberately keep $\epsilon_{\text{low}}$ small**, since relaxing it would crush low-probability tokens to 0 and shrink the sampling space | 对称裁剪对低概率 token 的提升上限极苛刻 → 策略熵单调下降、rollout 高度同质化 → **熵坍缩**、探索死亡 / Symmetric clipping imposes a brutally tight growth ceiling on low-probability tokens → monotonically decreasing entropy, near-identical rollouts → **entropy collapse** and dead exploration |
| **② Dynamic Sampling**（动态采样 / dynamic sampling） | 过采样后**过滤掉准确率为 0 或 1 的组**（组内 $\hat{A}\equiv 0$，梯度为零），持续重采样直到凑满一个全是"有效样本"的 batch，以约束 $0<|\{\text{correct}\}|

组件 ④ 的软惩罚公式（论文 Eq. 13）为：

The soft penalty of component ④ (paper Eq. 13) is:

$$
R_{\text{length}}(y)=\begin{cases}0, & |y|\le L_{\max}-L_{\text{cache}}\\[4pt] \dfrac{(L_{\max}-L_{\text{cache}})-|y|}{L_{\text{cache}}}, & L_{\max}-L_{\text{cache}}<|y|\le L_{\max}\\[6pt] -1, & |y|>L_{\max}\end{cases}
$$

最终奖励 $R_i = R_{\text{correct}}(o_i, a) + R_{\text{length}}(o_i)$，其中 $R_{\text{correct}}\in\{+1,-1\}$ 由规则验证器（答案等价性判定）给出。这个设计的精神是：**长度约束应表达为"渐进变贵"而不是"悬崖式判死"**，避免在 $L_{\max}$ 处出现奖励函数的阶跃不连续。

The final reward is $R_i = R_{\text{correct}}(o_i, a) + R_{\text{length}}(o_i)$, where $R_{\text{correct}}\in\{+1,-1\}$ comes from a rule-based verifier (answer-equivalence checking). The spirit of the design: **a length constraint should be expressed as "progressively more expensive," not as a cliff-edge death sentence**, avoiding a step discontinuity in the reward function at $L_{\max}$.

---

### 2. 两个常被漏掉的"隐性"组件 / Two Frequently Omitted "Implicit" Components

**⑤ 彻底移除 KL 惩罚项。** 这是 DAPO 与经典 RLHF 最哲学性的分歧。在 RLHF 中，KL 项的作用是"别偏离 SFT 模型太远，保住语言质量与安全性"；但在 long-CoT reasoning RL 中，模型**本来就要**从 base 模型的行为分布上大幅漂移（学会自我检查、回溯、超长推理），此时 KL 项就是纯粹的枷锁。DAPO 直接令 $\beta=0$。这与 Dr.GRPO / Open-Reasoner-Zero 等同期工作的结论一致，也是 2025 年"零 KL 训练"成为 reasoning RL 默认配置的关键一步。

**⑤ Complete removal of the KL penalty.** This is DAPO's most philosophical break with classical RLHF. In RLHF, the KL term says "don't drift too far from the SFT model; preserve language quality and safety." But in long-CoT reasoning RL, the model **is supposed to** drift massively from the base model's behavioral distribution (learning self-checking, backtracking, very long deliberation), so the KL term becomes pure shackles. DAPO simply sets $\beta=0$. This matches contemporaneous findings in Dr.GRPO / Open-Reasoner-Zero, and was a key step in making "zero-KL training" the default configuration for reasoning RL in 2025.

**⑥ 纯规则奖励 + 答案等价性验证器。** DAPO 不用任何神经奖励模型，只用可验证的最终答案匹配（AIME 类数学题转换为整数答案），从根上消除 reward hacking。这一点常被当作"实验设置"而非"算法组件"，但它其实是前四个 trick 能成立的前提：只有当奖励绝对可信时，才敢像 ② 那样激进地丢弃样本、像 ④ 那样直接给奖励做手术。

**⑥ Purely rule-based rewards + an answer-equivalence verifier.** DAPO uses no neural reward model at all, only verifiable final-answer matching (AIME-style problems converted to integer answers), eliminating reward hacking at the root. This is often filed under "experimental setup" rather than "algorithmic component," but it is really the precondition for the other four tricks: only when the reward is absolutely trustworthy can you afford to discard samples as aggressively as ② does, or perform surgery directly on the reward as ④ does.

---

<details>
<summary><b>消融实验与超参（点击展开）/ Ablations and Hyperparameters (click to expand)</b></summary>

论文 Table 1，Qwen2.5-32B **base** 模型起训，AIME 2024 avg@32：

Paper Table 1, trained from the Qwen2.5-32B **base** model, AIME 2024 avg@32:

| 配置 / Setting | AIME24 avg@32 | 增量 / Δ |
|---|---|---|
| DeepSeek-R1-Zero-Qwen-32B（对照 / reference） | 47 | — |
| Naive GRPO | 30 | — |
| + Overlong Filtering | 36 | +6 |
| + Clip-Higher | 38 | +2 |
| + Soft Overlong Punishment | 41 | +3 |
| + Token-level Loss | 42 | +1 |
| + Dynamic Sampling | **50 (DAPO)** | +8 |

两点值得注意：**(a)** Token-level Loss 的分数增益最小（+1），但论文明确指出它的价值在于**训练稳定性与"健康"的长度增长曲线**，而非直接刷分——这是评价 RL trick 时容易被单一指标误导的典型例子。**(b)** Dynamic Sampling 贡献最大（+8），说明在后期"有效梯度稀疏化"是长程 RL 最主要的瓶颈之一。

Two things worth noting: **(a)** Token-level Loss gives the smallest score gain (+1), but the paper explicitly states its value lies in **training stability and a "healthy" length-growth curve**, not in raw score — a textbook case of how a single metric can mislead when judging RL tricks. **(b)** Dynamic Sampling contributes the most (+8), indicating that late-stage "effective-gradient sparsification" is one of the primary bottlenecks in long-horizon RL.

**关键超参 / Key hyperparameters:** verl 框架；AdamW，恒定 lr $1\times10^{-6}$ + 20 步线性 warm-up；rollout prompt batch = 512，每 prompt 采 $G=16$；mini-batch = 512（即每次 rollout 做 16 次梯度更新）；$\epsilon_{\text{low}}=0.2,\ \epsilon_{\text{high}}=0.28$；期望最大长度 16384 + 4096 缓冲 = 生成上限 20480；评测 temperature 1.0、top-$p$ 0.7、重复 32 次取 avg@32。

**Key hyperparameters:** verl framework; AdamW with constant lr $1\times10^{-6}$ plus a 20-step linear warm-up; rollout prompt batch = 512 with $G=16$ samples per prompt; mini-batch = 512 (i.e., 16 gradient updates per rollout step); $\epsilon_{\text{low}}=0.2,\ \epsilon_{\text{high}}=0.28$; expected max length 16384 + 4096 cache = 20480 generation cap; evaluation at temperature 1.0, top-$p$ 0.7, repeated 32× for avg@32.

</details>

---

### 3. 与 GRPO / Dr.GRPO 的精确对位 / Precise Alignment with GRPO / Dr.GRPO

把三者放在一起看，DAPO 与 Dr.GRPO 其实**部分诊断相同、处方不同**，这是理解 2025 年这条技术线的关键：

Viewing the three side by side, DAPO and Dr.GRPO in fact share **part of the diagnosis but differ in the prescription** — this is the key to understanding this 2025 technical thread:

| 维度 / Dimension | GRPO | Dr.GRPO | DAPO |
|---|---|---|---|
| 长度归一化 $1/\lvert o_i\rvert$ / length normalization | 有（引入长度偏置）/ present (length bias) | **删除**，改用常数 $1/L_{\max}$ 式的无偏聚合 / **removed**, unbiased aggregation | **删除**，改为全局 token 平均 / **removed**, global token averaging |
| 优势的 std 归一化 / std normalization of advantage | 有 / yes | **删除**（认为它给难/易题错误加权）/ **removed** (argued to mis-weight hard/easy questions) | **保留** / **retained** |
| 裁剪区间 / clipping range | 对称 / symmetric | 对称 / symmetric | **非对称（Clip-Higher）** / **asymmetric** |
| KL 项 / KL term | 有 / yes | 通常置零 / typically zero | **置零** / **zero** |
| 样本过滤 / sample filtering | 无 / none | 无 / none | **动态采样** / **dynamic sampling** |
| 长度/截断处理 / length & truncation handling | 无 / none | 无 / none | **软惩罚 + 过滤** / **soft penalty + filtering** |

**一句话总结分歧：** Dr.GRPO 的立场是"从统计估计的无偏性出发，把 GRPO 里所有引入偏置的项都删掉"，是一种**理论洁癖式的减法**；DAPO 的立场是"从大规模训练的病理现象（熵坍缩、零梯度、长度暴涨、截断噪声）出发，针对每种病开一味药"，是一种**工程实证式的加法**。两者在"删除 $1/\lvert o_i\rvert$ 长度偏置"上英雄所见略同，但在 std 归一化上分道扬镳——DAPO 保留它，说明在实际大规模训练中，std 归一化带来的方差缩减收益可能超过它引入的加权偏置代价。这个分歧至今没有被完全定论，也是后续工作（如 GSPO、CISPO、以及各类序列级重要性采样变体）继续争论的战场。

**The disagreement in one sentence:** Dr.GRPO's position is "start from unbiasedness of statistical estimation and delete every bias-introducing term in GRPO" — a **theory-purist subtraction**. DAPO's position is "start from the observed pathologies of large-scale training (entropy collapse, zero gradients, length explosion, truncation noise) and prescribe one drug per disease" — an **empirically driven addition**. The two agree on removing the $1/\lvert o_i\rvert$ length bias but part ways on std normalization — DAPO's retention of it suggests that at real scale, the variance-reduction benefit of std normalization may outweigh the weighting bias it introduces. This disagreement is still unsettled and remains the battleground for follow-up work (GSPO, CISPO, and various sequence-level importance-sampling variants).

### 什么问题是 DAPO 能治而 Dr.GRPO 治不了的？

**核心区别（一句话）**：
Dr.GRPO 只改"组内已有差异的缩放"（`adv=(R-mean)/std` → `adv=R-mean`），
不改"哪些数据参与训练"；DAPO 的四个组件全部作用在**数据选择 / 梯度配平**层面。
如果组内 reward 全相等，`R-mean=0` 同样是 0——Dr.GRPO 一样空转。

| 维度  | Dr.GRPO | DAPO |
| --- | --- | --- |
| 改变什么 | 组内 advantage 的缩放 | 组构成、裁剪、损失聚合、reward 尺度 |
| 组内全同分（std=0） | `adv=0`，照样空转 | `filter_groups` 整组丢弃 |
| 跨组数据选择 | 无   | Dynamic Sampling 过滤 + 重新生成 |
| 正样本抑制 / 坍缩 | 无   | Clip-Higher 非对称裁剪 |
| reward 尺度异常 | 无   | Sparse Reward Normalization |
| 超长轨迹 | 无   | overlong_buffer 惩罚 |

两者在"删除 1/∣o_i∣ 长度偏置"上英雄所见略同，但在 std 归一化上分道扬镳。
DAPO 保留它，说明在实际大规模训练中，std 归一化带来的方差缩减收益可能超过它引入的加权偏置代价。这个分歧至今没有被完全定论，也是后续工作（如 GSPO、CISPO、以及各类序列级重要性采样变体）继续争论的战场。

### 关键工程：Observation Token Masking（多轮 RL 的命门）

在计算 log-prob 和 policy loss 时，**必须对 `<information>` 内的检索 token 置零 mask**，只对模型自己生成的 think/search/answer 计 loss。这是 Search-R1 最关键的创新之一：RL 期间检索内容被排除在优化之外，只有模型自己的推理参与更新，迫使模型"对检索结果做推理"而非"照抄"，从而提升稳定性与泛化。不做 mask 会让模型去拟合外部网页内容，导致学偏/坍缩。实验显示做 masking 训练更稳、提升更大。

### 可以改进的多轮 credit assignment

轨迹级单标量 reward 广播到所有 turn 是最简做法，但 GRPO 在多轮设定下被广泛报告不稳定。
更细的做法是 **turn-level 优化**：整条轨迹含多个"模型生成 + 环境反馈"回合，在环境反馈上做优化会引入不稳定，因此解耦模型生成 $o_t$ 与环境反馈 $f_t$，只在 $o_t$ 上做定向优化。

代表工作：RAGEN 的 StarPO-s 用比例化轨迹过滤，GiGPO 结合状态级与轨迹级优势，MT-GRPO 展示 turn-level credit assignment 的收益。

### hyperparameter 选择

| 项 | Gemini 值 | **建议值** | 理由 |
| :--- | :--- | :--- | :--- |
| Group size $G$ | 5 | **8~16** | 统计更稳；配合 dynamic sampling |
| Advantage std 归一 | 除 std | **关闭**（Dr.GRPO）或保留但知其偏置 | 消除难度偏置 |
| Dynamic sampling | 无 | **开启** | 消灭零方差空梯度组 |
| KL coef | 0.001 | **0（DAPO 路线）或 1e-3（保守路线），二选一并能解释** | 见下 |
| Actor LR | 1e-6 | 1e-6 ~ 5e-7 + linear warmup | Agent 训练要保守 |
| 单轮生成上限 | "2048" | **区分**：单轮 512~1024，整条轨迹总长 8k~16k | 多轮会累积 |
| Loss 聚合 | 未提 | token-level（DAPO） | 长序列更精确 |

> **KL 二选一话术**
> 路线 A（DAPO/Dr.GRPO）——"结果奖励可验证 + 参考模型已是好起点，去 KL 让策略充分移动、避免拖后腿"；
> DAPO 在其方法中移除了 KL 散度。
> 路线 B（保守）——"保留小 KL 防止在稀疏奖励早期策略崩溃/复读，代价是探索受限"。

---

## Q2: Reward 怎么设计的？

**三层 Reward 结构**：

1. 无 `<answer>` 标签 → `reward = 0.0`
2. 有 `<answer>` 但答案错 → `reward = 0.1`（format_score，引导格式）
3. 答案完全匹配（EM） → `reward = 1.0`

**为什么不直接用 0/1 二值 reward？**

- RL 有"冷启动"问题：如果模型从不会输出 `<answer>`，所有 reward = 0
- GRPO 组内标准化后所有 advantage = 0 → PG loss = 0 → 模型不更新
- `format_score=0.1` 给了一个"梯子"，模型先学会格式，再学内容

## Q3: Tool Server 怎么工作？Agent 怎么调用工具的？

**架构**：

```
veRL Training Process
    │ vLLM 生成 → 检测到 </search>
    │ POST /get_observation
    ▼
Tool Server (verl_tool.servers.serve)
    │ 路由请求到 wiki_search 工具
    │ 调用 Wikipedia API
    │ 返回 <information>...</information>
    ▼
vLLM 继续生成（observation 拼接到对话中）
    │ 检测到 </answer>
    ▼
Reward Manager 计算得分
```

**关键设计**：

- Tool Server 是独立进程，HTTP 通信，支持多 worker 并发
- 使用 Ray 分发工具调用（生产环境）或线程池（开发环境）
- `action_stop_tokens="</search>,</answer>"` → vLLM 遇到这些 token 时暂停生成
- Agent Loop 最多 `max_turns=3` 轮交互

## Q6: multi-turn Agent 的数据流是怎样的？

```
Step 1: [system prompt + "Question: ..."] → vLLM →
    "<think>I need to search...</think><search>query</search>"

Step 2: Tool call → Wikipedia → "<information>Doc 1: ...</information>"
    [对话 + observation] → vLLM →
    "<think>Found the answer: Paris</think><answer>Paris</answer>"

Episode end → Reward Manager:
    extract_solution → "Paris" → normalize → EM check → reward
```

Loss 只计算生成 token（不包括 prompt 和 observation token），由 `mask_observations=True` 控制。

## Q7: 怎么判断训练是否在"真的学习"？

**关键信号**：

- `critic/score/mean` 趋势上升 → 模型在提高答题准确率
- `actor/pg_loss` 非零且有波动 → 有梯度信号
- `verl_tool/num_turns/mean` 从 1 增加到 >1 → 模型在多轮搜索
- `verl_tool/tool_calls/mean` 非零 → 工具被实际调用
- `actor/entropy` 不退化到 0 → 保持探索

**反信号（我们实际遇到的）**：

- PG loss = 0.0 持续 99 步 → 空转训练
- score = 0.1 不变 → 所有回复格式分相同，无区分度 → 说明模型没有真正学习，需要检查 reward 设计或模型能力

## Q10: pg_loss 是什么？为什么有时是负的、有时是正的、有时是零？

`pg_loss = -Σ(advantage_i × log_prob_i) / N`

这是 GRPO 训练中最重要的指标，直接反映"模型是否在学习"。

**三种情况**：

| pg_loss | 含义  | 实际表现 |
| --- | --- | --- |
| **= 0** | 无学习信号。所有回复 reward 相同 → advantage = 0 | 宿主机训练 99 步全是 0 |
| **< 0** | **正向学习！** 高 reward 回复的概率在上升 | Docker Step 9: -0.028 ✅ |
| **> 0** | 反向学习。模型在惩罚好行为（信号反了） | 偶尔出现，正常波动 |

**为什么绝对值这么小（0.01-0.03）？**

1. 小 batch（8 prompts × 4 回复）噪声大
2. 平均到 500+ token 后数值变小
3. GRPO 除以组内 std 会缩小 advantage
4. 随着训练进行、更多轨迹答对，pg_loss 绝对值会增大

**怎么识别"假学习"vs"真学习"？**

- 假学习：pg_loss ≠ 0 但 score 始终不涨 → 模型在过拟合噪声
- 真学习：pg_loss 波动 + score 趋势上升 → 我们在 Docker 训练中看到的

## Q11: 为什么需要 Docker 环境？vLLM 0.8.5 和 0.11 有什么区别？

**核心矛盾**：宿主机 PyPI 镜像最高 torch 2.6 → 无法装 vLLM ≥ 0.9。

**vLLM 0.8.5 的问题**（宿主机）：

- V1 引擎是实验性的，对 Qwen2.5 系列有 token 生成 bug
- HuggingFace 直接推理正常，vLLM 推理输出多语言乱码
- 需要 5 个兼容性补丁才能运行 veRL

**vLLM 0.11 的解决**（Docker）：

- 使用 `vllm/vllm-openai:v0.11.0` 镜像（torch 2.8 + CUDA 12.8）
- 5 个补丁全部不需要（vLLM API 一致）
- 模型正确输出 `<think>...<search>query</search>` 格式
- 100% valid_traj vs 宿主机 3-9%

**为什么 Docker 在宿主机上行不通？**

- 8×A100 + FSDP + Ray + vLLM 的内存峰值接近宿主机 503GB
- Docker 容器被 OOM Killer 杀掉（exit code 137）
- 最终方案：无 `--memory` 限制 + `--shm-size=20g` + `--ipc=host`

## Q12: Coder 模型输出乱码，你是怎么排查到根因的？

**排查过程（面试重点！）**：

1. **现象**：训练日志显示 `score/mean=0.1` 持续 99 步，`num_turns=1`，模型从不调用搜索
2. **怀疑 1**：模型能力不够 → 用简单英文提问 "What is the capital of France?" → 正常回答 ✅
3. **怀疑 2**：prompt 格式不对 → 三轮迭代（Search-R1 原始 → 加 system message → 去掉示例答案）→ 仍有乱码
4. **怀疑 3**：tokenizer 有问题 → 检查 chat_template 正确应用 ✅
5. **怀疑 4**：vLLM 推理和 HuggingFace 推理不一致 → **关键验证！** HF 推理正常，vLLM 推理乱码
6. **结论**：vLLM 0.8.5 V1 引擎的 token 生成 bug → 切换到 Docker vLLM 0.11 → 完美解决

**面试技巧**：展示系统性的排查思路——逐一排除假设，最终定位到推理引擎层。

## Q13: 训练 43 步后 Score 从 0.10 涨到 0.72，这意味着什么？

**实际数据（Docker vLLM 0.11 + Instruct）**：

| 阶段  | Steps | Score 均值 | 含义  |
| --- | --- | --- | --- |
| 探索期 | 1-11 | 0.148 | 模型偶尔答对，大部分是格式分 |
| 学习期 | 12-25 | 0.35 | 搜索+答案正确率上升 |
| 稳定期 | 26-43 | 0.44 | 持续超过 0.4，峰值 0.72 |

**Score 0.72 的分解**（n=4, format_score=0.1）：

- 3 条回复答对 (1.0×3) + 1 条格式分 (0.1) → (3.1/4) = 0.775
- 或 2 条答对 + 1 条格式分 + 1 条无标签 → (2.1/4) = 0.525
- Score 0.72 ≈ 平均每个 prompt 有 2-3 条回复答对

**num_turns 的阶段性变化**：

- Step 1: 2.28（多轮搜索）
- Step 43: 1.0（直接回答）
- 解读：模型学会了判断问题难度——简单问题直接答，复杂问题才搜索

## Q14: SFT 预热为什么对 Coder 无效但对 Instruct 没必要？

**Coder 模型的 SFT**（500 条 × 3 epochs）：

- 目标：教会模型 `<think>` → `<search>` → `<answer>` 格式
- 结果：学会了 `<answer>` 格式（valid_traj 提升 10%），但没学会 `<search>`
- 原因：Coder 的代码先验太强，500 条不够翻转 14B 参数

**Instruct 模型不需要 SFT**：

- Instruct 模型天然理解指令格式
- 第一次生成就正确使用 `<think>`, `<search>`, `<answer>`
- 100% valid_traj from step 1
- 结论：选择正确的基座模型比做更多 SFT 更重要

**面试金句**：*"RL 训练的成功与否，80% 取决于基座模型是否理解任务格式。Instruct 模型零样本就能正确输出 XML 标签，Coder 模型 500 条 SFT 都不够。"*

## Q15: 如果重新做这个项目，你会怎么优化流程？

**当前流程的问题**：

1. 在宿主机 vLLM 0.8.5 上浪费了 99+200 步（~15 小时 GPU）才发现乱码问题
2. Docker 环境配置反复试错（OOM、工具服务器连接、pip install）
3. SFT 数据生成遇到 Wikipedia 限流

**优化后的流程**：

1. **第一步**：用 HuggingFace 直接测试模型对 Search-R1 格式的理解（5 分钟）
2. **第二步**：用 vLLM 直接测试（确认推理引擎兼容性，10 分钟）
3. **第三步**：小规模 RL 验证（10 步，确认 reward + tool + multi-turn 都正常）
4. **第四步**：大规模训练（200-1000 步）

**关键节省**：第 1-2 步可以避免在错误模型和错误 vLLM 上浪费 GPU 时间。

## Q17: 训练数据分布不均会有什么后果？你是怎么发现的？

**现象回顾**：训练 123 步后 Score 从 0.10 升到 0.44 但随后停涨，num_turns 从 2.28 降到 1.0。

**排查过程**：逐一排除可能性（reward 设计、模型能力、vLLM 兼容性）后，检查数据分布 → 发现训练集的前 79,168 条全是 NQ 单跳问题，后 73,485 条全是 HotpotQA 多跳问题——中间只有一条分界线，没有任何混合。

**后果**：

- 200 步 × batch_size=8 = 1,600 条全部是 NQ → 模型从未见过 HotpotQA
- num_turns=1.0 是对的——NQ 单跳不需要多轮搜索
- Score ≈ 0.44 是 NQ 的天花板——太简单，没有提升空间
- 多跳能力完全未训练

**教训**：**先检查数据分布再开始 RL 训练**——否则可能花几天时间在一个"假"任务上。

## Q18: RL 训练需要多少 epoch？和 SFT 有什么不同？

**SFT 需要完整 epoch**——模型在标注数据上做教师强制学习，每个样本通常被看到 1-3 次。

**RL 完全不同**——模型通过"尝试→观察 reward→比较"来学习，数据只提供问题和 ground truth。

| 维度  | SFT | RL (GRPO/DAPO) |
| --- | --- | --- |
| 学习方式 | 模仿正确回答 | 从 reward 中试错 |
| 数据作用 | 直接监督 | 只提供问题，答案用于 reward |
| 过拟合风险 | 大（死记硬背） | 小（模型在探索） |
| epoch 必要性 | 必须  | 非必须 |
| 多 epoch 风险 | 中等  | 背答案会降低探索→reward 虚高 |

**RL 的 epoch 设计原则**：不是"学多少遍"，而是"每个问题给几次尝试机会"。

实验 2 的设计：
`1526 条 × 2 epochs = 382 步 × 8 batch = 每个问题被看到 2 次`

- NQ 单跳：1 次学会格式，1 次学会搜索
- HotpotQA 多跳：1 次发现需要多轮搜索，1 次学会搜索链

### 当前数据集（短轨迹 ~700 tok、n=4、reward∈ {0.1,1.0}）

| 问题（记录中已出现） | DAPO 怎么治 | Dr.GRPO 为什么治不了 |
| --- | --- | --- |
| **全同分空转**（Exp1 卡 99 步 `pg_loss=0`：全 0.1 格式分） | `filter_groups` 按组内 std=0 **整组丢弃**，不足 batch 时**继续生成直到攒够有差异的组** | `adv=R-mean=0.1-0.1=0`，不去除法也还是零 |
| **冷启动死锁**（不会格式→全 0.1→学不会） | 强制每个更新步的组里至少有一条 ≥1.0，把"碰运气"变成"结构化等待" | 不做数据选择，纯靠某条回复偶然答对才打破僵局 |
| **行为坍缩风险**（num_turns 2.28→1.0 这类过早收敛） | Clip-Higher：正 advantage 裁剪上限高于负侧，防止稀疏环境下正 token 梯度被对称裁剪压没 | 完全不碰裁剪机制 |

⚠️ 注意：Exp2 的成功主要是换模型/数据 + GRPO 的 `/std` 在"偶然答对时放大 1.5 倍"立功。
短轨迹下 DAPO 是**增强鲁棒性**而非必需；Dr.GRPO 连鲁棒性都谈不上。

### 假设的长序列、多跳轨迹数据（>10K token、多轮搜索、小 batch）

这是 Dr.GRPO 的"主场"，但它的收益与 DAPO 仍然**零重叠**——两者治的是正交的极端问题：

| 问题  | DAPO 怎么治 | Dr.GRPO 怎么处理（或无能为力） |
| --- | --- | --- |
| **绝大多数 group 全失败**（多跳难，组内全 0） | 过滤 + 重新生成，丢弃海量无信息轨迹，省算力且去噪 | 全部训练；去掉 `/std` 后凑巧出现的微小伪差异（0.1 vs 0.15）被**原值保留 → 注入噪声** |
| **reward 随轨迹长度/token 数强相关** | Sparse Reward Normalization 压到 [0,1]，压缩离群尺度 | 保留原始尺度——单条长轨迹的 adv 直接主导更新，10K+ token 上更易梯度爆炸 |
| **中间步骤几乎零信号**（只有终局 EM 奖励） | Token-mean 聚合 + 只保留有终局成功的组，梯度集中到有效 token | 不改造损失聚合，无数据选择 |
| **超长/截断轨迹**（无限搜索循环） | overlong_buffer 惩罚超长轨迹，学"适可而止" | 无长度控制，截断时 reward 尖峰直接进梯度 |

### 面试金句

*"Dr.GRPO 只在'组内已有差异但 std 估计不可靠'这一种窄场景有效；DAPO 覆盖所有
'组内无差异'和'跨组数据配平'类问题。两者不互斥——长序列场景的正解是
DAPO 打底（采样/裁剪/归一化）+ `norm_adv_by_std_in_grpo=False` 叠加，而不是二选一。"*

## Q20: DAPO 与 Dr.GRPO 各自的贡献 + 简历怎么写？

### 两篇论文的原始贡献（面试核对用，别张冠李戴）

| 论文  | 出处  | 唯一/核心贡献 |
| --- | --- | --- |
| **DAPO** | arXiv 2503.14476 | 在 GRPO 之上改 4 件事：① Clip-Higher 非对称裁剪 ② Dynamic Sampling（filter_groups）③ Token-level 损失聚合 ④ Sparse Reward Normalization。**不含**"去掉 std 归一化" |
| **Dr.GRPO** | arXiv 2503.20783 | 独立论文，唯一贡献：GRPO 的 `adv=(R-mean)/std` 在中长轨迹/小 batch 下 std 估计不可靠、会放大噪声，应改为 `adv=R-mean` |

⚠️ **`norm_adv_by_std_in_grpo` 开关本身不是 DAPO 原算法自带的**——它是 verl 框架的配置项，对应 Dr.GRPO 的贡献。把开关说成"DAPO 原算法特性"会被懂行的人抓。

### 叠加后的语义（一句话）

> DAPO 管"选哪批数据"（过滤全同分组、防坍缩、归一化 reward），
> Dr.GRPO 管"这批数据内的 advantage 要不要除 std"，分属流水线**不同层**，可叠加。

Work output：

**中文**

> 在 8×A100 上基于 veRL 搭建 Agentic RL 管线（Qwen2.5-14B + vLLM + Wikipedia 检索工具），
> 训练多跳 QA 搜索智能体，GRPO 将得分从 0.10 提升至 0.44（峰值 0.72）。
> 系统性拆解两类 RL 失败模式——冷启动空转（组内 reward 全同分导致 pg_loss=0）
> 与小 batch/长轨迹下组内 std 估计不可靠——据此设计 **DAPO × Dr.GRPO 组合方案**：
> DAPO 动态采样（filter_groups）丢弃零方差分组、非对称裁剪（Clip-Higher）
> 防止稀疏奖励下行为坍缩，并叠加 Dr.GRPO 的去 std 归一化优势
> （`norm_adv_by_std_in_grpo=False`）以稳定长轨迹训练。

**English**

> Built an agentic RL pipeline on veRL (Qwen2.5-14B + vLLM + Wikipedia search tool)
> for multi-hop QA, improving score from 0.10→0.44 (peak 0.72) via GRPO.
> Diagnosed two RL failure modes—cold-start stagnation (uniform group rewards →
> zero PG loss) and unreliable group-std in small-batch/long-horizon rollouts—and
> designed a combined **DAPO × Dr.GRPO** recipe: DAPO dynamic sampling (filter_groups)
> to discard zero-variance groups and asymmetric Clip-Higher to prevent token collapse,
> layered with Dr.GRPO's std-free advantage normalization
> (`norm_adv_by_std_in_grpo=False`) for stable long-trajectory training.

**用法注意**：用"设计并论证"，别写"实验验证提升 X%"——目前实际跑的是 GRPO，
DAPO×Dr.GRPO 组合是设计论证、尚未跑通出数字。

把"设计并论证"换成"**实验验证可行，消融对比显示收敛更快/更稳，score 提升 X%**"，
并补上组合 vs 单独 DAPO vs 单独 Dr.GRPO 的具体数字。

### 面试核对点（套用本结构必被问）

1. `norm_adv_by_std_in_grpo`：

- `True`=GRPO：`(R-mean)/std`
- `False`=Dr.GRPO：`R-mean`

2. 叠加语义：DAPO 管数据选择，Dr.GRPO 管组内 advantage 缩放，不同层不冲突。
3. 冷启动时两者都救不了"组内全同分"——只有 DAPO 的 filter_groups 能救。
4. Dr.GRPO 的唯一适用窗：组内有差异、但 std 估计不可靠（小 batch / 长轨迹）。

---

*Rollout 与 Training 物理解耦** 的分布式架构。通过 Ray 统一调度 8 卡 A100 的计算资源，规避传统单节点 RL 显存不足和推理吞吐低下的问题。

> Idea: 生成轨迹的推理引擎 和 更新参数的训练引擎 分开，
> Ray 负责集群资源调度和进程管理，避免推理慢、训练 OOM 互相拖累。

```
[ Ray Cluster (8x A100) ]
 ┌────────────────────────────────────────────────────────┐
 │ ┌──────────────────────┐      ┌──────────────────────┐ │
 │ │   Actor / Rollout    │      │    Learner Engine    │ │
 │ │ (vLLM Engine, 2xGPU) │      │ (FSDP / DS3, 6xGPU)  │ │
 │ └──────────┬───────────┘      └──────────▲───────────┘ │
 └────────────┼─────────────────────────────┼─────────────┘
              │ 1. Generate Trajectory      │ 3. Update Policy
              ▼                             │
   ┌────────────────────┐                   │
   │   Custom Tool/     │                   │
   │  Environment Loop  │                   │
   └──────────┬─────────┘                   │
              │ 2. Compute Rewards          │
              ▼                             │
   ┌────────────────────┐                   │
   │   Reward Service   ├───────────────────┘
   │   (Flask/FastAPI)  │
   └────────────────────┘
```

> 

这里有两种部署形态可以选择：

1. Colocated / Hybrid Engine（veRL 默认）：同一批 GPU 上，rollout（vLLM）与训练（FSDP）分时复用，rollout 时把训练权重 offload，训练时收回。显存利用率最高，对短文本而言很合适，如果训练状态。	7B 首选，8 卡全用于 hybrid engine。
2. Disaggregated（解耦）：少量卡常驻 vLLM，其余卡常驻 FSDP（如 Gemini 的 2+6）。省去权重搬运，但推理卡在训练时闲置。	仅当模型大 / 推理是瓶颈时考虑，我们的长链思考任务会占用大量 KV Cache，这部分交给vLLM的独占显存。

> Anchor:2 卡 vLLM + 6 卡 FSDP 常驻是解耦式。为什么不用 colocated？单节点 8 卡训 7B，colocated 通常吞吐更高，因为解耦式在训练阶段那 2 张推理卡在空转。
> 权衡点：解耦式省 reshard 开销但浪费卡；colocated 省卡但有权重 offload/reload 开销。而且我们的穿刺实验要上集群，到时候的模型 offload/reload 会成为严重瓶颈。

veRL 官方在底层将训练后端抽象成了通用的 Engine/Worker 接口，以通过配置参数直接切换底层训练后端。
深度集成 Megatron-LM（替换strategy就可以） 或 MindSpeed-LLM（适配中，不过老模型还挺顺利的），可以在集群替换掉 FSDP。

### 2.1 核心实验流程

1. **数据就绪**：使用 HotpotQA 的本地 Wikipedia 支撑段落，利用 BM25 算法在本地搭建轻量高并发检索服务（`search_server.py`），避免联网延迟。
2. **轨迹生成（Decoupled Rollout）**：
   * Actor（在 vLLM 引擎中运行）生成文本。
   * 当遇到 `</call:search>` 停止符时，vLLM 暂停生成。
   * 通过 veRL 的 `AgentLoop` 机制拦截输出，解析检索词并请求本地检索服务。
   * 将检索结果包裹在 `<observation>...</observation>` 中拼回 Prompt，唤醒 vLLM 继续生成，直到输出 `<answer>` 标签。
3. **异步打分**：轨迹生成完毕后，异步发送至打分服务（`agent_reward.py`），进行**奖励塑造（Reward Shaping）**计算。
4. **模型更新**：Learner 收集一个 Batch 的轨迹和奖励值，利用 FSDP 引擎对 Policy（Actor）模型进行梯度更新。

---

## 3. 算法选择与技术细节

本方案没有使用 PPO，而是尝试了 GRPO 后选择了改良的 Dr. GRPO。

### 3.1 Why GRPO?

1. **极大地节省显存**：PPO 需要维护一个与 Actor 相同规模的 **Critic（评论员）模型** 来预测状态价值（State Value），这在 8 卡节点上微调 7B+ 模型时极易造成 OOM。GRPO 取消了 Critic 模型，将显存和计算资源全部释放给 Actor。
2. **相对优势估算**：对每一个输入 $Prompt$，让模型并行 Rollout 产生一组成员（采样数 $G = 5$）。通过这组轨迹的奖励均值和标准差，计算组内的相对优势（Advantage）：
   
   $$
   A_i = \frac{r_i - \text{mean}(R)}{\text{std}(R)}
   $$
   
   这自然地建立了一个基线（Baseline），极大地稳定了强化学习的梯度更新。

### 3.2 现代改良：从 vanilla GRPO → Dr.GRPO / DAPO

vanilla GRPO 的两个已知偏置，务必知道：

1. **难度偏置**：除以 $\operatorname{std}(\mathbf{r})$ 会放大"极易/极难"题的权重。Dr. GRPO 取消这个缩放，平等对待所有题目。→ veRL 配置 `algorithm.norm_adv_by_std_in_grpo: False`。
2. **长度偏置**：按序列长度平均会让"更长的错误答案"被低估惩罚。GRPO 按序列长度归一化会导致更长的错误回答被惩罚不足。Dr.GRPO 改用全局常数归一化以消除长度偏置。

**DAPO 的四件套**（ByteDance，基于 verl 实现，长 CoT / 多轮场景强烈推荐）：Clip-Higher（非对称裁剪、上界更高）、Dynamic Sampling（重采样至组内有对有错）、Token-Level Policy Gradient Loss、Overlong Reward Shaping（惩罚过长回答）。其中：

- **Clip-Higher** 治**熵坍缩**：初期观察到熵坍缩现象，通过增大重要性采样比的上裁剪范围来缓解。
- **Dynamic Sampling** 就是 §索引③ 里 σ=0 空梯度的正解。
- **Overlong Reward Shaping** 是软惩罚：设一个最大长度，对超过阈值（如 4096）的多余 token 温和降分，这种"软惩罚"避免模型啰嗦又不过于严厉。

### 3.3 关键工程：Observation Token Masking（多轮 RL 的命门）

在计算 log-prob 和 policy loss 时，**必须对 `<information>` 内的检索 token 置零 mask**，只对模型自己生成的 think/search/answer 计 loss。这是 Search-R1 最关键的创新之一：RL 期间检索内容被排除在优化之外，只有模型自己的推理参与更新，迫使模型"对检索结果做推理"而非"照抄"，从而提升稳定性与泛化。不做 mask 会让模型去拟合外部网页内容，导致学偏/坍缩。实验显示做 masking 训练更稳、提升更大。

### 3.4 多轮 credit assignment

轨迹级单标量 reward 广播到所有 turn 是最简做法，但 GRPO 在多轮设定下被广泛报告不稳定。更细的做法是 **turn-level 优化**：整条轨迹含多个"模型生成 + 环境反馈"回合，在环境反馈上做优化会引入不稳定，因此解耦模型生成 $o_t$ 与环境反馈 $f_t$，只在 $o_t$ 上做定向优化。代表工作：RAGEN 的 StarPO-s 用比例化轨迹过滤，GiGPO 结合状态级与轨迹级优势，MT-GRPO 展示 turn-level credit assignment 的收益。

### 3.5 推荐超参（8×A100 / 14B / HotpotQA）

| 项 | My Choice  | **Opus 4.8 建议值** | 理由 |
| :--- | :--- | :--- | :--- |
| Group size $G$ | 8 | **8~16** | 统计更稳；配合 dynamic sampling |
| Advantage std 归一 | 除 std | **关闭**（Dr.GRPO）或保留但知其偏置 | 消除难度偏置 |
| Dynamic sampling | 无 | **开启** | 消灭零方差空梯度组 |
| KL coef | 0.001 | **0（DAPO 路线）或 1e-3（保守路线），二选一并能解释** | 见下 |
| Actor LR | 1e-6 | 1e-6 ~ 5e-7 + linear warmup | Agent 训练要保守 |
| max_response_length | 6144 | 完全没道理我删了 | 多轮会累积，对单轮考虑装下 think + search + answer 的空间 |
| max_prompt_length | 4096 | Cover HotpotQA 和我们的问题即可 |
| Loss 聚合 | 未提 | token-level（DAPO） | 长序列更精确 |
| max_turns | 6 | | 和实际业务保持一致 |

> **KL 二选一**：
> 路线 A（DAPO/Dr.GRPO）——"结果奖励可验证 + 参考模型已是好起点，去 KL 让策略充分移动、避免拖后腿"；DAPO 在其方法中移除了 KL 散度。
> 路线 B（保守）——"保留小 KL 防止在稀疏奖励早期策略崩溃/复读，代价是探索受限"。

### 3.6 对多卡集群的放缩方案

将 **DeepResearch Agent（多跳问答/搜索智能体）** 的 GRPO 训练从 **单机 8 卡 Qwen-2.5-14B** 迁移到 **64 卡集群 Qwen-2.5-72B**，这不仅是卡数和参数量的缩放，更涉及 **Agent 交互特异性（多轮调用与环境通信）**、**长上下文显存墙** 以及 **72B 超大模型分布式拓扑** 的全面升级。

以下为您整理的**必须修改的内容清单与技术理由**，分为 4 大模块：

---

### 一、 核心超参数对照与修改清单（参数表）

| 参数 / 配置项 | 单机 8 卡 (Qwen-2.5-14B) | 64 卡集群 (Qwen-2.5-72B) | 修改理由与避坑要点 |
|---|---|---|---|
| **并行拓扑 (Training Parallelism)** | `TP=1` 或 `TP=2`, `FSDP (DP=8)` | **`TP=8` (单机内) + `FSDP/ZeRO-3 (DP=8)` (跨机)** | 72B 模型权重加优化器状态超 1.2TB。**TP 必须严格锁在单机 8 卡内（走 NVLink）**，跨机走 DP/FSDP。若跨机做 TP 会因机间带宽不足导致通信崩塌。 |
| **推理并行 (Rollout Engine)** | 8 卡启动 1 个 `vLLM (TP=8)` 或 4 个 `TP=2` | **8 个独立 vLLM 实例，每台机器 1 个 (`TP=8`)** | 72B 在生成时如果 TP 不为 8，单卡 KV Cache 显存会直接 OOM（尤其在多跳 Agent 生成长文本时）。 |
| **学习率 (Learning Rate)** | $4 \times 10^{-6} \sim 6 \times 10^{-6}$ | **$1 \times 10^{-6} \sim 2 \times 10^{-6}$ (显著调小)** | 72B 模型的优化曲面非常敏感，大模型的策略更新容错率极低，过大 LR 会瞬间引发输出崩溃（乱码/死循环）；同时 Warmup 步数要重新换算为总 Steps 的 $3\%\sim 5\%$。 |
| **微批次 (`micro_batch_size`)** | 1 ~ 2 | **强制设为 1，开启 Activation Checkpointing** | 72B 在长上下文反向传播时，显存中保存的中间激活值极其庞大。设为 1 并配合选择性重计算是防 OOM 的底线。 |
| **梯度累积步数 (`grad_accum_steps`)** | 较高（例如 8 ~ 16） | **降低（例如 2 ~ 4）** | 64 卡时跨机 DP 已经提供了 8 的数据并行放大，若保持原累积步数，会导致全局更新步数过少、策略反馈滞后。 |
| **组采样大小 ($G$ / Group Size)** | $G = 4 \sim 8$ | **建议 $G = 8$ (保持稳定)** | 很多团队在扩卡时盲目将 $G$ 调到 16 或 32，对于 72B + Agent 多轮长链，这会导致生成阶段耗时成倍增加，形成严重的 Rollout 瓶颈。建议保持 $G=8$ 换取更快的迭代频率。 |
| **更新轮数 (`ppo_epochs`)** | 1 ~ 2 | **强制设为 1 (纯 On-policy)** | 72B 表达能力极强，同一批 Agent 数据反复迭代超过 1 轮会极易产生过拟合和策略漂移，彻底破坏基座的泛化检索能力。 |

---

### 二、 DeepResearch Agent 专属的算法与数据修改（至关重要）

#### 1. 必须对环境返回内容做 Loss 掩码（Observation Masking）

* **做法**：在多跳 Agent 交互轨迹中，Prompt 结构包含模型生成的 `Thought`、`Tool Call` 以及环境返回的 `Tool Response (Search Snippets)`。**反向传播计算 Loss 时，必须对 `Tool Response` 进行 Mask（即 Token 权重置 0）**。
* **理由**：14B 时代可能因为数据量小被忽视；但在 72B 上，若不对检索回来的网页/维基片段做掩码，72B 的强拟合能力会尝试去**预测/背诵搜索结果**，导致策略梯度被严重污染，模型丧失推理和搜索能力。

#### 2. 多步搜索的步数惩罚与死循环截断（Step Penalty & Loop Break）

* **做法**：
  1. 引入单轮负奖励：每执行一次 `<search>` 赋予微小的步数惩罚（如 $-0.02$）；
  2. 强制最大搜索轮数（Max Turns 如 $4 \sim 6$ 轮），超过上限直接强行进入 Answer 阶段。
* **理由**：72B 模型在 RL 探索初期，容易为了追求“信息完整度”而陷入**无休止的搜索死循环**，极度拖慢 Rollout 速度并占满上下文长度。

#### 3. 截断过滤（Overlong Filtering）与掩码

* **做法**：因为环境返回的搜索内容长度不可控，一旦达到 `max_seq_len`（如 16k/32k）被硬截断的轨迹，**直接在本次更新中 Mask 丢弃，不计算梯度**。
* **理由**：多跳问答如果在推理中途被强行掐断，其最后的 Advantage 信号是极其嘈杂的负样本，72B 会因此学到“放弃思考”或“草率输出”的次优策略。

---

### 三、 系统与工程基础设施修改

#### 1. 外部检索服务（Retriever）的并发吞吐能力

* **修改项**：单机 8 卡时，瞬时搜索并发（QPS）只有几百；**64 卡 72B 触发多跳并发时，瞬时 QPS 会达到数千至上万**。
* **做法**：
  * 切勿直接调用公网搜索 API（会瞬间触发 Rate Limit 或被封禁）；
  * 离线搭建基于 Elasticsearch / Qdrant 的 **本地 Wikipedia 检索集群**，并配置独立的多节点负载均衡。

#### 2. Rollout 阶段的机间通信优化（Weight Sync）

* **修改项**：开启 Actor 训练引擎向 Rollout 推理引擎的**跨机模型权重广播加速**。
* **理由**：14B 单机内同步权重几乎瞬时完成；72B 跨 8 台机器同步权重（约 144GB）若走普通的 TCP 会导致每次 Rollout 前等待数分钟。必须确认配置：
  * `NCCL_IB_DISABLE=0`（强制启用 InfiniBand/RoCE）；
  * 设置训练框架（如 `verl`）的权重同步方式为基于 Ray/NCCL 的分布式广播通道。

#### 3. 应对长尾效应（Tail Latency / Straggler Problem）

* **修改项**：在 vLLM/SGLang 侧开启 **动态批处理（Continuous Batching）** 和 **组内异步超时截断**。
* **理由**：Agent 执行 HotpotQA 等多跳任务时，各样本的搜索步数和思考长度差异巨大（有的 2 步搜完，有的 6 步搜满）。在 64 卡集群上，最慢的样本会卡住整个集群的同一步伐。必须设置超时直接终止该样本的 Rollout 并赋予兜底奖励。

---

### 四、 快速部署前自检 Checklist

- [ ] **拓扑**：训练与推理均设置为单机内 `TP=8`，跨机纯走 `DP/FSDP`？
- [ ] **掩码**：Agent 轨迹中的外部搜索结果（Observation）是否已在 Loss 计算中被完全 Mask？
- [ ] **学习率**：是否已从 14B 的 $5\times 10^{-6}$ 安全下调至 72B 的 $1.5\times 10^{-6}$ 附近？
- [ ] **更新轮数**：`update_epochs / ppo_epochs` 是否设为 1？
- [ ] **检索支撑**：本地知识库/检索集群是否能抗住 64 卡带来的高并发查询冲击？
- [ ] **显存防爆**：`micro_batch_size=1` 且已开启 `Activation Checkpointing`（选择性重计算）？

---

> 但当前主流的检索类 Agentic RL 恰恰相反——Search-R1[1] 明确采用"简单的、基于结果（outcome-based）的奖励函数"，并证明这比复杂奖励更稳、更能泛化。Search-R1 optimizes LLM reasoning trajectories with multi-turn search interactions, leveraging retrieved token masking for stable RL training and a simple outcome-based reward function.
> 
> 其核心论点是：复杂的神经奖励模型容易被钻空子（gamed）或需要过度工程；只需定义答案正确性即可扩展到新领域。你手动加的每一个 shaping 项（尤其 diversity、step penalty）都是一个可被 hack 的攻击面。面试正确答案不是"我设计了 5 个奖励"，而是"我优先用 outcome reward，只保留最小格式约束，把复杂偏好交给相对优势去自然涌现"。 我在修订版把 shaping 降级为"可选辅助项 + 明确风险标注"。

### 4.2 如何防御 Reward Hacking（奖励作弊）？

* **作弊表现 A**：模型学会了疯狂检索，故意拉长交互步数，在检索历史中反复塞入极微小差异的词来刷前期的 $R_{format}$ 和探索奖励。
  * *防御方案*：引入**二次惩罚（Quadratic Step Penalty）**，且步数惩罚与步数呈二次非线性关系；同时对 Query 进行强语义去重约束（Rouge-L 阈值）。
* **作弊表现 B**：模型在回答中疯狂堆砌所有可能相关的实体，试图在 $R_{accuracy}$ 匹配中“蒙混过关”。
  * *防御方案*：强制提取 `<answer>...</answer>` 内部的单一实体，抛弃外部的所有冗余废话，使其无法进行模糊匹配。

---

## 5. 典型面试挑战题与技术对线

### 挑战 1：你们如何解决 RL 训练的“冷启动（Cold Start）”问题？如果模型一上来随机探索，拿不到任何正反馈怎么办？

* **答**：这是 Agentic RL 最经典的痛点。我们的解决方案是 **SFT Bootstrapping（冷启动微调）**。我们没有直接让基础模型去跑 RL，而是先利用少量的精选高质量 ReAct 多步数据（约 1000~2000 条），对模型进行了一个 epoch 的 SFT。这让模型先具备“只要看到问题，就一定会尝试用 `<call:search>` 交互”的温和先验概率。在这个基础上再进行 GRPO 训练，可以保证首个 Batch 的采样中，至少有 20% 以上的轨迹能顺利拿分，从而让 Policy 梯度的更新方向能够立足。

### 挑战 2：外部环境（如 Search Tool 报错、超时或服务崩溃）是不可导的，强化学习怎么把梯度回传给模型？

* **答**：强化学习（如 PPO/GRPO）属于 **无模型（Model-Free）强化学习**。它本身就不需要对外部环境进行显式求导。梯度回传的本质是：
  1. 策略模型做出动作（产生 Token）；
  2. 环境给出一个标量 Reward；
  3. 算法根据策略梯度定理，利用优势函数对产生高 Reward 对应的 Token 的 **对数概率（Log Probabilities）** 进行放大，对低 Reward 的进行抑制。
     因此，即使环境完全是一个不透明的黑盒（甚至可以是人类反馈），只要能输出一个标量 Reward，就完全不影响梯度的正常更新。

### 挑战 3：在多步交互中，环境返回的 Observation 是外部产生的（非模型生成），计算梯度时如何处理这部分 Token？

* **答**：这是一个关键的工程细节。对于多步 Agent 来说，轨迹中会混入大量的 `Observation`（外部网页内容）。在计算 Actor 模型的 `log_prob` 和计算 Policy 梯度（Loss）时，**必须对 Observation 部分的 Token 进行 Mask（置零）**，只对模型自己生成的 `Thought`、`Action` 和 `Final Answer` 的 Token 计算 Loss 并回传梯度。如果不对这部分进行 Mask，模型会尝试去拟合和预测外部环境返回的数据，从而导致严重的学偏或策略崩溃。

---

## 6. 实践避坑指南与解决方案

### 6.1 显存泄漏与 vLLM 内存碎片问题

* **现象**：由于多步 Agent 的序列长度（Context Length）随着交互轮数不断拉长（最高可能逼近几十k），在进行几万步采样后，Ray 集群中负责 vLLM Rollout 的节点会频繁出现无预警的 OOM 崩溃。
* **原因**：vLLM 的 KV Cache 分配在多步动态增长时，由于并发请求的不同步，可能产生严重的内存碎片。
* **方案**：在 veRL 的配置文件中，合理调低 `vllm_gpu_memory_utilization`（例如从 0.9 降到 0.6），并在 `ray_config` 中设置 `max_concurrency` 限制，同时启用 vLLM 的 `enforce_eager=True` 强制执行即时显存回收，虽然会牺牲极少部分的吞吐量，但极大提升了大规模训练的稳定性。

### 6.2 训练发散与“NaN Loss”

* **现象**：训练到第 2 个 Epoch 时，WandB 看板上的 Loss 突然变成 NaN，Actor 模型的输出全部变成重复的空格或标点符号（模型崩溃）。
* **原因**：优势函数 $A_i = \frac{r_i - \mu}{\sigma}$ 中，当组内所有成员的奖励完全一样时，分母标准差 $\sigma = 0$。如果没有对其进行极小值保护（$\epsilon$ epsilon），就会导致梯度变成无穷大，破坏参数。
* **方案**：在代码中计算相对优势时，务必对标准差加上一个微小的保护项：`std = np.std(R) + 1e-8`。此外，将 `kl_ctrl.kl_coef` 从 `0.0005` 适当提高到 `0.001` 或 `0.002`，加强对模型背离参考模型（Reference Model）的惩罚约束。

# Reference

[1] Search-R1: https://arxiv.org/abs/2503.09516

---

# 修订说明：Gemini 版的 5 个致命问题（面试会被打的点）

<details>
<summary><b>① 最重要：重度 Reward Shaping 与当前 SOTA 方向相反（会被直接质疑"你在制造 Reward Hacking 表面积"）</b></summary>

Gemini 设计了 5 项密集奖励（format / validity / diversity / step / accuracy）。但当前主流的检索类 Agentic RL 恰恰相反——**Search-R1 明确采用"简单的、基于结果（outcome-based）的奖励函数"**，并证明这比复杂奖励更稳、更能泛化。Search-R1 optimizes LLM reasoning trajectories with multi-turn search interactions, leveraging retrieved token masking for stable RL training and a simple outcome-based reward function.

其核心论点是：复杂的神经奖励模型容易被钻空子（gamed）或需要过度工程；只需定义答案正确性即可扩展到新领域。你手动加的每一个 shaping 项（尤其 diversity、step penalty）都是一个可被 hack 的攻击面。**面试正确答案不是"我设计了 5 个奖励"，而是"我优先用 outcome reward，只保留最小格式约束，把复杂偏好交给相对优势去自然涌现"。** 我在修订版把 shaping 降级为"可选辅助项 + 明确风险标注"。

</details>

<details>
<summary><b>② GRPO 优势公式除以 std 会引入"难度偏置"，长度归一化会引入"长度偏置"（Dr.GRPO 的核心批评）</b></summary>

Gemini 直接用 $A_i=(r_i-\mu)/\sigma$ 并把 NaN 归因于"σ=0 除零、加 ε 解决"。这只对了一半。真正的学术批评是：用 std(r) 归一化会给标准差低的题目（极易或极难）不成比例的权重，Dr. GRPO 取消这个缩放以平等对待所有题目。同时 GRPO 的长度归一化会带来长度偏置，Dr.GRPO 改用全局常数归一化来消除它。而 σ=0 的真正工程解法不是加 ε，而是 **DAPO 的 Dynamic Sampling**（见 ③）。

</details>

<details>
<summary><b>③ σ=0 / NaN 的真正解法是 Dynamic Sampling，不是加 ε、也不是调高 KL</b></summary>

当一组 rollout 全对或全错时，优势全为 0，这一组**贡献零梯度**（不是 NaN，加 ε 后是 0/ε=0）。DAPO 的正解是重采样直到组内有对有错。Dynamic Sampling：重采样直到一个 group 同时包含正确与错误答案。该策略过滤掉准确率为 1 和 0 的 prompt 组，保持每个 batch 中有效梯度的 prompt 数量一致。把 KL 系数调高来"救 NaN"是错误因果——KL 是防漂移的，不解决零方差。

</details>

<details>
<summary><b>④ G=5 太小；KL 是否保留是有争议的设计选择，不能想当然</b></summary>

G=5 统计意义偏弱，检索类任务常用 8~16。更重要：**DAPO 和 Dr.GRPO 都直接去掉了 KL 项**。DAPO 提出了一系列基于 GRPO 的改进，包括 dynamic sampling、token 级梯度损失、clip-higher、overlong reward shaping 以及移除 KL 散度。去 KL 的理由是：ground-truth reward 已经是硬约束，R1-zero 类训练希望模型分布大幅移动，KL 反而拖后腿。你要能讲清"我为什么保留/去掉 KL"，而不是背一个 0.001。

</details>

<details>
<summary><b>⑤ 若干工程表述不准确</b></summary>

- **`enforce_eager=True` 不是"即时显存回收"**：它是**关闭 CUDA Graph 捕获**，从而省掉 CUDA Graph 占用的显存、避免其固定内存块带来的碎片，代价是牺牲吞吐。表述要改对，否则面试一问原理就露馅。
- **Max Response Length=2048 与"上下文逼近几十 k"自相矛盾**：要区分「单轮生成上限 / 整条轨迹总长（prompt+所有 turn）」两个概念。
- **观测 token mask** 是对的，但要点名这是 Search-R1 的 **retrieved token loss masking**，并给出实验依据：应用 retrieved token masking 带来更大的模型提升、缓解非预期优化、训练更稳定，且始终优于不做 masking 的变体。

</details>

---

# LLM Agentic RL（Deep Research / ReAct）训练全栈实战手册 v2

> 面向 **8×A100 单节点**、基于 **veRL（Ray + vLLM/SGLang + FSDP）** 的多轮 Agent（ReAct/Deep Research）强化学习。
> 阅读约定：不熟悉 RL 的读者，先看每节开头的 **"一句话直觉"**；准备面试的读者，重点看 **▶ 面试锚点**。

---

## 1. 问题定义与 MDP 建模

**一句话直觉**：让模型在"想—查—看结果—再想"的循环里，靠"答对了给糖、答错了打手"来学会高效检索并给出有据可查的答案。

### 1.1 核心目标

训练具备 **Deep Research（多步检索推理）** 能力的 Agent：面对多跳、模糊问题（如 HotpotQA）时，自主决定何时检索、检索什么、如何过滤与纠偏，并产出高准确、带真实引用的终局答案。这是一个"由任务成功而非僵化预定义流程驱动、学会智能推理与检索"的系统，从而能泛化到新问题与新领域。

### 1.2 MDP 形式化

把多轮 ReAct 建模为有限步 MDP：

| 元素 | 定义 |
| :--- | :--- |
| 状态 $s_t$ | 到当前轮为止的完整上下文：初始 prompt + 历史 `` + `` + `` |
| 动作 $a_t$ | 模型本轮生成的 token 序列：内部动作 `…`（推理）+ 外部动作 `query`（触发环境） |
| 转移 $P(s_{t+1}\mid s_t,a_t)$ | 生成外部动作时暂停，检索服务执行并把结果包进 `…` 拼回，形成 $s_{t+1}$ |
| 奖励 $r$ | 通常在输出 `…` 或达到 max-turns 时对整条轨迹给一个标量（见 §4） |

> ▶ **面试锚点**：面试官常问"这跟单轮 RLHF 有何不同？"。答：状态在**转移中被环境注入了非模型生成的 token（observation）**，因此必须做 **loss masking**（§3.3）；且奖励是**轨迹级稀疏信号**，credit assignment 更难（§3.4）。

---

## 2. 系统架构

**一句话直觉**：把"生成轨迹的推理引擎"和"更新参数的训练引擎"分开，Ray 调度，避免推理慢、训练 OOM 互相拖累。

### 2.1 两种部署形态（必须能对比，面试高频）

| 形态 | 说明 | 8×A100 建议 |
| :--- | :--- | :--- |
| **Colocated / Hybrid Engine**（veRL 默认） | 同一批 GPU 上，rollout（vLLM）与训练（FSDP）**分时复用**，rollout 时把训练权重 offload，训练时收回。显存利用率最高。 | **7B 首选**。8 卡全用于 hybrid engine |
| **Disaggregated（解耦）** | 少量卡常驻 vLLM，其余卡常驻 FSDP（如 Gemini 的 2+6）。省去权重搬运，但推理卡在训练时闲置。 | 仅当模型大 / 推理是瓶颈时考虑 |

> ▶ **面试锚点**：Gemini 图里的"2 卡 vLLM + 6 卡 FSDP 常驻"是**解耦式**。面试官会问"为什么不用 colocated？单节点 8 卡训 7B，colocated 通常吞吐更高，因为解耦式在训练阶段那 2 张推理卡在空转"。你要能说出权衡点：**解耦式省 reshard 开销但浪费卡；colocated 省卡但有权重 offload/reload 开销**。

### 2.2 轨迹生成（Agent Loop）

veRL 已用 **Agent Loop** 统一多轮 rollout 接口。Agent Loop 是多轮 rollout 与 agentic RL 的通用接口：给定 prompt，运行用户自定义循环（调 LLM 生成、调工具……）并返回最终输出，随后计算 reward 作为 RL 训练轨迹。关键工程点：为避免等待工具返回时 GPU 空转，采用基于 asyncio 的协程机制异步执行各 rollout 请求，从而提升训练吞吐。并且 Server 提供基于 token 的 API，让 Client 维护"工具调用文本"与"LLM 返回 token"的对应关系，以便训练时输出正确的 token（及其 mask）。

你自研 ReAct 系统接入时，只需把它包装成一个 Agent Loop / `BaseTool`：对于自定义环境交互工具，可基于 `verl.tools.base_tool.BaseTool` 实现自己的工具。

```
[ Ray Cluster · 8×A100 ]
        ┌─────────────── Colocated Hybrid Engine ───────────────┐
        │  vLLM Rollout (分时)  ⇄  权重同步  ⇄  FSDP Actor (分时)  │
        └───────┬──────────────────────────────────────▲────────┘
                │ 1. Agent Loop 生成轨迹                  │ 4. GRPO 更新
                ▼ (async, tool 调用不阻塞 GPU)            │
        ┌────────────────────┐   ┌──────────────────────┴──────┐
        │ Local Search Server │   │ Reward: EM/F1 (规则可验证)   │
        │ (BM25, FastAPI)     │──▶│ + 最小格式校验 (可选 shaping) │
        └────────────────────┘   └─────────────────────────────┘
                        2. observation 拼回      3. 轨迹级标量 reward
```

---

## 4. 奖励设计（重写：Outcome-first，Shaping 谨慎）

**一句话直觉**：奖励越简单越难被钻空子。先只奖励"格式对 + 答案对"，其余偏好交给相对优势自然涌现，除非确有必要再加 shaping。

### 4.1 主推方案：最小化 Outcome-based Reward

$$
R_{total}=R_{format}+R_{accuracy}
$$

| 维度 | 设计 | 说明 |
| :--- | :--- | :--- |
| **格式 $R_{format}$** | ReAct 标签闭合且顺序合法：**0**（合法，不额外给分）；非法：小负分或直接判负 | 只做**门槛**，不做诱饵。避免模型刷格式分 |
| **准确率 $R_{accuracy}$** | 抽取最后一个 `` 内实体，与 GT 做 **EM / F1**：正确 +1，错误 0（或 -0.x） | 唯一主信号 |

Search-R1 证明：outcome-based reward + retrieved token masking 就能实现稳定、可扩展的学习，无需昂贵的数据/奖励工程。多项研究推荐 exact match 这类更简单的奖励，已被证明能有效激发推理能力。

### 4.2 可选辅助 shaping（每一项都标注 hack 风险）

> ⚠️ 这些是 Gemini 原方案的项，我保留但**明确标注风险**——面试官问到时，你要主动说"我知道这些会引入 hack 表面积，所以默认关闭 / 仅小权重 / 上线前 A/B 验证"。

| 维度 | 作用 | **它能被怎么 hack（必须主动讲）** |
| :--- | :--- | :--- |
| 工具合规惩罚 | 调不存在的工具/空 query 扣分 | 风险低，可保留 |
| 步数效率惩罚 $-c\cdot\text{steps}$ | 逼模型别磨蹭 | 模型学会"少查但瞎猜"，牺牲准确率换步数分。**建议改为：只在答对前提下才计效率，或用 overlong shaping 替代** |
| 去重/多样性惩罚 | 防死循环重复 query | 语义去重阈值本身可被 hack（凑到阈值边缘）；**且这是把"环境行为"塞进奖励，是最脆弱的一项，建议默认关闭** |

### 4.3 Reward Hacking 防御（升级版）

- **根因认知**：复杂/神经奖励最容易被 gamed。因此第一防线是**减少 shaping 项**，而非叠加更多惩罚。
- **答案堆砌 hack**：强制抽取 `<answer>` 内**单一实体**再匹配，禁止模糊命中（Gemini 这条是对的，保留）。
- **过长/复读 hack**：用 **DAPO Overlong Reward Shaping** 软惩罚替代二次步数惩罚——它对超过长度阈值的回答施加惩罚，帮助防止模型生成过长输出的同时保持解答质量。它还能防复读：overlong reward shaping 惩罚过长回答，从而防止模型陷入灾难性重复循环。
- **监控指标**：训练中盯 **response length 曲线 + 熵**。若长度暴涨而准确率不涨 → 长度 hack；若熵骤降 → 熵坍缩，上 Clip-Higher。

---

## 5. 面试挑战题（修订 + 扩充）

### Q1｜冷启动：随机探索拿不到正反馈怎么办？

**答**：三层解法。(1) **SFT Bootstrapping**：先用 1k~2k 条高质量 ReAct 轨迹 SFT 1 个 epoch，注入"看到问题就会 `<search>`"的先验，保证首个 batch 有非零命中率。(2) 但要知道 R1-zero 路线证明**可不用 SFT 直接 RL**：该范式直接对 base LLM 做 RL 而不依赖 SFT 作为前置步骤，因其简洁性与 RL scaling 现象而有吸引力。(3) **Dynamic Sampling** 保证每个 batch 都有"有对有错"的有效梯度组，等价于自动过滤掉"全错拿不到信号"的题。

### Q2｜环境不可导（工具超时/崩溃），梯度怎么回传？

**答**：GRPO/PPO 是 **model-free** 策略梯度，**不对环境求导**。链路是：策略产生 token → 环境吐一个标量 reward → 策略梯度定理按优势放大高 reward token 的 log-prob、抑制低 reward 的。环境是黑盒（甚至可以是人）都无所谓，只要能给标量 reward。工具报错时，把该轨迹按"失败"给对应 reward 即可（相当于负样本），照常更新。

### Q3｜Observation 是外部注入的 token，怎么处理？

**答**：**Retrieved token loss masking**（§3.3）。只有模型生成的 token 参与 loss，检索复制来的 token 梯度被 mask，防止 RL 更新经由被动观测传播，聚焦于决策与推理步。不 mask 会导致模型拟合外部内容、坍缩。

### Q4｜为什么不直接多加几个 shaped reward 把行为教到位？（新增，高频陷阱）

**答**：因为每个 shaping 项都是一个 hack 攻击面。SOTA（Search-R1）刻意用**最简 outcome reward** 就拿到 Qwen2.5-7B 相对 RAG 基线约 41% 的平均提升。我的原则是"能被 GT 验证的就用规则奖励，偏好交给组内相对优势涌现"，只在监控发现具体病态（如复读）时，用**针对性、可解释**的 shaping（如 overlong）打补丁。

### Q5｜训练发散 / NaN，你怎么定位？（修订：纠正 Gemini 的错误因果）

**答**：分清两种情况。(1) **零方差组**：一组全对/全错时优势全 0，是**空梯度**不是 NaN——正解是 **Dynamic Sampling** 重采样至有对有错，而非加 ε 或调 KL。(2) **真 NaN/坍缩**：多因熵坍缩或 lr 过大。查熵曲线——熵骤降上 **Clip-Higher**；它用分离的上下裁剪阈值，允许更好的探索。并降 lr、开 grad clip。**把"调高 KL"当 NaN 解药是错误因果**，KL 只管防漂移。

### Q6｜on-policy 还是 off-policy？rollout 和训练的策略不一致怎么办？（新增）

**答**：GRPO 近 on-policy，但一次采样跑多个 mini-batch 会造成 $\pi_\theta$ 与采样用的 $\pi_{\theta_{old}}$ 偏移，靠**重要性采样比 + clip**（PPO surrogate）约束。多轮 + 异步 rollout 下偏移更大，这也是 GSPO / turn-level 方法出现的动因（verl-agent 已支持 GSPO/DAPO/GiGPO 等，verl-agent 支持 GiGPO、GRPO、PPO、DAPO、GSPO、RLOO、REINFORCE++、dynamic sampling 与 clip-higher。）。

---

## 6. 避坑指南（修订）

### 6.1 vLLM 显存碎片 / 多轮长上下文 OOM

- **区分两个长度**：单轮生成上限（如 1024）与整条轨迹总长（prompt+所有 turn，可达 8k~16k）。KV Cache 按**总长**预算。
- **方案**：调低 `gpu_memory_utilization`（0.9→0.6~0.7）；限制并发；`max_num_seqs` 设合理值。
- **`enforce_eager=True` 的正确表述**：它**关闭 CUDA Graph 捕获**，省掉 CUDA Graph 固定占用的显存、减少其带来的碎片，代价是牺牲部分吞吐——**不是"即时显存回收"**。稳定性优先时可开，追吞吐时关掉并改用分页 KV 优化。

### 6.2 训练发散 / NaN（见 Q5，此处给 checklist）

1. 开 Dynamic Sampling（治零方差空梯度）
2. `norm_adv_by_std_in_grpo: False`（治难度偏置）+ token-level loss（治长度偏置）
3. 熵坍缩 → Clip-Higher（`clip_ratio_high` 抬到 ~0.28）
4. 优势/奖励做数值保护 + `grad_clip`，lr 保守 + warmup
5. 盯 WandB：reward、response_length、entropy、KL 四条曲线联动看

### 6.3 评测（Gemini 缺失，务必补，面试常问）

- 用 **held-out** 集（HotpotQA dev）报 EM/F1，别只看训练 reward（reward 涨≠泛化涨）。
- vLLM 批量推理对 batch size 敏感：用 vLLM 做贪心解码（temperature=0）保证可复现；由于 vLLM 批量推理在不同 batch size 下对同一输入产生不同输出，需固定验证 batch size 并独立评测每个数据集以保证一致。
- 额外报 **平均 turn 数 / 检索次数**，证明"效率"而非只看准确率。

---

## 一页速记（面试前 5 分钟看）

- **框架**：veRL Agent Loop，colocated hybrid engine 优先；自研 ReAct 包成 `BaseTool`。
- **算法**：GRPO 起步 → 上 **Dr.GRPO（关 std 归一）+ DAPO（dynamic sampling / clip-higher / token-level loss / overlong shaping）**。
- **奖励**：**outcome-first**（format 门槛 + EM/F1），shaping 谨慎且标注 hack 风险。
- **多轮命门**：**observation token masking**（Search-R1）。
- **稳定性**：零方差→dynamic sampling；熵坍缩→clip-higher；`enforce_eager`=关 CUDA Graph。
- **credit assignment**：知道 turn-level（GiGPO/MT-GRPO）比 trajectory 广播更稳。

---

# 番外：Search R1


## Search-R1: Contributions and RL Implementation

### Core Contributions

Search-R1's central motivation is that existing approaches either rely on prompting (poor generalization) or supervised fine-tuning on large-scale annotated trajectories (hard to scale), while RL offers a path where the model learns to call a search engine autonomously using only outcome rewards.

The contributions span three levels:

**1. Framework: Modeling the Search Engine as Part of the RL Environment**

The standard RL objective for LLMs treats the entire output sequence $y$ as generated solely by the model:

$$\max_{\pi_\theta} \mathbb{E}_{x \sim D, y \sim \pi_\theta(\cdot|x)}\left[r_\phi(x,y)\right] - \beta D_{\text{KL}}\left[\pi_\theta(y|x) \| \pi_{\text{ref}}(y|x)\right]$$

Search-R1 extends this to incorporate an external search engine $\mathcal{R}$:

$$\max_{\pi_\theta} \mathbb{E}_{x \sim D,\, y \sim \pi_\theta(\cdot|x;\mathcal{R})}\left[r_\phi(x,y)\right] - \beta D_{\text{KL}}\left[\pi_\theta(y|x;\mathcal{R}) \| \pi_{\text{ref}}(y|x;\mathcal{R})\right]$$

where $\pi_\theta(\cdot|x;\mathcal{R}) = \pi_\theta(\cdot|x) \bowtie \mathcal{R}$, with $\bowtie$ denoting the interleaving of LLM generation and retrieval. The full rollout trajectory thus contains both model-generated tokens and retrieved document content.

> "In this revised objective, the trajectory $y \sim \pi_\theta(\cdot|x;\mathcal{R})$ includes interleaved reasoning steps and retrieved content, reflecting a multi-turn interaction between the LLM and the search engine." [RL Formulation](https://www.alphaxiv.org/abs/2503.09516?page=15)

**2. Training Stability: Retrieved Token Loss Masking**

This is a key engineering contribution. A rollout sequence contains two types of tokens: those generated by the LLM, and those returned by the search engine. Applying policy gradient loss to retrieved tokens causes unintended learning dynamics, since the model has no control over that content.

The solution is a masking function $I(y_t)$: set to $1$ for LLM-generated tokens and $0$ for retrieved tokens, so loss is only computed where $I(y_t)=1$. The paper demonstrates this clearly:

> "Applying retrieved token masking results in greater LLM improvements, mitigating unintended optimization effects and ensuring more stable training." [Token Masking](https://www.alphaxiv.org/abs/2503.09516?page=9)

Concretely, with masking the 7B model's average EM rises from 0.343 to 0.431. [Token Masking Ablation](https://www.alphaxiv.org/abs/2503.09516?page=18)

**3. Multi-Turn Interleaved Reasoning and Search Protocol**

The model uses special tokens to structure its output:
- `<think>...</think>` — reasoning process
- `<search>...</search>` — search query issued by the model
- `<information>...</information>` — retrieved results injected by the system
- `<answer>...</answer>` — final answer

> "Upon detecting these tokens in the generated sequence, the system extracts the search query, queries the search engine, and retrieves relevant results. The retrieved information is then enclosed within special retrieval tokens, `<information>` and `</information>`, and appended to the ongoing rollout sequence, serving as additional context for the next generation step." [Generation Protocol](https://www.alphaxiv.org/abs/2503.09516?page=5)

This loop continues until `</answer>` is generated or the maximum action budget $B=4$ is reached.

---

### How the RL Is Done

Search-R1 supports both **PPO** and **GRPO**. Here is a detailed breakdown of each.

#### PPO Implementation

$$J_{\text{PPO}}(\theta) = \mathbb{E}\left[\frac{1}{\sum_t I(y_t)} \sum_{t: I(y_t)=1} \min\left(\frac{\pi_\theta(y_t|x,y_{<t};\mathcal{R})}{\pi_{\text{old}}(y_t|x,y_{<t};\mathcal{R})} A_t,\ \text{clip}(\cdot, 1\pm\epsilon)\, A_t\right)\right]$$

The advantage $A_t$ is computed using Generalized Advantage Estimation (GAE, $\lambda = \gamma = 1$), requiring a separate Value LLM. The policy LLM uses a learning rate of 1e-6, while the value LLM uses 1e-5. Training runs for 500 steps on 8×H100 GPUs with a total batch size of 512. [PPO Setup](https://www.alphaxiv.org/abs/2503.09516?page=16)

#### GRPO Implementation

GRPO eliminates the value model entirely. For each question, $G=5$ responses are sampled, and the advantage is computed from within-group relative rewards:

$$J_{\text{GRPO}}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^G \frac{1}{\sum_t I(y_{i,t})} \sum_{t: I(y_{i,t})=1} \min\left(\frac{\pi_\theta}{\pi_{\text{old}}} \hat{A}_{i,t},\ \text{clip}(\cdot, 1\pm\epsilon)\hat{A}_{i,t}\right) - \beta D_{\text{KL}}[\pi_\theta \| \pi_{\text{ref}}]\right]$$

Notably, the KL divergence is added directly to the loss function rather than used as a reward penalty term, and the retrieved token mask is also applied when computing the KL term.

> "Instead of incorporating KL divergence as a penalty within the reward function, GRPO regularizes by directly adding the KL divergence between the trained policy and the reference policy to the loss function. The retrieved token masking is also applied when calculating the KL divergence loss $D_{\text{KL}}$." [GRPO KL](https://www.alphaxiv.org/abs/2503.09516?page=5)

#### Reward Function

The reward design is intentionally minimal — outcome reward only, no process reward, no neural reward model:

$$r_\phi(x, y) = \text{EM}(a_{\text{pred}}, a_{\text{gold}})$$

> "We adopt a rule-based reward system that consists solely of final outcome rewards, which assess the correctness of the model's response." [Reward Design](https://www.alphaxiv.org/abs/2503.09516?page=6)

#### PPO vs. GRPO Trade-offs

The paper provides a clear empirical comparison:

| Dimension | PPO | GRPO |
|---|---|---|
| Convergence speed | Slower (value model warm-up needed) | Faster |
| Training stability | Stable throughout 500 steps | Reward collapse after extended training |
| Final performance (7B-base avg) | **0.431** | 0.350 |

> "GRPO converges faster than PPO across all cases... PPO demonstrates greater training stability. As shown in Figure 2(a), GRPO leads to reward collapse after training for many steps, whereas PPO remains stable... Despite differences in convergence speed and stability, both methods achieve similar final train reward and performance." [PPO vs GRPO](https://www.alphaxiv.org/abs/2503.09516?page=8)

PPO is used as the default in the paper due to its superior stability.

---

### Emergent Behaviors During Training

Response length follows a **decrease → increase → stabilize** pattern: in the first ~100 steps the model drops filler words; after step 100, as it learns to call the search engine more frequently, response length grows due to retrieved passages, and training reward rises in tandem.

> "Early Stage (First 100 Steps): The response length sharply decreases, while the training reward exhibits a slight increase. During this phase, the base model learns to eliminate excessive filler words and begins adapting to the task requirements. Later Stage (After 100 Steps): Both response length and training reward increase significantly. At this point, the LLM learns to call the search engine frequently, resulting in longer responses due to retrieved passages." [Response Length](https://www.alphaxiv.org/abs/2503.09516?page=9)

The case studies also reveal that the model spontaneously develops **self-verification** behavior — continuing to issue additional search queries even after finding a plausible answer, mirroring what DeepSeek-R1 exhibits in pure reasoning settings.

---

# References

- [veRL](https://github.com/volcengine/verl) — RL training framework
- [verl-tool](https://github.com/TIGER-AI-Lab/verl-tool) — Tool-augmented RL on veRL
- [Search-R1](https://github.com/PeterGriffinJin/Search-R1) — Original search-augmented RL
- [DAPO](https://arxiv.org/abs/2503.14476) — Dynamic sampling for RL
- [HotpotQA](https://hotpotqa.github.io/) — Multi-hop QA benchmark

