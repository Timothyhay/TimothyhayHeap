---
layout: modern-article
title: LLM Agentic RL w/ hand-made vanilla Deep Researcher Agent
tags: LLM
comments: true
---

本文记录了在 8 卡 A100 节点上，基于 **veRL (Ray + vLLM + FSDP)** 框架，对大语言模型进行 Multi-turn Agent 强化学习训练的实践，
以及后续放缩到 910B 集群的迁移方案。

## 1. 我们为何出发

[书接上回](/2026/08/21/DeepResearcher.html)，我们从第一性原理出发，手搓了一个架构和工具都保持简单的 ReAct DeepResearch Agent。

我们的最终目标是训练其中的 Researcher Agent，让他具备在复杂、模糊、需要多步推理的问题中自主、高效使用本地 search 工具的能力。
同时能过滤掉冗余信息、自动纠正检索方向，并能最终获得引用正式的高准确度回答。

接下来我们谈从问题建模、训练环境搭建开始，如何从单节点到多卡集群构建自定义 Agent 的训练。


### 1.1 MDP 形式化建模
在 Agentic RL 中，我们将多轮 ReAct 交互建模为一个步长有限的 MDP：
*   **状态空间 $S$**：当前轮次之前的完整对话历史，包括初始 $Prompt$、历史思考过程 $Thought_t$、历史动作 $Action_t$（搜索） 和环境反馈 $Observation_t$。
*   **动作空间 $A$**：模型在当前步骤生成的 Token 序列，think + tool_call(search)：
    1.  **内部动作（思考）**：`<think>...</think>` 内的推理文本。
    2.  **外部动作（工具调用）**：触发环境反馈的标签，如 `<call:search>query</call:search>`。
*   **状态转移 $P(S_{t+1}|S_t, a_t)$**：当模型生成外部动作时，生成被暂停、检索服务执行(Action)，将返回的 $Observation_t$ `<observation>...</observation>`拼接到 $S_t$ 后，形成下一轮的状态 $S_{t+1}$。
*   **奖励 $R(S, A)$**：环境在模型输出最终答案 `<answer>...</answer>` 或达到最大步数时，对整条轨迹进行综合打分。

> Anchor: 这跟单轮 RLHF 有何不同？
> 状态在转移中被环境注入了非模型生成的 token（observation），因此必须做 loss masking（§3.3）；且奖励是轨迹级稀疏信号，credit assignment 更难（§3.4）。




---

## Q1: 为什么用 Dr.GRPO 思路修复 DAPO？

初次我们尝试了 GRPO，一开始的思路 以及和 PPO 的区别：

**核心区别**：PPO 需要 Critic（Value Network）来估计 Advantage，而 GRPO 用**组内相对比较**。

- PPO：`Advantage = Reward - V(s)`，需要额外训练一个 Critic 模型（参数量 ≈ Actor）
- GRPO：同一 prompt 生成 n 个回复，组内标准化 `Advantage = (R - mean_group) / std_group`
- GRPO 省掉 Critic → 节省 ~50% GPU 内存和训练时间，更适合 14B 大模型

> PPO 的 Critic 是一份与 Actor 同量级的模型，极易 OOM；GRPO 省下这份显存全给 Actor。其优势估计为组内相对：
>
> $$A_i=\frac{r_i-\operatorname{mean}(\mathbf{r})}{\operatorname{std}(\mathbf{r})}$$
> 相比 PPO，组内相对归一化给训练提供比单一 reward + 不可靠 critic 更清晰的信号


**为什么选择 GRPO**：

- 14B 模型已经很大，再训练一个 14B Critic 会超出 GPU 内存
- GRPO 天然支持 outcome-level reward（只有最终答案才有 reward），适合 QA 任务

### 改良实践

vanilla GRPO 的两个已知偏置：

1. **难度偏置**：除以 $\operatorname{std}(\mathbf{r})$ 会放大"极易/极难"题的权重。Dr. GRPO 取消这个缩放，平等对待所有题目。→ veRL 配置 `algorithm.norm_adv_by_std_in_grpo: False`。
2. **长度偏置**：按序列长度平均会让"更长的错误答案"被低估惩罚。GRPO 按序列长度归一化会导致更长的错误回答被惩罚不足。Dr.GRPO 改用全局常数归一化以消除长度偏置。

**DAPO 的四件套**（ByteDance，基于 verl 实现，长 CoT / 多轮场景强烈推荐）：Clip-Higher（非对称裁剪、上界更高）、Dynamic Sampling（重采样至组内有对有错）、Token-Level Policy Gradient Loss、Overlong Reward Shaping（惩罚过长回答）。其中：
- **Clip-Higher** 治**熵坍缩**：初期观察到熵坍缩现象，通过增大重要性采样比的上裁剪范围来缓解。
- **Dynamic Sampling** 就是 §索引③ 里 σ=0 空梯度的正解。
- **Overlong Reward Shaping** 是软惩罚：设一个最大长度，对超过阈值（如 4096）的多余 token 温和降分，这种"软惩罚"避免模型啰嗦又不过于严厉。

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
路线 A（DAPO/Dr.GRPO）——"结果奖励可验证 + 参考模型已是好起点，去 KL 让策略充分移动、避免拖后腿"；
DAPO 在其方法中移除了 KL 散度。
> 路线 B（保守）——"保留小 KL 防止在稀疏奖励早期策略崩溃/复读，代价是探索受限"。

**关键是你选哪条要给理由，别报数字。**

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

## Q4: 遇到了哪些兼容性问题？怎么解决的？

核心矛盾：veRL 0.7.0.dev 要求 vLLM 0.9+，但所有 PyPI 镜像只有 torch 2.6.0 ± vLLM 0.8.5。

**8 个关键补丁**（详见 PATCHES.md）：

1. `get_tcp_uri` → 手动实现 `f"tcp://{host}:{port}"`
2. `CoreEngineProcManager` → No-op wrapper（vLLM 0.8.5 引擎运行在进程内）
3. `AsyncLLM.reset_mm_cache()` → No-op
4. `AsyncLLM.wait_for_requests_to_drain()` → `abort_all_requests()` 替代
5. `VllmConfig.max_model_len` → 改用 `VllmConfig.model_config`
6. `flash_attn.bert_padding` → 纯 PyTorch 回退实现
7. `calculate_debug_metrics` 空 tensor → try/except
8. `attn_implementation=sdpa` → PyTorch 原生 attention

## 遇到的问题

为什么 Coder 模型失败了而 Instruct 成功了？

- Qwen2.5-Coder-14B 完全不理解 `<think>/<search>/<answer>` 标签 → 输出乱码
- 根本原因：Coder 版本是代码模型，XML 标签触发代码生成模式
- 解决：切到 Qwen2.5-14B-Instruct → 立即正常工作

有可能是上面的 patch 把 vllm 框架给我改坏了。不过也可能是-

**根因分析**（从日志直接追踪到）：

- Qwen2.5-Coder 看到 `<search>`, `<think>` 等 XML 标签 → 进入"代码生成模式"
- 输出：`.0000.0.0.0.0` + 中韩希伯来文字符混合 → 完全乱码
- 500 条 SFT 数据用了 3 epochs 仍然不够覆盖 14B 代码先验

**验证方法**：

- 直接加载模型，用简单自然语言提问 → 回答正常（"The capital of France is Paris"）
- 加上 `<think>/<search>` 标签 → 立即输出乱码
- → 排除了 tokenizer/model 损坏的可能，确认是 prompt 格式不匹配

**面试要点**：展示你区分了"模型没能力"vs"模型和 prompt 不匹配"

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
- score = 0.1 不变 → 所有回复格式分相同，无区分度
- → 说明模型没有真正学习，需要检查 reward 设计或模型能力

## Q8: Wikipedia API 限流怎么处理？

发现了 429 Too Many Requests 并修复：

- **不超时**：API 响应 0.4-1.0s
- **不限流窗口**：每 ~3 请求后 HTTP 429，`Retry-After: 13s`
- **修复**：读 Retry-After → 等待指定秒数 → 重试
- **SFT 数据生成**用 3-5s 延迟 → 100% 成功率（vs 0.3s = 8%）

## Q9: 如果要上线这个系统，还需要做什么？

1. **Reward 优化**：增加 search_bonus（使用搜索就加分），format_score 随步数衰减
2. **多工具**：可能会加入 calculator 工具（处理数值问题），或者参考 Kimi Researcher 加入代码工具
3. **更大规模搜索**：搭建本地 Wikipedia 向量库（不用外部 API）
4. **评估**：在 HotpotQA 官方测试集上报告 EM 和 F1

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

## Q16: GRPO vs Dr.GRPO 有什么区别？什么场景该用哪个？

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

## Q19: 什么问题是 DAPO 能治而 Dr.GRPO 治不了的？

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

### 当前数据集（短轨迹 ~700 tok、n=4、reward∈{0.1,1.0}）

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

### 简历写法：版本 B（跑通并消融对比后升级，现在别写）

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

---

## 2. 系统架构与实验流程

系统采用 **Rollout 与 Training 物理解耦** 的分布式架构。通过 Ray 统一调度 8 卡 A100 的计算资源，规避传统单节点 RL 显存不足和推理吞吐低下的问题。

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
1.  **数据就绪**：使用 HotpotQA 的本地 Wikipedia 支撑段落，利用 BM25 算法在本地搭建轻量高并发检索服务（`search_server.py`），避免联网延迟。
2.  **轨迹生成（Decoupled Rollout）**：
    *   Actor（在 vLLM 引擎中运行）生成文本。
    *   当遇到 `</call:search>` 停止符时，vLLM 暂停生成。
    *   通过 veRL 的 `AgentLoop` 机制拦截输出，解析检索词并请求本地检索服务。
    *   将检索结果包裹在 `<observation>...</observation>` 中拼回 Prompt，唤醒 vLLM 继续生成，直到输出 `<answer>` 标签。
3.  **异步打分**：轨迹生成完毕后，异步发送至打分服务（`agent_reward.py`），进行**奖励塑造（Reward Shaping）**计算。
4.  **模型更新**：Learner 收集一个 Batch 的轨迹和奖励值，利用 FSDP 引擎对 Policy（Actor）模型进行梯度更新。

---

## 3. 算法选择与技术细节

本方案没有使用 PPO，而是尝试了 GRPO 后选择了改良的 Dr. GRPO。

### 3.1 Why GRPO?
1.  **极大地节省显存**：PPO 需要维护一个与 Actor 相同规模的 **Critic（评论员）模型** 来预测状态价值（State Value），这在 8 卡节点上微调 7B+ 模型时极易造成 OOM。GRPO 取消了 Critic 模型，将显存和计算资源全部释放给 Actor。
2.  **相对优势估算**：对每一个输入 $Prompt$，让模型并行 Rollout 产生一组成员（采样数 $G = 5$）。通过这组轨迹的奖励均值和标准差，计算组内的相对优势（Advantage）：
    $$A_i = \frac{r_i - \text{mean}(R)}{\text{std}(R)}$$
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

## 4. 核心：奖励设计与塑造（Reward Shaping）



### 4.1 密集奖励公式设计 (Shaped Reward)
最终奖励由多个惩罚项与一个终局奖励累加而成，以解决信号稀疏问题：
$$R_{total} = R_{format} + R_{validity} + R_{diversity} + R_{step} + R_{accuracy}$$

| 奖励维度 | 设计逻辑 (Formulation) | 提点效果 (Hints) |
| :--- | :--- | :--- |
| **格式对齐奖励 ($R_{format}$)** | 成功输出并闭合 `thought` 和 `call:search` 给 **$+0.2$**；格式崩溃、标签不匹配给 **$-0.5$** 并强制截断。 | 让模型在第 1~2 个 Epoch 迅速学会 ReAct 语法，降低后续随机探索成本。 |
| **工具合规奖励 ($R_{validity}$)** | 调用不存在的工具，或检索词为空时给 **$-0.2$**。 | 约束模型不产生“幻觉工具”调用。 |
| **探索去重惩罚 ($R_{diversity}$)** | 若当前步骤的 Search Query 在历史中已出现过（或语义相似度 $> 0.8$），给 **$-0.4$**。 | **对抗“死循环作弊”**。强制模型在检索受阻时改变思路（试用其他关键词）。 |
| **步数效率惩罚 ($R_{step}$)** | 每发生一步交互，惩罚 **$-0.05 \times Steps$**。 | **对抗“赖着不走作弊”**。逼迫模型在“获取更多信息”和“尽快结束”之间权衡。 |
| **终局准确率 ($R_{accuracy}$)** | 解析最后一个 `<answer>`。若与 Ground Truth 匹配（EM/F1）给 **$+1.5$**；若错误给 **$-0.5$**；若未生成最终答案给 **$-1.0$**。 | 强化学习的终极目标信号。 |

> 但当前主流的检索类 Agentic RL 恰恰相反——Search-R1[1] 明确采用"简单的、基于结果（outcome-based）的奖励函数"，并证明这比复杂奖励更稳、更能泛化。Search-R1 optimizes LLM reasoning trajectories with multi-turn search interactions, leveraging retrieved token masking for stable RL training and a simple outcome-based reward function.
> 
> 其核心论点是：复杂的神经奖励模型容易被钻空子（gamed）或需要过度工程；只需定义答案正确性即可扩展到新领域。你手动加的每一个 shaping 项（尤其 diversity、step penalty）都是一个可被 hack 的攻击面。面试正确答案不是"我设计了 5 个奖励"，而是"我优先用 outcome reward，只保留最小格式约束，把复杂偏好交给相对优势去自然涌现"。 我在修订版把 shaping 降级为"可选辅助项 + 明确风险标注"。

### 4.2 如何防御 Reward Hacking（奖励作弊）？
*   **作弊表现 A**：模型学会了疯狂检索，故意拉长交互步数，在检索历史中反复塞入极微小差异的词来刷前期的 $R_{format}$ 和探索奖励。
    *   *防御方案*：引入**二次惩罚（Quadratic Step Penalty）**，且步数惩罚与步数呈二次非线性关系；同时对 Query 进行强语义去重约束（Rouge-L 阈值）。
*   **作弊表现 B**：模型在回答中疯狂堆砌所有可能相关的实体，试图在 $R_{accuracy}$ 匹配中“蒙混过关”。
    *   *防御方案*：强制提取 `<answer>...</answer>` 内部的单一实体，抛弃外部的所有冗余废话，使其无法进行模糊匹配。

---

## 5. 典型面试挑战题与技术对线

### 挑战 1：你们如何解决 RL 训练的“冷启动（Cold Start）”问题？如果模型一上来随机探索，拿不到任何正反馈怎么办？
*   **答**：这是 Agentic RL 最经典的痛点。我们的解决方案是 **SFT Bootstrapping（冷启动微调）**。我们没有直接让基础模型去跑 RL，而是先利用少量的精选高质量 ReAct 多步数据（约 1000~2000 条），对模型进行了一个 epoch 的 SFT。这让模型先具备“只要看到问题，就一定会尝试用 `<call:search>` 交互”的温和先验概率。在这个基础上再进行 GRPO 训练，可以保证首个 Batch 的采样中，至少有 20% 以上的轨迹能顺利拿分，从而让 Policy 梯度的更新方向能够立足。

### 挑战 2：外部环境（如 Search Tool 报错、超时或服务崩溃）是不可导的，强化学习怎么把梯度回传给模型？
*   **答**：强化学习（如 PPO/GRPO）属于 **无模型（Model-Free）强化学习**。它本身就不需要对外部环境进行显式求导。梯度回传的本质是：
    1. 策略模型做出动作（产生 Token）；
    2. 环境给出一个标量 Reward；
    3. 算法根据策略梯度定理，利用优势函数对产生高 Reward 对应的 Token 的 **对数概率（Log Probabilities）** 进行放大，对低 Reward 的进行抑制。
    因此，即使环境完全是一个不透明的黑盒（甚至可以是人类反馈），只要能输出一个标量 Reward，就完全不影响梯度的正常更新。

### 挑战 3：在多步交互中，环境返回的 Observation 是外部产生的（非模型生成），计算梯度时如何处理这部分 Token？
*   **答**：这是一个关键的工程细节。对于多步 Agent 来说，轨迹中会混入大量的 `Observation`（外部网页内容）。在计算 Actor 模型的 `log_prob` 和计算 Policy 梯度（Loss）时，**必须对 Observation 部分的 Token 进行 Mask（置零）**，只对模型自己生成的 `Thought`、`Action` 和 `Final Answer` 的 Token 计算 Loss 并回传梯度。如果不对这部分进行 Mask，模型会尝试去拟合和预测外部环境返回的数据，从而导致严重的学偏或策略崩溃。

---

## 6. 实践避坑指南与解决方案

### 6.1 显存泄漏与 vLLM 内存碎片问题
*   **现象**：由于多步 Agent 的序列长度（Context Length）随着交互轮数不断拉长（最高可能逼近几十k），在进行几万步采样后，Ray 集群中负责 vLLM Rollout 的节点会频繁出现无预警的 OOM 崩溃。
*   **原因**：vLLM 的 KV Cache 分配在多步动态增长时，由于并发请求的不同步，可能产生严重的内存碎片。
*   **方案**：在 veRL 的配置文件中，合理调低 `vllm_gpu_memory_utilization`（例如从 0.9 降到 0.6），并在 `ray_config` 中设置 `max_concurrency` 限制，同时启用 vLLM 的 `enforce_eager=True` 强制执行即时显存回收，虽然会牺牲极少部分的吞吐量，但极大提升了大规模训练的稳定性。

### 6.2 训练发散与“NaN Loss”
*   **现象**：训练到第 2 个 Epoch 时，WandB 看板上的 Loss 突然变成 NaN，Actor 模型的输出全部变成重复的空格或标点符号（模型崩溃）。
*   **原因**：优势函数 $A_i = \frac{r_i - \mu}{\sigma}$ 中，当组内所有成员的奖励完全一样时，分母标准差 $\sigma = 0$。如果没有对其进行极小值保护（$\epsilon$ epsilon），就会导致梯度变成无穷大，破坏参数。
*   **方案**：在代码中计算相对优势时，务必对标准差加上一个微小的保护项：`std = np.std(R) + 1e-8`。此外，将 `kl_ctrl.kl_coef` 从 `0.0005` 适当提高到 `0.001` 或 `0.002`，加强对模型背离参考模型（Reference Model）的惩罚约束。

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
| 状态 $s_t$ | 到当前轮为止的完整上下文：初始 prompt + 历史 `<think>` + `<search>` + `<observation>` |
| 动作 $a_t$ | 模型本轮生成的 token 序列：内部动作 `<think>…</think>`（推理）+ 外部动作 `<search>query</search>`（触发环境） |
| 转移 $P(s_{t+1}\mid s_t,a_t)$ | 生成外部动作时暂停，检索服务执行并把结果包进 `<information>…</information>` 拼回，形成 $s_{t+1}$ |
| 奖励 $r$ | 通常在输出 `<answer>…</answer>` 或达到 max-turns 时对整条轨迹给一个标量（见 §4） |

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

$$R_{total}=R_{format}+R_{accuracy}$$

| 维度 | 设计 | 说明 |
| :--- | :--- | :--- |
| **格式 $R_{format}$** | ReAct 标签闭合且顺序合法：**0**（合法，不额外给分）；非法：小负分或直接判负 | 只做**门槛**，不做诱饵。避免模型刷格式分 |
| **准确率 $R_{accuracy}$** | 抽取最后一个 `<answer>` 内实体，与 GT 做 **EM / F1**：正确 +1，错误 0（或 -0.x） | 唯一主信号 |

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

需要的话，我可以进一步：**(a)** 把这份 v2 导出成 Markdown/PDF 文件；**(b)** 写一份对应的 veRL 配置样例（GRPO+DAPO 开关、masking、dynamic sampling 的 yaml）；**(c)** 针对你自研 ReAct 系统，给出接入 `BaseTool` / Agent Loop 的代码骨架。要哪个？