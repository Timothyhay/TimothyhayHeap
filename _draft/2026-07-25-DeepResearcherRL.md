---
layout: modern-article
title: SFT Theory Note
tags: LLM
comments: true
---

# LLM Agentic RL w/ vanilla ReAct Deep Researcher Agent

本文记录了在 8 卡 A100 节点上，基于 **veRL (Ray + vLLM + FSDP)** 框架，对大语言模型进行 Multi-turn Agent 强化学习训练的完整方案。

---

## 1. 项目目标与问题定义

### 1.1 核心目标
训练一个具备 Deep Research （也就是多轮检索推理）能力的 Agent。模型需要学会在面对复杂、模糊、需要多步推理的问题（如 HotpotQA）时，自主调用本地 Search 工具进行探索，过滤冗余信息，纠正检索方向，并最终提炼出高准确度、含真实引用的最终报告。

### 1.2 MDP（马尔可夫决策过程）形式化建模
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

## 2. 系统架构与实验流程

系统采用 **Rollout 与 Training 物理解耦** 的分布式架构。通过 Ray 统一调度 8 卡 A100 的计算资源，规避传统单节点 RL 显存不足和推理吞吐低下的问题。

> Idea: "生成轨迹的推理引擎"和"更新参数的训练引擎"分开，Ray 调度，避免推理慢、训练 OOM 互相拖累。

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

这里有两种部署形态可以选择：

1. Colocated / Hybrid Engine（veRL 默认）：同一批 GPU 上，rollout（vLLM）与训练（FSDP）分时复用，rollout 时把训练权重 offload，训练时收回。显存利用率最高，对短文本而言很合适，如果训练状态。	7B 首选，8 卡全用于 hybrid engine。
2. Disaggregated（解耦）：少量卡常驻 vLLM，其余卡常驻 FSDP（如 Gemini 的 2+6）。省去权重搬运，但推理卡在训练时闲置。	仅当模型大 / 推理是瓶颈时考虑，我们的长链思考任务会占用大量 KV Cache，这部分交给vLLM的独占显存。

> Anchor:2 卡 vLLM + 6 卡 FSDP 常驻是解耦式。为什么不用 colocated？单节点 8 卡训 7B，colocated 通常吞吐更高，因为解耦式在训练阶段那 2 张推理卡在空转。
> 权衡点：解耦式省 reshard 开销但浪费卡；colocated 省卡但有权重 offload/reload 开销。而且我们的穿刺实验要上集群，到时候的模型 offload/reload 会成为严重瓶颈。

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

### 3.6 对多卡集群的放缩计划



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

この手册には良い骨格がありますが、いくつかの技術的な正確性の問題と、面接で突っ込まれやすい「現在のSOTAとのズレ」があります。まず最新の関連研究を確認してから、全面的に最適化・修正した版を作ります。

良い情報が得られました。もう少し最新のagentic RL手法とveRLのmulti-turn実装、DAPOの詳細を確認します。

レビュー完了しました。Gemini版は骨格は良いのですが、**現在のSOTA（Search-R1 / DeepSeek-R1 / DAPO / Dr.GRPO）と逆行している設計判断がいくつか**あり、面接官に突かれると崩れる箇所があります。以下、まず「何を・なぜ直したか」を明示し、その上で修正版フルテキストを出します。

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

## 3. 算法：GRPO 及其现代改良

**一句话直觉**：不训练"打分裁判"（Critic），而是让同一道题采样一组答案，**组内互相比较**——比平均好的鼓励，比平均差的抑制。

### 3.1 为什么用 GRPO（而非 PPO）
GRPO 通过取消独立的价值函数（Critic）来改进 PPO。在 8 卡训 7B 时，PPO 的 Critic 是一份与 Actor 同量级的模型，极易 OOM；GRPO 省下这份显存全给 Actor。其优势估计为组内相对：

$$A_i=\frac{r_i-\operatorname{mean}(\mathbf{r})}{\operatorname{std}(\mathbf{r})}$$

这个"组内相对归一化"给训练提供了比单一 reward + 不可靠 critic 更清晰的信号。

### 3.2 现代改良：从 vanilla GRPO → Dr.GRPO / DAPO（面试加分项）

vanilla GRPO 的两个已知偏置，务必知道：

1. **难度偏置**：除以 $\operatorname{std}(\mathbf{r})$ 会放大"极易/极难"题的权重。Dr. GRPO 取消这个缩放，平等对待所有题目。→ veRL 配置 `algorithm.norm_adv_by_std_in_grpo: False`。
2. **长度偏置**：按序列长度平均会让"更长的错误答案"被低估惩罚。GRPO 按序列长度归一化会导致更长的错误回答被惩罚不足。Dr.GRPO 改用全局常数归一化以消除长度偏置。

**DAPO 的四件套**（ByteDance，基于 verl 实现，长 CoT / 多轮场景强烈推荐）：Clip-Higher（非对称裁剪、上界更高）、Dynamic Sampling（重采样至组内有对有错）、Token-Level Policy Gradient Loss、Overlong Reward Shaping（惩罚过长回答）。其中：
- **Clip-Higher** 治**熵坍缩**：初期观察到熵坍缩现象，通过增大重要性采样比的上裁剪范围来缓解。
- **Dynamic Sampling** 就是 §索引③ 里 σ=0 空梯度的正解。
- **Overlong Reward Shaping** 是软惩罚：设一个最大长度，对超过阈值（如 4096）的多余 token 温和降分，这种"软惩罚"避免模型啰嗦又不过于严厉。

### 3.3 关键工程：Observation Token Masking（多轮 RL 的命门）
在计算 log-prob 和 policy loss 时，**必须对 `<information>` 内的检索 token 置零 mask**，只对模型自己生成的 think/search/answer 计 loss。这是 Search-R1 最关键的创新之一：RL 期间检索内容被排除在优化之外，只有模型自己的推理参与更新，迫使模型"对检索结果做推理"而非"照抄"，从而提升稳定性与泛化。不做 mask 会让模型去拟合外部网页内容，导致学偏/坍缩。实验显示做 masking 训练更稳、提升更大。

### 3.4 多轮 credit assignment（进阶，能讲就是高级信号）
轨迹级单标量 reward 广播到所有 turn 是最简做法，但 GRPO 在多轮设定下被广泛报告不稳定。更细的做法是 **turn-level 优化**：整条轨迹含多个"模型生成 + 环境反馈"回合，在环境反馈上做优化会引入不稳定，因此解耦模型生成 $o_t$ 与环境反馈 $f_t$，只在 $o_t$ 上做定向优化。代表工作：RAGEN 的 StarPO-s 用比例化轨迹过滤，GiGPO 结合状态级与轨迹级优势，MT-GRPO 展示 turn-level credit assignment 的收益。

### 3.5 推荐超参（8×A100 / 7B / HotpotQA）

| 项 | Gemini 值 | **建议值** | 理由 |
| :--- | :--- | :--- | :--- |
| Group size $G$ | 5 | **8~16** | 统计更稳；配合 dynamic sampling |
| Advantage std 归一 | 除 std | **关闭**（Dr.GRPO）或保留但知其偏置 | 消除难度偏置 |
| Dynamic sampling | 无 | **开启** | 消灭零方差空梯度组 |
| KL coef | 0.001 | **0（DAPO 路线）或 1e-3（保守路线），二选一并能解释** | 见下 |
| Actor LR | 1e-6 | 1e-6 ~ 5e-7 + linear warmup | Agent 训练要保守 |
| 单轮生成上限 | "2048" | **区分**：单轮 512~1024，整条轨迹总长 8k~16k | 多轮会累积 |
| Loss 聚合 | 未提 | token-level（DAPO） | 长序列更精确 |

> ▶ **KL 二选一话术**：路线 A（DAPO/Dr.GRPO）——"结果奖励可验证 + 参考模型已是好起点，去 KL 让策略充分移动、避免拖后腿"；DAPO 在其方法中移除了 KL 散度。路线 B（保守）——"保留小 KL 防止在稀疏奖励早期策略崩溃/复读，代价是探索受限"。**关键是你选哪条要给理由，别报数字。**

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