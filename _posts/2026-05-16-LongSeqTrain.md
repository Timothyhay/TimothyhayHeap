---
layout: modern-article
title: Experience of Long Sequence SFT 
date: 2026-04-24
tags: LLM
comments: true
---

[WIP]

本文介绍使用长序列 Agent Trajectory Data 训练 LLM，来赋予其复杂任务规划、工具调用/Agentic能力的流程、经验与注意点。

这里的 Agent 轨迹指交替出现的序列：**状态 (State) -> 思考 (Thought) -> 动作 (Action) -> 观察/环境反馈 (Observation) -> ...**

由于序列极长且具有强烈的时序因果关系，训练过程与普通的文本对话微调有显著不同。以下是具体的训练方法和核心注意事项：

---

### 一、总体流程


#### 1. 数据重构与格式化 (Data Formatting)
在训练前，必须将复杂的轨迹数据标准化。通常采用类似 **ReAct (Reasoning and Acting)** 的框架：
*   **引入特殊 Token：** 为了让模型明确界限，建议使用如 `<|thought|>`, `<|action|>`, `<|observation|>` 等特殊标识符切分不同的阶段。
*   **统一结构：** 确保每一轮的交互都遵循严格的 Schema（例如 JSON 格式的工具调用），以降低模型的解析难度。

#### 2. 监督微调 (SFT / Behavioral Cloning)
这是最基础的训练方式，即通过行为克隆（Behavioral Cloning）让模型模仿成功的轨迹。
*   **Loss Masking（极其关键）：** 在计算损失（Loss）时，**只能对模型生成的 Thought 和 Action 计算 Loss**，必须将 User 的 Prompt 和环境返回的 Observation 的 Loss 给 Mask（掩码）掉。因为模型不需要学习如何“生成环境反馈”，只需要学习如何应对环境。
*   **过滤与提纯：** 只使用最终成功完成任务的轨迹（Success Trajectories）进行 SFT。

#### 3. 强化学习与偏好对齐 (RL / DPO)
由于长序列存在多条路径可以达到目标，仅靠 SFT 容易产生泛化问题。
*   **轨迹级 DPO (Trajectory-level DPO)：** 对于同一个任务，收集一个成功的长轨迹（Chosen）和一个中途失败或效率低下的轨迹（Rejected），使用 DPO (Direct Preference Optimization) 算法进行对比学习。
*   **步骤级优化 (Step-level RL/MCTS)：** 使用 PPO 或 Step-DPO。在长序列中，单独评估某一个 Action 的好坏（依赖于 Reward Model 或蒙特卡洛树搜索 MCTS 的反馈）。

#### 4. 处理超长上下文的架构支持
*   **注意力机制优化：** 必须启用 FlashAttention-2 或 FlashAttention-3 以降低显存消耗。如果是极长序列（如 100k+ tokens），需要引入序列并行（Sequence Parallelism）如 RingAttention。
*   **位置编码扩展：** 如果轨迹长度超出预训练窗口，需要使用 RoPE Scaling（如 YaRN）或长文本预训练阶段进行窗口扩展。

---

### 二、 训练时必须注意的核心问题（避坑指南）

#### 1. 误差累积与曝光偏差 (Exposure Bias & Compounding Errors)
*   **问题：** 纯 SFT 训练只见过“完美”的成功轨迹。但在实际推理时，模型一旦某一步动作出错，环境会返回它从未见过的负面反馈（Observation），导致后续步骤彻底崩溃。
*   **对策：** **加入“纠错数据”**。在训练集中，刻意保留一些“模型犯错 -> 环境报错 -> 模型反思错误并修正”的次优轨迹。让模型学习如何从 Error 中恢复，这比只学完美轨迹更重要。

#### 2. 信用分配问题 (Credit Assignment)
*   **问题：** 在一个 50 步的长序列中，任务最终成功或失败了。到底哪一步的 Action 起了决定性作用？（稀疏奖励问题）。传统 SFT 会对成功轨迹中的所有步骤给予相同的权重，这往往是不合理的。
*   **对策：** 引入价值模型（Value Model）或者使用 Q-learning/Decision Transformer 等机制，对轨迹中的关键节点（Sub-goals）赋予更高的权重。

#### 3. 捷径学习与因果混淆 (Shortcut Learning & Causal Confusion)
*   **问题：** 模型可能没有真正学会“规划”，而是学会了“背诵”或“抄袭”环境反馈。例如，环境返回了一个冗长的错误日志，模型下一步直接原样输出日志中的一段话，而不是针对性地修改代码或参数。
*   **对策：** 监控训练集的重复率。可以通过在数据层面截断冗长的 Observation（例如只保留头尾或关键信息），迫使模型关注核心因果关系。

#### 4. “Lost in the Middle”（中间注意力丢失）
*   **问题：** 长序列轨迹中，任务的最核心指令通常在最开头（Step 0），而模型在经历几十步的 Action-Observation 循环后，可能在 Step 30 忘记了最初的目标，导致无限循环或幻觉。
*   **对策：**
    *   **System Prompt 强化：** 训练数据中在中间步骤定期重申（或 Summarize）全局目标。
    *   **状态总结 (State Tracker)：** 教会大模型定期输出类似 `<|summary|>` 的内容，将过往长达几万 token 的历史浓缩成当前的状态摘要。

#### 5. 灾难性遗忘 (Catastrophic Forgetting)
*   **问题：** 如果全是极长的 Agent 格式数据，模型可能会丧失原本的通用对话能力和常识推理能力，甚至只会输出 JSON 或特定格式。
*   **对策：** 在 Agent SFT 或 RL 阶段，必须**混合通用语料（General Domain Data）**。保持 Agent 轨迹数据与通用对话数据的一定比例（例如 7:3 或 8:2），以维持模型底座的灵活性。

#### 6. 显存与 Batch Size 瓶颈
*   **问题：** 长序列会导致显存爆炸（OOM）。例如 32k 的上下文，常规单卡甚至无法塞下一个 Batch Size = 1 的样本进行 Backward。
*   **对策：**
    *   重度依赖**梯度累积（Gradient Accumulation）**以维持等效 Batch Size。
    *   开启**梯度检查点（Gradient Checkpointing）**，用计算时间换取显存空间。
    *   丢弃历史不重要的环境反馈，采用滑动窗口（Sliding Window）阶段性截断不再使用的极早期 Observation。



## 二、参数选择


## 三、环境与实际问题

### 总结
用长序列 Agent 轨迹训练 LLM，本质上不仅是在教模型“说话”，而是在教它“做事”。**高质量的轨迹数据（尤其是包含试错与反思的轨迹）、精准的 Loss Masking 策略、以及对长序列上下文能力的底层优化**，是决定最终 Agent 智商高低的关键三要素。