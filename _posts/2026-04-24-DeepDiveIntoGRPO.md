---
layout: modern-article
title: 公式与代码样式测试 - 就用 GRPO 来测试吧
date: 2026-04-24
tags: LLM
comments: true
---

在大型语言模型（LLM）的后训练阶段，强化学习（RL）已成为提升模型逻辑推理能力的关键。最近，DeepSeek 推出的 **GRPO (Group Relative Policy Optimization)** 算法因其在 DeepSeek-R1 中的出色表现而备受关注。

本文将深入分析 GRPO 的核心逻辑、数学公式，并提供代码实现的伪逻辑。

## 1. 为什么需要 GRPO？

在传统的 PPO（Proximal Policy Optimization）算法中，我们通常需要四个模型：
1. **Policy Model**: 我们要训练的主模型。
2. **Reference Model**: 用于计算 KL 散度约束，防止模型跑偏。
3. **Reward Model**: 为输出评分。
4. **Critic Model**: 用于估计状态价值（Value Function），帮助计算优势（Advantage）。

**Critic 模型的大小通常与 Policy 模型相当**。这意味着在训练 671B 这样的超大模型时，Critic 模型会占用极高的显存。GRPO 的核心创新在于：**通过组内相对评分取消了 Critic 模型**，从而显著降低了计算资源消耗。

## 2. 数学原理

GRPO 的核心思想是：对于每一个提示词（Prompt）$q$，模型生成一组输出 $\{o_1, o_2, ..., o_G\}$。然后，它根据这组输出的奖励值（Rewards）来计算相对优势，而不是依赖一个单独的价值模型。

### 2.1 目标函数

GRPO 的目标函数定义如下，我们通过折行处理使其更易阅读：

$$
\begin{aligned}
\mathcal{J}_{GRPO}(\theta) = \mathbb{E} \left[ q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{old}}(O|q) \right] \sum_{i=1}^G \{ & \min \left[ \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)} \hat{A}_i, \text{clip} \left( \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}, 1-\epsilon, 1+\epsilon \right) \hat{A}_i \right] \\
& - \beta \mathbb{D}_{KL}(\pi_\theta || \pi_{ref}) \}
\end{aligned}
$$

其中：
- $\hat{A}_i$ 是第 $i$ 个输出在当前组内的**相对优势**。
- $\beta$ 是 KL 散度的惩罚系数。

### 2.2 优势计算（Advantage Calculation）

这是 GRPO 最精妙的地方。它不通过神经网络预测 $V(s)$，而是直接对组内奖励进行归一化：

$$
\hat{A}_i = \frac{r_i - \text{mean}(\{r_1, r_2, ..., r_G\})}{\text{std}(\{r_1, r_2, ..., r_G\})}
$$

通过这种方式，模型能够识别出在同一个 Prompt 下，哪些回答是优秀的，哪些是平庸的。

---

## 3. 代码示例：组内优势计算逻辑

以下是使用 PyTorch 风格编写的 GRPO 核心优势计算代码：

```python
import torch

def compute_grpo_advantages(rewards, eps=1e-8):
    """
    计算组内相对优势
    rewards: Tensor of shape (batch_size, group_size)
    """
    # 计算组内的均值和标准差
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True)
    
    # 归一化奖励，得到 Advantage
    # 这是 GRPO 取消 Critic 模型的关键步骤
    advantages = (rewards - mean) / (std + eps)
    
    return advantages

# 示例数据：假设 Batch 为 1，组大小为 8
# 这 8 个回答的 Reward 分数
group_rewards = torch.tensor([[0.5, 0.2, 0.9, 0.1, 0.5, 0.4, 0.8, 0.3]])

advantages = compute_grpo_advantages(group_rewards)
print(f"Group Advantages:\n{advantages}")
```

## 4. GRPO 与 PPO 的对比

| 特性 | PPO | GRPO |
| :--- | :--- | :--- |
| **价值模型 (Critic)** | 必须，与 Policy 规模一致 | **无需** |
| **内存开销** | 高 | **低 (减少约 50%)** |
| **优势估计** | 依赖神经网络学习 GAE | **基于样本组的统计分布** |
| **训练稳定性** | 容易受 Critic 训练质量影响 | 依赖样本组（Group Size）的大小 |

## 5. 总结

GRPO 证明了在 RLHF 阶段，我们并不总是需要复杂的价值网络。通过将奖励建模为一种相对的、竞争性的指标，GRPO 不仅大幅提升了训练效率，还使得模型在数学推理等具有“硬性评价指标”的领域表现更加出色。

在 DeepSeek-R1 的训练中，这种“自我竞争”的机制是其涌现出复杂思维链（CoT）的重要推手。
