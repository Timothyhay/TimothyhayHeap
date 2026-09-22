---
layout: modern-article
title: Past and Present of Reinforcement Learning
date: 2026-04-24
tags: LLM
comments: true
---
The Evolution of Policy Optimization Algorithms from the Perspective of LLM RL Development

## 发展脉络总览 / Overview of the Evolution

```
2017  PPO          ── 经典 RL，InstructGPT/ChatGPT 的 RLHF 基石
                      Classic RL, the RLHF cornerstone of InstructGPT/ChatGPT
2023  DPO          ── 绕过 RL，把偏好对齐变成监督学习
                      Bypasses RL, turns preference alignment into supervised learning
2024  ORPO         ── 更激进：连 SFT 和参考模型都合并掉
                      More radical: merges away even SFT and the reference model
2024  GRPO         ── DeepSeekMath 提出，去掉 Critic 的 RL 回归
                      Proposed in DeepSeekMath, RL returns without a Critic
2025  Dr.GRPO      ── 修正 GRPO 的梯度偏差（长度/难度偏置）
                      Fixes GRPO's gradient biases (length/difficulty bias)
2025  DAPO         ── 工程化大成：长 CoT 大规模 RL 的稳定训练配方
                      Engineering culmination: a stable recipe for large-scale long-CoT RL
2025+ GSPO/CISPO…  ── 面向 MoE、off-policy 复用等场景的后续演化
                      Later evolution targeting MoE, off-policy reuse, etc.
```

两条主线：

1. **计算/工程复杂度的钟摆**：PPO（4 个模型）→ DPO（2 个）→ ORPO（1 个）→ GRPO（2 个但无 Critic）。
2. **奖励信号来源的变迁**：人类偏好（RLHF）→ 偏好数据集（DPO 系）→ **可验证奖励**（数学/代码答案对错，R1 时代的 RLVR）。
3. **The pendulum of computational/engineering complexity**: PPO (4 models) → DPO (2) → ORPO (1) → GRPO (2, but no Critic).
4. **The shift in reward signal sources**: human preferences (RLHF) → preference datasets (the DPO family) → **verifiable rewards** (correctness of math/code answers, i.e., RLVR in the R1 era).

---

## 1. PPO（2017 / RLHF 化 2022）：一切的起点

**背景**：PPO（Proximal Policy Optimization, Schulman et al.）原本是通用 RL 算法，被 OpenAI 在 InstructGPT 中用于 RLHF，成为 ChatGPT 的核心训练技术。

PPO 产生的背景是 PG（所有直接对策略参数求导优化期望回报的算法统称 PG）步长极难选，策略易崩溃、TRPO 二阶优化太重、难以工程化： 
> 1. 标准策略梯度（Policy Gradient / PG）的痛点：
> - 样本利用率极低：严格的 On-policy，采样一次数据更新一次梯度后就必须丢弃。（更新后策略变了，数据就 off-policy 了）
> - 步长极其脆弱（Policy Collapse）：在参数空间走了一大步，可能导致策略在概率分布空间发生剧烈突变，一旦策略变差，采出的数据更差，导致模型迅速崩溃且无法恢复。
> 2. TRPO 的痛点：
> - 为了限制更新幅度，TRPO 施加了硬性的平均 KL 散度约束 (总之 E[D_KL]≤δ)
> - 计算代价极高：求解这个约束优化问题需要计算二阶海森矩阵（Fisher Information Matrix）并使用共轭梯度法（CG），无法与现代深度学习的一阶优化器（如 Adam）无缝结合，且极难拓展到超大模型。

PPO 的诞生动机：
保留 TRPO 的信赖域（Trust Region）思想，但完全丢弃二阶计算，仅用一阶梯度（Clip 机制）实现同等甚至更好的稳定性。

**核心思想**：策略梯度方法容易因单步更新过大而崩溃。PPO 用**裁剪的重要性采样比率**限制每次更新幅度：

PPO was originally a general-purpose RL algorithm. OpenAI applied it to RLHF in InstructGPT, and it became the core training technique behind ChatGPT.

**Core idea**: Policy gradient methods can easily collapse due to overly large single-step updates. PPO uses a **clipped importance-sampling ratio** to bound each update:

$$
\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t\left[\min\Big(r_t(\theta)\,\hat{A}_t,\ \text{clip}\big(r_t(\theta),\, 1-\epsilon,\, 1+\epsilon\big)\,\hat{A}_t\Big)\right],
\quad r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}
$$

其中优势 $\hat{A}_t$ 通常用 **GAE**（广义优势估计）计算，需要一个**价值网络（Critic）**估计 $V(s_t)$。

Here the advantage $\hat{A}_t$ is typically computed with **GAE** (Generalized Advantage Estimation), which requires a **value network (Critic)** to estimate $V(s_t)$.

**The full PPO stack in RLHF** requires maintaining **4 models** simultaneously:


| 模型 / Model           | 作用 / Role                                                          | 是否更新 / Updated? |
| ---------------------- | -------------------------------------------------------------------- | ------------------- |
| Actor（策略 / Policy） | 生成回复 / Generates responses                                       | ✅                  |
| Critic（价值 / Value） | 估计每个 token 前缀的价值 / Estimates the value of each token prefix | ✅                  |
| Reward Model           | 给完整回复打分 / Scores complete responses                           | ❌ 冻结 / Frozen    |
| Reference Model        | 计算 KL 惩罚，防止漂移 / Computes the KL penalty to prevent drift    | ❌ 冻结 / Frozen    |

The reward actually being optimized is:

$$
r_{\text{total}} = r_{\text{RM}}(x, y) - \beta\, \mathrm{KL}\big(\pi_\theta \,\|\, \pi_{\text{ref}}\big)
$$

**痛点**（正是后续所有算法的出发点）：

- **显存与工程成本极高**：4 个模型 + 采样-训练交替流水线；
- **Critic 难训**：LLM 的稀疏奖励（只在句末打分）让逐 token 价值估计非常困难，长 CoT 下价值误差被放大；
- **超参敏感**、reward hacking 等经典 RLHF 问题。

**Pain points** (which motivated every subsequent algorithm):

- **Extremely high memory and engineering cost**: 4 models plus an alternating sampling–training pipeline;
- **The Critic is hard to train**: the sparse reward of LLMs (scored only at the end of the sequence) makes per-token value estimation very difficult, and value errors get amplified under long CoT;
- **Hyperparameter sensitivity**, reward hacking, and other classic RLHF problems.

---

## 2. DPO（2023）：把 RL 从 RLHF 中"消掉"

**背景**：Rafailov et al. 的 *Direct Preference Optimization* 提出：既然 RLHF 的最优解有闭式形式，为什么还要跑 RL？

asks: since the RLHF optimum has a closed-form solution, why run RL at all?

换句话说，DPO 的 idea 是 在带有 KL 正则项的强化学习优化目标下，语言模型本身就可以直接作为隐式奖励模型（Implicit Reward Model）。

**核心推导**：KL 约束下的 RLHF 目标，其最优策略为

**Core derivation**: Under the KL-constrained RLHF objective, the optimal policy is

$$
\pi^*(y \mid x) \propto \pi_{\text{ref}}(y \mid x)\exp\left(\frac{1}{\beta} r(x, y)\right)
$$

可以反解出 reward $r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \text{const}$，代入 Bradley–Terry 偏好模型，（配分函数 $Z(x)$ 相消）奖励函数被**隐式地重参数化为策略本身**，得到纯监督损失：

Inverting this gives $r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \text{const}$. Substituting into the Bradley–Terry preference model, the reward function is **implicitly reparameterized as the policy itself**, yielding a purely supervised loss:

$$
\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)\right]
$$



DPO 能够跳过显式奖励模型，核心在于它将 **KL 约束下的 RLHF 最优解** 与 **Bradley-Terry 偏好模型** 进行了巧妙的数学结合：首先，带有 KL 正则的传统 RLHF 目标存在一个闭式解（Closed-form solution），将该解反向推导即可得到一个**隐式奖励函数**——它等于策略模型与参考模型的对数概率比，加上一个难以计算的配分函数（归一化因子）$Z(x)$；然而关键的一步是，当把这个隐式奖励代入 Bradley-Terry 偏好概率模型时，胜出项与失败项做差使得**配分函数 $Z(x)$ 被精确抵消**。这就直接建立了人类偏好概率与模型自身生成概率的等价映射，从而将复杂的强化学习与奖励建模，彻底简化为一个无需采样的二分类交叉熵优化任务。

> "The core insight of DPO lies in analytically combining the **closed-form solution of the KL-constrained RL objective** with the **Bradley-Terry preference model**. Specifically, the optimal policy under standard RLHF can be inverted to express an **implicit reward** as the log-ratio between the policy and reference model, plus an intractable prompt-dependent partition function $Z(x)$. Crucially, when plugging this implicit reward back into the Bradley-Terry preference formulation, the partition function $Z(x)$ **perfectly cancels out** in the reward difference $r(x, y_w) - r(x, y_l)$. This allows us to parameterize human preference probabilities directly through the language model's own token probabilities, thereby collapsing the complex, multi-stage RL pipeline into a simple, stable binary cross-entropy loss."

keywords：

* **Closed-form solution / 闭式解**（体现数理底子）
* **Implicit reward / 隐式奖励**（体现对 DPO 核心概念的理解）
* **Partition function cancels out / 配分函数对消**（最核心的技术巧思）
* **Binary cross-entropy / 二分类交叉熵**（点出工程落地的最终形态）





**意义与局限**：

**Significance and limitations**:


| ✅ 优点 / Pros                                                                                                                                                 | ❌ 局限 / Limitations                                                                                                                                                                                                |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 无需 RM、Critic、采样循环，训练如同 SFT 般简单稳定 / No RM, Critic, or sampling loop needed; training is as simple and stable as SFT                           | **离线（off-policy）**：只学静态偏好对，无法从自己的新样本中学习 / **Offline (off-policy)**: learns only from static preference pairs, cannot learn from its own fresh samples                                       |
| 成为 2023–2024 开源社区对齐的事实标准（Zephyr、Tulu 等）/ Became the de facto alignment standard in the 2023–2024 open-source community (Zephyr, Tulu, etc.) | 已知会**同时降低 $y_w$ 和 $y_l$ 的概率**（只要差距拉大即可），可能损害生成质量 / Known to **decrease the probabilities of both $y_w$ and $y_l$** (as long as the gap widens), potentially harming generation quality |
| 理论优雅：偏好数据 → 最优策略的直接映射 / Theoretically elegant: a direct mapping from preference data to the optimal policy                                  | 对分布外提示泛化不如 on-policy RL；上限通常低于精调良好的 PPO / Generalizes worse to out-of-distribution prompts than on-policy RL; its ceiling is usually below well-tuned PPO                                      |

DPO 引发了庞大的 "xPO" 家族（IPO、KTO、SimPO……见补充部分），但也让社区最终意识到：**离线偏好学习替代不了在线 RL**——这为 GRPO 时代埋下伏笔。

DPO spawned a vast "xPO" family (IPO, KTO, SimPO… see the supplementary section), but it also led the community to a final realization: **offline preference learning cannot replace online RL** — foreshadowing the GRPO era.

---

## 3. ORPO（2024）：连参考模型和 SFT 阶段都不要了

**背景**：Hong et al. 的 *Odds Ratio Preference Optimization* 追求极致简化——把 **SFT + 对齐合并为单阶段**，且**无参考模型**。

**Background**: *Odds Ratio Preference Optimization* by Hong et al. pursues ultimate simplification — **merging SFT + alignment into a single stage**, with **no reference model**.

**核心思想**：用**几率（odds）**而非概率来构造对比。定义

**Core idea**: Build the contrast using **odds** rather than probabilities. Define

$$
\text{odds}_\theta(y \mid x) = \frac{P_\theta(y \mid x)}{1 - P_\theta(y \mid x)}
$$

总损失 = SFT 损失 + 几率比惩罚项：

Total loss = SFT loss + an odds-ratio penalty term:

$$
\mathcal{L}_{\text{ORPO}} = \underbrace{\mathcal{L}_{\text{SFT}}(y_w)}_{\text{学好回答 / learn good answers}} + \lambda \cdot \underbrace{\left(-\log \sigma\left(\log \frac{\text{odds}_\theta(y_w \mid x)}{\text{odds}_\theta(y_l \mid x)}\right)\right)}_{\text{压制坏回答 / suppress bad answers}}
$$

用 odds ratio 而非概率比的原因：在 SFT 同时进行时，概率比的梯度会过度打压 $y_l$；odds ratio 的惩罚更温和，二者可以共存于一个损失中。

**定位**：ORPO 代表了"去 RL 化"路线的极简终点（1 个模型、1 个阶段），在中小规模对齐上性价比很高；但和 DPO 一样是离线方法，天花板受限于偏好数据质量。

The reason for using the odds ratio instead of the probability ratio: when SFT runs simultaneously, the gradient of a probability ratio would over-suppress $y_l$; the odds-ratio penalty is gentler, allowing the two objectives to coexist within one loss.

**Positioning**: ORPO represents the minimalist endpoint of the "de-RL-ification" path (1 model, 1 stage), offering great cost-effectiveness for small-to-medium-scale alignment; but like DPO, it is an offline method whose ceiling is bounded by preference data quality.

---

## 4. GRPO（2024）：RL 的轻量化回归，R1 时代的引擎

The Lightweight Return of RL, the Engine of the R1 Era.

**背景**：DeepSeekMath 提出 *Group Relative Policy Optimization*，随后因 **DeepSeek-R1** 而闻名。此时奖励来源已经变了：不再是人类偏好 RM，而是**可验证奖励（RLVR）**——数学答案对/错、代码过/不过测试。

**核心思想**：PPO 最大的负担是 Critic。GRPO 的替代方案：对同一提示 $q$ 采样一组 $G$ 个回复，用**组内相对表现**作为优势基线，彻底删掉价值网络：

**Background**: *Group Relative Policy Optimization* was proposed in DeepSeekMath and later became famous through **DeepSeek-R1**. By this point, the reward source had changed: no longer a human-preference RM, but **verifiable rewards (RLVR)** — math answers right/wrong, code passing/failing tests.

**Core idea**: PPO's biggest burden is the Critic. GRPO's alternative: sample a group of $G$ responses for the same prompt $q$, and use **within-group relative performance** as the advantage baseline, completely removing the value network:

$$
\hat{A}_i = \frac{r_i - \text{mean}(r_1, \dots, r_G)}{\text{std}(r_1, \dots, r_G)}
$$

目标函数保留 PPO 式裁剪，外加显式 KL 正则：

The objective retains PPO-style clipping, plus an explicit KL regularizer:

$$
\mathcal{J}_{\text{GRPO}} = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \Big(\min\big(\rho_{i,t}\hat{A}_i,\ \text{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon)\hat{A}_i\big) - \beta\, \mathbb{D}_{\text{KL}}[\pi_\theta \| \pi_{\text{ref}}]\Big)\right]
$$

**为什么它成了 2025 年的默认算法**：

- 去掉 Critic → 显存减半、无价值估计误差问题；
- 与 RLVR 完美契合：二值奖励下"组内对比"就是天然的基线；
- 实现简单，配合 verl / OpenRLHF 等框架极易复现 R1-style 训练。

**Why it became the default algorithm of 2025**:

- Removing the Critic → memory halved, no value-estimation error problem;
- A perfect fit for RLVR: under binary rewards, "within-group comparison" is a natural baseline;
- Simple to implement — with frameworks like verl / OpenRLHF, R1-style training is easy to reproduce.

但它埋了几个坑由后来者修正。

But it buried a few traps — fixed by its successors.

---

## 5. Dr.GRPO（2025）：修 Bug——"GRPO Done Right"

**背景**：*Understanding R1-Zero-Like Training*（Liu et al., 2025）对 GRPO 的梯度估计做了数学分析，发现**两个系统性偏差**：

mathematically analyzed GRPO's gradient estimator and found **two systematic biases**:

1. **长度偏差**：损失中的 $\frac{1}{|o_i|}$ 归一化导致——**答错时，回复越长每 token 惩罚越小**。模型学会"错了就啰嗦"，这是 R1 类训练中回复长度不断膨胀的重要原因之一（并非全是"更深入思考"）。
2. **难度偏差**：除以组内 $\text{std}(r)$ 会放大接近全对或全错（低方差）问题的权重，使不同难度的问题获得失真的梯度权重。

**Length bias**: the $\frac{1}{|o_i|}$ normalization in the loss means that **for incorrect answers, the longer the response, the smaller the per-token penalty**. The model learns to "ramble when wrong" — a major cause of ever-growing response lengths in R1-style training (not all of it is "deeper thinking").
**Difficulty bias**: dividing by the within-group $\text{std}(r)$ amplifies the weight of questions that are nearly all-correct or all-wrong (low variance), giving questions of different difficulty distorted gradient weights.

**修正**极其简单：**删掉这两个归一化项** remove both normalization terms


$$
\hat{A}_i = r_i - \text{mean}(r_1, \dots, r_G) \quad (\text{不再除以 std / no longer divided by std}), \qquad \text{损失聚合去掉 / loss aggregation drops } \tfrac{1}{|o_i|}
$$

从而恢复无偏的策略梯度估计。结果：**相同性能下 token 效率显著提升**，错误回答不再无意义变长。

thereby restoring an unbiased policy gradient estimator. Result: **significantly better token efficiency at the same performance**, and incorrect answers no longer grow meaninglessly long.

**意义**：Dr.GRPO 提醒社区，GRPO 目标函数的细节（归一化位置）并非无关紧要的实现选择，而是直接塑造模型行为的偏置来源。

**Significance**: Dr.GRPO reminded the community that the details of GRPO's objective (where normalization is applied) are not trivial implementation choices — they are sources of bias that directly shape model behavior.

---

## 6. DAPO（2025）：大规模长 CoT RL 的工程配方

## 6. DAPO (2025): An Engineering Recipe for Large-Scale Long-CoT RL

**背景**：ByteDance Seed 的 *DAPO: An Open-Source LLM Reinforcement Learning System at Scale*。与 Dr.GRPO 的"理论修正"路线不同，DAPO 是**四个工程技巧的组合拳**（Decoupled Clip and Dynamic sAmpling Policy Optimization），目标是解决大规模长 CoT 训练中的**熵坍缩、梯度消失、训练发散**。用 Qwen2.5-32B base 在 AIME 2024 上达到 50 分并完全开源（代码 + 数据，基于 verl）。

**Background**: ByteDance Seed's *DAPO: An Open-Source LLM Reinforcement Learning System at Scale*. Unlike Dr.GRPO's "theoretical correction" route, DAPO is a **combination punch of four engineering tricks** (Decoupled Clip and Dynamic sAmpling Policy Optimization), aimed at solving **entropy collapse, vanishing gradients, and training divergence** in large-scale long-CoT training. It reached a score of 50 on AIME 2024 with a Qwen2.5-32B base and was fully open-sourced (code + data, built on verl).

The four components:


| 技巧 / Technique                                                         | 做法 / Method                                                                                                                                                                                                                                                                                                                                                     | 关键机制细节 / Key Mechanism                                                                                                                                                                                                                                                                                                                                                            | 解决的问题 / Problem Solved                                                                                                                                                                                                                                                                                                                                                                        |
| ------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **① Clip-Higher**（解耦裁剪 / decoupled clipping）                      | 把 PPO 单一$\epsilon$ 拆成非对称区间 $[1-\epsilon_{\text{low}},\,1+\epsilon_{\text{high}}]$，论文取 $\epsilon_{\text{low}}=0.2,\ \epsilon_{\text{high}}=0.28$ / Split PPO's single $\epsilon$ into an asymmetric range $[1-\epsilon_{\text{low}},\,1+\epsilon_{\text{high}}]$; the paper uses $\epsilon_{\text{low}}=0.2,\ \epsilon_{\text{high}}=0.28$           | 上界$\epsilon_{\text{high}}$ 放宽以给低概率 token "上升空间"；下界 $\epsilon_{\text{low}}$ **故意不放宽**，否则会把低概率 token 直接压到 0、缩小采样空间 / Raise $\epsilon_{\text{high}}$ to give low-probability tokens headroom to grow; **deliberately keep $\epsilon_{\text{low}}$ small**, since relaxing it would crush low-probability tokens to 0 and shrink the sampling space | 对称裁剪对低概率 token 的提升上限极苛刻 → 策略熵单调下降、rollout 高度同质化 →**熵坍缩**、探索死亡 / Symmetric clipping imposes a brutally tight growth ceiling on low-probability tokens → monotonically decreasing entropy, near-identical rollouts → **entropy collapse** and dead exploration                                                                                              |
| **② Dynamic Sampling**（动态采样 / dynamic sampling）                   | 过采样后**过滤掉准确率为 0 或 1 的组**（组内 $\hat{A}\equiv 0$，梯度为零），持续重采样直到凑满一个全是"有效样本"的 batch，以约束 $0<                                                                                                                                                                                                                              | \{\text{correct}\}                                                                                                                                                                                                                                                                                                                                                                      | <G$ 写进目标 / Over-sample, then **filter out groups with accuracy 0 or 1** (where $\hat{A}\equiv 0$ and the gradient vanishes), and keep resampling until the batch is full of "effective" samples; encoded as the constraint $0<                                                                                                                                                                 |
| **③ Token 级损失 / Token-level Loss**                                   | 从 GRPO 的 sample-level$\frac{1}{G}\sum_i \frac{1}{\lvert o_i\rvert}\sum_t$ 改为 token-level $\frac{1}{\sum_i \lvert o_i\rvert}\sum_i\sum_t$ / Replace GRPO's sample-level $\frac{1}{G}\sum_i \frac{1}{\lvert o_i\rvert}\sum_t$ with token-level $\frac{1}{\sum_i \lvert o_i\rvert}\sum_i\sum_t$                                                                  | 每个 token 权重相等，长回复对梯度的**总贡献按其长度线性增加**，不再被 $1/\lvert o_i\rvert$ 摊薄 / Every token gets equal weight, so a long response's **total gradient contribution scales linearly with its length** instead of being diluted by $1/\lvert o_i\rvert$                                                                                                                | 双向失衡：(a) 高质量长 CoT 中的推理模式被系统性降权，学不动；(b) 长回复中的**乱码与重复也惩罚不足** → 熵与回复长度出现"不健康"暴涨 / A two-sided failure: (a) reasoning patterns inside high-quality long CoT are systematically down-weighted and under-learned; (b) **gibberish and repetition in long responses are also under-penalized** → unhealthy blow-up of entropy and response length |
| **④ Overlong Reward Shaping**（超长奖励整形 / overlong reward shaping） | 两级方案：先用**Overlong Filtering**（把截断样本的 loss 直接 mask 掉），再用 **Soft Overlong Punishment**（分段软惩罚，见下方公式）叠加到规则奖励上 / A two-tier scheme: first **Overlong Filtering** (mask out the loss of truncated samples), then **Soft Overlong Punishment** (a piecewise soft penalty, formula below) added on top of the rule-based reward | 设$L_{\max}=20480$、缓冲 $L_{\text{cache}}=4096$，即 16384 token 以内不罚、16384–20480 线性递增罚至 $-1$ / With $L_{\max}=20480$ and cache $L_{\text{cache}}=4096$: no penalty below 16384 tokens, then a linearly increasing penalty up to $-1$ over 16384–20480                                                                                                                     | 默认把截断回复判为错误，会**惩罚推理本身正确、只是没写完**的样本 → 奖励噪声污染梯度、训练不稳 / Scoring truncated responses as wrong by default **punishes samples whose reasoning is actually sound but merely unfinished** → reward noise pollutes the gradient and destabilizes training                                                                                                      |

另外 DAPO **直接删掉了 KL 惩罚项**——理由是长 CoT 推理训练本来就期望策略大幅偏离初始分布，KL 约束反而碍事。这一做法此后被广泛沿用。

Additionally, DAPO **removed the KL penalty term entirely** — the rationale being that long-CoT reasoning training is expected to drift the policy far from the initial distribution, so the KL constraint only gets in the way. This practice has since been widely adopted.

**Dr.GRPO vs DAPO 的关系**：二者部分重叠（都处理长度偏置），但哲学不同——Dr.GRPO 是"从梯度估计的无偏性出发做最小修正"；DAPO 是"针对每个训练崩溃症状打一个补丁"。实践中两者的技巧经常被混搭使用。

**The relationship between Dr.GRPO and DAPO**: the two partially overlap (both address length bias) but differ in philosophy — Dr.GRPO makes "minimal corrections from the standpoint of unbiased gradient estimation," while DAPO "patches each training-collapse symptom one by one." In practice, their techniques are often mixed and matched.

---

## 7. 值得补充的算法 / Algorithms Worth Adding

<details>
<summary><b>7.1 REINFORCE / RLOO / REINFORCE++ —— "PPO 是否本来就过度设计" / "Was PPO Over-Engineered All Along?"</b>（点击展开 / click to expand）</summary>

**RLOO**（Ahmadian et al., 2024, *Back to Basics*）：论证在 LLM 场景下（初始化于强 SFT 模型、每回合单次奖励），PPO 的很多组件是不必要的。RLOO 用同一提示的其余 $k-1$ 个样本的平均奖励作 leave-one-out 基线：

**RLOO** (Ahmadian et al., 2024, *Back to Basics*): argues that in the LLM setting (initialized from a strong SFT model, one reward per episode), many of PPO's components are unnecessary. RLOO uses the average reward of the other $k-1$ samples for the same prompt as a leave-one-out baseline:

$$
\hat{A}_i = r_i - \frac{1}{k-1}\sum_{j \neq i} r_j
$$

概念上是 GRPO 的近亲（更早、无 std 归一化、无裁剪），无偏且极简。

Conceptually a close relative of GRPO (earlier, no std normalization, no clipping) — unbiased and minimalist.

**REINFORCE++**（2025）：不做 per-prompt 分组，改用**全局批次均值/标准差**归一化优势，配合 PPO 式裁剪与 token 级 KL，宣称比 GRPO 更稳、更抗 reward hacking。

**REINFORCE++** (2025): instead of per-prompt grouping, it normalizes advantages using the **global batch mean/std**, combined with PPO-style clipping and token-level KL, claiming to be more stable and more resistant to reward hacking than GRPO.

这一支的意义在于：它和 GRPO 一起证明了 **Critic-free 的策略梯度就是 LLM RL 的正确抽象层级**。

The significance of this branch: together with GRPO, it demonstrated that **Critic-free policy gradients are the right level of abstraction for LLM RL**.

</details>

<details>
<summary><b>7.2 DPO 家族的重要变体：IPO / KTO / SimPO / Important Variants in the DPO Family: IPO / KTO / SimPO</b>（点击展开 / click to expand）</summary>

- **IPO**（Azar et al., 2023）：指出 DPO 在偏好确定性高时会过拟合（把 $y_l$ 概率推向 0），改用有界的平方损失回归目标，理论上更稳。
- **KTO**（Ethayarajh et al., 2024）：基于前景理论（Kahneman-Tversky），**不需要成对偏好数据**——只需单条样本的"好/坏"二元标签即可训练，大幅降低数据收集门槛。
- **SimPO**（Meng et al., 2024）：去掉参考模型，用**长度归一化的平均 log 概率**作为隐式奖励，加上目标间隔 $\gamma$：
- **IPO** (Azar et al., 2023): points out that DPO overfits when preferences are highly deterministic (pushing the probability of $y_l$ toward 0), and instead uses a bounded squared-loss regression objective, which is theoretically more stable.
- **KTO** (Ethayarajh et al., 2024): based on prospect theory (Kahneman-Tversky), it **does not require paired preference data** — a binary "good/bad" label per single sample suffices, greatly lowering the data-collection barrier.
- **SimPO** (Meng et al., 2024): removes the reference model and uses the **length-normalized average log-probability** as the implicit reward, plus a target margin $\gamma$:

$$
\mathcal{L}_{\text{SimPO}} = -\log \sigma\left(\frac{\beta}{|y_w|}\log \pi_\theta(y_w \mid x) - \frac{\beta}{|y_l|}\log \pi_\theta(y_l \mid x) - \gamma\right)
$$

有趣的对照：SimPO 的长度归一化是**防止偏好学习偏爱长回复的解药**，而 Dr.GRPO 在 RL 侧删掉长度归一化——同一个"长度偏置"问题在不同目标函数结构下需要相反的处理，说明这些细节必须结合梯度形式具体分析。

An interesting contrast: SimPO's length normalization is **the antidote to preference learning favoring long responses**, whereas Dr.GRPO removes length normalization on the RL side — the same "length bias" problem requires opposite treatments under different objective structures, showing these details must be analyzed concretely with the gradient form in mind.

</details>

### 7.3 GSPO（2025，Qwen 团队）：序列级重要性采样 ⭐ 重点补充

### 7.3 GSPO (2025, Qwen Team): Sequence-Level Importance Sampling ⭐ Key Addition

GRPO 的重要性比率是 **token 级**的，但奖励是**序列级**的——粒度错配导致每个 token 的比率噪声在长序列上累积，在 **MoE 模型**上尤其严重（专家路由波动使 token 级比率剧烈抖动，甚至需要"Routing Replay"等补丁）。GSPO（Group Sequence Policy Optimization）改用**序列级比率**（几何平均）并在序列级裁剪：

GRPO's importance ratio is **token-level**, but the reward is **sequence-level** — this granularity mismatch causes per-token ratio noise to accumulate over long sequences, especially severe on **MoE models** (expert-routing fluctuations make token-level ratios jitter violently, even requiring patches like "Routing Replay"). GSPO (Group Sequence Policy Optimization) switches to a **sequence-level ratio** (geometric mean) and clips at the sequence level:

$$
s_i(\theta) = \left(\frac{\pi_\theta(y_i \mid x)}{\pi_{\theta_{\text{old}}}(y_i \mid x)}\right)^{1/|y_i|}
$$

使优化粒度与奖励粒度对齐，是 Qwen3 系列训练的核心算法，已成为 MoE 大规模 RL 的主流选择。

This aligns the optimization granularity with the reward granularity. It is the core algorithm behind Qwen3-series training and has become the mainstream choice for large-scale MoE RL.

### 7.4 CISPO（2025，MiniMax-M1）：裁剪权重而非裁剪 token ⭐ 重点补充

### 7.4 CISPO (2025, MiniMax-M1): Clip the Weights, Not the Tokens ⭐ Key Addition

关键观察：PPO/GRPO 的裁剪会**永久杀死**某些低概率但关键的"反思/分叉"token（如 "wait"、"however"——恰是长推理能力的种子）的梯度。CISPO 不裁剪 token 更新，而是**裁剪重要性采样权重本身**（类似截断 IS 的 REINFORCE），保证**所有 token 都保留梯度**，同时控制方差。附带好处是对 off-policy 数据复用（一批 rollout 训多步）更友好，能显著降低生成瓶颈成本。

Key observation: PPO/GRPO clipping can **permanently kill** the gradients of certain low-probability but critical "reflection/forking" tokens (e.g., "wait", "however" — precisely the seeds of long-form reasoning). CISPO does not clip token updates; instead it **clips the importance-sampling weight itself** (akin to REINFORCE with truncated IS), guaranteeing **all tokens retain gradients** while controlling variance. A side benefit: it is friendlier to off-policy data reuse (training multiple steps on one batch of rollouts), significantly reducing the cost of the generation bottleneck.

<details>
<summary><b>7.5 其他值得知道的方向 / Other Directions Worth Knowing</b>（点击展开 / click to expand）</summary>

- **VAPO**（ByteDance, 2025）：与"去 Critic"潮流反向——论证**训好的价值网络仍是天花板更高的方案**，通过 value-pretraining、解耦 GAE 等技巧使 value-based 方法在长 CoT 上反超 DAPO。说明 Critic-free 是工程妥协而非理论最优。
- **GDPO**：多奖励目标下先对每个奖励独立归一化再合并，解决多目标对齐（如工具调用 + 格式 + 正确性）时的优势坍缩。
- **熵控制技术**（Clip-Cov / KL-Cov 等）：直接针对 RLVR 训练中的熵坍缩机制做干预，与 DAPO 的 clip-higher 属于同一问题的不同解法。
- **过程奖励（PRM）与 RLVR 之争**：R1 报告了 PRM 的 reward hacking 难题而选择纯结果奖励；但步骤级信号如何安全利用仍是活跃方向。
- **Agentic RL / 多轮 RL**：当前（2025–2026）前沿正从"单轮解题"转向多轮工具调用轨迹的 RL，信用分配和异步 rollout 是新战场，上述算法大多正在被移植改造。
- **VAPO** (ByteDance, 2025): runs counter to the "de-Critic" trend — arguing that **a well-trained value network still offers a higher ceiling**, using tricks like value-pretraining and decoupled GAE to make value-based methods surpass DAPO on long CoT. This shows Critic-free is an engineering compromise, not a theoretical optimum.
- **GDPO**: under multiple reward objectives, normalize each reward independently before merging, solving advantage collapse in multi-objective alignment (e.g., tool use + format + correctness).
- **Entropy-control techniques** (Clip-Cov / KL-Cov, etc.): directly intervene in the entropy-collapse mechanism of RLVR training — different solutions to the same problem DAPO's clip-higher addresses.
- **The PRM vs. RLVR debate**: R1 reported PRMs' reward-hacking difficulties and chose pure outcome rewards; but how to safely exploit step-level signals remains an active direction.
- **Agentic RL / multi-turn RL**: the current (2025–2026) frontier is shifting from "single-turn problem solving" to RL over multi-turn tool-use trajectories; credit assignment and asynchronous rollouts are the new battleground, and most of the above algorithms are being ported and adapted.

</details>

---

## 8. 总览对比 / Overall Comparison


| 算法 / Algorithm | 年份 / Year | 范式 / Paradigm               | Critic | 参考模型/KL / Ref Model/KL | 数据来源 / Data Source          | 优势估计 / Advantage Estimation                                      | 一句话定位 / One-Line Positioning                                               |
| ---------------- | ----------- | ----------------------------- | ------ | -------------------------- | ------------------------------- | -------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| **PPO**          | 2017/22     | 在线 RL / Online RL           | ✅     | ✅                         | RM 打分 / RM scores             | GAE                                                                  | RLHF 黄金标准，贵而难调 / The RLHF gold standard — expensive and hard to tune  |
| **DPO**          | 2023        | 离线偏好 / Offline preference | ❌     | ✅                         | 偏好对 / Preference pairs       | —（隐式奖励 / implicit reward）                                     | 把 RLHF 变成监督学习 / Turns RLHF into supervised learning                      |
| **ORPO**         | 2024        | 离线偏好 / Offline preference | ❌     | ❌                         | 偏好对 / Preference pairs       | —（odds ratio）                                                     | 单模型单阶段的极简对齐 / Minimalist single-model, single-stage alignment        |
| **GRPO**         | 2024        | 在线 RL / Online RL           | ❌     | ✅（KL 项 / KL term）      | 可验证奖励 / Verifiable rewards | 组内均值+std 归一化 / Group mean + std normalization                 | R1 时代默认算法 / The default algorithm of the R1 era                           |
| **Dr.GRPO**      | 2025        | 在线 RL / Online RL           | ❌     | 可选 / Optional            | 可验证奖励 / Verifiable rewards | 组内均值（去归一化）/ Group mean (de-normalized)                     | 无偏梯度，治长度膨胀 / Unbiased gradients, cures length inflation               |
| **DAPO**         | 2025        | 在线 RL / Online RL           | ❌     | ❌（删 KL / KL removed）   | 可验证奖励 / Verifiable rewards | GRPO 式 + 动态采样 / GRPO-style + dynamic sampling                   | 大规模长 CoT 稳定配方 / A stable recipe for large-scale long CoT                |
| GSPO             | 2025        | 在线 RL / Online RL           | ❌     | ❌                         | 可验证奖励 / Verifiable rewards | 组内相对 +**序列级比率** / Group-relative + **sequence-level ratio** | MoE / 大规模稳定性 / MoE / large-scale stability                                |
| CISPO            | 2025        | 在线 RL / Online RL           | ❌     | ❌                         | 可验证奖励 / Verifiable rewards | 裁 IS 权重不裁 token / Clip IS weights, not tokens                   | 保住反思 token、off-policy 复用 / Preserves reflection tokens, off-policy reuse |

## 9. 实践选型建议 / Practical Selection Advice

- **通用对话对齐、有偏好数据、算力有限** → DPO（或 SimPO/KTO），仍是性价比之王；ORPO 适合想省掉 SFT 阶段的场景。
- **推理/代码等有可验证奖励的任务** → GRPO 起步，几乎必然要叠加 Dr.GRPO 的去偏 + DAPO 的 clip-higher / dynamic sampling / token 级损失（现代框架如 verl 中这些是可组合开关）。
- **MoE 模型或超大规模训练不稳** → GSPO。
- **rollout 成本是瓶颈、想一批数据训多步** → CISPO。
- **有精细过程奖励、追求上限且工程能力强** → 别忘了 PPO/VAPO 这条 value-based 路线并没有死。
- **General dialogue alignment, with preference data, limited compute** → DPO (or SimPO/KTO) remains the king of cost-effectiveness; ORPO suits scenarios where you want to skip the SFT stage.
- **Reasoning/code tasks with verifiable rewards** → start with GRPO, and almost inevitably stack Dr.GRPO's debiasing + DAPO's clip-higher / dynamic sampling / token-level loss (in modern frameworks like verl, these are composable switches).
- **MoE models or instability in ultra-large-scale training** → GSPO.
- **Rollout cost is the bottleneck and you want multiple training steps per batch of data** → CISPO.
- **Fine-grained process rewards, chasing the ceiling, with strong engineering capabilities** → don't forget the value-based line of PPO/VAPO is far from dead.

**一个收束性的观察**：这十年的演进本质上是在回答同一个问题——*在"奖励只在序列末尾出现、动作空间是整个词表、单条轨迹极长"的特殊 MDP 里，经典 RL 的哪些组件是必需的？* 答案逐渐清晰：Critic 可以用组内采样替代（GRPO），KL 约束在推理训练中可以扔掉（DAPO），但**梯度估计的归一化细节（Dr.GRPO）和裁剪粒度（GSPO/CISPO）会直接塑造模型行为**，一点都马虎不得。

**A concluding observation**: this decade of evolution is essentially answering one question — *in the special MDP where "rewards appear only at the end of the sequence, the action space is the entire vocabulary, and a single trajectory is extremely long," which components of classic RL are truly necessary?* The answer has gradually become clear: the Critic can be replaced by within-group sampling (GRPO), the KL constraint can be discarded in reasoning training (DAPO), but **the normalization details of gradient estimation (Dr.GRPO) and the clipping granularity (GSPO/CISPO) directly shape model behavior** — and cannot be treated carelessly at all.

如果你想深入某一个算法的推导细节（比如 DPO 的闭式解推导、或 Dr.GRPO 的偏差分析），我可以单独展开。

If you'd like to dive into the derivation details of any particular algorithm (e.g., DPO's closed-form derivation, or Dr.GRPO's bias analysis), I can expand on it separately.

# Quick Questions

LoRA, DPO & ORPO？

> **LoRA 是一种 PEFT 方法，通过冻结 Base Model，仅训练低秩矩阵 \(BA\) 来近似参数更新 \(\Delta W\)，从而显著降低微调的显存和参数成本。**
>
> **DPO 是一种 preference optimization 方法，它利用 chosen/rejected response pair，绕过 Reward Model 和 PPO，直接优化 policy，同时使用 frozen reference model 约束 policy 不要偏离原始分布过多。**
>
> **ORPO 可以看作进一步简化的 preference optimization，它不需要 reference model，并把 SFT loss 和 preference loss 融合到一个目标中。**
>
> **三者不是互斥的：LoRA 解决的是‘怎么高效更新模型’，DPO/ORPO 解决的是‘根据偏好数据把模型往什么方向优化’。**

重要性采样为何在 RL 中需要

From DJ：

1. 训推不一致问题：推理时我们使用vLLM，但训练框架不同，计算方式也不同
2. MoE模型专家随机路由
3. tokenizer 分词结果也不一样



# 公式推导与实现

# PPO（Proximal Policy Optimization）详解

下面按「问题 → 推导 → 设计动机 → 面试手推」的顺序来讲，核心是把每一步"为什么要这么做"说清楚。

---

## 一、PPO 要解决什么问题

PPO 属于 **on-policy 的策略梯度方法**。它的前身是 REINFORCE / A2C 和 TRPO，这两条线各有痛点：

| 方法 | 痛点 |
|---|---|
| 原始策略梯度（REINFORCE/A2C） | 每批数据只能做**一次**梯度更新（更新后策略变了，数据就 off-policy 了），样本效率低；步长很难调，步子大了策略崩掉（performance collapse）且难以恢复 |
| TRPO | 用 KL 约束保证单调改进，理论漂亮，但需要二阶信息（Fisher 矩阵 + 共轭梯度 + 线搜索），实现复杂、计算贵、不易与 dropout / 参数共享等结合 |

PPO 的目标：**用一阶方法，近似达到 TRPO "限制策略更新幅度" 的效果，同时允许一批数据被复用多个 epoch。**

---

## 二、从零手推（面试版）

### 第 1 步：策略梯度定理（log-derivative trick）

目标 $J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}[R(\tau)]$。关键技巧：

$$
\nabla_\theta \mathbb{E}_{x\sim p_\theta}[f(x)]
=\int f(x)\nabla_\theta p_\theta(x)\,dx
=\int f(x)\,p_\theta(x)\,\nabla_\theta\log p_\theta(x)\,dx
=\mathbb{E}_{x\sim p_\theta}\big[f(x)\nabla_\theta\log p_\theta(x)\big]
$$

轨迹概率 $p_\theta(\tau)=\rho(s_0)\prod_t \pi_\theta(a_t|s_t)P(s_{t+1}|s_t,a_t)$，取 log 后环境转移项与 $\theta$ 无关，求导消失，得到：

$$
\nabla_\theta J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}\Big[\sum_t \nabla_\theta\log\pi_\theta(a_t|s_t)\,\Psi_t\Big]
$$

其中 $\Psi_t$ 可以是回报 $G_t$，也可以是优势 $A^{\pi}(s_t,a_t)$。

**为什么可以减 baseline / 用优势？** 因为对任意只依赖状态的 $b(s)$：

$$
\mathbb{E}_{a\sim\pi_\theta(\cdot|s)}\big[\nabla_\theta\log\pi_\theta(a|s)\,b(s)\big]
=b(s)\nabla_\theta\sum_a\pi_\theta(a|s)=b(s)\nabla_\theta 1=0
$$

期望不变，方差降低。取 $b(s)=V^\pi(s)$ 就得到优势 $A=Q-V$。

> 面试要点：这一步说明策略梯度是**无偏但高方差**的，而且它是在 $\pi_\theta$ 自己采的数据上计算的期望——**数据一旦更新过一次就不再是 $\pi_\theta$ 的样本**，这就是 on-policy 的根源。

### 第 2 步：性能差分引理（Performance Difference Lemma）

我们真正想优化的是"新策略比旧策略好多少"：

$$
J(\pi')-J(\pi)=\mathbb{E}_{\tau\sim\pi'}\Big[\sum_{t}\gamma^t A^{\pi}(s_t,a_t)\Big]
$$

**推导**（telescoping）：注意 $A^\pi(s_t,a_t)=\mathbb{E}_{s_{t+1}}[r_t+\gamma V^\pi(s_{t+1})]-V^\pi(s_t)$，于是

$$
\mathbb{E}_{\tau\sim\pi'}\Big[\sum_t\gamma^t A^\pi(s_t,a_t)\Big]
=\mathbb{E}_{\tau\sim\pi'}\Big[\sum_t\gamma^t r_t+\sum_t\big(\gamma^{t+1}V^\pi(s_{t+1})-\gamma^tV^\pi(s_t)\big)\Big]
$$

第二项裂项相消只剩 $-V^\pi(s_0)$，所以

$$
=\mathbb{E}_{\tau\sim\pi'}\Big[\sum_t\gamma^t r_t\Big]-\mathbb{E}_{s_0}[V^\pi(s_0)]=J(\pi')-J(\pi)\quad\blacksquare
$$

写成状态分布形式（$d^{\pi'}$ 为折扣状态访问分布）：

$$
J(\pi')-J(\pi)=\frac{1}{1-\gamma}\,\mathbb{E}_{s\sim d^{\pi'},\,a\sim\pi'}\big[A^{\pi}(s,a)\big]
$$

> 这个式子的含义：**用旧策略的优势函数评价新策略的动作**，只要新策略在自己到达的状态上平均优势为正，就一定改进。问题是 $d^{\pi'}$ 依赖新策略，没法在更新前采样。

### 第 3 步：两个近似 → 替代目标（surrogate objective）

**近似 1：状态分布用旧策略的。** 假设 $\pi'$ 与 $\pi$ 接近，则 $d^{\pi'}\approx d^{\pi}$。

**近似 2：动作上做重要性采样。** 我们只有 $a\sim\pi$ 的样本，但要算 $a\sim\pi'$ 的期望：

$$
\mathbb{E}_{a\sim\pi'(\cdot|s)}[A^\pi(s,a)]
=\sum_a \pi'(a|s)A^\pi(s,a)
=\sum_a \pi(a|s)\frac{\pi'(a|s)}{\pi(a|s)}A^\pi(s,a)
=\mathbb{E}_{a\sim\pi(\cdot|s)}\Big[\frac{\pi'(a|s)}{\pi(a|s)}A^\pi(s,a)\Big]
$$

于是得到 TRPO/PPO 共用的替代目标：

$$
L_{\pi_{\text{old}}}(\theta)=\mathbb{E}_{s\sim d^{\pi_{\text{old}}},\,a\sim\pi_{\text{old}}}\Big[\underbrace{\frac{\pi_\theta(a|s)}{\pi_{\text{old}}(a|s)}}_{r_t(\theta)}\,\hat A_t\Big]
$$

**为什么只对单步动作做重要性比，而不是整条轨迹？** 整条轨迹的比值 $\prod_t \frac{\pi_\theta}{\pi_{old}}$ 会指数级爆炸/消失，方差不可控；单步比值是把 $d^{\pi'}\approx d^{\pi}$ 这个偏差"吃掉"换来的方差可控——这是一个刻意的 bias-variance 取舍。

**这个替代目标有两个关键性质**（面试常被追问）：

1. $L_{\pi_{\text{old}}}(\theta_{\text{old}})=J(\theta_{\text{old}})$（此时比值恒为 1，优势期望为 0）。
2. 一阶梯度匹配：

$$
\nabla_\theta r_t(\theta)\Big|_{\theta_{\text{old}}}
=\frac{\nabla_\theta\pi_\theta(a|s)}{\pi_{\text{old}}(a|s)}\Big|_{\theta_{\text{old}}}
=\nabla_\theta\log\pi_\theta(a|s)\Big|_{\theta_{\text{old}}}
$$

所以 $\nabla_\theta L_{\pi_{\text{old}}}(\theta)\big|_{\theta_{\text{old}}}=\nabla_\theta J(\theta)\big|_{\theta_{\text{old}}}$，**在旧策略处替代目标和真实目标切线相同**，即它是真实目标的局部一阶近似。

### 第 4 步：为什么必须限制步长（TRPO 的下界）

TRPO 证明了：

$$
J(\pi')\;\ge\;L_{\pi}(\pi')-C\cdot\max_s D_{\text{KL}}\big(\pi(\cdot|s)\,\|\,\pi'(\cdot|s)\big),\qquad C=\frac{4\epsilon\gamma}{(1-\gamma)^2}
$$

含义：替代目标只在 $\pi'$ 离 $\pi$ 不远时可信，走远了近似 1 就失效，替代目标上升不代表真实回报上升。所以 TRPO 做**信赖域优化**：

$$
\max_\theta L_{\pi_{\text{old}}}(\theta)\quad\text{s.t.}\quad \bar D_{\text{KL}}(\pi_{\text{old}}\|\pi_\theta)\le\delta
$$

这需要二阶方法求解——太重了。

### 第 5 步：PPO 的解法——把约束"焊进"目标函数

**PPO-Clip（主流）：**

$$
L^{\text{CLIP}}(\theta)=\hat{\mathbb{E}}_t\Big[\min\Big(r_t(\theta)\hat A_t,\;\text{clip}\big(r_t(\theta),1-\epsilon,1+\epsilon\big)\hat A_t\Big)\Big]
$$

**PPO-Penalty（自适应 KL）：**

$$
L^{\text{KL}}(\theta)=\hat{\mathbb{E}}_t\Big[r_t(\theta)\hat A_t-\beta\, D_{\text{KL}}(\pi_{\text{old}}\|\pi_\theta)\Big]
$$

$\beta$ 根据实际 KL 与目标值 $d_{\text{targ}}$ 的比较动态调大/调小（实测不如 clip 好用）。

---

## 三、Clip 目标为什么这样设计（逐项拆解）

分四种情形看 $L^{\text{CLIP}}$ 对 $r_t$ 的行为：

| 情形 | $\hat A_t$ | $r_t$ | $\min(\cdot)$ 取哪项 | 梯度 | 直觉 |
|---|---|---|---|---|---|
| ① | $>0$ | $\le 1+\epsilon$ | 未裁剪项 | 正常 | 好动作，提高概率 |
| ② | $>0$ | $>1+\epsilon$ | 裁剪项（常数） | **0** | 好动作概率已经提高够多了，别再贪 |
| ③ | $<0$ | $\ge 1-\epsilon$ | 未裁剪项 | 正常 | 坏动作，降低概率 |
| ④ | $<0$ | $<1-\epsilon$ | 裁剪项（常数） | **0** | 坏动作概率已经压得够低了，停 |

三个设计要点：

1. **`clip` 的作用**：让比值超出 $[1-\epsilon,1+\epsilon]$ 后目标函数变平，梯度为零，策略在这些样本上就"不再有动力"继续偏离——用一阶方法软性地实现了信赖域。注意它**不是硬约束**，比值确实可能超出区间（因为多个 epoch 的 minibatch 更新），只是超出后不再被推得更远。

2. **`min` 的作用（悲观下界）**：单纯 clip 是不够的。如果只用裁剪项，那么在情形"$\hat A_t<0$ 且 $r_t>1+\epsilon$"（坏动作概率反而变大了）时，裁剪项是常数、梯度为 0，**错误无法被纠正**。加了 `min` 后这种情况会选未裁剪项 $r_t\hat A_t$（更小），梯度恢复，把概率往回拉。所以 `min` 保证了 $L^{\text{CLIP}}\le L_{\pi_{\text{old}}}$，是替代目标的**下界**——"把改进的一面裁掉，把变坏的一面保留"，这正是 TRPO 下界思想的廉价版。

3. **允许多 epoch 复用数据**：因为比值和 clip 天然处理了"数据来自 $\pi_{\text{old}}$，参数已是 $\pi_\theta$"这一 off-policy 偏移，同一批数据可以做 K 个 epoch 的 minibatch SGD，样本效率大幅高于 A2C。

---

## 四、完整算法与配套部件

**优势估计：GAE（Generalized Advantage Estimation）**

$$
\delta_t=r_t+\gamma V_\phi(s_{t+1})-V_\phi(s_t),\qquad
\hat A_t^{\text{GAE}(\gamma,\lambda)}=\sum_{l=0}^{\infty}(\gamma\lambda)^l\delta_{t+l}
$$

$\lambda=0$ 退化为一步 TD（低方差高偏差），$\lambda=1$ 退化为 MC 回报减 baseline（高方差无偏差）。$\lambda\approx0.95$ 是在两者间折中。实现上从后向前递推 $\hat A_t=\delta_t+\gamma\lambda\hat A_{t+1}$。

**总损失**

$$
L(\theta,\phi)=-L^{\text{CLIP}}(\theta)+c_1\,\underbrace{\big(V_\phi(s_t)-\hat R_t\big)^2}_{\text{value loss}}-c_2\,\underbrace{\mathcal H[\pi_\theta](s_t)}_{\text{entropy bonus}}
$$

价值损失训练 critic（$\hat R_t=\hat A_t+V_{\text{old}}(s_t)$），熵项防止过早收敛到确定性策略。

<details>
<summary><b>算法流程与实现细节（点击展开）</b></summary>

```text
for iteration = 1, 2, ...:
    用 π_old 与环境交互 N 步（或 N 条轨迹），记录 (s, a, logπ_old(a|s), r, done, V_old(s))
    用 GAE 计算 Â_t，并得到 value target R̂_t = Â_t + V_old(s_t)
    （常做优势标准化：Â ← (Â - mean)/std）
    for epoch = 1..K:
        将数据打乱切成 minibatch
        for each minibatch:
            r_t = exp(logπ_θ(a|s) - logπ_old(a|s))
            L_clip = mean(min(r_t·Â, clip(r_t,1-ε,1+ε)·Â))
            L_v    = mean((V_φ(s) - R̂)²)
            L      = -L_clip + c1·L_v - c2·entropy
            梯度下降（常配合梯度裁剪 max_norm=0.5）
    π_old ← π_θ
```

常见的"隐性"实现细节（论文没写但对性能影响很大）：
- 优势标准化（per-batch）
- 价值函数 clipping（争议性，有时有害）
- 学习率线性衰减、Adam $\epsilon=10^{-5}$
- 观测/回报归一化
- 正交初始化，策略头输出层小增益
- 用 `exp(logp_new - logp_old)` 而非直接比值，数值稳定
- 监控 approx KL 和 clip fraction 作为早停信号（比值超出区间的样本比例正常在 0.1–0.3）

典型超参：$\epsilon=0.2$，$K=3\sim10$，$\gamma=0.99$，$\lambda=0.95$，$c_1=0.5$，$c_2=0.01$。
</details>

<details>
<summary><b>PPO 在 LLM / RLHF 中的变体（点击展开）</b></summary>

在 RLHF 里，PPO 的公式形式不变，但语义映射为：
- 状态 = prompt + 已生成 token，动作 = 下一个 token，一条 response 是一条轨迹
- 奖励通常只在序列末尾由 reward model 给出，中间步骤为 0
- 额外加一项对参考模型的 KL 惩罚 $r_t \leftarrow r_t-\beta\log\frac{\pi_\theta(a_t|s_t)}{\pi_{\text{ref}}(a_t|s_t)}$，防止 reward hacking 和分布漂移（注意这个 KL 是相对 **ref 模型**，与 clip 中相对 **old 策略**的比值是两个不同的东西）
- 需要同时维护 actor、critic、reward model、reference model 四个模型，显存和工程开销大，这也是后来 GRPO、DPO 等方法出现的动机——GRPO 去掉了 critic，用同一 prompt 下多个采样的组内标准化奖励作为优势估计，其他部分（比值、clip、KL）与 PPO 基本一致。
</details>

---

## 五、面试高频追问

<details>
<summary><b>点击展开常见问题与要点回答</b></summary>

**Q1：PPO 是 on-policy 还是 off-policy？**
本质是 on-policy：数据必须来自当前（或非常接近当前）的策略，多 epoch 复用靴带在"$\pi_\theta$ 与 $\pi_{\text{old}}$ 差距很小"这个前提上，靠 clip 保证。它不能像 DQN/SAC 那样用很旧的 replay buffer。

**Q2：clip 是硬约束吗？比值一定在 $[1-\epsilon,1+\epsilon]$ 里吗？**
不是。clip 只是让越界样本梯度为零，但其他样本的更新仍会顺带改变这些样本的比值，所以实际比值经常越界，KL 也可能持续增长。这也是为什么工程上会监控 KL 做早停。

**Q3：为什么用 $\min$ 而不是直接 clip？**
见第三节：没有 $\min$ 就无法纠正"往错的方向走过头"的情况；有 $\min$ 才构成替代目标的下界，继承 TRPO 的悲观改进思想。

**Q4：为什么重要性采样只做在动作级别？**
轨迹级比值方差指数爆炸；动作级比值配合"状态分布近似不变"假设，用少量偏差换来可控方差。这个偏差正是要靠信赖域/clip 限制步长来控制的。

**Q5：$\hat A_t$ 的梯度要不要传回去？**
不传，优势是常数（`detach`）。critic 单独用 value loss 训练。

**Q6：PPO vs TRPO？**
TRPO：硬 KL 约束、二阶（Fisher-vector product + CG + line search）、有单调改进保证；PPO：一阶、clip 软约束、无理论保证但实测更好、更易实现、可与参数共享/dropout/大 batch 结合。

**Q7：如果去掉 clip，多 epoch 更新会怎样？**
比值无限制，前几个 minibatch 就可能把策略推得很远，后续样本严重 off-policy，替代目标和真实目标脱钩，常见现象是策略熵瞬间坍缩、性能崩溃且难以恢复。

**Q8：$\epsilon$ 大小的影响？**
$\epsilon$ 越大，更新越激进、样本利用越充分但越容易不稳定；越小越保守，需要更多迭代。常配合 KL 早停一起用。
</details>

---

## 一句话总结

PPO 的逻辑链是：**策略梯度只能用一次数据 → 用性能差分引理写出"新旧策略的差" → 用旧状态分布 + 动作级重要性采样得到可以在旧数据上估计的替代目标 → 这个替代目标只在局部可信，所以要限制步长 → TRPO 用硬 KL 约束（贵），PPO 用 $\min$ + $\text{clip}$ 构造替代目标的悲观下界，让越界样本梯度归零（便宜）→ 于是可以安全地对同一批数据做多轮一阶优化。**

# 说明

文章大量使用了来自 Claude Fable 5 的回答，整理而成。
