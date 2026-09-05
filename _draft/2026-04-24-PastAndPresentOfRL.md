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

**核心推导**：KL 约束下的 RLHF 目标，其最优策略为

**Core derivation**: Under the KL-constrained RLHF objective, the optimal policy is

$$
\pi^*(y \mid x) \propto \pi_{\text{ref}}(y \mid x)\exp\left(\frac{1}{\beta} r(x, y)\right)
$$

反解出 $r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \text{const}$，代入 Bradley–Terry 偏好模型，奖励函数被**隐式地重参数化为策略本身**，得到纯监督损失：

Inverting this gives $r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \text{const}$. Substituting into the Bradley–Terry preference model, the reward function is **implicitly reparameterized as the policy itself**, yielding a purely supervised loss:

$$
\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)\right]
$$

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
3. **Length bias**: the $\frac{1}{|o_i|}$ normalization in the loss means that **for incorrect answers, the longer the response, the smaller the per-token penalty**. The model learns to "ramble when wrong" — a major cause of ever-growing response lengths in R1-style training (not all of it is "deeper thinking").
4. **Difficulty bias**: dividing by the within-group $\text{std}(r)$ amplifies the weight of questions that are nearly all-correct or all-wrong (low variance), giving questions of different difficulty distorted gradient weights.

**修正**极其简单：**删掉这两个归一化项**——

**The fix** is extremely simple: **remove both normalization terms** —

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

# 说明

文章大量使用了来自 Claude Fable 5 的回答，整理而成。
