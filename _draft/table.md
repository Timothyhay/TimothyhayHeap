# 从 LLM RL 发展脉络看策略优化算法演进

这是一条非常清晰的技术演进线：**从"重装备"的经典 RL（PPO），到"去 RL 化"的偏好优化（DPO/ORPO），再到"轻量 RL 回归"的可验证奖励时代（GRPO 及其变体）**。下面按时间与逻辑顺序详细展开。

---

## 发展脉络总览

```
2017  PPO          ── 经典 RL，InstructGPT/ChatGPT 的 RLHF 基石
2023  DPO          ── 绕过 RL，把偏好对齐变成监督学习
2024  ORPO         ── 更激进：连 SFT 和参考模型都合并掉
2024  GRPO         ── DeepSeekMath 提出，去掉 Critic 的 RL 回归
2025  Dr.GRPO      ── 修正 GRPO 的梯度偏差（长度/难度偏置）
2025  DAPO         ── 工程化大成：长 CoT 大规模 RL 的稳定训练配方
2025+ GSPO/CISPO…  ── 面向 MoE、off-policy 复用等场景的后续演化
```

两条主线值得先记住：

1. **计算/工程复杂度的钟摆**：PPO（4 个模型）→ DPO（2 个）→ ORPO（1 个）→ GRPO（2 个但无 Critic）。
2. **奖励信号来源的变迁**：人类偏好（RLHF）→ 偏好数据集（DPO 系）→ **可验证奖励**（数学/代码答案对错，R1 时代的 RLVR）。

---

## 1. PPO（2017 / RLHF 化 2022）：一切的起点

**背景**：PPO（Proximal Policy Optimization, Schulman et al.）原本是通用 RL 算法，被 OpenAI 在 InstructGPT 中用于 RLHF，成为 ChatGPT 的核心训练技术。

**核心思想**：策略梯度方法容易因单步更新过大而崩溃。PPO 用**裁剪的重要性采样比率**限制每次更新幅度：

$$
\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t\left[\min\Big(r_t(\theta)\,\hat{A}_t,\ \text{clip}\big(r_t(\theta),\, 1-\epsilon,\, 1+\epsilon\big)\,\hat{A}_t\Big)\right],
\quad r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}
$$

其中优势 $\hat{A}_t$ 通常用 **GAE**（广义优势估计）计算，需要一个**价值网络（Critic）**估计 $V(s_t)$。

**RLHF 中的 PPO 全家桶**需要同时维护 **4 个模型**：

| 模型 | 作用 | 是否更新 |
|---|---|---|
| Actor（策略） | 生成回复 | ✅ |
| Critic（价值） | 估计每个 token 前缀的价值 | ✅ |
| Reward Model | 给完整回复打分 | ❌ 冻结 |
| Reference Model | 计算 KL 惩罚，防止漂移 | ❌ 冻结 |

实际优化的奖励为：

$$
r_{\text{total}} = r_{\text{RM}}(x, y) - \beta\, \mathrm{KL}\big(\pi_\theta \,\|\, \pi_{\text{ref}}\big)
$$

**痛点**（正是后续所有算法的出发点）：
- **显存与工程成本极高**：4 个模型 + 采样-训练交替流水线；
- **Critic 难训**：LLM 的稀疏奖励（只在句末打分）让逐 token 价值估计非常困难，长 CoT 下价值误差被放大；
- **超参敏感**、reward hacking 等经典 RLHF 问题。

---

## 2. DPO（2023）：把 RL 从 RLHF 中"消掉"

**背景**：Rafailov et al. 的 *Direct Preference Optimization* 提出：既然 RLHF 的最优解有闭式形式，为什么还要跑 RL？

**核心推导**：KL 约束下的 RLHF 目标，其最优策略为

$$
\pi^*(y \mid x) \propto \pi_{\text{ref}}(y \mid x)\exp\left(\frac{1}{\beta} r(x, y)\right)
$$

反解出 $r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \text{const}$，代入 Bradley–Terry 偏好模型，奖励函数被**隐式地重参数化为策略本身**，得到纯监督损失：

$$
\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)\right]
$$

**意义与局限**：

| ✅ 优点 | ❌ 局限 |
|---|---|
| 无需 RM、Critic、采样循环，训练如同 SFT 般简单稳定 | **离线（off-policy）**：只学静态偏好对，无法从自己的新样本中学习 |
| 成为 2023–2024 开源社区对齐的事实标准（Zephyr、Tulu 等） | 已知会**同时降低 $y_w$ 和 $y_l$ 的概率**（只要差距拉大即可），可能损害生成质量 |
| 理论优雅：偏好数据 → 最优策略的直接映射 | 对分布外提示泛化不如 on-policy RL；上限通常低于精调良好的 PPO |

DPO 引发了庞大的 "xPO" 家族（IPO、KTO、SimPO……见补充部分），但也让社区最终意识到：**离线偏好学习替代不了在线 RL**——这为 GRPO 时代埋下伏笔。

---

## 3. ORPO（2024）：连参考模型和 SFT 阶段都不要了

**背景**：Hong et al. 的 *Odds Ratio Preference Optimization* 追求极致简化——把 **SFT + 对齐合并为单阶段**，且**无参考模型**。

**核心思想**：用**几率（odds）**而非概率来构造对比。定义

$$
\text{odds}_\theta(y \mid x) = \frac{P_\theta(y \mid x)}{1 - P_\theta(y \mid x)}
$$

总损失 = SFT 损失 + 几率比惩罚项：

$$
\mathcal{L}_{\text{ORPO}} = \underbrace{\mathcal{L}_{\text{SFT}}(y_w)}_{\text{学好回答}} + \lambda \cdot \underbrace{\left(-\log \sigma\left(\log \frac{\text{odds}_\theta(y_w \mid x)}{\text{odds}_\theta(y_l \mid x)}\right)\right)}_{\text{压制坏回答}}
$$

用 odds ratio 而非概率比的原因：在 SFT 同时进行时，概率比的梯度会过度打压 $y_l$；odds ratio 的惩罚更温和，二者可以共存于一个损失中。

**定位**：ORPO 代表了"去 RL 化"路线的极简终点（1 个模型、1 个阶段），在中小规模对齐上性价比很高；但和 DPO 一样是离线方法，天花板受限于偏好数据质量。

---

## 4. GRPO（2024）：RL 的轻量化回归，R1 时代的引擎

**背景**：DeepSeekMath 提出 *Group Relative Policy Optimization*，随后因 **DeepSeek-R1** 而闻名。此时奖励来源已经变了：不再是人类偏好 RM，而是**可验证奖励（RLVR）**——数学答案对/错、代码过/不过测试。

**核心思想**：PPO 最大的负担是 Critic。GRPO 的替代方案：对同一提示 $q$ 采样一组 $G$ 个回复，用**组内相对表现**作为优势基线，彻底删掉价值网络：

$$
\hat{A}_i = \frac{r_i - \text{mean}(r_1, \dots, r_G)}{\text{std}(r_1, \dots, r_G)}
$$

目标函数保留 PPO 式裁剪，外加显式 KL 正则：

$$
\mathcal{J}_{\text{GRPO}} = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \Big(\min\big(\rho_{i,t}\hat{A}_i,\ \text{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon)\hat{A}_i\big) - \beta\, \mathbb{D}_{\text{KL}}[\pi_\theta \| \pi_{\text{ref}}]\Big)\right]
$$

**为什么它成了 2025 年的默认算法**：
- 去掉 Critic → 显存减半、无价值估计误差问题；
- 与 RLVR 完美契合：二值奖励下"组内对比"就是天然的基线；
- 实现简单，配合 verl / OpenRLHF 等框架极易复现 R1-style 训练。

但它埋了几个坑——由后来者修正。

---

## 5. Dr.GRPO（2025）：修 Bug——"GRPO Done Right"

**背景**：*Understanding R1-Zero-Like Training*（Liu et al., 2025）对 GRPO 的梯度估计做了数学分析，发现**两个系统性偏差**：

1. **长度偏差**：损失中的 $\frac{1}{|o_i|}$ 归一化导致——**答错时，回复越长每 token 惩罚越小**。模型学会"错了就啰嗦"，这是 R1 类训练中回复长度不断膨胀的重要原因之一（并非全是"更深入思考"）。
2. **难度偏差**：除以组内 $\text{std}(r)$ 会放大接近全对或全错（低方差）问题的权重，使不同难度的问题获得失真的梯度权重。

**修正**极其简单：**删掉这两个归一化项**——

$$
\hat{A}_i = r_i - \text{mean}(r_1, \dots, r_G) \quad (\text{不再除以 std}), \qquad \text{损失聚合去掉 } \tfrac{1}{|o_i|}
$$

从而恢复无偏的策略梯度估计。结果：**相同性能下 token 效率显著提升**，错误回答不再无意义变长。

**意义**：Dr.GRPO 提醒社区，GRPO 目标函数的细节（归一化位置）并非无关紧要的实现选择，而是直接塑造模型行为的偏置来源。

---

## 6. DAPO（2025）：大规模长 CoT RL 的工程配方

**背景**：ByteDance Seed 的 *DAPO: An Open-Source LLM Reinforcement Learning System at Scale*。与 Dr.GRPO 的"理论修正"路线不同，DAPO 是**四个工程技巧的组合拳**（Decoupled Clip and Dynamic sAmpling Policy Optimization），目标是解决大规模长 CoT 训练中的**熵坍缩、梯度消失、训练发散**。用 Qwen2.5-32B base 在 AIME 2024 上达到 50 分并完全开源（代码 + 数据，基于 verl）。

四大组件：

| 技巧 | 做法 | 解决的问题 |
|---|---|---|
| **① Clip-Higher** | 裁剪区间非对称：$[1-\epsilon_{\text{low}},\ 1+\epsilon_{\text{high}}]$，如 $\epsilon_{\text{low}}=0.2,\ \epsilon_{\text{high}}=0.28$ | 对称裁剪压制低概率探索 token 的概率提升 → **熵坍缩**。放宽上界让探索性 token 能"长起来" |
| **② Dynamic Sampling** | 过滤掉全对或全错的组（$\hat{A}=0$，零梯度），持续重采样直到批次填满有效样本 | 训练后期大量提示变成"全对"，有效梯度信号越来越稀 |
| **③ Token 级损失** | 在**所有回复的所有 token** 上平均，而非先按回复平均（即也去掉了 GRPO 的样本内 $\frac{1}{|o_i|}$ 再平均的结构） | 长回复中每个 token 被系统性降权，长 CoT 学习信号不足 |
| **④ Overlong Reward Shaping** | 对超长截断的回复用软性长度惩罚，而非直接判 0 分 | 截断带来的奖励噪声污染训练信号 |

另外 DAPO **直接删掉了 KL 惩罚项**——理由是长 CoT 推理训练本来就期望策略大幅偏离初始分布，KL 约束反而碍事。这一做法此后被广泛沿用。

**Dr.GRPO vs DAPO 的关系**：二者部分重叠（都处理长度偏置），但哲学不同——Dr.GRPO 是"从梯度估计的无偏性出发做最小修正"；DAPO 是"针对每个训练崩溃症状打一个补丁"。实践中两者的技巧经常被混搭使用。

---

## 7. 值得补充的算法

<details>
<summary><b>7.1 REINFORCE / RLOO / REINFORCE++ —— "PPO 是否本来就过度设计"</b>（点击展开）</summary>

**RLOO**（Ahmadian et al., 2024, *Back to Basics*）：论证在 LLM 场景下（初始化于强 SFT 模型、每回合单次奖励），PPO 的很多组件是不必要的。RLOO 用同一提示的其余 $k-1$ 个样本的平均奖励作 leave-one-out 基线：

$$
\hat{A}_i = r_i - \frac{1}{k-1}\sum_{j \neq i} r_j
$$

概念上是 GRPO 的近亲（更早、无 std 归一化、无裁剪），无偏且极简。

**REINFORCE++**（2025）：不做 per-prompt 分组，改用**全局批次均值/标准差**归一化优势，配合 PPO 式裁剪与 token 级 KL，宣称比 GRPO 更稳、更抗 reward hacking。

这一支的意义在于：它和 GRPO 一起证明了 **Critic-free 的策略梯度就是 LLM RL 的正确抽象层级**。
</details>

<details>
<summary><b>7.2 DPO 家族的重要变体：IPO / KTO / SimPO</b>（点击展开）</summary>

- **IPO**（Azar et al., 2023）：指出 DPO 在偏好确定性高时会过拟合（把 $y_l$ 概率推向 0），改用有界的平方损失回归目标，理论上更稳。
- **KTO**（Ethayarajh et al., 2024）：基于前景理论（Kahneman-Tversky），**不需要成对偏好数据**——只需单条样本的"好/坏"二元标签即可训练，大幅降低数据收集门槛。
- **SimPO**（Meng et al., 2024）：去掉参考模型，用**长度归一化的平均 log 概率**作为隐式奖励，加上目标间隔 $\gamma$：

$$
\mathcal{L}_{\text{SimPO}} = -\log \sigma\left(\frac{\beta}{|y_w|}\log \pi_\theta(y_w \mid x) - \frac{\beta}{|y_l|}\log \pi_\theta(y_l \mid x) - \gamma\right)
$$

有趣的对照：SimPO 的长度归一化是**防止偏好学习偏爱长回复的解药**，而 Dr.GRPO 在 RL 侧删掉长度归一化——同一个"长度偏置"问题在不同目标函数结构下需要相反的处理，说明这些细节必须结合梯度形式具体分析。
</details>

### 7.3 GSPO（2025，Qwen 团队）：序列级重要性采样 ⭐ 重点补充

GRPO 的重要性比率是 **token 级**的，但奖励是**序列级**的——粒度错配导致每个 token 的比率噪声在长序列上累积，在 **MoE 模型**上尤其严重（专家路由波动使 token 级比率剧烈抖动，甚至需要"Routing Replay"等补丁）。GSPO（Group Sequence Policy Optimization）改用**序列级比率**（几何平均）并在序列级裁剪：

$$
s_i(\theta) = \left(\frac{\pi_\theta(y_i \mid x)}{\pi_{\theta_{\text{old}}}(y_i \mid x)}\right)^{1/|y_i|}
$$

使优化粒度与奖励粒度对齐，是 Qwen3 系列训练的核心算法，已成为 MoE 大规模 RL 的主流选择。

### 7.4 CISPO（2025，MiniMax-M1）：裁剪权重而非裁剪 token ⭐ 重点补充

关键观察：PPO/GRPO 的裁剪会**永久杀死**某些低概率但关键的"反思/分叉"token（如 "wait"、"however"——恰是长推理能力的种子）的梯度。CISPO 不裁剪 token 更新，而是**裁剪重要性采样权重本身**（类似截断 IS 的 REINFORCE），保证**所有 token 都保留梯度**，同时控制方差。附带好处是对 off-policy 数据复用（一批 rollout 训多步）更友好，能显著降低生成瓶颈成本。

<details>
<summary><b>7.5 其他值得知道的方向</b>（点击展开）</summary>

- **VAPO**（ByteDance, 2025）：与"去 Critic"潮流反向——论证**训好的价值网络仍是天花板更高的方案**，通过 value-pretraining、解耦 GAE 等技巧使 value-based 方法在长 CoT 上反超 DAPO。说明 Critic-free 是工程妥协而非理论最优。
- **GDPO**：多奖励目标下先对每个奖励独立归一化再合并，解决多目标对齐（如工具调用 + 格式 + 正确性）时的优势坍缩。
- **熵控制技术**（Clip-Cov / KL-Cov 等）：直接针对 RLVR 训练中的熵坍缩机制做干预，与 DAPO 的 clip-higher 属于同一问题的不同解法。
- **过程奖励（PRM）与 RLVR 之争**：R1 报告了 PRM 的 reward hacking 难题而选择纯结果奖励；但步骤级信号如何安全利用仍是活跃方向。
- **Agentic RL / 多轮 RL**：当前（2025–2026）前沿正从"单轮解题"转向多轮工具调用轨迹的 RL，信用分配和异步 rollout 是新战场，上述算法大多正在被移植改造。
</details>

---

## 8. 总览对比

| 算法 | 年份 | 范式 | Critic | 参考模型/KL | 数据来源 | 优势估计 | 一句话定位 |
|---|---|---|---|---|---|---|---|
| **PPO** | 2017/22 | 在线 RL | ✅ | ✅ | RM 打分 | GAE | RLHF 黄金标准，贵而难调 |
| **DPO** | 2023 | 离线偏好 | ❌ | ✅ | 偏好对 | —（隐式奖励） | 把 RLHF 变成监督学习 |
| **ORPO** | 2024 | 离线偏好 | ❌ | ❌ | 偏好对 | —（odds ratio） | 单模型单阶段的极简对齐 |
| **GRPO** | 2024 | 在线 RL | ❌ | ✅（KL 项） | 可验证奖励 | 组内均值+std 归一化 | R1 时代默认算法 |
| **Dr.GRPO** | 2025 | 在线 RL | ❌ | 可选 | 可验证奖励 | 组内均值（去归一化） | 无偏梯度，治长度膨胀 |
| **DAPO** | 2025 | 在线 RL | ❌ | ❌（删 KL） | 可验证奖励 | GRPO 式 + 动态采样 | 大规模长 CoT 稳定配方 |
| GSPO | 2025 | 在线 RL | ❌ | ❌ | 可验证奖励 | 组内相对 + **序列级比率** | MoE / 大规模稳定性 |
| CISPO | 2025 | 在线 RL | ❌ | ❌ | 可验证奖励 | 裁 IS 权重不裁 token | 保住反思 token、off-policy 复用 |

## 9. 实践选型建议

- **通用对话对齐、有偏好数据、算力有限** → DPO（或 SimPO/KTO），仍是性价比之王；ORPO 适合想省掉 SFT 阶段的场景。
- **推理/代码等有可验证奖励的任务** → GRPO 起步，几乎必然要叠加 Dr.GRPO 的去偏 + DAPO 的 clip-higher / dynamic sampling / token 级损失（现代框架如 verl 中这些是可组合开关）。
- **MoE 模型或超大规模训练不稳** → GSPO。
- **rollout 成本是瓶颈、想一批数据训多步** → CISPO。
- **有精细过程奖励、追求上限且工程能力强** → 别忘了 PPO/VAPO 这条 value-based 路线并没有死。

**一个收束性的观察**：这十年的演进本质上是在回答同一个问题——*在"奖励只在序列末尾出现、动作空间是整个词表、单条轨迹极长"的特殊 MDP 里，经典 RL 的哪些组件是必需的？* 答案逐渐清晰：Critic 可以用组内采样替代（GRPO），KL 约束在推理训练中可以扔掉（DAPO），但**梯度估计的归一化细节（Dr.GRPO）和裁剪粒度（GSPO/CISPO）会直接塑造模型行为**，一点都马虎不得。

如果你想深入某一个算法的推导细节（比如 DPO 的闭式解推导、或 Dr.GRPO 的偏差分析），我可以单独展开。