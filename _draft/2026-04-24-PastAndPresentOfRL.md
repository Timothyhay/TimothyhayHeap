---
layout: modern-article
title: Past and Present of Reinforcement Learning
date: 2026-04-24
tags: LLM
comments: true
---


# 2. DPO：Direct Preference Optimization

## 2.1 它解决什么问题？

SFT 的目标是：

> 给模型一个 prompt，让它学习“正确答案”。

但对于很多任务：

> **两个答案都正确，但一个明显比另一个好。**

例如：

```text
Prompt
 ├── chosen：回答准确、简洁、有帮助
 └── rejected：回答啰嗦、存在幻觉
```

这时候可以使用**偏好数据（preference data）**。

传统 RLHF：

$$
SFT
\rightarrow
Reward\ Model
\rightarrow
PPO
\rightarrow
Policy
$$

问题是：

* 需要训练 Reward Model
* PPO 实现复杂
* RL 训练不稳定
* 工程成本高

DPO 的核心贡献：

> **不显式训练 Reward Model，也不做 PPO，直接利用 preference pair 优化 policy。**

---

# 3. DPO 的核心思想

数据形式：

$$
(x,y_w,y_l)
$$

其中：

* \(x\)：prompt
* \(y_w\)：winner / chosen
* \(y_l\)：loser / rejected

DPO 通常保留一个冻结的 Reference Model：

$$
\pi_{\text{ref}}
$$

同时训练：

$$
\pi_\theta
$$

目标是让：

$$
\frac{\pi_\theta(y_w|x)}
{\pi_{\text{ref}}(y_w|x)}
$$

相对于：

$$
\frac{\pi_\theta(y_l|x)}
{\pi_{\text{ref}}(y_l|x)}
$$

更大。

---

## 3.1 DPO Loss

经典形式：

$$
\mathcal L_{\mathrm{DPO}}
=
-\mathbb E
\left[
\log
\sigma
\left(
\beta
\left[
\log\frac{\pi_\theta(y_w|x)}
{\pi_{\mathrm{ref}}(y_w|x)}
-
\log\frac{\pi_\theta(y_l|x)}
{\pi_{\mathrm{ref}}(y_l|x)}
\right]
\right)
\right]
$$

可以把它理解成：

> **提高 chosen 的相对概率，同时降低 rejected 的相对概率，并且相对于 reference model 进行约束。**

---

## 3.2 为什么需要 Reference Model？

这是 DPO 面试中非常值得问的一点。

如果只做：

$$
\pi_\theta(y_w|x) \uparrow
$$

模型可能会不断偏向训练数据，导致：

* distribution shift
* 过拟合
* 模型能力退化
* reward hacking 类问题

Reference Model 提供一个锚点：

$$
\pi_\theta \approx \pi_{\mathrm{ref}}
$$

同时允许模型向更符合人类偏好的方向移动。

这与 RLHF 中的 **KL constraint** 有非常深的联系。

---

# 4. DPO 与 RLHF 的关系

经典 RLHF 可以写成：

$$
\max_\pi
\mathbb E_{x,y\sim\pi}[r(x,y)]
-
\beta
D_{KL}
(\pi||\pi_{\mathrm{ref}})
$$

DPO 的重要理论结果是：

> 在特定的 Bradley-Terry preference model 和 KL-regularized RL 假设下，可以把这个 RL 问题转换成直接的 classification-like objective。

所以：

**DPO 不是“简单的二分类微调”。**

它背后实际上是在解决一个**KL-regularized preference optimization** 问题。

---

# 5. ORPO：Odds Ratio Preference Optimization

## 5.1 ORPO 解决什么问题？

DPO 已经去掉了：

* Reward Model
* PPO

但仍然需要：

* Reference Model

也就是说：

```text
DPO

Policy Model
     │
     ├──── Reference Model
     │
     └──── Preference Pair
```

ORPO 进一步提出：

> **能不能连 Reference Model 都不要？**

答案是可以。

---

# 6. ORPO 的核心思想

ORPO = **Odds Ratio Preference Optimization**

它将：

> **SFT + Preference Optimization**

合并到一个训练目标中。

其核心可以粗略理解成：

$$
\mathcal L_{\mathrm{ORPO}}
=
\mathcal L_{\mathrm{SFT}}
+
\lambda
\mathcal L_{\mathrm{OR}}
$$

其中：

### SFT 部分

让模型提高 chosen response 的 likelihood：

$$
\mathcal L_{\mathrm{SFT}}
=
-\log P_\theta(y_w|x)
$$

### Preference 部分

利用 chosen/rejected 的 **odds ratio** 进行区分。

对于 response \(y\)：

$$
Odds(y)
=
\frac{P(y|x)}
{1-P(y|x)}
$$

然后比较：

$$
\frac{Odds(y_w)}
{Odds(y_l)}
$$

通过这个 ratio，让：

$$
y_w > y_l
$$

---

# 7. ORPO 与 DPO 最重要的区别

这是面试里非常值得直接对比的地方：

|                 | SFT | DPO        | ORPO     |
| --------------- | --- | ---------- | -------- |
| 正样本             | ✓   | chosen     | chosen   |
| rejected        | ✗   | ✓          | ✓        |
| Reward Model    | ✗   | ✗          | ✗        |
| PPO             | ✗   | ✗          | ✗        |
| Reference Model | ✗   | **✓**      | **✗**    |
| SFT Loss        | ✓   | 通常单独阶段     | **直接结合** |
| Preference Loss | ✗   | ✓          | ✓        |
| 训练流程            | 简单  | 两阶段/独立偏好优化 | **单阶段**  |

所以一句话记忆：

> **DPO = 不要 Reward Model 和 PPO，但保留 Reference Model。**

> **ORPO = 连 Reference Model 也不要，并把 SFT 和 Preference Optimization 合在一起。**

---

# 8. 三者放在一起怎么理解？

可以把 LLM Alignment Pipeline 理解成：

```text
Pretrained Model
       │
       ▼
      SFT
       │
       │  教模型“应该怎么回答”
       ▼
   Instruction Model
       │
       ├───────────────┐
       │               │
      DPO             ORPO
       │               │
 Preference Pair   Preference Pair
       │               │
       ▼               ▼
  Reference Model      无 Reference
       │
       ▼
 Aligned Model
```

而 **LoRA 是另外一个维度的问题**。

它回答的是：

> **“怎么高效地更新模型参数？”**

而 DPO / ORPO 回答的是：

> **“模型应该朝什么方向更新？”**

所以实际工程中完全可以组合：

$$
\boxed{\text{LoRA}+\text{DPO}}
$$

或者：

$$
\boxed{\text{LoRA}+\text{ORPO}}
$$

例如：

```text
Base LLM
   │
   ├── Frozen Base
   │
   └── LoRA Adapter
          │
          ▼
     DPO / ORPO Training
```

---

# 9. 面试官最关心的几个问题

### Q1：LoRA 为什么能减少显存？

不是单纯因为“参数少”。

关键是：

> **冻结 Base Model 后，不需要为 Base Model 的参数保存梯度和 Optimizer States。**

训练时主要维护 LoRA 参数对应的：

* gradient
* optimizer states

所以显存节省非常明显。

---

### Q2：LoRA rank 越大越好吗？

不是。

\(r\) 越大：

* 表达能力更强
* 参数量增加
* 显存增加
* 训练成本增加
* 过拟合风险可能增加

因此 \(r\) 是**适配能力与参数效率之间的 trade-off**。

---

### Q3：DPO 为什么比 PPO 简单？

因为 DPO 把：

```text
Reward Model
      ↓
Reward
      ↓
PPO
      ↓
Policy Update
```

变成：

```text
Chosen / Rejected
        ↓
Direct Preference Loss
        ↓
Policy Update
```

避免了 RL 中复杂的 rollout、value model、advantage estimation 等过程。

---

### Q4：DPO 的 chosen/rejected 是什么？

不是：

> “正确答案 / 错误答案”

更准确地说是：

> **两个候选回答之间的人类/模型偏好关系。**

因此 preference data 可以表达：

* helpfulness
* harmlessness
* style
* reasoning quality
* factuality
* instruction following

等。

---

### Q5：DPO 为什么还需要 Reference Model？

为了提供一个**原始策略分布的参照和约束**，避免 policy 为了 preference objective 过度偏移。

---

### Q6：ORPO 为什么不需要 Reference Model？

因为 ORPO 直接在当前模型的概率分布上构造 preference objective，通过 chosen/rejected 的 odds-ratio 差异进行优化，同时包含 SFT objective。

因此不需要：

$$
\pi_{\mathrm{ref}}
$$

---

### Q7：LoRA、DPO、ORPO 是竞争关系吗？

**不是。**

这是一个很容易答错的面试点。

* **LoRA：参数更新方式**
* **DPO：偏好优化方法**
* **ORPO：偏好优化方法**

因此可以组合：

$$
\boxed{\text{LoRA + DPO}}
$$

$$
\boxed{\text{LoRA + ORPO}}
$$

---

# 10. 最后压缩成面试版

如果面试官只给你 **30 秒**，可以这样回答：

> **LoRA 是一种 PEFT 方法，通过冻结 Base Model，仅训练低秩矩阵 \(BA\) 来近似参数更新 \(\Delta W\)，从而显著降低微调的显存和参数成本。**
>
> **DPO 是一种 preference optimization 方法，它利用 chosen/rejected response pair，绕过 Reward Model 和 PPO，直接优化 policy，同时使用 frozen reference model 约束 policy 不要偏离原始分布过多。**
>
> **ORPO 可以看作进一步简化的 preference optimization，它不需要 reference model，并把 SFT loss 和 preference loss 融合到一个目标中。**
>
> **三者不是互斥的：LoRA 解决的是‘怎么高效更新模型’，DPO/ORPO 解决的是‘根据偏好数据把模型往什么方向优化’。**

如果要继续往**算法工程师面试深度**准备，下一层最值得掌握的是 **SFT → RLHF/PPO → DPO → IPO/KTO/ORPO/SimPO** 的演进关系，以及这些方法的 **loss 推导、数据格式、训练稳定性和 reward hacking**。
