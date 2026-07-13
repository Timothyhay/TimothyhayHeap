---
layout: modern-article
title: A Casual Explanation of Common Attention Mechanisms
tags: LLM
comments: true
---

本文简单谈了谈笔者对注意力机制的理解，以及几种主流注意力机制做了统一风格的最小实现。

# 注意力机制的发展

2017 年之前：要解决的是**模型如何高效建模长程依赖**问题，传统 RNN 很难利用几千个 token 之前的信息，而 Transformer 允许当前位置直接访问上下文中的任意位置。这一阶段诞生了注意力机制本身，以及 Transformer 的 Q/K/V 框架和多头设计。

2019 年至今：Transformer 已成事实标准，主要矛盾转移到了"自回归推理时如何降低 KV Cache 的显存占用与访存带宽压力"。MQA、GQA、MLA 都是为解决这一矛盾而生，它们并不改变注意力的数学本质，而是在 K/V 的"存储与共享方式"上做文章。

---
### 在开始之前，注意力是什么

$$
Attention(Q,K,V)=softmax
\left(
    \frac{QK^T}{\sqrt d}
\right)V
$$

Attention 对于当前 token，模型首先计算它与历史所有 token 的相关性，然后根据相关性加权聚合历史信息。经过大量训练后，不同的 Attention Head 会逐渐形成专门化功能。有些 Head 负责代词指代，有些负责括号匹配，有些负责代码结构，有些负责数学符号关系。这些现象已经被大量 Mechanistic Interpretability 研究观察到。

真正有趣的地方在于，当参数规模和数据规模持续增长时，模型内部会逐渐形成越来越抽象的表示。早期层可能学习词法和语法，中间层学习概念和实体关系，高层则学习更复杂的抽象结构。这种层级表示与人脑皮层中从低级特征到高级语义的处理方式有一定相似性，虽然两者机制完全不同。

从训练工程的角度看，预训练最重要的问题其实不是模型结构，而是数据分布。因为模型学到的不是“真理”，而是数据中的统计规律。如果某类模式在数据中占比更高，模型就会更倾向于学习它。例如一个模型如果训练语料中代码占 30%，另一个模型代码只占 5%，即使参数量完全相同，它们的代码能力也会有巨大差异。今天很多开源模型之间的性能差距，很大程度上来自数据配方（Data Mixture）而不是架构创新。

这也是为什么现代预训练越来越强调数据工程。预训练团队花费大量时间进行去重、质量过滤、文档分类和数据配比，并不是因为这些工作很优雅，而是因为数据分布最终决定了模型学到什么样的世界模型。对于一个 70B 参数模型而言，如果训练数据中几乎没有高质量数学证明，那么它几乎不可能凭空涌现出很强的数学推理能力。

进一步说，预训练实际上存在一种知识压缩过程。假设一个 70B 模型使用 bfloat16 存储，其参数容量大约只有 140GB。但它可能阅读了十几万 GB 的文本数据。这意味着模型不可能逐字记忆训练集，而必须把大量冗余信息压缩成更抽象的结构。因此我们看到模型能够泛化到从未见过的问题。它记住的不是具体句子，而是生成这些句子的底层规律。

在大规模训练中，还有一个极其重要的经验规律叫 Scaling Law。研究者发现，当参数量、数据量和训练计算量同步增加时，训练损失会按照幂律下降，大致满足：

$$
L(C)
====

A C^{-\alpha}
+
B
$$

其中 (C) 是计算预算。这意味着模型性能不会突然饱和，而是持续改善，只是边际收益越来越低。这一发现极大改变了整个行业的发展方向，因为它说明堆算力、堆数据、堆参数并非盲目行为，而是一种有理论依据的工程路线。

后来 DeepMind 的 Chinchilla 工作进一步发现，很多模型实际上处于“参数过多、数据不足”的状态。对于固定计算预算而言，适当减小模型规模、增加训练 token 数量，反而可以获得更好的结果。这也是为什么现代模型越来越重视数据量，许多模型的预训练 token 已经达到十万亿级别。

如果把整个 LLM 训练流程放在一起理解，我认为最准确的认知框架是：预训练阶段负责建立压缩后的世界模型，使模型能够理解语言、知识和现实世界中的统计结构；SFT 阶段负责把这个世界模型转换为符合人类期望的行为模式；而 DPO、RLHF、GRPO、RFT 等后训练阶段，则是在行为空间中进一步优化偏好、推理和决策能力。

因此，对于真正参与训练的人来说，预训练最值得深入理解的其实不是“预测下一个 token”这句话本身，而是三个更深层的问题：第一，为什么仅靠预测 token 就能形成世界模型；第二，数据分布如何决定最终能力边界；第三，梯度下降为什么会在超大规模参数空间中自发形成抽象表示。这三个问题至今仍然是大模型研究最核心、也最有趣的部分。


# 实现

这里重在讨论：

- **多头注意力（MHA）**：2017 年开山之作《Attention Is All You Need》提出，是当前主流 LLM 的基石。每个头都有自己独立的 Q、K、V。
> - **缩放点积注意力（SDPA）**：注意力计算的最小单元，在 2017 年《Attention Is All You Need》提出，在点积注意力基础上加入 $\frac{1}{\sqrt{d_k}}$ 缩放。注意力机制本身最早可追溯到 2014 年 Bahdanau 等人的**加性注意力**，但 Q/K/V 框架与缩放点积形式均来自 Transformer。
- **多查询注意力（MQA）**：出自 2019 年《Fast Transformer Decoding: One Write-Head is All You Need》。所有头共享同一份 K、V，只保留各自独立的 Q，从而大幅压缩 KV Cache。代表模型有 PaLM、StarCoder、Falcon 等。
- **分组查询注意力（GQA）**：出自 2023 年《GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints》，是 MHA 与 MQA 的折中。设头数为 $h$、组数为 $g$（$g$ 整除 $h$）：$g=h$ 时退化为 MHA，$g=1$ 时退化为 MQA，$1<g<h$ 时 KV Cache 压缩到 $\frac{g}{h}$。代表模型有 Llama-2-70B（$g=8$）与 Llama-3 全系列。
- **多头隐注意力（MLA）**：出自 2024 年 DeepSeek-V2 技术报告，用低秩线性变换压缩 K、V，只缓存一个低维潜向量，进一步压缩 KV Cache。

> **关于 KV Cache**：在自回归生成中，新预测的第 $t$ 个 token 不会改变已经算好的前 $t-1$ 个位置的 K、V，因此可以把它们缓存下来，避免重复计算。Prefill（预填充）阶段会一次性算出 prompt 全部 token 的 K、V 并写入缓存，KV Cache 的加速收益主要体现在逐 token 的 decode 阶段。

> 缩放因子写作 `head_dim ** 0.5`、掩码统一用 `masked_fill`（约定 `True` 表示被遮蔽）、注释和形状标注统一。
---

## 一、缩放点积注意力（SDPA）

它是 MHA 的内部基本组件：先计算 query 与每个 key 的关联度作为权重，再对 value 加权求和。

```python
import torch
from torch import nn


class ScaledDotProductAttention(nn.Module):
    """缩放点积注意力 (Scaled Dot-Product Attention)。

    注意力计算的最小单元。在普通点积注意力基础上加入 1/sqrt(d_k) 缩放，
    避免 d_k 较大时点积数值过大、使 softmax 落入梯度极小的饱和区。
    本身不区分多头，可被 MHA/MQA/GQA 复用。
    """

    def forward(self, query, key, value, causal_mask=None, padding_mask=None):
        # query / key / value 形状: (..., seq_len, head_dim)
        # 其中 "..." 可为 (batch,) 或 (batch, num_heads)，逻辑完全相同
        d_k = query.size(-1)

        # 1) 注意力分数 QK^T / sqrt(d_k)，形状: (..., seq_len, seq_len)
        scores = torch.matmul(query, key.transpose(-1, -2)) / (d_k ** 0.5)

        # 2) 因果掩码: True 表示"未来 token"，置为 -inf。形状需可广播到 scores
        if causal_mask is not None:
            scores = scores.masked_fill(causal_mask, float("-inf"))

        # 3) padding 掩码: True 表示 padding 位置，形状需可广播到 scores
        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask, float("-inf"))

        # 4) softmax 归一化得到注意力权重，再对 value 加权求和
        probs = torch.softmax(scores, dim=-1)            # (..., seq_len, seq_len)
        output = torch.matmul(probs, value)              # (..., seq_len, head_dim)
        return output


def test_sdpa():
    torch.manual_seed(0)
    batch_size, seq_len, head_dim = 2, 5, 16
    query = torch.randn(batch_size, seq_len, head_dim)
    key = torch.randn(batch_size, seq_len, head_dim)
    value = torch.randn(batch_size, seq_len, head_dim)

    # 因果掩码: 上三角(不含对角线)为 True，即屏蔽未来位置
    causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()

    output = ScaledDotProductAttention()(query, key, value, causal_mask=causal_mask)
    print("SDPA  in:", tuple(query.shape), " out:", tuple(output.shape))


if __name__ == "__main__":
    test_sdpa()
```

---

## 二、多头注意力（MHA）

「多头」指并行计算多组注意力，每个头从不同的子空间「视角」捕获信息。设输入为 $X$（一批 token 向量），$W^Q,W^K,W^V\in\mathbb{R}^{d_{model}\times d_{model}}$，将输出切分为 $h$ 个头、每个头维度 $d_k=d_{model}/h$。计算步骤如下：

**1）线性投影得到 Q、K、V：**

$$Q = XW^Q,\qquad K = XW^K,\qquad V = XW^V$$

**2）计算第 $i$ 个头的缩放点积得分（$d_k$ 为每个头的维度）：**

$$\text{scores}_i=\frac{Q_iK_i^{\top}}{\sqrt{d_k}}$$

**3）用 Softmax 得到注意力权重：**

$$A_i=\mathrm{softmax}\!\left(\frac{Q_iK_i^{\top}}{\sqrt{d_k}}\right)$$

**4）用注意力权重对 $V$ 加权求和，得到每个头的输出：**

$$\text{head}_i = A_iV_i=\mathrm{softmax}\!\left(\frac{Q_iK_i^{\top}}{\sqrt{d_k}}\right)V_i$$

**5）拼接所有头的输出，再乘以输出投影矩阵 $W^O$，得到最终结果：**

$$\mathrm{MultiHead}(X)=\mathrm{Concat}(\text{head}_1,\dots,\text{head}_h)\,W^O$$

实现代码如下：

```python
import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size 必须能被 num_heads 整除"
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads      # 每个头的维度

        # Q/K/V/O 投影矩阵，维度均保持为 hidden_size
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        self.o_linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_state, causal_mask=None, padding_mask=None):
        # hidden_state 形状: (batch_size, seq_len, hidden_size)
        batch_size = hidden_state.size(0)

        # 1) 线性投影 -> (batch_size, seq_len, hidden_size)
        query = self.q_linear(hidden_state)
        key = self.k_linear(hidden_state)
        value = self.v_linear(hidden_state)

        # 2) 拆分多头 -> (batch_size, num_heads, seq_len, head_dim)
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 3) 缩放点积得分 -> (batch_size, num_heads, seq_len, seq_len)
        scores = torch.matmul(query, key.transpose(-1, -2)) / (self.head_dim ** 0.5)

        # 4) 掩码 (True 处填 -inf)
        if causal_mask is not None:                          # (seq_len, seq_len)，自动广播
            scores = scores.masked_fill(causal_mask, float("-inf"))
        if padding_mask is not None:                         # (batch_size, seq_len)
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(1)  # -> (batch_size, 1, 1, seq_len)
            scores = scores.masked_fill(padding_mask, float("-inf"))

        # 5) softmax + 加权求和 -> (batch_size, num_heads, seq_len, head_dim)
        probs = torch.softmax(scores, dim=-1)
        # 如需 dropout 可加在这里: probs = self.dropout(probs)
        output = torch.matmul(probs, value)

        # 6) 合并多头 -> (batch_size, seq_len, hidden_size)，再做输出投影
        #    注意: transpose 后张量非连续，必须用 reshape(而非 view)
        output = output.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_dim)
        output = self.o_linear(output)                       # (batch_size, seq_len, hidden_size)
        return output


def test_MHA():
    torch.manual_seed(0)
    batch_size, seq_len, hidden_size, num_heads = 4, 16, 64, 8

    hidden_state = torch.randn(batch_size, seq_len, hidden_size)
    causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()

    mha = MultiHeadAttention(hidden_size, num_heads)
    output = mha(hidden_state, causal_mask=causal_mask)

    print("MHA   in:", tuple(hidden_state.shape), " out:", tuple(output.shape))


if __name__ == "__main__":
    test_MHA()
```

---

## 三、带 KV Cache 的 MHA

KV Cache 通过缓存并逐步追加 K、V，在 decode 阶段以空间换时间。下面演示逐 token 解码：每一步只输入一个新 token，与缓存的历史 K、V 拼接后做注意力——由于新 query 天然只能看到自己和历史，所以无需再额外加因果掩码。

```python
import torch
from torch import nn


class MultiHeadAttentionWithCache(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        self.o_linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_state, past_key_value=None, use_cache=False):
        # decode 阶段 hidden_state 形状: (batch_size, 1, hidden_size)
        batch_size = hidden_state.size(0)

        # 1) 只对当前(新)token 计算 Q、K、V
        query = self.q_linear(hidden_state)
        key = self.k_linear(hidden_state)
        value = self.v_linear(hidden_state)

        # 2) 拆分多头 -> (batch_size, num_heads, q_len, head_dim)
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 3) 若存在缓存，沿 seq 维拼接历史 K、V
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key = torch.cat([past_key, key], dim=2)          # (batch_size, num_heads, seq_len, head_dim)
            value = torch.cat([past_value, value], dim=2)

        # 4) 保存更新后的缓存
        new_past_key_value = (key, value) if use_cache else None

        # 5) 缩放点积注意力: 当前 query 关注全部历史 key，无需额外因果掩码
        scores = torch.matmul(query, key.transpose(-1, -2)) / (self.head_dim ** 0.5)
        probs = torch.softmax(scores, dim=-1)
        output = torch.matmul(probs, value)                  # (batch_size, num_heads, q_len, head_dim)

        # 6) 合并多头 + 输出投影
        output = output.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_dim)
        output = self.o_linear(output)

        return (output, new_past_key_value) if use_cache else output


def test_MHA_with_cache():
    torch.manual_seed(0)
    batch_size, seq_len, hidden_size, num_heads = 2, 5, 64, 4

    hidden_state = torch.randn(batch_size, seq_len, hidden_size)
    mha = MultiHeadAttentionWithCache(hidden_size, num_heads)   # 注意: 必须先实例化模型

    past_key_value = None
    outputs = []
    for i in range(seq_len):
        current_input = hidden_state[:, i:i + 1, :]            # 当前单个 token
        output_step, past_key_value = mha(
            current_input,
            past_key_value=past_key_value,
            use_cache=True,
        )
        outputs.append(output_step)

    output = torch.cat(outputs, dim=1)                          # 合并各步输出
    print("MHA+KV in:", tuple(hidden_state.shape), " out:", tuple(output.shape))


if __name__ == "__main__":
    test_MHA_with_cache()
```

---

## 四、多查询注意力（MQA）

所有头共享同一份 K、V（投影维度仅为 `head_dim`），Q 仍按头独立。计算时让单头的 K、V 通过广播与多头 Q 相乘。

```python
import torch
from torch import nn


class MultiQueryAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # 注意: K、V 只投影到 head_dim(单头)，所有头共享
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, self.head_dim)
        self.v_linear = nn.Linear(hidden_size, self.head_dim)
        self.o_linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_state, causal_mask=None, padding_mask=None):
        batch_size = hidden_state.size(0)

        query = self.q_linear(hidden_state)   # (batch_size, seq_len, hidden_size)
        key = self.k_linear(hidden_state)     # (batch_size, seq_len, head_dim)
        value = self.v_linear(hidden_state)   # (batch_size, seq_len, head_dim)

        # 拆分头部: Q 拆成 num_heads 个头，K、V 仅 1 个头
        query = self.split_head(query)        # (batch_size, num_heads, seq_len, head_dim)
        key = self.split_head(key, 1)         # (batch_size, 1, seq_len, head_dim)
        value = self.split_head(value, 1)     # (batch_size, 1, seq_len, head_dim)

        # 缩放点积得分: 单头 K 自动广播到所有头 -> (batch_size, num_heads, seq_len, seq_len)
        scores = torch.matmul(query, key.transpose(-1, -2)) / (self.head_dim ** 0.5)

        if causal_mask is not None:
            scores = scores.masked_fill(causal_mask, float("-inf"))
        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(padding_mask, float("-inf"))

        probs = torch.softmax(scores, dim=-1)
        output = torch.matmul(probs, value)   # (batch_size, num_heads, seq_len, head_dim)

        # 合并多头 + 输出投影
        output = output.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_dim)
        output = self.o_linear(output)        # (batch_size, seq_len, hidden_size)
        return output

    def split_head(self, x, head_num=None):
        batch_size = x.size(0)
        if head_num is None:
            head_num = self.num_heads
        # -> (batch_size, head_num, seq_len, head_dim)
        return x.view(batch_size, -1, head_num, self.head_dim).transpose(1, 2)


def test_MQA():
    torch.manual_seed(0)
    batch_size, seq_len, hidden_size, num_heads = 4, 16, 64, 8

    hidden_state = torch.randn(batch_size, seq_len, hidden_size)
    causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()

    mqa = MultiQueryAttention(hidden_size, num_heads)
    output = mqa(hidden_state, causal_mask=causal_mask)
    print("MQA   in:", tuple(hidden_state.shape), " out:", tuple(output.shape))


if __name__ == "__main__":
    test_MQA()
```

---

## 五、分组查询注意力（GQA）

把 $h$ 个头分成 $g$ 组，每组共享一对 K、V，组内的多个 query 头复用同一份 K、V。

```python
import torch
from torch import nn


class GroupedQueryAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, group_num):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size 必须能被 num_heads 整除"
        assert num_heads % group_num == 0, "num_heads 必须能被 group_num 整除"
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.group_num = group_num                         # KV 组数 (g=num_heads 即 MHA, g=1 即 MQA)

        # K、V 投影到 group_num * head_dim (折中)
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, group_num * self.head_dim)
        self.v_linear = nn.Linear(hidden_size, group_num * self.head_dim)
        self.o_linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_state, causal_mask=None, padding_mask=None):
        batch_size = hidden_state.size(0)

        query = self.q_linear(hidden_state)   # (batch_size, seq_len, hidden_size)
        key = self.k_linear(hidden_state)     # (batch_size, seq_len, group_num * head_dim)
        value = self.v_linear(hidden_state)   # (batch_size, seq_len, group_num * head_dim)

        # 拆分头部: Q 为 num_heads 个头; K、V 先按组拆分再复制扩展到 num_heads
        query = self.split_head(query)                       # (batch_size, num_heads, seq_len, head_dim)
        key = self.split_head(key, self.group_num)           # (batch_size, num_heads, seq_len, head_dim)
        value = self.split_head(value, self.group_num)       # (batch_size, num_heads, seq_len, head_dim)

        scores = torch.matmul(query, key.transpose(-1, -2)) / (self.head_dim ** 0.5)

        if causal_mask is not None:
            scores = scores.masked_fill(causal_mask, float("-inf"))
        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(padding_mask, float("-inf"))

        probs = torch.softmax(scores, dim=-1)
        output = torch.matmul(probs, value)   # (batch_size, num_heads, seq_len, head_dim)

        output = output.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_dim)
        output = self.o_linear(output)        # (batch_size, seq_len, hidden_size)
        return output

    def split_head(self, x, group_num=None):
        batch_size, seq_len = x.size()[:2]

        # Q: 直接拆成 num_heads 个头
        if group_num is None:
            return x.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # K/V: 先拆成 group_num 个组 -> (batch_size, group_num, seq_len, head_dim)
        x = x.view(batch_size, -1, group_num, self.head_dim).transpose(1, 2)
        # 每组复制 num_heads // group_num 份，扩展到 num_heads
        # 注意: expand 后张量非连续，必须用 reshape(而非 view)
        x = (
            x[:, :, None, :, :]
            .expand(batch_size, group_num, self.num_heads // group_num, seq_len, self.head_dim)
            .reshape(batch_size, self.num_heads, seq_len, self.head_dim)
        )
        return x                              # (batch_size, num_heads, seq_len, head_dim)


def test_GQA():
    torch.manual_seed(0)
    batch_size, seq_len, hidden_size, num_heads, group_num = 4, 16, 64, 8, 2

    hidden_state = torch.randn(batch_size, seq_len, hidden_size)
    causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()

    gqa = GroupedQueryAttention(hidden_size, num_heads, group_num)
    output = gqa(hidden_state, causal_mask=causal_mask)
    print("GQA   in:", tuple(hidden_state.shape), " out:", tuple(output.shape))


if __name__ == "__main__":
    test_GQA()
```

---

## 六、多头隐注意力（MLA）

MLA 的核心是**低秩压缩**：把 K、V 联合压缩成一个低维潜向量 $c^{KV}$，推理时**只缓存这个潜向量**，需要时再升维还原出各头的 K、V，从而把 KV Cache 压到极小。

由于 RoPE 是位置相关的、无法与「升维」矩阵直接合并，MLA 采用**解耦 RoPE**：额外用一小段、所有头共享的维度专门承载位置编码，与不带位置编码的「内容」维度拼接。简化的公式（$h$ 为输入隐状态）：

- 查询低秩压缩：$c^{Q}=hW^{DQ}$，再升维并拆成内容与位置两部分 $[q^{C},\,q^{R}]=c^{Q}W^{UQ}$
- KV 低秩压缩：$c^{KV}=hW^{DKV}$（**被缓存**），升维得到 $k^{C}=c^{KV}W^{UK}$、$v^{C}=c^{KV}W^{UV}$
- 解耦 RoPE：$q^{R}=\mathrm{RoPE}(q^{R})$，共享的 $k^{R}=\mathrm{RoPE}(hW^{KR})$（**也被缓存**）
- 拼接得到完整 Q、K：$q=[q^{C};\,q^{R}]$，$k=[k^{C};\,k^{R}]$，再走标准缩放点积注意力

```python
import torch
from torch import nn


class RMSNorm(nn.Module):
    """DeepSeek 中低秩压缩后使用的归一化层。"""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * self.weight


def precompute_rope(seq_len, dim, base=10000.0):
    """预计算 RoPE 的 cos/sin，dim 必须为偶数。返回形状均为 (seq_len, dim)。"""
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)            # (seq_len, dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)     # (seq_len, dim)
    return emb.cos(), emb.sin()


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x, cos, sin):
    # x: (batch, num_heads, seq_len, dim); cos/sin: (seq_len, dim) 自动广播
    return x * cos + rotate_half(x) * sin


class MultiHeadLatentAttention(nn.Module):
    def __init__(self, hidden_size, num_heads,
                 q_lora_rank, kv_lora_rank,
                 qk_nope_head_dim, qk_rope_head_dim, v_head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.qk_nope_head_dim = qk_nope_head_dim       # 不含位置编码的 Q/K 维度
        self.qk_rope_head_dim = qk_rope_head_dim       # 承载 RoPE 的 Q/K 维度
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.kv_lora_rank = kv_lora_rank

        # ---- Query 低秩压缩: hidden -> q_lora_rank -> num_heads * qk_head_dim ----
        self.q_down = nn.Linear(hidden_size, q_lora_rank, bias=False)
        self.q_norm = RMSNorm(q_lora_rank)
        self.q_up = nn.Linear(q_lora_rank, num_heads * self.qk_head_dim, bias=False)

        # ---- KV 低秩压缩: hidden -> [kv_lora_rank | qk_rope_head_dim] ----
        #   kv_lora_rank 段是被缓存的潜向量 c_kv
        #   qk_rope_head_dim 段是所有头共享、携带位置编码的 key
        self.kv_down = nn.Linear(hidden_size, kv_lora_rank + qk_rope_head_dim, bias=False)
        self.kv_norm = RMSNorm(kv_lora_rank)
        self.kv_up = nn.Linear(kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim), bias=False)

        self.o_linear = nn.Linear(num_heads * v_head_dim, hidden_size, bias=False)

    def forward(self, hidden_state, cos, sin, causal_mask=None):
        bsz, seq_len, _ = hidden_state.shape

        # ===== Query 分支: 下投影 -> 归一化 -> 上投影 =====
        q = self.q_up(self.q_norm(self.q_down(hidden_state)))            # (b, s, n_h*qk_head_dim)
        q = q.view(bsz, seq_len, self.num_heads, self.qk_head_dim).transpose(1, 2)
        q_nope, q_rope = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # ===== KV 分支: 下投影得到 [潜向量 c_kv | 共享 rope key] =====
        kv = self.kv_down(hidden_state)                                  # (b, s, kv_lora_rank + qk_rope_head_dim)
        c_kv, k_rope = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        # 推理时只需缓存 c_kv (b, s, kv_lora_rank) 与 k_rope (b, s, qk_rope_head_dim)

        kv = self.kv_up(self.kv_norm(c_kv))                              # (b, s, n_h*(qk_nope+v))
        kv = kv.view(bsz, seq_len, self.num_heads,
                     self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2)
        k_nope, value = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # 共享的 rope key -> (b, 1, s, qk_rope_head_dim)
        k_rope = k_rope.view(bsz, seq_len, 1, self.qk_rope_head_dim).transpose(1, 2)

        # ===== 解耦 RoPE: 只作用在 rope 子空间 =====
        q_rope = apply_rope(q_rope, cos, sin)                            # (b, n_h, s, qk_rope_head_dim)
        k_rope = apply_rope(k_rope, cos, sin)                            # (b, 1,   s, qk_rope_head_dim)

        # ===== 拼接 nope 与 rope，得到完整 Q / K =====
        query = torch.cat([q_nope, q_rope], dim=-1)                      # (b, n_h, s, qk_head_dim)
        k_rope = k_rope.expand(bsz, self.num_heads, seq_len, self.qk_rope_head_dim)
        key = torch.cat([k_nope, k_rope], dim=-1)                        # (b, n_h, s, qk_head_dim)

        # ===== 缩放点积注意力 =====
        scores = torch.matmul(query, key.transpose(-1, -2)) / (self.qk_head_dim ** 0.5)
        if causal_mask is not None:
            scores = scores.masked_fill(causal_mask, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, value)                                 # (b, n_h, s, v_head_dim)

        out = out.transpose(1, 2).reshape(bsz, seq_len, self.num_heads * self.v_head_dim)
        return self.o_linear(out)                                        # (b, s, hidden_size)


def test_MLA():
    torch.manual_seed(0)
    batch_size, seq_len, hidden_size, num_heads = 2, 6, 128, 4
    qk_nope_head_dim, qk_rope_head_dim, v_head_dim = 16, 8, 16   # qk_rope_head_dim 须为偶数

    mla = MultiHeadLatentAttention(
        hidden_size, num_heads,
        q_lora_rank=48, kv_lora_rank=32,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
    )

    hidden_state = torch.randn(batch_size, seq_len, hidden_size)
    cos, sin = precompute_rope(seq_len, qk_rope_head_dim)
    causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()

    output = mla(hidden_state, cos, sin, causal_mask=causal_mask)
    print("MLA   in:", tuple(hidden_state.shape), " out:", tuple(output.shape))


if __name__ == "__main__":
    test_MLA()
```

---

### 几点统一说明

- **掩码约定**：`causal_mask`/`padding_mask` 中 `True` 表示「需要被遮蔽」的位置，用 `masked_fill(mask, -inf)` 屏蔽；`causal_mask = torch.triu(ones, diagonal=1).bool()` 即屏蔽未来位置。
- **合并多头**：`transpose` 之后张量内存非连续，统一用 `reshape`（等价于 `.contiguous().view(...)`），避免 `view` 报错。
- **MLA 简化**：为聚焦核心思想，这里省略了 DeepSeek-V2 实际使用的 YaRN/`mscale` 缩放修正与「矩阵吸收（absorb）」等推理加速技巧；`qk_rope_head_dim` 需为偶数以适配 RoPE。
- 每个文件独立 `import`、独立 `test_*`，可单独运行；上面的形状逻辑我已用 NumPy 等价模拟验证（GQA 扩展、MLA 低秩压缩与解耦 RoPE 的形状/广播均正确）。建议你在本地装好 PyTorch 后再各跑一遍 `test_*` 做最终确认。

如果你愿意，我可以再补一段把这些模块统一在一起、加上前馈层与残差/归一化的「完整 Transformer Decoder Block」示例，作为教材的收尾章节。

---

## 一、史实与概念错误

### 1.（重要）缩放点积注意力的出处搞错了

原文：

> 缩放点积注意力（Scaled Dot-Product Attention）：2014 年《Neural Machine Translation by Jointly Learning to Align and Translate》提出的单头注意力……

这里有两个事实错误：

- **2014 年 Bahdanau 等人的那篇论文提出的是「加性注意力 / Additive (Bahdanau) Attention」**，用一个前馈网络（concat + tanh）来计算对齐分数，并**不是**点积形式，更没有 $1/\sqrt{d_k}$ 的缩放。
- **「缩放点积注意力」这个具体形式恰恰是 2017 年《Attention is All You Need》提出的**，缩放因子 $\frac{1}{\sqrt{d_k}}$ 正是该论文的贡献。点积式注意力的雏形一般追溯到 Luong 等人 2015 年的工作，但「缩放版」来自 Transformer。

另外，原文后面还有一句：

> 缩放点积注意力早于 Transformer 被提出，受到的关注并不多……

这句也不准确——**注意力机制**确实早于 Transformer（2014/2015），但**「缩放点积注意力」本身就是 Transformer 提出的**，并非早于它。

**建议改为：**

> 缩放点积注意力（Scaled Dot-Product Attention）是 2017 年《Attention is All You Need》中定义的注意力计算单元，其核心是在点积注意力的基础上加入了 $\frac{1}{\sqrt{d_k}}$ 缩放因子，以缓解 $d_k$ 较大时点积数值过大、导致 softmax 梯度消失的问题。它是多头注意力（MHA）的内部基本组件。

并把概念段落里「2014 年那篇论文提出单头注意力」的归属删掉或改成「注意力机制最早可追溯到 2014 年 Bahdanau 等人的加性注意力，而缩放点积形式由 Transformer 提出」。同时注意：原文用 Query/Key/Value 的框架去描述 2014 年的工作其实是「时代错位」——Q/K/V 的统一表述也是 Transformer 才有的。

### 2. MQA 的使用者列表存疑

原文：

> 使用 MQA 的模型包括 PaLM、StarCoder、Gemini 等。

- PaLM、StarCoder 使用 MQA 是论文/技术报告里明确写过的，没问题。
- **Gemini 的注意力结构并未公开披露**，把它列为 MQA 的代表缺乏可靠依据。更稳妥、且公开确认使用 MQA 的代表是 **Falcon（如 Falcon-40B）**。

建议把 Gemini 换成 Falcon，或至少去掉 Gemini。

### 3. KV Cache 与 Prefill 的描述容易误导

原文：

> 即使是 Decoder-only 的模型，在预处理输入（Prefill）的时候也不需要利用 KV Cache（P/D 分离）……

这句不准确。实际上：

- **Prefill 阶段会一次性计算 prompt 所有 token 的 K、V，并把它们写入 KV Cache**，供后续 decode 复用。所以并不是「prefill 不需要 KV Cache」，而是「KV Cache 的*加速收益*主要体现在逐 token 的 decode 阶段」。
- **P/D 分离（Prefill/Decode disaggregation）** 是一种把预填充和解码放到不同实例/硬件上执行的部署优化，和「prefill 是否需要 KV Cache」是两回事，放在这里当作括号注解会让读者混淆。

建议改为：

> KV Cache 主要用于 Decoder 在逐 token 生成（decode）阶段减少重复计算：Prefill 阶段会一次性算出 prompt 所有 token 的 K、V 并写入缓存，decode 阶段每生成一个新 token，只需计算它自己的 Q、K、V，再与缓存中的历史 K、V 拼接即可。本代码仅作示例。

### 4. GQA 段落里的公式变量缺失

补全后应为（设头数为 $h$，组数为 $g$）：

> 将所有 Head 分为 $g$ 个组（要求 $g$ 整除 $h$），每组共享同一对 K、V。当 $g=h$ 时退化为 MHA，$g=1$ 时退化为 MQA；当 $1<g<h$ 时，KV Cache 压缩到 MHA 的 $\frac{g}{h}$，压缩率不如 MQA 的 $\frac{1}{h}$，但保留了更大的自由度，效果更有保证。在 Llama-2/3-70B 中 $g=8$。

---

## 二、代码 Bug（会直接报错）

### Bug 1：`test_MHA_with_cache()` 里 `mha` 未定义

这个测试函数里循环调用了 `mha(...)`，但**从头到尾没有实例化 `mha`**，运行会直接抛 `NameError: name 'mha' is not defined`。需要在循环前补上模型实例化：

```python
def test_MHA_with_cache():
    batch_size = 2
    seq_len = 5
    hidden_size = 64
    num_heads = 4

    hidden_state = torch.randn(batch_size, seq_len, hidden_size)
    causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()

    mha = MultiHeadAttention(hidden_size, num_heads)   # ← 缺了这一行

    past_key_value = None
    outputs = []
    for i in range(seq_len):
        current_input = hidden_state[:, i:i+1, :]
        current_causal_mask = causal_mask[i:i+1, :i+1]   # 全 False，等价于不掩码（用全量缓存时本就无需再掩码）
        output_step, past_key_value = mha(
            current_input,
            causal_mask=current_causal_mask,
            past_key_value=past_key_value,
            use_cache=True
        )
        outputs.append(output_step)

    output = torch.cat(outputs, dim=1)
    print("Input shape:", hidden_state.shape)
    print("Output shape:", output.shape)
```

（顺带说明：用全量 KV Cache 做 decode 时，单个 query 本来就可以看到所有已缓存的历史 key，`current_causal_mask` 实际上全是 `False`、不起任何作用，这里保留只是为了和接口对齐，逻辑上没问题。）

### Bug 2：GQA 的 `split_head` 中 `.expand(...).view(...)` 会报错

```python
x = x[:, :, None, :, :].expand(...).view(batch_size, self.num_heads, seq_len, self.head_dim)
```

`expand()` 返回的是**非连续（non-contiguous）张量**（被扩展维度的 stride 为 0），紧接着对它调用 `.view()` 去合并维度时，PyTorch 会抛出：

```
RuntimeError: view size is not compatible with input tensor's size and stride
(at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
```

把最后的 `.view` 改成 `.reshape` 即可（`reshape` 会在必要时自动 `contiguous`）：

```python
def split_head(self, x, group_num=None):
    batch_size, seq_len = x.size()[:2]

    if group_num is None:
        return x.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
    else:
        # (batch_size, group_num, seq_len, head_dim)
        x = x.view(batch_size, -1, group_num, self.head_dim).transpose(1, 2)
        # 每组 K/V 复制 num_heads // group_num 份
        x = (
            x[:, :, None, :, :]
            .expand(batch_size, group_num, self.num_heads // group_num, seq_len, self.head_dim)
            .reshape(batch_size, self.num_heads, seq_len, self.head_dim)   # ← view 改成 reshape
        )
        return x
```

---

## 三、可选的小改进（不影响正确性）

这些不算错误，但作为教材会更严谨：

- **缩放因子的设备/写法**：`torch.sqrt(torch.tensor(d_k, dtype=torch.float32))` 每次都在 CPU 上新建张量，放到 GPU 训练时会有设备不一致风险。更简洁稳妥的写法是直接 `attention_scores = ... / (self.head_dim ** 0.5)`。
- **掩码写法**：`attention_scores += mask * -1e9` 在混合精度（fp16/bf16）下，`-1e9` 可能溢出/精度不佳，且 `+=` 原地操作偶尔会干扰 autograd。教材里更推荐 `attention_scores = attention_scores.masked_fill(mask, float('-inf'))`，语义也更清晰。
- **SDPA 代码注释**：`ScaledDotProductAttention` 里几处注释写了 `(batch_size, num_heads, seq_len, ...)`，但这个类本身并没有 `num_heads` 维度（heads 是 MHA 才拆出来的），建议把注释里的 `num_heads` 去掉，避免读者困惑。
- **MLA 部分仍是 `TODO`**：如果要作为完整教材，这一节需要补全（DeepSeek-V2 的低秩 KV 压缩 + 解耦 RoPE 部分是 MLA 的关键，容易讲错，值得单独花篇幅）。

---

### 小结

最需要修的是这几处：

1. **SDPA 出处**（最严重的史实错误：应归于 2017 年 Transformer，而非 2014 年 Bahdanau 论文）；
2. **GQA `split_head` 的 `.view` → `.reshape`**（必崩）；
3. **`test_MHA_with_cache` 漏了 `mha = MultiHeadAttention(...)`**（必崩 `NameError`）；
4. Prefill/KV Cache 表述、MQA 模型列表（Gemini 存疑）、GQA 公式变量补全这几处概念性瑕疵。

把以上几点改掉后，这篇博客作为入门教材就比较扎实了。需要的话，我可以帮你把整篇（含 MLA 一节）重写成一份可直接运行、注释统一的完整版本。

Reference:
https://hwcoder.top/Manual-Coding-1


下面是扩写后的版本。我在保留专业性的基础上，补充了每种机制的设计动机，并用一条主线——**KV Cache 的显存与带宽瓶颈**——把这条演进脉络串了起来，让你能看出"为什么"会这样一步步发展。

---

### 演进主线：从"如何算注意力"到"如何让推理负担得起注意力"

理解这条发展线，关键是抓住两个不同阶段的核心矛盾：

- **2017 年之前到 2017 年**：要解决的是"**怎样让模型高效地建模长程依赖**"。这一阶段诞生了注意力机制本身，以及 Transformer 的 Q/K/V 框架和多头设计。
- **2019 年至今**：Transformer 已成事实标准，主要矛盾转移到了"**自回归推理时如何降低 KV Cache 的显存占用与访存带宽压力**"。MQA、GQA、MLA 都是为解决这一矛盾而生，它们并不改变注意力的数学本质，而是在 **K/V 的"存储与共享方式"上做文章**。

把这两个阶段分开看，后面四种机制的逻辑就非常清楚了。

---

### 1. 缩放点积注意力（SDPA）：注意力的最小计算单元

注意力计算的最小单元，由 2017 年《Attention Is All You Need》提出，在点积注意力基础上加入 $\frac{1}{\sqrt{d_k}}$ 缩放因子，其完整形式为：

$$\text{Attention}(Q,K,V)=\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

**为什么要除以 $\sqrt{d_k}$？** 当维度 $d_k$ 较大时，$Q$ 与 $K$ 的点积结果方差会随 $d_k$ 线性增大（若各分量独立、均值为 0、方差为 1，则点积方差约为 $d_k$）。点积数值过大会把 softmax 推向饱和区，使其输出接近 one-hot，导致梯度极小、训练难以推进。除以 $\sqrt{d_k}$ 正是为了把点积方差重新归一到 1 附近，稳定梯度。

**注意力的更早源头。** 注意力机制本身最早可追溯到 2014 年 Bahdanau 等人在神经机器翻译中提出的**加性注意力**（用一个小型前馈网络计算对齐分数），其动机是缓解 seq2seq 中"把整句压进一个固定向量"的信息瓶颈。但真正奠定今天范式的——**显式的 Q/K/V 抽象**与**缩放点积**这种可高度并行化的打分形式——均来自 Transformer。点积形式相比加性形式的最大优势在于可直接用高度优化的矩阵乘法实现，在 GPU 上吞吐远高于加性注意力，这也是它能成为主流的关键工程原因。

---

### 2. 多头注意力（MHA）：把表示空间拆成多个子空间

2017 年开山之作《Attention Is All You Need》提出，是当前主流 LLM 的基石。其做法是把 $Q$、$K$、$V$ 分别线性投影到 $h$ 个低维子空间，在每个子空间独立做一次 SDPA，再把 $h$ 个头的输出拼接并线性变换。**每个头都有自己独立的 Q、K、V 投影。**

**为什么要"多头"而不是"一个大头"？** 单个注意力头在做 softmax 加权时，本质上倾向于聚焦到少数位置上，表达能力受限。多头允许模型在不同子空间中并行关注不同类型的关系——例如有的头捕捉语法依赖、有的头捕捉指代或长程语义关联——相当于"集成"了多组互补的注意力模式，而且由于每个头维度更小（$d_k=d_{\text{model}}/h$），总计算量与单个全维注意力基本持平。

正是 MHA 这种"每个头各存一份 K/V"的设计，埋下了后续所有优化的伏笔：**在自回归解码时，所有头、所有层、所有历史 token 的 K/V 都必须缓存下来**，KV Cache 随序列长度和头数线性膨胀，成为长上下文推理的主要显存与带宽瓶颈。

---

### 3. 多查询注意力（MQA）：让所有头共享同一份 K/V

出自 2019 年《Fast Transformer Decoding: One Write-Head is All You Need》。其核心改动是：**所有头共享同一份 K、V，只保留各自独立的 Q**，从而把 KV Cache 压缩到原来的 $\frac{1}{h}$。

**为什么这样能行、又为什么需要它？** 论文的出发点是一个关键观察：自回归解码时性能往往**不受算力限制，而受访存带宽限制**——每生成一个 token 都要把全部 KV Cache 从显存读进计算单元，参数和缓存的"搬运"成本主导了延迟。既然瓶颈在 K/V 的读取量，那么让所有头共用一份 K/V 就能成倍减少需要搬运和存储的数据，显著加快解码、增大可容纳的 batch 与上下文长度。代价是表达能力略有损失（K/V 不再各头独立），在部分任务上有小幅质量下降。代表模型有 PaLM、StarCoder、Falcon 等。

---

### 4. 分组查询注意力（GQA）：在质量与效率之间取折中

出自 2023 年《GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints》，是 MHA 与 MQA 之间的**可调折中**。设头数为 $h$、组数为 $g$（要求 $g$ 整除 $h$），让每 $\frac{h}{g}$ 个查询头共享一组 K/V：

- $g=h$ 时，每头各有一份 K/V，退化为 **MHA**；
- $g=1$ 时，全部头共享一份 K/V，退化为 **MQA**；
- $1<g<h$ 时，KV Cache 压缩到 $\frac{g}{h}$。

**为什么需要它？** MQA 虽快，但把 $h$ 份 K/V 压成 1 份过于激进，在大模型上质量损失明显且训练不稳定。GQA 引入"组"这个旋钮，让我们能在"接近 MHA 的质量"和"接近 MQA 的速度"之间平滑取舍。论文还提出一个很实用的工程贡献：**可以从已训练好的 MHA 检查点出发，通过对每组内的 K/V 投影做均值池化来初始化，再用极少量算力继续训练（uptraining）**，无需从零重训就能得到 GQA 模型。这种"低成本改造既有模型"的特性，使它迅速成为业界默认选择。代表模型有 Llama-2-70B（$g=8$）与 Llama-3 全系列。

---

### 5. 多头隐注意力（MLA）：用低秩压缩换取更小的缓存

出自 2024 年 DeepSeek-V2 技术报告。它换了一个思路：不再靠"让多个头共享 K/V"来省缓存，而是**用低秩线性变换把 K、V 联合压缩到一个低维潜向量（latent vector）**，推理时只缓存这个潜向量，使用时再通过上投影矩阵恢复出各头所需的 K/V。

**为什么这是又一次进步？** GQA/MQA 是通过"减少 K/V 的份数"来省缓存，但这必然以牺牲头间多样性为代价。MLA 的目标是**在把 KV Cache 压得比 GQA 更小的同时，尽量保住接近 MHA 的表达能力**——因为各头仍可从同一潜向量解出不同的 K/V，而非简单共享同一份。此外，得益于矩阵吸收等技巧，上投影可以被合并进其他权重，从而避免显式重建完整 K/V 的额外开销。这使得 MLA 在极小缓存占用下仍保持较强性能，成为 DeepSeek 系列长上下文与高效推理的关键设计。

---

### 一句话总结这条脉络

| 机制 | 年份 | 核心动机 | K/V 处理方式 | KV Cache 相对量 |
|---|---|---|---|---|
| SDPA | 2017 | 稳定梯度、可并行打分 | —（单次计算） | — |
| MHA | 2017 | 多子空间建模、表达力 | 每头独立 K/V | $1$（基准） |
| MQA | 2019 | 缓解解码访存带宽瓶颈 | 全部头共享一份 K/V | $\frac{1}{h}$ |
| GQA | 2023 | 质量与效率可调折中 | 分组共享 K/V | $\frac{g}{h}$ |
| MLA | 2024 | 小缓存 + 保表达力 | 低秩潜向量压缩 | 进一步压缩 |

可以看到：**SDPA 与 MHA 解决"怎么算得好"，而 MQA → GQA → MLA 这条线，本质上都是在解决同一个工程难题——如何在自回归推理中既压住 KV Cache、又尽量不损失模型质量**，只是手段从"粗暴共享"逐步演进到"分组折中"再到"低秩压缩"，越来越精细。


# Special Thank

本文中的代码验证包含 Claude Opus 4.8 的参与。
