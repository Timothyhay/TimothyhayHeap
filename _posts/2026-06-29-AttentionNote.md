---
layout: modern-article
title: A Casual Explanation of Common Attention Mechanisms
tags: LLM
comments: true
---
俗话说得好， Resnet 就是 LSTM 旋转了 90 度，从时序变成了深度； transformer 就是 attention + resnet，大道至简这一块。

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
L(C)==A C^{-\alpha}+B
$$

其中 (C) 是计算预算。这意味着模型性能不会突然饱和，而是持续改善，只是边际收益越来越低。这一发现极大改变了整个行业的发展方向，因为它说明堆算力、堆数据、堆参数并非盲目行为，而是一种有理论依据的工程路线。

后来 DeepMind 的 Chinchilla 工作进一步发现，很多模型实际上处于“参数过多、数据不足”的状态。对于固定计算预算而言，适当减小模型规模、增加训练 token 数量，反而可以获得更好的结果。这也是为什么现代模型越来越重视数据量，许多模型的预训练 token 已经达到十万亿级别。

如果把整个 LLM 训练流程放在一起理解，我认为最准确的认知框架是：预训练阶段负责建立压缩后的世界模型，使模型能够理解语言、知识和现实世界中的统计结构；SFT 阶段负责把这个世界模型转换为符合人类期望的行为模式；而 DPO、RLHF、GRPO、RFT 等后训练阶段，则是在行为空间中进一步优化偏好、推理和决策能力。

因此，对于真正参与训练的人来说，预训练最值得深入理解的其实不是“预测下一个 token”这句话本身，而是三个更深层的问题：第一，为什么仅靠预测 token 就能形成世界模型；第二，数据分布如何决定最终能力边界；第三，梯度下降为什么会在超大规模参数空间中自发形成抽象表示。这三个问题至今仍然是大模型研究最核心、也最有趣的部分。

# 实现

这里重在讨论：

- **多头注意力（MHA）**：2017 年开山之作《Attention Is All You Need》提出，是当前主流 LLM 的基石。每个头都有自己独立的 Q、K、V。

> - **缩放点积注意力（SDPA）**：注意力计算的最小单元，在 2017 年《Attention Is All You Need》提出，在点积注意力基础上加入 $\frac{1}{\sqrt{d_k}}$ 缩放，缓解 softmax 梯度消失的问题。注意力机制本身最早可追溯到 2014 年 Bahdanau 等人的**加性注意力**，但 Q/K/V 框架与缩放点积形式均来自 Transformer。它是多头注意力（MHA）的内部基本组件。

- **多查询注意力（MQA）**：出自 2019 年《Fast Transformer Decoding: One Write-Head is All You Need》。所有头共享同一份 K、V，只保留各自独立的 Q，从而大幅压缩 KV Cache。代表模型有 PaLM、StarCoder、Falcon 等。
- **分组查询注意力（GQA）**：出自 2023 年《GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints》，是 MHA 与 MQA 的折中。设头数为 $h$、组数为 $g$（$g$ 整除 $h$）：$g=h$ 时退化为 MHA，$g=1$ 时退化为 MQA，$1<g<h$ 时 KV Cache 压缩到 $\frac{g}{h}$。代表模型有 Llama-2-70B（$g=8$）与 Llama-3 全系列。
- **多头隐注意力（MLA）**：出自 2024 年 DeepSeek-V2 技术报告，用低秩线性变换压缩 K、V，只缓存一个低维潜向量，进一步压缩 KV Cache。

> **关于 KV Cache**：在自回归生成中，新预测的第 $t$ 个 token 不会改变已经算好的前 $t-1$ 个位置的 K、V，因此可以把它们缓存下来，避免重复计算。Prefill（预填充）阶段会一次性算出 prompt 全部 token 的 K、V 并写入缓存，KV Cache 的加速收益主要体现在逐 token 的 decode 阶段。

> 缩放因子写作 `head_dim ** 0.5`、掩码统一用 `masked_fill`（约定 `True` 表示被遮蔽）、注释和形状标注统一。

---

## 一、缩放点积注意力（SDPA）

Scaled Dot-Product Attention 是 MHA 的内部基本组件：先计算 query 与每个 key 的关联度作为权重，再对 value 加权求和。

给定输入 $X \in \mathbb{R}^{n \times d}$ （n=序列长度/token数，d=模型隐藏层维度/embedding维度），做三次线性投影：$Q = XW_Q,\; K = XW_K,\; V = XW_V$，然后

$$
\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V
$$

计算顺序：Given an input matrix $X$,
we first linearly project it into $Q$, $K$, $V$ representations using learnable projection matrices;
next, we compute the raw attention logits by taking the dot product $QK^T$(Q K transpose),
scale them by $\sqrt{d_k}$(The square root of d sub k) to prevent gradient vanishing caused by softmax saturation,
and optionally plus a causal or padding mask with $-\infty$;
finally, we apply the Softmax function row-wise along the sequence dimension to obtain a normalized attention probability matrix,
and multiply by the Value matrix $V$ to produce the final output.

其中 $M$ 是掩码（因果掩码为上三角 $-\infty$）。

直觉：每个 query 用点积衡量与所有 key 的相关性，softmax 归一化为权重，再对 value 加权求和。

```
Input X
      │
      ├─── W_Q ───> Q (Query) ──┐
      ├─── W_K ───> K (Key)   ──┴─> MatMul (Q·K^T) ──> Scale (/√d_k) ──> [Mask] ──> Softmax ──┐
      └─── W_V ───> V (Value) ─────────────────────────────────────────────────────────────┴─> MatMul (Score·V) ──> Output
```

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

为什么除以 $\sqrt{d_k}$：防止点积数值过大导致 Softmax 函数进入梯度饱和区（梯度消失），从而保证训练过程的稳定性。

若 $q, k$ 各分量独立、均值 0、方差 1，则 $q\cdot k = \sum_i q_i k_i$ 的方差为 $d_k$。$d_k$ 大时，点积量级很大，softmax 进入饱和区（近似 one-hot），梯度趋近于 0。除以 $\sqrt{d_k}$ 把方差拉回 1。


注意力的更早源头。 注意力机制本身最早可追溯到 2014 年 Bahdanau 等人在神经机器翻译中提出的加性注意力（用一个小型前馈网络计算对齐分数），其动机是缓解 seq2seq 中"把整句压进一个固定向量"的信息瓶颈。但真正奠定今天范式的——显式的 Q/K/V 抽象与缩放点积这种可高度并行化的打分形式——均来自 Transformer。点积形式相比加性形式的最大优势在于可直接用高度优化的矩阵乘法实现，在 GPU 上吞吐远高于加性注意力，这也是它能成为主流的关键工程原因。

---

## 二、多头注意力（MHA）

**为什么要"多头"而不是"一个大头"？** 单个注意力头在做 softmax 加权时，本质上倾向于聚焦到少数位置上，表达能力受限。多头允许模型在不同子空间中并行关注不同类型的关系——例如有的头捕捉语法依赖、有的头捕捉指代或长程语义关联——相当于"集成"了多组互补的注意力模式，而且由于每个头维度更小（$d_k=d_{\text{model}}/h$），总计算量与单个全维注意力基本持平。

正是 MHA 这种"每个头各存一份 K/V"的设计，埋下了后续所有优化的伏笔：**在自回归解码时，所有头、所有层、所有历史 token 的 K/V 都必须缓存下来**，KV Cache 随序列长度和头数线性膨胀，成为长上下文推理的主要显存与带宽瓶颈。

「多头」指并行计算多组注意力，每个头从不同的子空间「视角」捕获信息。设输入为 $X$（一批 token 向量），$W^Q,W^K,W^V\in\mathbb{R}^{d_{model}\times d_{model}}$，将输出切分为 $h$ 个头、每个头维度 $d_k=d_{model}/h$。计算步骤如下：

相比于只做一次点积，MHA 将 $Q, K, V$ 投影到 $h$ 个不同的低维子空间，分别独立计算注意力（即执行 $h$ 次 SDPA），最后将所有头的输出拼接（Concat）并通过一个线性层融合。
* 目的是让模型在不同的位置同时关注来自不同子空间（语义、语法、相对距离等）的信息。

**1）线性投影得到 Q、K、V：**

$$
Q = XW^Q,\qquad K = XW^K,\qquad V = XW^V
$$

**2）计算第 $i$ 个头的缩放点积得分（$d_k$ 为每个头的维度）：**

$$
\text{scores}_i=\frac{Q_iK_i^{\top}}{\sqrt{d_k}}
$$

**3）用 Softmax 得到注意力权重：**

$$
A_i=\mathrm{softmax}\!\left(\frac{Q_iK_i^{\top}}{\sqrt{d_k}}\right)
$$

**4）用注意力权重对 $V$ 加权求和，得到每个头的输出：**

$$
\text{head}_i = A_iV_i=\mathrm{softmax}\!\left(\frac{Q_iK_i^{\top}}{\sqrt{d_k}}\right)V_i
$$

**5）拼接所有头的输出，再乘以输出投影矩阵 $W^O$，得到最终结果：**

$$
\mathrm{MultiHead}(X)=\mathrm{Concat}(\text{head}_1,\dots,\text{head}_h)\,W^O
$$

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

KV Cache 主要用于 Decoder 在逐 token 生成（decode）阶段减少重复计算：Prefill 阶段会一次性算出 prompt 所有 token 的 K、V 并写入缓存，decode 阶段每生成一个新 token，只需计算它自己的 Q、K、V，再与缓存中的历史 K、V 拼接即可。本代码仅作示例。

实际上：

- **Prefill 阶段会一次性计算 prompt 所有 token 的 K、V，并把它们写入 KV Cache**，供后续 decode 复用。所以并不是「prefill 不需要 KV Cache」，而是「KV Cache 的*加速收益*主要体现在逐 token 的 decode 阶段」。
- **P/D 分离（Prefill/Decode disaggregation）** 是一种把预填充和解码放到不同实例/硬件上执行的部署优化，和「prefill 是否需要 KV Cache」是两回事。


逐 token 解码：每一步只输入一个新 token，与缓存的历史 K、V 拼接后做注意力。
由于新 query 天然只能看到自己和历史，所以无需再额外加因果掩码。

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
    '''
    用全量 KV Cache 做 decode 时，单个 query 本来就可以看到所有已缓存的历史 key，`current_causal_mask` 实际上全是 `False`、不起任何作用，这里保留只是为了和接口对齐，逻辑上没问题。
    '''
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


if __name__ == "__main__":
    test_MHA_with_cache()
```

---

## 四、多查询注意力（MQA）

所有头共享同一份 K、V（投影维度仅为 `head_dim`），Q 仍按头独立。计算时让单头的 K、V 通过广播与多头 Q 相乘。
公开确认使用 MQA 的代表是 **Falcon（如 Falcon-40B）**。


出自 2019 年《Fast Transformer Decoding: One Write-Head is All You Need》。其核心改动是：**所有头共享同一份 K、V，只保留各自独立的 Q**，从而把 KV Cache 压缩到原来的 $\frac{1}{h}$。

**为什么这样能行、又为什么需要它？** 论文的出发点是一个关键观察：自回归解码时性能往往**不受算力限制，而受访存带宽限制**——每生成一个 token 都要把全部 KV Cache 从显存读进计算单元，参数和缓存的"搬运"成本主导了延迟。既然瓶颈在 K/V 的读取量，那么让所有头共用一份 K/V 就能成倍减少需要搬运和存储的数据，显著加快解码、增大可容纳的 batch 与上下文长度。代价是表达能力略有损失（K/V 不再各头独立），在部分任务上有小幅质量下降。


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

将所有 Head $h$ 分为 $g$ 个组（要求 $g$ 整除 $h$），每组共享同一对 K、V（组内的多个 query 头复用同一份 K、V。）。当 $g=h$ 时退化为 MHA，$g=1$ 时退化为 MQA；当 $1<g<h$ 时，KV Cache 压缩到 MHA 的 $\frac{g}{h}$，压缩率不如 MQA 的 $\frac{1}{h}$，但保留了更大的自由度，效果更有保证。在 Llama-2/3-70B 中 $g=8$。

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

出自 2024 年 DeepSeek-V2 技术报告。它换了一个思路：不再靠"让多个头共享 K/V"来省缓存，而是**用低秩线性变换把 K、V 联合压缩到一个低维潜向量（latent vector）**，推理时只缓存这个潜向量，使用时再通过上投影矩阵恢复出各头所需的 K/V。

**为什么这是又一次进步？** GQA/MQA 是通过"减少 K/V 的份数"来省缓存，但这必然以牺牲头间多样性为代价。MLA 的目标是**在把 KV Cache 压得比 GQA 更小的同时，尽量保住接近 MHA 的表达能力**——因为各头仍可从同一潜向量解出不同的 K/V，而非简单共享同一份。此外，得益于矩阵吸收等技巧，上投影可以被合并进其他权重，从而避免显式重建完整 K/V 的额外开销。这使得 MLA 在极小缓存占用下仍保持较强性能，成为 DeepSeek 系列长上下文与高效推理的关键设计。

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

- **掩码约定**：`causal_mask`/`padding_mask` 中 `True` 表示「需要被遮蔽」的位置，用 `masked_fill(mask, -inf)` 屏蔽；`causal_mask = torch.triu(ones, diagonal=1).bool()` 即屏蔽未来位置。有关掩码的写法，`attention_scores += mask * -1e9` 在混合精度（fp16/bf16）下，`-1e9` 可能溢出/精度不佳，且 `+=` 原地操作偶尔会干扰 autograd。教材里更推荐 `attention_scores = attention_scores.masked_fill(mask, float('-inf'))`，语义也更清晰。
- **合并多头**：`transpose` 之后张量内存非连续，统一用 `reshape`（等价于 `.contiguous().view(...)`），避免 `view` 报错。
- **MLA 简化**：为聚焦核心思想，这里省略了 DeepSeek-V2 实际使用的 YaRN/`mscale` 缩放修正与「矩阵吸收（absorb）」等推理加速技巧；`qk_rope_head_dim` 需为偶数以适配 RoPE。


### 脉络


| 机制 | 年份 | 核心动机             | K/V 处理方式       | KV Cache 相对量 |
| ---- | ---- | -------------------- | ------------------ | --------------- |
| SDPA | 2017 | 稳定梯度、可并行打分 | —（单次计算）     | —              |
| MHA  | 2017 | 多子空间建模、表达力 | 每头独立 K/V       | $1$（基准）     |
| MQA  | 2019 | 缓解解码访存带宽瓶颈 | 全部头共享一份 K/V | $\frac{1}{h}$   |
| GQA  | 2023 | 质量与效率可调折中   | 分组共享 K/V       | $\frac{g}{h}$   |
| MLA  | 2024 | 小缓存 + 保表达力    | 低秩潜向量压缩     | 进一步压缩      |

可以看到：**SDPA 与 MHA 解决"怎么算得好"，而 MQA → GQA → MLA 这条线，本质上都是在解决同一个工程难题——如何在自回归推理中既压住 KV Cache、又尽量不损失模型质量**，只是手段从"粗暴共享"逐步演进到"分组折中"再到"低秩压缩"，越来越精细。


# 交叉注意力

在传统的 Self-attention（自注意力） 中，Query、Key 和 Value 均来自于同一个序列/模态。
它的作用是让序列内的元素互相交互（例如一句话中的主语去注意动词）。

而 Cross-attention（交叉注意力） 的核心区别在于：

Q 来自一个序列（或模态）X - Encoder-Decoder 架构中，Q 来自 decoder；

K 和 V 来自另一个序列（或模态）Y - Encoder-Decoder 架构中，K/V 来自 encoder。

Cross-attention 的本质是：让一个序列去查询并“吸收”另一个序列的信息。

Reranker 语境中的"cross-attention"：通常指 cross-encoder——把 query 和 document 拼成一个序列 [CLS] q [SEP] d [SEP]，做全量自注意力，query 的每个 token 都能和 document 的每个 token 交互。这是"early interaction"。


假设输入序列 $X \in \mathbb{R}^{N \times d_x}$（目标序列，长度为 $N$）和 $Y \in \mathbb{R}^{M \times d_y}$（上下文/源序列，长度为 $M$）：

1. **线性映射**：
   $$Q = X W_Q \quad (Q \in \mathbb{R}^{N \times d_k})$$
   $$K = Y W_K \quad (K \in \mathbb{R}^{M \times d_k})$$
   $$V = Y W_V \quad (V \in \mathbb{R}^{M \times d_v})$$
   *(注意：$W_Q$ 映射自 $X$，而 $W_K, W_V$ 映射自 $Y$；$X$ 和 $Y$ 的序列长度 $N$ 和 $M$ 可以完全不相等，但映射后的 $Q$ 和 $K$ 维度必须匹配)*

2. **计算注意力权重并加权**：
   $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
    * 输出的维度是 $N \times d_v$，其形状只与 Query 的序列长度 $N$ 相关，而与 Key/Value 的长度 $M$ 无关。

典型场景是机器翻译（原始 Transformer 的 Encoder-Decoder）、文本生成图像（如 Stable Diffusion）、多模态感知（如 BLIP-2、Flamingo、Perceiver）。


### Cross-attention vs. MHA vs. SDPA：核心区别

Cross-attention 和其他注意力也不是一个维度的互斥概念。

1. **Cross-attention vs. SDPA**：
    * SDPA 是**算法实现/内核**，Cross-attention 是**应用模式**。
    * Cross-attention 的底层计算，不管是单头还是多头，核心都依赖 SDPA 算子来计算注意力分布。

2. **Cross-attention vs. MHA**：
    * MHA **既可以是 Self-attention，也可以是 Cross-attention**。
    * 如果传入 MHA 的是同一个输入 `MHA(x, x, x)`，它就是 **Multi-Head Self-attention**；
    * 如果传入 MHA 的是不同输入 `MHA(x, y, y)`，它就是 **Multi-Head Cross-attention**。
    * 在现代神经网络中，**Cross-attention 几乎百分之百都是以 Multi-Head（MHA）的形式来实现的**。

---

# KVCache 相关


### 一、 核心原理与运行机制

#### 考点 1：为什么只需要 Cache $K$ 和 $V$，而不需要 Cache $Q$？
* **自回归与因果掩码（Causal Mask）**：在生成第 $t$ 个 Token 时，只有当前第 $t$ 个 Token 作为查询向量 $q_t$，去和历史所有 Token 的 $k_1, \dots, k_t$ 计算注意力权重，
并加权求和 $v_1, \dots, v_t$。
* **历史 $Q$ 已经完成使命**：过去生成的 $q_{<t}$ 不会再参与未来任何 Token 的注意力计算；
但过去的 $k_{<t}$ 和 $v_{<t}$ 在未来的每一轮 Decode 都要被重复使用。
因此**缓存 $K, V$ 可以避免 $O(N^2)$ 的重复投影计算，将单步解码复杂度降为 $O(N)$**。

#### 考点 2：Prefill 与 Decode 阶段在计算/访存特征上的本质差异
| 阶段 | 输入/输出特征 | 瓶颈类型 | 矩阵运算类型 |
| :--- | :--- | :--- | :--- |
| **Prefill（预填充/Prompt）** | 输入所有 Prompt Token（长序列），并行计算 | **计算密集型（Compute-bound）** | GEMM（通用矩阵乘法，高算力利用率） |
| **Decode（自回归解码）** | 每次只输入 1 个 Token（Seq_len=1） | **访存/带宽密集型（Memory-bound）** | GEMV（矩阵-向量乘法，受显存带宽限制） |

* **延伸追问**：为什么 Decode 阶段 GPU 算力利用率（MFU）通常很低？
  * *答*：因为每次计算只处理一个 Token，但每个 Layer 都要从 HBM（显存）中把庞大的历史 KV Cache 全量加载到 SRAM 中一次，算力被内存带宽卡死（Arithmetic Intensity 极低）。

---

### 二、 KV Cache 显存占用计算（高频手撕推导）

#### 考点 3：单请求 KV Cache 显存公式
$$\text{Memory (Bytes)} = 2 \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times \text{seq\_len} \times \text{bytes\_per\_element} \times \text{batch\_size}$$

* **各项含义**：
  * **$2$**：分别存 $K$ 和 $V$；
  * **$n_{\text{layers}}$**：Transformer 层数；
  * **$n_{\text{kv\_heads}}$**：KV 头数（注意不是 Q 头数！）；
  * **$d_{\text{head}}$**：每个 Head 的隐层维度（通常为 $\frac{d_{\text{model}}}{n_{\text{q\_heads}}}$）；
  * **$\text{bytes\_per\_element}$**：精度（FP16/BF16 为 2 字节，FP8 为 1 字节，INT4 为 0.5 字节）。

> **面试现场速算题举例**：
> LLaMA-3-8B（32 层, GQA 8 个 KV heads, $d_{\text{head}}=128$, BF16 精度），在上下文长度为 8K、Batch Size=1 时，KV Cache 占用多少显存？
> $$\text{Size} = 2 \times 32 \times 8 \times 128 \times 8192 \times 2 \text{ Bytes} \approx 1,073,741,824 \text{ Bytes} = \mathbf{1\text{ GB}}$$

---

### 三、 旨在减少 KV Cache 的注意力架构演进（重中之重）

面试官最喜欢问：“为了减少 KV Cache 的显存瓶颈，模型结构经历了哪些演进？”

```
MHA (1:1)  ──►  MQA (N:1)  ──►  GQA (N:G)  ──►  MLA (低秩压缩)
```

1. **MHA (Multi-Head Attention)**：$Q, K, V$ 头数相同（$n_Q = n_{KV}$），KV Cache 显存开销极大。
2. **MQA (Multi-Query Attention)**：所有 $Q$ 头共享同一组 $K$ 和 $V$（$n_{KV} = 1$）。KV Cache 减少为原来的 $\frac{1}{n_Q}$，极大提升推理吞吐，但模型容量和表达能力有所下降。
3. **GQA (Grouped-Query Attention)**：折中方案，将 $Q$ 头分组，每组共享一个 KV 头（如 LLaMA-2/3、Mistral、Qwen）。通常 8 个 $Q$ 共享 1 个 $KV$，显存压缩到 $\frac{1}{8}$ 且几乎不损耗模型精度。
4. **MLA (Multi-head Latent Attention，DeepSeek-V2/V3 核心)**：
   * **原理**：将 $K$ 和 $V$ 联合通过低秩矩阵压缩为一个低维的隐向量 $c_t^{KV}$，只缓存这个隐向量（外加一个带解耦 RoPE 的 Key 向量 $k_t^R$）。
   * **效果**：将 KV Cache 的大小压缩到甚至低于 MQA 的级别，同时保留了强于 MHA 的模型表达能力。

---

### 四、 长上下文与 KV Cache 压缩/淘汰算法（算法层）

当上下文达到 32k/128k/1M 时，全量保留 KV Cache 不可承受，业内常见的算法方案：

#### 考点 4：Attention Sink 与 StreamingLLM
* **现象**：当去除最早几个 Token 的 KV Cache 时，模型的注意力会剧烈崩溃。
* **原因**：Softmax 计算具有“无惩罚吸收多余注意力分值”的特性，模型倾向于将首部无意义的几个 Token（如 `<s>`）作为**注意力汇聚点（Attention Sink）**。
* **方案（StreamingLLM）**：**固定保留前几个 Sink Tokens + 滑动窗口局域 Tokens**，即可实现近乎无限长序列的稳定流式推理。

#### 考点 5：基于重要性的稀疏与剪枝（H2O, SnapKV, PyramidKV）
* **Heavy-Hitter Oracle (H2O)**：根据累积的注意力权重打分，保留得分最高的少数“重击者（Heavy-Hitter）”Token 和最近的局部 Token，其余丢弃。
* **KV 量化（KV Cache Quantization）**：
  * 将 KV Cache 转为 FP8、INT8 或 INT4 存储。
  * *考点细节*：Key 向量通常存在特定的“异常通道（Outlier Channels）”，直接按张量量化容易精度崩塌，通常需要 Per-Channel 量化或结合旋转变换（如 QuaRot）。

---

### 五、 显存管理与系统级工程考点（工程框架层）

#### 考点 6：PagedAttention（vLLM 的核心突破）
* **解决的痛点**：传统框架连续预分配显存导致的**内部内存碎片**（为最大 seq_len 预分配空间却没用完）和**外部内存碎片**。
* **核心思路**：借鉴操作系统虚拟内存分页机制。将 KV Cache 划分为固定大小的 **Block（如 16 或 32 个 Token）**，逻辑上连续，物理上通过 Block Table 离散存储，实现显存的按需动态分配与零浪费。

#### 考点 7：Prompt Caching / RadixAttention（SGLang 等）
* **核心思想**：在多轮对话、Few-shot 任务、多 Agent 场景中，大量 Prompt 前缀是完全重复的。
* **实现**：用基数树（Radix Tree）维护和缓存共享前缀的 KV Cache，避免重复计算 Prefill。

---

### 💡 面试答题套路总结（如何拿高分）

当面试官问到 KV Cache 时，推荐按照**“痛点 $\to$ 瓶颈本质 $\to$ 架构演进 $\to$ 系统优化”**的主线来回答：

> 1. **定义与动机**：自回归因果特性决定了只需缓存 $K,V$，避免 $O(N^2)$ 重复计算；
> 2. **指出痛点**：Decode 阶段受限于显存带宽（Memory-bound），且显存开销随序列线性暴涨；
> 3. **算法/架构解法**：模型端从 **MHA $\to$ GQA $\to$ MLA** 压缩维度，算法端用 **StreamingLLM / H2O / 量化** 淘汰稀疏 Token；
> 4. **系统工程落地**：底层借助 **PagedAttention** 解决内存碎片，利用 **Prompt Caching** 实现跨请求复用。


# Special Thank

本文中的代码验证包含 Claude Opus 4.8 的参与。
