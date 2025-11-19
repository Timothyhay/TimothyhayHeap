---
layout: blogpage
title: Retrieval and Reranking for Code Snippet
comments: true
tags: Deep-Learning
---

I need a plan for effective code retrieval to extract the most relevant part of code file(s) before send it to LLM - today we call it **Context Management** (code as most the context window)

For most of the reranking task, a 2-step retrieval is a usual combo:
1. Embedding model roughly sort 
第一阶段 bi-encoder（快速粗排）：unixcoder-base、grapecodebert-base、bge-m3、nv-embed-v2 等代码专用 embedding。
2. Reranker model finely sort  
第二阶段 reranker（精排 Top 50→Top 10）： cross-encoder
对海量文档的处理往往需要先使用Bi-Encoder快速召回Top-K（like Top 100），然后再用一个Cross-Encoder（like BGE）对着K个文档精排。

jina-embeddings-v2-base-code is an multilingual embedding model speaks English and 30 widely used programming languages. Same as other jina-embeddings-v2 series, it supports 8192 sequence length.

CodeT5+ is a new family of open code large language models with an encoder-decoder architecture that can flexibly operate in different modes (i.e. encoder-only, decoder-only, and encoder-decoder) to support a wide range of code understanding and generation tasks. It is introduced in the paper:

CodeT5+: Open Code Large Language Models for Code Understanding and Generation by Yue Wang*, Hung Le*, Akhilesh Deepak Gotmare, Nghi D.Q. Bui, Junnan Li, Steven C.H. Hoi (* indicates equal contribution).

Compared to the original CodeT5 family (base: 220M, large: 770M), CodeT5+ is pretrained with a diverse set of pretraining tasks including span denoising, causal language modeling, contrastive learning, and text-code matching to learn rich representations from both unimodal code data and bimodal code-text data. Additionally, it employs a simple yet effective compute-efficient pretraining method to initialize the model components with frozen off-the-shelf LLMs such as CodeGen to efficiently scale up the model (i.e. 2B, 6B, 16B), and adopts a "shallow encoder and deep decoder" architecture. Furthermore, it is instruction-tuned to align with natural language instructions (see our InstructCodeT5+ 16B) following Code Alpaca.

** 使用没有微调过的模型作为Reranker **
以 CodeT5+ 为例：CodeT5+ 的重排序能力主要来自于其 "Text-Code Matching" (TCM) 预训练任务。在该模式下，模型不仅独立编码文本和代码，还通过 Decoder 的 Cross-Attention 机制深度融合两者的信息，判断它们是否匹配。

**选择模型：**
必须使用经过双模态（Bimodal）训练的checkpoint，例如 Salesforce/codet5p-220m-bimodal 或 Salesforce/codet5p-770m。纯 Encoder 模型（如 embedding 版本）通常用于粗排（向量检索），而 Reranker 需要 Decoder 参与。
**数据构造：**
Reranker 是 Cross-Encoder 模式。我们需要将 (Query, Code_Candidate) 拼成一对输入。
Encoder 输入：自然语言查询（Query）。
Decoder 输入：代码候选（Code Snippet）。
**获取匹配分数：**
 CodeT5+ 的匹配分数的计算逻辑如下：
- 将 Query 输入 Encoder。
- 将 Code 输入 Decoder。
- 取 Decoder 输出序列中 [EOS] Token 的 Hidden State。
- 将该向量通过一个二分类线性层（Projector），输出“匹配（Match）”和“不匹配（Mismatch）”的 Logits。取“匹配”类的 Logit 或概率作为相关性得分。

这里 Logit（未归一化的原始预测值）和 Probability（经过 Sigmoid/Softmax 归一化的概率值）或 Cosine Similarity 是两码事。Logit 的绝对值没有物理意义（不像概率代表置信度）。这就是所谓 Calibration 问题，Logit之间的相关关系并不线性对应于相关性，除非模型校准过。虽然 Logit 对排序（A > B）有效，但如果需要过滤低质量结果，Logit 很难确定截断点。以及 Logit 跨模型不可比（那是自然的）。

不过，如果在重排序阶段，我们只关心候选文档的相对顺序（谁比谁好），Logit 提供了最细粒度的区分度，避免了 Softmax 在高分段的挤压效应。这样也能用。

### LLM-based Reranker

CodeRankLLM is a 7B LLM fine-tuned for listwise code-reranking. When combined with performant code retrievers like CodeRankEmbed, it significantly enhances the quality of retrieved results for various code retrieval tasks.


## Reference

https://huggingface.co/nomic-ai/CodeRankEmbed

https://gangiswag.github.io/cornstack/