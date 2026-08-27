---
layout: modern-article
title: "Explore and Exercise: Improve LLM Performance by RAG"
comments: true
tags: RAG
---

提高 RAG 效果的几种方式：

# 1. 索引阶段

- 语义切片（Semantic Chunking）：利用句子相似度断句，来保持语义完整。
- 父子切片（Parent-Child / Hierarchical Chunking）：使用小切片（Child）进行精准相似度检索，但在喂给 LLM 时加载其所属的更大上下文切片（Parent）。
- 句子窗口检索（Sentence-Window Retrieval）：以单个句子建立索引，检索到后前后各扩展

# 2. 查询阶段

- HyDE - Hypothetical Document Embeddings）：
先让 LLM 生成一段假设的答案，再用该假设答案向量去检索真实文档（利用“答案搜答案”弥合“问题与答案”的语义差距）。
- 重写+多查询：从不同角度重写用户 Query，对复杂问题则分解，总之分别检索后再合并去重

# 3. 检索阶段

- 混合检索（Hybrid Search）：稠密检索（Dense Retrieval，基于向量语义） 和 稀疏检索（Sparse Retrieval，如 BM25 关键词匹配），
通过 RRF（Reciprocal Rank Fusion）等算法加权融合，弥补向量检索在专有名词、编号、型号匹配上的短板。
- metadata 前置过滤
- Reranking：使用 Cross-Encoder 架构的 Rerank 模型（如 BGE-Reranker、Cohere Rerank）对召回的前 K 个文档进行深层交叉注意力打分，
重排后截取前 N 个（N≪K)

*有必要的话还可以做后处理，使用 LLMLingua 等提示词压缩工具，或利用小模型做抽取式摘要，去除冗余停用词与无关上下文。

# 4. 生成阶段

- prompt 优化
- 针对性调模型：比如 RAFT（Retrieval Augmented Fine-Tuning），训练模型学会从混杂着干扰文档的上下文中选正确答案。