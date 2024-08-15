---
layout: blogpage
title: When Should We Use Counter-fitting Word Vector
comments: true
tags: Note
---

简单整理一下这篇文章：

## 主要内容和贡献

- 反向拟合方法：提出了一种新颖的反向拟合方法，用于将反义词和同义词的约束注入到词向量空间中，以改善词向量在语义相似性判断上的能力。
- SimLex-999数据集表现：将此方法应用于公开可用的预训练词向量后，在SimLex-999数据集上达到了新的最佳性能。
- 对话状态跟踪：展示了如何使用该方法为下游任务定制词向量空间，例如对话状态跟踪（DST），并在不同对话领域中实现了稳健的改进。
- 开源工具和词向量：作者提供了工具和经过反向拟合的词向量，以便其他研究者可以利用和进一步研究。

 can inject knowledge of dialogue domain ontologies into word vector space representations to facilitate the construction of semantic dictionaries which improve DST performance across two different dialogue domains. 
反向拟合⽅法可以将对话领域本体的知识注⼊到词向量空间表示中，以促进语义词典的构建，从⽽提⾼跨两个不同对话领域的 DST 性能（dialogue state tracking task）。

## 同类工作







## 潜在改进点
