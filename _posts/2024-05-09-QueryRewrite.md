---
layout: blogpage
title: Query Rewrite in RAG
comments: true
tags: RAG
---

Review of query rewite/reformatting methods to increase retrieval accuracy during query stage.

查询重写的意义：将用户查询的语义空间与文档的语义空间对齐，使LLM提供更准确的回答。

## 借助 LLM 的方法

## 1. HyDE

假设文档嵌入（HyDE）方法：生成假设文档，以便最终查询向量v与向量空间中的实际文档尽可能紧密地对齐。

步骤：

1.使用LLM基于查询生成k个假设文档。这些生成的文件可能不是事实，也可能包含错误，但它们应该于相关文件相似。此步骤的目的是通过LLM解释用户的查询。

2.将生成的假设文档输入编码器，将其映射到密集向量f(dk)，编码器具有过滤功能，过滤掉假设文档中的噪声。这里，dk表示第k个生成的文档，f表示编码器操作。

3.计算k个矢量的平均值，得到向量v。该向量保存来自用户的查询和所需答案模式的信息。

4.使用向量v从文档库中检索答案。


Src: Precise Zero-Shot Dense Retrieval without Relevance Labels

## 2. Rewrite with LLM

原始查询可能并不总是用于LLM检索的最佳查询。可以首先使用LLM来重写查询，然后再进行检索和生成答案，而非以原始查询中检索内容并生成答案。

Src: Query Rewriting for Retrieval-Augmented Large Language Models

## 3. Step-Back Prompting

一种提示技术，使LLM能够从包含特定细节的实例中抽象、提取高级概念和基本原理。其思想是将“step-back问题”定义为从原始问题派生出的更抽象的问题。

例如，如果查询包含大量细节，LLM很难检索相关事实来解决任务。如图5中的第一个例子所示，对于物理问题“如果温度增加2倍，体积增加8倍，理想气体的压力P会发生什么？”在直接推理该问题时，LLM可能会偏离理想气体定律的第一原理。

包含两步：

1. 抽象：最初，我们提示LLM提出一个关于高级概念或原理的广泛问题，而不是直接响应查询。然后，我们检索关于所述概念或原理的相关事实。

2. 推理：LLM可以根据这些关于高级概念或原理的事实推导出原始问题的答案。我们称之为抽象推理。


## 4. Query2doc

提出了Query2doc方法来进行query改写，它使用LLM的一些提示生成伪文档，然后将它们与原始查询组合以创建新的查询。

Query2doc认为，HyDE隐含地假设groundtruth文档和伪文档用不同的单词表达相同的语义，这可能不适用于某些查询。

Query2doc和HyDE之间的另一个区别是，Query2doc训练有监督的密集检索器。

Src: Query2doc: Query Expansion with Large Language Models

## 5. Iteration

提出了ITER-RETGEN方法，它使用生成的内容来指导检索。它在检索-读取-检索-读取流中迭代地实现“检索增强的生成”和“生成增强的检索”。

在每次迭代t中，我们首先使用上一次迭代的生成yt-1，将其与q组合，并检索前k个段落。接下来，我们提示LLM生成输出yt，该输出yt将检索到的段落（表示为Dyt-1||q）和q合并到提示中。一共进行 T 次检索生成迭代 ，

Src: Enhancing Retrieval-Augmented Large Language Models with Iterative Retrieval-Generation Synergy

## 基于知识体系的方法

通过query中词的同义词、上下位词替换改写query.


```
苹果6手机多少钱 -> iphone6多少钱 
新鲜水果->新鲜苹果 
```

同义词挖掘方法：如语料对齐挖掘，上下文挖掘等 

语料对齐法：可以拿点击日志、session日志、anchor语料 挖掘到对齐语料。再使用机器翻译模型(比如 IBM Model1)等从对齐语料中挖掘同义词。 

上下文挖掘：简单来说就是同义词往往有着相似的上下文。通过计算词的上下文相似程度来挖掘同义词

e.g. 
> 折抵换购 iPhone XR 仅 RMB 176/月起  
> 折抵换购 苹果 XR 仅 RMB 176/月起

## 建立知识体系

 基于日志挖掘的方法：基于用户的行为数据（如query点击日志和query session日志），挖掘query和query之间的关系。 
 
 对于点击同一文档的query建立query共现关系，以及基于同一个session下的query建立共现关系。 

 ## 基于翻译思想

  query 作为源语言，rewrite作为目标语言，训练翻译模型，将query通过推理翻译达到rewrite目的

  语料对齐+机器翻译：原理就是把原语言翻译成目标语言概率最大化。这里需要单词对齐的方法。
  
  Ref: 宗成庆:《自然语言理解》讲义 11章

Reference: https://zhuanlan.zhihu.com/p/685981587