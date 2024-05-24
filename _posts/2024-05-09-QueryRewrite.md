---
layout: blogpage
title: Query Rewrite in RAG
comments: true
tags: RAG
---

Review of query rewite methods to increase retrieval accuracy during query stage.

The significance of query rewriting: aligning the semantic space of a user query with the semantic space of a document - allows LLM to provide more accurate answers.

查询改写的意义：将用户查询的语义空间与文档的语义空间对齐，使LLM提供更准确的回答。

在搜索场景中，由于用户搜索词Query和检索文本Document之间存在大量表述不一的情况，在文本检索框架下，此类文本不匹配导致的漏召回问题严重影响着用户的体验。对这类问题业界一般有两种方案：用户端拓展用户的查询词——即查询改写，或Document端拓展文档关键词——即Document标签。考虑到RAG服务难以对用户入库知识做修改，笔者经手的服务也暂不聚焦在增加知识元数据的功能，这里主要收集前一种解决漏召回的方案：查询改写（Query Rewriting，或称为查询扩展Query Expansion）。查询改写的应用方式是对原始Query拓展出与用户需求关联度高的改写词，多个改写词与用户搜索词一起做检索，从而用更好的表述，帮用户搜到更多符合需求的内容。

In the search scenario, due to the large number of inconsistent expressions between the user's search term Query and the retrieved text Document, under the text retrieval framework, the missed recall problem caused by such text mismatch seriously affects the user experience. The industry generally has two solutions for this kind of problem: the user side expands the user's query terms - that is, query rewriting, or the Document side expands the document keywords - that is, Document tags. Considering that it is difficult for the RAG service to modify the user's knowledge stored in the database, the services I handle do not focus on the function of adding knowledge metadata for the time being. Here we mainly collect the former solution to missing recall: Query Rewriting (also known as Query Rewriting / Query Expansion). The application method of query rewriting is to expand the original Query into rewritten words that are highly relevant to the user's needs. Multiple rewritten words are searched together with the user's search terms, so as to use better expressions to help users find more content that meets their needs. 

## 1 借助 LLM 的方法

### 1.1 HyDE

假设文档嵌入（HyDE）方法：生成假设文档，以便最终查询向量v与向量空间中的实际文档尽可能紧密地对齐。

步骤：

1.使用LLM基于查询生成k个假设文档。这些生成的文件可能不是事实，也可能包含错误，但它们应该于相关文件相似。此步骤的目的是通过LLM解释用户的查询。

2.将生成的假设文档输入编码器，将其映射到密集向量f(dk)，编码器具有过滤功能，过滤掉假设文档中的噪声。这里，dk表示第k个生成的文档，f表示编码器操作。

3.计算k个矢量的平均值，得到向量v。该向量保存来自用户的查询和所需答案模式的信息。

4.使用向量v从文档库中检索答案。


Src: Precise Zero-Shot Dense Retrieval without Relevance Labels

### 1.2 Rewrite with LLM

原始查询可能并不总是用于LLM检索的最佳查询。可以首先使用LLM来重写查询，然后再进行检索和生成答案，而非以原始查询中检索内容并生成答案。

Src: Query Rewriting for Retrieval-Augmented Large Language Models

### 1.3 Step-Back Prompting

一种提示技术，使LLM能够从包含特定细节的实例中抽象、提取高级概念和基本原理。其思想是将“step-back问题”定义为从原始问题派生出的更抽象的问题。

例如，如果查询包含大量细节，LLM很难检索相关事实来解决任务。如图5中的第一个例子所示，对于物理问题“如果温度增加2倍，体积增加8倍，理想气体的压力P会发生什么？”在直接推理该问题时，LLM可能会偏离理想气体定律的第一原理。

包含两步：

1. 抽象：最初，我们提示LLM提出一个关于高级概念或原理的广泛问题，而不是直接响应查询。然后，我们检索关于所述概念或原理的相关事实。

2. 推理：LLM可以根据这些关于高级概念或原理的事实推导出原始问题的答案。我们称之为抽象推理。


### 1.4 Query2doc

提出了Query2doc方法来进行query改写，它使用LLM的一些提示生成伪文档，然后将它们与原始查询组合以创建新的查询。

Query2doc认为，HyDE隐含地假设groundtruth文档和伪文档用不同的单词表达相同的语义，这可能不适用于某些查询。

Query2doc和HyDE之间的另一个区别是，Query2doc训练有监督的密集检索器。

Src: Query2doc: Query Expansion with Large Language Models

### 1.5 Iteration

提出了ITER-RETGEN方法，它使用生成的内容来指导检索。它在检索-读取-检索-读取流中迭代地实现“检索增强的生成”和“生成增强的检索”。

在每次迭代t中，我们首先使用上一次迭代的生成yt-1，将其与q组合，并检索前k个段落。接下来，我们提示LLM生成输出yt，该输出yt将检索到的段落（表示为 Dy_t-1 || q ）和q合并到提示中。一共进行 T 次检索生成迭代。


Src: Enhancing Retrieval-Augmented Large Language Models with Iterative Retrieval-Generation Synergy

## 2 基于知识体系的方法

通过query中词的同义词、上下位词替换改写query.

e.g. query 替换例子
> 苹果13手机多少钱 -> iphone13多少钱  
> 新鲜水果->新鲜苹果 


同义词挖掘方法：如语料对齐挖掘，上下文挖掘等 

语料对齐法：可以拿点击日志、session日志、anchor语料 挖掘到对齐语料。再使用机器翻译模型(比如 IBM Model1)等从对齐语料中挖掘同义词。 （语料对齐+机器翻译：原理就是把原语言翻译成目标语言概率最大化。）


上下文挖掘：简单来说就是同义词往往有着相似的上下文。通过计算词的上下文相似程度来挖掘同义词

e.g. 
> 折抵换购 iPhone XR 仅 RMB 176/月起  
> 折抵换购 苹果 XR 仅 RMB 176/月起

Ref: 宗成庆:《自然语言理解》讲义 11章

有关同义词替换，实际处理时需要注意语料内容。如果Query内容是有高度结构化要求的，则不宜进行替换（或删除）。这类语料包括成段的代码内容。
所谓删除指的是：

## 删除低权重token

完成分词后，在原始知识库中计算各token权重（如以IDF作为指标），同时以原始query大致筛选出一部分目标文档。将低权重token去除，组成新的query再在更相关的目标文档进行二次检索，重新获得一组结果。


## 建立知识体系

 基于日志挖掘的方法：基于用户的行为数据（如query点击日志和query session日志），挖掘query和query之间的关系。 
 
 对于点击同一文档的query建立query共现关系，以及基于同一个session下的query建立共现关系。 

 ## 基于翻译思想

query 作为源语言，rewrite作为目标语言，训练翻译模型，将query通过推理翻译达到rewrite目的

语料对齐+机器翻译：原理就是把原语言翻译成目标语言概率最大化。这里需要单词对齐的方法。
  


## Reference

[1] https://zhuanlan.zhihu.com/p/685981587
[2] 美团搜索中查询改写技术的探索与实践 - https://tech.meituan.com/2022/02/17/exploration-and-practice-of-query-rewriting-in-meituan-search.html