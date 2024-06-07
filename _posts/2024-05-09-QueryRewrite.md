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

### 2.1 语料挖掘

高质量的数据可以显著改善头部流量的改写效果，并且决定了后续模型性能的天花板。在候选集生成方面，基于搜索日志的挖掘、基于翻译思想、基于图计算、基于Embedding都是工业界和学术界常用的方法；在候选集过滤判别方面则有句间关系分类、Embedding相似度计算等方法。我们结合美团搜索场景总结了各个方法的优缺点，并在每个挖掘算法组件都结合了用户行为和语义两方面信息，下文将对离线语料挖掘做具体介绍。

#### 2.1.1 搜索日志挖掘候选语料
搜索日志挖掘是工业界常用的同义词获取手段，挖掘的主要方向有：

用户搜索后点击共同商户： 利用两个点击相同Document的Query构建相关关系。这种相关关系可以挖掘到大量词对，但这种简单的假设缺点也很明显，点击共现的Query可能有不同程度的漂移。在美团场景下提供综合服务的店铺很多，会有两种类型团单大量出现在相同商户下的情况，挖掘到“拔牙”→“补牙”这种有语义漂移噪声的可能性更大。此外，这个方法依赖现有搜索的效果，无法挖掘到无结果Query的改写词。
从搜索Session中挖掘： Session是指用户在一段时间内“打开App→多个页面的浏览，多个功能的点击、支付等行为→离开App”的一次交互过程。该方法是利用用户在整次App访问过程中连续输入的Query来构建相关关系。Session挖掘依赖搜索结果程度低，因此泛化能力更强。但相应的缺点是，Session时间切割不好确定，并且序列中每个搜索词之间的关联方式比较隐蔽，甚至可能没有相关关系。需要结合业务特点设计时长、引入点击（例如一次Session在有点击前的搜索词都无点击，可能是有具体需求未被满足）等条件做挖掘。
词对齐： 词对齐借鉴了翻译的思想，具体方法是将Query召回的商户标题去除了商户名部分后剩余的部分做为平行语料，设计一些对齐策略如字对齐（包含相同的字）、拼音对齐（相同拼音）、结构对齐（分词后词位置相同）。该方法的缺点是强依赖于现有搜索的效果。

商户/商品内SEO： 商品场景下，部分商家上架时会做SEO，如：“加长 狗狗牵引绳 狗绳 狗项圈 遛狗泰迪金毛宠物大型中型小型犬 狗链子”。这一类挖掘来源的缺点是有比较大的噪声，并且噪音关联性较大比较难分辨（存在上下位类型、同位词类型、作弊等噪音类型）。
以上简单方法均可以挖掘到大量相关词对，但基于的假设和设计的规则都很简单，优缺点都非常明显。下面介绍几种优化的挖掘方法。

#### 2.1.2 基于图方法挖掘
图方法如经典的协同过滤以及Graph Embedding等，在推荐场景中通过利用用户和Document的关系构建图结构来推荐更相似的Document。在搜索场景下用户的搜索Query以及平台的Document通过点击、下单等方式同样也可以建模成图结构。在美团搜索的使用场景下，我们对构图方式做了如下两个改进：① Query和Document之间的边权重使用Query点击Document的点击次数和点击率进行Wilson平滑的结果，而不只是Query点击Document的次数，从而提高相关性；② 在二部图中，将用户在Session中自行改写的Query也视为Document节点，与点击的Document标题一起进行构图，从而提高挖掘的数据量。

我们早期用SimRank++算法[2]验证了构图方式两个优化点的可行性，SimRank++算法是一种同构信息网络中的相似度量算法，它的思想是：如果两个用户相似，则与这两个用户相关联的物品也类似；如果两个物品类似，则与这两个物品相关联的用户也类似。该算法的优点是可以使用Spark进行大规模全局优化，并且边权重可以根据需要调整。优化构图后人工评测SimRank++优化前后查询改写数据量提升了约30%，同时准确率从72%提升到83%。

后续，我们用相同的思路尝试了其他图神经网络模型（GNN）。DeepWalk[3]在构造Sentence上下文采用随机游走的方法。 随机游走一般是将Query之间的关系建立成图，通过从一个点随机游走，建立起多条路径，每条路径上的Query组成一个句子，再使用上下文相关原理训练Query的Embedding。随机游走的优点就是关系具有传递性，和Query共现不同，可以将间接关系的Query建立联系。少量的数据经过游走能够产生够多的训练数据。例如在Session1中用户先搜索Query1后改为Query2再查询，在Session2中用户先搜索Query2后改为Query3再查询，共现的方法无法直接建立Query1和Query3的关联关系，而随机游走能够很好地解决。在改写词挖掘任务中，基于图的方法相较于直接从搜索日志挖掘词对的方法，挖掘的效率和准确率均有所提升。

#### 2.1.3 基于语义向量挖掘
在word2vec[4]后，Embedding的思想迅速从NLP领域扩散到几乎所有机器学习的领域，号称“万物皆可Embedding”，只要是一个序列问题均可以从上下文的角度表示其中的节点。此外，Embedding在数据稀疏性表示上的优势也有利于后续深度学习的探索。将Query Embedding到低维语义空间，通过计算Embedding间的相似度查找相关词，在挖掘相似词的任务中是常见且易于实践的挖掘方法。除了简单的在用户评论等语料上训练大规模词向量外（即图7a），在实践中还尝试了以下两种构建上下文的方法：

通过Query召回商户构建Doc2Vec[5]：通过Query召回或点击的商户作为上下文训练Embedding表征Query（即图7b）。由于美团场景下同一商户提供的服务、商品繁多，该方法在没有考虑Query本身类目意图的情况下，噪声比较大。
通过用户Session构建改写序列[6]：通过一个Session序列作为上下文训练Embedding表征Query（即图7c）。该方法的优点是有效的利用了用户自行换词的限制条件，挖掘覆盖率和准确率都相对更高。

设计不同的上下文结构得到Embedding后，为了进一步提高准确率后续的基本的步骤是：① 训练语料通过分词后，利用fastText训练Query的词向量，fastText训练时考虑了字级别的Ngram特征，可以将未登录Query的字、词Embedding进行简单求和或求平均，解决OOV（Out-Of-Vocabulary）的问题；② 在目标词表中，用词向量表示该词；③ 利用LSH，查找向量cosine相似度高于一定阈值候选词，或使用DSSM双塔模型[7]，通过有监督训练提高精度；④ XGBoost结合特征工程进一步过滤。

BERT[8]自提出以来深刻改变了自然语言处理领域的研究应用生态，我们尝试了一些使用BERT Embedding的方法，其中比较有效的是通过Fine-Tuning的Sentence-BERT[9]或SimCSE[10]模型获取词向量。

BERT计算语义相似度是通过句间关系下游任务完成的，方法是用特殊字符将两个句子连接成一个整体做分类，带来的问题是使用时需要两两组合造成大量冗余计算，因此不适合做语义相似度搜索或无监督聚类任务。Sentence-BERT借鉴了孪生网络模型的框架，将不同的句子输入到两个参数共享的BERT模型中，获取到每个句子的表征向量，该向量可以用于语义相似度计算，也可以用于无监督的聚类任务。

我们实践的方法基本与Sentence-BERT思想大致相同，使用下图中左图的方法构造有监督的改写对训练数据，用右图的方法在不同意图类型的历史搜索Query进行向量计算。

此处应该有一个 图8 Sentence-BERT训练和预测结构示意图

相比于前面的方法，双塔结构BERT的方法捕捉语义的能力更强，并且有监督训练的方式结合一些模型结构上的调整，能够减少各类漂移严重的Case。此外Sentence-BERT不依赖统计特征和平行语料，在任何业务上均可以比较方便的迁移和Fine-Tuning，对一些冷启动的业务场景非常友好。在此基础上利用Faiss[11]向量检索方法构建离线检索流程，能够支持在亿级别候选池中高效检索，通过该方法构建的改写候选能达到千万甚至亿级别数据量，且实测准确率较高。近几年的对比学习等方法在文本表示领域不断刷新榜单，从向量构建和向量交互方式等方面均可做持续的探索。

其他还有语料对齐挖掘，上下文挖掘等方法。 

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
  

# 3 操作

使用过滤Query中低词权重的token，使用压缩过的Query重新查询。


## Reference

[1] https://zhuanlan.zhihu.com/p/685981587
[2] 美团搜索中查询改写技术的探索与实践 - https://tech.meituan.com/2022/02/17/exploration-and-practice-of-query-rewriting-in-meituan-search.html