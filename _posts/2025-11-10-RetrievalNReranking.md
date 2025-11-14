---
layout: blogpage
title: Retrieval and Reranking for Code Snippet
comments: true
tags: Deep-Learning
---

I need a plan for effective code retrieval to extract the most relevant part of code file(s) before send it to LLM - today we call it **Context Management** (code as most the context window)

For most of the reranking task, a 2-step retrieval is a usual combo:
1. Embedding model roughly sort 
第一阶段 bi-encoder（快速粗排）：unixcoder-base、grapecodebert-base、bge-m3、nv-embed-v2 等代码专用 embedding
2. Reranker model finely sort  
第二阶段 reranker（精排 Top 50→Top 10）： cross-encoder


jina-embeddings-v2-base-code is an multilingual embedding model speaks English and 30 widely used programming languages. Same as other jina-embeddings-v2 series, it supports 8192 sequence length.

CodeT5+ is a new family of open code large language models with an encoder-decoder architecture that can flexibly operate in different modes (i.e. encoder-only, decoder-only, and encoder-decoder) to support a wide range of code understanding and generation tasks. It is introduced in the paper:

CodeT5+: Open Code Large Language Models for Code Understanding and Generation by Yue Wang*, Hung Le*, Akhilesh Deepak Gotmare, Nghi D.Q. Bui, Junnan Li, Steven C.H. Hoi (* indicates equal contribution).

Compared to the original CodeT5 family (base: 220M, large: 770M), CodeT5+ is pretrained with a diverse set of pretraining tasks including span denoising, causal language modeling, contrastive learning, and text-code matching to learn rich representations from both unimodal code data and bimodal code-text data. Additionally, it employs a simple yet effective compute-efficient pretraining method to initialize the model components with frozen off-the-shelf LLMs such as CodeGen to efficiently scale up the model (i.e. 2B, 6B, 16B), and adopts a "shallow encoder and deep decoder" architecture. Furthermore, it is instruction-tuned to align with natural language instructions (see our InstructCodeT5+ 16B) following Code Alpaca.


### LLM-based Reranker

CodeRankLLM is a 7B LLM fine-tuned for listwise code-reranking. When combined with performant code retrievers like CodeRankEmbed, it significantly enhances the quality of retrieved results for various code retrieval tasks.


## Reference

https://huggingface.co/nomic-ai/CodeRankEmbed

https://gangiswag.github.io/cornstack/