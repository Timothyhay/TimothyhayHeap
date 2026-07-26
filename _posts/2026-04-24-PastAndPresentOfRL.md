---
layout: modern-article
title: Past and Present of Reinforcement Learning
date: 2026-04-24
tags: LLM
comments: true
---





# 相关技术

R3 

vllm 有一个参数可以开起来，会返回一个 routed expert 值。这是一个包含类似 [batch_size, seq_len, num_layers, top_k] 的索引张量。详细记录了moe的时候怎么走的通路。在训练的时候吧这个 routed expert 经过后处理后输入给训练框架来激活对应专家。