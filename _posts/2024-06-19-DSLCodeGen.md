---
layout: blogpage
title: My Patent - A DSL based Test-Unit Generation Framework
comments: true
tags: Deep-Learning 
---

# How to design a nice DSL

1. Service Interaction 

选择了在服务原有调试脚本上扩展，以确保DSL能和服务各功能模块方便交互。

如访问环境变量/能自定义规则的Mock功能/获取请求或响应，以及继承原有JavaScript语法脚本的可拓展性。

2. LLM Adaptation

作为对 LLM 友好的 DSL，需要在 DSL 上做一些考虑。

语义一致性：也就是在用户输入阶段使用的 Language Chain 需要经过预处理，让不同顺序的  Language Chain 表达修改为同一顺序再作为模型的输入，以减少歧义。

逻辑压缩：压缩原生 JavaScript 中循环/判断以及一些跨度较高的关键词或较长的函数用法，使每个statement拥有独立功能，压缩上下文长度。

自然语言风格：选择 BDD 风格的脚本，在确保结构化性质的同时，用贴近自然语言的方式描述DSL。

3. Robustness

限制：对原生 JavaScript 以白名单方式限制了不安全用法，去除无法解析的语句和测试 API 无关的内容；同时减少了解析时的语法树复杂程度。

自动纠错：借助修改实例自动匹配规则进行纠错，在解析 AST 的过程中匹配规则，确保解析结果符合预期。