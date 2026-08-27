---
layout: modern-article
title: 禅，RSI 与 Agent 最简实现艺术
tags: Agent
comments: true
---

2023年6月，OpenAI Head of Safety Systems - Lilian Weng 发表了一篇讨论非常广的文章：LLM Powered Autonomous Agents，
以至于一整年内有关 Agent 组成部分的截图都持续地在各大公司/高校组会/咨询公司或者公众号的材料中出现。

她将 LLM Agent 的核心公式定义为：
> Agent = LLM + Planning + Memory + Tool Use

时过境迁，LLM 本身的输出答案能力和驱动状态机流转的能力，加上设计过的规划与反思(Planning)、长短期记忆(Memory)、各种组件/钩子/Skills相关的工具调用(ToolCall)的能力依然可以概况一个Agent的组成部分。

再后来吴恩达把这几个模块抽象成了构建 Agent 的设计模式，加上了 Multi-agent Collaboration 的一种模式。旨在说明复杂任务分给多个 Agent 的状态。

还有各种各样的公司和角色在之后的时间争夺 Agent 定义的话语权。
笔者的一贯风格是辨别并摒弃不需要了解的无意义内容，比如 Google Vertex AI Agent 重新造了一些词来说明生产环境的 Agent 需要  Runtime Orchestration 和基于真实数据的 grounding 机制。
有些多 Agent 交互（比如 tau-bench，或者其他 HCI 领域的讨论）会提到人本身、人的约束或者LLM约束在 Agent 中的作用，在这个概念的基础上再加一层，
结合 Human-in-the-loop，监控和安全相关的机制做一些文章。但笔者认为这些都属于 Harness 的细节部分，不应该做为核心 Agent 模块在此讨论。

因此，笔者认为 Agent 实际上只需要考虑决策、规划层、存储层、执行层、协同层五个方面的设计即可。
也就是对应 -

$
Agent = LLM + Planning + Memory + Tool Use + Collaboration
$

这样的设计。

本文会谈谈笔者从 Agent 概念大爆发之后，从收集了几十篇 Planning 相关文章使劲读开始，到熟悉社区雨后春笋般出现的 Agent SDK，以及带工程团队手搓业务 Agent，
和最后动手做 long-horizon trajectory SFT & Agentic RL 的全流程里对 Agent 框架设计的一些理解。

*这里预计不含 long-horizon LLM 后训练相关内容，将在其他文章中展开。




# 1. Memory

我们从 Memory 开始。



