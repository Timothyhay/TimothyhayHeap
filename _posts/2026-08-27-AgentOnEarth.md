---
layout: modern-article
title: 禅，RSI 与 Agent 最简实现艺术
tags: Agent
comments: true
---

Note: 这篇文章还在施工中。
[WIP]

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

先来介绍一下今天（2026年8月）在简洁方面有口皆碑的 Pi Agent 的实践。

没有太多个性化的内容，主要包含会话压缩和持久化。

## 1.1 Compaction

有关压缩的描述可以在这里(`packages/coding-agent/docs/compaction.md`)找到。
做法和主流 Agent 完全一样，就是 LLM 黑盒压缩 + 凑点东西进新上下文。
具体而言，流程分为 5 步：

1. 寻找切割点：从最新消息回溯，累积 token 估算直到达到 keepRecentTokens（默认 20k）
2. 提取消息：收集从上次保留边界到切割点的消息
3. 生成摘要：调用 LLM 按结构化格式总结，并把上一次摘要作为迭代上下文传入
4. 追加条目：保存 CompactionEntry，包含摘要和 firstKeptEntryId
5. 重建上下文：下次请求时用 摘要 + firstKeptEntryId 之后的消息。

所谓 Compaction Entry，是一个包含压缩状态的 JSON，它存储旧消息的摘要，并标记从哪个位置开始的消息要还原到上下文。

```js
interface CompactionEntry<T = unknown> {  
  type: "compaction";  
  id: string;  
  parentId: string;  
  timestamp: number;  
  summary: string;  
  firstKeptEntryId: string;  
  tokensBefore: number;  
  usage?: Usage;  
  fromHook?: boolean;  
  details?: T;  
}
```

CompactionEntry 数据结构负责记录摘要。

摘要使用固定的结构化格式（Goal / Progress / Key Decisions / Next Steps / Critical Context 等），并在多次压缩时保留之前的信息、更新进展状态。

**分割 Turn 的处理**
有一个细节处理（甚至比 CC 还好）：如果单个 turn 本身超出 keepRecentTokens，会产生"split turn"，
此时会生成两份摘要（历史摘要 + turn 前缀摘要）并合并。

也有一些 pi 的分支实现做到了并发地compaction来提升体验；具体在压缩的时候怎么存的问题，TencentDB Memory 有一个存 mermaid 图的设计。
总之，存什么和怎么存是一个重要问题。

## 1.2 Session Persistence

pi 有一个树状的会话历史记录设计。
整个会话以 JSONL 文件树形结构存储在 ~/.pi/agent/sessions/，每个条目有 id/parentId，支持用户原地分支（/tree, /fork, /clone）。

**这个压缩是有损的，但是支持完整还原。** 完整历史仍保留在 JSONL 文件中，可以通过 /tree 找回。这就是所谓上下文持久化。

# 2. Planning

pi 的文档[2]提到 pi 有意跳过了 sub agents 和 plan mode 这类功能。pi 的核心因此只是一个持续的简单 agent loop：

user query → 输出（可能带工具调用的）assistant message → 执行工具 → observation 给 LLM → 继续下一轮，直到没有工具调用为止。
这套 loop 没规划步骤，只是逐轮响应式地根据 LLM 的工具调用推进，而不是提前生成一个多步计划再执行。

p.s. 这个流程中有钩子（beforeToolCall/afterToolCall/shouldStopAfterTurn）可以介入执行流程。


# Reference

[2] https://pi.dev/