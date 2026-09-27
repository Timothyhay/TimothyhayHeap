---
layout: modern-article
title: Context Engineering in Conding Agent
tags: Agent
comments: true
---


Andrej Karpathy 说过：

<blockquote class="twitter-tweet" data-align="center"><p lang="en" dir="ltr">+1 for &quot;context engineering&quot; over &quot;prompt engineering&quot;.<br><br>People associate prompts with short task descriptions you&#39;d give an LLM in your day-to-day use. When in every industrial-strength LLM app, context engineering is the delicate art and science of filling the context window… <a href="https://t.co/Ne65F6vFcf">https://t.co/Ne65F6vFcf</a></p>&mdash; Andrej Karpathy (@karpathy) <a href="https://x.com/karpathy/status/1937902205765607626?ref_src=twsrc%5Etfw">June 25, 2025</a></blockquote> <script async src="https://platform.x.com/widgets.js" charset="utf-8"></script>
上下文工程是一门精妙的艺术和科学，它需要用恰到好处的信息填充上下文窗口，以支持下一步操作。

- the delicate art and science of filling the context window with just the right information for the next step. 

Science because doing this right involves task descriptions and explanations, few shot examples, RAG, related (possibly multimodal) data, tools, state and history, compacting... Too little or of the wrong form and the LLM doesn't have the right context for optimal performance. Too much or too irrelevant and the LLM costs might go up and performance might come down. Doing this well is highly non-trivial. And art because of the guiding intuition around LLM psychology of people spirits.

说它是科学，是因为做好上下文工程涉及任务描述和解释、few-shot、RAG分类、相关（可能是多模态）数据、工具、状态和历史记录、信息压缩等等。

信息太少或形式不对，LLM 就无法获得最佳性能所需的上下文；信息太多或与主题无关，则 LLM 的成本可能会增加，性能可能会下降。

说它是艺术，是因为它需要运用围绕 LLM 心理学和用户心理的直觉。

原则上，一般从预算控制（预留合理输入输出长度，防止lost in the middle）、信噪比控制（防止上下文污染，剔除幻觉）、
记忆分层与压缩（长短期记忆、滑窗、配合黑盒压缩）、安全问题（防止 prompt注入）考虑。

Btw，有关 Memory：
- 短期记忆：当前会话历史、临时 Scratchpad（草稿本）。
- 长期记忆：通过向量数据库（RAG）或知识图谱，按需检索并动态注入。


在 Coding Agent 中，我设计的 Context Engineering 特性包括下面三类：
- 上下文隔离：主题检测、子Agent机制、Agent Skills（比如渐进式压缩的方法）、沙箱
- 上下文整理：压缩（多轮对话摘要、异构文件检索、场景摘要等）；分层动作空间（基于文件指针的可逆压缩）
- 上下文丰富：待办事项管理（planning，to-do list 工具及跨session存储）；IDE 集成上下文（Linter工具、terminal 环境信息 etc.）动态上下文拼接（AGENT.md，引用片段注入)


另外，软件工程领域也有上下文管理，通常是并发追踪、资源生命周期相关的管理。
从这里我们也可以获得一些启发，在 LLM 上下文窗口和 软工资源 之间有一些通用原则：

- 最小必要原则（Principle of Least Privilege）：只给当前执行单元提供它必须知道的上下文，过多信息即是噪声与隐患。
- 生命周期明确：上下文必须有清晰的创建、使用、销毁边界，不可悬挂或长久积压。
- 可观测与可追踪：给每次上下文流转赋予唯一追踪标识（Trace ID / Session ID），或者说文件指针（指向完整对话历史的）便于排查问题。