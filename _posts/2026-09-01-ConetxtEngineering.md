---
layout: modern-article
title: Context Engineering in Conding Agent
tags: Agent
comments: true
---


Andrej Karpathy 说过：

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">+1 for &quot;context engineering&quot; over &quot;prompt engineering&quot;.<br><br>People associate prompts with short task descriptions you&#39;d give an LLM in your day-to-day use. When in every industrial-strength LLM app, context engineering is the delicate art and science of filling the context window… <a href="https://t.co/Ne65F6vFcf">https://t.co/Ne65F6vFcf</a></p>&mdash; Andrej Karpathy (@karpathy) <a href="https://x.com/karpathy/status/1937902205765607626?ref_src=twsrc%5Etfw">June 25, 2025</a></blockquote> <script async src="https://platform.x.com/widgets.js" charset="utf-8"></script>

the delicate art and science of filling the context window with just the right information for the next step. 

Science because doing this right involves task descriptions and explanations, few shot examples, RAG, related (possibly multimodal) data, tools, state and history, compacting... Too little or of the wrong form and the LLM doesn't have the right context for optimal performance. Too much or too irrelevant and the LLM costs might go up and performance might come down. Doing this well is highly non-trivial. And art because of the guiding intuition around LLM psychology of people spirits.

说它是科学，是因为做好上下文工程涉及任务描述和解释、少量示例、红黄绿（RAG）分类、相关（可能是多模态）数据、工具、状态和历史记录、信息压缩等等。信息太少或形式不对，LLM 就无法获得最佳性能所需的上下文。信息太多或与主题无关，则 LLM 的成本可能会增加，性能可能会下降。做好上下文工程绝非易事。说它是艺术，是因为它需要运用围绕 LLM 心理学和用户心理的直觉。



上下文隔离：主题检测、子Agent机制、Agent Skills、沙箱
上下文整理：压缩（多轮对话摘要、异构文件检索、场景摘要等）；分层动作空间（基于文件指针的可逆压缩）
上下文丰富：待办事项管理（planning，to-do list 工具及跨session存储）；IDE 集成上下文（Linter工具、terminal 环境信息 etc.）动态上下文拼接（AGENT.md，引用片段注入)
