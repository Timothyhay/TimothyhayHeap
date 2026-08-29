---
layout: modern-article
title: 从零开始连接世界知识 - 我的 DeepResearch 产品成长记录
tags: Agent
comments: true
---


最近在总结之前做过的工作。对我来说最有意思也最有成就感的项目就是这个从零开始手搓自定义 Agent 框架，然后实现一个构想中的 DeepResearch 产品。
感觉好感慨，这段时间也是真正把曾经的洞察完完整整付诸实践，带上工程团队的小朋友们一起实现一个复杂系统的过程。
当时就是看到 Kimi Researcher 给 Agentic RL 留了方便的入口，但因为各种麻烦一直到很后来才开始真正开始做 RL。
从 Agent 到模型都以问答领域的工作为基础完成了一些突破，感觉说是自己问答工作的总结项目好像也没问题。

我是一个非常不喜欢为概念买账的人，但是还是*第一次*尝试用了第一性原理(First Principles)的方法思考一个DeepResearch系统应该是什么样的。

它确实在这段时间帮到了我很多 - 从我开始从最根本的问题开始思考：“研究的本质是什么？人类在研究过程中最核心的问题在哪？”

从这个本原问题出发，我将在本文讨论 DeepResearch 产品的设计、实现和调优方案。当然，对*我的*产品，请让我在下文以它曾经的代号'DMI'来命名。

如果你想跳过感悟和设计思考部分，也可以直接看[多Agent架构实现章节](#2-架构实现)、以及基于SFT+以Dr.GRPO思路优化后的[DAPO+调优结果](#4-模型调优)。

---

# 0. 研究的本质是什么：从第一性原理出发

我的理解是，研究的本质是一个系统性的过程，旨在**发现、解释和理解**某个主题。
整个过程不应该是信息的堆砌，而应该要组成一个从信息获取、处理、整合、表达或者呈现的完整闭环。

在这个过程中，我们很自然地会遇到一些问题，即所谓现代研究的痛点：

1. 信息太多了 - 低价值的信息过载问题
2. 信息找不到 - 信息孤岛问题，异构与隔离的信息
3. 信息看不懂 - 对人类而言太复杂的领域问题难懂，而且理解、记忆和关联都需要消耗大量时间
4. 信息难总结 - 提炼出关键脉络、核心矛盾很花时间
5. 信息难辨别 - 查到的东西真真假假难以验证溯源，但是正经研究又需要保证严谨性

> 总算是都压缩到了五个字。不过我也准备了专业一些的表达以应对潜在需要聊这个的英文和面试之类的正式场合，也就是：
>
> 1. 信息过载 / Info Overload
> 2. 信息孤岛 / Info Silos
> 3. 高认知负担 / High Cognitive Load
> 4. 提炼困难 / Synthesis Difficulty
> 5. 验证繁琐 / Tedious Verification

总之呢，一个合格的 DeepResearch 产品一定要能做到能够**参与研究的全流程**，确保生成的内容真的能**降低认知负担**，提高使用者的调研效率。
这个结果必须对得起复杂多跳任务的长时间开销，绝对不能只是信息的搬运工而已。

# 1. 产品蓝图

基于以上分析，我把 DMI 的开发分成了几个核心模块，这些模块的设计都关联到了后续的实现模块和开发流程中。

## 1.1 问题定义和方向

研究的起点需要明确方向和剪枝，一个确定的方向天然可以避开大量无关信息。

在这个阶段，用户（或者说我们）可能会有一个模糊的想法，需要转成一个**清晰、可研究的问题**，那么这个问题的核心议题应该是什么？

**应有功能：**

- 问题探索 / Query Exploration：允许用户问题改写、放缩，确认用户要的是大方案还是小回答。
- 研究地图生成 / Research Landscape：以知识图谱或者LLM直接生成关联子问题，彼此存在依赖，构造研究地图。让用户开局就有一个全局视野

## 1.2 信息获取与预处理

所谓 garbage in garbage out，我们需要全面查询所有**需要的**内容，无论他们**在哪里**，以何种**格式**。

**应有功能：**

- 多源数据接口 / Cross-Source Data Aggregator：准备好 arXiv、外部搜索引擎以及公司内部的文档、个人知识库的 API。理论上最好也要有尽可能多的媒体、商业数据源接口。
  搜索由 AI 驱动，在不同数据库中尝试最合适关键词和语义搜索。
- 质量分析与评估 / Quality Analysis：信息来源的权威性、内容倾向性需要有可信度评估，同时需要基本的信息去重去噪等预处理，对公司内部的内容社区给质量特征标签。

## 1.3 信息深加工

所谓研究的核心。面对成千上万份文档，我们需要想办法先转化成LLM好处理的形态的知识，然后做处理。

**应有功能：**

- 主题摘要 / Thematic Summarization：对多文档的检索结果，最好由一个综合性的回答来描述资料中提到的观点，来让LLM判断是否能从结果中找到需要的重点。
  这个摘要过程对检索结果而言可能是必须的，但是对每个子 Agent 的处理步骤而言也会需要，也就是任何需要在 LLM 对接的场合都要求。
- 结构化信息提取 / Structured Data Extraction：从文中收集话题、实体之间的关系等关键信息。
- 地图更新 / Landscape Update：将获得的实体与关系用于更新子问题研究地图。

## 1.4 分析结果

从已经获得的结果中，提出自己的理解，即使是 AI 我们也要这么要求。

**应有功能：**

- 观点对比分析 / Conflict Analysis：这里概况了几种分析方法，但核心是要求 LLM 基于冲突观点、时间线、假设进行知识的自动矫正，形成研究证据链。
- 眼前一亮的独到理解 / Surprise Me: 通过综合分析的结论理解事物的发展趋势、难以察觉的实体关系等，提出更惊喜的发现。

## 1.5 结果呈现

好的思想也需要好的表达，应该用各种方式可视化，同时能准确溯源。

**应有功能：**

- 生成报告 / Report Generation：生成带引用链接的长文本，同时应该有图表、摘要等专业内容。
- 自定义形式 / Customize Report：允许类似 NotebookLM 一样以 HTML、PPT等形式生成结果。

# 2. 架构实现

DMI 能对复杂问题进行任务分解、并行研究、信息汇总和报告生成。整个 DeepResearch 系统为分层架构，由几个 Agent 和跨层的公共能力组成。

**核心 Agent 包括：**

* Orchestrator (DeepResearch Agent)：负责宏观规划和管理。将主问题分解为子任务，创建并调度Researcher Agent，并对结果进行组织和综合。
* Researcher (DeepSearch Agent)：负责微观执行。每个Researcher是一个独立的 ReAct Agent，专注于一个具体的子任务，通过搜索、评估、筛选等操作得出针对某个单一话题的结论。
* Rewriter：结合对话历史，把内容改写为独立、完整、无歧义的研究查询
* Clarifier：负责用户提出问题后的澄清
* Formatter：负责把最终的结论整理成研究报告，PPT或者HTML

**公共能力：**
提供跨模块的通用服务，如预算控制、摘要等。这些特性实际被 Orchestrator 和 Researcher 等 Agent 调用。

我们从请求的执行链路开始，大约会经历 4 个阶段：

首先用户原始查询 query 进来 -

- Step 1: Clarifier 先判断用户请求是否清晰，不清楚的话请求用户澄清
- Step 2: Rewriter 结合对话历史，改写为独立、完整、无歧义的研究查询
- Step 3: Orchestrator （背后是 Qwen3-Coder-480B）
  - **规划**：预研究与问题分解。判断任务类型（广度、深度、直接回答）；计划一个解决通路，然后选择 Worker(Researcher) 数量（1~20），然后把初始节点子 Agent 和依赖的子 Agent 要研究的话题润色好，标记好依赖关系送进队列。
  - **实例化 Worker(Researcher)**：
    - 循环寻找并执行无依赖 + pending 状态的节点启动。
    - 每次启动会实例化一个 Researcher 并发执行，最多执行 7 步 ReAct循环。Researcher 的行为包括：
      - **ReAct 循环**：Thought -> Action:search[query] -> Observation
      - **检索时**：首先进行 Action 去重（查字典） + 跨搜索缓存提取；RAG 检索 + 内部门户网站检索 + LLM 评估再过滤。
      - **完成后**：返回一个包含 `{summary, source, new_topics}` 的结果，结果的总结 `summary `会追加到 Orchestrator 的 Factlist 里。这里的 `new_topics`来自于 Researcher的判断，会动态创建新的研究话题到队列中。把这个节点的任务标记为 complete。
  - **认知更新**：研究计划不是一成不变的，Orchestrator会监听Researcher的 `decompose`动作，并实时更新研究地图（DAG）。
  - **总结**：分为 快速总结 `QuickSummarize` 和 链路总结 `ChainSummarize`，分别用于总结某个节点的关键信息，以及一条有依赖关系的链路中的脉络，呈现形式不同；快速总结强调要点，链路总结要求像短文。
- Step 4：格式化所有链路的报告，使用长思考模型（temperature=0.2，较为确定）。对较长的文本分步处理，返回最终字符串。

<br>

**Quick questions:** 为什么选了 Orchestrator-Worker 架构的多 Agent 设计，你是 A\ 粉丝吗？

我不是，我没有。核心原因在于研究问题天然具有可变的**分解粒度**和**依赖拓扑，单 Agent 没法并行、Workflow 无法做到动态可变**。

在**并行性**上，调研类的任务需要多 Agent (Worker) 来实现分时动态（在不同时间启动）并发。对单 Agent 只能严格顺序运行， Workflow 可预设并行但流程更死。
在**上下文管理**上，单个 Agent 上下文长度有限，我们就很自然地希望每个子 Agent 只负责一个话题的调查，一方面最大程度使用了每个Agent的上下文窗口，另一方面也不会相互污染。
在**依赖管理**上，我们希望实现的非常重要的一个特性是**含依赖关系的，以 DAG 格式呈现的**知识网络动态拓扑。同时依赖关系不需要在单个 Agent 内部用上下文描述，我们用系统内结构化存储的 DAG 网络来描述。同时配合跨 Agent 的已知事实列表，预算控制等机制可以让每个子 Agent (Researcher) 都专注地把自己的研究任务做到极致，其他的事情交给专业的公共能力和 Orchestator 处理。

这里我想插一句，在完成 多Agent/单Agent 架构选型之前，看到了两篇观点相反的文章：

Cognition (Devin 的开发团队) 在他们的博客里直接说 "Don't Build Multi-Agents"[^dont-build-multi-agents]，但是同期 Anthropic 有一篇详细介绍如何构造 Agent 系统的博客[^building-effective-agents]，也有介绍他们多 Agent Research System 实践的[^multi-agent-research-system]。

这两篇文章都是在相近的时间被我看到的，当时在思考这个问题的也不止我一个[^reddit-single-or-multi]。最后我选择了多 Agent 架构，倒不是因为 Anthropic 直接给出了很详细的实践，而是我认为其实他们的观点本质上并不冲突。

Congition （很有可能他们接的外部模型效果也一般）的场景主要是代码仓、SWE相关的任务，文章里提到的问题主要和写操作相关（write-heavy 任务），属于代码依赖关系复杂的工程任务，盲目引入复杂的并行多 Agent，确实很容易导致每个 Agent 想的不一样导致隐式冲突，和他们最为抨击的上下文碎片化（多 Agent 无法看到彼此工作细节，导致犯错）。对写代码这个操作而言，确实不能把长链任务交给多 Agent，完了还要保证其他 Agent 总结回来的信息一点不丢失细节、彼此都对复杂项目做了最优修改。

但是 Anthropic 的话一直是在一个读相关（read-heavy 任务）的场景做推广，比如 DeepResearch 任务，Agent 之间本身是弱耦合、高并行性的，像我们严格控制每个 Agent 研究的内容，反而需要做严格的上下文压缩和隔离，这种和需要面对同一个代码仓做修改、需要了解全部上下文中修改细节的 write-heavy 任务完全不一样。

所以其实多 Agent 的选型是不矛盾的，只是场景不同选择不同罢了。

# 3. 模块详细设计

这里我补充一些几个核心 Agent 的设计细节。

## 3.1 Orchestrator

**规划部分**：即预研究与问题分解。也就是判断任务类型（广度、深度、直接回答）；
计划一个解决通路，然后分析用户查询复杂度，从而选择 Worker(Researcher) 数量（1~20）；

**子问题准备**：把初始节点子 Agent 和依赖的子 Agent 要研究的话题润色好，标记好依赖关系送进任务队列。

**认知更新**：研究计划不是一成不变的，根据研究中其他Researcher的新发现动态更新研究地图（DAG）。同时维护一个 Factlist，随着子任务的结束不断更新认知。
这里还有一个**已经调研过的子任务话题**和**已经查过的内容**的缓存，避免重复操作（主要是为了省钱）。

**并发管理**：同时调度多个Researcher来处理没有依赖关系的子任务（一次性尽可能多的获取所有可执行的任务），来缩短研究总耗时。这是我们选多Agent架构的最核心原因。

**组织结果**：即总结。分为 快速总结 `QuickSummarize` 和 链路总结 `ChainSummarize`，分别用于总结某个节点的关键信息，以及一条有依赖关系的链路中的脉络，呈现形式不同；快速总结强调要点，链路总结要求像短文。

## 3.2 Researcher

**决策去重**

**爬虫与搜索**

首先需要在ResearcherAgent的Action中调用search[query]工具时触发，这个工具自带一个重复动作去重/读取缓存的功能。
RAG核心流程是多路召回 + rerank top5；

> 如果判断挂载的知识库超过1个，原本的 DMI 外（工程团队选择的策略）是走一个全量召回的升级策略，这是一个现成的接口：
>
> 分库独立召回topK，然后每库结果min-max归一化，再RRF做跨库合并，最后去重+rerank精排。

向量路的query对象是一定要模型改写很多次之后才能传的，有例子提供给LLM改写。模型自己为1~3种问法各自生成一路检索请求。

对部分内部来源（比如个人知识库）做多路召回 + rerank；余下的结果我们简单extend拼接，不做加权融合。

因为调用现场搜索接口时已经按相关性排序了，我们都只选top 5结果，让模型看一下。这里召回的也不是完整网页，是大小合适的chunk。

后续我们统一去重，过滤掉重复的文档块。

然后LLM会多检索、爬取结果打分，配合playwright爬虫补全正文和 `crawl_record`跨轮缓存。

> 关于为什么要补爬虫正文：因为我们的内部门户网站接口 API 只返回摘要，得我们自己去抓内网界面转成 markdown 再供 LLM 消费。
>
> 没有什么特别的技术，无非是给界面注入Cookie，然后随机延迟模拟人类行为，智能滚动一下防止懒加载DOM里没有获取到全文。
> 最后用readability算法获取主体的正文，然后 markdownify 转 markdown。拿到之后交给 `crawl_record`，它会记录有没有爬过、以及有没有用。

异常处理：检索失败给提示，不阻塞流程；重排失败给默认分。

**语料评估**：一次对话内对大量URL进行打分，考虑4个因素：
- 时效性：网页时间戳
- 路径结构：分析URL路径结构判断聚合情况、结构高频出现认为权重变高
- 语义相关性：上文检索时完成，但 LLM 会在选择中再次考虑
- 语料质量：LLM 整体评价时也考虑语料质量


### 搜索去重

BTW，三层去重机制：

1. 单 Researcher 内去重：使用执行过的 Action 字典，以工具名、参数为key（实际上就是query内容）重复动作跳过并返回之前查过的结果。
2. 话题去重：由 TaskManager 处理，在 append 话题时按 topic 去重，相同任务不添加。
3. prompt 级别引导：Orchestrator prompt 要求子话题之间边界清晰且易于理解，避免重叠。需要研究的新话题不在当前上下文出现，而是选择研究完成后请求新话题放进研究队列。


### 长网页提取最优文本段

全部内容一股脑塞进 LLM 上下文虽然很省事，但是考虑到 token 成本和生成速度，肯定不是最优选择。
在实际应用汇总，我们其实只需要与问题最相关的部分就可以了。这部分知识会被添加到 Agent 的上下文中。

首先我们对长文档进行按预设规则（段落标记优先，否则定长）切块，根据 query 内容分别请求 embedding model，分别计算每块的余弦相似度。
然后我们选择总得分最高的连续块作为命中单元，所谓连续块就是在一定窗口内的多个相邻 chunk。考虑在阈值内选择 0 ~ 3 个块来保证文章关键信息不遗漏。


## 3.3 Formatter

**流程**：

已获得的参考文献块 → ~~Embedding 粗筛（BGE-M3 ）~~ → NLI / 精排信息关联度核验（BGE-Reranker-v2-m3）→ LLM 兜底

实际上 LLM 兜底是很后期才做的，并行起来效率还可以。


# 4. 模型调优

我们的最终目标是训练其中的 Researcher Agent，让他具备在复杂、模糊、需要多步推理的问题中自主、高效使用本地 search 工具的能力。
同时能过滤掉冗余信息、自动纠正检索方向，并能最终获得引用正式的高准确度回答。

这里我定义：
- 状态空间 $S$ 问题文本 + 已有历史轨迹。
- 总之空间 $A$ 模型是工具调用`<search>query</search>`或`<answer>ans</answer>` + 中间的推理内容。
- 奖励 $R$ 输出最终答案或达到最大步数时打分。完全匹配 1.0；有`<answer>`标签 0.1；标签都没 0；多次出现得分 /= 出现次数，最多/4 。
- 价值函数：在 DAPO 的基础上过滤了组内奖励无差异（std=0）的group，避免 advantage 浪费更新。

整个流程参考了 Search-R1，实现 test EM@50=0.508(+14.2%), EM@100 0.587(+13.3%)，显著高于训练前基线。
平均轮次@100 2.51（避免了直答坍缩）。
正确响应的长度从 250(GRPO) -> 592，说明模型在搜索后给出了更完整的答案。
期间 pg_loss 持续为负(-0.03 ~ -0.14)，entropy 有维度波动(0.33-0.43)。

包括数据准备、rollout 环境、advantage 计算、验证的细节，可以看这篇[完整的 DMI RL 笔记](/2026/08/25/DeepResearcherRL.html)。


# Reference

[^dont-build-multi-agents]: Cognition (Devin 开发团队) 工程博客：Walden Yan, [*Don't Build Multi-Agents*](https://cognition.com/blog/dont-build-multi-agents), 2025.
    
[^building-effective-agents]: Anthropic 工程博客：Erik Schluntz & Barry Zhang, [*Building Effective Agents*](https://www.anthropic.com/engineering/building-effective-agents), 2024.
    
[^multi-agent-research-system]: Anthropic 工程博客：[*How we built our multi-agent research system*](https://www.anthropic.com/engineering/multi-agent-research-system), 2025.
    
[^reddit-single-or-multi]: Reddit r/AI_Agents 社区讨论：[*Multi Agent or Single Agent?*](https://www.reddit.com/r/AI_Agents/comments/1lb0zb3/multiagent_or_single_agent/), 2025.
