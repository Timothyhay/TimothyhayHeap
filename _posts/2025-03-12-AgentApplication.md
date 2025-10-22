---
layout: blogpage
title: Mechanism of and between Agents 
comments: true
tags: LLM Note
---

离 LLM Agent 的概念出现已经过去了许多个月。在模型能力一般，Agent 应用还停留在简单套壳的时候，这个概念看起来颇有一点没活硬整的意思。
虽然到现在它的本质也没有摆脱 “LLM + Prompt Engineering + if-else” 的简单框架（meme_让我看看你的真面目.jpg），
但确实让落地的应用见到了一些 LLM 应用开发的可能方向。

我想来谈一谈以 Manus 通稿营销爆火的背景下如今 Agent 的有趣特点，以及最近的研究里一些我的发现。

# 1 是什么让通用 Agent 如此不同 / What makes general agent general

迄今为止，不仅是研发领域的 Agent 应用，其他领域甚至宣称自己是通用 Agent 的产品的发展趋势都很明显 - 即扩大作用域。

所谓作用域，即该 Agent 能观察或影响的范围。封装一个 LLM 到一个 chatbot 里，它只能与聊天框作用，最多再看看历史上下文。
但是看看 Cline，bolt.new 这些辅助研发新生代产品，作用域被拓展到 git 仓库、terminal、文件系统、MCP 协议或其他工具，总之是更广阔的上下文。  
Agent 一定能让 LLM 在除了只有模型本身以外发挥出更大能力的核心就在于此 - 如果把调用工具的结果也视为上下文的扩展的话。

再看 OpenManus 的初版实现，可调用的工具也就是：`PythonExecute(), GoogleSearch(), BrowserUseTool(), FileSaver(), Terminate()`

这里的核心在于`brower_use`，极大地拓展了 Agent 作用域，这个简单的工具因为直接与浩瀚的互联网（和它的用户友好界面）相连，  
配合`google_search`让它可以借助各种现成的网页应用和信息完成复杂任务，本质上这个工具不能算*1个*工具 - 而是庞大工具及其数据的集合。这也成为了 OpenManus 通用能力的来源。

Microsoft Copilot 出来的时候也是主打了一个系统交互功能，将作用域拓展到操作系统本身，以及种种不同的应用（虽然现在适配的情况好像不是那么回事）。
在基本对话功能上，接入系统设置、加上截图工具、闹钟、Edge等等纷繁复杂的系统工具来实现操作系统级的通用[1]。  
这里的核心在于可以自由和用户当前界面内容和的复杂多样的微软操作系统内置软件交互的能力 - 同样也是扩展作用域的直接体现。

再看刚刚（2025/03/12）OpenAI 发布的 OpenAI Agents SDK[2]：

> OpenAI offers a few built-in tools when using the OpenAIResponsesModel:
>
> The WebSearchTool lets an agent search the web.
> The FileSearchTool allows retrieving information from your OpenAI Vector Stores.
> The ComputerTool allows automating computer use tasks.

看到 `WebSearchTool` 和 `ComputerTool` 的时候，通用的概念从何而来已经显而易见了。  
换言之，如果需要让 Agent 具有真正意义上的通用能力，就一定要又足够大的作用域来获得足够多的"observation"和"action"来填充模型上下文，如果内置工具不够丰富，就需要接入丰富信息源（或者工具源）来达成这一点。

而接入浏览器和用户个人电脑就是非常取巧也非常自然的做法。

另外，如果没有合适的工具可用，除了让模型自己生成回答，还有另一种思路 - 即现场生成代码，在沙箱执行并获取结果。qwen-agent 和 smol-agent 都基本具备相应能力，这种方案从解决问题的方向增强了 Agent 能力，

# 2 现有 Agent 项目架构的异同 / Similarities and differences between Agent architectures

前段时间组内思考过的一个问题 - 有关 Agent 框架如何选型？事到如今其实形态已经大差不差。

根据梁新兵（MGX/OpenManus作者）的观点[3]：一个极简的Agent框架，应该是可插拔的Tools和 Prompt的组合。

纵观上文类似 Cline，bolt，OmniParser，Manus 的作用域不同的项目级代码生成产品、辅助研发 Agent 或通用 Agent 框架或项目，越往后出现的产品，实现框架都越呈现出一种同样的范式。
因为大量做法已经成为业界共识 -

## multi-agent 协作思路

如 Manus -

它首先先使用`PlanningTool`做规划，形成一个包含多个任务的线性结构的计划，然后顺序执行每一个任务并动态分配给相应的 Agent。  
Agent 在执行每个任务的过程中，以 ReAct 循环的形式调用工具以完成每一个任务。

Manus 引入了 PlanningTool 作为单独工具 -
> 这很重要，比如 Claude-3.7 在 SWEBench 上达到 70% 解决率的效果，此前是 49% 的。这里一部分提升是来源于模型，另一部分就来源于规划。

## signle agent 思路

如 OpenManus -

继承了 Manus 的规划优势，通过 PlanningTool 实现任务分解，会使用`PlanningTool`形成一个包含多个任务的线性结构的计划，写入至一个 markdowns 文件中去.  
然后 OpenManus 查看这个 Plan，从中依次取出每个任务，在执行任务时，会将该任务分配给最适合处理该任务的 Agent(工具类)。

![openManusProcedure](/images/illustration/2025-03-12/openManusProcedure.png)


本质上决定一个ReAct Agent(Reason & Act)的效果的关键是提示词引导和工具使用。
在 OpenManus 中，Prompt 控制了 Agent 整体的行为逻辑，Tools 给定了 Agent 的行动空间，二者被定义就能完整诠释一个 ReAct Agent。  
在 React Agent 的基础上，基于 Function Call 实现了一个轻量的 ToolCall Agent，它以更格式化的方式去选择和执行工具 - smolAgent 和 OpenManus 也基本继承了这样的思路。

** 这里最值得优化的点在于，如何正确选择工具（或其他Agent）？ **

从目前的经验来看，至少有以下几点可以考虑：

1. 动态选择 - 在执行每个原子任务时临时选择工具或分配其他子Agent，这样能避免任务被早起规划框死，用更长上下文和更具体的状况决定工具选择
2. 意图识别 - 使用额外的模型对上下文进行分析，决定接下来是否需要调用工具、调什么工具
3. 正则匹配 - （OpenManus的做法）对工具类用途详细描述，根据描述内容进行正则匹配，快速判断是否调用

# 3 一些悬而未决的问题

1. 成本 - 首先大量的应用厂商，包括MGX，都会使用如缓存或 memory 压缩技术对 token 消耗进行优化，尽量减少每次调用的上下文长度。
此外，未来大家很可能会部署大量小模型，基于已有数据进行微调或强化，专注于优化某些节点或工具使用的能力。通过集成多个小模型的能力，实现完整甚至超过大模型的效果。
这里的成本也直接影响响应的效率。

2. Memory - 在处理复杂、长程任务（e.g. 浏览代码仓、访问长网页）时，如何压缩上下文并存储到记忆中的同时避免遗漏。而且 memory 存在多种类型，不仅仅是语义信息的记忆，还包括工具使用时产生的程序性记忆以及事件关联关系的记忆。
早期一些研究工作指出，memory 会随着时间或任务步骤的增加而消退。

> MemoryGPT 对一定的上下文进行总结，这是一种非常朴素但也有效的思想。Mem0则在 memory 更新过程中主动使用工具，涉及 memory删除、memory 更新和新增等操作。

# 4 补充一点想说的

## 关于 Function Call 和 MCP 协议

- Function Call 是一个更大的概念，让 LLM 从聊天工具变成能使用其他工具的智能。
- MCP 通过统一协议和请求格式访问不同的特定工具。

```
Function Call -> Tool

MCP -> [MCP Server] -> Tool 1
                    -> Tool 2
                    -> Tool 3
```


## SPO

SPO 是一套提示词优化工具。与传统需要大量数据集的优化方法不同，SPO适用于没有准确评分或数据集有限的场景。例如，写小红书文案或SEO优化时，我们可能只有几个满意的样例。SPO能在这种有限样例条件下进行提示词优化。该工具已经开源，在国内魔搭平台和Hugging Face上均有良好反馈。


# Reference

[1] [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/)
[2] [微软Copilot详细指南——你必须尝试的Copilot技能大全](https://zhuanlan.zhihu.com/p/673884336)
[3] [上线一天Github过万星，OpenManus核心作者聊Agent发展趋势](https://mp.weixin.qq.com/s?__biz=Mzg4Nzc3NjY0Mw==&mid=2247492035&idx=1&sn=5f3e1dedebc61bd545e45716e10e9ea1&from=groupmessage&scene=1&subscene=10000&clicktime=1741785126&enterid=1741785126&ascene=1&devicetype=android-33&version=28003844&nettype=WIFI&abtest_cookie=AAACAA%3D%3D&lang=en&countrycode=CN&exportkey=n_ChQIAhIQxudc8N60JPF%2FtC%2FUBzbbzhL0AQIE97dBBAEAAAAAADkbOKaO7%2BcAAAAOpnltbLcz9gKNyK89dVj0oiikdy4xoQhcdCIqHF0RAAqMDLcSmca0nbeDUmBfjv29lv7T%2FxzvqZX%2FHQ46Y5P%2Bl5QUU0ltyWrC%2BD9yZBM56L96eGULx46LDLGYIIyieAcq%2BlIL8KxVNDcEpN3NKKJ8Yz%2Fuu5CVaur3RAMx5QerpDrmhjwDXm%2FRq1KTkLHTSA0M50snllOnnW1st2fEwMO3sZYBPkJdwU4hr725mg1I0G%2FzQ6556RWl%2B23gZwLDNdYYbVKYAKjGYQitou5aS9kU6huKksSJgRcgUIPm9iY%3D&pass_ticket=4gUaH%2FLCVN%2FdD4AobB8zcPbYIMDlcTl2ARF%2FoNNN0OAlDQEV40UFlN57rfvi9Y8z&wx_header=3)
[4] [SPO]( https://www.modelscope.cn/studios/AI-ModelScope/SPO)