---
layout: modern-article
title: Coding Field OPSD 复现实践笔记
tags: LLM
comments: true
---

# 0. Why This Article? Why OPSD?

目前业界主要通过 GRPO 这种基于结果奖励的 RL 作为后训练提升 Agent coding 能力，但存在几个问题：

首先是 reward 过于稀疏，一个任务最终只给出一个通过/不通过的 0/1 信号，整条轨迹跑完才有反馈，中间过程关键决策的信号被严重稀释，模型难以学习到局部合理行为；
同时，这种稀疏性还会引发 GRPO 探索，也就是训练过程中 reward 先上升后骤降，输出退化为重复 token 或者无意义工具调用。

除了 reward 稀疏以外，某些信号本身也不可信，比如模型的 reward hacking 绕过验证脚本而非真正解决问题的行为 - 
有关 [SWE 任务 reward hacking 大赏]()可以直走左拐点击链接来隔壁特展展厅观看。

总之呢，和 [DeepResearch 的 RL 任务]()不同，Coding Agent 的多轮决策过程中容易出错，并且这些错误都很可能对结果有影响，
因此更依赖过程监督。蒸馏能提供 token 级稠密监督，正好弥补 reward 信号的不足。

<br>

最近 OPD(On-Policy Distillation) 突然开始火起来，一方面是学术界跟风，另一方面 Cursor 也发了一篇博客[1]提到通过 OPSD 帮助 composer 提点的事。
OPD 是说相比 RLVR 的稀疏结果奖励，可以提供 token 级信号。 但标注 OPD 有一个前提：teacher 和 student 需要词表相同。
因为 token 级监督要求两者在每个位置给出可对齐的分布，如果是不同词表的模型，词表在切分方式、special token 上处理各异，
同一文本会被切成不同的 token 序列，这样 token 级 比骄傲会把语义分期和 tokenizer 差异混在一起，扭曲监督信号。

跨词表蒸馏有 SimCT、X-Token、GOLD 等对齐/投影方法，但需要额外投影矩阵和对齐过程，而且仍引入失真，效果不稳定。
我们的目标模型 Qwen 3.6-27B 在 SWE-Bench-Verified 上达到 77.2%，也已经在开源模型前列了。
这种时候开源模型模型里没有和他词表相同且更强的可以作为教师模型，于是我们选择 OPSD 让模型自蒸馏，自己当 teacher。

这个过程，teacher 靠特权信息（gold patch、工具列表、局部纠偏提示）获得比 student 更准的分布，而教师模型和学生模型天然同词表，
token 一一对应，用教师模型和学生模型的 token 级 KL 散度作为 loss 反向传播并优化模型，用来替代结果奖励。


更多背景可以参阅文末参考文献[2~4]。

# 1. 特权信息构造

在 OPSD 中，我们通过分析自己的 Coding Agent 产品用户的真实轨迹，在 badcase 归纳出常见的失败模式，
一共定义了 17 种 issue 类型：
- 调用不存在工具
- grep 未提供 pattern
- 编辑了无关文件
- 长时间不执行测试
- 重复执行失败命令
- 过早放弃探索任务

etc. 
总之没见到一种 issue，都在教师模型的上下文注入一条针对性 hint。相比人工经验拍出的 hint，或者合成的数据，
这种基于真实用户 badcase 的 hint 更容易让模型在实际 coding 场景进行提升。

实际用起来，大概就是：

学生模型某轮调用了不存在的工具 ReadLints，报错。下一轮如果随意猜测工具整条轨迹将失效。
教师模型的处理是：在改轮 assistant 生效前注入一条 hint （`<system_reminder>可用工具列表如下：bash, grep, read...`）
同一组权重，获得 hint 后教师模型分布中 bash、grep 这些 token 的概率显著上升，而学生分布依然集中于 ReadLints。
该位置的 KL 较大，梯度将学生分布向**不再猜测不存在的工具**的方向修正。

值得注意的是，hint 的注入是**局部**的，只影响那一个 bad turn 的分布，不会影响整个轨迹。
这里参考了 Cursor 针对局部行为进行修正的方案，而非把全局 hint 全部注入，
通过 hint_mask 标记 hint 窗口内的 token（也就是出错轮的 assistant 部分）。
蒸馏 loss 仅在这些 token 上计算，其余 token 不参与蒸馏。

由于一条 rollout 轨迹能展开成多个训练样本，目前一条轨迹大概可用生成 3~5 个 sample，这样也提升了数据产出的效率及监督信号的密度。

# 2. 环境与架构

- 训练侧：16 节点 96 NPU 运行 FSDP Actor（PPO training），把 27B 参数切分到 128 张 NPU
- 推理侧：6 节点 32 NPU 运行 vLLM，8 replica，每个 replica TP=4，给 student 做 on-policy rollout
- 执行机：2 节点，各运行 10 各并发 Docker 容器（一共20并发），每个容器运行一共 OpenCode Agent 完成真实代码任务。

训练和推理节点之间通过权重同步（HCCL broadcast）保持学生模型策略一致，rollout 轨迹通过 HTTP 回传给训练集群做蒸馏。

**当前方案的问题：**

比如 on-policy 要求每个 step 都用最新权重采样，但 rollout 耗时长、成本高，因此训练与 rollout 应该采用异步流水线的方式，
也就是 trainer 执行当前 step 梯度更新时，rollout 集群已经在使用上一版权重采样下一批轨迹比较好。

# 3. 总结

OPSD 的思路很简洁优雅：利用同一个模型构造教师和学生，通过 hint 提供额外特权信息，再用 token-level KL 替代系数 reward，但实际训练还是发现很多优化点。
比如如何构造高质量 hint，全量 logits 计算时 OOM 问题，训练效率问题 etc. 等一个有缘的时间优化。


# Reference

[1] Cursor 的实践文章：https://cursor.com/blog/composer-2-5#fnref-1

[2~4] 有关 OPSD，可以看看 Self-Distillation Enables Continual Learning、Reinforcement Learning via Self-Distillation 和 Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models。