---
layout: modern-article
title: Agentic Data Building Note
tags: Agent Note
comments: true
---

稍微记录一下 Coding Agent 的训练语料构建要点。

目标流程：数据生成、质量控制、效果验证、归因分析、反馈循环。

# 数据生成

以我们的 SWE-factory 为例。
- 首先发现、采集高质量仓库，按照 star、仓库类型、测试覆盖率、语言等业务目标来选。
- 选择合适的真实 Issue-PR对。有关联 issue、修复 PR、gold patch、diff规模合理、有测试的patch。对于图片、压缩包等二进制测试文件，SWE-Factory 会直接下载真实文件，并从补丁中移除不完整的二进制 diff，避免测试补丁无法应用。这里以 fail-to-pass 验证任务质量，在未应用 gold patch 时目标测试必须失败，应用后必须通过。
- 理想状态下也可以用 LLM 分类打上更多标签
- rollout：SWE-factory 会自动搭建每个任务的运行环境（由四个Agent协作：分别搜集依赖和测试方式、写Dockerfile、写测试脚本，实际构建并运行后根据报错迭代修改。它还会复用同一仓库、相近版本已成功的环境配置，以减少重复探索。）总之准备好舞台跑起来。

# 数据质量控制
- 清洗异常格式、去重
- 对 rollout 得到的轨迹做选择，分别层级筛选：
  - 筛掉无测试、环境问题的；
  - issue 模糊、diff 过大过小或无关的；
  - 依赖解析失败、测试有误的；
  - 轨迹质量问题 - 我的 STITCH/SWE-Prime 算法
这样最后能有 <1% 左右转化率吧（从候选仓库开始算的话）
- 根据任务目标确认配比，比如按任务类型、难度、特殊标签采样

# 效果验证
- 多 Benchmark 验证
- 多 Agent 验证，mini-SWE-Agent，OpenCode
- 多模型评测
- 不同算法、数据版本的 A/B Test，以及统计显著性分析

# 归因分析
做错误分类、根因分析 etc.
- 依然用我的 STITCH Stage 1，分析 failure feature 贡献
- 通用地说可以是 LLM-as-a-judge 方法

# 效果反馈
这一步我们希望建立机制把 workflow 变成 flywheel：
- 分析结论生成可执行数据生成指令（比如顶下搜索关键词、难度分布、场景 etc.）
- 数据配比优化
- 人工审核
- 看板：比如 ROI - 不同算法每轮迭代的算力花费、数据用量对应的提升

理想状态是系统发现问题、系统自动验证并解决。