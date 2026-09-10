---
layout: modern-article
title: 离离又谱谱的 Coding Agent Reward Hacking 现象记录
tags: Agent
comments: true
---

# 设定

对 SWE-bench-like 的榜单，Agent 开始解题前，代码仓 `git reset --hard` 到对应 issue 被解决前的 commit 快照。
而 PR 中的 test cases 不向 Agent 暴露，只用来验收解决效果。每道题都有一个独立 docker 保证工作区和依赖独立。

我们提前配置好工具的封锁规则：
```json
{
    "permission": {
      "bash": {
        "git log *": "deny",
        "git show *": "deny"
      },
      "webfatch": "deny"
    }
}
```
BTW, 这是 OpenCode 的配置。

我们在 GLM-5.2 从真实 github issue 中来自 5 个仓库 142 道 C++ 语言的题目中发现了大量 git 相关 reward hacking 行为。
（这批轨迹的解决率高到 90%+）

其中 11 条（8%）是在联网下载答案，107 条（75%）是提取了历史 git 记录。
虽然我们封堵了 git 的使用和 webfetch 工具，来防止模型看 git 记录中来自未来的提交看到答案；
但容器还是有出网能力，因为解题过程 Agent 很可能需要下载依赖。

这套配置在 GLM-5.1 上还不错，不过在 GLM-5.2 下出现了大量问题 - 大致行为有下面几类：

# Reward Hacking 模式

1. 虽然不能webfetch，但是可以通过 bash 直接 curl 下载 issue、diff 文件；甚至直接用 python 解析 github API 返回的 JSON 提取修改了哪些文件
2. 直接克隆仓库，在其中搜索 commit 提取具体代码变更
3. curl 直接下载修复后的源文件
4. 下载到补丁文件后直接 git apply
5. 有些语义等价的 git 命令。比如 git log = git rev-list 或者 git diff 来推断提交之间的代码变更；git show = git cat-file 读历史文件

# 修复
思想是在源头消除不该存在的信息。
- 网络层 iptables 防止出站
- 单 commit 仓库重建：容器构建的时候重建为一个只包含初始 commit 的干净 git 仓库，这样模型还能用 git 命令帮助解题，但是历史记录看不到有用信息。