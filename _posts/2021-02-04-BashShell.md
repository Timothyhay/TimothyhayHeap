---
layout: blogpage
title: 从 Kaldi 开始的 Shell 编程
comments: true
tags: Shell Skill
---

Kaldi 这样纯粹的命令行软件的使用方式除了让我觉得非常程序员、非常pro以外，用一个个散装 Shell 脚本组合成的工具也确实让一个很久没有写过 
Shell 脚本的人有些懵逼。上次写 Shell 好像还是大三学操作系统的时候吧 .. 

接着这个机会把遇到的语法问题记录下来。这篇文章将只包含 Shell 的相关知识，Kaldi部分的问题大概会写一篇新文章来放。

## 特殊变量与命令行参数

首先是关于特殊变量(started with $)的问题:

e.g.

```shell script
echo "$0 $@"  # Print the command line for logging

if [ "$1" == "--fake" ]; then
  fake=true
  shift
fi
```
