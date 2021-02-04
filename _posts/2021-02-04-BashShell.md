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

```shell
echo "$0 $@"  # Print the command line for logging

if [ "$1" == "--fake" ]; then
  fake=true
  shift
fi
```

- $0 - 当前脚本的文件名

- $n - 传递给脚本或函数的参数。n 是一个数字，表示第几个参数。

例如，第一个参数是$1，第二个参数是$2。

- $# - 传递给脚本或函数的参数个数。

- $* - 传递给脚本或函数的所有参数。

- $@ - 传递给脚本或函数的所有参数。

被双引号(" ")包含时，与 $* 稍有不同，下面将会讲到。

- $? - 上个命令的退出状态，或函数的返回值。

- $$ - 当前Shell进程ID。

对于 Shell 脚本，就是这些脚本所在的进程ID。