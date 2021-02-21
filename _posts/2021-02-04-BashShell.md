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

首先是关于特殊变量(begin with $)的问题:

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


## echo 的神奇用法

### 新建文件并写入内容

```shell
# Create plp.conf and Set Fs for make_plp.sh
echo "--sample-frequency=16000 " > conf/plp.conf
```

使用 `echo` 也可以把信息写入文件中。这个写入操作包含新建和写入，在文件不存在的时候会自动新建。

- '>' 重新创建
- '>>' 追加

追加操作即在文件尾写入。

### 防止换行的参数

Using the backslash interpreter -e, you can manipulate how the line appears on the output. 

For example, to print a new line, use the ‘\n‘ escape character option as shown below:

```shell
echo -e "hello w\norld!"


    hello w
    orld!

```shell
echo -e "hello w\c"
echo "orld!"
```

    hello world!

## 输出命令结果

Shell中使用反引号进行命令替换，命令替换使Shell可以将命令字符替换为命令执行结果的输出内容。

```shell
# Make acwt a str like 0.xxx 
weightstr="0""`echo "scale=3;$acousticWeight/1000" | bc`"
```
			
同样的功能也可以使用$()来实现。

```shell
echo "Today is `date +%D`"
```
    Today is 02/09/13

```shell
echo "Today is $(date +%D)"
```
    Today is 02/09/13

## 比较运算符

- 大于 -gt (greater than)
- 小于 -lt (less than)
- 大于或等于 -ge (greater than or equal)
- 小于或等于 -le (less than or equal)
- 不相等 -ne （not equal）


## Reference

[1] Linux Bash Shell 特殊变量 - https://www.cnblogs.com/chjbbs/p/6393805.html
[2] 16 Echo Command Examples in Linux - https://www.linuxtechi.com/echo-command-examples-in-linux/