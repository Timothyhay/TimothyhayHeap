---
layout: blogpage
title: How do I set up my own secret base?
comments: true
tags: Skill Jekyll
---

Oh.. Actually I mean - What platforms or interesting techniques did I use to build this website!

## Jekyll

All the pages has been set up with the help of Jekyll framework.

## LeanCloud

<font color="blue">LeanCloud</font> 提供的是一种 Serverless 云服务，包括数据存储、云侧简单程序或者脚本的容器化部署、即使通信和消息推送服务等等。
部署起来相对简单，中文官网写的甚至是“只差程序员，如何把想法实现”。事实上差一个能部署这个的人可能还是和程序员有一点区别吧。

### Visitor Counter

借助这类工具通过 JS 存储访问数据，实现了 Visitor Counter 的功能。


## CI/CD

实际工作中的 CI/CD 是一系列流程与平台组成的规范过程，用来支持云服务的部署与维护。对这个小网站而言，在我研究环境变量的时候发现——

GitHub Action **（居然）**自带一个 CI(continuous integration) 工作流！
可以在这个 [文档](https://docs.github.com/en/actions/about-github-actions/about-continuous-integration-with-github-actions) 看到详情。
借助这个功能可以让这个静态网站的部署和使用变得更灵活一些，一些稍微需要一些技巧的事情也可以借助这个完成。