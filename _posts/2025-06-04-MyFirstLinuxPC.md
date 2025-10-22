---
layout: blogpage
title: Before You Wanna Act Like a Hacker (in those movies)
comments: true
tags: Skill
---

I installed my first Win 11 & Arch Linux dual boot laptop yesterday.
Setting up Arch Linux cost me too much time that I did not go to bed until 2 am. So I think there's something i can take notes.

First things first, read [official wiki](https://wiki.archlinux.org/title/Main_page) carefully.
It covered most(90%+) of the solutions to the trouble may get during the installation.
Especially when you need to install the latest version, the official guide will never be outdated.

For CN region users, a [Chinese Ver.](https://wiki.archlinuxcn.org/wiki/%E5%AE%89%E8%A3%85%E6%8C%87%E5%8D%97) also available and sometimes more useful
because of the nice localization - it tells which dependencies may not be accessible due to the network issue, and give out an alternative.

Here I'd like to mention something may not be the most detailed in the official wiki.

## 1. Disk Partition

首先要进行一次磁盘分区。尤其是双系统的电脑，需要的分区类型也需要查一下。

## 2. Resize and Moving

太小的分区可能放不了引导。具体要多大的分区呢？看看Wiki！

大家都说不要动 - 但其实飞动不可的分区，是可以先格式化再重新新建来移动的！

## 3. The Dependency Relationship when Mounting

如果要取消挂载磁盘，注意需要先取消它的子节点。
