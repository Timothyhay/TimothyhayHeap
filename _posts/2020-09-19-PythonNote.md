---
layout: blogpage
title: 用惯了C/C++的人不习惯的Python语法笔记！
comments: true
tags: Skill Note Python
---

## 枚举方法enumerate ##

enumerate(X, [start=0])

When using the iterators, we need to keep track of the number of items in the iterator. This is achieved by an in-built method called enumerate(). The enumerate() method adds counter to the iterable. The returned object is a enumerate object.

	days= {'Mon', 'Tue', 'Wed', 'Thu'}
	enum_days = enumerate(days)
	# enumearte using loop
	for enum_days in enumerate(days):
	   print(enum_days)

Output:
	
	(0, 'Thu')
	(1, 'Tue')
	(2, 'Wed')
	(3, 'Mon')

for 循环有两个变量参与时，第一个输出序号。序号可以从给定值开始。

	for count, enum_days in enumerate(days, 5): 
	   print(count, enum_days)

Output:

	5 Thu
	6 Tue
	7 Wed
	8 Mon


使用枚举方法来遍历字典吧！


	a = {1: 111, 2: 222, 3: 333}
	  ...: for i , item in enumerate(a):
	  ...:     print (i, item)

这种做法和字典的 dict.items() 方法是等效的：

	a = {1: 111, 2: 222, 3: 333}
	  ...: for i , item in a.items():
	  ...:     print (i, item)

Output:

	1 111
	2 222
	3 333

Reference:
----------
[1]Enumerate() in Python - https://www.tutorialspoint.com/enumerate-in-python
