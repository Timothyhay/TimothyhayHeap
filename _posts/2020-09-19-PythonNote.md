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

	for count, enum_days in enumerate(days, 5): # Start from 5
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


## 函数内修改参数变量，如果是可变类型，会修改到函数外噢！ ##
当我们传的参数是int、字符串(string)、float、（数值型number）、元组（tuple) 时，无论函数中对其做什么操作，都不会改变函数外这个参数的值；

当传的是字典型(dictionary)、列表型(list)时，如果是重新对其进行赋值，则不会改变函数外参数的值，如果是对其进行操作，则会改变。


## print() 语法 ##

	print(*objects, sep=' ', end='\n', file=sys.stdout, flush=False)

param:

objects – 一次输出多个对象。输出多个对象时，需要用 , 分隔。

sep – 用来间隔多个对象，默认值是一个空格。

end – 用来设定以什么结尾。默认值是换行符 \n，我们可以换成其他字符串。

file: It should be an object with a write(str) method. If this value is not mentioned, it prints objects on the standard output device i.e. screen.

flush: The stream gets forcibly flushed if this value is True. By default, this value is False.

Example:

	>>> a = 'Hello'
	>>> b = 'World'
	
	>>> print(a,b)      # 直接打印，分隔符默认有一个空格
	Hello World
	
	>>> print(a,b,sep='') # 分隔符 设置为空
	HelloWorld
	 
	>>> print(a,b,sep=".")  # 分隔符设置为.
	Hello.World



Reference:
----------
[1]Enumerate() in Python - https://www.tutorialspoint.com/enumerate-in-python
[2]Introduction to Print Statement in Python - https://www.educba.com/print-statement-in-python/