---
layout: blogpage
title: 关于用惯了C/C++的人不习惯的Python语法和特性
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

end – 用来设定以什么结尾。默认值是换行符 \n，可以换成其他字符串。

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

格式化输出方法：

	print("%d records removed in %s:" %(removal_count, column_name))


## 原地逆序遍历list ##

从列表末尾元素往前循环，步长为-1，直到下标为0的元素。
无需额外内存开销存放reversed(list)副本。

	>>> for zi in range(len(crt_eplist)-1, -1, -1):  # len(crt_eplist) == 4
	    	print(zi)  

	3
	2
	1
	0


## lambda operator 的用法 ##

使用lambda operator 本质上是使用一种内联的函数的定义方式。

比如：

	list_a = [1, 2, 3]
	list_b = [10, 20, 30]
	  
	map(lambda x, y: x + y, list_a, list_b) # Output: [11, 22, 33]

或者用于操作字典：

	dict_a = [{'name': 'python', 'points': 10}, {'name': 'java', 'points': 8}]
	  
	map(lambda x : x['name'], dict_a) # Output: ['python', 'java']
	  
	map(lambda x : x['points']*10,  dict_a) # Output: [100, 80]
	
	map(lambda x : x['name'] == "python", dict_a) # Output: [True, False]

> We can’t access the elements of the map object with index nor we can use len() to find the length of the map object.
> We can, however, force convert the map output, i.e. the map object, to list by list().


## copy & deepcopy ##

	import copy
	a = [1, 2, 3, 4, ['a', 'b']] #原始对象
	 
	b = a                       #赋值，传对象的引用
	c = copy.copy(a)            #对象拷贝，浅拷贝
	d = copy.deepcopy(a)        #对象拷贝，深拷贝
	 
	a.append(5)                 #修改对象a
	a[4].append('c')            #修改对象a中的['a', 'b']数组对象
	 
	print( 'a = ', a )
	print( 'b = ', b )
	print( 'c = ', c )
	print( 'd = ', d )

Output:

	('a = ', [1, 2, 3, 4, ['a', 'b', 'c'], 5])
	('b = ', [1, 2, 3, 4, ['a', 'b', 'c'], 5])
	('c = ', [1, 2, 3, 4, ['a', 'b', 'c']])
	('d = ', [1, 2, 3, 4, ['a', 'b']])


Reference:
----------
[1] Enumerate() in Python - https://www.tutorialspoint.com/enumerate-in-python

[2] Introduction to Print Statement in Python - https://www.educba.com/print-statement-in-python/

[3] https://medium.com/better-programming/lambda-map-and-filter-in-python-4935f248593

[4] Python 直接赋值、浅拷贝和深度拷贝解析 - https://www.runoob.com/w3cnote/python-understanding-dict-copy-shallow-or-deep.html