---
layout: blogpage
title: Spark RDD 和 DataFrame 处理数据的一点探索笔记
comments: true
tags: Spark Skill Python
---

虽然说，曾经也听本科的C语言老师说过国外不看重上课教语言这样的说法，但是完全零基础啃API开始使用 Apache Spark 着实还是让我费了一番力气。

说的就是你！CS5344！

在这里记录一下在Lab1里用到的 RDD 和 DataFrame 的各种初级用法以及踩过的坑。

The following content are basically practiced in PySpark.

## RDD

- sparkContext.parallelize()
 
用于并行化驱动程序中的现有集合。

这是创建RDD的基本方法，主要在POC或原型制作时使用，它要求在创建RDD之前将所有数据都存在于驱动程序中，因此它并不是最常用于生产应用程序的。

    drinks = [("HoneyLemon", 1.5), ("Mojito", 6), ("TehC", 2)]
    rdd = sc.parallelize(drinks)

### RDD Transformation

- foreach(func)

仅返回满足`foreach`内函数条件的元素。

    def f(x): print(x)
    p = rdd.foreach(f)

这个例子在foreach中调用`print`函数，打印RDD中的所有元素。

- reduce(f)

执行指定的可交换和关联二元操作后，将返回RDD中的元素。

- flatMap(f)

A transformation operation that flattens the RDD/DataFrame (array/map DataFrame columns) 
after applying the function on every element and returns a new PySpark RDD/DataFrame.

where <f> is the transformation function that could return multiple elements to new RDD for each of the element of source RDD.

- filter(f)

返回一个包含元素的新RDD，它满足过滤器内函数条件。

    words_filter = rdd.filter(lambda x: 'Mojito' in x)
    filtered = words_filter.collect()
    print("Fitered RDD -> %s" % (filtered))  # Fitered RDD -> [('Mojito', 6)]

### RDD Actions 

- collect()

以数据形式返回RDD中的所有数据。在处理带有成千上万亿数据的巨大的RDD时，可能会耗尽驱动程序上的内存。

- take()

返回指定数目的记录。

## Pair RDD

Pair RDD is nothing but RDD containing a key-value pair. 

### Convert a DataFrame back to normal RDD 

    rdd = df.rdd

It only returns a Row object.

So if you want to have the regular RDD format.

Try this:
 
    rdd = df.rdd.map(tuple)
 
 or
 
    rdd = df.rdd.map(list)
    
have the regular RDD format.

### Actions of Pair RDD

Pair RDD 包含一些特有的函数。

- reductByKey

reduceByKey用指定的函数来合并相同key对应的value值。

- sortByKey

sortByKey转换是对RDD的key列进行排序。

## Demo Code

```python
# -*- coding:utf-8 -*-
from pyspark import SparkContext
sc = SparkContext("local", "Scratch")
#SparkContext is the entry point to any spark functionality.
#When we run any Spark application, a driver program starts, 
#which has the main function and your SparkContext gets initiated here. 
#The driver program then runs the operations inside the executors on worker nodes.

import numpy as np
import pandas as pd
# Before import python packages, you should determine the python driver for pyspark
# make sure you have set your env variables in ./conf/spark-env.sh 
#(if it doesn't exist you can use spark-env.sh.template as a base.)

def count(rdd):
	#Count() returns the number of elements in the RDD.
	rdd_cnt = rdd.count()
	print("Number of elements in RDD: %i" %(rdd_cnt))

def collect(rdd):
	# Collect returna all the elements in the RDD.
	rdd_col = rdd.collect()
	print("Elements in RDD: %s" %(rdd_col))

def reduce(rdd):
	#reduce() doesn’t return a new iterable but uses the function 
	#called to reduce the iterable to a single value.
	concat = rdd.reduce(lambda x,y:  x + ', ' + y)
	print("List of all companies : %s" %(concat))

def filter(rdd):
	# filter() returns only those elements which meet the condition of the function inside.
	rdd_filter = rdd.filter(lambda x: 'Go' in x)
	filtered = rdd_filter.collect()
	print("Fitered RDD: %s" %(filtered))

def foreach(rdd):
	#foreach() is like a for loop that iterates through every elemnt in the RDD
	def func(x): print(x)
	each_rdd = rdd.foreach(func)


def map(rdd):
	#map() returns a new RDD by applying a function to each element in the current RDD.
	rdd_map = rdd.map(lambda x: (x, 1))
	mapping = rdd_map.collect()
	print("Key value pair: %s" % (mapping))

def join(rdd1,rdd2):
	#join() returns RDD with a pair of elements with the matching keys 
	#and all the values for that particular key.
	joined = rdd1.join(rdd2)
	final = joined.collect()
	print("Joined RDD: %s" %(final))
	return joined

def groupbykey(rdd):
	# groupByKey() transformation is used to Group the values for each key 
	# in the RDD into a single sequence.
	group_rdd = rdd.groupByKey()
	for element in group_rdd.map(lambda x : (x[0], list(x[1]))).collect():
		print(element)

def reducebykey(rdd):
	# reduceByKey() transformation is used to merge the values of each key 
	# using an associative reduce function and learned it is a wider transformation 
	# that shuffles the data across RDD partitions.
	rdd2=rdd.reduceByKey(lambda a,b: a+b)
	for element in rdd2.collect():
		print(element)

if __name__ == '__main__':
	# define rdd
	rdd = sc.parallelize (
		["Amazon",
		"Apple", 
		"Banjo", 
		"DJI", 
		"Facebook",
		"Google", 
		"HiSilicon",
		"IBM",
		"Intel",
		"Microsoft",
		"Nvidia",
		"OpenAI",
		"Qualcomm",
		"SenseTime",
		"Twitter"]
		)
	# action
	# count(rdd)
	# collect(rdd)
	# foreach(rdd)
	# reduce(rdd)
	
	#transformation
	map(rdd)
	filter(rdd)
	# define rdd1 rdd2, which are used for join
	rdd1 = sc.parallelize([("Amazon", 1), ("IBM", 4)])
	rdd2 = sc.parallelize([("Amazon", 2), ("IBM", 5)])
	rdd_joined = join(rdd1,rdd2)
	groupbykey(rdd_joined)
	reducebykey(rdd_joined)```


## Reference

[1] 大数据入门与实战-PySpark的使用教程 - https://www.jianshu.com/p/5a42fe0eed4d

[2] PySpark flatMap() Transformation - https://sparkbyexamples.com/pyspark/pyspark-flatmap-transformation/

[3] How to convert a DataFrame back to normal RDD in pyspark? - https://stackoverflow.com/questions/29000514/how-to-convert-a-dataframe-back-to-normal-rdd-in-pyspark