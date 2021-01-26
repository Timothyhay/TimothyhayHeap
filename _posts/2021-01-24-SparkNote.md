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

 Basic Methods

- sparkContext.parallelize()
 
用于并行化驱动程序中的现有集合。

这是创建RDD的基本方法，主要在POC或原型制作时使用，它要求在创建RDD之前将所有数据都存在于驱动程序中，因此它并不是最常用于生产应用程序的。

    drinks = [("HoneyLemon", 1.5), ("Mojito", 6), ("TehC", 2)]
    rdd = sc.parallelize(drinks)

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

### Actions of Pair RDD - 

Pair RDD 包含一些特有的函数。

- reductByKey

reduceByKey用指定的函数来合并相同key对应的value值。

- sortByKey

sortByKey转换是对RDD的key列进行排序。

## Reference

[1] 大数据入门与实战-PySpark的使用教程 - https://www.jianshu.com/p/5a42fe0eed4d

[2] PySpark flatMap() Transformation - https://sparkbyexamples.com/pyspark/pyspark-flatmap-transformation/

[2] How to convert a DataFrame back to normal RDD in pyspark?
 - https://stackoverflow.com/questions/29000514/how-to-convert-a-dataframe-back-to-normal-rdd-in-pyspark