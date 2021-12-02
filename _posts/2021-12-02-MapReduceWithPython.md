---
layout: blogpage
title: Map & Reduce with Python
comments: true
tags: Python Skill Note
---

Note about how to use `map()` and `reduce()` in Python Standard Library. May add the comparation with PySpark's when I have time!


## Overview

[○ ○ ○ ○ ○ ○ ○]

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓ map

[□ □ □ □ □ □ □]

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓ reduce

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;☆

## Anonymous Function lambda 

```python
def add(x,y):
    return x + y
```
Can be translated to:

```python
    lambda x, y: x + y
```

## map()

`map()` 方法会把序列中的每一个元素通过一个函数映射生成一个新序列，其值为所有函数返回值。

The `map()` function iterates through all items in the given iterable and executes the function we passed as an argument on each of them.

```python
map(function, iterable)  
```
e.g.

```python
items = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, items))
```

*Note*: We cast `map_object` to a list to print each element's value. We did this because calling `print()` on a list will print the actual values of the elements. Calling `print()` on `map_object` would print the memory addresses of the values instead.

p.s. `map_object` is also iterable. Use `for value in map_object: print(value)` can also see the values.

## filter()

`filter()` 和 `map()` 方法类似（参数也一样），但是只返回映射结果为真的元素。

As the name suggests, filter() forms a new list that contains only elements that satisfy a certain condition, i.e. the function we passed returns True.

```python
fruit = ["Apple", "Banana", "Pear", "Apricot", "Orange"]
filter_object = filter(lambda s: s[0] == "A", fruit)

print(list(filter_object))
```

It returns:

    ['Apple', 'Apricot']

If we use map, the answer will be:

    [True, False, False, True, False]


## Reduce

In Python 3 `reduce()` isn't a built-in function anymore, we need to import it from functools module:

```python
from functools import reduce

reduce(function, sequence[, initial])
```

`reduce()` 在迭代序列的过程中，把前两个元素（只两个）传给函数，然后把得到的结果和第三个元素作为两个参数再次传给函数，以此类推，最后得到一个结果。

e.g. Get factorial(阶乘) of 10:

```python
from functools import reduce 

def f(x, y):
    return x * y
# Sequence with 1~10
items = range(1,11)

result = reduce(f, items)
print(result)
```
And we can give an initial value! It will be firstly computed:

```python
from functools import reduce

list = [2, 4, 7, 3]
print(reduce(lambda x, y: x + y, list))
print("With an initial value: " + str(reduce(lambda x, y: x + y, list, 10)))
```

The code would result in:

    16
    With an initial value: 26

## Reference

[1] https://zhuanlan.zhihu.com/p/77311224

[2] https://stackabuse.com/map-filter-and-reduce-in-python-with-examples/