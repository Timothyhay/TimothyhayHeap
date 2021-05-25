---
layout: blogpage
title: LeetCode 用 Cheatsheet
comments: true
tags: Python
---

## 定义一个装满0的定长数组 

b = [0 for _ in range(N)]



Use Data Structure in Python!

## 使用普通列表实现栈

>>> s = []
>>> s.append('eat')
>>> s.append('sleep')
>>> s.append('code')

>>> s
['eat', 'sleep', 'code']

>>> s.pop()
'code'
>>> s.pop()
'sleep'
>>> s.pop()
'eat'

>>> s.pop()
IndexError: "pop from empty list"


    from collections import deque

## 用deque双端队列做队列的事

deque 类实现了一个双端队列，支持在O(1)时间（非均摊）中从任一端添加和删除元素。由于deque 支持从两端添加和移除元素，因此既可用作队列也可用作栈。

Python 的deque 对象以双向链表实现。这为插入和删除元素提供了出色且一致的性能，但是随机访问位于栈中间元素的性能很差，耗时为O(n)。

>>> from collections import deque
>>> q = deque()
>>> q.append('eat')
>>> q.append('sleep')
>>> q.append('code')

>>> q
deque(['eat', 'sleep', 'code'])

>>> q.popleft()
'eat'
>>> q.popleft()
'sleep'
>>> q.popleft()
'code'

>>> q.popleft()


## 优先队列

from queue import PriorityQueue

q = PriorityQueue()

q.put((2, 'code'))
q.put((1, 'eat'))
q.put((3, 'sleep'))

while not q.empty():
    next_item = q.get()
    print(next_item)

    # 结果：
    # (1, 'eat')
    # (2, 'code')
    # (3, 'sleep')


## 位运算 ##

    x << y

Returns x with the bits shifted to the left by y places (and new bits on the right-hand-side are zeros). This is the same as multiplying x by 2**y.


    x >> y

Returns x with the bits shifted to the right by y places. This is the same as //'ing x by 2**y.


    x & y

Does a "bitwise and". Each bit of the output is 1 if the corresponding bit of x AND of y is 1, otherwise it's 0.


    x | y
    
Does a "bitwise or". Each bit of the output is 0 if the corresponding bit of x AND of y is 0, otherwise it's 1.


    ~ x
Returns the complement of x - the number you get by switching each 1 for a 0 and each 0 for a 1. This is the same as -x - 1.


    x ^ y
Does a "bitwise exclusive or". Each bit of the output is the same as the corresponding bit in x if that bit in y is 0, and it's the complement of the bit in x if that bit in y is 1.

