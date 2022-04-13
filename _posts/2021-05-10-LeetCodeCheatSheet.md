---
layout: blogpage
title: LeetCode 用 Cheatsheet
comments: true
tags: Python LeetCode
---

## 定义一个装满0的定长数组 

    b = [0 for _ in range(N)]



Use Data Structure in Python!

## Stack / 使用普通列表实现栈

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

## deque / 用deque双端队列做队列的事

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


## Priority Queue / 优先队列

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

        # Get Top
        q.queue[0]

e.g. 

    def maxEvents(self, events: List[List[int]]) -> int:
        
        class Event(object):
            def __init__(self, date):
                self.date = date
            def __lt__(self, other):
                if self.date[1] == other.date[1]:
                    return self.date[0] < other.date[0]
                else:
                    return self.date[1] < other.date[1]
        events.sort(reverse=False)  
        day = events[0][0]
        finday = events[-1][1]
        evptr = 0
        attend = 0
        from queue import PriorityQueue
        pq = PriorityQueue()
        while evptr < len(events) or not pq.empty():
            if pq.empty():
                day = events[evptr][0]
            while not pq.empty() and pq.queue[0].date[1] < day:
                pq.get()
            # print(evptr, pq.queue, day)
            while evptr < len(events) and events[evptr][0] <= day and events[evptr][1] >= day:
                pq.put(Event(events[evptr]))
                evptr += 1

            if not pq.empty() and pq.queue[0].date[1] >= day:
                a = pq.get()
                # print("attend", a)
                attend += 1

            day += 1

        return attend

## Bit Operation / 位运算 ##

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

# Boolean Algebra

XOR异或，NOR或非，NAND与非，XNOR异或非

## XOR

XOR = a⊕b = (¬a ∧ b) ∨ (a ∧¬b)

若x是二进制数0101，y是二进制数1011；
则x⊕y=1110
只有在两个比较的位不同时其结果是1，否则结果为0
即“两个输入相同时为0，不同则为1”。

| a | b | a⊕b |
|---|---|-----|
| 0 | 0 | 0   |
| 0 | 1 | 1   |
| 1 | 0 | 1   |
| 1 | 1 | 0   |

- p.s. reduce() 的用法

函数将一个数据集合（链表，元组等）中的所有数据进行下列操作：用传给 reduce 中的函数 function（有两个参数）先对集合中的第 1、2 个元素进行操作，得到的结果再与第三个数据用 function 函数运算，最后得到一个结果。

在 Python 3 需要import! 
    from functools import reduce

```python
from functools import reduce

def add(x, y) :            # 两数相加
    return x + y
sum1 = reduce(add, [1,2,3,4,5])   # 计算列表和：1+2+3+4+5
sum2 = reduce(lambda x, y: x+y, [1,2,3,4,5])  # 使用 lambda 匿名函数
print(sum1)
print(sum2)
```

To call `operator.xor` in python:

    reduce(operator.xor, nums)

- String Opr. / 字符串操作

s = "String"

s.strip(rm) 删除s字符串中开头、结尾处，位于 rm删除序列的字符

s.lstrip(rm) 删除s字符串中开头处，位于 rm删除序列的字符

s.rstrip(rm) 删除s字符串中结尾处，位于 rm删除序列的字符

rm为空时，默认删除空白符（包括'\n', '\r', '\t', ' ')

e.g.

    >>> '   spacious   '.strip()
    'spacious'
    >>> 'www.example.com'.strip('cmowz.')
    'example'


- x = ord("h")

The ord() function returns the number representing the unicode code of a specified character.

    curr_digit = ord(s[index]) - ord('0')

    # Overflow
    if num > MAX_NUM // 10 or (num == MAX_NUM // 10 and curr_digit > 7): # here we do the check before adding current digit
            return MAX_NUM if sign == 1 else MIN_NUM


## Formatting Output in Python

相对基本格式化输出采用‘%’的方法，format()功能更强大，该函数把字符串当成一个模板，通过传入的参数进行格式化，并且使用大括号‘{}’作为特殊字符代替‘%’

（1）不带编号，即“{}”

（2）带数字编号，可调换顺序，即“{1}”、“{2}”

（3）带关键字，即“{a}”、“{tom}”


    >>> print('{} {}'.format('hello','world'))  # 不带字段
    hello world

    >>> print('{0} {1}'.format('hello','world'))  # 带数字编号
    hello world

    >>> print('{0} {1} {0}'.format('hello','world'))  # 打乱顺序
    hello world hello

    >>> print('{1} {1} {0}'.format('hello','world'))
    world world hello

    >>> print('{a} {tom} {a}'.format(tom='hello',a='world'))  # 带关键字
    world hello world

- 格式转换

'b' - 二进制。将数字以2为基数进行输出。
'c' - 字符。在打印之前将整数转换成对应的Unicode字符串。
'd' - 十进制整数。将数字以10为基数进行输出。
'o' - 八进制。将数字以8为基数进行输出。
'x' - 十六进制。将数字以16为基数进行输出，9以上的位数用小写字母。
'e' - 幂符号。用科学计数法打印数字。用'e'表示幂。
'g' - 一般格式。将数值以fixed-point格式输出。当数值特别大的时候，用幂形式打印。
'n' - 数字。当值为整数时和'd'相同，值为浮点数时和'g'相同。不同的是它会根据区域设置插入数字分隔符。
'%' - 百分数。将数值乘以100然后以fixed-point('f')格式打印，值后面会有一个百分号。

    >>> print('{0:b}'.format(3))
    11
    >>> print('{:c}'.format(20))
    
    >>> print('{:d}'.format(20))
    20
    >>> print('{:o}'.format(20))
    24
    >>> print('{:x}'.format(20))
    14
    >>> print('{:e}'.format(20))
    2.000000e+01
    >>> print('{:g}'.format(20.1))
    20.1
    >>> print('{:f}'.format(20))
    20.000000
    >>> print('{:n}'.format(20))
    20
    >>> print('{:%}'.format(20))
    2000.000000%


## 进制转换

↓	2进制	8进制	10进制	16进制
2进制	-	bin(int(n,8))	bin(int(n,10))	bin(int(n,16))
8进制	oct(int(n,2))	-	oct(int(n,10))	oct(int(n,16))
10进制	int(n,2)	int(n,8)	-	int(n,16)
16进制	hex(int(n,2))	hex(int(n,8))	hex(int(n,10))	-


n=int(input())
print(bin(n))