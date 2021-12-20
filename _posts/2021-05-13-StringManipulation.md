---
layout: blogpage
title: Manipulation of String 
comments: true
tags: LeetCode
---

Note about some classic String problems.

## Basic knowledge ##

- Check null string in Python 

```python
string str
if(str.empty()) //成立则为空
```

Don't use `str==NULL` to check:

NULL只用于指针的比较和赋值；而string是类，传参进函数时str调用默认的构造函数已经初始化了。并且str已经是对象了，它不可能为NULL。




## Longest Common Substring / 最长公共子串 ##

Create a 2D array with length of `len(s1)+1` and `len(s2)+1`. `+1` for convience of avoiding index out of range when letting `charCount[iid][jid] = charCount[iid-1][jid-1] + 1`.

```python
def getLongestCommonSubstring(s1, s2):
    # Array for DP
    # rather than charCount = [[0 for _ in range(len(s1) + 1)] for _ in range(len(s2) + 1)]
    charCount = [[0 for _ in range(len(s2) + 1)] for _ in range(len(s1) + 1)]
    
    # Max LCS length
    maxlength = 0
    # End of the LCS (when enumerate from 1, this ptr should be the index of end char +1)
    endptr = 0
    # enumerate from 1
    for iid, i in enumerate(s1, 1):
        for jid, j in enumerate(s2, 1):
            if i == j:
                charCount[iid][jid] = charCount[iid-1][jid-1] + 1
                if charCount[iid][jid] > maxlength:
                    maxlength = charCount[iid][jid]
                    endptr = iid
                    
    # Return LCS and the length
    return s1[endptr - maxlength: endptr], maxlength
```

When char from s1 == char from s2, note in the charCount matrix and move the ptr to the lower right corner of the matrix - by `charCount[iid][jid] = charCount[iid-1][jid-1] + 1`.

    charCount
    Out[1]:
    [[0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 2, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 2, 0],
    [0, 0, 0, 0, 0, 3]]


