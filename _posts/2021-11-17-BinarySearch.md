---
layout: blogpage
title: Questions about Binary Search
comments: true
tags: LeetCode
---

BinarySearch is a O(logN) searching method, but need to be applied in ordered array.

The Original BinarySearch, and finding the First & Last index of target element:

```python

# Original Method - Can only handle for array of elements occur once
def binSearchOri(nums, target):
    start = 0
    end = len(nums) - 1
    while start <= end:
        mid = start + (end - start) // 2
        if nums[mid] > target:
            end = mid - 1
        elif nums[mid] < target:
            start = mid + 1
        else:
            return mid
    return -1

# binSearchOri([5,7,7,8,8,10], 8)
# Out[3]: 4
# binSearchOri([5,7,7,8,8,8,8,8,8,10], 8)
# Out[4]: 4
# binSearchOri([5,7,7,8,8,8,8,8,8,8,8,8,8,10], 8)
# Out[5]: 6
# binSearchOri([5,7,7,8,10], 8)
# Out[6]: 3

# Output the first index of the same matched elements
def binSearchFirst(nums, target):
    start = 0
    end = len(nums) - 1
    while start <= end:
        mid = start + (end - start) // 2
        if nums[mid] >= target:
            end = mid - 1
        elif nums[mid] < target:
            start = mid + 1
    if nums[start] != target:
        return -1
    else:
        return start

# binSearchFirst([5,7,7,8,10], 8)
# Out[8]: 3
# binSearchFirst([5,7,7,8,8,8,8,8,8,8,8,8,8,10], 8)
# Out[9]: 3
# binSearchFirst([5,7,7,8,8,8,8,8,8,10], 8)
# Out[10]: 3
# binSearchFirst([5,7,7,8,8,10], 8)
# Out[11]: 3

# Output the last index of the same matched elements
def binSearchLast(nums, target):
    start = 0
    end = len(nums) - 1
    while start <= end:
        mid = start + (end - start) // 2
        if nums[mid] <= target:
            start = mid + 1
        else:
            end = mid - 1
    if nums[end] != target:
        return -1
    else:
        return end

# binSearchLast([5,7,7,8,8,8,8,8,8,10], 8)
# Out[19]: 8
# binSearchLast([5,7,7,8,8,10], 8)
# Out[20]: 4
# binSearchLast([5,7,7,8,8,8,8,8,10], 8)
# Out[21]: 7
# binSearchLast([5,7,7,8,8,8,8,10], 8)
# Out[22]: 6
```