---
layout: blogpage
title: All about Divide and Conquer
comments: true
tags: LeetCode Note
---


```python
def divide_and_conquer( S ):
    # (1). Divide the problem S into a set of subproblems {S1, S2, ... Sn}.
    [S1, S2, ... Sn] = divide(S)

    # (2). Solve the subproblem recursively,
    #   obtain the results of subproblems as [R1, R2... Rn].
    rets = [divide_and_conquer(Si) for Si in [S1, S2, ... Sn]]
    [R1, R2,... Rn] = rets

    # (3). combine the results from the subproblems.
    #   and return the combined result.
    return combine([R1, R2,... Rn])
```

## Validate Binary Search Tree ##

Sometimes, tree related problems can be solved using divide-and-conquer algorithms.
