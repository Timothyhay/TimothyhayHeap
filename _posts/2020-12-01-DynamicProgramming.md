---
layout: blogpage
title: All about Dynamic Programming
comments: true
tags: LeetCode Note
---

> Those who cannot remember the past are condemned to repeat it.
>
> —— George Santayana


<p>Usually, solving and fully understanding a dynamic programming problem is a 4 step process:</p>

<ol>
<li>Start with the recursive backtracking solution</li>
<li>Optimize by using a memoization table (top-down[2] dynamic programming)</li>
<li>Remove the need for recursion (bottom-up dynamic programming)</li>
<li>Apply final tricks to reduce the time / memory complexity </li>
</ol>

<p> All solutions presented below produce the correct result, but they differ in run time and memory requirements. </p>

<!--
<p>
    As the problem has an <strong>optimal substructure</strong>, it is natural to cache intermediate results.
    We ask the question dp(i, j): does text[i:] and pattern[j:] match?
    We can describe our answer in terms of answers to questions involving smaller strings.
</p>
-->

<h2>Take <code>55. Jump Game</code> for example:</h2>

<p>Most of the contents come from its LeetCode solution[1].</p>

<br />

### Approach 1: Backtracking ###
```cpp
// TLE - TC O(2^n) - Approach 1: Backtracking
class Solution {
public:
    bool canJumpFromPosition(vector<int>& nums, int pos){
        if (pos == nums.size() - 1)
            return true;
        for (int i = pos + 1; i <= pos + nums[pos] && i < nums.size(); ++i) {
            if(canJumpFromPosition(nums, i))
                return true;
        }
        return false;
    }

    bool canJump(vector<int>& nums) {
        return canJumpFromPosition(nums, 0);
    }
};
```
Time complexity : O(2^n). There are 2^n (upper bound) ways of jumping from the first position to the last, where nn is the length of array nums.

Space complexity : O(n). Recursion requires additional memory for the stack frames.

<br />

### Approach 2: Dynamic Programming Top-down ###

Top-down Dynamic Programming can be thought of as optimized backtracking. It relies on the observation that once we determine that a certain index is good / bad, this result will never change. This means that we can store the result and not need to recompute it every time.

```cpp
// TLE - TC O(n^2) -  Approach 2: Dynamic Programming Top-down
class Solution {
public:
    map<int, int> memo;
    bool canJumpFromPosition(vector<int>& nums, int pos){
        if (memo[pos] == -1) {
            if (pos == nums.size() - 1) {
                memo[pos] = 1;
                return true;
            }
            for (int i = pos + 1; i <= pos + nums[pos] && i < nums.size(); ++i) {
                if (canJumpFromPosition(nums, i)){
                    memo[pos] = 1;
                    return true;
                }

            }
            memo[pos] = -1;
            return false;
        } else if (memo[pos] == 1)
            return true;
        else
            return false;
    }

    bool canJump(vector<int>& nums) {
        for (int i = 0; i < nums.size(); ++i) {
            // -1: Unknown, 1: True, 0: False
            memo[i] = -1;
        }

        return canJumpFromPosition(nums, 0);
    }
};
```

Time complexity : O(n^2). For every element in the array, say i, we are looking at the next nums[i] elements to its right aiming to find a `True` index. nums[i] can be at most nn, where nn is the length of array nums.

Space complexity : O(2n) = O(n). First n originates from recursion. Second n comes from the usage of the memo table.

<br />

### Approach 3: Dynamic Programming Bottom-up ###

Top-down to bottom-up conversion is done by eliminating recursion. In practice, this achieves better performance as we no longer have the method stack overhead and might even benefit from some caching. More importantly, this step opens up possibilities for future optimization. The recursion is usually eliminated by trying to reverse the order of the steps from the top-down approach.

```cpp
// TLE - Approach 3: Dynamic Programming Bottom-up
class Solution {
public:
    bool canJump(vector<int>& nums) {
        map<int, int> memo;
        for(int i = 0; i < nums.size(); ++i){
            // -1: Unknown, 1: True, 0: False
            memo[i] = -1;
        }
        memo[nums.size()-1] = 1;
        for(int i = nums.size() - 2; i >= 0; --i){
            for (int j = i + 1; j <= i + nums[i] && j < nums.size(); ++j) {
                if (memo[j] == 1){
                    memo[i] = 1;
                    break;
                }
            }
        }
        return memo[0] == 1;
    }

};
```

We start from the right of the array, every time we will query a position to our right, that position has already be determined as being GOOD or BAD. 

This means we don't need to recurse anymore, as we will always hit the memo table

<br />

### Approach 4: Dynamic Programming Bottom-up ###

```cpp
// AC - Approach 4: Greedy
class Solution {
public:
    bool canJump(vector<int>& nums) {
        int lastJmpPos = nums.size() - 1;
        for (int i = nums.size() - 2; i >= 0; --i){
            if (nums[i] + i >= lastJmpPos)
                lastJmpPos = i;
        }
        return lastJmpPos == 0;
    }

};
```

Only Need: 

Time complexity : O(n). We are doing a single pass through the nums array, hence nn steps, where nn is the length of array nums.
     
Space complexity : O(1). We are not using any extra memory.

<br />

## Other Examples ##

When need to consider a buy-sell situation, where sell action must be after the buy action, use a `for` loop with `if-else` stmt to avoid a O(n^2) check.

```cpp
// 121. Best Time to Buy and Sell Stock
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int minprice = INT_MAX, maxprofit = 0;
        for (int i = 0; i < prices.size(); ++i) {
            if (prices[i] < minprice )
                minprice = prices[i];
            if (maxprofit < prices[i] - minprice)
                maxprofit = prices[i] - minprice;
        }
        return maxprofit;
    }
};
```

`#include <climits>` when you need `INT_MAX`. But here it can actually be replaced by `prices[0]`.






##  Reference ##
[1] jump-game/solution - https://leetcode.com/problems/jump-game/solution/