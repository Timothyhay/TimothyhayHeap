---
layout: blogpage
title: Python关于list.remove()方法的小坑
comments: true
tags: Skill Python
---
话不多说，请看做5340的作业代码：

    # Alert! If use original factorlist as outer loop, after removal, the elem index in the list changed
    # but the loop counter does not - will cause step over matching some factors
    # For every factor, find a cluque which factor is a subset of
    for factor in factorlist[:]:  # Every matching obtain a copy of factorlist
        fvar = factor.var
        for cid in range(len(cliques)):
            clique = cliques[cid]
            #print("Now matching set_fvar",set(fvar))
            #print("With set_clique", set(clique))
            if set(fvar).issubset(set(clique)):
                # The clique potential is the product of its assigned factors
                clique_factors[cid] = factor_product(clique_factors[cid], factor)
                factorlist.remove(factor)
                break

在外层循环中，如果使用`for factor in factorlist:`的写法，会导致内层处理里移除factorlist元素让所有元素往移除位（向前/数组下标0的位置）移动。而外层循环的计数器依然不变，这会导致循环的过程中跳过某些元素（当有连续需要删除的元素时）。

解决方法如上述代码所示，使用`for factor in factorlist[:]:`在循环过程中临时获取整个`factorlist`的副本，动态更新每次迭代中的数组下标。但是这种方法对较长数组可能影响效率。

此外，也可以使用`for factor in factorlist[::-1]:`，倒序遍历list。这是个非常tricky的做法，因为倒序遍历时移除元素，哪怕集体往前移动，对从后往前走的指针来说次序依然没有改变。

在CS5340 Assignment2里，虽然倒序遍历一样能得到正确中间结果，但是和正序的结果可能有些不同。好看起见，我还是选择了正序~

