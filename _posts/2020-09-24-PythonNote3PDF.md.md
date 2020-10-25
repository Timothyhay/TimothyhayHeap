---
layout: blogpage
title: 没想到吧！Gaussian PDF 的 Vectorization，已经做好了！
comments: true
tags: Skill Python
---

话不多说，再请看做5340 ASS3的作业代码：

```python
emission_list = []
for obid in range(len(x_list)):
    # First fill out Emission probabilities - p(x|z, phi)
    observation = x_list[obid]
    eplist = []

    # Use scipy.stats.norm.pd to calculate Gaussian PDF
    # Alert! Instead of calculate one by one, this function can feed in array instead of scalar
    eplist = scipy.stats.norm.pdf(observation[:, None], phi['mu'], phi['sigma'])

    emission_list.append(eplist)
```

在计算 Gaussian probability density functions 的时候，如果调用`scipy.stats.norm.pdf()`方法，它的参数，`x`，`mu`，`sigma`都可以是向量！恰当处理可以省去所有循环。

一开始我发现这一点的时候，是这样写的：

```python
    # for x in observation:
    #     # Use scipy.stats.norm.pd to calculate Gaussian PDF
    #     p = scipy.stats.norm.pdf(x, phi['mu'], phi['sigma'])
    #     eplist.append(p)
```
但实际上，仅仅针对`x`的对 Gaussian PDF 的多次调用也会造成非常恐怖的性能开销。

这个作业要求整个算法在20s之内跑完，我在向量化处理之内能跑70s.. 化成下述做法之后缩短到了20s。直觉告诉我，没有理由不能一起把`x`也一起向量化。于是在我写出最终结果之后，算法用时提升到了13s。

酷！

要实现把一组输入`x`分别对不同的`mu`，`sigma`调用 Gaussian PDF，使用`observation[:, None]`的格式将其变为一个长度为数组元素个数的列向量即可。

一个简单例子演示这个取法的效果：

	np.arange(5).shape
	Out[7]: (5,)

	np.arange(5)[:, None].shape
	Out[8]: (5, 1)


	np.arange(5)[:, None]
	Out[9]: 
	array([[0],
	       [1],
	       [2],
	       [3],
	       [4]])

酷中酷！