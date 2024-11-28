---
layout: blogpage
title: 到底应该怎么评价模型的好坏！！
comments: true
tags: Note Skill
---

真是个让人头大的问题。对不同类型的模型有各种不同的技巧，总之我来做一点笔记。


在完成模型训练到生产环境使用前，应该考虑将原始数据中的一部分作为训练数据、另一部分作为测试数据。使用训练数据训练模型，再用测试数据看好坏。即通过测试数据判断模型好坏，然后再不断对模型进行修改。

## 一些通用方法


### Metrics

This section describes the metrics returned for the specific types of models supported for use with Evaluate Model:

- classification models
- regression models
- clustering models

### Metrics for classification models

#### Confusion Matrix / 混淆矩阵:


TN：真实值是0，预测值也是0，即我们预测是negative，预测正确了。

FP：真实值是0，预测值是1，即我们预测是positive，但是预测错误了。

FN：真实值是1，预测值是0，即我们预测是negative，但预测错误了。

TP：真实值是1，预测值是1，即我们预测是positive，预测正确了。

The following metrics are reported when evaluating binary classification models.

**Accuracy** measures the goodness of a classification model as the proportion of true results to total cases.

**Precision** is the proportion of true results over all positive results. Precision = TP/(TP+FP)

**Recall** is the fraction of the total amount of relevant instances that were actually retrieved. Recall = TP/(TP+FN)

**F1 score** is computed as the weighted average of precision and recall between 0 and 1, where the ideal F1 score value is 1.

> 一些思考：

1、是否只关注accuracy？

虽然准确率可以判断总的正确率，但是如果存在样本不均衡的情况下，就不能使用accuracy来进行衡量了。比如，一个总样本中，正样本占90%,负样本占10%，那么只需要将所有样本预测为正，就可以拿到90%的准确率，而这显然是无意义的。正因为如此，我们就需要精准率和召回率了。

2、那么精准率和召回率这两个指标如何解读？如果对于一个模型来说，两个指标一个低一个高该如何取舍呢？

> 精准率的应用场景：预测癌症。医生预测为癌症，患者确实患癌症的比例。
> 
> 召回率的应用场景：网贷违约率，相对好用户，我们更关心坏用户。召回率越高，代表实际坏用户中被预测出来的概率越高。
> 
> 至于两个指标如何使用，需要看具体场景，实际上这两个指标是互斥的，一个高，必有另一个低。


AUC measures the area under the curve plotted with true positives on the y axis and false positives on the x axis. This metric is useful because it provides a single number that lets you compare models of different types. AUC is classification-threshold-invariant. It measures the quality of the model's predictions irrespective of what classification threshold is chosen.

Metrics for regression models
The metrics returned for regression models are designed to estimate the amount of error. A model is considered to fit the data well if the difference between observed and predicted values is small. However, looking at the pattern of the residuals (the difference between any one predicted point and its corresponding actual value) can tell you a lot about potential bias in the model.

The following metrics are reported for evaluating regression models.

Mean absolute error (MAE) measures how close the predictions are to the actual outcomes; thus, a lower score is better.

Root mean squared error (RMSE) creates a single value that summarizes the error in the model. By squaring the difference, the metric disregards the difference between over-prediction and under-prediction.

Relative absolute error (RAE) is the relative absolute difference between expected and actual values; relative because the mean difference is divided by the arithmetic mean.

Relative squared error (RSE) similarly normalizes the total squared error of the predicted values by dividing by the total squared error of the actual values.

Coefficient of determination, often referred to as R2, represents the predictive power of the model as a value between 0 and 1. Zero means the model is random (explains nothing); 1 means there is a perfect fit. However, caution should be used in interpreting R2 values, as low values can be entirely normal and high values can be suspect.

Metrics for clustering models
Because clustering models differ significantly from classification and regression models in many respects, Evaluate Model also returns a different set of statistics for clustering models.

The statistics returned for a clustering model describe how many data points were assigned to each cluster, the amount of separation between clusters, and how tightly the data points are bunched within each cluster.

The statistics for the clustering model are averaged over the entire dataset, with additional rows containing the statistics per cluster.

The following metrics are reported for evaluating clustering models.

The scores in the column, Average Distance to Other Center, represent how close, on average, each point in the cluster is to the centroids of all other clusters.

The scores in the column, Average Distance to Cluster Center, represent the closeness of all points in a cluster to the centroid of that cluster.

The Number of Points column shows how many data points were assigned to each cluster, along with the total overall number of data points in any cluster.

If the number of data points assigned to clusters is less than the total number of data points available, it means that the data points could not be assigned to a cluster.

The scores in the column, Maximal Distance to Cluster Center, represent the max of the distances between each point and the centroid of that point's cluster.

If this number is high, it can mean that the cluster is widely dispersed. You should review this statistic together with the Average Distance to Cluster Center to determine the cluster's spread.

The Combined Evaluation score at the bottom of each section of results lists the averaged scores for the clusters created in that particular model.



Reference

[1] https://docs.microsoft.com/en-us/azure/machine-learning/component-reference/evaluate-model
[2] https://zhuanlan.zhihu.com/p/110950916
