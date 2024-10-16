---
layout: blogpage
title: Perplexity & Importance Sampling
comments: true
tags: Note 
---



## Importance Sampling for Perplexity Evaluation ##

### Perplexity ###

Perplexity The standard evaluation metric for
language models is perplexity:

<!-- 
$$
\mathrm{PPL}=\exp \left(-\frac{1}{T} \sum_{t=1}^{T} \log p\left(x_{t} \mid x_{<t}\right)\right)
$$



where \\( p(\boldsymbol{x_{t}}, \boldsymbol{x_{<t}}) \\) is the marginal likelihood of the token \\(x_{t} \\) conditioned on the previous tokens \\(x_{<t} \\) . By the chain rule of probabilities \\( p(\boldsymbol{x})=\prod_{t=1}^{T} p\left(x_{t} \mid x_{<t}\right) . \\) Perplexity can be intractable to compute for latent language models since it requires marginalizing out the latent variable (e.g., \\( p(\boldsymbol{x})=\sum_{z} p(\boldsymbol{x}, \boldsymbol{z}) \\)  whose state space is often exponential in the length of the text.

-->

困惑度作为语言模型的一种衡量标准，本身在衡量隐式语言模型时由于要边缘化隐变量，会因为该变量指数倍文本长度的状态空间而无法计算，因此我们需要改用抽样的方法。

As a measure of language model, perplexity itself cannot be calculated due to the marginalization of latent variables when measuring implicit language models. Because the state space of this variable is exponential times of the text length, so we need to use the sampling method instead.


This is usually done by splitting the dataset into two parts: one for training, the other for testing.  
For LDA, a test set is a collection of unseen documents wd, and the model is described by the topic matrix Φ and the hyperparameter α for topic-distribution of documents. The LDA parameters Θ is not taken into consideration as it represents the topic-distributions for the documents of the training set, and can therefore be ignored to compute the likelihood of unseen documents.

对于 LDA 来说，测试集是未知文档 wd 的集合，模型由主题矩阵 Φ 和文档主题分布的超参数 α 描述。 LDA 参数 Θ 未考虑在内，因为它表示训练集文档的主题分布，因此计算没有见过的文档的可能性的时候可以忽略。

Therefore, we need to evaluate the log-likelihood of a set of unseen documents \\( \boldsymbol{w}_{d} \\) given the topics Φ and the hyperparameter α for topic-distribution θd of documents. Likelihood of unseen documents can be used to compare models; higher likelihood implying a better model.


因此，我们需要在给定主题 Φ 和文档主题分布 θd 的超参数 α 的情况下评估一组未知文档 <span>\\( \boldsymbol{w}_{d} \\)<span> 的对数似然。 未见文档的可能性可用于比较模型；更高的可能性意味着更好的模型。上文的对数似然在LDA下可以写成：

<!-- 

$$ \log p\left(x_{t} \mid x_{<t}\right) = \log p(\boldsymbol{w} \mid \boldsymbol{\Phi}, \alpha)=\sum_{d} \log p\left(\boldsymbol{w}_{d} \mid \boldsymbol{\Phi}, \alpha\right) $$

-->

which is a decreasing function of the log-likelihood of the unseen documents \\( \boldsymbol{w}_{d} \\); the lower the perplexity, the better the model.



### Importance Sampling  ###

Existing approaches instead use importance sampling (Kahn, 1950) to estimate an approximate marginal probability:

<!-- 

$$
\hat{p}(\boldsymbol{x})=\frac{1}{K} \sum_{k=1}^{K} \frac{p\left(\boldsymbol{x}, \boldsymbol{z}_{k}\right)}{q\left(\boldsymbol{z}_{k}\right)}
$$

-->

where q(z) is an arbitrary proposal distribution and
z1, . . . , zK ∼ q(z). It is well known that \\(\hat{p}(x)\\) is an unbiased estimator:

<!-- 

$$
\mathbb{E}_{z_{k} \sim q(z)}[\hat{p}(x)]=p(x)
$$

-->


provided that q(z) > 0 whenever p(z) > 0.

这里的提议分布q(z)应当是方便采样的简单分布。


### Perplexity is not strongly correlated to human judgment

[Chang09] have shown that, surprisingly, predictive likelihood (or equivalently, perplexity) and human judgment are often not correlated, and even sometimes slightly anti-correlated.

They ran a large scale experiment on the Amazon Mechanical Turk platform. For each topic, they took the top five words (ordered by frequency p(w|k)=ϕkw) of that topic and added a random sixth word. Then, they presented these lists of six words to participants asking them to identify the intruder word.

If every participant could identify the intruder, then we could conclude that the topic is good at describing an idea. If on the other hand, many people identified one of the topic top five word as the intruder, it means that they could not see the logic in the association of words, and we can conclude the topic was not good enough.

It's important to understand what this experiment is proving. The result proves that, given a topic, the five words that have the largest frequency p(w|k)=ϕkw withing their topic are usually not good at describing one coherent idea; at least not good enough to be able to recognize an intruder.

## Reference ##

https://ucinlp.github.io/files/papers/impsample-acl20.pdf

http://qpleple.com/perplexity-to-evaluate-topic-models/

http://dirichlet.net/pdf/wallach09evaluation.pdf

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>