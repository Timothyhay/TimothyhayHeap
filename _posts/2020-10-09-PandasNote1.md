---
layout: blogpage
title: 作业里积累的Pandas笔记
comments: true
tags: Skill Note Python
---

## From CS5228 Knowledge Discovery and Data Mining Assignment 2 ##

## 关于条件判断 - 这两个表达式得到的真值不一样： ##

Input:

	print(((pima == 0) | (pima.isnull())).sum(axis=0))

Output:

	Pregnancies                 111
	Glucose                       5
	BloodPressure                36
	SkinThickness               228
	Insulin                     374
	BMI                          11
	DiabetesPedigreeFunction      1
	Age                           0
	Outcome                     500
	dtype: int64

Input:

	print((pima == 0 | pima.isnull()).sum(axis=0))

Output:

	Pregnancies                 111
	Glucose                       5
	BloodPressure                35
	SkinThickness               227
	Insulin                     374
	BMI                          11
	DiabetesPedigreeFunction      0
	Age                           0
	Outcome                     500
	dtype: int64

不把两个表达式都bracket起来的话，DiabetesPedigreeFunction中的NaN项无法被检测到


## 对DataFrame的copy ##

使用`pandas.DataFrame.copy`方法！参数deep=True代表深拷贝。

	DataFrame.copy(deep=True)

deep=false时相当于引用，原值改变时复制的结果随着改变。

	data = DataFrame.copy(deep=False)
	# Actually equals to:
	data = DataFrame


## pandas.cut ##

pandas.cut(x, bins, right=True, labels=None, retbins=False, precision=3, include_lowest=False, duplicates='raise', ordered=True)


**Examples**

Discretize into three equal-sized bins.

	pd.cut(np.array([1, 7, 5, 4, 6, 3]), 3)
	
	[(0.994, 3.0], (5.0, 7.0], (3.0, 5.0], (3.0, 5.0], (5.0, 7.0], ...
	Categories (3, interval[float64]): [(0.994, 3.0] < (3.0, 5.0] ...
	pd.cut(np.array([1, 7, 5, 4, 6, 3]), 3, retbins=True)
	
	([(0.994, 3.0], (5.0, 7.0], (3.0, 5.0], (3.0, 5.0], (5.0, 7.0], ...
	Categories (3, interval[float64]): [(0.994, 3.0] < (3.0, 5.0] ...
	array([0.994, 3.   , 5.   , 7.   ]))

Discovers the same bins, but assign them specific labels. Notice that the returned Categorical’s categories are labels and is ordered.
	
	pd.cut(np.array([1, 7, 5, 4, 6, 3]),
	       3, labels=["bad", "medium", "good"])
	['bad', 'good', 'medium', 'medium', 'good', 'bad']
	Categories (3, object): ['bad' < 'medium' < 'good']


In Assignment 2, I did like this：

- Normal:  <140 mg/dl of glucose,
- Prediabetes:  ≥140  but  <200 mg/dl of glucose,
- Diabetes:  >200 mg/dl of glucose

To do this, setting inf = float('inf'). Use pd.cut with bins of [-inf,139.99,199.99,inf] and labels of ['Normal', 'Prediabetes', 'Diabetes'].

	# 1) Transform continuous variables into categorical variables
	inf = float('inf')
	pima1 = pima.copy(deep=True)
	
	pima1.Glucose = pd.cut(pima.Glucose, [-inf, 139.99, 199.99, inf], labels=['Normal', 'Prediabetes', 'Diabetes'])

Reference:
----------
[1]pandas.cut - https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.cut.html