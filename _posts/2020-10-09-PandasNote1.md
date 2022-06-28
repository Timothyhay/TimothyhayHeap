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



## Ex. Pandas Tutorial ##

### Series

Pandas Series 类似表格中的一个**列（column）**，类似于一维数组，可以保存任何数据类型。

Series 由索引（index）和列组成，函数如下：

	pandas.Series( data, index, dtype, name, copy)

- data：一组数据(ndarray 类型)。
- index：数据索引标签，如果不指定，默认从 0 开始。
- dtype：数据类型，默认会自己判断。
- name：设置名称。
- copy：拷贝数据，默认为 False。

基本用法：

```py

a = [1, 2, 3]
myvar = pd.Series(a)
# 不适用索引
print(myvar[1])

a = ["Google", "Runoob", "Wiki"]
myvar = pd.Series(a, index = ["x", "y", "z"])

# 使用索引
print(myvar["y"])

# 使用键值对初始化，key 成为 index
sites = {1: "Google", 2: "Runoob", 3: "Wiki"}
myvar = pd.Series(sites)
```

### DataFrame

	pandas.DataFrame( data, index, columns, dtype, copy)

- data：一组数据(ndarray、series, map, lists, dict 等类型)。
- index：索引值，或者可以称为行标签。
- columns：列标签，默认为 RangeIndex (0, 1, 2, …, n) 。
- dtype：数据类型。
- copy：拷贝数据，默认为 False。




Reference:
----------
[1]pandas.cut - https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.cut.html