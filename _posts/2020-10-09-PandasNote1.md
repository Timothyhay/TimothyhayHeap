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

Oytput:

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