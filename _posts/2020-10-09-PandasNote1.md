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

使用 `list` 或者 `dict` 初始化:

```py
# Use list to init
data = [['Google', 10], ['Runoob', 12], ['Wiki', 13]]

# Use dict to init, key will be the field name
data = {'Site': ['Google', 'Runoob', 'Wiki'], 'Age': [10, 12, 13]}

df0 = pd.DataFrame(data, columns=['Site', 'Age'])

data = [{'a': 1, 'b': 2}, {'a': 5, 'b': 10, 'c': 20}]  # This will make row 0 col 'c' NaN
```


使用 `loc` 返回行:

```py
# 使用 loc 属性返回指定行的数据，如果没有设置索引，第一行索引为 0，第二行索引为 1，以此类推：
days = {
    "calories": [420, 380, 390],
    "duration": [50, 40, 45]
}

df = pd.DataFrame(days, index=["day1", "day2", "day3"])

# 指定索引
print(df.loc["day2"])
print(df0.loc[1])
# 返回结果其实就是一个 Pandas Series 数据。

# 也可以返回多行数据，使用 [[ ... ]] 格式，... 为各行的索引，以逗号隔开：
print("Multi-line:")
print(df0.loc[[0, 1]])
# 返回结果其实就是一个 Pandas DataFrame 数据。

# loc[] 接受两个参数，并以','分隔。第一个位置表示行，第二个位置表示列
print(df0.loc[1, 'Age'])         # 12
print(df0.loc[[0, 1], ['Age']])  # 当然列数组也可以选多列 不止是'Age'
'''
   Age
0   10
1   12
'''
```
**.iloc[]**

df.iloc[] 只能使用整数索引，不能使用标签索引，通过整数索引切片选择数据时，前闭后开(不包含边界结束值)。同 Python 和 NumPy 一样，它们的索引都是从 0 开始。

e.g.

```py
print(df.iloc[[1, 3, 5], [1, 3]])
print(df.iloc[1:3, :])
print(df.iloc[:,1:3])

```

一些高级用法：

```py
'''
     Site  Age
0  Google   10
1  Runoob   12
2    Wiki   13
'''
# 取满足条件的条目 ('Age' > 2)
print(df[df['Age'] > 2])
# df['Age'] > 2] 是一个 True/False Series
# 会得到满足的所有行(的所有字段) - 

# 设置索引
df.set_index('school_code')
# 输出表格但是不显示索引
print("DataFrame without index:")
print(df.to_string(index=False))

# 在指定位置(索引为3的列)插入列
date_of_birth = ['15/05/2002','17/05/2002','16/02/1999']
df.insert(loc=3, column='date_of_birth', value=date_of_birth)
```

## concat and Merge

**concat/append - 大体上用于纵向连接，是 df 的直接拼接：**

```py
student_data1 = pd.DataFrame({
        'student_id': ['S1', 'S2', 'S3', 'S4', 'S5'],
         'name': ['Danniella Fenton', 'Ryder Storey', 'Bryce Jensen', 'Ed Bernal', 'Kwame Morin'], 
        'marks': [200, 210, 190, 222, 199]})

student_data2 = pd.DataFrame({
        'student_id': ['S4', 'S5', 'S6', 'S7', 'S8'],
        'name': ['Scarlette Fisher', 'Carla Williamson', 'Dante Morse', 'Kaiser William', 'Madeeha Preston'], 
        'marks': [201, 200, 198, 219, 201]})

# 直接把 df 按行连接， id 可以有重复
print("Join the said two dataframes along rows:")
result_data = pd.concat([student_data1, student_data2])
print(result_data)

# 按列合并结果中会有 6 列，即同名的列 student_id，name，marks
print("\nJoin the said two dataframes along columns:")
result_data = pd.concat([student_data1, student_data2], axis = 1)
# Note: 如果设置了axis = 1(按列合并) 不同的 index 会各占一列 空值会以 NaN 填补

# 添加一行
s6 = pd.Series(['S6', 'Scarlette Fisher', 205], index=['student_id', 'name', 'marks'])
combined_data = student_data1.append(s6, ignore_index = True)
```

注意使用 `concat` 和 `append` 时若让两个df纵向合并在一起， 因为concat()保留了每个子 DataFrame 的 index ， 所以合并之后的DataFrame中， 每个index也会出现两次。

可以通过设置参数中 `ignore_index=False` 来解决这个问题：

	result = pd.concat([df1, df2], ignore_index=True)

这会让 index 变得有序不重复，但是条目中的其他列重复记录不会影响。

**merge 才是真正的join：**

merge()的how参数可以设置DataFrame的四种连接方式：左连接(left), 右连接(right), 外连接(outer), 内连接(inner)。

横向合并left和right两个子DataFrame：

```py

# on='key' 把两个子DataFrame中key列相同的值连接到一行上
result = pd.merge(left, right, on='key')
# on=['key1', 'key2'] 则只会保留 'key1','key2' 相同的值
result = pd.merge(left, right, on=['key1', 'key2'])
# 以上省略how='inner'默认参数 - 实际为 inner join
# 即 key值不完全相同的行便不会显示在最终的结果里 

# 外连接(outer)则会保留两个DataFrame中所有的行：
result = pd.merge(left, right, on=['key1', 'key2'],
                  how='outer')

# 左连接(left)则会保留左边的子DataFrame中所有的行：
result = pd.merge(left, right, on=['key1', 'key2'],
                  how='left')

# 右连接(right)则会保留右边的子DataFrame中所有的行：
result = pd.merge(left, right, on=['key1', 'key2'],
                  how='right')
```

输出行列相关：

```
print(w_a_con.head())
print('Shape of the dataframe: ',w_a_con.shape) # (100, 5)
print('Number of rows: ',w_a_con.shape[0])   # 100
print('Number of column: ',w_a_con.shape[1]) # 5
print("Extract Column Names:")
print(w_a_con.columns)

```


Reference:
----------
[1]pandas.cut - https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.cut.html

[2] merge/concat 区别 - https://zhuanlan.zhihu.com/p/70438557

[3] https://www.w3resource.com/python-exercises/pandas/joining-and-merging/index.php