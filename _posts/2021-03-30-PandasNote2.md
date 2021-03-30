---
layout: blogpage
title: 看一眼Shopee面试分析的数据都长什么样
comments: true
tags: Skill Python
---

卓君给我发了个Shopee的面试题，不得不感叹现在学数学的小朋友要写的代码真是越来越多了。
之前学数据分析的玉米说他想投算法岗，我还不信，现在我有点信了。

这篇文章不算我的Pandas笔记，大概算是发给卓君的代码的详细版注释。希望可以在这里把面试题的流程说得更清楚一些。

首先我们读数据！

```python
# Read data first
# Datafrme user - length: 10855
user = pd.read_excel(r'SQL Test_2020_case_study_1.xlsx', sheet_name=1, engine='openpyxl').dropna(axis=1, how='all')
# Datafrme order - length: 30041
order = pd.read_excel(r'SQL Test_2020_case_study_1.xlsx', sheet_name=2, engine='openpyxl').dropna(axis=1, how='all')
print("Raw Data:", user.head(), order.head(), sep='\n', end='\n')
```

在最近的2.0.1版本的xlrd里，Pandas的`read_excel()`函数已经不支持显式对xlsx的读取了。
解决办法除了降级到1.2.0这种听起来就很奇怪的做法以外，就是另外安装openpyxl，提供参数`engine='openpyxl'`。

这样做会带来空列，我们使用`dropna(axis=1, how='all')`来去掉全部值都为空值的列，如果要对行这样做改成`axis=0`就好。

使用`head(n)`方法可以看一眼前n行Dataframe的内容，如果想看点统计数据可以试试`describe()`方法。


输出结果是：

    Raw Data:
       userid register_time country
    0   10310    2017-03-02      TW
    1   10313    2017-03-20      SG
    2   10323    2017-03-26      TW
    3   10330    2017-05-02      VN
    4   10333    2017-01-10      TH
       orderid  userid   itemid  gmv order_time
    0  1030132   64177  3366770   27 2017-04-24
    1  1030137   10475  6130641   69 2017-02-02
    2  1030147   28286  6770063   87 2017-04-25
    3  1030153   28282  4193426   82 2017-05-11
    4  1030155   64970  8825994   29 2017-03-07
    
    
接着对这两张表格做一下连接。

```python
# Specify param how='outer' to perform outer join
data_outerjoin = pd.merge(order, user, how='outer')
print("Empty Row Count:", data_outerjoin.isna().sum(), sep='\n', end='\n')

# 1 - Show users without any order
result = data_outerjoin['userid'][data_outerjoin.T.isnull().any()]
print(result.count(), "users has no orders.\nNo. \t UserID")
print(result, end='\n')

# Left join
data_leftjoin = pd.merge(order, user, how='left')
```

要做外连接，可以用`pd.merge(order, user, how='outer')`实现。当需要内连接的时候，修改参数`how='inner'`就好。
对表格在左还是右有要求的是左右连接，这里`pd.merge(order, user, how='left')`就是把user连接到左边的order上。

`how`参数的所有取值有 {'left', 'right', 'outer', 'inner', 'cross'}, 默认是 'inner'，内连接。
还可以指定连接用键，具体可以看一眼[官方文档](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.merge.html?highlight=merge#pandas.merge) 。


```python
# Get interval time between order_time and register_time
data_leftjoin["interval"] = data_leftjoin["order_time"] - data_leftjoin["register_time"]

# 2 - Show users has orders before registration
sign = data_leftjoin["interval"] < pd.Timedelta('0 day')
result = data_leftjoin["userid"][sign]
print(result.count(), "users has orders before registration.\nNo. \t UserID")
print(result, end='\n')
```

这里直接用Datetime（日期时间）类型变量相减，会得到Timedelta（时间差）类型变量。Timedelta型变量不能和整型变量对比。
所以判断用户订单时间是否小于注册时间的时候，需要实例化一个`pd.Timedelta('0 day')`的 0天 时间差变量来判断 interval 字段的正负。

这样得到的结果`sign`是和`data_leftjoin`一样长的带30041个bool变量的Series型变量。

长这样：

    0         True
    1         True
    2        False
    3        False
    4         True
             ...  
    30036    False
    30037     True
    30038    False
    30039     True
    30040    False
    Name: interval, Length: 30041, dtype: bool

我们只要用中括号筛选就可以筛选出`sign`为True的条目。

    14969 users has orders before registration.
    No. 	 UserID
    0        64177
    1        10475
    4        64970
    5        37113
    7        28074
             ...  
    30030    50180
    30031    27216
    30033    69791
    30037    43512
    30039    26588

这里用波浪号把`sign`取反，放进'type'字段里，就可以用来表示用户的正常与否啦。

```python
# Tag them as fake users
data_leftjoin["type"] = ~sign
# Show the count of 0s and 1s in Outcome using a bar graph.
sb.countplot(x="type", data=data_leftjoin)
plt.show()
# Result: The number of real and fake users are almost identical
```

后面就是分析数据的过程，用`sb.countplot()`数了数正常和不正常的用户数量，发现居然差不多一样多。
然后我就想，这数据不会是50/50随机生成的吧..

```python
# Generate a 5*5 heatmap plot showing the correlation between any two features
sb.clustermap(data_leftjoin.corr(), annot=True)
plt.show()
# Result: Almost no relationship
```

不死心地看了一眼热力图，嗯好的果然数据之间没有任何关系，连敷衍一点的关系都没有，全是负值。

这种情况也很难出现，实际处理数据的时候，我记得以前处理过一个集邮政编码啥的都能和目标分类有些许联系... 除非这些数据真的就是乱生成的。

```python
# 3 - Then check out the distribution of GMV
sections = np.arange(0, 111, 5)
data_leftjoin['gmv_range'] = pd.cut(data_leftjoin.gmv, sections)
data_leftjoin['gmv_range'].value_counts().plot(kind='bar')
plt.show()
# Result: The quantity of order in each GMV-span is basically the same

# Interesting.
# 4 - Then finally check out the distribution of countries of users
data_leftjoin['country'].value_counts().plot(kind='bar')
plt.show()
# Result: Also almost the same
```

做到这里，基本上可以确定数据集都是假数据了.. 所有输出的图表都一样高；用户也平均分布在7个国家，属实没意思。

虽然可以理解保护商业机密或者运营状况的考虑，那把表得内容替换得看不太出来是什么感觉也行啊... 以前做过一个体彩机构的投注分析算法设计的比赛，
他们操作的流水我觉得可能和Shopee真差不多，背后还是香港马会，反正有点东西，人给的数据也没有这么敷衍啊！

害。

## Reference

[1] pandas.merge - https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.merge.html?highlight=merge#pandas.merge

[2] 时间序列与日期用法 - 

https://www.pypandas.cn/docs/user_guide/timeseries.html#%E6%97%B6%E9%97%B4%E6%88%B3-vs-%E6%97%B6%E9%97%B4%E6%AE%B5