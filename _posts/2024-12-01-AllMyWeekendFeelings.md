---
layout: blogpage
title: All My Weekend Feelings in 2024
tags: Life-Fragments
---

绵绵不绝的工作压力让我对每一段休息时间都非常珍视，同时在自己意识到周末过得很快的时候又非常懊悔。我喜欢待在家里、想要休息，但是一直在家睡觉又感觉缺乏出门探险的新鲜感.. 让人很是纠结。我想要快乐，想知道怎么做才可以在仅有的空闲时间收获快乐。

于是听从乐忻的建议做了一些记录。我尽可能记录了 2024 年给我留下印象的周末，包含 74 条日记。记录、主观心情以及其他元数据可以被我存在[我的 Notion 笔记里](https://timothyhay.notion.site/298dfef81f834bfaa335e65e7c1c7249?v=cba1cfb5a5c44ffd954d88ec5edbfcb8&pvs=4)。Art Kohn 认为 70% 的经历会在 24h 后忘记[1]，因此部分待在家里的记录，或者没有当天留下日记或者照片的条目，就真的无论如何也想不起来了.. 更不要说客观描述心情。因此 2024 年的 70+ 条记录对我来说已经是极限了 T T。

![Notion Preview](/images/illustration/2025-03-12/Notion%20Preview.png)

## 1 Task Intro

但好在要进行简单数据分析，对我需要的 NLP 任务而言这个数量已经足够了。

### 1.1 Target

我希望至少弄明白两件事：

1. 到底周末做什么才能开心 - 至少不那么焦虑也行
2. 出门或者不出门能让我感觉在过更有意义的生活吗

### 1.2 Data Preparation

针对上文的目标，在做日常记录时，我只会简单记录如下字段：

```
{
    'Week': '记录所在周序号', 
    'Title': '记录标题，通常无意义', 
    'OutsideHour': '出门时长', 
    'Mood': '主观心情打分，是介于0~5之间的浮点数，越高表示记录时认为自己的心情越好', 
    'Description': '当日日程的详细描述', 
    'CreatedTime': '记录时间'
}
```

这里我通过 Notion 表格的触发器让数据库每个周末增加两条对应时间的空白记录，来确保筛选和排序时通过`CreatedTime`字段可以精准过滤到具体哪个周末，对没能及时记录的条目也不用想上个周末是几号了。

## 2 Statistical Analysis

使用传统描述性统计分析方法用仅有的量化数据进行一些分析。值得一提的是心情(Mood)字段是主观填写的，这里的分析只能说明一种偏主观的关系。

但谁在意呢？心情当然是主观的。

### raw_data.info()

如果单独看看数据集中心情和外出时长(OutsideHour)的分布，容易看出心情的分布比较正态，而且感觉确实自己在评价平静温和的一天的时候倾向于给一个3.0。
实际上3分以上的日子对我来说应该显著包含一些快乐的因素 *（虽然只是凭感觉在评，完全没有自己提前订好标准）* 。

这样看来，快乐的日子还是比难过的多好多 - 平均心情比我的预设平均值要多0.53的样子。

![MoodAndOutsideHourDistribution.png](/images/illustration/2024-12-05/MoodAndOutsideHourDistribution.png)

相比之下出门时长分布就偏了很多，大多数时候我都不怎么出门，或者只是短短地体验了一下“超市25分钟”这样的半月/月度活动。

同时打印了一下出门时长和心情值的全年趋势。可以注意到几个高峰有一些重合，但是由于比例尺的关系看得不是很清楚。

![Mood and OutsideHour Over Weeks.png](/images/illustration/2024-12-05/MoodAndOutsideHourOverWeeks.png)

但单独看心情变化的话，可以尝试观察一下季节对自己的心情影响。为了让趋势更易读我进一步以窗口大小为 5 周做了平滑处理：

![Mood Moving Average (Window=5)](/images/illustration/2024-12-05/MoodMovingAverageWindow5.png)


显然还是没有发现什么特别的影响。硬要说的话年初好像真的挺开心的，也许是因为过年和元旦给我很多特别的仪式感和假期..？

### Correlation Analysis

但心情和外出时长确实有一些关系。

![MoodAndOutsideHourScatter](/images/illustration/2024-12-05/MoodAndOutsideHourScatter.png)

从上图的散点可以看出一个非常明显的相关关系：**不出门不一定不快乐，但出远门一定很开心！**

不过相关性不能代表因果，因为往往能吸引我出门很长时间的事情往往是有意思的线下活动或长途旅游。而且对我来说有计划的一次出行会塞满各种我喜欢的事情，即使旅途可能会比较累或者有其他未知的麻烦，但计划中的快乐事件赋予了行程很大程度的稳定快乐。比如 5月1日区天津泡泡岛听 YOASOBI 的事，就是累得要死站得要不行了结束还打不到车，在防波堤一样什么也没有的滨海新区狂走了数公里才打上车回酒店.. 即使这样也很难说是不快乐的回忆。

最后对上文提到的时间（周数）/心情/外出时长绘制了 Pearson 和 Spearman 相关系数的热力图。这里展示了 Pearson 相关度。

![PearsonCorrelationMatrix](/images/illustration/2024-12-05/PearsonCorrelationMatrix.png)

同时体现了另一个数值 Score，这其实是我后来使用 LLM 进行的心情打分。

根据热力图，可以说我的出门时长和心情其实很有关系，但是心情随时间基本没影响。同时 LLM 的判断似乎还不如我的样子。

理论上如果衡量单调关系这里选择 Spearman 相关度更好，它不要求正态分布（因为我有一些`OutsideHour`集中在 0 侧），对异常值也更鲁棒。不过从结果上来看影响不大：

```
Pearson Correlation Matrix:
                 Week      Mood  OutsideHour
Week         1.000000 -0.093533    -0.060592
Mood        -0.093533  1.000000     0.587222
OutsideHour -0.060592  0.587222     1.000000

Spearman Correlation Matrix:
                 Week      Mood  OutsideHour
Week         1.000000 -0.171926    -0.150199
Mood        -0.171926  1.000000     0.402214
OutsideHour -0.150199  0.402214     1.000000
```

因为 Pearson 明显高于 Spearman，表明关系更偏向线性而非非线性单调。

**0.58 表示中等偏强正相关：** 出门时间对心情有积极影响，但不是极强。
同时 p < 0.01，认为关系显著。


## 3 Context Analysis



![MoodAndScoreDistribution.png](/images/illustration/2024-12-05/MoodAndScoreDistribution.png)



## Reference

https://www.nasbaregistry.org/cpe-monitor-newsletters/dr-art-kohn-explains-how-to-achieve-the-optimal-learning-experience-with-boosts-and-bursts#:~:text=On%20average%2C%2050%20percent%20of%20information%20is%20forgotten,is%20more%20important%20than%20what%20you%20do%20during.