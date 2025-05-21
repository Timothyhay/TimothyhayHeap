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
同时有 p < 0.01，认为关系显著。


## 3 Context Analysis


接下来挖掘一下日记具体内容中包含的语义信息。尤其在我完全不记得以前写过了什么的现在，感觉像拿着刮刮乐的小朋友.. 希望能发现一些意外惊喜。

### Tokenization

在没有去除 stopwords 的情况下做了词频统计，因为这里可以发现一些更有趣的事：即使是记录心情，我的语言特征也很明显。比如“..”的写法真的已经用了十几年，被我用来作为一种淡淡的语气衔接标记，成为胜过“啊”/“emmm”等各种赛博逗号的挚爱用法。

![Top10FreqWords](/images/illustration/2024-12-05/Top10FreqWords.png)

在去除部分无意义 stopwords 后，构建`Description`字段的词云。结合词频统计表可以看到更全面的日程偏好（口癖）。

![Wordcloud](/images/illustration/2024-12-05/Wordcloud.png)

首先就是和探索动作有关的“深圳” “出门”和“商场”。大量市区的线下活动（比如宝可梦快闪）也安排在商场里，出门和逛商场对生活在深圳的人来说一定程度上是等价的。至少对我很是如此。

“自己”，在描述一切的时候讨论的最终目标。

以及和探索过程有关的内容，“好吃” “二次元” 和“活动”看起来像很多次日常的主题。


完成分词后，可以尝试使用感兴趣的关键词针对性地判断特殊事件对心情的影响。这里我选择了出现次数相对多，又能说明问题的两个主题：工作与学习。


![MoodByActivityKeywords.png](/images/illustration/2024-12-05/MoodByActivityKeywords.png)

对我来说，工作之后在周末的学习通常是一些感兴趣的事情，往往是自驱的活动；但加班真的不是。虽然不代表加班一定会有坏心情，但加班给心情带来的波动方差很大。

这里的活动关键词我还尝试了天气、运动类型的主题，发现自己其实不怎么提这些内容。尤其是天气这种事，对我完全没有影响.. 似乎只是生活背景色的一部分。毕竟我至今也没有装过天气预报的APP。

### Sentiment Analysis

情感分析是最开始打算做这个记录的时候就计划做的事，作为经典 NLP 任务实现起来有很多现成的轮子，而且也很有意思（NLP魂狠狠动了）。

考虑到调用方便和准确度，这里同时使用 `SnowNLP().sentiments` 基于贝叶斯模型，和今天的第一梯队 LLM `gemini-2.5-flash-preview-04-17` 进行了日记`Description`字段的情感分析。贝叶斯模型的方法主要用于筛选情感相关的关键词，用于本阶段的其他实验中；LLM 方法生成的结果会更多给我自己参考。

顺带一提，这里我使用的 prompt 如下：

```python
    system_prompt = dedent('''
    你是能力极强的情感分析师。接下来你会收到用户关于某个休息日的描述，你需要分析用户的日程描述，给出一段简短的评价，以及一个由你决定的当日心情量化得分。
    接收到的输入以一个JSON表示，包含的键与含义如下：
    {
        'Week': '记录所在周序号', 
        'Title': '记录标题，通常无意义', 
        'OutsideHour': '出门时长', 
        'Mood': '用户的主观心情打分，是0~5之间的浮点数，越高表示记录时认为自己的心情越好', 
        'Description': '当日日程的详细描述，你应当以此为主要依据来分析用户心情', 
        'Created_time': '记录时间'
    }

    请注意：
    1. 你的最终目的是积极地引导用户理解如何过上更快乐、更有意义的生活，因此你需要在评价中用温和的语气对用户进行正面引导，同时你的分析应当尽可能专业、详细。
    2. 你仍然需要客观地根据'Description'字段内容给出一个客观的用户一日心情评价。这个评价不需要参考'Mood'字段，而应该由你的独立分析给出。这里的评分也应该是0~5之间的浮点数，越高表示你认为当日记录的心情越好。
    ''').strip()


response = client.models.generate_content(
                model="gemini-2.5-flash-preview-04-17",
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_schema=SentimentEvaluation,
                    response_mime_type="application/json",
                ),
                contents=str(record)
            )
```
借助 Google `genai` 的


![MoodAndScoreDistribution.png](/images/illustration/2024-12-05/MoodAndScoreDistribution.png)



## 4 

使用了马卡龙配色。


## Reference

https://www.nasbaregistry.org/cpe-monitor-newsletters/dr-art-kohn-explains-how-to-achieve-the-optimal-learning-experience-with-boosts-and-bursts#:~:text=On%20average%2C%2050%20percent%20of%20information%20is%20forgotten,is%20more%20important%20than%20what%20you%20do%20during.