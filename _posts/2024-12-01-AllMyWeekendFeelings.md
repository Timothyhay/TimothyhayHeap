---
layout: blogpage
title: All My Weekend Feelings in 2024
tags: Life-Fragments
---

绵绵不绝的工作压力让我对每一段休息时间都非常珍视，同时在自己意识到周末过得很快的时候又非常懊悔。我喜欢待在家里、想要休息，但是一直在家睡觉又感觉缺乏出门探险的新鲜感.. 让人很是纠结。我想要快乐，想知道怎么做才可以在仅有的空闲时间收获快乐。

于是听从乐忻的建议做了一些记录。我尽可能记录了 2024 年给我留下印象的周末，包含 74 条日记。记录、主观心情以及其他元数据可以被我存在[我的 Notion 笔记里](https://timothyhay.notion.site/298dfef81f834bfaa335e65e7c1c7249?v=cba1cfb5a5c44ffd954d88ec5edbfcb8&pvs=4)。Art Kohn 认为 70% 的经历会在 24h 后忘记[1]，因此部分待在家里的记录，或者没有当天留下日记或者照片的条目，就真的无论如何也想不起来了.. 更不要说客观描述心情。因此 2024 年的 70+ 条记录对我来说已经是极限了 T T。

![Notion Preview](../images/illustration/2025-03-12/Notion%20Preview.png)

## 1 Task Intro

但好在要进行简单数据分析，对我需要的 NLP 任务而言这个数量已经足够了。

### Target

我希望至少弄明白两件事：

1. 到底周末做什么才能开心 - 至少不那么焦虑也行
2. 出门或者不出门能让我感觉在过更有意义的生活吗

### Data Preparation

针对上文的目标，在做日常记录时，我只会简单记录如下字段：

```json
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



## Reference

https://www.nasbaregistry.org/cpe-monitor-newsletters/dr-art-kohn-explains-how-to-achieve-the-optimal-learning-experience-with-boosts-and-bursts#:~:text=On%20average%2C%2050%20percent%20of%20information%20is%20forgotten,is%20more%20important%20than%20what%20you%20do%20during.