---
layout: blogpage
title: 当我在写爬虫的时候，我写些什么
comments: true
tags: Skill Note 
---

需要在短时间里完成一个特定网站的爬虫，从没什么基础开始记录一些坑、经验和中途写的用的其他小工具。

## What I Code About When I Code About Crawling ##

### JavaScript Selector ###

有的时候需要对出异常的网页在前端验证选择器路径到底对不对，用原生JS简单写一些。

- CSS Selector

CSS Selector 有现成方法可以实现。参数只要带上选择器路径即可：

```js
document.querySelector(CSS selectors)
```

Example:

```js
document.querySelector('body > div.comMain > div.conlf > div.conTop > div.crumb > span:nth-child(1) > a')
```

- Xpath Selector

用 `document.evaluate()` 就可以！但是参数麻烦点，用下面的函数直接找吧：

```js
function getElementByXpath(path) {
  return document.evaluate(path, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
}
```

Example:
```js
console.log(getElementByXpath("//html[1]/body[1]/div[1]") );
```

### Can not find exist UUID in MongoDB ###

`_id` 字段是用UUID对象存的，在Python里调用PyMongoDB，怎么转换对象也查不到。

对一条刚刚从数据库里查询到的记录 `x`，获取ID后，参照数据库内的Binary类型还要再做转换。原本写爬虫的人用的是UUID3类型，老UUID了属于是，默认生成的UUID对象应该是UUID4，换成Binary对象的时候需要指定 `SUBTYPE` 参数。

最后的查询条件是这样的：

```py
myquery = {"_id": Binary((x.get('_id').bytes), 3)
```

是看到数据库里的 `_id` 字段.. 下面的 `"$type": "3"` 才改了改试试，成功查到。

```json
{
    "_id": {
        "$binary": "DO1Yk9xnMJCAYkkxspFLEA==",
        "$type": "3"
    },
    "Date": "2021-08-23",
    "Title": "报告：60%以上的红树林损失由人类活动直接导致",
    "Author": "财新网",
    "Content": "\n　　“60% 以上的红树林损失由人类活动直接导致，主要的原因包括将红树林区域改成农田、水产养殖，以及城市化。”由国际自然保护联盟 （IUCN）、大自然保护协会（TNC）等机构共同成立的全球红树林联盟（GMA）近日发布的《2021年全球红树林状况》报告（下称《报告》）中，得出了上述结论。\n　　红树林生长在海洋与陆地相交的潮间带，是一种由各种树木和灌木构成，生物多样性极其丰富的植物群落，具有防风护岸、减少洪水的重要作用。《报告》估计，红树林每年可防止超过 650 亿美元的财产损失，并为约 1500 万人减少洪水风险。此外，红树林还能支持渔业养殖，并为包括老虎、海马等 341 个国际濒危物种在内的多种动物提供栖息地。《报告》称，在许多国家，超过 80% 的渔民依赖红树林，而全球则有超过 410 万的红树林渔民。\n\n",
    "Category": "ESG-绿色经济",
    "Link": "http://www.caixin.com/2021-08-23/101759160.html",
    "Processed": 0
}
```

附上 `SUBTYPE` 参数含义：

Constants

| Constant Name |	Value |	Description |
| ---- | ---- | ---- |
| Binary.SUBTYPE_DEFAULT |	0 |	Default BSON type| 
| Binary.SUBTYPE_FUNCTION	| 1 |	Function BSON type |
| Binary.SUBTYPE_BYTE_ARRAY	| 2 |	Byte Array BSON type |
| Binary.SUBTYPE_UUID_OLD	| 3 |	OLD UUID BSON type |
| Binary.SUBTYPE_UUID |	4	| UUID BSON type |
| Binary.SUBTYPE_MD5	| 5 |	MD5 BSON type |
| Binary.SUBTYPE_USER_DEFINED	| 128 |	User BSON type |


### When soup can not handle Xpath ###

TBC