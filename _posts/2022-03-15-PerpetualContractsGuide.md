---
layout: blogpage
title: Perpetual Contracts on Earth
comments: true
tags: Crypto Note
---

This is a reading note about Contracts Guide from BitMEX[1]. I took notes on strange concepts here 😶!


永续合约(Perpetual Contracts)是一种特殊的期货合约。 与传统合约(traditional <ins>Futures Contract</ins>)不同，永续合约没有到期日，用户可以选择一直持仓。

在展开说明之前，其中的期货合约(Futures Contract) 中的到期日具体指什么？

What is Futures Expiration and Settlement then:

## 所谓 **到期(Expiration)** 和 **结算(Settlement)** [2]

> **到期**：所有期货合约都有指定的到期日期。在到期日之前，交易者有多个选择，可以平仓或延长其持仓，而无需持有交易至到期，但部分交易者会选择持有合约并结算。

All futures contracts have a specified date on which they expire. Prior to the expiration date, traders have a number of options to either close out or extend their open positions without holding the trade to expiration, but some traders will choose to hold the contract and go to settlement.


> **结算**：结算是指**履行**与原始合约相关的法律交割义务。
对于一些合约，这种交割将以标的商品的实物交割形式进行。 例如，寻求获得谷物的食品生产商可能在寻求交割实物玉米或小麦，而农民可能在寻求将谷物交付给该生产者。虽然实物交割是某些能源、金属和农产品的重要机制，但所有商品期货合约中只有一小部分是实物交割的。

Settlement is the **fulfillment** of the legal delivery obligations associated with the original contract.  For some contracts, this delivery will take place in the form of physical delivery of the underlying commodity. For example, a food producer looking to acquire grain may be looking to take delivery of physical corn or wheat, and a farmer may be looking to deliver his grain to that producer. Although physical delivery is an important mechanism for certain energy, metals and agriculture products, only a small percent of all commodities futures contracts are physically delivered.

> <ins>在大多数情况下，交割将以现金结算的形式进行。</ins>当合约是现金结算时，结算以**合约到期时**贷记或贷借合约价值的形式进行。最常见的现金结算产品是股票指数和利率期货，不过贵金属、外汇和一些农产品也可以现金结算。 

<ins>In most cases, delivery will take place in the form of cash settlement.</ins> When a contract is cash-settled, settlement takes place in the form of a credit or debit made for the value of the contract **at the time of contract expiration**. The most commonly cash-settled products are equity index and interest rate futures, although precious metals, foreign exchange, and some agricultural products may also be settled in cash.

> 对于选择去结算的交易者，交割形式将高度依赖于每个交易者的需求以及交易产品的独特特征。

For traders choosing to go to settlement, the form of delivery will be highly dependent on the needs of each trader, as well as the unique characteristics of the product being traded.

## 永续合约与传统期货合约的具体区别

The Perpetual Contract is similar to a traditional Futures Contract, but has a few differences:

- There is no expiry or settlement (subject to the Early Settlement).  不存在到期与结算。

- Perpetual Contracts mimic a margin-based<1> spot market<2> and hence trade close to the underlying reference Index Price.

永续合约模拟基于保证金的现货市场，因此交易价格接近基础参考指数价格(underlying reference Index Price)。

The Funding mechanism is used to tether contracts to their underlying spot price. 

资金机制用于将合约与其基础现货价格挂钩。

This is in contrast to a Futures Contract which may trade at significantly different prices due to basis<3>. 
这与期货合约形成对比，期货合约可能因基差而以显着不同的价格交易。

- Each perpetual contract has its own details which can be found in its Contract Specifications. These details include: 

每个永续合约都有自己的详细信息，可以在其合约规范中找到。这些详情包括：

    - Reference Index / 参考索引
    - Funding Rate / 资金费率
    - Maximum Leverage / 最大杠杆

<br />

<1> 这里的 margin 指保证金。

**期货保证金(Futures Margin)** 是指期货结算会员按照结算规则存入制定账户的一定数量的资金或缴存符合标准的一定数量的有价证券，以作为期货交易的结算和履约的保证。

<2> 现货市场(Spot Markets) 是指市场上的买卖双方成交后须在若干个交易日内办理交割的金融市场。

<3> 这里的 basis 指基差[5]，基差(Basis)是某一特定商品于某一特定的时间和地点的现货价格与期货价格之差。即 **基差 = 现货价格 - 期货价格**

另外：

> 在「正常市場」（Normal Market / Contango Market）中，基差為負值，即期貨價格高於現貨價格。由於在正常情形下，期貨價格包含了儲存、保存、持有和利息等成本。因此，期貨價格高於現貨價格為正常市場下所發生的情形。

> 在「逆價市場」（Inverted Market / Backwardation Market）中，係指當供給嚴重不足之下，可能會出現現貨價格較期貨價格高，即基差為正值的不正常情況。

*BitMEX下的做法

### Leverage
Perpetual contracts do not require traders to post 100% of collateral as margin, because of this you can trade with leverage of up to 100x on some of BitMEX’s contracts. All margin on BitMEX is denominated in Bitcoin, allowing traders to speculate on the future value of its products only using Bitcoin.

### Payout
BitMEX offers perpetual contracts that have inverse, linear and quanto payouts. This document explains the key differences between these payouts, and some implications for traders.

The product suits traders who prefer to hold positions for a long time and do not want their positions to fluctuate in value due to large swings in basis.

[1] Perpetual Contracts Guide - https://www.bitmex.com/app/perpetualContractsGuide#Funding
> BitMEX is a P2P crypto-products trading platform. 


[2] Get to Know Futures Expiration and Settlement - https://www.cmegroup.com/cn-s/education/learn-about-trading/courses/introduction-to-futures/get-to-know-futures-expiration-and-settlement.html

[3] https://wiki.mbalib.com/wiki/%E6%9C%9F%E8%B4%A7%E4%BF%9D%E8%AF%81%E9%87%91

[5] 基差 - https://www.moneydj.com/kmdj/wiki/wikiviewer.aspx?keyid=e6856deb-5ca2-4968-bff1-7f4c12e0d10a