---
layout: blogpage
title: Perpetual Contracts on Earth
comments: true
tags: Crypto Note
---

This is a reading note about Contracts Guide from BitMEX[1] (A P2P crypto-products trading platform). I took notes on strange concepts here 😶! 
 

永续合约(Perpetual Contracts)是一种特殊的期货合约。 与传统合约(traditional <ins>Futures Contract</ins>)不同，永续合约没有到期日，用户可以选择一直持仓。

在展开说明之前，其中的期货合约(Futures Contract) 中的到期日具体指什么？

What is Futures Expiration and Settlement then:

## 所谓 **到期(Expiration)** 和 **结算(Settlement)** [2]

**到期**：所有期货合约都有指定的到期日期。在到期日之前，交易者有多个选择，可以平仓或延长其持仓，而无需持有交易至到期，但部分交易者会选择持有合约并结算。

All futures contracts have a specified date on which they expire. Prior to the expiration date, traders have a number of options to either close out or extend their open positions without holding the trade to expiration, but some traders will choose to hold the contract and go to settlement.


**结算**：结算是指**履行**与原始合约相关的法律交割义务。

对于一些合约，这种交割将以标的商品的实物交割形式进行。 例如，寻求获得谷物的食品生产商可能在寻求交割实物玉米或小麦，而农民可能在寻求将谷物交付给该生产者。虽然实物交割是某些能源、金属和农产品的重要机制，但所有商品期货合约中只有一小部分是实物交割的。

Settlement is the **fulfillment** of the legal delivery obligations associated with the original contract.  For some contracts, this delivery will take place in the form of physical delivery of the underlying commodity. For example, a food producer looking to acquire grain may be looking to take delivery of physical corn or wheat, and a farmer may be looking to deliver his grain to that producer. Although physical delivery is an important mechanism for certain energy, metals and agriculture products, only a small percent of all commodities futures contracts are physically delivered.

<ins>在大多数情况下，交割将以现金结算的形式进行。</ins>当合约是现金结算时，结算以**合约到期时**贷记或贷借合约价值的形式进行。最常见的现金结算产品是股票指数和利率期货，不过贵金属、外汇和一些农产品也可以现金结算。 

<ins>In most cases, delivery will take place in the form of cash settlement.</ins> When a contract is cash-settled, settlement takes place in the form of a credit or debit made for the value of the contract **at the time of contract expiration**. The most commonly cash-settled products are equity index and interest rate futures, although precious metals, foreign exchange, and some agricultural products may also be settled in cash.

对于选择去结算的交易者，交割形式将高度依赖于每个交易者的需求以及交易产品的独特特征。

For traders choosing to go to settlement, the form of delivery will be highly dependent on the needs of each trader, as well as the unique characteristics of the product being traded.

## 永续合约与传统期货合约的具体区别

The Perpetual Contract is similar to a traditional Futures Contract, but has a few differences:

- There is no expiry or settlement (subject to the Early Settlement).  不存在到期与结算。

- Perpetual Contracts mimic a *margin-based<1> spot market<2>* and hence trade close to the underlying reference Index Price.

永续合约模拟基于保证金的现货市场，因此交易价格接近标的物参考指数价格(underlying reference Index Price)。

The Funding mechanism is used to tether contracts to their underlying spot price. 

资金机制用于将合约与其标的物现货价格挂钩。

This is in contrast to a Futures Contract which may trade at significantly different prices due to *basis<3>*. 

这与期货合约形成对比，期货合约可能因基差而以显着不同的价格交易。

<br/>

- Each perpetual contract has its own details which can be found in its Contract Specifications. These details include: 

每个永续合约都有自己的详细信息，可以在其合约规范中找到。这些详情包括：

        - Reference Index / 参考索引
        - Funding Rate / 资金费率
        - Maximum Leverage / 最大杠杆

<br />

💬 注释 / Annotation:

<1> 这里的 margin 指保证金。

**期货保证金(Futures Margin)** 是指期货结算会员按照结算规则存入制定账户的一定数量的资金或缴存符合标准的一定数量的有价证券，以作为期货交易的结算和履约的保证。

<2> 现货市场(Spot Markets) 是指市场上的买卖双方成交后须在若干个交易日内办理交割的金融市场。

<3> 这里的 basis 指基差[5]，基差(Basis)是某一特定商品于某一特定的时间和地点的现货价格与期货价格之差。即 **基差 = 现货价格 - 期货价格**

另外：

> 在「正常市場」（Normal Market / Contango Market）中，基差為負值，即期貨價格高於現貨價格。由於在正常情形下，期貨價格包含了儲存、保存、持有和利息等成本。因此，期貨價格高於現貨價格為正常市場下所發生的情形。

> 在「逆價市場」（Inverted Market / Backwardation Market）中，係指當供給嚴重不足之下，可能會出現現貨價格較期貨價格高，即基差為正值的不正常情況。

The funding mechanism will be explained later. 🍃

## * BitMEX 平台的做法

### Leverage 杠杆
Perpetual contracts do not require traders to post 100% of collateral as margin, because of this you can trade with leverage of up to 100x on some of BitMEX’s contracts. All margin on BitMEX is denominated in Bitcoin, allowing traders to speculate on the future value of its products only using Bitcoin.

永续合约不需要交易者将 100% 的抵押品作为保证金，因此您可以在 BitMEX 的某些合约上使用高达 100 倍的杠杆进行交易。 BitMEX 的所有保证金均以比特币计价，允许交易者仅使用比特币来推测其产品的未来价值。

<br />

### Payout 支付
BitMEX offers perpetual contracts that have inverse, linear and *quanto<4>* payouts. This document[1] explains the key differences between these payouts, and some implications for traders.

The product suits traders who prefer to hold *positions<5>* for a long time and do not want their positions to fluctuate in value due to large swings in basis.

BitMEX 提供具有反向、线性和 *quanto* 支付的永续合约。本文件[1]解释了这些支出之间的主要区别，以及对交易者的一些影响。

该产品适合喜欢长期持有*头寸*且不希望头寸因基差大幅波动而波动的交易者。

<br />

💬 注释 / Annotation:

<4> Quanto[6] 是 Quantity Adjusting Option 的简称，指：标的物（underlying）是以货币A计价，但是以货币B来结算的现金交割型衍生性金融商品。如果某投资人想要投资国外商品，例如美国投资人要投资日经指数，最好的方式是投资CME的日经指数Nikkei 225期货。当Nikkei 225指数变动1点（1日元），期货的价值会变动500美元。如此一来可以<ins>投资海外商品，又可以规避汇率风险。</ins>Quanto期货或选择权其实是一般的期货或选择权再加上一组远期汇率合约。

A quanto[7] is a type of derivative in which the underlying is denominated in one currency, but the instrument itself is settled in another currency at some rate. Such products are attractive for speculators and investors who wish to <ins>have exposure to a foreign asset, but without the corresponding exchange rate risk.</ins>

在 BitMEX 平台：

> These contracts are designed to be easy to trade and understand, but keep in mind as you trade them that your underlying margin and PNL are in Bitcoin. You are still exposed to Bitcoin/USD price risk when trading Quanto Perpetuals, even though the underlying and quote currencies are not Bitcoin.

仍然有 Bitcoin/USD 的价格风险，即使标的和报价货币不是比特币。

<5> Position[8] 这里译为头寸，
指金融单位所拥有的款项。收大于支叫多头寸(long position)，收小于支叫缺/空头寸(short position)。


## What is an Inverse Contract?

<ins>An inverse contract is worth a fixed amount of the quote currency.</ins> In the case of the XBTUSD perpetual, each contract is worth $1 of Bitcoin at any price. XBTUSD is an inverse contract because it is quoted as XBT/USD but the underlying is USD/XBT or 1 / (XBT/USD). It is quoted as an inverse to facilitate hedging US Dollar amounts while the spot market convention is to quote the number of US Dollars per Bitcoin.

<ins>反向合约价值固定金额的报价货币。</ins>对于 XBTUSD 永续合约来说，每份合约在任何价格都价值 1 美元的比特币。 XBTUSD 是反向合约，因为它以 XBT/USD 报价，但标的为 USD/XBT 或 1 / (XBT/USD)。它被引用为倒数以促进对冲美元金额，而现货市场惯例是引用每个比特币的美元数量。

This product is suitable for traders who want to go long or short US Dollars against Bitcoin.

该产品适合想要做多或做空美元兑比特币的交易者。

## What is a Linear Contract?

A linear payout is the simplest to describe, and is used for many *swaps<6>*. The price of a linear contract is expressed as the price of the underlying against the base currency.

线性支付是最容易描述的，并且用于许多掉期。线性合约的价格表示为标的物相对于基础货币的价格。

<6> Swap[9] 这里译为掉期，
掉期是指在外汇市场上买进即期外汇的同时又卖出同种货币的远期外汇，或者卖出即期外汇的同时又买进同种货币的远期外汇。（掉期交易（Swap Transaction）是指交易雙方約定在未來某一時期相互交換某種資產的交易形式。）也就是说在同一笔交易中将一笔即期和一笔远期业务合在一起做，或者说在一笔业务中将借贷业务合在一起做。


## Mechanics of Perpetual Markets

需要清楚的期货市场机制！

When trading perpetual contracts, a trader needs to be aware of several mechanics of the futures market. The key components a trader needs to be aware of are:

**Multiplier**: How much is one contract worth? You can see this information under the Contract Specifications for each instrument. 

一份合约值多少钱。

**Position Marking**: Perpetual contracts are marked according to the Fair Price Marking method. The mark price determines Unrealised PNL and liquidations.

永续合约按照公允价格标记方法进行标记。标记价格决定未实现的盈亏和清算。

**Initial and Maintenance Margin**: These key margin levels determine how much leverage one can trade with and at what point liquidation occurs.

这些关键的保证金水平决定了一个人可以交易多少杠杆以及在什么时候发生清算。

**Funding**: Any position in a perpetual swap that is open when Funding occurs (every 8 hours) will pay or receive funding.

在 Funding 发生时（每 8 小时）打开的永久掉期中的任何头寸都将支付或接收资金。

<br />

### Funding

Funding occurs every 8 hours at 04:00 UTC, 12:00 UTC and 20:00 UTC. You will **only pay or receive funding if you hold a position at one of these times.** If you close your position prior to the funding exchange then you will not pay or receive funding.

只有在 Funding 发生时持有头寸会支付或接收资金。如果在资金交换之前平仓，那么你将不会支付或接收资金。

The funding you pay or receive is calculated as:

        Funding = Mark Value * Funding Rate

Your Mark Value is irrespective of leverage. For example, if you hold 100 XBTUSD contracts, <ins>funding is charged/received on the notional value of those contracts</ins>, and is not based on how much margin you have assigned to the position.

你的标记值与杠杆无关。e.g.,如果你持有 100 张 XBTUSD 合约，则<ins>根据这些合约的名义价值收取/收取资金</ins>，而不是基于您分配给该头寸的保证金多少。

When the `Funding Rate` is positive, longs pay shorts. When it is negative, shorts pay longs. 

当资金费率为正时，多头支付空头。当它为负时，空头支付多头。

<br />

#### Funding Rate Calculations

The Funding Rate is comprised of two main parts: the Interest Rate and the Premium / Discount. This rate aims to keep the traded price of the perpetual contract in line with the underlying reference price. In this way, the contract mimics how margin-trading markets work as <ins>buyers and sellers of the contract exchange interest payments periodically.</ins>

资金费率由两个主要部分组成：利率和溢价/折扣。该利率旨在使永续合约的交易价格与标的参考价格保持一致。通过这种方式，合约模仿了保证金交易市场的运作方式，因为<ins>合约的买卖双方定期交换利息。</ins>

**Interest Rate Component**

Every contract traded on BitMEX consists of two instruments: a Base currency and a Quote currency. For example, on XBTUSD, the Base currency is XBT while the quote currency is USD. The Interest Rate is a **function** of interest rates between these two currencies:

在 BitMEX 上交易的每份合约都包含两种工具：基础货币和报价货币。例如，在 XBTUSD 上，基础货币是 XBT，而报价货币是美元。利率是这两种货币之间利率的**函数**：

    Interest Rate (I) = (Interest Quote Index - Interest Base Index) / Funding Interval

    where
        Interest Base Index = The Interest Rate for borrowing the Base currency
        Interest Quote Index = The Interest Rate for borrowing the Quote currency
        Funding Interval = 3 (Since funding occurs every 8 hours)

Note: Under each Contract Specification page, the source borrow market is stated for each Interest Index.
在每个合约规范页面下，都会针对每个利息指数说明来源借入市场。


**Premium / Discount Component**

The perpetual contract may trade at a significant premium or discount to the Mark Price. In those situations, a **Premium Index** will be used to raise or lower the next Funding Rate to levels consistent with where the contract is trading. Each contract’s Premium Index is available on the specific instrument’s Contract Specifications page and is calculated as follows:

永续合约可能以相对于标记价格显着溢价或折价进行交易。在这些情况下，**溢价指数**将用于将下一个资金费率提高或降低到与合约交易时一致的水平。每个合约的溢价指数可在特定工具的合约规格页面上找到，计算如下：

    Premium Index (P) = 
        (Max(0, Impact Bid Price - Mark Price) - Max(0, Mark Price - Impact Ask Price)) / Spot Price + Fair Basis used in Mark Price


#### Final Funding Rate Calculation

BitMEX calculates the Premium Index (P) and Interest Rate (I) every minute and then performs a 8-Hour Time-Weighted-Average-Price (TWAP) over the series of minute rates.

The Funding Rate is next calculated with the 8-Hour Interest Rate Component and the 8-Hour Premium / Discount Component. A +/-0.05% dampener is added.

BitMEX 每分钟计算溢价指数 (P) 和利率 (I)，然后对一系列分钟利率执行 8 小时时间加权平均价格 (TWAP)。

接下来资金费率使用 8 小时利率成分(Interest Rate Component)和 8 小时溢价/折扣成分(Premium / Discount Component)计算。添加了一个 +/-0.05% 的阻尼(dampener)。

    Funding Rate (F) = 
        Premium Index (P) + clamp(Interest Rate (I) - Premium Index (P), 0.05%, -0.05%)

Hence, if (I - P) is within +/-0.05% then F = P + (I - P) = I. In other words, the Funding Rate will equal the Interest Rate. 

(I - P) 在 +/-0.05% 之内时认为资金费率(Funding Rate) 和利率(Interest Rate)相等。

This calculated Funding Rate is then applied to a trader’s XBT Position Value to determine the Funding Amount to be paid or received at the Funding Timestamp.

然后将计算出的资金费率应用于交易者的 XBT 头寸价值，以确定在进行 Funding 的时间戳时支付或接收的资金金额。

**Funding Rate Caps 资金费率上限**

BitMEX imposes caps on the Funding Rate to ensure the maximum leverage can still be utilized. To do this, two caps are imposed:

BitMEX 对资金费率设置上限，以确保仍然可以使用最大杠杆。
为实现设有两个上限：

1. <ins>The absolute Funding Rate is **capped at** 75% of the Initial Margin - Maintenance Margin. </ins>If the Initial Margin is 1% and the Maintenance Margin is 0.5%, the maximum Funding Rate will be 75% * (1% - 0.5%)= 0.375%.
（绝对资金费率**上限** = （初始保证金 - 维持保证金）* 75%）

2. The Funding Rate may not change by more than 75% of the Maintenance Margin between Funding Intervals.
（资金费率在资金间隔之间的变化不得超过维持保证金的 75%。）





## Reference

[1] Perpetual Contracts Guide - https://www.bitmex.com/app/perpetualContractsGuide#Funding

[2] Get to Know Futures Expiration and Settlement - https://www.cmegroup.com/cn-s/education/learn-about-trading/courses/introduction-to-futures/get-to-know-futures-expiration-and-settlement.html

[3] MBA智库 期货保证金 - https://wiki.mbalib.com/wiki/%E6%9C%9F%E8%B4%A7%E4%BF%9D%E8%AF%81%E9%87%91

[5] 基差 - https://www.moneydj.com/kmdj/wiki/wikiviewer.aspx?keyid=e6856deb-5ca2-4968-bff1-7f4c12e0d10a

[6] 什么是 Quanto - https://zhidao.baidu.com/question/24572297.html

[7] Quanto From Wikipedia, the free encyclopedia - https://en.wikipedia.org/wiki/Quanto

[8] Position From Wikipedia, the free encyclopedia -  https://en.wikipedia.org/wiki/Position_(finance)

[9] Swap - Baidu Baike - https://baike.baidu.com/item/%E6%8E%89%E6%9C%9F/3048850