---
layout: blogpage
title: Perpetual Contracts on Earth
comments: true
tags: Crypto Note
---

This is a reading note about Contracts Guide from BitMEX[1]. I took notes on strange concepts here 😶!

永续合约(Perpetual Contracts)是一种特殊的期货合约。 与传统合约(traditional <ins>Futures Contract</ins>)不同，永续合约没有到期日，用户可以选择一直持仓。

其中，期货合约(Futures Contract) 中的到期日具体指什么？

What is Futures Expiration and Settlement then:

## 所谓**到期(Expiration)**和**结算(Settlement)** [2]

> **到期** <br/>
> 所有期货合约都有指定的到期日期。在到期日之前，交易者有多个选择，可以平仓或延长其持仓，而无需持有交易至到期，但部分交易者会选择持有合约并结算。

All futures contracts have a specified date on which they expire. Prior to the expiration date, traders have a number of options to either close out or extend their open positions without holding the trade to expiration, but some traders will choose to hold the contract and go to settlement.


> **结算** <br/>
结算是指履行与原始合约相关的法律交割义务。对于一些合约，这种交割将以标的商品的实物交割形式进行。 例如，寻求获得谷物的食品生产商可能在寻求交割实物玉米或小麦，而农民可能在寻求将谷物交付给该生产者。虽然实物交割是某些能源、金属和农产品的重要机制，但所有商品期货合约中只有一小部分是实物交割的。

Settlement is the fulfillment of the legal delivery obligations associated with the original contract. For some contracts, this delivery will take place in the form of physical delivery of the underlying commodity. For example, a food producer looking to acquire grain may be looking to take delivery of physical corn or wheat, and a farmer may be looking to deliver his grain to that producer. Although physical delivery is an important mechanism for certain energy, metals and agriculture products, only a small percent of all commodities futures contracts are physically delivered.

> 在大多数情况下，交割将以现金结算的形式进行。当合约是现金结算时，结算以合约到期时贷记或贷借合约价值的形式进行。最常见的现金结算产品是股票指数和利率期货，不过贵金属、外汇和一些农产品也可以现金结算。 

In most cases, delivery will take place in the form of cash settlement. When a contract is cash-settled, settlement takes place in the form of a credit or debit made for the value of the contract at the time of contract expiration. The most commonly cash-settled products are equity index and interest rate futures, although precious metals, foreign exchange, and some agricultural products may also be settled in cash.

> 对于选择去结算的交易者，交割形式将高度依赖于每个交易者的需求以及交易产品的独特特征。

For traders choosing to go to settlement, the form of delivery will be highly dependent on the needs of each trader, as well as the unique characteristics of the product being traded.


The Perpetual Contract is similar to a traditional Futures Contract, but has a few differences:

- There is no expiry or settlement (subject to the Early Settlement).



- Perpetual Contracts mimic a margin-based spot market and hence trade close to the underlying reference Index Price. 
    - The Funding mechanism is used to tether contracts to their underlying spot price.
    - This is in contrast to a Futures Contract which may trade at significantly different prices due to basis.
Each perpetual contract has its own details which can be found in its Contract Specifications. These details include:
Reference Index
Funding Rate
Maximum Leverage


[1] Perpetual Contracts Guide - https://www.bitmex.com/app/perpetualContractsGuide#Funding
* BitMEX is a P2P crypto-products trading platform. 


[2] Get to Know Futures Expiration and Settlement - https://www.cmegroup.com/cn-s/education/learn-about-trading/courses/introduction-to-futures/get-to-know-futures-expiration-and-settlement.html