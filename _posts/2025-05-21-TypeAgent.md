
# TypeAgent

TypeAgent 提供了例子与指导原则帮助使用结构化 LLM 与 prompting 方法构建 Agent 架构。

旨在使用单个独立Agent应用至所有应用中

## 仓库


## 设计原则

*部分描述比较抽象，需要结合示例打开研究。

1. 蒸馏模型至逻辑架构（）
- Actions: 应用预定义特征替换模型调用搜寻特征
- Memory: 从文本中构建本体（图） - 
- Plans: 人、机、模型均使用 TOT 交互


2. 使用架构控制信息密度
- Actions:由Action集合+描述定义相互独立的Action种类
- Memory: 紧密语义结构 （？）
- Plans: 每个搜索树节点代表一个关注的子问题


3. 使用架构控制协作交互 <已涉及相关场景>
- Actions: 用户决定如何澄清Action请求
- Memory: 单模型提取文本逻辑结构
- Plans: 模型（quality models, advantage models, language models）、人、机协同扩展每个BFS节点


## 特性

1. 新Memory机制: Structured RAG，一种索引与查询方法

优势： 回答历史对话中的问题效果更好

*此部分文档缺失


自然语言用户请求（可以通过LLM）被转换为搜索查询表达式。然后计算这些查询表达式。查询结果用于生成用户请求的答案。

对非结构化数据，查询将包含范围表达式和树模式表达式。

---
