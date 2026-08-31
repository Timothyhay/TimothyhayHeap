---
layout: modern-article
title: Retrieval and Reranking for Code Snippet
comments: true
tags: Deep-Learning
---

I need a plan for effective code retrieval to extract the most relevant part of code file(s) before send it to LLM - today we call it **Context Management** (code as most the context window)

对典型的 RAG 任务而言，系统会包含离线索引和在线查询两阶段。对我们的代码检索场合，由于用户有固定的查找范围，且文档少，chunk少，所以不太需要召回。
这里我补充一下一般意义上 RAG 方法的流程。

# 1. General RAG

- 离线：数据获取/解析清洗 → 切分(Chunking) → 向量化(Embedding) → 建索引(Indexing)
- 在线：查询理解/改写 → 召回(Retrieval) → 重排(Reranking) → 上下文组装 → LLM 生成 → (评估/反馈)

传统 RAG 的优化方法（有用的、没大用的、学术界流行的）我会单独整理到这里：
[_Explore and Exercise: Improve LLM Performance by RAG_](/2024/05/21/RAGOptimization.html)

## 离线阶段

### 1 Data Ingestion & Parsing

把各种异构数据转成干净的结构化文本。

* PDF 解析：版面分析（Layout Analysis，如 LayoutLM、PP-Structure）、OCR、表格抽取。
* HTML：DOM 树清洗、正文抽取（Readability 算法、boilerplate removal）。
* 保留元数据（标题、章节层级、时间、作者、URL），作为后续过滤和引用的基础。

### 2 Chunking

切分需要在两个目标之间权衡：chunk 要*小*到语义聚焦（利于向量检索精度），又要*大*到自包含（才有利于 LLM 理解）。

一般来说兜底方案是按 定长token 数切，overlap 缓解边界截断，通用文档可以按结构切，比如章节 → 段落 → 句子 层级递归下探。
另外还有两种经典做法：

- 语义切分 (Semantic Chunking) 计算相邻句子 embedding 的余弦相似度，在相似度骤降处断开，适用主题跳跃明显的长文
- 父子切分 (Parent-Child)	用小 chunk 检索，命中后返回其所属的大 chunk 或父文档

### 3 Embedding

这里的 embedding 通常在这个场景翻译成向量化。

**核心模型：双塔/Bi-encoder。** 查询 $q$ 和文档 $d$ 分别独立编码为向量，相关性用余弦相似度或内积衡量
(p.s. 当向量被做过 L2 归一化（即模长为 1 的单位向量）时，二者完全等价)：

$$
\text{sim}(q, d) = \frac{E_q(q) \cdot E_d(d)}{\|E_q(q)\| \, \|E_d(d)\|}
$$

**训练原理：对比学习（Contrastive Learning）。** 主流用 InfoNCE 损失，拉近正样本对、推远负样本对：

$$
\mathcal{L} = -\log \frac{e^{\text{sim}(q, d^+)/\tau}}{e^{\text{sim}(q, d^+)/\tau} + \sum_{i} e^{\text{sim}(q, d_i^-)/\tau}}
$$

其中 $\tau$ 是温度系数。训练效果高度依赖**难负例挖掘**（hard negative mining，用 BM25 或上一轮模型召回的"看似相关但不相关"样本）和**批内负例**（in-batch negatives）。

这里可以展示一下我的专利。等有空我画个图 ;)

**相关 terms 速查**

- **非对称编码**：查询短、文档长，语义分布不同。很多模型（如 E5、BGE）在查询侧加前缀 `"query: "`、文档侧加 `"passage: "`，或干脆用两套编码器。
- **Matryoshka Representation Learning (MRL)**：训练时让向量前 $k$ 维就携带主要信息，推理时可截断降维，用精度换存储/速度。
- **Learned Sparse Retrieval（如 SPLADE）**：神经模型输出词表维度的稀疏权重向量，兼具倒排索引的效率和语义扩展能力（会自动激活同义词的权重）。

### 4. Indexing

索引。海量向量的精确 kNN 搜索是 $O(N)$，所以必须用**近似最近邻（ANN）**：

- **HNSW（Hierarchical Navigable Small World）**：多层跳表式图结构。上层是稀疏的"高速公路"用于快速定位区域，逐层下沉到底层稠密图做精细搜索。查询复杂度约 $O(\log N)$，是目前召回率/延迟综合最优的主流选择。
- **IVF（倒排文件）**：先用 k-means 把向量空间聚成 $n_{list}$ 个簇，查询时只搜最近的 $n_{probe}$ 个簇。
- **PQ（Product Quantization，乘积量化）**：把 $D$ 维向量切成 $m$ 段，每段独立做 k-means 量化为 8-bit 码本 ID，把向量压缩为 $m$ 字节，距离计算通过查表完成。常与 IVF 组合成 IVF-PQ，用于十亿级规模。
- **倒排索引（Inverted Index）**：为稀疏检索（BM25）服务，词 → 文档列表的映射，配合跳表求交。

> HNSW 将跳表的分层思想引入高维小世界图，通过‘顶层稀疏图大步贪心跳跃、逐层下潜并在全量底层做Beam Search’，
> 以 $O(logN)$ 的对数时间复杂度实现高召回率的向量检索。

## 在线阶段

### 5. Query Rewrite

查询改写。用户查询往往口语化、依赖上下文，与文档表述存在语义上的 gap。
一般来说多用 LLM 改写或者优化：

- **Query Rewriting**：结合多轮对话历史消解指代（"它的价格呢？" → "iPhone 16 的价格"）。
- **Multi-Query / Query Expansion**：把一个查询扩展成多个不同角度的变体，分别召回后合并。
- **HyDE（Hypothetical Document Embeddings）**：让 LLM 生成假想答案文档，再用假想文档的向量去检索：因为**文档和文档之间的向量距离比查询和文档更近**。
- **查询分解（Decomposition）**：多跳问题拆成子问题串行检索。
- **意图路由（Routing）**：判断该查哪个数据源/是否需要检索。

### 6. Retrieval

召回。

**稀疏检索：BM25**，基于词频统计的经典算法：

$$
\text{score}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t,d)\,(k_1+1)}{f(t,d) + k_1 \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}
$$

其中 $f(t,d)$ 是词频，$k_1$ 控制词频饱和，$b$ 控制文档长度归一化。优点：精确匹配专有名词、编号、罕见词非常可靠；无需训练。缺点：无法处理同义改写。

**稠密检索**：用（第 3 步的）bi-encoder 向量 + ANN 索引。优点：语义泛化；缺点：对未见过的专名、长尾词弱。
可以用比如 Sentence-BERT 或者 bge-large-zh-v1.5 / bge-m3 (Bi-Encoder)。注意这里是双塔架构的 BGE 模型，和 bge-reranker Cross-Encoder（交叉编码单塔架构）不一样的。

**混合检索（Hybrid Search）**：两路并行召回后融合。最常用 **RRF（Reciprocal Rank Fusion）**，只看排名不看分数，天然规避了两路分数量纲不可比的问题：

$$
\text{RRF}(d) = \sum_{r \in \text{rankers}} \frac{1}{k + \text{rank}_r(d)}, \quad k \approx 60
$$

此外还可加**元数据过滤**（时间、来源、权限）做预过滤或后过滤。

> Quick Question: 什么不只用 Embedding 模型从头搜到底？
> 既然 bge-embedding 速度那么快，为什么最后还要加一个 bge-reranker 呢？
> 
> A：双塔模型（Embedding）有信息瓶颈：
> 文档和 Query 在编码时是完全不知道对方存在的，各自压缩成 1024 维的向量，丢失了大量 Token 级别的局部交互细节（无法感知细微的逻辑修饰词、长尾专名对齐）。
> 
> 而交叉编码器（Reranker）精度极高：
> Query 的每个 Token 可以通过 全量自注意力机制（Self-Attention） 去逐一关注 Document 的每一个 Token，能捕捉到非常深层的因果、否定、指代和上下文关联。

### 7. Reranking

召回阶段为了速度牺牲了精度，主要是因为**查询和文档独立编码，没有交互**。

重排用更贵的模型对 top-K（如 100 条）精排出 top-N（如 5 条）：

- **Cross-Encoder**：把 `[query, document]` 拼接成一个序列输入 Transformer，让 query 与 document 的 token 之间做**全量交叉注意力**，输出单一相关性分数。精度远高于 bi-encoder，但每对都要一次前向计算，只能用于小候选集。
  典型模型就是 bge-reranker-v2-m3 或者 jina-reranker-v2-multilingual。
- **ColBERT（Late Interaction）**：折中方案。文档 token 向量可离线预计算，在线只算 MaxSim 交互：
  ColBERTv2  / RAGatouille（一个开源实现库）可以用上。

$$
\text{score}(q,d) = \sum_{i \in q} \max_{j \in d} \; E_{q_i} \cdot E_{d_j}
$$

- **LLM Rerank / Listwise Rerank**：直接让 LLM 对候选列表排序或打分，效果最好但最贵。
  除了 LLM 以外，也有基于开源 LLM 微调的 Reranker，比如 bge-reranker-v2-gemma、Qwen2-7B-Reranker 等。

> Quick Question: 为什么不能用 Reranker 去做 Dense Retrieval？
> A:
> Reranker 无法离线建立向量索引（他是返回一个一个标量相似度分数的，embedding model返回一个高维稠密向量）：
> 因为bge-reranker 的输出是一个相关性分数，它根本不产生固定长度的单条文档向量。
> 它必须同时吃进 `(Query, Document)`才能算分。
> 
> 假设你的知识库有 100 万篇文档，计算量会直接导致系统崩溃：
> 如果用 Reranker 去做全库检索，用户发来一个问题，你需要把这个问题和 100 万篇文档分别拼接，运行 100 万次完整的 Transformer 前向推理，搜一次可能要花几十分钟。
> 所以，Reranker 只能在小范围（比如召回出来的 Top 50~100 条）候选集上使用。

### 8. Generation

Context augmentation & generation 阶段。

- **Prompt 组装**：无非就是系统指令 + 带编号的检索片段 + 用户问题；一般要要求模型**引用来源编号**、**无据可依时明确说不知道**（抑制幻觉）。
- **位置策略**：LLM 存在 "Lost in the Middle" 现象，对上下文首尾的信息利用率高于中间，重要片段应放两端。不过对现在（2026年8月）的模型而言好很多了。
- **上下文压缩**：对超长候选做抽取式压缩（如 LLMLingua）或让 LLM 先摘要再拼接。
- **其他高级模式**：Self-RAG / CRAG（模型自评检索结果质量，不合格则重新检索或改用 web 搜索）、迭代式检索（生成中途触发新检索）。

### 9. Evaluation

目前业内最主流的评估体系是围绕 RAG 三元组（RAG Triad） 和 Ragas / TruLens 等开源评估框架展开的。

所谓 RAG 三元组：QC relevance, QA relevance, CA faithfulness -

```
[ 用户 Query ]
        /          \
  (Context Relevance) (Answer Relevance)
      /              \
[ 检索 Context ] ─── [ 生成 Answer ]
        (Faithfulness/忠实度)
```

- 检索质量（Retrieval）: Recall@K（Hit Rate@K，Top-K 结果中是否至少包含一个标准答案切片）、MRR（平均倒数排名，one over the rank of the first hit，$\frac{1}{\text{rank of first hit}}$）、nDCG(归一化折损累计增益，Normalized Discounted Cumulative Gain) - 这个指标不强求 0/1 二分类，能体现“非常相关”与“略微相关”的区别。不过依赖标注：需要人工或大模型打出多级相关度评分
- 生成质量（Generation）: Faithfulness（答案是否忠于上下文）、Answer Relevance、Context Precision/Recall（RAGAS 框架，基本是LLM-as-a-Judge算的）
- 端到端（业务指标，E2E performance）: 人工评测、LLM-as-a-Judge

---

## 自然语言召回代码片段时的调整

用自然语言查询召回代码与普通文本 RAG 的区别在于：

1. **模态鸿沟更大**：查询是自然语言，目标是形式语言代码片段，两者词面几乎零重叠，语法结构完全不同。
2. **代码有严格的结构**：AST、作用域、调用关系、类型系统，这些是纯文本没有的强信号。
3. **代码片段不自包含**：一个函数的语义依赖 import、类定义、被调用的辅助函数，孤立的 chunk 语义残缺。
4. **标识符携带浓缩语义**：`parseHttpRequestHeader` 一个词等于一句话。

针对每个阶段的调整如下：

### Chunking: By Structure

By Length -> By Structure. 对代码做固定长度滑窗切分会把函数拦腰截断，产生语法不完整、语义残缺的 chunk。很容易想到几个要点：

- **基于 AST 切分**：用 `tree-sitter` 等解析器构建语法树，以**函数 / 方法 / 类**为天然切分单元。超长函数再沿 AST 子树递归下切（保证每个 chunk 仍是完整语法节点）。
- **上下文注入**：每个 chunk 附带其关键信息（和父类信息）——文件路径、所属类名、函数签名、docstring、相关 import。
- **Parent-Child 结构特别适合代码**：用函数级小 chunk 检索，命中后返回整个类或整个文件给 LLM。

实际动手时，我设计了5个分治切分的原则：

1. 自顶向下：优先把一个完整的代码块（比如一个类）作为切片，即从树顶top-down搜索。能装下就是最好的上下文单元。
2. 长度检测优先：把一个代码块作为切片前先确定大小，大了我们再切
3. 动态分治切分：- 如果整个chunk的token数小于threshold，整个chunk（包括其中所有方法/字段)成为一个单独切片  - 如果~大于threshold呢，我们就把内部的子块（比如方法）作为切片单元
4. 智能聚合：在处理子块的时候，我们可能会切得很细，我们会先尝试聚合连续的多个子块，保持token数 < threshold
5. 保留结构：切片时尽量保留高级结构。如子切片保留所在类名，docstring。同时把父节点的结构也放入切片，内容用 … 代替。确保单独拿出LLM能懂。这里我们优先保证语义完整性，添加的一点冗余信息也是必要的，有助于模型定位的。

### Embedding：use bi-encoder for Code

通用文本 embedding 模型对代码效果差，需要在 **(NL 描述, 代码)** 平行语料上做过对比学习的模型：

- 代表模型：CodeBERT / UniXcoder / CodeT5+，以及商用的 voyage-code、jina-embeddings-code 等。
- 训练数据天然存在：**docstring/注释 ↔ 函数体**、**commit message ↔ diff**、**Stack Overflow 问题 ↔ 采纳答案中的代码**。对比学习的正样本对就是"NL 描述与其实现"，这正是在直接优化"NL 查询 → 代码"这个非对称任务。

<br>

用户经常在查询里直接提到 API 名、错误信息、变量名，这些精确匹配是稠密检索的弱项。看起来需要稀疏检索，但是这种情况其实grep就完事了。理论上可以改造分词器 + BM25，但是没必要。

### Reranker：use cross-encoder/LLM for Code

重排用**代码感知的 cross-encoder**（在 NL-code 相关性数据上微调），或直接用 LLM 判断会比较好。现代 LLM 本身读代码能力很强，LLM rerank 在代码场景收益明显。

e.g.
CodeRankLLM is a 7B LLM fine-tuned for listwise code-reranking. When combined with performant code retrievers like CodeRankEmbed, it significantly enhances the quality of retrieved results for various code retrieval tasks.


实际上我们没有使用 LLM 架构的 reranker，还是出于了性能考虑。

### Evaluation

- 检索指标不变（MRR、Recall@K），但数据集换用 **CodeSearchNet、CoSQA、SWE-bench**（检索子任务）等 NL→Code 基准。
- 增加**功能正确性**维度：检回的代码能否真正解决问题，可用执行测试验证，而非仅看语义相似。

### 调整总结表

| 阶段 | 通用文本 RAG | NL → 代码检索 |
|---|---|---|
| 切分 | 按长度/语义切 | **按 AST 语法单元切**，注入路径/签名/imports 上下文 |
| 嵌入 | 通用 embedding | **代码专用双塔模型**（NL-code 对比学习训练） |
| 查询改写 | 改写、扩展 | **NL→伪代码 HyDE**；索引期做代码→NL 摘要 |
| 稀疏检索 | 标准 BM25 | **拆分标识符的分词器**，索引错误信息字面量 |
| 结构信号 | 无 | **符号索引、调用图扩展、agentic grep** |
| 重排 | cross-encoder | 代码感知 cross-encoder / LLM 判定功能相关性 |
| 过滤 | 时间、来源 | **语言、仓库、路径、版本** |
| 评估 | MS MARCO 等 | CodeSearchNet / SWE-bench，加执行正确性 |

**一句话总结**：通用 RAG 的骨架（切分→嵌入→混合召回→重排→生成）在代码场景完全保留，但每个组件都要"代码化"——切分尊重语法结构，嵌入弥合 NL-代码模态鸿沟（专用模型 + 双向翻译），稀疏检索理解标识符，并额外引入文本世界不存在的杀手锏：**AST、符号表和调用图**这些精确的结构化信号。实践中最强的方案往往是"向量检索 + BM25 + 符号索引 + agentic 探索"的多路混合体系。

## Note

- Bi-Encoder
  Query 和 Code/Doc 分别独立通过模型，被压缩成两个固定维度的向量（Embedding）。最后只通过简单的“向量点积”或“余弦相似度”来计算分数。这种方式强制将复杂的语义压缩到一个向量中。Query 中的每个 Token 无法直接“看见” Code 中的 Token。

由于缺乏深层的 Cross-Attention，它难以捕捉精细的语义匹配（例如：代码中的变量名是否对应 Query 中的某个特定约束）。

如果不适合 Rerank，为什么还要用它？ 因为它快。你可以预先计算好库里几百万个代码的向量，检索时只需做矩阵乘法。
相对的 Cross-Encoder 必须实时计算每一对 (Query, Candidate)，无法预计算。

- Cross-Encoder
  Query 和 Code 拼接在一起输入模型。模型中的每一层 Transformer 都在让 Query 和 Code 的 Token 进行互相注意（Attention）。这样模型可以逐字逐句地比对细节。

jina-embeddings-v2-base-code is an multilingual embedding model speaks English and 30 widely used programming languages. Same as other jina-embeddings-v2 series, it supports 8192 sequence length.

CodeT5+ is a new family of open code large language models with an encoder-decoder architecture that can flexibly operate in different modes (i.e. encoder-only, decoder-only, and encoder-decoder) to support a wide range of code understanding and generation tasks. It is introduced in the paper:

CodeT5+: Open Code Large Language Models for Code Understanding and Generation by Yue Wang*, Hung Le*, Akhilesh Deepak Gotmare, Nghi D.Q. Bui, Junnan Li, Steven C.H. Hoi (* indicates equal contribution).

Compared to the original CodeT5 family (base: 220M, large: 770M), CodeT5+ is pretrained with a diverse set of pretraining tasks including span denoising, causal language modeling, contrastive learning, and text-code matching to learn rich representations from both unimodal code data and bimodal code-text data. Additionally, it employs a simple yet effective compute-efficient pretraining method to initialize the model components with frozen off-the-shelf LLMs such as CodeGen to efficiently scale up the model (i.e. 2B, 6B, 16B), and adopts a "shallow encoder and deep decoder" architecture. Furthermore, it is instruction-tuned to align with natural language instructions (see our InstructCodeT5+ 16B) following Code Alpaca.

### 使用没有微调过的模型作为Reranker

以 CodeT5+ 为例：CodeT5+ 的重排序能力主要来自于其 Text-Code Matching预训练任务。在该模式下，模型不仅独立编码文本和代码，还通过 Decoder 的 Cross-Attention 机制深度融合两者的信息，判断它们是否匹配。

**选择模型：**
必须使用经过双模态（Bimodal）训练的checkpoint，例如 Salesforce/codet5p-220m-bimodal 或 Salesforce/codet5p-770m。纯 Encoder 模型（如 embedding 版本）通常用于粗排（向量检索），而 Reranker 需要 Decoder 参与。

**数据构造：**
Reranker 是 Cross-Encoder 模式。我们需要将 (Query, Code_Candidate) 拼成一对输入：

> [CLS] Query [SEP] Code

Encoder 输入：自然语言查询（Query）。
Decoder 输入：代码候选（Code Snippet）。

**获取匹配分数：**
CodeT5+ 的匹配分数的计算逻辑如下：

- 将 Query 输入 Encoder。
- 将 Code 输入 Decoder。
- 取 Decoder 输出序列中 [EOS] Token 的 Hidden State。
- 将该向量通过一个二分类线性层（Projector），输出“匹配（Match）”和“不匹配（Mismatch）”的 Logits。取“匹配”类的 Logit 或概率作为相关性得分。

这里 Logit（未归一化的原始预测值）和 Probability（经过 Sigmoid/Softmax 归一化的概率值）或 Cosine Similarity 是两码事。Logit 的绝对值没有物理意义（不像概率代表置信度）。这就是所谓 Calibration 问题，Logit之间的相关关系并不线性对应于相关性，除非模型校准过。虽然 Logit 对排序（A > B）有效，但如果需要过滤低质量结果，Logit 很难确定截断点。以及 Logit 跨模型不可比（那是自然的）。

不过，如果在重排序阶段，我们只关心候选文档的相对顺序（谁比谁好），Logit 提供了最细粒度的区分度，避免了 Softmax 在高分段的挤压效应。这样也能用。


## Reference

https://huggingface.co/nomic-ai/CodeRankEmbed

https://gangiswag.github.io/cornstack/

