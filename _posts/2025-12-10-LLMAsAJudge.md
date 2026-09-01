---
layout: modern-article
title: LLM-as-a-judge .. reliable or not?
comments: true
tags: LLM
---

# Work in progress

LLM-as-a-judge: a complete guide to using LLMs for evaluations

It’s worth noting that LLM-as-a-judge is not an evaluation metric in the same sense as, say, accuracy, precision, or NDCG. In machine learning, a metric is a well-defined, objective measure: they precisely quantify how well a model’s predictions match the ground truth.

In contrast, LLM-as-a-judge is a general technique where you use LLM to approximate human labeling. When you ask an LLM to assess qualities like "faithfulness to source," "correctness," or "helpfulness," you define what these terms mean in the evaluation prompt and rely on the semantic relationships the LLM learned from training data. 

LLM evaluators in several scenarios:

- Pairwise comparison: give LLM two responses and ask to choose the better one. This lets you compare models, prompts, or configurations to see which performs best.
成对比较 ：给学习模型 (LLM) 两个答案，并要求其选择更优的答案。这样可以比较不同的模型、提示或配置，从而确定哪个性能最佳。

- Evaluation by criteria (reference-free): ask the LLM to assess a response or conversation based on tone, clarity, correctness, or other dimensions.
按标准评估（无参考） ：让 LLM 根据语气、清晰度、正确性或其他维度评估回复或对话。

- Evaluation by criteria (reference-based): provide extra context, like a source document or reference, and ask the LLM to score the response.
按标准（参考资料）进行评估 ：提供额外的背景信息，例如来源文件或参考资料，并让 LLM 对回复进行评分。

prompting techniques:
1. Use binary or low-precision scoring.
2. Explain the meaning of each score.
3. Simplify evaluation by splitting criteria.
4. Add examples to the prompt. 
5. Encourage step-by-step reasoning.
6. Set a low temperature.


## Reference

[1] https://www.evidentlyai.com/llm-guide/llm-as-a-judge