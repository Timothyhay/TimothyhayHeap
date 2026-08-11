# WhalePod reasoning_strip 缓存命中率评测

> 测试时间: 2026-08-11 | 模型: deepseek-chat (deepseek-v4-flash) | API: api.deepseek.com
> 测试方法: 每种模式独立运行 10 次 12 轮工具调用 session，记录服务器报告的真实缓存命中率

---

## 三种模式

| 策略 | 含义 | 适用场景 |
|:---|:---|:---|
| `never` | reasoning_content **永不**回传 API | 默认。当前 deepseek-v4-flash 未强制要求回传 |
| `tool` | 仅带 tool_calls 的 assistant 消息回传 reasoning | DeepSeek 官方文档要求：工具调用时继续推理需要回传 |
| `always` | **所有** assistant 消息均回传 reasoning | 极端情况下需完整保留思维链上下文 |

---

## 汇总结果

| 指标 | never | tool | always |
|:---|:---|:---|:---|
| **session 命中率（中位）** | **94.3%** | 96.4% | 97.1% |
| **命中率 P10** | 93.5% | 96.1% | 96.8% |
| **命中率 P90** | 95.5% | 96.7% | 97.3% |
| 单 session prompt tokens | 1,416,947 | 1,604,825 | 2,202,948 |
| 单 session cached tokens | 1,331,584 | 1,541,248 | 2,141,184 |
| 单 session fresh tokens | 85,363 | 63,577 | 61,764 |
| vs never 基准 (fresh) | 100% | 74.5% | 72.4% |
| vs never 基准 (prompt) | 100% | 113.3% | 155.5% |
| 推理输出 token | 3,018 | 4,408 | 9,830 |
| vs never 基准 (推理输出) | 100% | 146.1% | 325.7% |
| 请求数 | 28 | 30 | 37 |

---

## 核心发现

### 1. 命中率是假象——分母膨胀推高了比率

`always` 命中率 97.1%，比 `never` 的 94.3% 高出 3 个百分点。但 prompt 总量膨胀了 **55%**（1.42M → 2.20M），推理输出膨胀了 **226%**（3,018 → 9,830）。

```
hit_rate = (S + R) / (S + R + tail)
```
`R` 是回传的 reasoning 字节。`R` 在连续请求间字节完全不变 → 天然 100% 可缓存 → 分子分母同步膨胀 → 命中率虚高。

### 2. 真正该看的是 fresh tokens 绝对值

- `never` 每 session：**85,363 fresh tokens** 需要按原价计费
- `always` 每 session：**61,764 fresh tokens**（省了 28%）
- 但 `always` 同时把 cached tokens 从 1.33M 推到 **2.14M**（+61%），cached 也有计费
- `never` 的总 prompt tokens 是 **1.42M**，`always` 是 **2.20M** — 多了 78 万 token

### 3. 模型看到自己的推理后会"话多"

`reasoning_tokens` 从 3,018 涨到 9,830（+226%）。这全是**输出 token**，按 completion 费率计费，比 prompt 贵。同时请求数从 28 涨到 37——模型在思维链上下文中会更积极地调用工具探索。

### 4. 当前 deepseek-v4-flash API 不强制要求回传

`never` 模式下 10 次 session 共 ~280 次请求全部成功，未出现 DeepSeek 文档中描述的 400 错误。因此默认 `never` 在线上是实际可行的。

---

## 结论

| 模式 | 推荐度 | 说明 |
|:---|:---|:---|
| `never` | **推荐默认** | 最少 token 总量，最低成本。当前 API 不强制回传 |
| `tool` | 兼容性开关 | 如果未来 API 严格执行文档要求，切到此模式 |
| `always` | 不推荐 | prompt 膨胀 55%，推理输出膨胀 226%，得不偿失 |

---

## 原始数据

```
bench/results/live_acceptance.json           # never 模式 (10 次重复)
bench/results/live_acceptance_rs_tool.json   # tool 模式 (10 次重复)
bench/results/live_acceptance_rs_always.json # always 模式 (10 次重复)
```

## 代码位置

- `whalepod/endpoints/vllm.py:71-74` — `_encode_message()` 根据 `reasoning_strip` 决定是否写入 `reasoning_content`
- `whalepod/core/agent.py:79,121-141` — `AgentConfig.reasoning_strip` / `Agent.set_reasoning_strip()` / `toggle_reasoning_strip()`
- `whalepod/cli.py:752-761` — `/mode reasoning [never|tool|always]` 命令
- `bench/live_acceptance.py` — `--reasoning-strip` 参数驱动本次测试
