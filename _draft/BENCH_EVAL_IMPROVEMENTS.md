[bench_eval.md](../../whale-pod/docs/bench_eval.md)# KVCache Bench Improvements (changelog)

This documents the improvements applied in 2026-08 to the benchmark/evaluation
layer that measure (offline) and validate (live) WhalePod's prefix-caching
design. For the full architecture, experiment flow, and results, see the
[technical report](../docs/bench_eval.md).

All six fixes target the gap between "our model of what the server does" and
"what the server actually does". The first two (`tokenizer`, `wire encoding`)
directly shrink the prediction-to-measurement error; the remainder add
instrumentation that makes the remaining gap diagnosable.

## 1. Tokenizer → DeepSeek V4 official BPE

**Before:** `whalepod.core.tokenizer.estimate_tokens` used either
`tiktoken.cl100k_base` (GPT-4's vocabulary) or a fast character heuristic
(CJK ~1 token/char, Latin ~3.2 chars/token). DeepSeek V4's vocabulary differs
from both — `cl100k_base` was not accurate for V3 either — so every benchmark
token count (including the 64-block cache alignment in the offline bench) was
an approximation. Committed results recorded `"tokenizer": "heuristic"`, which
acknowledged the uncertainty.

**After:** The resolver now prefers the official `deepseek-ai/DeepSeek-V4-Flash-0731`
BPE (via `tokenizers.Tokenizer.from_file`) when a local `tokenizer.json` is
present. The estimator still falls back to tiktoken → heuristic with no network
and no dependency, so the offline bench remains fully offline once the
32 ktoken file is on disk.

- `estimate_tokens` → returns V4 BPE counts when available
- `active_tokenizer_name()` → `"deepseek-v4 (tokenizer.json)"` in result meta
- `bench/fetch_tokenizer.py` downloads `tokenizer.json` once from HuggingFace
  Hub (public, no auth needed) to `~/.whalepod/tokenizers/deepseek-v4-flash/`;
  override with `WHALEPOD_TOKENIZER_JSON` env var
- `pyproject.toml` extra: `pip install -e ".[tokenizer]"` to add `tokenizers` dependency

The runtime context_stats and prune thresholds now see the same token count the
server does, so the window-budget arithmetic is accurate for the first time.
When the tokenizer is missing, the heuristic fallback is the same as before.

**Effect on offline results:** as-built prompt-token total dropped from
~1,027,000 (heuristic) to ~911,000 (V4 BPE), and the predicted-vs-measured
MAE should narrow.

## 2. `wire_text` → DeepSeek V4 official message encoding

**Before:** The offline bench modeled the server's input string as
`json.dumps(tools) + "\n" + json.dumps(each message)` — a hand-rolled approximation
that had no special tokens, no role delimiters, and no tool-merge logic.

**After:** The bench now uses the official `encode_messages()` from
`deepseek-ai/DeepSeek-V4-Flash-0731` (vendored verbatim as
`bench/dsv4_encoding.py`, MIT licensed) to build the **exact** input string the
model tokenizes:
- BOS + reasoning-effort prefix (when `thinking_mode="thinking"`)
- System message gets `tools` injected directly (not sent as a separate JSON key)
- User-turn delimiters (`<|User|>` / `<|Assistant|>`) replace bare JSON role keys
- Tool results are **merged into user messages** (V4 has no standalone `tool` role)
- Tool schemas appear as the `"## Tools"` block on the system message

Byte-prefix measurement now runs on the same bytes the server fast-tokenizes,
so the offline reusable-prefix fraction is a direct predictor of the live
cache-hit rate (rather than a proxy for it).

**Effect on offline results:** The prefix reuse pattern changes shape — the
as-built line now spikes to 99.6% on the second request instead of 97.8%.

## 3. Hit-rate reconciliation and cache-write tracking

**Before:** The live bench read `cached_tokens` from the provider but never
checked whether the provider's own split (`hit + miss`) reconciled to their
`prompt_tokens`. If the server reported `hit=800, prompt=1000` and silently
under-counted the miss, the `hit_rate` column would be inflated.

**After:**
- `LiveRequest.miss_tokens` records `prompt_cache_miss_tokens` (or reconstructs
  from `prompt - cached` as fallback)
- `LiveRequest.accounting_consistent` flags requests where hit + miss ≠ prompt
- `report_run` prints a warning when any request is inconsistent
- `LiveRequest.cache_write_tokens` now appears in per-request JSON for billing audit
- The per-request table in the report includes an `idle` column (see #5)

## 4. Dynamic pricing for offline bench

**Before:** `PRICE_PROMPT` / `PRICE_CACHE_READ` were hardcoded constants in
`bench/validate.py` — a 2026-08-04 snapshot that could drift from the live
provider's pricing model.

**After:** The offline bench reads `results/live_acceptance.json` for its
server-fetched `/models` pricing when available, falling back to the hardcoded
snapshot only when running offline before any live run has been recorded.
Prices from the live bench's pricing table drive cost calculations in the
offline bench, keeping the two benches on the same basis.

## 5. Request timestamps and idle-gap correlation

**Before:** The README attributed two low-hit requests (#16 and #24) to
"provider eviction" but this was an interpretation (offline model predicted
≥91%) rather than measured evidence. There was no way to test the eviction
claim from the data.

**After:**
- `RecordingEndpoint` records monotonic timestamps (`t0`, `t1`) per request
- `LiveRequest.idle_s` is the wall-clock gap from the end of the previous request
  to the start of this one
- `report_run` surfaces the largest idle gap preceding a low-hit request as a
  quantifiable proxy for KV-cache eviction
- The per-request table now includes an `idle` column

## 6. Multi-session statistics (`--repeat N`)

**Before:** The live bench ran one 29-request session, producing a point
estimate of the cache hit rate, cost, and agreement. Provider state
(load, eviction) changes between runs, so a single run is a sample, not a
distribution.

**After:** `--repeat N` runs the main (and optionally prune) session N times
and reports:
- Per-request hit rate across runs: median / P10 / P90
- Session-level hit rate and cost distributions
- `bench/results/live_acceptance.json` includes a `_aggregate` section
- The chart output still shows the first run (for a clean picture);
  the aggregate is a number table in the report

## Running

```bash
# One-time: download the tokenizer
python bench/fetch_tokenizer.py

# Offline validation (no network — uses the local tokenizer now)
python bench/validate.py

# Live acceptance (needs API key; repeat for statistical strength)
python bench/live_acceptance.py --repeat 10
```

## Results snapshot

Re-ran after the improvements (tokenizer = V4 BPE, encoding = official encoder,
prices = synced from live_acceptance.json):

| Variant | Prompt tokens | Reusable | Reuse % | Prompt cost |
| :--- | ---: | ---: | ---: | ---: |
| as-built | 911,083 | 858,240 | 94.2% | $0.02020 |
| no-ledger | 1,099,805 | 1,030,528 | 93.7% | $0.02478 |
| rolling-summary | 518,504 | 465,216 | 89.7% | $0.01317 |
| three-zone | 913,223 | 405,312 | 44.4% | $0.05301 |

These numbers replace the heuristic-based results that were checked in before
(1,027,163 prompt tokens for as-built). The cost ordering is unchanged; the
absolute numbers are now the first figures measured with the real V4 tokenizer
on the real model-input bytes.
