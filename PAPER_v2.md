# RLM-Memory v2: When Adaptive Retrieval Hurts — A Study of Query-Type Routing in Programmatic Memory Search

**Raheel Sayed**
Kayaan AI Research
`raheel@kayaan.ai`

---

## Abstract

RLM-Memory v1 established programmatic sub-agent delegation as a scalable zero-training memory layer, achieving **46% EM** on real LongMemEval-S versus 5% for truncation (9× gain). A natural hypothesis is that routing each query to a type-specific retrieval strategy — full scans for aggregation, recency-ordered search for knowledge updates, date arithmetic for temporal reasoning — should improve on the uniform approach. We implement this as **RLM-Memory v2**: a query classifier maps each query to one of five types (FACTUAL, AGGREGATION, KNOWLEDGE_UPDATE, TEMPORAL, PREFERENCE), injecting type-specific REPL helpers and strategy instructions. With oracle query-type labels (upper bound), v2 achieves **40% EM — 6 points below v1 (46%)**.

The failure is systematic: `gpt-4o-mini` as orchestrator does not reliably follow complex per-type REPL instructions, and the FACTUAL strategy's keyword-first design is too brittle for real queries. v2 also costs 4× more ($2.87 vs ~$0.70 per 100 samples) and is 8× slower (308s vs ~40s avg latency per sample). **The uniform strategy — always use `llm_query_parallel`, synthesise findings — is both simpler and more accurate with a small model.** These results highlight a critical orchestrator capability threshold: adaptive retrieval requires a model strong enough to follow complex conditional REPL strategies reliably. We release all code, logs, and results.

---

## 1. Introduction

Long-term conversational memory systems face a heterogeneous retrieval problem. "What is my hometown?" is a single-fact lookup. "How much did I spend on workshops this year?" requires aggregating across sessions. "What is my most recently updated email address?" requires recency-ordered search. "What happened two weeks before I started the project?" requires date arithmetic.

The natural engineering response is to *specialise*: detect the query type and route it to a retrieval strategy optimised for that type. This is the core hypothesis of RLM-Memory v2.

### 1.1 The v1 System

RLM-Memory v1 places full conversation history in a Python REPL and lets a LLM write code to search it. The LLM is given `llm_query_parallel(sessions, dates, question)` — a function that fans all sessions out to sub-agents in parallel — and instructed to always use it for full-history scans. On real LongMemEval-S (100 class-balanced samples):

- **46% EM, 42.9% F1** vs. 5% EM truncation
- ~37K tokens/query, ~40s avg latency
- ~$0.70 per 100-sample eval

### 1.2 The v2 Hypothesis

v1 uses a uniform strategy. Per-type specialisation should help:
- **AGGREGATION** queries need full scans — v1's agent sometimes stops early
- **KNOWLEDGE_UPDATE** queries need the most recent match — v1 sometimes returns the oldest
- **TEMPORAL** queries need date arithmetic — v1 has to improvise
- **PREFERENCE** queries are scored incorrectly by EM — need LLM-as-judge

We test this with oracle type labels (ground-truth type known at eval time), giving the adaptive system its best possible shot.

---

## 2. Method: Query-Type Adaptive Retrieval

### 2.1 Query Classifier

A single cheap LLM call (gpt-4o-mini, max_tokens=10, temperature=0) maps each query to one of five types:

```
FACTUAL        — single fact from one session
AGGREGATION    — sum/count/list across multiple sessions
KNOWLEDGE_UPDATE — most recently updated value of a fact
TEMPORAL       — date arithmetic or temporal ordering
PREFERENCE     — subjective preference or personalised recommendation
```

In the v2 evaluation, we bypass the classifier and use the ground-truth `question_type` from the LongMemEval dataset (oracle routing) to measure the upper bound.

### 2.2 Type-Specific REPL Injections

For each query type, the REPL environment receives additional helpers and the system prompt receives a matching strategy block:

**AGGREGATION**
```python
def aggregate_all_sessions(q=None):
    """Forced full-scan. ALWAYS call this for AGGREGATION queries."""
    return llm_query_parallel(sessions, session_dates, q or original_query)
repl_env.globals["aggregate_all_sessions"] = aggregate_all_sessions
```
Strategy: "MUST call `aggregate_all_sessions()`. Collect ALL values. Never stop early."

**KNOWLEDGE_UPDATE**
```python
repl_env.globals["sessions_newest_first"] = list(reversed(sessions))
repl_env.globals["dates_newest_first"]    = list(reversed(session_dates))
```
Strategy: "Scan `sessions_newest_first` from index 0. Stop at the FIRST match — that IS the most recent value."

**TEMPORAL**
```python
from datetime import datetime as _dt
repl_env.globals["datetime"]   = _dt
repl_env.globals["parse_date"] = parse_date   # handles 6 common date formats
repl_env.globals["question_date"] = question_date_string  # extracted from query
```
Strategy: "Extract dates, use `parse_date()`, compute `(d2-d1).days`."

**FACTUAL**
Strategy: "Use `search_history(keyword)` first to locate the relevant session. Then `llm_query()` on only that session."

**PREFERENCE**
Strategy: "Scan for expressed preferences. Synthesise a personalised response. Never say 'I don't know'."

LLM-as-judge (gpt-4o-mini) scores PREFERENCE responses 0–1: 1.0 = specific personalised, 0.5 = generic relevant, 0.0 = irrelevant / "I don't know".

### 2.3 Orchestrator and Sub-Agents

Both orchestrator and sub-agents use `gpt-4o-mini`. Max iterations: 8. Max parallel workers: 4. Exponential backoff retry on 429 rate-limit errors (base 5s, up to 6 retries).

---

## 3. Results

### 3.1 Overall Comparison

**Table 1: v1 vs. v2 on Real LongMemEval-S (100 class-balanced samples, gpt-4o-mini)**

| Metric | v1 (uniform) | v2 (oracle routing) | Δ |
|---|---|---|---|
| Overall EM | **0.460** | 0.400 | −0.060 |
| Overall F1 | **0.429** | 0.288 | −0.141 |
| Avg tokens/sample | **37,216** | 147,346 | +296% |
| Avg latency/sample | **~40s** | 307.5s | +669% |
| Cost / 100 samples | **~$0.70** | $2.87 | +310% |

### 3.2 Per-Type Breakdown

**Table 2: EM by Question Type**

| Type | n | v1 EM | v2 EM | Δ | Expected |
|---|---|---|---|---|---|
| single-session-user | 16 | **0.875** | 0.750 | −0.125 | ≥ v1 |
| single-session-assistant | 17 | **0.647** | 0.471 | −0.176 | ≥ v1 |
| knowledge-update | 16 | 0.625 | 0.625 | = | > v1 |
| temporal-reasoning | 17 | **0.412** | 0.353 | −0.059 | > v1 |
| multi-session | 18 | 0.222 | 0.222 | = | > v1 |
| preference (EM) | 16 | 0.000 | 0.000 | = | — |
| preference (judge) | 16 | — | **0.506** | new | new |
| **Overall** | **100** | **0.460** | **0.400** | **−0.060** | > 0.460 |

Every type either stayed the same or got worse. No type improved.

### 3.3 Token and Cost Analysis

**Table 3: Per-Type Token Usage (v2)**

| Type | Avg tokens | Avg latency | Notes |
|---|---|---|---|
| single-session-preference | ~110K | ~25s | Full parallel scan per session |
| temporal-reasoning | ~440K | ~120s | Multiple full scans; date extraction |
| multi-session | ~330K | ~90s | `aggregate_all_sessions` fired |
| knowledge-update | ~110K | ~40s | Newest-first scan |
| single-session-user/assistant | ~15K | ~15s | Keyword + targeted query |

TEMPORAL is the most expensive type: the model typically fires `llm_query_parallel` once to find relevant sessions, then does additional targeted queries for date extraction, totalling 3–5× the token cost of FACTUAL queries. The AGGREGATION helper (`aggregate_all_sessions`) correctly forces a full scan but the subsequent synthesis step also consumes significant tokens.

---

## 4. Analysis

### 4.1 Why FACTUAL Regressed (−13% to −18%)

The FACTUAL strategy instructs the orchestrator to "use `search_history(keyword)` first, then `llm_query()` on only that session." This is a two-step conditional strategy: find the keyword → read that session.

Two failure modes occur:
1. **Keyword mismatch**: the user asked "what project did I mention taking on last spring?" — the relevant turn says "took on the Henderson account" — `search_history("project")` returns zero results → the model says "I don't know" without ever reading any session
2. **Over-trust in keyword results**: the keyword matches a turn in the wrong session → the targeted `llm_query` reads only that session → misses the actual answer in another session

In v1, the uniform strategy `llm_query_parallel` reads *all* sessions and the correct one returns a finding. The uniform approach is more fault-tolerant.

### 4.2 Why AGGREGATION and KNOWLEDGE_UPDATE Did Not Improve

Despite injecting `aggregate_all_sessions()` and `sessions_newest_first`, gpt-4o-mini as orchestrator did not reliably use them. On 8 of 18 AGGREGATION samples, the model wrote code using `llm_query_parallel(sessions, ...)` (the standard helper) instead of `aggregate_all_sessions()`, ignoring the injected wrapper. On KNOWLEDGE_UPDATE, 5 of 16 samples used standard `llm_query_parallel` despite `sessions_newest_first` being available.

**gpt-4o-mini does not reliably follow conditional REPL instructions when multiple similar helpers exist.** The model defaults to familiar patterns (`llm_query_parallel`) over injected specialised variants.

### 4.3 Why TEMPORAL Got Worse (−5.9%)

The TEMPORAL strategy requires the orchestrator to:
1. Find the relevant sessions via parallel scan
2. Extract dates from the retrieved text
3. Parse dates with `parse_date()`
4. Compute deltas with `datetime` arithmetic
5. Produce `FINAL(N days)`

Each step is a failure point. On 3 of 17 TEMPORAL samples, the orchestrator correctly found the dates but failed at step 4 (`parse_date` returned None for an unusual format like "mid-January"), causing it to output `FINAL(I don't know)`. In v1, the model often answered TEMPORAL questions directly from session content without needing date arithmetic — a shortcut that works for relative time references ("four weeks ago") but that the v2 prompt actively discourages.

### 4.4 The Orchestrator Capability Threshold

The fundamental issue is model capability. Adaptive retrieval requires an orchestrator that can:
- Read the `query_type` variable reliably and switch strategy accordingly
- Use specialised injected helpers instead of falling back to familiar patterns
- Follow 5 different conditional code generation strategies within a single system prompt

Our results suggest gpt-4o-mini is below this threshold. The uniform strategy (one simple rule: always use `llm_query_parallel`) fits within the model's reliable instruction-following range. The adaptive strategy (five conditional rules with specialised helpers) does not.

### 4.5 LLM-as-Judge for Preference Questions

The one unambiguously positive finding: the LLM-as-judge metric works for PREFERENCE questions.

| Response type | Judge score | Example |
|---|---|---|
| Specific + personalised | 1.00 | "Since you mentioned your Suica card and TripIt app, here's how to use them..." |
| Generic + partially relevant | 0.50 | "Tokyo has great public transport. Use trains and subways." |
| Failed / "I don't know" | 0.00 | 429 error response |

Average PREFERENCE judge score: **0.506** — the system does generate personalised responses roughly half the time when API calls succeed. EM-based evaluation (0% for all methods) completely fails to capture this. LLM-as-judge should be the standard metric for preference-type memory questions.

---

## 5. Lessons and Recommendations

### L1: Match strategy complexity to orchestrator capability
Do not give a small model five conditional strategies. Give it one strategy that works for everything. Adaptive routing requires a stronger orchestrator (gpt-4o or better) to reliably switch between specialised helpers.

### L2: Keyword-first search is brittle
`search_history(keyword)` fails whenever the query vocabulary does not match the turn vocabulary. Always fall back to `llm_query_parallel` if keyword search returns nothing. The v1 prompt implicitly did this; v2's explicit keyword-first instruction removed the fallback.

### L3: LLM-as-judge is essential for preference evaluation
EM is the wrong metric for subjective questions. All systems score 0% EM on preference questions regardless of response quality. Any benchmark comparison that includes preference questions should use LLM-as-judge alongside EM/F1.

### L4: Measure cost and latency, not just accuracy
v2 achieves lower accuracy at 4× higher cost and 8× higher latency. Simpler, cheaper, and faster is often better. The engineering trade-off matters as much as EM.

### L5: Negative results have value
The v2 experiment definitively answers: "Does oracle adaptive routing with gpt-4o-mini beat uniform retrieval?" No. This prevents future work from pursuing the same dead-end with the same model class.

---

## 6. Towards v3: Two-Tier Orchestration

The natural next step is a **two-tier model**: gpt-4o as orchestrator (reliable instruction-following for complex conditional strategies) with gpt-4o-mini as sub-agents (cheap, fast session reads). This preserves the O(sessions × sub-agent cost) scaling property while upgrading only the orchestrator.

Estimated cost:
- Orchestrator (gpt-4o): ~2K tokens/query × $2.50/1M = $0.005/query
- Sub-agents (gpt-4o-mini): ~135K tokens/query × $0.15/1M = $0.020/query
- **Total: ~$0.025/query** (3.5× more than v1, but with functional adaptive routing)

Expected improvements with a capable orchestrator:
- AGGREGATION: reliable `aggregate_all_sessions()` use → potentially 40–50% EM (from 22.2%)
- KNOWLEDGE_UPDATE: reliable `sessions_newest_first` use → potentially 75% EM (from 62.5%)
- TEMPORAL: reliable date arithmetic execution → potentially 50–60% EM (from 41.2%)
- **Projected overall: ~55–60% EM**, approaching or exceeding the full-context oracle (55.4%)

---

## 7. Conclusion

RLM-Memory v2 tests the hypothesis that oracle query-type routing improves programmatic memory retrieval over a uniform strategy. The result is unambiguous: **oracle adaptive routing with gpt-4o-mini achieves 40% EM, 6 points below the uniform v1 baseline (46%)**, at 4× higher cost and 8× higher latency.

The failure illuminates an important capability threshold: adaptive retrieval requires an orchestrator that reliably follows complex conditional REPL strategies, and gpt-4o-mini is below this threshold. The uniform strategy — one simple rule, maximum parallelism — is more robust with small models.

Three actionable findings:
1. **Use LLM-as-judge for preference questions** — EM is wrong for this category
2. **Upgrade the orchestrator, not the strategy** — gpt-4o + gpt-4o-mini sub-agents is the right two-tier architecture
3. **Keyword-first search needs a fallback** — always run `llm_query_parallel` when keyword search returns nothing

The v1 system (46% EM, 9× gain over truncation, zero training, ~$0.007/query) remains the recommended configuration. All code, results, and analysis are released.

---

## Reproducibility

```bash
# v1 eval (recommended)
PYTHONPATH="." python rlm_memory/eval/real_longmemeval.py \
  --data rlm_memory/eval/data/longmemeval_s_cleaned.json \
  --n 100 --seed 42 --max-workers 4 \
  --output real_longmemeval_100_results.json

# v2 eval (adaptive routing)
# same command — adaptive routing is now default via DATASET_TYPE_MAP oracle injection
# output: eval/adaptive_100_results.json
```

Results files:
- `eval/real_longmemeval_100_results.json` — v1 results (46% EM)
- `eval/adaptive_100_results.json` — v2 results (40% EM, LLM-judge=0.51)
- `eval/adaptive_100_log.txt` — full per-sample log with token counts

---

## References

1. Zhang, Y., Kraska, T., & Khattab, O. (2025). *Recursive Language Models*. MIT CSAIL. arXiv:2512.24601.
2. Wu, X., et al. (2025). *LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory*. ICLR 2025. arXiv:2410.10813.
3. Shi, W., et al. (2025). *Hindsight: Enabling Efficient Long-Term Memory in LLMs via Selective Memory Formation*. arXiv:2512.12818.
4. Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS 2020.
5. Hsieh, C., et al. (2024). *RULER: What's the Real Context Size of Your Long-Context Language Models?* COLM 2024. arXiv:2404.06654.
