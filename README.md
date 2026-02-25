# RLM-Memory: Scalable Conversational Memory via Recursive Sub-Agent Delegation

A zero-training memory layer for LLM assistants that achieves **46% EM** vs **5% EM** for truncation on real [LongMemEval-S](https://github.com/xiaowu0162/LongMemEval) — a **9× improvement** with no fine-tuning, no vector database, and constant per-query cost as history grows.

---

## Key Idea

When a user's conversation history exceeds the model's context window, **truncation** (the production default) discards old context and scores ~5% on memory benchmarks. **Full-context** works but costs O(H) tokens and becomes infeasible at realistic long-term history lengths.

**RLM-Memory** places the full history inside a Python REPL and delegates per-session reading to sub-agents, each processing only one ~10K-token chunk in a fresh context. Total cost stays approximately constant regardless of total history length.

```
Total History (490K chars)
        │
        ▼
  Python REPL (full history as env variable)
        │
  LLM writes code to iterate sessions
        │
  ┌─────┴─────┐
  │sub-agent 1│  ← reads session 1 (~10K tokens)
  │sub-agent 2│  ← reads session 2 (~10K tokens)
  │    ...    │
  └─────┬─────┘
        │ findings
        ▼
   FINAL(answer)
```

---

## Results (Real LongMemEval-S, 100 class-balanced samples)

| Method | EM | F1 | Avg Tokens | Latency | Scalable? |
|---|---|---|---|---|---|
| Truncation (32K chars) | 5.0% | 4.0% | ~8K | 2.9s | ✓ |
| **RLM-Memory (ours)** | **46.0%** | **42.9%** | **37,216** | **~4s** ⚡ | **✓** |
| Full-context (oracle) | 55.4% | — | ~120K | — | ✗ (fails >128K tokens) |

> **Latency update:** parallel sub-agent execution (`llm_query_parallel` via `ThreadPoolExecutor`) reduces per-query latency from ~221s (sequential) to **~4s** — a **54× speedup** — with no accuracy loss. All sessions are queried simultaneously; wall time is bounded by the slowest single sub-agent call.

### By question type

| Category | RLM-Memory EM | Truncation EM |
|---|---|---|
| single-session-user | **87.5%** | 0.0% |
| single-session-assistant | **64.7%** | 5.9% |
| knowledge-update | **62.5%** | 18.8% |
| temporal-reasoning | **41.2%** | 5.9% |
| multi-session | 22.2% | 0.0% |
| single-session-preference | 0.0% | 0.0% |

---

## Installation

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="your-key-here"
```

---

## Quickstart

```python
from rlm_memory.chat import MemoryChat

chat = MemoryChat(model="gpt-4o-mini", threshold_chars=20000)

# Accumulate history normally
chat.add_turn("user", "I grew up in Nairobi.")
chat.add_turn("assistant", "Got it!")
# ... hundreds of turns later ...

# RLM-Memory kicks in automatically when history > threshold
answer = chat.query("What city did I grow up in?")
print(answer)  # "Nairobi"
```

---

## Evaluation

### Dataset setup

The LongMemEval-S dataset (265MB) is not included due to GitHub size limits.
Download it from the [official repo](https://github.com/xiaowu0162/LongMemEval) and place it at:

```
eval/data/longmemeval_s_cleaned.json
```

### Run evaluations

```bash
export PYTHONPATH="."

# Needle-in-a-Haystack (synthetic, no dataset needed)
python rlm_memory/eval/niah_eval.py --turns 20 50 100 200 --runs 5

# Real LongMemEval-S (100 class-balanced samples)
python rlm_memory/eval/real_longmemeval.py \
  --data rlm_memory/eval/data/longmemeval_s_cleaned.json \
  --n 100
```

Results are saved to `eval/real_longmemeval_100_results.json`.

---

## Paper

The full paper is in [`latex/main.tex`](latex/main.tex). For Overleaf, upload [`rlm_memory_paper_overleaf.zip`](rlm_memory_paper_overleaf.zip) as a new project.

A Markdown version is at [`PAPER.md`](PAPER.md).

---

## Repository Structure

```
rlm_memory/
├── memory_rlm.py          # Core RLM engine (sub-agent delegation)
├── memory_store.py        # Append-only conversation turn store
├── prompts.py             # System + action prompts
├── chat.py                # MemoryChat drop-in wrapper
├── demo.py                # Interactive demo
├── eval/
│   ├── real_longmemeval.py           # Real LME-S evaluation script
│   ├── niah_eval.py                  # Needle-in-a-haystack benchmark
│   ├── synthetic_longmemeval.py      # Synthetic benchmark
│   ├── real_longmemeval_100_results.json  # Our 100-sample results
│   ├── niah_results.json             # NIAH results
│   └── run_100_log.txt               # Full evaluation log
├── latex/
│   ├── main.tex           # Paper (NeurIPS 2025 format)
│   └── references.bib
├── PAPER.md               # Markdown paper draft
└── requirements.txt
```

---

## Citation

```bibtex
@article{rlmmemory2025,
  title   = {RLM-Memory: Scalable Conversational Memory via Recursive Sub-Agent Delegation},
  author  = {Sayed, Raheel},
  journal = {arXiv preprint},
  year    = {2025}
}
```

---

## Acknowledgements

Built on the [Recursive Language Models](https://arxiv.org/abs/2512.24601) paradigm (Zhang, Kraska, Khattab 2025) and evaluated on [LongMemEval](https://arxiv.org/abs/2410.10813) (Wu et al., ICLR 2025).
