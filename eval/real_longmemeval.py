"""
eval/real_longmemeval.py

Evaluation on the REAL LongMemEval benchmark (ICLR 2025).
Dataset: xiaowu0162/longmemeval-cleaned (longmemeval_s_cleaned.json)

Question types:
  - single-session-user       (70 samples)
  - single-session-assistant  (56 samples)
  - single-session-preference (30 samples)
  - multi-session             (133 samples)
  - temporal-reasoning        (133 samples)
  - knowledge-update          (78 samples)
  Total: 500 samples

Usage:
  python real_longmemeval.py --n 100         # run on 100 class-balanced samples
  python real_longmemeval.py --n 500         # full benchmark
  python real_longmemeval.py --n 50 --types temporal-reasoning multi-session
"""

import os
import sys
import json
import time
import random
import argparse
import re
from typing import List, Dict, Tuple, Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MINIMAL = os.path.join(_ROOT, "..", "Recursive_language_model_rlm-minimal")
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.abspath(_MINIMAL))

import openai as _openai

from rlm_memory import MemoryStore
from rlm_memory.memory_rlm import MemoryRLM
from rlm_memory.query_classifier import DATASET_TYPE_MAP
from rlm.utils.llm import OpenAIClient


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "longmemeval_s_cleaned.json")

QUESTION_TYPES = [
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
]


def load_dataset(path: str = DATA_FILE) -> List[Dict]:
    print(f"Loading dataset from {path} ...")
    with open(path) as f:
        data = json.load(f)
    print(f"  Loaded {len(data)} samples.")
    return data


def sample_stratified(data: List[Dict], n: int, types: List[str], seed: int) -> List[Dict]:
    """Pick n samples class-balanced by question type (roughly equal per type)."""
    random.seed(seed)
    by_type: Dict[str, List[Dict]] = {}
    for s in data:
        qt = s["question_type"]
        if qt not in types:
            continue
        by_type.setdefault(qt, []).append(s)

    per_type = max(1, n // len(types))
    selected = []
    for qt in types:
        pool = by_type.get(qt, [])
        k = min(per_type, len(pool))
        selected.extend(random.sample(pool, k))

    # If we're short (unequal type counts), top up from any type
    leftover = n - len(selected)
    if leftover > 0:
        remaining = [s for s in data if s not in selected and s["question_type"] in types]
        selected.extend(random.sample(remaining, min(leftover, len(remaining))))

    random.shuffle(selected)
    return selected[:n]


# ---------------------------------------------------------------------------
# History builder: convert haystack_sessions → MemoryStore
# ---------------------------------------------------------------------------

def build_memory_store(sample: Dict) -> MemoryStore:
    """
    Flatten all haystack_sessions into a MemoryStore.
    Each session is separated by a [--- Session N (DATE) ---] marker.
    Dates are included so temporal reasoning questions can be answered.
    """
    store = MemoryStore()
    sessions = sample["haystack_sessions"]
    dates = sample.get("haystack_dates", [""] * len(sessions))

    for sess_idx, (session, date) in enumerate(zip(sessions, dates)):
        # Add session boundary marker as a system-style turn
        marker = f"[--- Session {sess_idx + 1} | {date} ---]"
        store.add("system", marker)

        for turn in session:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if role not in ("user", "assistant"):
                role = "user"
            store.add(role, content)

    return store


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _parse_gold_alternatives(gold: str) -> List[str]:
    """
    Some gold answers have multiple acceptable forms, e.g.:
      "21 days. 22 days (including the last day) is also acceptable."
    Split these into a list of candidate answers and return all of them.
    """
    gold = str(gold)
    # Split on common 'also acceptable' patterns
    parts = re.split(
        r"(?:\.|;)\s*(?:\d+ \w+ )?(?:\([\w\s]+\) )?(?:is also acceptable|are also acceptable|also ok|or)",
        gold, flags=re.IGNORECASE
    )
    # Also add the full string as a candidate
    candidates = [gold] + [p.strip() for p in parts if p.strip()]
    return candidates


def token_f1(pred: str, gold: str) -> float:
    p_toks = normalize(pred).split()
    g_toks = normalize(gold).split()
    if not p_toks or not g_toks:
        return float(p_toks == g_toks)
    common = set(p_toks) & set(g_toks)
    if not common:
        return 0.0
    prec = len(common) / len(p_toks)
    rec  = len(common) / len(g_toks)
    return 2 * prec * rec / (prec + rec)


def exact_match(pred: str, gold: str) -> float:
    n_pred = normalize(pred)
    n_gold = normalize(gold)
    if not n_gold:          # never match against an empty gold
        return 0.0
    return float(n_gold in n_pred or n_pred == n_gold)


def score_answer(pred: str, gold) -> Dict:
    """Score pred against gold, handling multi-alternative gold answers."""
    pred = str(pred)
    gold = str(gold)
    alternatives = [a for a in _parse_gold_alternatives(gold) if normalize(a)]
    if not alternatives:
        alternatives = [gold]
    best_em = max(exact_match(pred, alt) for alt in alternatives)
    best_f1 = max(token_f1(pred, alt) for alt in alternatives)
    return {"exact_match": best_em, "f1": best_f1}


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

TRUNC_SYSTEM = (
    "You are a helpful assistant. Answer the question based on the conversation "
    "history provided below. Be concise — a few words or one sentence is fine. "
    "If the information was never mentioned in the conversation, say exactly: "
    "\"I don't know.\""
)


def truncation_baseline(store: MemoryStore, query: str, model: str,
                        max_chars: int = 32_000) -> str:
    hist = store.to_string()
    truncated = hist[-max_chars:] if len(hist) > max_chars else hist
    client = OpenAIClient(model=model)
    msgs = [
        {"role": "system", "content": TRUNC_SYSTEM},
        {"role": "user", "content": (
            f"Conversation history (may be truncated — only the most recent "
            f"portion is shown):\n\n{truncated}\n\n"
            f"Question: {query}"
        )},
    ]
    return client.completion(msgs)


# ---------------------------------------------------------------------------
# LLM-as-judge for preference questions
# ---------------------------------------------------------------------------

_JUDGE_PROMPT = """\
You are evaluating an AI assistant's response to a preference-style memory question.

Question: {question}
AI Response: {response}

Does this response meaningfully address the question based on what the user likely \
shared about their preferences? Score from 0.0 to 1.0:
  1.0 — directly answers with specific, personalised details
  0.5 — partially relevant but generic or vague
  0.0 — irrelevant, empty, or "I don't know"

Reply with just a single decimal number (e.g. 0.8).\
"""


def llm_judge_preference(
    pred: str,
    question: str,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
) -> float:
    """Score a preference-question answer 0.0–1.0 using an LLM judge."""
    try:
        client = _openai.OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY")
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": _JUDGE_PROMPT.format(
                    question=question, response=pred
                ),
            }],
            max_tokens=10,
            temperature=0,
        )
        raw = resp.choices[0].message.content.strip()
        return min(1.0, max(0.0, float(raw)))
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------

def run_eval(samples: List[Dict], model: str, sub_model: str,
             max_iter: int, verbose: bool, trunc_chars: int,
             max_workers: int = 10) -> Dict:

    type_results: Dict[str, Dict] = {}
    all_rlm_em, all_rlm_f1 = [], []
    all_trunc_em, all_trunc_f1 = [], []
    rlm_tokens, rlm_latencies, trunc_latencies = [], [], []
    pref_judge_scores: List[float] = []

    errors = 0
    api_key = os.environ.get("OPENAI_API_KEY")

    for i, sample in enumerate(samples):
        qtype    = sample["question_type"]
        question = sample["question"]
        answer   = sample["answer"]
        qid      = sample["question_id"]

        # Oracle query type mapping — uses ground-truth type from dataset
        oracle_type = DATASET_TYPE_MAP.get(qtype, "FACTUAL")

        store = build_memory_store(sample)
        total_chars = store.total_chars()
        total_turns = store.total_turns()

        # Inject question date for temporal reasoning
        question_date = sample.get("question_date", "")
        if question_date:
            augmented_q = f"[Today's date: {question_date}]\n{question}"
        else:
            augmented_q = question

        print(f"\n[{i+1}/{len(samples)}] id={qid} | type={qtype} ({oracle_type}) | "
              f"turns={total_turns} | chars={total_chars:,}")
        print(f"  Q: {question}")
        print(f"  A: {answer}")

        if qtype not in type_results:
            type_results[qtype] = {
                "rlm_em": [], "rlm_f1": [],
                "trunc_em": [], "trunc_f1": [],
                "llm_judge": [],
            }

        # ---- RLM-Memory (with oracle query type) ----
        rlm = MemoryRLM(
            model=model, sub_model=sub_model,
            max_iterations=max_iter, verbose=verbose,
            max_workers=max_workers,
        )
        t0 = time.time()
        try:
            rlm_ans = rlm.completion(store, augmented_q, query_type=oracle_type)
        except Exception as e:
            rlm_ans = f"[ERROR: {e}]"
            errors += 1
        rlm_lat = time.time() - t0
        rlm_s   = score_answer(rlm_ans, answer)
        toks    = rlm.stats().get("total_tokens", 0)

        all_rlm_em.append(rlm_s["exact_match"])
        all_rlm_f1.append(rlm_s["f1"])
        rlm_tokens.append(toks)
        rlm_latencies.append(rlm_lat)
        type_results[qtype]["rlm_em"].append(rlm_s["exact_match"])
        type_results[qtype]["rlm_f1"].append(rlm_s["f1"])

        # LLM-as-judge for preference questions
        judge_score = None
        if oracle_type == "PREFERENCE":
            judge_score = llm_judge_preference(rlm_ans, question, api_key, model)
            pref_judge_scores.append(judge_score)
            type_results[qtype]["llm_judge"].append(judge_score)

        judge_str = f" judge={judge_score:.2f}" if judge_score is not None else ""
        print(f"  RLM-v2 ({oracle_type}): EM={rlm_s['exact_match']:.2f} F1={rlm_s['f1']:.4f}"
              f"{judge_str} | {rlm_lat:.1f}s | {toks} tok | '{rlm_ans[:80]}'")

        # ---- Truncation baseline ----
        t0 = time.time()
        try:
            trunc_ans = truncation_baseline(store, augmented_q, model, trunc_chars)
        except Exception as e:
            trunc_ans = f"[ERROR: {e}]"
            errors += 1
        trunc_lat = time.time() - t0
        trunc_s   = score_answer(trunc_ans, answer)

        all_trunc_em.append(trunc_s["exact_match"])
        all_trunc_f1.append(trunc_s["f1"])
        trunc_latencies.append(trunc_lat)
        type_results[qtype]["trunc_em"].append(trunc_s["exact_match"])
        type_results[qtype]["trunc_f1"].append(trunc_s["f1"])
        print(f"  Truncation:           EM={trunc_s['exact_match']:.2f} F1={trunc_s['f1']:.4f} "
              f"| {trunc_lat:.1f}s | '{trunc_ans[:80]}'")

    def avg(lst): return round(sum(lst) / len(lst), 4) if lst else 0.0

    type_summary = {
        t: {
            "n":          len(r["rlm_em"]),
            "rlm_em":     avg(r["rlm_em"]),   "rlm_f1":   avg(r["rlm_f1"]),
            "trunc_em":   avg(r["trunc_em"]),  "trunc_f1": avg(r["trunc_f1"]),
            "llm_judge":  avg(r["llm_judge"]) if r["llm_judge"] else None,
        }
        for t, r in type_results.items()
    }

    return {
        "n_samples":            len(samples),
        "model":                model,
        "errors":               errors,
        "adaptive_routing":     True,
        "pref_llm_judge_avg":   avg(pref_judge_scores) if pref_judge_scores else None,
        "overall": {
            "rlm": {
                "em":            avg(all_rlm_em),
                "f1":            avg(all_rlm_f1),
                "avg_tokens":    avg(rlm_tokens),
                "avg_latency_s": avg(rlm_latencies),
            },
            "truncation": {
                "em":            avg(all_trunc_em),
                "f1":            avg(all_trunc_f1),
                "avg_latency_s": avg(trunc_latencies),
            },
        },
        "by_type": type_summary,
    }


# ---------------------------------------------------------------------------
# CLI + pretty-print
# ---------------------------------------------------------------------------

def print_results(results: Dict):
    print("\n" + "=" * 74)
    print("REAL LongMemEval — Results")
    print(f"  n={results['n_samples']} samples | model={results['model']} | "
          f"errors={results['errors']}")
    print("=" * 74)

    o = results["overall"]
    print(f"\n{'Method':<24} {'EM':>8} {'F1':>8} {'Tokens':>10} {'Latency':>10}")
    print("-" * 62)
    print(f"{'Truncation':<24} "
          f"{o['truncation']['em']:>8.4f} {o['truncation']['f1']:>8.4f} "
          f"{'—':>10} {o['truncation']['avg_latency_s']:>9.1f}s")
    print(f"{'RLM-Memory (ours)':<24} "
          f"{o['rlm']['em']:>8.4f} {o['rlm']['f1']:>8.4f} "
          f"{o['rlm']['avg_tokens']:>10.0f} {o['rlm']['avg_latency_s']:>9.1f}s")

    adaptive = results.get("adaptive_routing", False)
    pref_judge = results.get("pref_llm_judge_avg")
    if adaptive:
        print(f"\n  [v2] Adaptive routing: ON  |  "
              f"Preference LLM-judge avg: "
              f"{pref_judge:.3f}" if pref_judge else "  [v2] Adaptive routing: ON")

    print("\nBy Question Type:")
    print(f"  {'Type':<28} {'RLM EM':>7} {'RLM F1':>7} {'Judge':>7} {'Trunc EM':>9}")
    print("  " + "-" * 67)
    order = [
        "temporal-reasoning", "multi-session", "knowledge-update",
        "single-session-user", "single-session-assistant", "single-session-preference",
    ]
    for t in order:
        if t in results["by_type"]:
            r = results["by_type"][t]
            judge_str = f"{r['llm_judge']:>7.3f}" if r.get("llm_judge") else "     — "
            print(f"  {t:<28} {r['rlm_em']:>7.4f} {r['rlm_f1']:>7.4f} "
                  f"{judge_str} {r['trunc_em']:>9.4f}  (n={r['n']})")

    print("=" * 74)

    # Compare vs published
    print("\n--- Published LongMemEval Results (real benchmark, for context) ---")
    pub = [
        ("Observational Mem.", "GPT-5-mini",  0.949, "trained"),
        ("Hindsight",          "OS 20B+",     0.914, "trained"),
        ("MemBuilder",         "Qwen3-4B",    0.858, "fine-tuned"),
        ("Zep/Graphiti",       "GPT-4o",      0.712, "trained"),
        ("Full-ctx baseline",  "GPT-4o",      0.602, "no training"),
        ("Full-ctx baseline",  "GPT-4o-mini", 0.554, "no training"),
    ]
    print(f"  {'System':<22} {'Model':<15} {'Pub EM':>7}  {'Method'}")
    print("  " + "-" * 62)
    for name, mdl, acc, method in pub:
        print(f"  {name:<22} {mdl:<15} {acc:>7.3f}  {method}")
    rlm_em = results["overall"]["rlm"]["em"]
    print(f"  {'RLM-Memory (ours)':<22} {'gpt-4o-mini':<15} {rlm_em:>7.3f}  no training  ← THIS RUN")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RLM-Memory on the real LongMemEval benchmark."
    )
    parser.add_argument("--data",      default=DATA_FILE,
                        help="Path to longmemeval_s_cleaned.json")
    parser.add_argument("--n",         type=int, default=100,
                        help="Number of samples to evaluate (class-balanced per type, default 100)")
    parser.add_argument("--types",     nargs="+", default=QUESTION_TYPES,
                        help="Question types to include")
    parser.add_argument("--model",     default="gpt-4o-mini")
    parser.add_argument("--sub-model", default="gpt-4o-mini")
    parser.add_argument("--max-iter",  type=int, default=8)
    parser.add_argument("--trunc-chars", type=int, default=32_000,
                        help="Truncation window in chars (default 32K)")
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--output",    default="adaptive_100_results.json")
    parser.add_argument("--verbose",      action="store_true")
    parser.add_argument("--max-workers",  type=int, default=10,
                        help="Max parallel sub-agent threads per sample (default 10)")
    args = parser.parse_args()

    data    = load_dataset(args.data)
    samples = sample_stratified(data, args.n, args.types, args.seed)

    print(f"\nRLM-Memory — Real LongMemEval Evaluation")
    print(f"  Model: {args.model} | Sub-model: {args.sub_model}")
    print(f"  Samples: {len(samples)} (class-balanced from {len(data)} total)")
    print(f"  Types:   {', '.join(args.types)}")
    print(f"  Truncation window: {args.trunc_chars:,} chars")

    results = run_eval(
        samples,
        model=args.model,
        sub_model=args.sub_model,
        max_iter=args.max_iter,
        verbose=args.verbose,
        trunc_chars=args.trunc_chars,
        max_workers=args.max_workers,
    )

    out = os.path.join(os.path.dirname(__file__), args.output)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out}")

    print_results(results)


if __name__ == "__main__":
    main()
