"""
eval/rag_eval.py

RAG baseline evaluation on real LongMemEval-S.

Uses the same 100 class-balanced samples (seed=42) as real_longmemeval.py
for a direct apples-to-apples comparison.

Usage:
  python rag_eval.py
  python rag_eval.py --n 50 --top-k 30
"""

import os
import sys
import json
import time
import argparse

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MINIMAL = os.path.join(_ROOT, "..", "Recursive_language_model_rlm-minimal")
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.abspath(_MINIMAL))

# Reuse data loading + scoring from existing eval
from real_longmemeval import (
    load_dataset, sample_stratified, build_memory_store,
    score_answer, QUESTION_TYPES, DATA_FILE,
)
from rlm_memory.rag_baseline import RAGBaseline


def run_rag_eval(samples, model, embed_model, top_k, context_window):
    rag = RAGBaseline(
        model=model,
        embed_model=embed_model,
        top_k=top_k,
        context_window=context_window,
    )

    type_results = {}
    all_em, all_f1 = [], []
    all_tokens, all_latencies = [], []
    errors = 0

    for i, sample in enumerate(samples):
        qtype    = sample["question_type"]
        question = sample["question"]
        answer   = sample["answer"]
        qid      = sample["question_id"]

        store = build_memory_store(sample)

        question_date = sample.get("question_date", "")
        augmented_q = (
            f"[Today's date: {question_date}]\n{question}"
            if question_date else question
        )

        print(f"\n[{i+1}/{len(samples)}] id={qid} | type={qtype} | "
              f"turns={store.total_turns()} | chars={store.total_chars():,}")
        print(f"  Q: {question}")
        print(f"  A: {answer}")

        if qtype not in type_results:
            type_results[qtype] = {"em": [], "f1": []}

        try:
            rag_ans = rag.query(store, augmented_q)
        except Exception as e:
            rag_ans = f"[ERROR: {e}]"
            errors += 1

        s    = score_answer(rag_ans, answer)
        stat = rag.stats()

        all_em.append(s["exact_match"])
        all_f1.append(s["f1"])
        all_tokens.append(stat.get("total_tokens", 0))
        all_latencies.append(stat.get("latency_s", 0))
        type_results[qtype]["em"].append(s["exact_match"])
        type_results[qtype]["f1"].append(s["f1"])

        print(f"  RAG (top-{top_k}): EM={s['exact_match']:.2f} F1={s['f1']:.4f} "
              f"| {stat.get('latency_s',0):.1f}s "
              f"| {stat.get('total_tokens',0)} tok "
              f"| retrieved={stat.get('retrieved_turns',0)} turns "
              f"| '{rag_ans[:80]}'")

    def avg(lst): return round(sum(lst) / len(lst), 4) if lst else 0.0

    return {
        "n_samples":   len(samples),
        "model":       model,
        "embed_model": embed_model,
        "top_k":       top_k,
        "errors":      errors,
        "overall": {
            "em":            avg(all_em),
            "f1":            avg(all_f1),
            "avg_tokens":    avg(all_tokens),
            "avg_latency_s": avg(all_latencies),
        },
        "by_type": {
            t: {
                "n":  len(r["em"]),
                "em": avg(r["em"]),
                "f1": avg(r["f1"]),
            }
            for t, r in type_results.items()
        },
    }


def print_results(results):
    print("\n" + "=" * 70)
    print("RAG Baseline — Real LongMemEval-S Results")
    print(f"  n={results['n_samples']} | model={results['model']} | "
          f"embed={results['embed_model']} | top_k={results['top_k']} | "
          f"errors={results['errors']}")
    print("=" * 70)

    o = results["overall"]
    print(f"\n  Overall EM : {o['em']:.4f}")
    print(f"  Overall F1 : {o['f1']:.4f}")
    print(f"  Avg Tokens : {o['avg_tokens']:.0f}")
    print(f"  Avg Latency: {o['avg_latency_s']:.2f}s")

    print(f"\n  {'Type':<30} {'EM':>7} {'F1':>7} {'n':>4}")
    print("  " + "-" * 52)
    order = [
        "single-session-user", "single-session-assistant",
        "knowledge-update", "temporal-reasoning",
        "multi-session", "single-session-preference",
    ]
    for t in order:
        if t in results["by_type"]:
            r = results["by_type"][t]
            print(f"  {t:<30} {r['em']:>7.4f} {r['f1']:>7.4f} {r['n']:>4}")

    print("\n--- Comparison (same 100 samples, same scorer) ---")
    print(f"  {'Method':<25} {'EM':>7} {'F1':>7}")
    print("  " + "-" * 42)
    published = [
        ("Truncation (32K)",   0.050, 0.040),
        ("RAG (this run)",     o["em"], o["f1"]),
        ("RLM-Memory",         0.460, 0.429),
        ("Full-ctx (oracle)",  0.554, None),
    ]
    for name, em, f1 in published:
        f1_str = f"{f1:.4f}" if f1 is not None else "  —   "
        print(f"  {name:<25} {em:>7.4f} {f1_str:>7}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="RAG baseline evaluation on real LongMemEval-S."
    )
    parser.add_argument("--data",      default=DATA_FILE)
    parser.add_argument("--n",         type=int, default=100)
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--model",     default="gpt-4o-mini")
    parser.add_argument("--embed-model", default="text-embedding-3-small")
    parser.add_argument("--top-k",     type=int, default=20)
    parser.add_argument("--context-window", type=int, default=1)
    parser.add_argument("--output",    default="rag_results.json")
    args = parser.parse_args()

    data    = load_dataset(args.data)
    samples = sample_stratified(data, args.n, QUESTION_TYPES, args.seed)

    print(f"\nRAG Baseline — Real LongMemEval Evaluation")
    print(f"  Model: {args.model} | Embed: {args.embed_model}")
    print(f"  top_k={args.top_k} | context_window=±{args.context_window}")
    print(f"  Samples: {len(samples)} (class-balanced, seed={args.seed})")

    results = run_rag_eval(
        samples,
        model=args.model,
        embed_model=args.embed_model,
        top_k=args.top_k,
        context_window=args.context_window,
    )

    out = os.path.join(os.path.dirname(__file__), args.output)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out}")

    print_results(results)


if __name__ == "__main__":
    main()
