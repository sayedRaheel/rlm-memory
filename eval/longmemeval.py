"""
eval/longmemeval.py

LongMemEval evaluation for rlm_memory.

LongMemEval (ICLR 2025) is the standard benchmark for evaluating
long-term memory of chat assistants. It has 500 QA questions embedded
inside realistic multi-session chat histories.

Paper:  https://arxiv.org/abs/2410.10813
GitHub: https://github.com/xiaowu0162/LongMemEval
HuggingFace: https://huggingface.co/datasets/wu-et-al/LongMemEval

Setup:
    pip install datasets
    # Dataset downloads automatically on first run

Usage:
    python eval/longmemeval.py --split test --num-samples 50
    python eval/longmemeval.py --split test --num-samples 50 --model gpt-4o
    python eval/longmemeval.py --local-path /path/to/longmemeval.json
"""

import os
import sys
import json
import time
import argparse
import re
from typing import List, Dict, Optional, Tuple

# Add package root and rlm-minimal to path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MINIMAL = os.path.join(_ROOT, "..", "Recursive_language_model_rlm-minimal")
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.abspath(_MINIMAL))

from rlm_memory import MemoryChat, MemoryStore
from rlm_memory.memory_rlm import MemoryRLM
from rlm.utils.llm import OpenAIClient


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_longmemeval_hf(split: str = "test", num_samples: Optional[int] = None) -> List[Dict]:
    """
    Load LongMemEval from HuggingFace datasets.

    Each sample has:
      - question_id
      - question
      - answer (str or list)
      - sessions (list of {role, content} turns)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Install HuggingFace datasets: pip install datasets"
        )

    print(f"Loading LongMemEval ({split} split) from HuggingFace...")
    ds = load_dataset("wu-et-al/LongMemEval", split=split, trust_remote_code=True)

    samples = []
    for i, item in enumerate(ds):
        if num_samples and i >= num_samples:
            break
        samples.append(item)

    print(f"Loaded {len(samples)} samples")
    return samples


def load_longmemeval_local(path: str, num_samples: Optional[int] = None) -> List[Dict]:
    """Load LongMemEval from a local JSON file."""
    with open(path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        samples = data[:num_samples] if num_samples else data
    else:
        samples = list(data.values())[:num_samples] if num_samples else list(data.values())

    print(f"Loaded {len(samples)} samples from {path}")
    return samples


def build_memory_store(sessions) -> MemoryStore:
    """
    Convert LongMemEval sessions into a MemoryStore.

    sessions can be:
      - list of {role, content} dicts
      - list of lists of {role, content} dicts (multi-session)
    """
    store = MemoryStore()

    # Flatten multi-session format if needed
    if sessions and isinstance(sessions[0], list):
        turns = [turn for session in sessions for turn in session]
    else:
        turns = sessions

    for turn in turns:
        role = turn.get("role", "user").lower()
        content = turn.get("content", "")
        if role in ("user", "human"):
            store.add("user", content)
        elif role in ("assistant", "ai", "gpt"):
            store.add("assistant", content)

    return store


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def token_f1(predicted: str, expected: str) -> float:
    """Token-level F1 score (standard for open-domain QA)."""
    pred_tokens = normalize(predicted).split()
    exp_tokens = normalize(expected).split()
    if not pred_tokens or not exp_tokens:
        return float(pred_tokens == exp_tokens)
    common = set(pred_tokens) & set(exp_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(exp_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(predicted: str, expected: str) -> float:
    return float(normalize(predicted) == normalize(expected))


def score_answer(predicted: str, expected) -> Dict[str, float]:
    """Score predicted answer against expected (handles list of acceptable answers)."""
    if isinstance(expected, list):
        candidates = [str(e) for e in expected]
    else:
        candidates = [str(expected)]

    best_em = max(exact_match(predicted, c) for c in candidates)
    best_f1 = max(token_f1(predicted, c) for c in candidates)
    return {"exact_match": best_em, "f1": best_f1}


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def truncation_baseline(store: MemoryStore, query: str, model: str, max_chars: int = 16_000) -> str:
    history_str = store.to_string()
    truncated = history_str[-max_chars:] if len(history_str) > max_chars else history_str
    client = OpenAIClient(model=model)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. "
                "Answer the question based on the conversation history provided. "
                "Be concise. If unsure, say so."
            ),
        },
        {
            "role": "user",
            "content": f"Conversation history (may be truncated):\n{truncated}\n\nQuestion: {query}",
        },
    ]
    return client.completion(messages)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run_eval(
    samples: List[Dict],
    model: str,
    sub_model: str,
    max_rlm_iterations: int,
    verbose: bool,
    run_truncation: bool = True,
) -> Dict:
    rlm_scores = {"exact_match": [], "f1": []}
    trunc_scores = {"exact_match": [], "f1": []}
    rlm_latencies = []
    trunc_latencies = []
    rlm_tokens_list = []
    failed = 0

    for i, sample in enumerate(samples):
        question = sample.get("question", sample.get("query", ""))
        expected = sample.get("answer", sample.get("answers", ""))
        sessions = sample.get("sessions", sample.get("history", []))
        qid = sample.get("question_id", sample.get("id", i))

        if not question or not sessions:
            print(f"  [{i+1}] Skipping sample {qid} — missing question or sessions")
            failed += 1
            continue

        store = build_memory_store(sessions)

        print(f"\n[{i+1}/{len(samples)}] id={qid} | "
              f"{store.total_turns()} turns | {store.total_chars():,} chars")
        print(f"  Q: {question[:100]}")
        print(f"  A: {str(expected)[:80]}")

        # --- RLM Memory ---
        rlm_engine = MemoryRLM(
            model=model, sub_model=sub_model,
            max_iterations=max_rlm_iterations, verbose=verbose
        )
        t0 = time.time()
        try:
            rlm_answer = rlm_engine.completion(store, question)
        except Exception as e:
            rlm_answer = f"[ERROR: {e}]"
        rlm_latency = time.time() - t0
        rlm_s = score_answer(rlm_answer, expected)
        rlm_tokens = rlm_engine.stats().get("total_tokens", 0)

        rlm_scores["exact_match"].append(rlm_s["exact_match"])
        rlm_scores["f1"].append(rlm_s["f1"])
        rlm_latencies.append(rlm_latency)
        rlm_tokens_list.append(rlm_tokens)

        print(f"  RLM:        EM={rlm_s['exact_match']:.2f} F1={rlm_s['f1']:.2f} "
              f"| {rlm_latency:.1f}s | {rlm_tokens} tokens | '{rlm_answer[:80]}'")

        # --- Truncation baseline ---
        if run_truncation:
            t0 = time.time()
            try:
                trunc_answer = truncation_baseline(store, question, model)
            except Exception as e:
                trunc_answer = f"[ERROR: {e}]"
            trunc_latency = time.time() - t0
            trunc_s = score_answer(trunc_answer, expected)

            trunc_scores["exact_match"].append(trunc_s["exact_match"])
            trunc_scores["f1"].append(trunc_s["f1"])
            trunc_latencies.append(trunc_latency)

            print(f"  Truncation: EM={trunc_s['exact_match']:.2f} F1={trunc_s['f1']:.2f} "
                  f"| {trunc_latency:.1f}s | '{trunc_answer[:80]}'")

    def avg(lst):
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    n = len(samples) - failed
    summary = {
        "total_samples": len(samples),
        "evaluated": n,
        "failed": failed,
        "rlm": {
            "exact_match": avg(rlm_scores["exact_match"]),
            "f1": avg(rlm_scores["f1"]),
            "avg_latency_s": avg(rlm_latencies),
            "avg_tokens": avg(rlm_tokens_list),
        },
        "truncation": {
            "exact_match": avg(trunc_scores["exact_match"]),
            "f1": avg(trunc_scores["f1"]),
            "avg_latency_s": avg(trunc_latencies),
        } if run_truncation else {},
    }
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LongMemEval evaluation for rlm_memory")
    parser.add_argument("--split", default="test", help="Dataset split (default: test)")
    parser.add_argument("--num-samples", type=int, default=20, help="Samples to evaluate")
    parser.add_argument("--local-path", help="Local JSON file instead of HuggingFace")
    parser.add_argument("--model", default="gpt-4o-mini", help="Main LLM model")
    parser.add_argument("--sub-model", default="gpt-4o-mini", help="Sub-agent model")
    parser.add_argument("--max-iter", type=int, default=8, help="Max RLM iterations")
    parser.add_argument("--no-truncation", action="store_true", help="Skip truncation baseline")
    parser.add_argument("--verbose", action="store_true", help="Verbose RLM output")
    parser.add_argument("--output", default="longmemeval_results.json", help="Output JSON file")
    args = parser.parse_args()

    print("rlm_memory — LongMemEval Evaluation")
    print(f"Model: {args.model} | Sub-model: {args.sub_model}")
    print(f"Samples: {args.num_samples} | Max RLM iterations: {args.max_iter}")

    # Load dataset
    if args.local_path:
        samples = load_longmemeval_local(args.local_path, args.num_samples)
    else:
        samples = load_longmemeval_hf(args.split, args.num_samples)

    # Run evaluation
    summary = run_eval(
        samples=samples,
        model=args.model,
        sub_model=args.sub_model,
        max_rlm_iterations=args.max_iter,
        verbose=args.verbose,
        run_truncation=not args.no_truncation,
    )

    # Save results
    output_path = os.path.join(os.path.dirname(__file__), args.output)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Samples evaluated: {summary['evaluated']} / {summary['total_samples']}")
    print(f"\nRLM Memory:")
    print(f"  Exact Match: {summary['rlm']['exact_match']:.4f}")
    print(f"  F1 Score:    {summary['rlm']['f1']:.4f}")
    print(f"  Avg Latency: {summary['rlm']['avg_latency_s']:.1f}s")
    print(f"  Avg Tokens:  {summary['rlm']['avg_tokens']:.0f}")
    if summary["truncation"]:
        print(f"\nTruncation Baseline:")
        print(f"  Exact Match: {summary['truncation']['exact_match']:.4f}")
        print(f"  F1 Score:    {summary['truncation']['f1']:.4f}")
        print(f"  Avg Latency: {summary['truncation']['avg_latency_s']:.1f}s")
    print(f"\nResults saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
