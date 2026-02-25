"""
eval/niah_eval.py

Needle-in-a-Haystack evaluation for rlm_memory.

Generates a synthetic long conversation with hidden facts scattered at
specific positions, then asks recall questions. Tests how well MemoryChat
finds facts buried deep in conversation history.

Compares three approaches:
  1. Truncation baseline  — keep only the last N chars (standard LLM behaviour)
  2. Full-context baseline — feed entire history (only works if it fits in context)
  3. RLM memory           — our approach

Usage:
    python eval/niah_eval.py
    python eval/niah_eval.py --turns 100 --runs 10
    python eval/niah_eval.py --turns 200 --model gpt-4o-mini
"""

import os
import sys
import json
import time
import random
import argparse
import string
from typing import List, Dict, Tuple, Optional

# Add package root and rlm-minimal to path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MINIMAL = os.path.join(_ROOT, "..", "Recursive_language_model_rlm-minimal")
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.abspath(_MINIMAL))

from rlm_memory import MemoryChat, MemoryStore
from rlm_memory.memory_rlm import MemoryRLM
from rlm.utils.llm import OpenAIClient


# ---------------------------------------------------------------------------
# Synthetic conversation generator
# ---------------------------------------------------------------------------

FILLER_TOPICS = [
    "The weather today is quite pleasant.",
    "I finished reading that book you recommended.",
    "The meeting went well, we covered the agenda.",
    "I had coffee this morning and felt more awake.",
    "The new software update seems faster than before.",
    "I went for a walk in the park after lunch.",
    "The project timeline looks reasonable to me.",
    "I need to review the quarterly report later.",
    "The team seems motivated about the new goals.",
    "I ordered a new keyboard — the old one was worn.",
    "The conference call was rescheduled to Thursday.",
    "I fixed a small bug in the data pipeline.",
    "The client seemed happy with the presentation.",
    "I updated my notes from yesterday's discussion.",
    "The server load has been normal all week.",
]

FACT_TEMPLATES = [
    ("my favorite color", "My favorite color is {value}."),
    ("my lucky number", "My lucky number is {value}."),
    ("my cat's name", "My cat's name is {value}."),
    ("my hometown", "I grew up in {value}."),
    ("my project codename", "The project codename is {value}."),
    ("my budget", "The budget for this quarter is ${value}."),
    ("my target metric", "Our target metric is {value}%."),
    ("my anniversary date", "Our company anniversary is {value}."),
    ("my PIN", "The locker PIN is {value}."),
    ("the magic number", "The magic number is {value}."),
]

QUESTION_TEMPLATES = [
    "What is {fact_name}?",
    "Can you remind me of {fact_name}?",
    "What did I say about {fact_name}?",
    "Do you remember {fact_name}?",
]


def _random_value(fact_name: str) -> str:
    if "number" in fact_name or "PIN" in fact_name or "metric" in fact_name:
        return str(random.randint(100, 9999))
    if "budget" in fact_name:
        return f"{random.randint(50, 999)},000"
    if "date" in fact_name:
        months = ["January", "March", "June", "August", "October", "December"]
        return f"{random.choice(months)} {random.randint(1, 28)}"
    if "color" in fact_name:
        return random.choice(["blue", "green", "red", "purple", "orange", "teal"])
    if "name" in fact_name or "cat" in fact_name:
        names = ["Whiskers", "Luna", "Max", "Bella", "Charlie", "Mochi"]
        return random.choice(names)
    if "hometown" in fact_name:
        cities = ["Toronto", "Berlin", "Nairobi", "Seoul", "Lisbon", "Auckland"]
        return random.choice(cities)
    # Generic: short alphanumeric
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=6))


def build_conversation(
    num_turns: int,
    num_facts: int = 3,
    seed: int = 42,
) -> Tuple[MemoryStore, List[Dict]]:
    """
    Build a synthetic conversation of `num_turns` user+assistant pairs.

    Hides `num_facts` facts at random positions.
    Returns (MemoryStore, list of {fact_name, value, turn_index, question}).
    """
    random.seed(seed)

    # Pick facts and insertion positions
    chosen = random.sample(FACT_TEMPLATES, min(num_facts, len(FACT_TEMPLATES)))
    insert_positions = sorted(random.sample(range(0, num_turns - 1), len(chosen)))

    facts = []
    for (fact_name, template), pos in zip(chosen, insert_positions):
        value = _random_value(fact_name)
        facts.append({
            "fact_name": fact_name,
            "value": value,
            "turn_index": pos * 2,          # user turn index (0-indexed over all turns)
            "statement": template.format(value=value),
            "question": random.choice(QUESTION_TEMPLATES).format(fact_name=fact_name),
        })

    store = MemoryStore()
    fact_by_pos = {f["turn_index"] // 2: f for f in facts}

    for i in range(num_turns):
        if i in fact_by_pos:
            user_msg = fact_by_pos[i]["statement"] + " " + random.choice(FILLER_TOPICS)
        else:
            user_msg = random.choice(FILLER_TOPICS)
        store.add("user", user_msg)
        store.add("assistant", "Got it, thanks for sharing.")

    return store, facts


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def truncation_baseline(
    history: MemoryStore,
    query: str,
    model: str,
    max_chars: int = 16_000,
) -> str:
    """Keep only the last `max_chars` characters of history."""
    history_str = history.to_string()
    truncated = history_str[-max_chars:] if len(history_str) > max_chars else history_str

    client = OpenAIClient(model=model)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. "
                "Below is a partial conversation history (older parts were truncated). "
                "Answer the question based on what you can see."
            ),
        },
        {"role": "user", "content": f"Conversation history:\n{truncated}\n\nQuestion: {query}"},
    ]
    return client.completion(messages)


def fullcontext_baseline(
    history: MemoryStore,
    query: str,
    model: str,
) -> str:
    """Feed entire history as context (works only if it fits in context window)."""
    history_str = history.to_string()
    client = OpenAIClient(model=model)
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer based on the conversation history.",
        },
        {"role": "user", "content": f"Conversation history:\n{history_str}\n\nQuestion: {query}"},
    ]
    return client.completion(messages)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_answer(predicted: str, expected: str) -> float:
    """Simple exact-match after lowercasing and stripping."""
    pred = predicted.lower().strip()
    exp = expected.lower().strip()
    if exp in pred:
        return 1.0
    # Partial: check if any word of expected appears
    exp_words = set(exp.split())
    pred_words = set(pred.split())
    overlap = exp_words & pred_words
    return len(overlap) / len(exp_words) if exp_words else 0.0


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run_eval(
    turn_counts: List[int],
    runs_per_scale: int,
    model: str,
    sub_model: str,
    threshold_chars: int,
    verbose: bool,
) -> List[Dict]:
    results = []

    for num_turns in turn_counts:
        print(f"\n{'='*60}")
        print(f"Scale: {num_turns} turns")
        print(f"{'='*60}")

        scale_results = {
            "num_turns": num_turns,
            "rlm": {"scores": [], "latencies": [], "tokens": []},
            "truncation": {"scores": [], "latencies": []},
            "fullcontext": {"scores": [], "latencies": []},
        }

        for run in range(runs_per_scale):
            history, facts = build_conversation(
                num_turns=num_turns, num_facts=3, seed=run * 100 + num_turns
            )

            # Pick one fact to test
            fact = facts[0]
            query = fact["question"]
            expected = fact["value"]

            print(f"\n  Run {run+1}/{runs_per_scale} | fact='{fact['fact_name']}' "
                  f"| inserted at turn {fact['turn_index']} | expected='{expected}'")

            # --- RLM Memory ---
            rlm = MemoryRLM(model=model, sub_model=sub_model,
                            max_iterations=8, verbose=verbose)
            t0 = time.time()
            rlm_answer = rlm.completion(history, query)
            rlm_latency = time.time() - t0
            rlm_score = score_answer(rlm_answer, expected)
            rlm_tokens = rlm.stats().get("total_tokens", 0)

            scale_results["rlm"]["scores"].append(rlm_score)
            scale_results["rlm"]["latencies"].append(rlm_latency)
            scale_results["rlm"]["tokens"].append(rlm_tokens)

            print(f"    RLM:          score={rlm_score:.2f} | "
                  f"latency={rlm_latency:.1f}s | tokens={rlm_tokens} | answer='{rlm_answer[:60]}'")

            # --- Truncation baseline ---
            t0 = time.time()
            trunc_answer = truncation_baseline(history, query, model)
            trunc_latency = time.time() - t0
            trunc_score = score_answer(trunc_answer, expected)

            scale_results["truncation"]["scores"].append(trunc_score)
            scale_results["truncation"]["latencies"].append(trunc_latency)

            print(f"    Truncation:   score={trunc_score:.2f} | "
                  f"latency={trunc_latency:.1f}s | answer='{trunc_answer[:60]}'")

            # --- Full context baseline (skip if too large) ---
            chars = history.total_chars()
            if chars < 100_000:
                t0 = time.time()
                full_answer = fullcontext_baseline(history, query, model)
                full_latency = time.time() - t0
                full_score = score_answer(full_answer, expected)
                scale_results["fullcontext"]["scores"].append(full_score)
                scale_results["fullcontext"]["latencies"].append(full_latency)
                print(f"    Full-context: score={full_score:.2f} | "
                      f"latency={full_latency:.1f}s | answer='{full_answer[:60]}'")
            else:
                print(f"    Full-context: SKIPPED (history too large: {chars:,} chars)")

        # Summarise scale
        def avg(lst):
            return round(sum(lst) / len(lst), 3) if lst else 0.0

        summary = {
            "num_turns": num_turns,
            "history_chars": history.total_chars(),
            "rlm_accuracy": avg(scale_results["rlm"]["scores"]),
            "truncation_accuracy": avg(scale_results["truncation"]["scores"]),
            "fullcontext_accuracy": avg(scale_results["fullcontext"]["scores"]),
            "rlm_avg_latency_s": avg(scale_results["rlm"]["latencies"]),
            "truncation_avg_latency_s": avg(scale_results["truncation"]["latencies"]),
            "rlm_avg_tokens": avg(scale_results["rlm"]["tokens"]),
        }
        results.append(summary)

        print(f"\n  SUMMARY @ {num_turns} turns:")
        print(f"    RLM accuracy:          {summary['rlm_accuracy']:.3f}")
        print(f"    Truncation accuracy:   {summary['truncation_accuracy']:.3f}")
        print(f"    Full-context accuracy: {summary['fullcontext_accuracy']:.3f}")
        print(f"    RLM avg latency:       {summary['rlm_avg_latency_s']:.1f}s")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NIAH evaluation for rlm_memory")
    parser.add_argument(
        "--turns", nargs="+", type=int, default=[20, 50, 100],
        help="Conversation lengths to test (default: 20 50 100)"
    )
    parser.add_argument(
        "--runs", type=int, default=3,
        help="Runs per scale (default: 3)"
    )
    parser.add_argument("--model", default="gpt-4o-mini", help="Main LLM model")
    parser.add_argument("--sub-model", default="gpt-4o-mini", help="Sub-agent model")
    parser.add_argument("--threshold", type=int, default=20_000, help="RLM threshold chars")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--output", default="niah_results.json", help="Output JSON file")
    args = parser.parse_args()

    print("rlm_memory — NIAH Evaluation")
    print(f"Model: {args.model} | Sub-model: {args.sub_model}")
    print(f"Scales: {args.turns} | Runs per scale: {args.runs}")

    results = run_eval(
        turn_counts=args.turns,
        runs_per_scale=args.runs,
        model=args.model,
        sub_model=args.sub_model,
        threshold_chars=args.threshold,
        verbose=args.verbose,
    )

    # Save results
    output_path = os.path.join(os.path.dirname(__file__), args.output)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Final table
    print("\n" + "=" * 70)
    print(f"{'Turns':>8} {'Chars':>10} {'RLM':>8} {'Truncate':>10} {'FullCtx':>9}")
    print("=" * 70)
    for r in results:
        print(
            f"{r['num_turns']:>8} "
            f"{r['history_chars']:>10,} "
            f"{r['rlm_accuracy']:>8.3f} "
            f"{r['truncation_accuracy']:>10.3f} "
            f"{r['fullcontext_accuracy']:>9.3f}"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
