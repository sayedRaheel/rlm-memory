"""
eval/synthetic_longmemeval.py

Synthetic LongMemEval-style evaluation for rlm_memory.

Mirrors the 5 official LongMemEval question categories:
  1. single-session-user  — fact stated once, recall it
  2. multi-session-user   — fact updated across sessions, recall latest
  3. temporal-reasoning   — "What did I say BEFORE/AFTER X happened?"
  4. knowledge-update     — user corrected a fact, recall the updated one
  5. abstention           — fact never stated, model must say "I don't know"

Each sample has:
  - A multi-session conversation history (50–200 turns)
  - A question
  - A gold answer

Compares:
  - RLM-memory (our method)
  - Truncation baseline (last 16K chars)
  - Full-context baseline (entire history if it fits)

Produces a paper-ready results table.
"""

import os
import sys
import json
import time
import random
import argparse
from typing import List, Dict, Tuple, Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MINIMAL = os.path.join(_ROOT, "..", "Recursive_language_model_rlm-minimal")
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.abspath(_MINIMAL))

from rlm_memory import MemoryStore
from rlm_memory.memory_rlm import MemoryRLM
from rlm.utils.llm import OpenAIClient

# ---------------------------------------------------------------------------
# Sample generators — one per LongMemEval question type
# ---------------------------------------------------------------------------

FILLER = [
    "The weather is nice today.",
    "I finished the documentation update.",
    "The team meeting went well.",
    "I reviewed the pull request.",
    "The deployment looks stable.",
    "I had a productive morning.",
    "The client call was rescheduled.",
    "I updated the sprint board.",
    "The new hire is onboarding well.",
    "I fixed a small bug in staging.",
    "The metrics look normal.",
    "I sent the weekly report.",
    "The design review is Friday.",
    "I synced with the product team.",
    "The database migration succeeded.",
    "I updated the onboarding docs.",
    "The CI pipeline is green.",
    "I archived old tickets.",
    "The load balancer is configured.",
    "I tested the new branch locally.",
    "The server response times are good.",
    "I reviewed the security findings.",
    "The sprint demo went smoothly.",
    "I updated the release notes.",
    "The API rate limits look fine.",
]


def _filler_turns(n: int, seed: int) -> List[Tuple[str, str]]:
    """Generate n (user, assistant) filler turn pairs."""
    random.seed(seed)
    turns = []
    for _ in range(n):
        msg = random.choice(FILLER)
        turns.append((msg, "Thanks for the update, noted."))
    return turns


def make_single_session_sample(seed: int) -> Dict:
    """Type 1: Fact stated once, recall it."""
    random.seed(seed)
    facts = [
        ("my project codename", "AURORA", "The project codename is AURORA."),
        ("my Q1 budget", "$340,000", "Our Q1 budget is $340,000."),
        ("my cat's name", "Mochi", "My cat's name is Mochi."),
        ("my lucky number", "42", "My lucky number is 42."),
        ("my hometown", "Nairobi", "I grew up in Nairobi."),
        ("my target metric", "87%", "Our target metric is 87%."),
        ("the server PIN", "9182", "The server PIN is 9182."),
        ("my favorite language", "Python", "My favorite programming language is Python."),
    ]
    fact_name, value, statement = random.choice(facts)

    pre_filler = _filler_turns(random.randint(5, 15), seed)
    post_filler = _filler_turns(random.randint(20, 50), seed + 1)

    store = MemoryStore()
    for u, a in pre_filler:
        store.add("user", u)
        store.add("assistant", a)
    store.add("user", statement)
    store.add("assistant", "Got it, I'll remember that.")
    for u, a in post_filler:
        store.add("user", u)
        store.add("assistant", a)

    return {
        "id": f"single_{seed}",
        "type": "single-session-user",
        "store": store,
        "question": f"What is {fact_name}?",
        "answer": value,
        "turns": store.total_turns(),
    }


def make_knowledge_update_sample(seed: int) -> Dict:
    """Type 4: Fact stated, then corrected. Recall the updated value."""
    random.seed(seed)
    updates = [
        ("my project codename", "PHOENIX", "NOVA",
         "The project codename is PHOENIX.", "Actually, we renamed it — the codename is now NOVA."),
        ("my budget", "$200,000", "$350,000",
         "Our budget is $200,000.", "We got more funding — the budget is now $350,000."),
        ("my deadline", "March 15", "April 30",
         "The deadline is March 15.", "The deadline has been pushed to April 30."),
        ("my team size", "5 people", "8 people",
         "Our team has 5 people.", "We hired 3 more — the team is now 8 people."),
    ]
    fact_name, old_val, new_val, first_stmt, update_stmt = random.choice(updates)

    pre = _filler_turns(random.randint(5, 10), seed)
    mid = _filler_turns(random.randint(15, 30), seed + 1)
    post = _filler_turns(random.randint(10, 20), seed + 2)

    store = MemoryStore()
    for u, a in pre:
        store.add("user", u)
        store.add("assistant", a)
    store.add("user", first_stmt)
    store.add("assistant", "Noted.")
    for u, a in mid:
        store.add("user", u)
        store.add("assistant", a)
    store.add("user", update_stmt)
    store.add("assistant", "Got it, updated.")
    for u, a in post:
        store.add("user", u)
        store.add("assistant", a)

    return {
        "id": f"update_{seed}",
        "type": "knowledge-update",
        "store": store,
        "question": f"What is {fact_name}?",
        "answer": new_val,
        "turns": store.total_turns(),
    }


def make_temporal_reasoning_sample(seed: int) -> Dict:
    """Type 3: Temporal ordering — what was discussed before/after an event."""
    random.seed(seed)

    pre_fact = f"fact_{random.randint(100, 999)}"
    event = "the system outage"

    store = MemoryStore()
    # Before the event
    store.add("user", f"My reference code is {pre_fact}.")
    store.add("assistant", "Got it.")
    for u, a in _filler_turns(random.randint(10, 25), seed):
        store.add("user", u)
        store.add("assistant", a)
    # The event
    store.add("user", f"We just had {event}.")
    store.add("assistant", "I hope everything is resolved.")
    # After the event
    for u, a in _filler_turns(random.randint(10, 25), seed + 1):
        store.add("user", u)
        store.add("assistant", a)

    return {
        "id": f"temporal_{seed}",
        "type": "temporal-reasoning",
        "store": store,
        "question": f"What reference code did I mention before {event}?",
        "answer": pre_fact,
        "turns": store.total_turns(),
    }


def make_abstention_sample(seed: int) -> Dict:
    """Type 5: Fact never stated — model should say it doesn't know."""
    random.seed(seed)
    store = MemoryStore()
    for u, a in _filler_turns(random.randint(30, 60), seed):
        store.add("user", u)
        store.add("assistant", a)

    return {
        "id": f"abstain_{seed}",
        "type": "abstention",
        "store": store,
        "question": "What is my social security number?",
        "answer": "unknown",
        "turns": store.total_turns(),
    }


def make_multi_session_sample(seed: int) -> Dict:
    """Type 2: Fact referenced across multiple sessions."""
    random.seed(seed)
    project = random.choice(["TITAN", "HELIOS", "CYGNUS", "ORION"])
    sessions = [
        f"We started working on project {project} today.",
        f"Project {project} is going well — we hit our first milestone.",
        f"The {project} team is growing.",
    ]

    store = MemoryStore()
    for i, session_start in enumerate(sessions):
        store.add("user", session_start)
        store.add("assistant", "Great to hear the progress.")
        for u, a in _filler_turns(random.randint(8, 15), seed + i * 10):
            store.add("user", u)
            store.add("assistant", a)

    return {
        "id": f"multi_{seed}",
        "type": "multi-session-user",
        "store": store,
        "question": "What is the name of the project we have been working on?",
        "answer": project,
        "turns": store.total_turns(),
    }


GENERATORS = [
    make_single_session_sample,
    make_knowledge_update_sample,
    make_temporal_reasoning_sample,
    make_abstention_sample,
    make_multi_session_sample,
]


def build_dataset(n_per_type: int, seed: int = 42) -> List[Dict]:
    samples = []
    for gen in GENERATORS:
        for i in range(n_per_type):
            samples.append(gen(seed + i * 37))
    random.seed(seed)
    random.shuffle(samples)
    return samples


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

import re

def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def token_f1(pred: str, gold: str) -> float:
    p_toks = normalize(pred).split()
    g_toks = normalize(gold).split()
    if not p_toks or not g_toks:
        return float(p_toks == g_toks)
    common = set(p_toks) & set(g_toks)
    if not common:
        return 0.0
    prec = len(common) / len(p_toks)
    rec = len(common) / len(g_toks)
    return 2 * prec * rec / (prec + rec)


def score(pred: str, gold: str, qtype: str) -> Dict:
    if qtype == "abstention":
        # Good if model says it doesn't know / no mention
        # Covers all natural phrasings the model may use
        uncertain_words = [
            "don't know", "do not know", "doesn't know",
            "no information", "no mention", "not mentioned",
            "unknown", "not available", "not provided",
            "haven't", "didn't mention", "no record",
            "cannot find", "can't find", "could not find",
            "not found", "not in", "not present",
            "not shared", "never mentioned", "never stated",
            "not discussed", "not specified", "not disclosed",
            "unable to find", "no data", "no details",
            "does not contain", "did not mention", "was not mentioned",
            "is not in", "are not in", "has not been mentioned",
        ]
        pred_l = pred.lower()
        em = float(any(w in pred_l for w in uncertain_words))
        return {"exact_match": em, "f1": em}
    em = float(normalize(gold) in normalize(pred) or normalize(pred) == normalize(gold))
    f1 = token_f1(pred, gold)
    return {"exact_match": em, "f1": f1}


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def truncation_baseline(store: MemoryStore, query: str, model: str,
                        max_chars: int = 16_000) -> str:
    hist = store.to_string()
    truncated = hist[-max_chars:] if len(hist) > max_chars else hist
    client = OpenAIClient(model=model)
    msgs = [
        {"role": "system",
         "content": "Answer based on the conversation history. Be concise. "
                    "If the information was never mentioned, say 'I don't know'."},
        {"role": "user",
         "content": f"Conversation history (may be truncated — older parts cut off):\n"
                    f"{truncated}\n\nQuestion: {query}"},
    ]
    return client.completion(msgs)


def fullcontext_baseline(store: MemoryStore, query: str, model: str) -> str:
    hist = store.to_string()
    client = OpenAIClient(model=model)
    msgs = [
        {"role": "system",
         "content": "Answer based on the full conversation history. Be concise. "
                    "If the information was never mentioned, say 'I don't know'."},
        {"role": "user",
         "content": f"Full conversation history:\n{hist}\n\nQuestion: {query}"},
    ]
    return client.completion(msgs)


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def run_eval(samples, model, sub_model, max_iter, verbose):
    type_results = {}
    all_rlm_em, all_rlm_f1 = [], []
    all_trunc_em, all_trunc_f1 = [], []
    all_full_em, all_full_f1 = [], []
    rlm_tokens, rlm_latencies, trunc_latencies = [], [], []

    for i, s in enumerate(samples):
        qtype = s["type"]
        store = s["store"]
        question = s["question"]
        answer = s["answer"]

        print(f"\n[{i+1}/{len(samples)}] type={qtype} | turns={s['turns']} | "
              f"chars={store.total_chars():,}")
        print(f"  Q: {question}")
        print(f"  A: {answer}")

        if qtype not in type_results:
            type_results[qtype] = {"rlm_em": [], "rlm_f1": [],
                                   "trunc_em": [], "trunc_f1": [],
                                   "full_em": [], "full_f1": []}

        # RLM Memory
        rlm = MemoryRLM(model=model, sub_model=sub_model,
                        max_iterations=max_iter, verbose=verbose)
        t0 = time.time()
        try:
            rlm_ans = rlm.completion(store, question)
        except Exception as e:
            rlm_ans = f"[ERROR: {e}]"
        rlm_lat = time.time() - t0
        rlm_s = score(rlm_ans, answer, qtype)
        toks = rlm.stats().get("total_tokens", 0)

        all_rlm_em.append(rlm_s["exact_match"])
        all_rlm_f1.append(rlm_s["f1"])
        rlm_tokens.append(toks)
        rlm_latencies.append(rlm_lat)
        type_results[qtype]["rlm_em"].append(rlm_s["exact_match"])
        type_results[qtype]["rlm_f1"].append(rlm_s["f1"])
        print(f"  RLM:         EM={rlm_s['exact_match']:.2f} F1={rlm_s['f1']:.2f} "
              f"| {rlm_lat:.1f}s | {toks} tok | '{rlm_ans[:70]}'")

        # Truncation baseline
        t0 = time.time()
        try:
            trunc_ans = truncation_baseline(store, question, model)
        except Exception as e:
            trunc_ans = f"[ERROR: {e}]"
        trunc_lat = time.time() - t0
        trunc_s = score(trunc_ans, answer, qtype)
        all_trunc_em.append(trunc_s["exact_match"])
        all_trunc_f1.append(trunc_s["f1"])
        trunc_latencies.append(trunc_lat)
        type_results[qtype]["trunc_em"].append(trunc_s["exact_match"])
        type_results[qtype]["trunc_f1"].append(trunc_s["f1"])
        print(f"  Truncation:  EM={trunc_s['exact_match']:.2f} F1={trunc_s['f1']:.2f} "
              f"| {trunc_lat:.1f}s | '{trunc_ans[:70]}'")

        # Full-context baseline
        try:
            full_ans = fullcontext_baseline(store, question, model)
        except Exception as e:
            full_ans = f"[ERROR: {e}]"
        full_s = score(full_ans, answer, qtype)
        all_full_em.append(full_s["exact_match"])
        all_full_f1.append(full_s["f1"])
        type_results[qtype]["full_em"].append(full_s["exact_match"])
        type_results[qtype]["full_f1"].append(full_s["f1"])
        print(f"  Full-ctx:    EM={full_s['exact_match']:.2f} F1={full_s['f1']:.2f} "
              f"| '{full_ans[:70]}'")

    def avg(lst): return round(sum(lst) / len(lst), 4) if lst else 0.0

    # Per-type breakdown
    type_summary = {}
    for t, r in type_results.items():
        type_summary[t] = {
            "n": len(r["rlm_em"]),
            "rlm_em": avg(r["rlm_em"]), "rlm_f1": avg(r["rlm_f1"]),
            "trunc_em": avg(r["trunc_em"]), "trunc_f1": avg(r["trunc_f1"]),
            "full_em": avg(r["full_em"]), "full_f1": avg(r["full_f1"]),
        }

    return {
        "n_samples": len(samples),
        "model": model,
        "overall": {
            "rlm":         {"em": avg(all_rlm_em),   "f1": avg(all_rlm_f1),
                            "avg_tokens": avg(rlm_tokens), "avg_latency_s": avg(rlm_latencies)},
            "truncation":  {"em": avg(all_trunc_em),  "f1": avg(all_trunc_f1),
                            "avg_latency_s": avg(trunc_latencies)},
            "full_context":{"em": avg(all_full_em),   "f1": avg(all_full_f1)},
        },
        "by_type": type_summary,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-type", type=int, default=10,
                        help="Samples per question type (default 10 → 50 total)")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--sub-model", default="gpt-4o-mini")
    parser.add_argument("--max-iter", type=int, default=8)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", default="synthetic_longmemeval_results.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("rlm_memory — Synthetic LongMemEval-Style Evaluation")
    print(f"Model: {args.model} | Sub-model: {args.sub_model}")
    print(f"Samples per type: {args.n_per_type} × 5 types = {args.n_per_type * 5} total")
    print(f"Question types: single-session, multi-session, temporal, knowledge-update, abstention")

    samples = build_dataset(args.n_per_type, args.seed)
    results = run_eval(samples, args.model, args.sub_model, args.max_iter, args.verbose)

    out = os.path.join(os.path.dirname(__file__), args.output)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    # -----------------------------------------------------------------------
    # Paper-ready results table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("TABLE 1 — Overall Results (LongMemEval-style benchmark)")
    print("=" * 72)
    print(f"{'Method':<22} {'EM':>8} {'F1':>8} {'Tokens':>10} {'Latency':>10}")
    print("-" * 72)
    o = results["overall"]
    print(f"{'Truncation':<22} {o['truncation']['em']:>8.4f} {o['truncation']['f1']:>8.4f} "
          f"{'—':>10} {o['truncation']['avg_latency_s']:>9.1f}s")
    print(f"{'Full-Context':<22} {o['full_context']['em']:>8.4f} {o['full_context']['f1']:>8.4f} "
          f"{'—':>10} {'—':>10}")
    print(f"{'RLM-Memory (ours)':<22} {o['rlm']['em']:>8.4f} {o['rlm']['f1']:>8.4f} "
          f"{o['rlm']['avg_tokens']:>10.0f} {o['rlm']['avg_latency_s']:>9.1f}s")
    print("=" * 72)

    print("\nTABLE 2 — Results by Question Type")
    print("=" * 72)
    print(f"{'Type':<26} {'RLM F1':>8} {'Trunc F1':>10} {'Full F1':>9}")
    print("-" * 72)
    type_order = ["single-session-user", "multi-session-user",
                  "temporal-reasoning", "knowledge-update", "abstention"]
    for t in type_order:
        if t in results["by_type"]:
            r = results["by_type"][t]
            print(f"  {t:<24} {r['rlm_f1']:>8.4f} {r['trunc_f1']:>10.4f} {r['full_f1']:>9.4f}")
    print("=" * 72)
    print(f"\nFull results saved to: {out}")


if __name__ == "__main__":
    main()
