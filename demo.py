"""
demo.py

Quick demo of rlm_memory — runs a 60-turn synthetic conversation
with facts hidden early in the history, then asks recall questions.

Usage:
    python demo.py
    python demo.py --model gpt-4o-mini --threshold 5000
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rlm_memory import MemoryChat


def main():
    parser = argparse.ArgumentParser(description="rlm_memory demo")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model to use")
    parser.add_argument("--sub-model", default="gpt-4o-mini", help="Sub-agent model")
    parser.add_argument(
        "--threshold", type=int, default=5000,
        help="Chars before switching to RLM mode (low = trigger RLM sooner for demo)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("rlm_memory — Demo")
    print(f"Model: {args.model} | Threshold: {args.threshold:,} chars")
    print("=" * 60)

    chat = MemoryChat(
        model=args.model,
        sub_model=args.sub_model,
        threshold_chars=args.threshold,
        verbose=True,
    )

    # --- Phase 1: Plant facts early in conversation ---
    print("\n[Phase 1] Planting facts early in history...")

    facts = [
        ("user", "My project codename is AURORA."),
        ("user", "The deadline for AURORA is April 30th."),
        ("user", "Our Q1 budget is $240,000."),
    ]

    for role, msg in facts:
        response = chat.send(msg)
        print(f"  User: {msg}")
        print(f"  Asst: {response[:80]}\n")

    # --- Phase 2: Add lots of filler to push history past threshold ---
    print("[Phase 2] Adding filler turns to grow history...")

    filler = [
        "The weather has been nice lately.",
        "I updated the documentation for the API module.",
        "The CI pipeline is passing all checks.",
        "I reviewed the pull request from the team.",
        "The client call went well this morning.",
        "I need to update the roadmap slides.",
        "The new hire starts next Monday.",
        "I fixed a small alignment issue in the UI.",
        "The database migration ran smoothly.",
        "I sent the weekly status update to the team.",
        "The server metrics look stable today.",
        "I archived some old tickets in the backlog.",
        "The design review is scheduled for Wednesday.",
        "I updated the onboarding guide for new engineers.",
        "The integration tests passed after the refactor.",
        "I synced with the product team about priorities.",
        "The load balancer config was updated.",
        "I reviewed the security audit findings.",
        "The sprint retrospective is Friday afternoon.",
        "I tested the new feature branch locally.",
    ]

    for msg in filler:
        chat.send(msg)

    print(f"\n  History: {chat.store.total_turns()} turns | {chat.store.total_chars():,} chars")

    # --- Phase 3: Ask recall questions (should trigger RLM mode) ---
    print("\n[Phase 3] Asking recall questions (RLM mode should activate)...")
    print("-" * 60)

    questions = [
        ("What is the project codename?", "AURORA"),
        ("What is the deadline?", "April 30th"),
        ("What is the Q1 budget?", "$240,000"),
    ]

    correct = 0
    for question, expected in questions:
        print(f"\nQuestion: {question}")
        answer = chat.send(question)
        print(f"Answer:   {answer}")
        print(f"Expected: {expected}")
        hit = expected.lower() in answer.lower()
        correct += int(hit)
        print(f"Result:   {'CORRECT' if hit else 'WRONG'}")

    print("\n" + "=" * 60)
    print(f"Score: {correct}/{len(questions)}")
    s = chat.stats()
    print(f"Normal mode calls: {s['normal_mode_calls']}")
    print(f"RLM mode calls:    {s['rlm_mode_calls']}")
    if s["last_rlm_stats"]:
        rlm = s["last_rlm_stats"]
        print(f"Last RLM: {rlm.get('iterations')} iters | "
              f"{rlm.get('total_tokens')} tokens | "
              f"${rlm.get('cost_usd', 0):.5f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
