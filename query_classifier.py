"""
query_classifier.py

Single-call query type classifier for RLM-Memory adaptive retrieval.

Classifies a memory question into one of 5 types so the engine can
apply the optimal retrieval strategy for that question shape.
"""

from typing import Optional

QUERY_TYPES = [
    "FACTUAL",
    "AGGREGATION",
    "KNOWLEDGE_UPDATE",
    "TEMPORAL",
    "PREFERENCE",
]

# Map LongMemEval dataset question_type strings → our internal types
DATASET_TYPE_MAP = {
    "single-session-user":       "FACTUAL",
    "single-session-assistant":  "FACTUAL",
    "single-session-preference": "PREFERENCE",
    "multi-session":             "AGGREGATION",
    "temporal-reasoning":        "TEMPORAL",
    "knowledge-update":          "KNOWLEDGE_UPDATE",
}

_CLASSIFIER_PROMPT = """\
Classify this memory question into exactly one type.

FACTUAL         — recall a single fact from one session
                  (name, job title, city, item bought, yes/no fact)

AGGREGATION     — requires summing, counting, or listing across multiple sessions
                  (total spent, list of all hobbies, how many times did X happen,
                   names of all the workshops attended)

KNOWLEDGE_UPDATE — a fact was stated, then later changed or corrected;
                   need the most recent / updated value

TEMPORAL        — requires date arithmetic or chronological ordering
                  (how many days before X did Y happen, what did I do after Z date,
                   how long between event A and event B)

PREFERENCE      — subjective preference, opinion, or personalised recommendation
                  (suggest, recommend, what should I, based on my preferences)

Question: {question}

Reply with exactly one word from: FACTUAL, AGGREGATION, KNOWLEDGE_UPDATE, TEMPORAL, PREFERENCE\
"""


def classify_query(
    question: str,
    client,
    model: str = "gpt-4o-mini",
) -> str:
    """
    Classify a memory question into one of 5 retrieval types.

    Args:
        question : the user's memory question
        client   : openai.OpenAI instance
        model    : model to use (default gpt-4o-mini — cheap single call)

    Returns:
        One of: "FACTUAL", "AGGREGATION", "KNOWLEDGE_UPDATE", "TEMPORAL", "PREFERENCE"
    """
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": _CLASSIFIER_PROMPT.format(question=question),
            }],
            max_tokens=10,
            temperature=0,
        )
        raw = resp.choices[0].message.content.strip().upper()
        for t in QUERY_TYPES:
            if t in raw:
                return t
    except Exception:
        pass
    return "FACTUAL"  # safe default
