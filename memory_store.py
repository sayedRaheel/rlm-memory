"""
memory_store.py

Append-only conversation history store.
Stores every turn (user + assistant) with metadata.
Used as the source-of-truth that the MemoryRLM searches over.
"""

import time
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class Turn:
    role: str           # "user" or "assistant"
    content: str
    timestamp: float
    turn_index: int


class MemoryStore:
    """
    Append-only store for conversation turns.

    Provides serialization helpers so MemoryRLM can expose
    the history as both a plain string and a structured list
    inside the REPL environment.
    """

    def __init__(self):
        self.turns: List[Turn] = []

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add(self, role: str, content: str) -> None:
        """Append a new turn to the history."""
        self.turns.append(Turn(
            role=role,
            content=content,
            timestamp=time.time(),
            turn_index=len(self.turns),
        ))

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_string(self) -> str:
        """
        Flat text representation exposed as `context` in the REPL.

        Format per line:
            [Turn 0][USER]: hello
            [Turn 1][ASSISTANT]: hi there
        """
        lines = []
        for t in self.turns:
            lines.append(f"[Turn {t.turn_index}][{t.role.upper()}]: {t.content}")
        return "\n".join(lines)

    def to_dicts(self) -> List[Dict]:
        """
        Structured representation exposed as `history_turns` in the REPL.
        Each element: {"turn_index", "role", "content", "timestamp"}
        """
        return [
            {
                "turn_index": t.turn_index,
                "role": t.role,
                "content": t.content,
                "timestamp": t.timestamp,
            }
            for t in self.turns
        ]

    def to_openai_messages(self) -> List[Dict[str, str]]:
        """Return history in OpenAI messages format (for normal-mode chat)."""
        return [{"role": t.role, "content": t.content} for t in self.turns]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def total_chars(self) -> int:
        return sum(len(t.content) for t in self.turns)

    def total_turns(self) -> int:
        return len(self.turns)

    def __len__(self) -> int:
        return len(self.turns)

    def __repr__(self) -> str:
        return f"MemoryStore({len(self.turns)} turns, {self.total_chars():,} chars)"
