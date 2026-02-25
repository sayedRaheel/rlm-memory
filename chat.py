"""
chat.py

MemoryChat — drop-in long-context chat interface.

Works exactly like a normal LLM chat, but handles arbitrarily long
conversation histories by switching to RLM memory mode once the
history exceeds a configurable character threshold.

No vector database. No summarization. No truncation.

Usage:
    from rlm_memory import MemoryChat

    chat = MemoryChat(model="gpt-4o")
    chat.send("My project budget is $500K")
    # ... 80 more turns ...
    answer = chat.send("What was my project budget?")
    # → "$500K"
"""

import os
import sys
from typing import Optional, Dict, Any

_MINIMAL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "Recursive_language_model_rlm-minimal"
)
if os.path.abspath(_MINIMAL_PATH) not in sys.path:
    sys.path.insert(0, os.path.abspath(_MINIMAL_PATH))

from rlm.utils.llm import OpenAIClient

from .memory_store import MemoryStore
from .memory_rlm import MemoryRLM


class MemoryChat:
    """
    Drop-in long-context chat with automatic RLM memory retrieval.

    Two modes:
      - Normal mode  : history fits in context window — standard LLM call
      - RLM mode     : history too long — LLM searches history via REPL

    The switch happens automatically at `threshold_chars`.

    Args:
        model:             Main LLM for both normal chat and RLM orchestration
        sub_model:         Cheaper sub-agent model for semantic chunk queries
        threshold_chars:   Switch to RLM mode when history exceeds this (default 20K chars)
        max_rlm_iterations: Max REPL iterations per RLM query
        api_key:           OpenAI API key (falls back to OPENAI_API_KEY env var)
        verbose:           Print mode and iteration info
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        sub_model: str = "gpt-4o-mini",
        threshold_chars: int = 20_000,
        max_rlm_iterations: int = 10,
        api_key: Optional[str] = None,
        verbose: bool = False,
    ):
        self.model = model
        self.threshold_chars = threshold_chars
        self.verbose = verbose

        self.store = MemoryStore()
        self._normal_llm = OpenAIClient(api_key=api_key, model=model)
        self._memory_rlm = MemoryRLM(
            model=model,
            sub_model=sub_model,
            max_iterations=max_rlm_iterations,
            api_key=api_key,
            verbose=verbose,
        )

        self._normal_mode_count = 0
        self._rlm_mode_count = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def send(self, message: str) -> str:
        """
        Send a user message and return the assistant's response.

        Automatically uses RLM memory retrieval when history is long.
        """
        self.store.add("user", message)

        history_chars = self.store.total_chars()
        use_rlm = history_chars >= self.threshold_chars

        if self.verbose:
            mode = "RLM" if use_rlm else "normal"
            print(
                f"[MemoryChat] Turn {self.store.total_turns()} | "
                f"{history_chars:,} chars | mode={mode}"
            )

        if use_rlm:
            response = self._rlm_response(message)
            self._rlm_mode_count += 1
        else:
            response = self._normal_response()
            self._normal_mode_count += 1

        self.store.add("assistant", response)
        return response

    def reset(self) -> None:
        """Clear conversation history and reset counters."""
        self.store = MemoryStore()
        self._normal_mode_count = 0
        self._rlm_mode_count = 0

    def stats(self) -> Dict[str, Any]:
        """Return session statistics."""
        return {
            "total_turns": self.store.total_turns(),
            "total_chars": self.store.total_chars(),
            "normal_mode_calls": self._normal_mode_count,
            "rlm_mode_calls": self._rlm_mode_count,
            "threshold_chars": self.threshold_chars,
            "last_rlm_stats": self._memory_rlm.stats(),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normal_response(self) -> str:
        """Standard LLM call with full history in context."""
        messages = self.store.to_openai_messages()
        return self._normal_llm.completion(messages)

    def _rlm_response(self, query: str) -> str:
        """RLM memory search response."""
        # Build a history store that excludes the last user message
        # (that's already the query we're passing explicitly)
        search_history = MemoryStore()
        for turn in self.store.turns[:-1]:
            search_history.turns.append(turn)

        return self._memory_rlm.completion(
            history=search_history,
            query=query,
        )

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"MemoryChat(model={self.model}, "
            f"turns={self.store.total_turns()}, "
            f"chars={self.store.total_chars():,})"
        )
