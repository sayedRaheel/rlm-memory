"""
memory_rlm.py

Core RLM engine for conversation memory retrieval.

When history is too long to fit in context, MemoryRLM puts the full
conversation into a Python REPL environment and lets the LLM write
code to search/retrieve only the relevant parts — exactly as the
original RLM paper does for documents, but applied to live chat history.
"""

import os
import sys
import time
from typing import Optional, Dict, Any, List

# ---------------------------------------------------------------------------
# Import from existing rlm-minimal codebase
# ---------------------------------------------------------------------------
_MINIMAL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "Recursive_language_model_rlm-minimal"
)
if os.path.abspath(_MINIMAL_PATH) not in sys.path:
    sys.path.insert(0, os.path.abspath(_MINIMAL_PATH))

from rlm.repl import REPLEnv
from rlm.utils.llm import OpenAIClient
from rlm.utils.utils import (
    find_final_answer,
    add_execution_result_to_messages,
    format_execution_result,
)
import re as _re

def find_code_blocks(text: str) -> List[str]:
    """
    Extended version: accepts ```repl, ```python, ```py blocks.
    The original RLM only matches ```repl — this ensures the model's code
    executes regardless of which language tag it uses.
    """
    pattern = r'```(?:repl|python|py)\s*\n(.*?)\n```'
    return [m.group(1).strip() for m in _re.finditer(pattern, text, _re.DOTALL)]

from .memory_store import MemoryStore
from .prompts import MEMORY_SYSTEM_PROMPT, MEMORY_ACTION_PROMPT, MEMORY_FINAL_PROMPT


def _split_into_sessions(history_turns: List[Dict]) -> tuple:
    """
    Split flat history_turns into per-session text chunks and date labels.

    A new session starts whenever a 'system' turn contains '--- Session'.
    If no session markers exist, each 50-turn block becomes one session.

    Returns:
        sessions_text  : list[str]  — full text of each session
        session_dates  : list[str]  — date/label for each session
    """
    sessions_text: List[str] = []
    session_dates: List[str] = []
    current_lines: List[str] = []
    current_date: str = ""

    for turn in history_turns:
        content = turn.get("content", "")
        role    = turn.get("role", "user")

        # Session boundary marker injected by real_longmemeval.py / build_memory_store()
        if role == "system" and "--- Session" in content:
            if current_lines:
                sessions_text.append("\n".join(current_lines))
                session_dates.append(current_date)
            current_lines = []
            # Extract date from marker: [--- Session N | DATE ---]
            try:
                current_date = content.split("|")[1].strip().rstrip("-").strip()
            except Exception:
                current_date = content
        else:
            current_lines.append(f"[{role.upper()}]: {content}")

    # Flush last session
    if current_lines:
        sessions_text.append("\n".join(current_lines))
        session_dates.append(current_date)

    # Fallback: if no session markers, chunk into blocks of 50 turns
    if not sessions_text:
        chunk_size = 50
        all_lines = [
            f"[{t.get('role','user').upper()}]: {t.get('content','')}"
            for t in history_turns
        ]
        sessions_text = [
            "\n".join(all_lines[i:i+chunk_size])
            for i in range(0, len(all_lines), chunk_size)
        ]
        session_dates = [f"Block {i+1}" for i in range(len(sessions_text))]

    return sessions_text, session_dates


class MemoryRLM:
    """
    RLM-powered conversation memory retrieval engine.

    Given a MemoryStore (full conversation history) and a query,
    the LLM uses a Python REPL to programmatically search the history
    and retrieve the answer — no summarization, no truncation, no vector DB.

    Args:
        model:          Main LLM model (orchestrator)
        sub_model:      Sub-agent model (cheaper, for semantic chunk queries)
        max_iterations: Max REPL loop iterations per query
        api_key:        OpenAI API key (falls back to OPENAI_API_KEY env var)
        verbose:        Print iteration progress
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        sub_model: str = "gpt-4o-mini",
        max_iterations: int = 10,
        api_key: Optional[str] = None,
        verbose: bool = False,
    ):
        self.model = model
        self.sub_model = sub_model
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.llm = OpenAIClient(api_key=api_key, model=model)
        self._last_stats: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def completion(self, history: MemoryStore, query: str) -> str:
        """
        Search `history` for the answer to `query` using a REPL loop.

        Returns the answer as a plain string.
        """
        t0 = time.time()
        self.llm.reset_usage()

        # --- Build REPL environment with history as context ---
        history_str = history.to_string()
        history_turns = history.to_dicts()

        # Build per-session data structures for sub-agent delegation
        sessions_text, session_dates = _split_into_sessions(history_turns)

        repl_env = REPLEnv(
            context_str=history_str,
            recursive_model=self.sub_model,
        )

        # ---- Primary: sub-agent-ready session list ----
        repl_env.globals["sessions"]      = sessions_text   # list[str] one per session
        repl_env.globals["session_dates"] = session_dates   # list[str] date label per session

        # ---- Secondary: flat helpers for quick checks ----
        def search_history(keyword: str) -> List[Dict]:
            """Keyword scan across all turns (case-insensitive). Use for quick checks."""
            kw = keyword.lower()
            return [t for t in history_turns if kw in t["content"].lower()]

        def get_recent(n: int) -> List[Dict]:
            """Return the last n turns."""
            return history_turns[-max(1, int(n)):]

        repl_env.globals["history_turns"]   = history_turns
        repl_env.globals["search_history"]  = search_history
        repl_env.globals["get_recent"]      = get_recent

        # --- Conversation messages ---
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": MEMORY_SYSTEM_PROMPT}
        ]

        answer: Optional[str] = None
        iterations_used = 0

        for iteration in range(self.max_iterations):
            iterations_used = iteration + 1

            # Build action prompt for this iteration
            action = MEMORY_ACTION_PROMPT.format(
                query=query,
                iteration=iteration + 1,
                max_iter=self.max_iterations,
            )
            messages.append({"role": "user", "content": action})

            # Call main LLM
            response = self.llm.completion(messages)
            messages.append({"role": "assistant", "content": response})

            if self.verbose:
                preview = response[:120].replace("\n", " ")
                print(f"  [MemoryRLM iter {iteration+1}] {preview}...")

            # --- Execute code blocks FIRST so variables are in scope ---
            code_blocks = find_code_blocks(response)
            if code_blocks:
                for code in code_blocks:
                    result = repl_env.code_execution(code)
                    formatted = format_execution_result(
                        result.stdout, result.stderr, result.locals
                    )
                    messages = add_execution_result_to_messages(
                        messages, code, formatted
                    )

                    if self.verbose:
                        # Show more output so the full sub-agent loop is visible
                        out_preview = result.stdout[:500].replace(chr(10), " ")
                        err_preview = result.stderr[:200].replace(chr(10), " ") if result.stderr else ""
                        print(f"  [REPL out] {out_preview}")
                        if err_preview:
                            print(f"  [REPL err] {err_preview}")

                    # Check if REPL stdout declares a final answer
                    repl_final = find_final_answer(result.stdout)
                    if repl_final:
                        answer = repl_final[1]
                        break

                if answer:
                    break

            # --- Now check the response text for FINAL() / FINAL_VAR() ---
            final = find_final_answer(response)
            if final:
                answer_type, content = final
                if answer_type == "FINAL_VAR":
                    var_name = content.strip().strip('"').strip("'")
                    answer = str(
                        repl_env.locals.get(var_name, f"[Variable '{var_name}' not found]")
                    )
                else:  # FINAL
                    # If content looks like a bare variable name, resolve it
                    stripped = content.strip()
                    if stripped.isidentifier() and stripped in repl_env.locals:
                        answer = str(repl_env.locals[stripped])
                    else:
                        answer = content
                break

        # --- Force final answer if max iterations reached ---
        if not answer:
            if self.verbose:
                print("  [MemoryRLM] Max iterations reached — forcing final answer")
            messages.append({
                "role": "user",
                "content": MEMORY_FINAL_PROMPT.format(query=query),
            })
            response = self.llm.completion(messages)
            final = find_final_answer(response)
            answer = final[1] if final else response.strip()

        # --- Record stats ---
        duration = time.time() - t0
        usage = self.llm.get_usage_summary()
        self._last_stats = {
            "iterations": iterations_used,
            "duration_s": round(duration, 2),
            "input_tokens": usage["total_input_tokens"],
            "output_tokens": usage["total_output_tokens"],
            "total_tokens": usage["total_tokens"],
            "cost_usd": round(usage["total_cost"], 6),
            "history_turns": history.total_turns(),
            "history_chars": history.total_chars(),
        }

        return answer or ""

    def stats(self) -> Dict[str, Any]:
        """Return stats from the last completion() call."""
        return self._last_stats
