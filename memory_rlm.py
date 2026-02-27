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
import random
import threading
import concurrent.futures
from typing import Optional, Dict, Any, List


def _retry_openai_call(fn, max_retries: int = 6, base_delay: float = 5.0):
    """
    Call fn() with exponential backoff on 429 (rate limit) errors.
    Retries up to max_retries times, sleeping base_delay * 2^attempt seconds.
    """
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            err_str = str(e)
            if "429" in err_str and attempt < max_retries:
                sleep_time = base_delay * (2 ** attempt) + random.uniform(0, 2)
                time.sleep(sleep_time)
            else:
                raise

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
from .query_classifier import classify_query, DATASET_TYPE_MAP


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
        max_workers: int = 20,
    ):
        self.model = model
        self.sub_model = sub_model
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.max_workers = max_workers
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.llm = OpenAIClient(api_key=api_key, model=model)
        self._last_stats: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def completion(
        self,
        history: MemoryStore,
        query: str,
        query_type: Optional[str] = None,
    ) -> str:
        """
        Search `history` for the answer to `query` using a REPL loop.

        Args:
            history    : MemoryStore with full conversation history
            query      : the question to answer
            query_type : optional override — one of FACTUAL, AGGREGATION,
                         KNOWLEDGE_UPDATE, TEMPORAL, PREFERENCE.
                         If None, a single cheap LLM call classifies the query.

        Returns the answer as a plain string.
        """
        t0 = time.time()
        self.llm.reset_usage()

        # --- Classify query type (one cheap call if not provided) ---
        import openai as _openai_mod
        _client = _openai_mod.OpenAI(api_key=self._api_key)
        if query_type is None:
            query_type = classify_query(query, _client, self.model)
        if self.verbose:
            print(f"  [MemoryRLM] query_type={query_type}")

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

        # ---- Parallel query primitive ----
        # Shared token counter for parallel sub-agent calls
        _parallel_tokens: Dict[str, int] = {"input": 0, "output": 0}
        _token_lock = threading.Lock()

        _api_key  = self._api_key
        _submodel = self.sub_model
        _workers  = self.max_workers
        _verbose  = self.verbose

        def llm_query_parallel(
            sessions_list: List[str],
            dates_list: List[str],
            question: str,
            workers: Optional[int] = None,
        ) -> List[str]:
            """
            Query ALL sessions in parallel using a thread pool.
            Dramatically faster than a sequential for loop (50 sessions -> ~5s vs ~200s).

            Args:
                sessions_list : list of session texts — pass `sessions`
                dates_list    : list of date labels  — pass `session_dates`
                question      : the question to answer
                workers       : max parallel threads (default: min(n_sessions, 20))

            Returns:
                List of finding strings for sessions where relevant info was found.
                Sessions that returned NOT_FOUND are excluded.

            Example:
                findings = llm_query_parallel(sessions, session_dates, question)
                if findings:
                    print("\\n".join(findings))
                else:
                    print("NOT_FOUND in any session")
            """
            import openai as _oai

            n = len(sessions_list)
            w = workers or min(n, _workers)

            def _query_one(idx: int) -> Optional[str]:
                session_text = sessions_list[idx]
                date = dates_list[idx] if idx < len(dates_list) else f"Session {idx}"
                prompt = (
                    f"Conversation session {idx} (date: {date}):\n"
                    f"{session_text}\n\n"
                    f"Question: {question}\n"
                    f"If this session contains relevant information, extract it concisely.\n"
                    f"If nothing relevant, reply exactly: NOT_FOUND"
                )
                client = _oai.OpenAI(api_key=_api_key)

                def _call():
                    return client.chat.completions.create(
                        model=_submodel,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=300,
                        temperature=0,
                    )

                resp = _retry_openai_call(_call)
                content = resp.choices[0].message.content.strip()
                # Track token usage
                with _token_lock:
                    _parallel_tokens["input"]  += resp.usage.prompt_tokens
                    _parallel_tokens["output"] += resp.usage.completion_tokens
                if "NOT_FOUND" not in content.upper():
                    if _verbose:
                        print(f"  [parallel] session {idx} → {content[:80]}")
                    return f"[Session {idx} | {date}]: {content}"
                return None

            t_start = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=w) as executor:
                results = list(executor.map(_query_one, range(n)))
            elapsed = time.time() - t_start

            findings = [r for r in results if r is not None]
            if _verbose:
                print(f"  [parallel] {n} sessions in {elapsed:.1f}s | "
                      f"{len(findings)} hits | "
                      f"{_parallel_tokens['input']+_parallel_tokens['output']} tok")
            return findings

        repl_env.globals["llm_query_parallel"] = llm_query_parallel

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

        # ---- Adaptive: inject query_type + type-specific helpers ----
        repl_env.globals["query_type"] = query_type

        # Parse question_date from query string (injected as "[Today's date: ...]")
        import re as _re2
        _date_match = _re2.search(r"\[Today's date:\s*([^\]]+)\]", query)
        repl_env.globals["question_date"] = (
            _date_match.group(1).strip() if _date_match else None
        )

        if query_type == "AGGREGATION":
            # Force full-scan wrapper: always queries ALL sessions
            _orig_query = query

            def aggregate_all_sessions(q=None):
                """Scan ALL sessions in parallel. Always use this for AGGREGATION queries."""
                actual_q = q if q is not None else _orig_query
                return llm_query_parallel(sessions_text, session_dates, actual_q)

            repl_env.globals["aggregate_all_sessions"] = aggregate_all_sessions

        elif query_type == "KNOWLEDGE_UPDATE":
            # Inject sessions reversed so index 0 = most recent session
            repl_env.globals["sessions_newest_first"] = list(reversed(sessions_text))
            repl_env.globals["dates_newest_first"]    = list(reversed(session_dates))

        elif query_type == "TEMPORAL":
            # Inject datetime utilities for date arithmetic
            from datetime import datetime as _dt

            def parse_date(s):
                """Parse a date string into a datetime object. Tries common formats."""
                for fmt in ["%Y-%m-%d", "%B %d, %Y", "%b %d, %Y",
                            "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d"]:
                    try:
                        return _dt.strptime(str(s).strip(), fmt)
                    except Exception:
                        pass
                return None

            repl_env.globals["datetime"]   = _dt
            repl_env.globals["parse_date"] = parse_date

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

            # Call main LLM (with retry on rate limits)
            response = _retry_openai_call(lambda: self.llm.completion(messages))
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
            response = _retry_openai_call(lambda: self.llm.completion(messages))
            final = find_final_answer(response)
            answer = final[1] if final else response.strip()

        # --- Record stats ---
        duration = time.time() - t0
        usage = self.llm.get_usage_summary()
        parallel_tok = _parallel_tokens["input"] + _parallel_tokens["output"]
        self._last_stats = {
            "iterations": iterations_used,
            "duration_s": round(duration, 2),
            "input_tokens": usage["total_input_tokens"],
            "output_tokens": usage["total_output_tokens"],
            "total_tokens": usage["total_tokens"] + parallel_tok,
            "parallel_tokens": parallel_tok,
            "cost_usd": round(usage["total_cost"], 6),
            "history_turns": history.total_turns(),
            "history_chars": history.total_chars(),
            "query_type": query_type,
        }

        return answer or ""

    def stats(self) -> Dict[str, Any]:
        """Return stats from the last completion() call."""
        return self._last_stats
