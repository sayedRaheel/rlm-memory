"""
rag_baseline.py

Retrieval-Augmented Generation baseline for conversational memory.

Embeds all conversation turns using text-embedding-3-small, retrieves
the top-k most relevant turns at query time via cosine similarity,
and answers using gpt-4o-mini.

Used as a comparison point against RLM-Memory and truncation on
the LongMemEval-S benchmark.
"""

import os
import time
from typing import List, Dict, Optional, Any

import numpy as np
import openai

from .memory_store import MemoryStore


RAG_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question based only on the "
    "retrieved conversation excerpts provided below. Be concise — a few words "
    "or one sentence is sufficient. "
    "If the information is not present in the excerpts, say exactly: "
    "\"I don't know.\""
)


class RAGBaseline:
    """
    Embedding-based retrieval baseline for conversational memory.

    For each query:
      1. Embed all non-system turns in the MemoryStore (batched).
      2. Embed the query.
      3. Retrieve top-k turns by cosine similarity.
      4. Expand each retrieved turn by ±context_window surrounding turns.
      5. Pass retrieved context to the LLM and return the answer.

    Args:
        model:          LLM for answer generation (default gpt-4o-mini)
        embed_model:    Embedding model (default text-embedding-3-small)
        top_k:          Number of turns to retrieve (default 20)
        context_window: Surrounding turns to add per retrieved turn (default 1)
        api_key:        OpenAI API key (falls back to OPENAI_API_KEY env var)
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        embed_model: str = "text-embedding-3-small",
        top_k: int = 20,
        context_window: int = 1,
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.embed_model = embed_model
        self.top_k = top_k
        self.context_window = context_window
        self._client = openai.OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY")
        )
        self._last_stats: Dict[str, Any] = {}

    # ------------------------------------------------------------------

    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts in one API call. Returns (N, D) array."""
        # OpenAI allows up to 2048 inputs per batch
        resp = self._client.embeddings.create(
            model=self.embed_model,
            input=texts,
        )
        vecs = [e.embedding for e in sorted(resp.data, key=lambda x: x.index)]
        return np.array(vecs, dtype=np.float32)

    def _cosine_similarity(self, query_vec: np.ndarray,
                           turn_vecs: np.ndarray) -> np.ndarray:
        """Return cosine similarity between query_vec and each row of turn_vecs."""
        q = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        norms = np.linalg.norm(turn_vecs, axis=1, keepdims=True) + 1e-10
        t = turn_vecs / norms
        return t @ q  # shape (N,)

    # ------------------------------------------------------------------

    def query(self, store: MemoryStore, question: str) -> str:
        """
        Retrieve relevant turns from `store` and answer `question`.
        Returns the answer as a plain string.
        """
        t0 = time.time()

        # --- Filter to content turns (skip session boundary markers) ---
        all_turns = store.to_dicts()
        content_turns = [
            (i, t) for i, t in enumerate(all_turns)
            if t["role"] != "system" and t.get("content", "").strip()
        ]

        if not content_turns:
            return "I don't know."

        original_indices = [i for i, _ in content_turns]
        # Truncate individual turns to 2000 chars to stay within embedding limits
        texts = [t["content"][:2000] for _, t in content_turns]

        # --- Embed turns + query in one batched call ---
        all_texts = texts + [question]
        embeddings = self._embed_batch(all_texts)
        turn_vecs  = embeddings[:-1]
        query_vec  = embeddings[-1]

        embed_tok = len(all_texts) * 10  # rough estimate; not from API

        # --- Retrieve top-k by cosine similarity ---
        scores = self._cosine_similarity(query_vec, turn_vecs)
        k = min(self.top_k, len(content_turns))
        top_local_indices = np.argsort(scores)[-k:][::-1].tolist()

        # --- Expand with ±context_window surrounding turns (in original order) ---
        included: set = set()
        for local_idx in top_local_indices:
            orig_idx = original_indices[local_idx]
            for j in range(
                max(0, orig_idx - self.context_window),
                min(len(all_turns), orig_idx + self.context_window + 1),
            ):
                included.add(j)

        retrieved = [all_turns[j] for j in sorted(included)
                     if all_turns[j]["role"] != "system"]

        context = "\n".join(
            f"[{t['role'].upper()}]: {t['content']}"
            for t in retrieved
        )

        # --- Answer ---
        msgs = [
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Retrieved conversation excerpts:\n\n{context}\n\n"
                f"Question: {question}"
            )},
        ]
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=msgs,
            max_tokens=200,
            temperature=0,
        )
        answer = resp.choices[0].message.content.strip()
        latency = time.time() - t0

        self._last_stats = {
            "latency_s":       round(latency, 2),
            "total_tokens":    resp.usage.total_tokens,
            "retrieved_turns": len(retrieved),
            "top_k":           k,
        }

        return answer

    def stats(self) -> Dict[str, Any]:
        return self._last_stats
