"""
rlm_memory — RLM-powered long-context conversation memory.

Drop-in replacement for standard LLM chat that handles arbitrarily
long conversation histories using the Recursive Language Model paradigm.

No vector database. No summarization. No truncation.

Quick start:
    from rlm_memory import MemoryChat

    chat = MemoryChat(model="gpt-4o")
    chat.send("My project budget is $500K")
    answer = chat.send("What was my project budget?")
"""

from .chat import MemoryChat
from .memory_store import MemoryStore, Turn
from .memory_rlm import MemoryRLM

__all__ = ["MemoryChat", "MemoryStore", "Turn", "MemoryRLM"]
__version__ = "0.1.0"
