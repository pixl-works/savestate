"""A store in a Python list, for tests and examples. No extraction, no embeddings: it keeps what you give it."""

from __future__ import annotations

import re

from savestate.models import Memory


def _words(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


class InMemory:
    """`add` stores each user message as one memory; `search` ranks by shared words, newest first on ties.

    Real hosts extract facts from the conversation; this one does not, so it suits tests and examples, not quality.
    Seed it directly with `InMemory([Memory(...), ...])` or `store.memories.append(...)`.
    """

    def __init__(self, memories: list[Memory] | None = None):
        self.memories: list[Memory] = list(memories or [])

    def add(self, messages: list[dict[str, str]], date: str) -> list[Memory]:
        new = [
            Memory(m["content"].strip(), date, f"m{len(self.memories) + i + 1}")
            for i, m in enumerate(x for x in messages if x.get("role") == "user" and x.get("content", "").strip())
        ]
        self.memories += new
        return new

    def search(self, query: str, k: int) -> list[Memory]:
        q = _words(query)
        ranked = sorted(enumerate(self.memories), key=lambda p: (-len(q & _words(p[1].text)), -p[0]))
        return [m for _, m in ranked[:k]]
