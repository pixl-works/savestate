"""Whose fact is it? Drops memories about other people's own lives from a recall pool.

Hosts store everything the user said, including what they said about friends and relatives. A memory like "her
brother moved to Oslo" competes with the user's own "moved to Oslo" for the same question. The filter classifies each
memory text once (cached, optionally on disk) as "user" or "other"; unsure stays "user", so a miss costs a little
noise and never a lost fact. Questions about who the user or their family IS skip the filter (`IDENTITY_Q`),
because "does the user have a mother?" is answered by any memory about the mother.

It assumes the questions are about the user. If your agent is asked about other people, leave it off.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
from pathlib import Path

from savestate.card import write_private
from savestate.models import LLM, Memory
from savestate.prompts import OWNER_PROMPT

IDENTITY_Q = re.compile(
    r"\b(mother|mom|mum|father|dad|parents?|siblings?|brothers?|sisters?|born|birth\w*|name|gender|"
    r"male|female|degree|universit\w*|college|major|studied|stud(y|ies)|education\w*|grew up|"
    r"hometown|nationality)\b",
    re.I,
)


class Owners:
    """memory text -> "user" | "other", asked of the model once per distinct text."""

    def __init__(self, llm: LLM, path: str | os.PathLike[str] | None = None):
        self.llm = llm
        self.path = Path(path).expanduser() if path else None
        self.map: dict[str, str] = {}
        if self.path and self.path.exists():
            with contextlib.suppress(ValueError):  # a cut-off write; it is only a cache, the model is asked again
                cached = json.loads(self.path.read_text())
                if isinstance(cached, dict):
                    self.map = {t: a for t, a in cached.items() if a in ("user", "other")}

    def classify(self, texts: list[str]) -> None:
        todo = [t for t in dict.fromkeys(texts) if t not in self.map]
        if not todo:
            return
        handles = {f"m{i}": t for i, t in enumerate(todo, 1)}
        answer = self.llm.complete_json(
            OWNER_PROMPT + "\n".join(f"[{h}] {' '.join(t.split())}" for h, t in handles.items())
        )
        for o in answer.get("owners") or []:
            if isinstance(o, dict) and o.get("id") in handles and o.get("about") in ("user", "other"):
                self.map[handles[o["id"]]] = o["about"]
        if self.path:
            write_private(self.path, json.dumps(self.map, ensure_ascii=False))

    def keep(self, text: str) -> bool:
        return self.map.get(text) != "other"

    def filter(self, question: str, memories: list[Memory]) -> list[Memory]:
        """The memories that are about the user; everything passes for identity and family questions."""
        if IDENTITY_Q.search(question):
            return memories
        self.classify([m.text for m in memories])
        return [m for m in memories if self.keep(m.text)]
