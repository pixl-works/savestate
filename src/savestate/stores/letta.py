"""Letta (letta-ai) archival memory through its SDK (extra `letta`, letta-client==1.11.0).

This is the setup MemConflict measured, not Letta's agent loop: each conversation goes into one agent's ARCHIVAL
memory as raw dialogue chunks, and `search` is Letta's own archival search. Neither call uses the agent's LLM; Letta
only embeds. Measured on self-hosted Letta 0.16.8, 5 tuning personas, with the full layer (owner filter + reranker):
changing facts 44.5 -> 75.3, situational 61.7 -> 75.9. The card alone failed the never-change guardrail there, so
use `Layer(..., owner_filter=True, reranker=...)` on Letta.

Two things differ from stores that extract memories themselves:

* Facts: archival memory keeps raw text and reports nothing derived, so `add` asks `llm` for the user's facts in the
  conversation (FACTS_PROMPT) and returns those; that is what the card is fed. They are not written into Letta.
* Exchanges: a chunk holds up to 8 messages (1-2k characters), longer than a cross-encoder reads (256-512 tokens).
  `search` splits each chunk into exchanges (a user message and the replies that follow it), so the reranker scores
  short units. `k` is the number of chunks fetched; more exchanges than that come back.

Every conversation's passages share one tag, so after writing, `add` searches that tag alone to prove they can be found
again (a ranking-free check: the old global probe gave false alarms late in long histories). A passage without an
embedding or a failed check deletes what this call wrote and raises: nothing half-written stays behind.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
from typing import Any

from savestate.models import LLM, Memory, check_key_url, check_user_id
from savestate.prompts import FACTS_PROMPT

BATCH_SIZE = 8  # messages per passage, as MemConflict's eval_letta writes them
CHUNK_CHARS = 8000  # passage cap, as eval_letta
_SPLIT = re.compile(r"\n(?=(?:user|assistant): )")


class LettaError(RuntimeError):
    pass


def chunks(messages: list[dict[str, str]]) -> list[str]:
    """Batches of 8 messages as `role: content` lines, at most 8000 characters each."""
    msgs = [m for m in messages if m.get("role") in ("user", "assistant") and str(m.get("content") or "").strip()]
    out: list[str] = []
    for i in range(0, len(msgs), BATCH_SIZE):
        text = "\n".join(f"{m['role']}: {m['content']}" for m in msgs[i : i + BATCH_SIZE])
        out += [text[j : j + CHUNK_CHARS] for j in range(0, len(text), CHUNK_CHARS)]
    return out


def exchanges(text: str) -> list[str]:
    """One chunk -> its exchanges: each user message with the assistant replies that follow it."""
    units: list[str] = []
    for m in _SPLIT.split(text):
        if m.startswith("user: ") or not units:
            units.append(m)
        else:
            units[-1] += "\n" + m
    return units


class Letta:
    """Args:
    agent_id: the Letta agent whose archival memory this is (create it in Letta, with the embedder you want).
    url: Letta server base; Letta Cloud is https://api.letta.com.
    llm: extracts the user's facts for the card in `add` (search needs none).
    api_key: for Letta Cloud (or LETTA_API_KEY); a self-hosted server needs none. Refused over plain http to a host
        other than this machine unless `allow_http=True`.
    split: return exchanges instead of whole chunks from `search` (what was measured with the layer).
    verify: after writing, check the conversation's passages can be found again.
    """

    def __init__(
        self,
        agent_id: str,
        url: str = "http://127.0.0.1:8283",
        llm: LLM | None = None,
        *,
        api_key: str | None = None,
        split: bool = True,
        verify: bool = True,
        timeout: float = 300.0,
        allow_http: bool = False,
        client: Any = None,
    ):
        self.user_id = self.agent_id = check_user_id(agent_id)  # Layer binds the card to it
        self.llm, self.split, self.verify = llm, split, verify
        if client is None:
            key = api_key or os.environ.get("LETTA_API_KEY")
            check_key_url(url, key, allow_http)
            try:
                from letta_client import Letta as LettaSDK
            except ImportError as exc:
                raise ImportError("the Letta adapter needs the extra: pip install 'savestate-memory[letta]'") from exc
            client = LettaSDK(base_url=url, api_key=key or "local", timeout=timeout, max_retries=4)
        self._c = client

    def _found(self, query: str, top_k: int, tags: list[str] | None = None) -> list[Any]:
        kw = {"tags": tags} if tags else {}
        return list(self._c.agents.passages.search(self.agent_id, query=query, top_k=top_k, **kw).results)

    def search(self, query: str, k: int) -> list[Memory]:
        out: list[Memory] = []
        for r in self._found(query, k):
            date = str(r.timestamp or "")[:10] or None
            parts = exchanges(r.content) if self.split else [r.content]
            out += [Memory(p, date, f"{r.id}#{n}" if self.split else r.id) for n, p in enumerate(parts)]
        return out

    def facts(self, messages: list[dict[str, str]], date: str) -> list[Memory]:
        """The user's facts in one conversation, as the card wants them (one LLM call)."""
        if self.llm is None:
            raise TypeError("Letta.add needs llm=: archival memory stores raw text, so the card's facts come from it")
        dialogue = "\n".join(f"{m['role']}: {m['content']}" for m in messages if str(m.get("content") or "").strip())
        if not dialogue:
            return []
        ans = self.llm.complete_json(FACTS_PROMPT.format(date=date, dialogue=dialogue))
        if not isinstance(ans.get("facts"), list):
            raise LettaError(
                "fact extraction returned no JSON facts list (refusal or a parse failure); nothing written"
            )
        return [Memory(f.strip(), date) for f in ans["facts"] if isinstance(f, str) and f.strip()]

    def add(self, messages: list[dict[str, str]], date: str) -> list[Memory]:
        """Write one conversation into archival memory; return the user's facts in it (for the card)."""
        new = self.facts(messages, date)  # first: a failed extraction leaves Letta untouched
        texts = chunks(messages)
        if not texts:
            return new
        body = json.dumps([date, [[m.get("role"), m.get("content")] for m in messages]], sort_keys=True)
        tag = "savestate-" + hashlib.sha256(body.encode()).hexdigest()[:12]
        made: list[str] = []
        try:
            for text in texts:
                for p in self._c.agents.passages.create(
                    self.agent_id, text=text, tags=[tag], created_at=f"{date}T10:00:00Z"
                ):
                    made.append(p.id)
                    if not p.embedding:
                        raise LettaError(f"passage {p.id} came back without an embedding; check the agent's embedder")
            if self.verify:
                longest = max(texts, key=len)
                mid = max(0, len(longest) // 2 - 150)
                if not self._found(longest[mid : mid + 300], len(texts), [tag]):
                    raise LettaError(f"wrote {len(made)} passages (tag {tag}) but a search cannot find them again")
        except Exception:
            for pid in made:  # keep the original error; a delete that fails leaves the tag to find leftovers by
                with contextlib.suppress(Exception):
                    self._c.agents.passages.delete(pid, agent_id=self.agent_id)
            raise
        return new
