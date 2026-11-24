"""Honcho v3 (plastic-labs, self-hosted) through its SDK (extra `honcho`, honcho-ai==2.2.0).

Honcho derives "conclusions" about the user from a conversation in the background (a deriver queue). We read the
assistant->user scope, the one Honcho's own context and chat calls use. Four pitfalls, all measured, are handled here:

* Drain: work units are enqueued a moment AFTER add_messages returns, so polling right away sees an empty queue and
  looks done. `add` first waits for new work to appear, then for two quiet polls in a row (a finished unit can enqueue
  a follow-up such as the summary). If the queue never drains it raises: memory state is unknown.
* Dates: Honcho only returns the wall-clock insert time of a conclusion. Each session stores {"date": ...} in its
  metadata and message `created_at` is set to the conversation date, so conclusions are dated by their session.
  (Conclusions from a Honcho "dream" cite no session and fall back to their insert date, which is right live and
  wrong when replaying old conversations: replay with dreams off.)
* Empty ingest raises instead of silently returning nothing (usually a failing deriver LLM or embedder).
* Slow dialectic/LLM calls: the SDK's 30 s default silently empties answers, so the timeout is 300 s.
"""

from __future__ import annotations

import hashlib
import re
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from savestate.models import Memory, check_user_id

MESSAGE_MAX_CHARS = 25000  # Honcho's maximum message size; longer turns are split with a "[continued] " prefix
MAX_BATCH = 100  # add_messages takes at most 100 messages per call


class HonchoError(RuntimeError):
    pass


def _chunks(text: str) -> list[str]:
    parts = [text[i : i + MESSAGE_MAX_CHARS] for i in range(0, len(text), MESSAGE_MAX_CHARS)] or [text]
    return [parts[0], *("[continued] " + p for p in parts[1:])]


class Honcho:
    """Args:
    workspace: one Honcho workspace per person, so people never see each other (made filesystem/id safe).
    url: Honcho API base.
    drain_timeout: seconds to wait for the deriver before raising.
    client: an existing `honcho.Honcho` (tests, or a shared connection).
    """

    def __init__(
        self,
        workspace: str,
        url: str = "http://127.0.0.1:8820",
        user_peer: str = "user",
        ai_peer: str = "assistant",
        timeout: float = 300.0,
        drain_timeout: float = 900.0,
        enqueue_wait: float = 30.0,
        allow_empty: bool = False,
        client: Any = None,
    ):
        self.user_id = self.workspace = check_user_id(workspace)  # user_id stays the raw id; Layer binds the card to it
        safe = re.sub(r"[^a-zA-Z0-9_-]", "_", self.workspace)  # Honcho ids must match ^[a-zA-Z0-9_-]+$
        if safe != self.workspace:  # many ids map to one safe id; the hash keeps people apart
            safe += "_" + hashlib.sha256(self.workspace.encode()).hexdigest()[:12]
        self.workspace = safe
        self.user_peer, self.ai_peer = user_peer, ai_peer
        self.drain_timeout, self.enqueue_wait, self.allow_empty = drain_timeout, enqueue_wait, allow_empty
        if client is None:
            try:
                from honcho import Honcho as HonchoSDK
            except ImportError as exc:
                raise ImportError("the Honcho adapter needs the extra: pip install 'savestate[honcho]'") from exc
            client = HonchoSDK(
                api_key="local", base_url=url, workspace_id=self.workspace, timeout=timeout, max_retries=2
            )
        self._c = client
        self._session_dates: dict[str, str | None] = {}

    def _scope(self) -> Any:
        return self._c.peer(self.ai_peer).conclusions_of(self.user_peer)

    def _date(self, session_id: str | None) -> str | None:
        if not session_id:
            return None
        if session_id not in self._session_dates:
            self._session_dates[session_id] = (self._c.session(session_id).get_metadata() or {}).get("date")
        return self._session_dates[session_id]

    def _memory(self, c: Any) -> Memory:
        date = self._date(c.session_id) or (c.created_at.date().isoformat() if c.created_at else None)
        return Memory(c.content, date, c.id)

    def search(self, query: str, k: int) -> list[Memory]:
        return [self._memory(c) for c in self._scope().query(query, top_k=min(k, 100))]

    def add(self, messages: list[dict[str, str]], date: str) -> list[Memory]:
        """Ingest one conversation, wait for the deriver, return the conclusions derived from it."""
        from honcho.api_types import SessionPeerConfig

        self._c._ensure_workspace()
        user, ai = self._c.peer(self.user_peer), self._c.peer(self.ai_peer)
        sid = f"{date}-{uuid.uuid4().hex[:8]}"
        session = self._c.session(sid, metadata={"date": date})
        self._session_dates[sid] = date
        both = SessionPeerConfig(observe_me=True, observe_others=True)  # without this Honcho derives nothing
        session.add_peers([(user, both), (ai, both)])
        before = self._c.queue_status().total_work_units
        base = datetime.strptime(date, "%Y-%m-%d").replace(hour=12, tzinfo=UTC)
        payload = []
        for m in messages:
            peer = user if m["role"] == "user" else ai
            for chunk in _chunks(str(m["content"])):
                if chunk.strip():  # seconds apart: keeps dialogue order inside the conversation's own day
                    payload.append(peer.message(chunk, created_at=base + timedelta(seconds=2 * len(payload))))
        for i in range(0, len(payload), MAX_BATCH):
            session.add_messages(payload[i : i + MAX_BATCH])
        self._drain(before)
        new = [self._memory(c) for c in self._scope().list(session=sid, size=100)]
        if not new and not self.allow_empty:
            raise HonchoError(
                f"no conclusion derived from a {len(payload)}-message conversation (workspace "
                f"{self.workspace}); check the deriver's logs for an LLM or embedder error"
            )
        return new

    def _drain(self, total_before: int, sleep: Callable[[float], None] = time.sleep) -> None:
        """Block until the deriver has no pending or running work in this workspace (see the module docstring)."""
        start, seen, quiet = time.monotonic(), False, 0
        while True:
            st = self._c.queue_status()
            busy = st.pending_work_units > 0 or st.in_progress_work_units > 0
            seen = seen or busy or st.total_work_units > total_before
            if busy:
                quiet = 0
            elif seen or time.monotonic() - start > self.enqueue_wait:
                quiet += 1
                if quiet >= 2:
                    return
                sleep(2.0)
                continue
            if time.monotonic() - start > self.drain_timeout:
                raise HonchoError(f"deriver did not drain in {self.drain_timeout:.0f}s (workspace {self.workspace})")
            sleep(1.0)
