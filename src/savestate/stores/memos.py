"""MemOS (MemTensor, self-hosted) over REST: POST /product/add and /product/search."""

from __future__ import annotations

import time
import uuid
from typing import Any

import httpx

from savestate.models import Memory, check_user_id


class MemOSHttp:
    """Args:
    url: MemOS server base, e.g. http://localhost:8800.
    user_id: the MemOS user (cube) whose memories these are.
    """

    def __init__(
        self, url: str, user_id: str, timeout: float = 1200.0, retries: int = 4, client: httpx.Client | None = None
    ):
        self.user_id, self.retries = check_user_id(user_id), retries
        self._client = client or httpx.Client(base_url=url.rstrip("/"), timeout=timeout)

    def _post(self, path: str, body: dict[str, Any]) -> Any:
        """MemOS answers 5xx while it is busy extracting, so retry with growing waits and raise if it never recovers."""
        err: Exception | None = None
        for attempt in range(self.retries):
            try:
                resp = self._client.post(path, json=body)
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPError as exc:
                err = exc
                if attempt < self.retries - 1:
                    time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"MemOS {path} failed after {self.retries} attempts: {err}")

    def search(self, query: str, k: int) -> list[Memory]:
        # relativity 0: no relevance cut, because a reranker needs the tail of the pool
        data = self._post("/product/search", {"query": query, "user_id": self.user_id, "top_k": k, "relativity": 0})
        out = []
        for cube in data["data"].get("text_mem", []):
            for m in cube.get("memories", []):
                src = (m.get("metadata") or {}).get("sources") or [{}]
                out.append(Memory(m.get("memory", ""), str(src[0].get("chat_time", ""))[:10] or None, m.get("id")))
        return out

    def add(self, messages: list[dict[str, str]], date: str) -> list[Memory]:
        """Ingest one conversation and wait for extraction (sync mode); returns the memories MemOS reports as added."""
        if not messages:
            return []
        msgs = [{**m, "chat_time": f"{date} 10:00:00"} for m in messages]
        data = self._post(
            "/product/add",
            {
                "user_id": self.user_id,
                "session_id": uuid.uuid4().hex,
                "messages": msgs,
                "async_mode": "sync",
                "mode": "fine",
            },
        )
        return [
            Memory(d["memory"], date, d.get("memory_id") or d.get("id"))
            for d in data.get("data") or []
            if d.get("memory")
        ]
