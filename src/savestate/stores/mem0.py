"""Self-hosted mem0 server over REST (POST /memories, POST /search; auth header X-API-Key).

The key comes from `api_key=` or the MEM0_API_KEY environment variable; a server running with auth disabled needs none.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from savestate.models import Memory, check_key_url, check_user_id


class Mem0Http:
    """Args:
    url: server base, e.g. http://localhost:8888.
    user_id: the mem0 user whose memories these are.
    api_key: sent as X-API-Key; refused over plain http to a host other than this machine unless `allow_http=True`.
    stamp_date: mem0 stamps wall-clock time, not conversation time, so `add` also writes the date into the chat
        text (extraction then sees it) and into metadata (which is where `search` reads the date back from).
    """

    def __init__(
        self,
        url: str,
        user_id: str,
        api_key: str | None = None,
        stamp_date: bool = True,
        timeout: float = 120.0,
        client: httpx.Client | None = None,
        allow_http: bool = False,
    ):
        self.user_id, self.stamp_date = check_user_id(user_id), stamp_date
        key = api_key or os.environ.get("MEM0_API_KEY")
        if client is None:  # a client you pass in carries its own headers; ours are not sent
            check_key_url(url, key, allow_http)
        self._client = client or httpx.Client(
            base_url=url.rstrip("/"),
            timeout=timeout,
            transport=httpx.HTTPTransport(retries=2),
            headers={"X-API-Key": key} if key else {},
        )

    def _json(self, method: str, path: str, **kw: Any) -> Any:
        resp = self._client.request(method, path, **kw)
        resp.raise_for_status()
        return resp.json() if resp.content else {}

    @staticmethod
    def _memory(item: dict[str, Any]) -> Memory:
        date = (item.get("metadata") or {}).get("date") or str(item.get("created_at") or "")[:10] or None
        return Memory(item.get("memory", ""), date, item.get("id"))

    def search(self, query: str, k: int) -> list[Memory]:
        data = self._json("POST", "/search", json={"query": query, "top_k": k, "filters": {"user_id": self.user_id}})
        return [self._memory(r) for r in (data.get("results", []) if isinstance(data, dict) else data)]

    def add(self, messages: list[dict[str, str]], date: str) -> list[Memory]:
        """Send the conversation for server-side extraction; returns the memories mem0 reports as ADDed."""
        msgs = [dict(m) for m in messages]
        if self.stamp_date and msgs:
            msgs[0]["content"] = f"(Conversation date: {date})\n{msgs[0]['content']}"
        data = self._json(
            "POST",
            "/memories",
            json={"messages": msgs, "user_id": self.user_id, "infer": True, "metadata": {"date": date}},
        )
        added = data.get("results", []) if isinstance(data, dict) else data
        return [
            Memory(r["memory"], date, r.get("id")) for r in added if r.get("event", "ADD") == "ADD" and r.get("memory")
        ]

    def delete_all(self) -> None:
        """Delete every memory of this user (for throwaway users in tests; needs an admin key)."""
        self._json("DELETE", "/memories", params={"user_id": self.user_id})
