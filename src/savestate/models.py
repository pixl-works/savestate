"""The small vocabulary every part of savestate shares: a memory, and the four things you can plug in."""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlsplit


@dataclass(frozen=True)
class Memory:
    """One stored fact as the host reports it. `date` is when it was learned (ISO, or None if the host cannot say)."""

    text: str
    date: str | None = None
    id: str | None = None

    @classmethod
    def of(cls, item: Memory | str | dict[str, Any]) -> Memory:
        """Accept what callers naturally have: a Memory, a plain string, or a host's {"memory"/"text", "id"} dict."""
        if isinstance(item, Memory):
            return item
        if isinstance(item, str):
            return cls(item)
        text = item.get("text") or item.get("memory") or ""
        return cls(str(text), item.get("date"), item.get("id"))


def check_user_id(value: object) -> str:
    """The id that scopes a person's memories; an empty or non-string one could widen a query to everyone."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"user_id must be a non-empty string (got {value!r})")
    return value


def check_key_url(url: str, key: str | None, allow_http: bool = False) -> None:
    """Refuse to send an API key in clear text: with a `key`, an http:// `url` must point at this machine.

    https anywhere, http to localhost / 127.0.0.0/8 / ::1, and http without a key pass. `allow_http=True` is the
    explicit opt-out for a private network you trust. Every client that sends a key calls this before it can.
    """
    if not key or allow_http:
        return
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    if parts.scheme.lower() != "http" or host == "localhost":
        return
    try:
        ip = ipaddress.ip_address(host)
        if (getattr(ip, "ipv4_mapped", None) or ip).is_loopback:
            return
    except ValueError:  # a name, not an address
        pass
    raise ValueError(
        f"refusing to send an API key over plain http to {host or url!r}: use https, or allow_http=True "
        "(config: allow_http = true) if that network is private and you trust it"
    )


@runtime_checkable
class Store(Protocol):
    """A host memory system, read side. Everything savestate needs from it is `search`."""

    def search(self, query: str, k: int) -> list[Memory]: ...


@runtime_checkable
class Ingesting(Protocol):
    """A Store that can also take a conversation. It returns the memories the host reports as NEW."""

    def add(self, messages: list[dict[str, str]], date: str) -> list[Memory]: ...


class LLM(Protocol):
    """Anything that answers a prompt with a JSON object ({} when the model refused or sent no JSON)."""

    def complete_json(self, prompt: str) -> dict[str, Any]: ...


class TextLLM(LLM, Protocol):
    """An LLM that can also answer in prose (only the benchmark needs this)."""

    def complete(self, prompt: str, system: str = "", max_tokens: int = 400) -> str: ...


class Reranker(Protocol):
    """Reorders a pool of memories for one question, best first. Never drops or invents memories."""

    def rerank(self, query: str, memories: list[Memory]) -> list[Memory]: ...
