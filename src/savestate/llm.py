"""LLM adapters over plain HTTP (no vendor SDK, so no version pinning): Anthropic native and OpenAI-compatible.

Two facts about current Claude models drive the design: they REJECT `temperature` (it is never sent), and thinking
must be switched off or it eats the token budget and leaves an empty answer (Haiku wants "disabled", Sonnet and
Opus want "between_tools").
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import httpx

from savestate.models import check_key_url


class LLMError(RuntimeError):
    """The model call failed after its retries."""


def thinking_off(model: str) -> dict[str, Any]:
    """The request field that turns thinking off, which differs by model family."""
    return {"thinking": {"type": "disabled" if "haiku" in model else "between_tools"}}


def parse_json(raw: str) -> dict[str, Any]:
    """The first JSON object in `raw`, or {}. raw_decode, because models sometimes append a second object or prose,
    which breaks first-brace-to-last-brace slicing."""
    start = raw.find("{")
    if start < 0:
        return {}
    try:
        out, _ = json.JSONDecoder().raw_decode(raw[start:])
    except ValueError:
        return {}
    return out if isinstance(out, dict) else {}


def _scrub(text: str, headers: dict[str, str]) -> str:
    """`text` without the API key: an endpoint that echoes the request must not put the key into an error."""
    for name in ("x-api-key", "authorization"):
        key = headers.get(name, "").removeprefix("Bearer ")
        if key:
            text = text.replace(key, "***")
    return text


def _post(client: httpx.Client, url: str, headers: dict[str, str], body: dict[str, Any], retries: int = 4) -> Any:
    """POST with backoff on network errors, 429, 5xx and Anthropic's 529; any other error status raises at once."""
    delay, last = 2.0, ""
    for attempt in range(retries):
        try:
            resp = client.post(url, headers=headers, json=body)
        except httpx.TransportError as exc:
            last = f"{type(exc).__name__}: {exc}"
        else:
            if resp.status_code < 400:
                return resp.json()
            last = f"HTTP {resp.status_code}: {_scrub(resp.text, headers)[:300]}"
            if resp.status_code not in (429, 500, 502, 503, 504, 529):
                break
        if attempt < retries - 1:
            time.sleep(delay)
            delay *= 2
    raise LLMError(f"{url} failed: {last}")


class AnthropicLLM:
    """Anthropic Messages API. `base_url` may point at a proxy that injects the key (then any api_key string works).

    A key is never sent over plain http to a host other than this machine (ValueError); `allow_http=True` lifts that.
    """

    def __init__(
        self,
        model: str = "claude-haiku-5-5",
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 4000,
        timeout: float = 300.0,
        client: httpx.Client | None = None,
        allow_http: bool = False,
    ):
        self.model, self.max_tokens = model, max_tokens
        self.base_url = (base_url or os.environ.get("ANTHROPIC_BASE_URL") or "https://api.anthropic.com").rstrip("/")
        self._key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        check_key_url(self.base_url, self._key, allow_http)
        self._client = client or httpx.Client(timeout=timeout)

    def complete(self, prompt: str, system: str = "", max_tokens: int | None = None) -> str:
        if not self._key:  # checked here, not in __init__, so read-only commands work without a key
            raise ValueError("AnthropicLLM needs api_key= or the ANTHROPIC_API_KEY environment variable")
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens or self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            **thinking_off(self.model),
        }
        if system:
            body["system"] = system
        headers = {"x-api-key": self._key, "anthropic-version": "2023-06-01"}
        data = _post(self._client, f"{self.base_url}/v1/messages", headers, body)
        if data.get("stop_reason") == "refusal":
            return ""
        return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")

    def complete_json(self, prompt: str) -> dict[str, Any]:
        return parse_json(self.complete(prompt))


class OpenAICompatLLM:
    """Any OpenAI-compatible /chat/completions endpoint (OpenAI, vLLM, Ollama, a proxy). No temperature, no extras.

    A real key (api_key= or OPENAI_API_KEY) is never sent over plain http to a host other than this machine
    (ValueError); `allow_http=True` lifts that. Without a key the placeholder "none" is sent, so a LAN server is fine.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        max_tokens: int = 4000,
        timeout: float = 300.0,
        client: httpx.Client | None = None,
        allow_http: bool = False,
    ):
        self.model, self.max_tokens = model, max_tokens
        self.base_url = base_url.rstrip("/")
        key = api_key or os.environ.get("OPENAI_API_KEY")
        check_key_url(self.base_url, key, allow_http)
        self._key = key or "none"
        self._client = client or httpx.Client(timeout=timeout)

    def complete(self, prompt: str, system: str = "", max_tokens: int | None = None) -> str:
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        body = {"model": self.model, "max_tokens": max_tokens or self.max_tokens, "messages": messages}
        data = _post(self._client, f"{self.base_url}/chat/completions", {"authorization": f"Bearer {self._key}"}, body)
        return (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""

    def complete_json(self, prompt: str) -> dict[str, Any]:
        return parse_json(self.complete(prompt))
