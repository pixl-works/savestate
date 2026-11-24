"""Configuration: a small TOML file, overridden by SAVESTATE_* environment variables.

    [store]
    type = "mem0"                       # mem0 | memos | honcho
    url = "http://127.0.0.1:8888"
    user_id = "me"
    allow_http = false                  # same, for MEM0_API_KEY
    [llm]
    provider = "anthropic"              # anthropic | openai
    model = "claude-haiku-5-5"
    base_url = "https://api.anthropic.com"
    allow_http = false                  # true: send the key over plain http to a non-local host (default: refused)
    [card]
    path = "~/.savestate/card.json"
    subject = "User"
    names = ["residence", "employer"]   # optional: replaces the default attribute names
    blocklist = "tooling|deploy"        # optional regex: lines matching it never reach the card
    [recall]
    pool = 30
    k = 10
    owner_filter = false
    [rerank]
    url = "http://127.0.0.1:11600/v1/rerank"

Secrets are never read from the file: use MEM0_API_KEY, ANTHROPIC_API_KEY or OPENAI_API_KEY.
Search order for the file: --config, $SAVESTATE_CONFIG, ~/.config/savestate/config.toml. The current directory is
never searched: the file decides where the API key is sent, so only a file you named or your own config is read.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from savestate.card import DEFAULT_NAMES, Card
from savestate.layer import Layer
from savestate.llm import AnthropicLLM, OpenAICompatLLM
from savestate.models import LLM, Reranker, Store
from savestate.rerank import HttpReranker, NoRerank

ENV = {  # variable -> (section, key)
    "SAVESTATE_STORE": ("store", "type"),
    "SAVESTATE_STORE_URL": ("store", "url"),
    "SAVESTATE_USER_ID": ("store", "user_id"),
    "SAVESTATE_LLM": ("llm", "provider"),
    "SAVESTATE_MODEL": ("llm", "model"),
    "SAVESTATE_LLM_URL": ("llm", "base_url"),
    "SAVESTATE_LLM_ALLOW_HTTP": ("llm", "allow_http"),
    "SAVESTATE_STORE_ALLOW_HTTP": ("store", "allow_http"),
    "SAVESTATE_CARD": ("card", "path"),
    "SAVESTATE_SUBJECT": ("card", "subject"),
    "SAVESTATE_BLOCKLIST": ("card", "blocklist"),
    "SAVESTATE_POOL": ("recall", "pool"),
    "SAVESTATE_K": ("recall", "k"),
    "SAVESTATE_OWNER_FILTER": ("recall", "owner_filter"),
    "SAVESTATE_RERANK_URL": ("rerank", "url"),
}
_INTS, _BOOLS = {"pool", "k"}, {"owner_filter", "allow_http"}
DEFAULT_CARD = "~/.savestate/card.json"


def load_config(
    path: str | os.PathLike[str] | None = None, env: Mapping[str, str] | None = None
) -> dict[str, dict[str, Any]]:
    env = os.environ if env is None else env
    candidates = [path, env.get("SAVESTATE_CONFIG"), "~/.config/savestate/config.toml"]
    cfg: dict[str, dict[str, Any]] = {}
    for c in candidates:
        if c and Path(c).expanduser().is_file():
            cfg = tomllib.loads(Path(c).expanduser().read_text())
            break
    else:
        if path:
            raise FileNotFoundError(path)
    for var, (section, key) in ENV.items():
        if var in env:
            raw = env[var]
            cfg.setdefault(section, {})[key] = (
                int(raw) if key in _INTS else raw.lower() in ("1", "true", "yes") if key in _BOOLS else raw
            )
    return cfg


def build_llm(cfg: dict[str, dict[str, Any]]) -> LLM:
    c = cfg.get("llm", {})
    allow = c.get("allow_http") is True  # only a real true: "false" in quotes must not open the door
    if c.get("provider", "anthropic") == "openai":
        return OpenAICompatLLM(
            c.get("model", "gpt-4o-mini"), **({"base_url": c["base_url"]} if "base_url" in c else {}), allow_http=allow
        )
    return AnthropicLLM(c.get("model", "claude-haiku-5-5"), base_url=c.get("base_url"), allow_http=allow)


def build_store(cfg: dict[str, dict[str, Any]]) -> Store:
    from savestate.stores import Honcho, Mem0Http, MemOSHttp

    c = cfg.get("store", {})
    kind, user = c.get("type"), c.get("user_id", "default")
    if kind == "mem0":
        return Mem0Http(c.get("url", "http://127.0.0.1:8888"), user, allow_http=c.get("allow_http") is True)
    if kind == "memos":
        return MemOSHttp(c.get("url", "http://127.0.0.1:8800"), user)
    if kind == "honcho":
        return Honcho(user, c.get("url", "http://127.0.0.1:8820"))
    raise ValueError(
        f"[store] type must be mem0, memos or honcho (got {kind!r}); set it in the config or SAVESTATE_STORE"
    )


def build_reranker(cfg: dict[str, dict[str, Any]]) -> Reranker:
    url = cfg.get("rerank", {}).get("url")
    return HttpReranker(url) if url else NoRerank()


def build_card(cfg: dict[str, dict[str, Any]], llm: LLM | None = None) -> Card:
    c = cfg.get("card", {})
    return Card(
        llm or build_llm(cfg),
        c.get("path", DEFAULT_CARD),
        subject=c.get("subject", "User"),
        names=c.get("names", DEFAULT_NAMES),
        blocklist=c.get("blocklist"),
        user_id=cfg.get("store", {}).get("user_id"),
    )


def build_layer(cfg: dict[str, dict[str, Any]]) -> Layer:
    c, r = cfg.get("card", {}), cfg.get("recall", {})
    return Layer(
        build_store(cfg),
        build_llm(cfg),
        card_path=c.get("path", DEFAULT_CARD),
        subject=c.get("subject", "User"),
        names=c.get("names", DEFAULT_NAMES),
        blocklist=c.get("blocklist"),
        reranker=build_reranker(cfg),
        owner_filter=bool(r.get("owner_filter", False)),
        pool=int(r.get("pool", 30)),
        k=int(r.get("k", 10)),
    )
