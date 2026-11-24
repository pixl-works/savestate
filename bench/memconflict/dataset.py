"""MemConflict data access. The dataset is not redistributable and is never copied into this repo: point
MEMCONFLICT_DATA at your copy of Step4_4.jsonl (one persona per line, each a chain of ~53 dated sessions)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

ENV = "MEMCONFLICT_DATA"
TYPES = {"dynamic_conflict": "changing", "static_conflict": "never-change", "conditional_conflict": "situational"}


def load_personas(path: str | os.PathLike[str] | None = None) -> list[dict[str, Any]]:
    src = path or os.environ.get(ENV)
    if not src:
        raise SystemExit(f"set {ENV} to your copy of the MemConflict Step4_4.jsonl (see bench/memconflict/README.md)")
    return [json.loads(line) for line in Path(src).expanduser().read_text().splitlines() if line.strip()]


def session_messages(session: dict[str, Any]) -> list[dict[str, str]]:
    """A session's dialogue as chat messages, in turn order. Turns without a user/assistant role are skipped."""
    turns = sorted((session.get("Session_Dialogue") or {}).items(), key=lambda kv: int(kv[0].rsplit("_", 1)[-1]))
    return [
        {"role": m["role"], "content": m["content"]}
        for _, msgs in turns
        for m in msgs
        if m.get("content") and m.get("role") in ("user", "assistant")
    ]
