"""Test doubles: a scripted LLM, so card and recall logic can be tested offline and deterministically."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class FakeLLM:
    """Answers complete_json from a script: a list of dicts handed out in order (the last one repeats), or a callable
    `prompt -> dict`. Every prompt it saw is kept in `prompts`, so tests can assert what the model was shown."""

    def __init__(self, script: list[dict[str, Any]] | Callable[[str], dict[str, Any]] | None = None):
        self.script = script if script is not None else [{}]
        self.prompts: list[str] = []

    def complete_json(self, prompt: str) -> dict[str, Any]:
        self.prompts.append(prompt)
        if callable(self.script):
            return self.script(prompt)
        return self.script.pop(0) if len(self.script) > 1 else self.script[0]
