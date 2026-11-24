"""One opt-in end-to-end test against a real model. Run it with `uv run pytest -m live`.

It talks to Claude through a key-injecting proxy (default http://127.0.0.1:8890/anthropic; set ANTHROPIC_BASE_URL and
ANTHROPIC_API_KEY for your own endpoint) and uses only hand-written messages.
"""

import os

import pytest

from savestate import AnthropicLLM, InMemory, Layer

pytestmark = pytest.mark.live

SESSIONS = [
    ("2026-01-10", "I just moved to Melbourne from Darwin, and I work at Acme as a junior analyst."),
    ("2026-02-20", "Big news: I got promoted to senior analyst at Acme. I'm still in Melbourne."),
    ("2026-04-05", "I quit Acme last week and I'm unemployed for now. I also moved to Sydney."),
]


def test_card_gains_real_lines_and_context_returns_card_and_memories(tmp_path):
    llm = AnthropicLLM(
        base_url=os.environ.get("ANTHROPIC_BASE_URL", "http://127.0.0.1:8890/anthropic"),
        api_key=os.environ.get("ANTHROPIC_API_KEY", "via-proxy"),
    )
    layer = Layer(InMemory(), llm, card_path=tmp_path / "card.json")
    for date, text in SESSIONS:
        layer.after_conversation([{"role": "user", "content": text}, {"role": "assistant", "content": "Noted."}], date)
    assert layer.card.lines, "the card stayed empty"
    out = layer.context("Where does the user live and work now?")
    assert out.index("STATE CARD:") < out.index("MEMORIES")
    assert "Sydney" in out and "[2026-04-05]" in out
    assert any("Sydney" in ln.text for ln in layer.card.lines.values())
