"""Runs offline: no server, no API key. A scripted FakeLLM plays the model, so this shows the plumbing, not model
quality. Run it with: python offline_demo.py"""

from pathlib import Path

from savestate import InMemory, Layer
from savestate.testing import FakeLLM

# What a real model would answer to the card-update prompt, once per conversation.
# "m1" is the first new memory in that call; the change log turns it into the store's own id.
llm = FakeLLM(
    [
        {"lines": [{"name": "residence", "text": "User's residence (as of 2026-03-01): Darwin.", "ids": ["m1"]}]},
        {
            "lines": [
                {
                    "name": "residence",
                    "text": "User's residence (as of 2026-11-23): Melbourne. Before: Darwin until 2026-11-23.",
                    "ids": ["m1"],
                }
            ]
        },
    ]
)
layer = Layer(InMemory(), llm, card_path="demo-card.json")

layer.after_conversation([{"role": "user", "content": "I live in Darwin."}], date="2026-03-01")
layer.after_conversation([{"role": "user", "content": "I just moved to Melbourne."}], date="2026-11-23")

print(layer.context("Where do I live now?", rules=False))

print()
print(Path("demo-card.log.jsonl").read_text().splitlines()[-1])  # each change keeps the ids of its source memories
