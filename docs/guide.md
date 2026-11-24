# savestate guide

Everything the [README](../README.md) leaves out: the API in detail, every option, the command line, security and limits.

## The API

`AnthropicLLM` defaults to `claude-haiku-5-5` and reads `ANTHROPIC_API_KEY`; `Mem0Http` reads `MEM0_API_KEY` if your
server has auth. The two hooks are `after_conversation(messages, date)` and `context(question)`.

`context()` returns plain text for your prompt. Use `layer.rules` as the system prompt, or `context(question,
rules=False)` for the card and memories only. `layer.card.render()` prints the card alone: a host with a good context of
its own can pin just that (the "card only" column in [Results](../README.md#results)). If the host adds memories by itself (a
plugin, a background worker), call `layer.observe(new_memories, date)` instead of `after_conversation`, and
`layer.freshen()` once a day.

A card after the three conversations in [tests/test_live.py](../tests/test_live.py) (mem0 and Haiku; output copied from an earlier
run of that test, not re-run for this page):

```text
STATE CARD:
- User's residence (as of 2026-04-05): Sydney, Australia. Before: Melbourne, Australia until 2026-04-05.
- User's employer (as of 2026-04-05): none (left Acme around late March 2026). Before: Acme until 2026-04-05.
- User's job title (as of 2026-02-20): senior analyst. Before: junior analyst until 2026-02-20.
- User's employment status (as of 2026-04-05): unemployed.
```

**Subject trap.** Card lines read "User's residence ...". The subject must be the name your memories use for the person:
with memories saying "Sam moved" under a card about "User", Haiku read Sam as someone else and silently skipped the
update. Use `Layer(..., subject="Sam")`. The change log records an `{"error": ...}` entry when the model returns no
usable answer and a `{"rejected": ..., "reason": ...}` entry when code refuses a line. A model that answers "nothing
changed" is not an error, which is how this mismatch hides: check the card after the first few updates.

## Parts and options

The owner filter and reranker are both optional and off by default.

| Part | What is there |
|---|---|
| **Stores** (`savestate.stores`) | `Mem0Http`, `MemOSHttp`, `Honcho` (extra `honcho`), `InMemory` (tests and examples). A store needs `search(query, k) -> list[Memory]`; one that also has `add(messages, date) -> list[Memory]` works with `after_conversation`. |
| **LLMs** | `AnthropicLLM`, `OpenAICompatLLM` (any `/chat/completions` server), `savestate.testing.FakeLLM`. Anything with `complete_json(prompt) -> dict` works. |
| **Rerankers** | `NoRerank` (default), `HttpReranker` (1.5 s timeout, falls back to the host's order, 120 s cooldown after a failure), `LocalCrossEncoder` (extra `local`: sentence-transformers, pulls torch). |
| **Owner filter** | `owner_filter=True` drops memories about other people's own lives (a friend's move) by asking the model once per distinct memory text and caching the answer. **Off by default**: it assumes every question is about the user, which is how the benchmark asks. |

```python
Layer(
    store,
    llm,
    card_path="card.json",
    subject="Sam",  # the name your memories use
    names={"residence": "where Sam lives"},  # replaces the default attribute names
    blocklist=r"deploy|home assistant",  # lines matching this never reach the card
    reranker=HttpReranker("http://127.0.0.1:11600/v1/rerank"),
    owner_filter=True,
    pool=30,
    k=10,  # fetch 30, keep the best 10
)
```

Default attributes: residence, employer, job title, employment status, work status, income, savings, relationship,
health. The model may add names; a merge pass folds duplicates.

Answer rules: The benchmark-shaped answer rule "the earliest value of a fixed fact wins" is off by default (`benchmark_rules=False`): MemConflict contradicts fixed facts with later false claims, but real users correct honest mistakes.

## Command line and configuration

Configuration is a TOML file ([savestate.example.toml](../savestate.example.toml)) overridden by
`SAVESTATE_*` variables. The file is found only through `--config PATH`, `$SAVESTATE_CONFIG` or
`~/.config/savestate/config.toml`; a `savestate.toml` in the current directory is **not** picked up, so an untrusted
directory cannot redirect your API key. Keys are never read from the file: use `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
`MEM0_API_KEY`. A key is never sent over plain `http://` to another machine (see Security); for a private network you
trust, set `allow_http = true` in `[llm]` or `[store]` (or `SAVESTATE_LLM_ALLOW_HTTP` / `SAVESTATE_STORE_ALLOW_HTTP`).
The card is bound to `[store] user_id` (see Security).

```bash
mkdir -p ~/.config/savestate && cp savestate.example.toml ~/.config/savestate/config.toml   # or pass --config PATH
savestate card update --date 2026-11-23 < new_memories.txt   # one memory per line, or {"id": ..., "text": ...} lines
savestate card show                                          # the card as pinned (--full: all history)
savestate card freshen                                       # strike expired parts
savestate context "Where do I live now?"                     # the full block from the configured store
savestate serve-rerank --port 11600                          # needs savestate[local]; loopback, no auth
```

`user_id` must be a non-empty string. `serve-rerank` has no authentication and answers `GET` only on `/health`; keep it
on loopback or a private network.

## Security

The card, its change log and the owner cache are written with mode 0600, because they hold personal facts. API keys come
only from environment variables, never from the config file, and the CLI reads no config from the current directory.
If loading a card drops lines that no longer pass the guards (say, after changing `subject` or `blocklist`), the
original file is kept next to it as `card.dropped-*.json` before anything is saved.

A card belongs to one person. `card.json` records the `user_id` of the store it was built from (`Layer` takes it from
the store, or pass `Card(..., user_id=...)`). Loading a file that names a different user raises `ValueError` and leaves
the file alone, so a wrong `card_path` cannot mix two people. A card from an older version has no id and gets one on
its next save.

An API key is never sent over plain `http://` to a host other than this machine: `AnthropicLLM`, `OpenAICompatLLM` and
`Mem0Http` raise `ValueError` when they are built with a key and an `http://` URL that is not `localhost`, `127.0.0.0/8`
or `::1`. `https://`, loopback `http://`, and `http://` without a key still work. If your network is private and you
trust it, pass `allow_http=True` (config: `allow_http = true`). `MemOSHttp` and `Honcho` send no key.

## Limits

- One writer per card file: a second process (say a cron `card freshen`) overwrites the first one's saves.

## Development

```bash
git clone https://github.com/pixl-works/savestate.git && cd savestate
uv sync                 # add --extra honcho; --extra local pulls torch
uv run pytest           # offline; the live test is opt-in: uv run pytest -m live
uv run ruff check .
```
