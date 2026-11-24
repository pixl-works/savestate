# MemConflict: host alone vs host + savestate

[MemConflict](https://github.com/TaoZhen1110/MemConflict) (Tao Zhen et al., [arXiv:2605.20926](https://arxiv.org/abs/2605.20926)) is a benchmark of 30 personas, each a chain of about 53 dated
conversations with questions after some of them. Question types: *dynamic* (a fact really changed: residence, job,
relationship), *static* (a later false claim contradicts a fact that cannot change) and *conditional* (a preference that
depends on the situation). The report calls them **changing**, **never-change** and **situational**.

**The dataset is not redistributable and is not in this repo.** Get `Step4_4.jsonl` from the MemConflict authors (their repo) and
point at it:

```bash
export MEMCONFLICT_DATA=~/datasets/memconflict/Step4_4.jsonl
export MEMCONFLICT_REPO=~/src/MemConflict  # a clone of their repo: the host arm reads its answer prompt from there
export ANTHROPIC_API_KEY=...            # or ANTHROPIC_BASE_URL for a key-injecting proxy
```

## Run

```bash
# 1. replay a persona into a host and answer its questions (smoke test: 6 sessions, 1 sample, in-memory host)
uv run python -m bench.memconflict run --persona 0 --sessions 6 --samples 1 --store memory
# a real host (throwaway user id, deleted afterwards where the store supports it); optional reranker service
uv run python -m bench.memconflict run --persona 0 --store mem0 --store-url http://localhost:8888 \
    --rerank-url http://127.0.0.1:11600/v1/rerank --cleanup
# 2. grade the answers
uv run python -m bench.memconflict judge --persona 0
# 3. compare arms over the personas you judged
uv run python -m bench.memconflict report --personas 0,1,2,3,4 --arms host,layer,layer-generic
```

Results go to `./runs/` (git-ignored). A run is not resumable: delete `runs/memconflict/pNN` and start again.

## Arms

All arms are answered from the same store and the same retrieved pool, so only the reading differs.

| Arm | Context | Answer rules |
|---|---|---|
| `host` | the host's own top 10 | MemConflict's own answer prompt (the paper's setup) |
| `layer` | pool of 30 -> owner filter -> reranker -> 10, plus the pinned card | savestate's rules **including the benchmark-shaped "earliest value of a fixed fact wins"** (what the research measured) |
| `layer-generic` | the same | savestate's shipped default rules (benchmark-shaped rule off) |

`--store memory` is the in-memory test store: it keeps raw user turns and ranks by word overlap, so it is a **pipeline
smoke test, not a measurement**; its "host alone" score is meaningless. Use a real host (`mem0`, `memos`, `honcho`) for
numbers.

## Scoring (fixed before the runs)

- Half-credit LLM judge: 1 / 0.5 / 0, plus an "outdated value" flag. An empty or unparseable judgement is *unjudged*.
- Common exclusion: a question that is unjudged in any arm is dropped from all arms.
- Per persona and type, the mean over questions and samples; arms compared by per-persona difference against the base arm.
- 95% bootstrap CI over personas (10,000 draws, seed 0) and a sign-flip p-value. With one persona the CI is
  degenerate; the report prints how many personas it used.

The host arm answers with MemConflict's own mem0 answer prompt, read at run time from `$MEMCONFLICT_REPO/Evaluation/eval_memzero.py`.
MemConflict has no licence, so neither its data nor its prompt is copied into this repo.