"""Replay one persona's sessions into a host, ask each session's questions, and record the answers per arm.

Arms (all answered from the SAME store and the SAME retrieved pool, so only the reading differs):
  host           the host's own top-k, answered with MemConflict's own answer prompt (the paper's setup, top 10)
  layer          savestate: pool -> owner filter -> reranker -> top-k, pinned card, our answer rules INCLUDING the
                 benchmark-shaped "earliest value of a fixed fact wins" rule (what the research measured)
  layer-generic  the same context with the shipped default rules (benchmark-shaped rule off)
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from bench.memconflict.dataset import session_messages
from savestate import Layer, Recall, answer_rules
from savestate.models import TextLLM

ARMS = ("host", "layer", "layer-generic")

REPO_ENV = "MEMCONFLICT_REPO"


def host_system(repo: str | os.PathLike[str] | None = None) -> str:
    """MemConflict's own answer prompt for mem0, read from your clone (Evaluation/eval_memzero.py): the "host alone"
    arm must read memories the way the benchmark authors do. MemConflict has no licence, so it is never copied here."""
    src = repo or os.environ.get(REPO_ENV)
    if not src:
        raise SystemExit(f"set {REPO_ENV} to your clone of the MemConflict repo (the host arm uses its answer prompt)")
    code = (Path(src).expanduser() / "Evaluation" / "eval_memzero.py").read_text()
    found = re.search(r'MEMZERO_ANSWER_SYSTEM_PROMPT = """(.*?)"""', code, re.S)
    if not found:
        raise SystemExit(f"MEMZERO_ANSWER_SYSTEM_PROMPT not found in {src}/Evaluation/eval_memzero.py")
    return found.group(1)


def ask(llm: TextLLM, system: str, context: str, question: str) -> str:
    return llm.complete(f"Retrieved Memory Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:", system=system)


def run_persona(
    item: dict[str, Any],
    layer: Layer,
    llm: TextLLM,
    out: Path,
    arms: tuple[str, ...] = ARMS,
    sessions: int | None = None,
    samples: int = 3,
    workers: int = 8,
) -> dict[str, Any]:
    """Ingest session by session; after each, answer its questions in every arm. Writes out/predictions.jsonl."""
    chain = item["Full_Session_Chain"][:sessions]
    out.mkdir(parents=True, exist_ok=True)
    host = Recall(layer.store, None, k=layer.recall.k)  # the host alone: its own order, no card
    host_prompt = host_system() if "host" in arms else ""
    preds = out / "predictions.jsonl"
    preds.write_text("")
    t0, n_questions = time.time(), 0
    for i, s in enumerate(chain):
        msgs = session_messages(s)
        new = layer.after_conversation(msgs, s["Date"]) if msgs else []
        jobs = []
        for q in s.get("Session_Questions") or []:
            pool = layer.recall.search(q["question"])
            contexts = {
                "host": (host_prompt, host.render(layer.store.search(q["question"], layer.recall.k), rules=False)),
                "layer": (answer_rules(True, benchmark_rules=True), layer.recall.render(pool, rules=False)),
                "layer-generic": (answer_rules(True, benchmark_rules=False), layer.recall.render(pool, rules=False)),
            }
            jobs += [(arm, q, k, *contexts[arm]) for arm in arms for k in range(samples)]
            n_questions += 1

        def one(job: tuple[Any, ...], i: int = i, s: dict[str, Any] = s) -> dict[str, Any]:
            arm, q, k, system, context = job
            return {
                "session": i,
                "date": s["Date"],
                "qid": q["question_id"],
                "arm": arm,
                "sample": k,
                "type": q["conflict_type"],
                "difficulty": q.get("difficulty"),
                "question": q["question"],
                "gold": q["answer"],
                "answer": ask(llm, system, context, q["question"]),
            }

        with ThreadPoolExecutor(workers) as pool_:
            records = list(pool_.map(one, jobs))
        with preds.open("a") as f:
            f.writelines(json.dumps(rec, ensure_ascii=False) + "\n" for rec in records)
        print(
            f"session {i + 1}/{len(chain)} {s['Date']} new={len(new)} asked={len(jobs) // max(1, len(arms) * samples)} "
            f"card_lines={len(layer.card.lines)} ({time.time() - t0:.0f}s)",
            file=sys.stderr,
            flush=True,
        )
    return {"sessions": len(chain), "questions": n_questions, "card_lines": len(layer.card.lines)}
