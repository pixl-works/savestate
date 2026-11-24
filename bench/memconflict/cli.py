"""python -m bench.memconflict run|judge|report ... (see bench/memconflict/README.md)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from bench.memconflict.dataset import load_personas
from bench.memconflict.judge import judge_file
from bench.memconflict.report import report
from bench.memconflict.run import ARMS, run_persona
from savestate import AnthropicLLM, HttpReranker, InMemory, Layer, Letta, NoRerank, OpenAICompatLLM
from savestate.config import build_store


def _llm(a: argparse.Namespace) -> AnthropicLLM | OpenAICompatLLM:
    if a.provider == "openai":
        return OpenAICompatLLM(a.model, base_url=a.llm_url or "https://api.openai.com/v1")
    return AnthropicLLM(a.model, base_url=a.llm_url)


def _dir(a: argparse.Namespace, persona: int) -> Path:
    return Path(a.out) / "memconflict" / f"p{persona:02d}"


def cmd_run(a: argparse.Namespace) -> None:
    item = load_personas()[a.persona]
    out, llm = _dir(a, a.persona), _llm(a)
    uid = f"mc{a.persona}-{a.tag or int(time.time())}"
    host_store, letta_sdk = None, None
    if a.store == "letta":  # Letta needs an agent with an embedder: one throwaway agent per run
        uid, letta_sdk = _letta_agent(a, uid)
    cfg = {"store": {"type": a.store, "user_id": uid, **({"url": a.store_url} if a.store_url else {})}}
    store = InMemory() if a.store == "memory" else build_store(cfg, llm)
    if a.store == "letta":
        host_store = Letta(uid, client=letta_sdk, split=False)  # the host alone reads whole chunks, as measured
    (out / "card.json").unlink(missing_ok=True)  # a benchmark run starts from an empty card (before Layer reads it)
    layer = Layer(
        store,
        llm,
        card_path=out / "card.json",
        owner_filter=True,
        reranker=HttpReranker(a.rerank_url, timeout=120) if a.rerank_url else NoRerank(),
    )
    arms = tuple(a.arms.split(","))
    if not set(arms) <= set(ARMS):
        raise SystemExit(f"--arms must be a subset of {','.join(ARMS)}")
    result = run_persona(item, layer, llm, out, arms, a.sessions, a.samples, host_store=host_store)
    print(json.dumps({"persona": a.persona, "store": a.store, "user_id": uid, **result}))
    if a.cleanup and letta_sdk is not None:
        letta_sdk.agents.delete(uid)
    elif a.cleanup and hasattr(store, "delete_all"):
        store.delete_all()


def _letta_agent(a: argparse.Namespace, name: str) -> tuple[str, object]:
    """Create the run's Letta agent (archival memory empty; its own LLM is never called by the bench)."""
    from letta_client import Letta as LettaSDK

    sdk = LettaSDK(base_url=a.store_url or "http://127.0.0.1:8283", api_key="local", timeout=300, max_retries=4)
    emb = {
        "embedding_endpoint_type": "openai",
        "embedding_endpoint": a.letta_embed_url,
        "embedding_model": a.letta_embed_model,
        "embedding_dim": a.letta_embed_dim,
        "embedding_chunk_size": 300,
    }
    agent = sdk.agents.create(
        name=name,
        model=a.letta_model,
        embedding_config=emb,
        memory_blocks=[
            {"label": "human", "value": f"Identifier: {name}."},
            {"label": "persona", "value": "Assistant."},
        ],
    )
    return agent.id, sdk


def cmd_judge(a: argparse.Namespace) -> None:
    print(json.dumps({"persona": a.persona, **judge_file(_llm(a), _dir(a, a.persona))}))


def cmd_report(a: argparse.Namespace) -> None:
    personas = [int(x) for x in a.personas.split(",")]
    print(json.dumps(report({f"p{p:02d}": _dir(a, p) for p in personas}, a.arms.split(","), a.base), indent=1))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m bench.memconflict", description=__doc__)
    p.add_argument("--out", default="runs", help="where results go (default ./runs, which is git-ignored)")
    p.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic")
    p.add_argument("--model", default="claude-haiku-5-5")
    p.add_argument("--llm-url")
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="ingest a persona and record answers")
    r.add_argument("--persona", type=int, required=True)
    r.add_argument("--store", choices=["memory", "mem0", "memos", "honcho", "letta"], default="memory")
    r.add_argument("--store-url")
    r.add_argument("--rerank-url", help="e.g. http://127.0.0.1:11600/v1/rerank (savestate serve-rerank)")
    r.add_argument("--sessions", type=int, help="only the first N sessions (smoke test)")
    r.add_argument("--samples", type=int, default=3)
    r.add_argument("--arms", default="host,layer", help=f"any of: {','.join(ARMS)}")
    r.add_argument("--tag", help="suffix of the throwaway host user id (default: timestamp)")
    r.add_argument("--cleanup", action="store_true", help="delete the host user afterwards (stores that support it)")
    r.add_argument("--letta-embed-url", default="http://127.0.0.1:11500/v1", help="Letta: OpenAI-compatible embedder")
    r.add_argument("--letta-embed-model", default="bge-m3")
    r.add_argument("--letta-embed-dim", type=int, default=1024)
    r.add_argument("--letta-model", default="anthropic/claude-haiku-5-5", help="Letta: the agent model (never called)")
    r.set_defaults(fn=cmd_run)
    j = sub.add_parser("judge", help="grade a persona's answers")
    j.add_argument("--persona", type=int, required=True)
    j.set_defaults(fn=cmd_judge)
    rep = sub.add_parser("report", help="compare arms over judged personas")
    rep.add_argument("--personas", required=True, help="comma-separated persona numbers, e.g. 0,1,2,3,4")
    rep.add_argument("--arms", default="host,layer")
    rep.add_argument("--base", default="host")
    rep.set_defaults(fn=cmd_report)
    args = p.parse_args(argv)
    args.fn(args)
    return 0
