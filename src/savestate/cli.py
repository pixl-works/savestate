"""The `savestate` command line.

savestate card update --date 2026-11-23 < new_memories.txt     one memory per line (or {"id", "text"} JSON lines)
savestate card show
savestate card freshen
savestate context "Where do I live now?"
savestate serve-rerank --port 11600
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date as Date
from pathlib import Path

from savestate.config import build_card, build_layer, load_config
from savestate.models import Memory


def _memories(lines: list[str]) -> list[Memory]:
    return [
        Memory.of(json.loads(s)) if s.lstrip().startswith("{") else Memory(s) for s in (ln.strip() for ln in lines) if s
    ]


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="savestate", description="A self-maintaining state card on top of any agent memory."
    )
    p.add_argument(
        "--config",
        help="TOML config file (default: $SAVESTATE_CONFIG, then ~/.config/savestate/config.toml)",
    )
    sub = p.add_subparsers(dest="command", required=True)
    card = sub.add_parser("card", help="maintain and show the state card").add_subparsers(dest="action", required=True)
    up = card.add_parser("update", help="fold new memories into the card")
    up.add_argument(
        "--date", default=Date.today().isoformat(), help="when the memories were learned (YYYY-MM-DD, default today)"
    )
    up.add_argument("--file", help="read memories from this file instead of stdin")
    show = card.add_parser("show", help="print the card as it would be pinned to a prompt")
    show.add_argument("--full", action="store_true", help="include all history, not just the last change")
    fresh = card.add_parser("freshen", help="strike expired parts of the card")
    fresh.add_argument("--today", default=Date.today().isoformat())
    ctx = sub.add_parser("context", help="print the context block for a question")
    ctx.add_argument("question")
    ctx.add_argument("--no-rules", action="store_true", help="omit the answer rules")
    rr = sub.add_parser("serve-rerank", help="serve a local cross-encoder reranker over HTTP (needs savestate[local])")
    rr.add_argument("--host", default="127.0.0.1")
    rr.add_argument("--port", type=int, default=11600)
    rr.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    rr.add_argument("--max-length", type=int, default=256)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "serve-rerank":
        from savestate.server import serve

        try:
            serve(args.model, args.host, args.port, args.max_length)
        except ImportError as exc:  # the `local` extra is not installed
            raise SystemExit(str(exc)) from exc
        return 0
    cfg = load_config(args.config)
    if args.command == "context":
        print(build_layer(cfg).context(args.question, rules=not args.no_rules))
    else:
        card = build_card(cfg)
        if args.action == "update":
            text = Path(args.file).read_text() if args.file else sys.stdin.read()
            changes = card.update(_memories(text.splitlines()), args.date)
            print(json.dumps(changes, indent=1, ensure_ascii=False))
        elif args.action == "freshen":
            print(json.dumps(card.freshen(args.today), indent=1, ensure_ascii=False))
        else:
            print(card.render(full=args.full))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
