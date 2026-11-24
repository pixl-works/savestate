"""`savestate serve-rerank`: the reranking service HttpReranker talks to.

    POST /v1/rerank {"query": "...", "documents": ["...", ...]}  ->  {"scores": [...], "ms": 812.3}
    GET  /health                                                  ->  {"ok": true, "model": "...", "device": "..."}

Binds to loopback by default: it has no authentication. Run it on the machine with the GPU and reach it through an
SSH tunnel or a private network, not the open internet.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

MAX_DOCS = 100
MAX_BODY = 1_000_000  # bytes
TIMEOUT = 10.0  # seconds a connection may stall on one read

log = logging.getLogger(__name__)


def make_server(
    score: Callable[[str, list[str]], list[float]], info: dict[str, Any], host: str = "127.0.0.1", port: int = 11600
) -> ThreadingHTTPServer:
    """An HTTP server around any `score(query, docs) -> scores` function (serialised: one request scores at a time)."""
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        timeout = TIMEOUT  # per-connection socket timeout: a client that stalls loses its thread

        def log_message(self, format: str, *args: Any) -> None:
            pass

        def reply(self, code: int, out: dict[str, Any]) -> None:
            raw = json.dumps(out).encode()
            self.send_response(code)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(raw)))
            if code >= 400:  # the request body may be unread: close, or its bytes would parse as the next request
                self.send_header("connection", "close")
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:
            if self.path.split("?")[0] != "/health":
                return self.reply(404, {"error": "GET /health"})
            self.reply(200, {"ok": True, **info})

        def do_POST(self) -> None:
            if self.headers.get("transfer-encoding"):
                return self.reply(501, {"error": "transfer-encoding is not supported; send content-length"})
            if not self.path.rstrip("/").endswith("/rerank"):
                return self.reply(404, {"error": "POST /v1/rerank"})
            length = self.headers.get("content-length", "0")
            if not (length.isascii() and length.isdigit()):
                return self.reply(400, {"error": "bad content-length"})
            if int(length) > MAX_BODY:
                return self.reply(413, {"error": f"body over {MAX_BODY} bytes"})
            try:
                body = json.loads(self.rfile.read(int(length)) or b"{}")
                docs = [str(d) for d in body.get("documents", [])][:MAX_DOCS]
                t = time.perf_counter()
                with lock:
                    scores = score(str(body["query"]), docs) if docs else []
                self.reply(200, {"scores": scores, "ms": round(1000 * (time.perf_counter() - t), 1)})
            except Exception as exc:  # a bad request must not kill the service
                log.warning("rerank request failed: %r", exc)  # the detail stays here, not with the client
                self.reply(400, {"error": "bad request"})

    return ThreadingHTTPServer((host, port), Handler)


def serve(model: str, host: str = "127.0.0.1", port: int = 11600, max_length: int = 256) -> None:
    """Load the cross-encoder, warm it (the first call compiles kernels), and serve until interrupted."""
    from savestate.rerank import CrossEncoderScorer

    scorer = CrossEncoderScorer(model, max_length=max_length)
    scorer.score("warm up", ["the first call compiles the kernels"] * 8)
    server = make_server(scorer.score, {"model": model, "device": scorer.device}, host, port)
    print(f"savestate rerank on {host}:{port} ({model}, {scorer.device})", flush=True)
    server.serve_forever()
