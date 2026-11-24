"""Rerankers: put the memories that answer the question first.

Hosts rank by embedding similarity, which loves stale near-duplicates. A cross-encoder over a deeper pool (30) cut
to the best 10 pulls the right memory up from rank 11-30 (a host's own `rerank=True` can only reorder what it
already cut). Every reranker here returns the same memories reordered, never fewer.
"""

from __future__ import annotations

import time

import httpx

from savestate.models import Memory

DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"


class NoRerank:
    """Keep the host's order."""

    def rerank(self, query: str, memories: list[Memory]) -> list[Memory]:
        return list(memories)


def _by_score(memories: list[Memory], scores: list[float]) -> list[Memory]:
    return [memories[i] for i in sorted(range(len(memories)), key=lambda i: -scores[i])]


class HttpReranker:
    """Calls a reranking service (`savestate serve-rerank`): POST url {"query", "documents"} -> {"scores"}.

    A reranker is an improvement, never a dependency: on a timeout, an error or a malformed answer it returns the
    host's order and stays away for `cooldown` seconds, so a dead service costs one short timeout, not one per question.
    """

    def __init__(self, url: str, timeout: float = 1.5, cooldown: float = 120.0, client: httpx.Client | None = None):
        self.url, self.timeout, self.cooldown = url, timeout, cooldown
        self._client = client or httpx.Client()
        self._down_until = 0.0
        self.stats: dict[str, int | str | None] = {"reranked": 0, "fallback": 0, "last_error": None}

    def rerank(self, query: str, memories: list[Memory]) -> list[Memory]:
        if len(memories) <= 1:
            return list(memories)
        if time.monotonic() < self._down_until:
            self.stats["fallback"] = int(self.stats["fallback"] or 0) + 1
            return list(memories)
        try:
            resp = self._client.post(
                self.url, json={"query": query, "documents": [m.text for m in memories]}, timeout=self.timeout
            )
            resp.raise_for_status()
            scores = resp.json()["scores"]
            if len(scores) != len(memories):
                raise ValueError(f"{len(scores)} scores for {len(memories)} documents")
            ranked = _by_score(memories, scores)  # inside the try: non-numeric scores are a malformed answer too
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            self._down_until = time.monotonic() + self.cooldown
            self.stats.update(fallback=int(self.stats["fallback"] or 0) + 1, last_error=f"{type(exc).__name__}: {exc}")
            return list(memories)
        self.stats["reranked"] = int(self.stats["reranked"] or 0) + 1
        return ranked


class CrossEncoderScorer:
    """A local cross-encoder (extra `local`: sentence-transformers). Pairs run shortest first in small batches, so a
    long memory pads only its own batch (30 real-length memories: 1.06 s unsorted at batch 32, 0.83 s sorted at
    batch 8 on an Apple M4). fp16 on Apple MPS, whose allocator would otherwise grow without bound."""

    def __init__(self, model: str = DEFAULT_MODEL, device: str | None = None, max_length: int = 256):
        try:
            import torch
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError("a local reranker needs the extra: pip install 'savestate[local]'") from exc
        self.model_name = model
        self._model = CrossEncoder(model, device=device, max_length=max_length)
        self.device = str(self._model.model.device)
        self._mps = self.device.startswith("mps")
        if self._mps:
            self._model.model.half()
        self._torch = torch

    def score(self, query: str, docs: list[str]) -> list[float]:
        """Scores in input order."""
        order = sorted(range(len(docs)), key=lambda i: len(docs[i]))
        out = self._model.predict([(query, docs[i]) for i in order], batch_size=8, show_progress_bar=False)
        if self._mps:
            self._torch.mps.synchronize()
            self._torch.mps.empty_cache()  # measured: the service grew to 3.9 GB in 3 h without this
        scores = [0.0] * len(docs)
        for rank, i in enumerate(order):
            scores[i] = float(out[rank])
        return scores


class LocalCrossEncoder:
    """Rerank in this process with a local cross-encoder (loads the model on first use)."""

    def __init__(self, model: str = DEFAULT_MODEL, device: str | None = None):
        self.model, self.device = model, device
        self._scorer: CrossEncoderScorer | None = None

    def rerank(self, query: str, memories: list[Memory]) -> list[Memory]:
        if len(memories) <= 1:
            return list(memories)
        self._scorer = self._scorer or CrossEncoderScorer(self.model, self.device, max_length=512)
        return _by_score(memories, self._scorer.score(query, [m.text for m in memories]))
