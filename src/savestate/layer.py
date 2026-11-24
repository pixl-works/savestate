"""The facade: two hooks, `after_conversation` and `context`. Everything else in savestate is a part of this."""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import date as Date
from datetime import datetime
from typing import Any

from savestate.card import DEFAULT_NAMES, Card, iso_date
from savestate.models import LLM, Ingesting, Memory, Reranker, Store
from savestate.owners import Owners
from savestate.recall import Recall


class Layer:
    """A state card and a better recall on top of an existing memory store.

        layer = Layer(Mem0Http("http://localhost:8888", user_id="me"), AnthropicLLM(), card_path="card.json")
        layer.after_conversation(messages, date="2026-11-23")   # after each conversation
        prompt = layer.context("Where do I live now?")          # before each answer

    The layer never writes into the store beyond handing it the conversation (`after_conversation`); the card lives
    in its own file. Hosts that add memories by themselves (a plugin, a background worker) use `observe` instead.

    Args:
        store: the host memory system (a Store; an Ingesting one if you use `after_conversation`).
        llm: the model that maintains the card (and classifies owners).
        card_path: where the card is kept; None keeps it in memory only. The card is bound to the store's `user_id`
            (when the store has one): a card file that belongs to someone else raises ValueError.
        subject, names, blocklist: see `Card`.
        reranker: see `savestate.rerank`; default keeps the host's order.
        owner_filter: drop other people's facts from recall. Off by default: it assumes every question is about
            the user.
        pool, k: fetch `pool` memories, keep `k`.
        benchmark_rules: add the benchmark-shaped "earliest value of a fixed fact wins" rule to the answer rules.
    """

    def __init__(
        self,
        store: Store,
        llm: LLM,
        *,
        card_path: str | os.PathLike[str] | None = None,
        subject: str = "User",
        names: Mapping[str, str] | Sequence[str] = DEFAULT_NAMES,
        blocklist: str | re.Pattern[str] | None = None,
        reranker: Reranker | None = None,
        owner_filter: bool = False,
        pool: int = 30,
        k: int = 10,
        benchmark_rules: bool = False,
    ):
        self.store = store
        self.card = Card(
            llm, card_path, subject=subject, names=names, blocklist=blocklist, user_id=getattr(store, "user_id", None)
        )
        owners = Owners(llm, os.fspath(card_path) + ".owners.json" if card_path else None) if owner_filter else None
        self.recall = Recall(
            store, self.card, owners=owners, reranker=reranker, pool=pool, k=k, benchmark_rules=benchmark_rules
        )

    def after_conversation(
        self, messages: list[dict[str, str]], date: str | Date | datetime | None = None
    ) -> list[Memory]:
        """Give the conversation to the host, then fold the memories it reports as new into the card.

        Returns those new memories. Raises if the store cannot ingest; use `observe` for such hosts.
        """
        if not isinstance(self.store, Ingesting):
            raise TypeError(
                f"{type(self.store).__name__} has no add(); the host adds memories by itself, "
                "so pass them to Layer.observe() instead"
            )
        day = _day(date)
        new = self.store.add(messages, day)
        try:
            self.card.update(new, day)
        except Exception as exc:  # the store already holds these memories; hand them back so the caller can retry
            exc.add_note(
                f"the store kept {len(new)} new memories the card may not have seen; retry Layer.observe(): {new}"
            )
            raise
        return new

    def observe(
        self, new_memories: Iterable[Memory | str | dict[str, Any]], date: str | Date | datetime | None = None
    ) -> list[dict[str, Any]]:
        """Fold memories the host added on its own into the card. Returns the card changes."""
        return self.card.update(new_memories, _day(date))

    def freshen(self, today: str | Date | datetime | None = None) -> list[dict[str, Any]]:
        """Strike expired parts of the card (past dates, stale short-lived states). Run it daily."""
        return self.card.freshen(_day(today))

    def context(self, question: str, rules: bool = True) -> str:
        """Card, the best dated memories for `question`, and the answer rules (`rules=False` omits them)."""
        return self.recall.context(question, rules=rules)

    @property
    def rules(self) -> str:
        """The answer rules alone, for use as the answering model's system prompt."""
        return self.recall.rules()


def _day(value: str | Date | datetime | None) -> str:
    return iso_date(Date.today() if value is None else value)
