"""Build the context block for one question: card first, then the best dated memories, then the answer rules."""

from __future__ import annotations

from savestate.card import Card
from savestate.models import Memory, Reranker, Store
from savestate.owners import Owners
from savestate.prompts import RULE_CARD, RULE_FIXED_FACTS, RULES_BASE
from savestate.rerank import NoRerank


def numbered_rules(card: bool = True, benchmark_rules: bool = False) -> str:
    """The numbered answer rules.

    `benchmark_rules` adds "the earliest value of a fixed fact wins". That mirrors how MemConflict builds its
    contradictions and is wrong for a user who corrects an honest mistake, so it is off by default.
    """
    rules = list(RULES_BASE)
    if benchmark_rules:
        rules.insert(5, RULE_FIXED_FACTS)
    if card:
        rules.append(RULE_CARD)
    return "\n".join(f"{i}. {r}" for i, r in enumerate(rules, 1))


def answer_rules(card: bool = True, benchmark_rules: bool = False) -> str:
    """The rules as a system prompt for the answering model."""
    return "You answer questions about a user using only the retrieved memories.\n\nRules:\n" + numbered_rules(
        card, benchmark_rules
    )


class Recall:
    """Fetch a pool from the host, filter and rerank it, keep the best `k`.

    Args:
        store: the host memory system.
        card: the state card to pin first; None for memories only.
        owners: optional owner filter (drops other people's facts).
        reranker: optional reranker; without one the host's order stands.
        pool: how many memories to fetch when filtering or reranking has something to choose from.
        k: how many to keep.
    """

    def __init__(
        self,
        store: Store,
        card: Card | None = None,
        *,
        owners: Owners | None = None,
        reranker: Reranker | None = None,
        pool: int = 30,
        k: int = 10,
        benchmark_rules: bool = False,
    ):
        self.store, self.card, self.owners, self.reranker = store, card, owners, reranker or NoRerank()
        self.pool, self.k, self.benchmark_rules = pool, k, benchmark_rules

    def search(self, question: str) -> list[Memory]:
        """The k memories the answering model should see."""
        deep = self.owners is not None or not isinstance(self.reranker, NoRerank)
        hits = self.store.search(question, self.pool if deep else self.k)
        if self.owners:
            hits = self.owners.filter(question, hits)
        return self.reranker.rerank(question, hits)[: self.k]

    def rules(self) -> str:
        return answer_rules(card=self.card is not None, benchmark_rules=self.benchmark_rules)

    def context(self, question: str, rules: bool = True) -> str:
        """The block to give the answering model: card, memories (each with the date it was learned), rules."""
        return self.render(self.search(question), rules)

    def render(self, memories: list[Memory], rules: bool = True) -> str:
        """Format already-chosen memories the way `context` does (the benchmark reuses one search for several arms)."""
        parts = [self.card.render(), ""] if self.card else []
        parts.append("MEMORIES (date = when it was learned):")
        # one line each, whitespace collapsed: a memory is data and cannot start a line of its own (STATE CARD, RULES)
        parts += [
            f"{n}. [{' '.join((m.date or 'unknown').split())}] {' '.join(m.text.split())}"
            for n, m in enumerate(memories, 1)
        ]
        if not memories:
            parts.append("No relevant memories found.")
        if rules:
            parts += ["", "RULES:", numbered_rules(self.card is not None, self.benchmark_rules)]
        return "\n".join(parts)
