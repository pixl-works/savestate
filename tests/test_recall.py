from savestate import Layer, Memory, NoRerank, Owners, Recall, answer_rules
from savestate.card import Card, Line
from savestate.stores import InMemory
from savestate.testing import FakeLLM

MEMS = [
    Memory("Moved to Oslo in March", "2026-03-01", "1"),
    Memory("Her brother lives in Rome", "2026-02-01", "2"),
    Memory("Likes tea", "2026-01-05", "3"),
    Memory("Lives in Rome", "2025-01-01", "4"),
]


class Spy:
    """A store that records how deep it was asked to search."""

    def __init__(self, memories):
        self.memories, self.asked = memories, []

    def search(self, query, k):
        self.asked.append(k)
        return self.memories[:k]


class Reverse:
    def rerank(self, query, memories):
        return memories[::-1]


def test_context_puts_card_first_then_dated_memories_then_rules():
    card = Card(FakeLLM())
    card.lines = {
        "residence": Line("User's residence (as of 2026-03-01): Oslo. Before: Rome until 2026-03-01.", "2026-03-01")
    }
    out = Recall(InMemory(MEMS), card, k=2).context("where does she live in Oslo")
    assert out.index("STATE CARD:") < out.index("MEMORIES") < out.index("RULES:")
    assert "1. [2026-03-01] Moved to Oslo in March" in out
    assert out.count(". [") == 2  # k=2
    assert "STATE CARD lists" in out


def test_context_without_card_or_rules_and_with_no_memories():
    out = Recall(InMemory(), None).context("anything", rules=False)
    assert out == "MEMORIES (date = when it was learned):\nNo relevant memories found."


def test_undated_memories_say_unknown():
    assert "[unknown] x" in Recall(InMemory([Memory("x")]), None).context("x", rules=False)


def test_pool_is_deep_only_when_something_can_choose_from_it():
    plain = Spy(MEMS)
    Recall(plain, None, pool=30, k=10).search("q")
    ranked = Spy(MEMS)
    Recall(ranked, None, reranker=Reverse(), pool=30, k=2).search("q")
    assert plain.asked == [10] and ranked.asked == [30]


def test_reranker_orders_the_pool_and_k_cuts_after_it():
    assert [m.id for m in Recall(Spy(MEMS), None, reranker=Reverse(), k=2).search("q")] == ["4", "3"]
    assert [m.id for m in Recall(Spy(MEMS), None, reranker=NoRerank(), k=2).search("q")] == ["1", "2"]


def test_owner_filter_drops_other_peoples_facts_and_caches_per_text():
    llm = FakeLLM(
        [{"owners": [{"id": "m2", "about": "other"}, {"id": "m1", "about": "user"}, {"id": "m3", "about": "user"}]}]
    )
    rec = Recall(Spy(MEMS[:3]), None, owners=Owners(llm), k=10)
    assert [m.id for m in rec.search("what happened in March")] == ["1", "3"]
    rec.search("what happened in May")
    assert len(llm.prompts) == 1  # same texts: no second model call


def test_identity_questions_skip_the_owner_filter():
    llm = FakeLLM([{"owners": [{"id": "m1", "about": "other"}]}])
    rec = Recall(Spy(MEMS[1:2]), None, owners=Owners(llm))
    assert [m.id for m in rec.search("Does she have a brother?")] == ["2"] and llm.prompts == []
    assert rec.search("Where does she work?") == []


def test_owner_cache_survives_on_disk_and_unknown_stays(tmp_path):
    llm = FakeLLM([{"owners": [{"id": "m1", "about": "other"}, {"id": "zz", "about": "other"}]}])
    Owners(llm, tmp_path / "o.json").classify(["a", "b"])
    again = Owners(FakeLLM(), tmp_path / "o.json")
    assert not again.keep("a") and again.keep("b") and again.keep("never seen")


def test_benchmark_rules_are_off_by_default_and_opt_in():
    default, bench = answer_rules(card=True), answer_rules(card=True, benchmark_rules=True)
    assert "earliest recorded value" not in default and "earliest recorded value" in bench
    assert "STATE CARD" in default and "STATE CARD" not in answer_rules(card=False)
    assert Layer(InMemory(), FakeLLM()).rules == default
    assert Layer(InMemory(), FakeLLM(), benchmark_rules=True).rules == bench


def test_a_cut_off_owner_cache_is_ignored_not_fatal(tmp_path):
    (tmp_path / "o.json").write_text('{"a": "oth')
    owners = Owners(FakeLLM([{"owners": [{"id": "m1", "about": "other"}]}]), tmp_path / "o.json")
    owners.classify(["a"])
    assert not owners.keep("a") and Owners(FakeLLM(), tmp_path / "o.json").map == {"a": "other"}  # rewritten whole
