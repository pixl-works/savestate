from types import SimpleNamespace

import pytest

from savestate import Layer, Memory
from savestate.stores import InMemory
from savestate.testing import FakeLLM

ASK = {"lines": [{"name": "residence", "text": "User's residence (as of 2026-03-01): Oslo.", "ids": ["m1"]}]}


def test_after_conversation_ingests_then_updates_the_card_from_the_new_memories(tmp_path):
    store, llm = InMemory(), FakeLLM([ASK])
    layer = Layer(store, llm, card_path=tmp_path / "card.json")
    new = layer.after_conversation(
        [{"role": "user", "content": "I moved to Oslo"}, {"role": "assistant", "content": "Nice!"}], date="2026-03-01"
    )
    assert [m.text for m in new] == ["I moved to Oslo"] and new[0].date == "2026-03-01"
    assert "[m1] I moved to Oslo" in llm.prompts[0]
    assert layer.card.lines["residence"].text.endswith("Oslo.")
    assert '"ids": ["m1"]' in (tmp_path / "card.log.jsonl").read_text()


def test_context_answers_from_card_and_store(tmp_path):
    layer = Layer(InMemory(), FakeLLM([ASK]))
    layer.after_conversation([{"role": "user", "content": "I moved to Oslo"}], "2026-03-01")
    out = layer.context("Where do I live? Oslo?")
    assert "User's residence (as of 2026-03-01): Oslo." in out and "[2026-03-01] I moved to Oslo" in out
    assert "RULES:" not in layer.context("q", rules=False)


def test_observe_is_for_hosts_that_add_memories_themselves():
    layer = Layer(InMemory(), FakeLLM([ASK]))
    changes = layer.observe([Memory("I moved to Oslo", id="7"), "plain strings work too"], "2026-03-01")
    assert changes[0]["name"] == "residence" and layer.store.memories == []  # nothing written to the host


def test_a_store_without_add_says_what_to_do():
    class ReadOnly:
        def search(self, query, k):
            return []

    with pytest.raises(TypeError, match=r"Layer\.observe"):
        Layer(ReadOnly(), FakeLLM()).after_conversation([{"role": "user", "content": "x"}], "2026-01-01")


def test_date_defaults_to_today_and_must_be_iso():
    layer = Layer(InMemory(), FakeLLM())
    layer.observe(["x"])  # no date: today
    with pytest.raises(ValueError):
        layer.observe(["x"], "yesterday")


def test_owner_filter_is_opt_in(tmp_path):
    assert Layer(InMemory(), FakeLLM()).recall.owners is None
    assert Layer(InMemory(), FakeLLM(), owner_filter=True, card_path=tmp_path / "c.json").recall.owners is not None


def test_a_failed_card_update_hands_back_the_memories_the_store_took(tmp_path):
    class Down(FakeLLM):
        def complete_json(self, prompt):
            raise RuntimeError("529")

    layer = Layer(InMemory(), Down(), card_path=tmp_path / "card.json")
    with pytest.raises(RuntimeError) as err:
        layer.after_conversation([{"role": "user", "content": "I moved to Oslo"}], "2026-03-01")
    assert "Layer.observe()" in "\n".join(err.value.__notes__) and "I moved to Oslo" in "\n".join(err.value.__notes__)
    assert [m.text for m in layer.store.memories] == ["I moved to Oslo"]


def test_the_layer_binds_the_card_to_the_store_user_id(tmp_path):
    p = tmp_path / "card.json"
    p.write_text('{"user_id": "u1", "lines": {}}')
    store = SimpleNamespace(user_id="u2", search=lambda q, k: [])
    with pytest.raises(ValueError, match=r"'u1'.*'u2'"):
        Layer(store, FakeLLM([]), card_path=p)
    assert Layer(SimpleNamespace(user_id="u1", search=lambda q, k: []), FakeLLM([]), card_path=p).card.user_id == "u1"
    assert Layer(InMemory(), FakeLLM([]), card_path=p).card.user_id == "u1"  # a store without an id: the file's stays
    assert Layer(InMemory(), FakeLLM([])).card.user_id is None
