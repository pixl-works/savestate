import json
import re
from datetime import UTC, datetime
from types import SimpleNamespace

import httpx
import pytest

from savestate import Honcho, Ingesting, InMemory, Letta, Mem0Http, Memory, MemOSHttp, Store
from savestate.stores import memos as memos_mod
from savestate.stores.honcho import HonchoError, _chunks
from savestate.stores.letta import LettaError, chunks, exchanges


def test_every_adapter_satisfies_the_protocols():
    assert isinstance(InMemory(), Store) and isinstance(InMemory(), Ingesting)
    for cls in (Mem0Http, MemOSHttp, Honcho, Letta):
        assert hasattr(cls, "search") and hasattr(cls, "add")


def test_inmemory_ranks_by_shared_words_newest_first_on_ties():
    s = InMemory()
    s.add(
        [
            {"role": "user", "content": "I like green tea"},
            {"role": "assistant", "content": "tea is nice"},
            {"role": "user", "content": "I live in Oslo"},
            {"role": "user", "content": "Oslo is cold"},
        ],
        "2026-01-01",
    )
    assert [m.id for m in s.memories] == ["m1", "m2", "m3"]  # assistant turns are not memories
    assert [m.text for m in s.search("Oslo", 2)] == ["Oslo is cold", "I live in Oslo"]  # a tie: newest first
    assert s.search("where in Oslo", 1)[0].text == "I live in Oslo"  # two shared words beat one
    assert [m.text for m in s.search("zzz", 1)] == ["Oslo is cold"]


def mem0_client(handler):
    return httpx.Client(base_url="http://mem0", transport=httpx.MockTransport(handler))


def test_mem0_search_reads_the_date_from_metadata_else_created_at():
    def handler(req):
        body = json.loads(req.content)
        assert req.url.path == "/search" and body == {"query": "q", "top_k": 5, "filters": {"user_id": "u1"}}
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "id": "1",
                        "memory": "a",
                        "metadata": {"date": "2022-02-02"},
                        "created_at": "2026-11-23T10:00:00Z",
                    },
                    {"id": "2", "memory": "b", "metadata": None, "created_at": "2026-11-23T10:00:00Z"},
                ]
            },
        )

    assert Mem0Http("http://mem0", "u1", client=mem0_client(handler)).search("q", 5) == [
        Memory("a", "2022-02-02", "1"),
        Memory("b", "2026-11-23", "2"),
    ]


def test_mem0_add_stamps_the_date_and_returns_only_added_memories():
    seen = []

    def handler(req):
        seen.append(json.loads(req.content))
        return httpx.Response(
            200,
            json={
                "results": [
                    {"id": "1", "memory": "new", "event": "ADD"},
                    {"id": "2", "memory": "touched", "event": "UPDATE"},
                    {"id": "3", "memory": "", "event": "ADD"},
                ]
            },
        )

    store = Mem0Http("http://mem0", "u1", client=mem0_client(handler))
    messages = [{"role": "user", "content": "hello"}]
    assert store.add(messages, "2026-03-01") == [Memory("new", "2026-03-01", "1")]
    assert seen[0]["messages"][0]["content"] == "(Conversation date: 2026-03-01)\nhello"
    assert seen[0]["metadata"] == {"date": "2026-03-01"} and seen[0]["infer"] is True and seen[0]["user_id"] == "u1"
    assert messages[0]["content"] == "hello"  # the caller's messages are not modified


def test_mem0_api_key_goes_in_x_api_key_and_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("MEM0_API_KEY", "secret")
    assert Mem0Http("http://mem0", "u", allow_http=True)._client.headers["x-api-key"] == "secret"
    monkeypatch.delenv("MEM0_API_KEY")
    assert "x-api-key" not in Mem0Http("http://mem0", "u")._client.headers


def test_mem0_delete_all_targets_one_user():
    seen = []

    def handler(req):
        seen.append((req.method, req.url.path, dict(req.url.params)))
        return httpx.Response(200, json={"message": "ok"})

    Mem0Http("http://mem0", "tmp-user", client=mem0_client(handler)).delete_all()
    assert seen == [("DELETE", "/memories", {"user_id": "tmp-user"})]


def test_memos_add_and_search_use_the_product_endpoints(monkeypatch):
    monkeypatch.setattr(memos_mod.time, "sleep", lambda s: None)
    seen = []

    def handler(req):
        body = json.loads(req.content)
        seen.append((req.url.path, body))
        if req.url.path == "/product/add":
            return httpx.Response(200, json={"data": [{"memory": "likes tea", "memory_id": "x1"}, {"memory": ""}]})
        return httpx.Response(
            200,
            json={
                "data": {
                    "text_mem": [
                        {
                            "memories": [
                                {
                                    "memory": "m",
                                    "id": "x2",
                                    "metadata": {"sources": [{"chat_time": "2022-05-06 10:00:00"}]},
                                }
                            ]
                        }
                    ]
                }
            },
        )

    store = MemOSHttp(
        "http://memos", "u", client=httpx.Client(base_url="http://memos", transport=httpx.MockTransport(handler))
    )
    assert store.add([{"role": "user", "content": "hi"}], "2022-05-06") == [Memory("likes tea", "2022-05-06", "x1")]
    assert store.search("q", 30) == [Memory("m", "2022-05-06", "x2")]
    assert seen[0][1]["messages"][0]["chat_time"] == "2022-05-06 10:00:00" and seen[0][1]["async_mode"] == "sync"
    assert seen[1][1]["relativity"] == 0 and seen[1][1]["top_k"] == 30


def test_memos_raises_after_its_retries_without_a_pointless_last_wait(monkeypatch):
    waits = []
    monkeypatch.setattr(memos_mod.time, "sleep", waits.append)
    store = MemOSHttp(
        "http://memos",
        "u",
        retries=2,
        client=httpx.Client(base_url="http://memos", transport=httpx.MockTransport(lambda req: httpx.Response(503))),
    )
    with pytest.raises(RuntimeError, match="after 2 attempts"):
        store.search("q", 3)
    assert waits == [5]  # one wait between the two attempts, none after the last


def test_honcho_chunks_long_messages_with_a_continued_prefix():
    assert _chunks("short") == ["short"]
    parts = _chunks("x" * 60000)
    assert [len(p) for p in parts] == [25000, 25000 + len("[continued] "), 10000 + len("[continued] ")]


class FakeQueue:
    """queue_status as Honcho behaves: nothing visible for a moment after the send, then work, then drained."""

    def __init__(self, states):
        self.states, self.calls = list(states), 0

    def __call__(self):
        self.calls += 1
        pending, running, total = self.states.pop(0) if len(self.states) > 1 else self.states[0]
        return SimpleNamespace(pending_work_units=pending, in_progress_work_units=running, total_work_units=total)


def honcho_with(queue, **kw):
    return Honcho("ws one", client=SimpleNamespace(queue_status=queue), **kw)


def test_honcho_workspace_names_are_made_id_safe():
    # a changed id keeps its readable part and gains a hash of the raw id, so "a.b" and "a_b" stay apart
    assert re.fullmatch(r"ws_one_[0-9a-f]{12}", honcho_with(FakeQueue([(0, 0, 0)])).workspace)


def test_honcho_drain_does_not_trust_an_empty_queue_right_after_the_send():
    # the first polls show nothing (work not enqueued yet), then work appears, runs, and finishes
    queue = FakeQueue([(0, 0, 5), (0, 0, 5), (1, 0, 6), (0, 1, 6), (0, 0, 6), (0, 0, 6)])
    honcho_with(queue)._drain(total_before=5, sleep=lambda s: None)
    assert queue.calls == 6  # it kept polling through the empty start and needed two quiet polls at the end


def test_honcho_drain_gives_up_on_an_enqueue_that_never_comes_and_on_a_stuck_queue():
    honcho_with(FakeQueue([(0, 0, 5)]), enqueue_wait=0.0)._drain(
        total_before=5, sleep=lambda s: None
    )  # nothing to wait for
    with pytest.raises(HonchoError, match="did not drain"):
        honcho_with(FakeQueue([(3, 0, 9)]), drain_timeout=0.0)._drain(total_before=5, sleep=lambda s: None)


def test_honcho_dates_a_conclusion_by_its_session_then_by_insert_time():
    class Session:
        def __init__(self, meta):
            self.meta = meta

        def get_metadata(self):
            return self.meta

    client = SimpleNamespace(session=lambda sid: Session({"date": "2022-02-02"}))
    h = Honcho("ws", client=client)
    made = datetime(2026, 11, 23, tzinfo=UTC)
    assert h._memory(SimpleNamespace(content="a", session_id="s1", created_at=made, id="1")).date == "2022-02-02"
    assert h._memory(SimpleNamespace(content="b", session_id=None, created_at=made, id="2")).date == "2026-11-23"


def test_stores_expose_the_raw_user_id_for_the_card_binding():
    client = SimpleNamespace()
    assert Mem0Http("http://mem0", "me", client=mem0_client(None)).user_id == "me"
    assert MemOSHttp("http://memos", "me", client=mem0_client(None)).user_id == "me"
    h = Honcho("ws one", client=client)
    assert h.user_id == "ws one" and h.workspace != "ws one"  # the card is bound to the raw id, not the safe one
    assert getattr(InMemory(), "user_id", None) is None


def test_mem0_key_over_plain_http_to_another_host_is_refused(monkeypatch):
    monkeypatch.delenv("MEM0_API_KEY", raising=False)
    with pytest.raises(ValueError, match="plain http"):
        Mem0Http("http://mem0.example.com:8888", "me", api_key="secret-key")
    monkeypatch.setenv("MEM0_API_KEY", "from-env")
    with pytest.raises(ValueError, match="plain http"):
        Mem0Http("http://10.0.0.5:8888", "me")
    Mem0Http("http://10.0.0.5:8888", "me", allow_http=True)
    Mem0Http("https://mem0.example.com", "me")
    Mem0Http("http://127.0.0.1:8888", "me")
    monkeypatch.delenv("MEM0_API_KEY")
    Mem0Http("http://mem0.example.com:8888", "me")  # no key, nothing to protect


def test_memos_and_honcho_send_no_key_so_plain_http_stays_allowed():
    MemOSHttp("http://memos.lan:8800", "me")
    Honcho("ws", "http://honcho.lan:8820", client=SimpleNamespace())  # its SDK key is the fixed placeholder "local"


# --- Letta -------------------------------------------------------------------------------------------------------


class FakePassages:
    """agents.passages as Letta 0.16.8 behaves: create returns a list with the embedding, search honours tags."""

    def __init__(self, embed: bool = True, findable: bool = True):
        self.rows: list[SimpleNamespace] = []
        self.embed, self.findable, self.searches, self.deleted = embed, findable, [], []

    def create(self, agent_id, *, text, tags, created_at):
        p = SimpleNamespace(
            id=f"p{len(self.rows)}",
            text=text,
            tags=tags,
            created_at=created_at,
            embedding=[0.1] * 4 if self.embed else [],
        )
        self.rows.append(p)
        return [p]

    def search(self, agent_id, *, query, top_k, tags=None):
        self.searches.append({"query": query, "top_k": top_k, "tags": tags})
        hits = [r for r in self.rows if self.findable and (not tags or set(tags) & set(r.tags))][:top_k]
        return SimpleNamespace(
            results=[SimpleNamespace(id=r.id, content=r.text, timestamp=r.created_at, tags=r.tags) for r in hits]
        )

    def delete(self, memory_id, *, agent_id):
        self.deleted.append(memory_id)
        self.rows = [r for r in self.rows if r.id != memory_id]


class FactsLLM:
    def __init__(self, answer):
        self.answer, self.prompts = answer, []

    def complete_json(self, prompt):
        self.prompts.append(prompt)
        return self.answer


def _letta(passages, llm=None, **kw):
    return Letta("agent-1", llm=llm, client=SimpleNamespace(agents=SimpleNamespace(passages=passages)), **kw)


CHAT = [
    {"role": "user", "content": "I just moved to Melbourne."},
    {"role": "assistant", "content": "Big move!"},
    {"role": "user", "content": "And I started at Northern Logistics."},
    {"role": "assistant", "content": "Congrats."},
]


def test_letta_chunks_eight_messages_as_role_lines_capped_at_8000_chars():
    msgs = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"} for i in range(10)]
    out = chunks([*msgs, {"role": "system", "content": "skip me"}, {"role": "user", "content": " "}])
    assert out == ["\n".join(f"{m['role']}: {m['content']}" for m in msgs[:8]), "user: m8\nassistant: m9"]
    assert [len(c) for c in chunks([{"role": "user", "content": "x" * 9000}])] == [8000, 1006]


def test_letta_exchanges_keep_each_user_message_with_its_replies():
    text = "assistant: hello\nuser: I live in Darwin.\nassistant: Nice.\nassistant: More.\nuser: Bye."
    assert exchanges(text) == [
        "assistant: hello",
        "user: I live in Darwin.\nassistant: Nice.\nassistant: More.",
        "user: Bye.",
    ]


def test_letta_add_writes_dated_tagged_passages_and_returns_the_extracted_facts():
    ps, llm = (
        FakePassages(),
        FactsLLM({"facts": ["User lives in Melbourne.", " ", 3, "User works at Northern Logistics."]}),
    )
    new = _letta(ps, llm).add(CHAT, "2022-02-25")
    assert new == [
        Memory("User lives in Melbourne.", "2022-02-25"),
        Memory("User works at Northern Logistics.", "2022-02-25"),
    ]
    assert "dated 2022-02-25" in llm.prompts[0] and "user: I just moved to Melbourne." in llm.prompts[0]
    (row,) = ps.rows
    assert row.created_at == "2022-02-25T10:00:00Z" and row.tags[0].startswith("savestate-")
    assert ps.searches[-1]["tags"] == row.tags  # the check searches this conversation's passages only


def test_letta_add_needs_an_llm_and_writes_nothing_when_extraction_fails():
    ps = FakePassages()
    with pytest.raises(TypeError, match="needs llm"):
        _letta(ps).add(CHAT, "2022-02-25")
    with pytest.raises(LettaError, match="no JSON facts"):
        _letta(ps, FactsLLM({})).add(CHAT, "2022-02-25")
    assert ps.rows == []


def test_letta_add_deletes_what_it_wrote_when_a_passage_has_no_embedding_or_cannot_be_found():
    for ps, msg in ((FakePassages(embed=False), "without an embedding"), (FakePassages(findable=False), "cannot find")):
        with pytest.raises(LettaError, match=msg):
            _letta(ps, FactsLLM({"facts": []})).add(CHAT, "2022-02-25")
        assert ps.rows == [] and ps.deleted == ["p0"]


def test_letta_search_returns_dated_exchanges_or_whole_chunks():
    ps = FakePassages()
    _letta(ps, FactsLLM({"facts": []})).add(CHAT, "2022-02-25")
    split = _letta(ps).search("where do I live", 10)
    assert [m.text for m in split] == [
        "user: I just moved to Melbourne.\nassistant: Big move!",
        "user: And I started at Northern Logistics.\nassistant: Congrats.",
    ]
    assert {m.date for m in split} == {"2022-02-25"} and [m.id for m in split] == ["p0#0", "p0#1"]
    whole = _letta(ps, split=False).search("where do I live", 10)
    assert len(whole) == 1 and whole[0].text.count("user: ") == 2 and whole[0].id == "p0"


def test_letta_key_over_plain_http_to_another_host_is_refused(monkeypatch):
    monkeypatch.setenv("LETTA_API_KEY", "sk-test")
    with pytest.raises(ValueError, match="plain http"):
        Letta("agent-1", "http://letta.lan:8283")
    assert Letta("agent-1", client=SimpleNamespace()).user_id == "agent-1"
