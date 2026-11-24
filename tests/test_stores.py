import json
import re
from datetime import UTC, datetime
from types import SimpleNamespace

import httpx
import pytest

from savestate import Honcho, Ingesting, InMemory, Mem0Http, Memory, MemOSHttp, Store
from savestate.stores import memos as memos_mod
from savestate.stores.honcho import HonchoError, _chunks


def test_every_adapter_satisfies_the_protocols():
    assert isinstance(InMemory(), Store) and isinstance(InMemory(), Ingesting)
    for cls in (Mem0Http, MemOSHttp, Honcho):
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
