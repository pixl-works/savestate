import threading

import httpx

from savestate import HttpReranker, Memory, NoRerank
from savestate import rerank as rerank_mod
from savestate.server import make_server

MEMS = [Memory("old stale fact", id="a"), Memory("the right fact", id="b"), Memory("noise", id="c")]


def client_for(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_http_reranker_orders_by_score():
    def handler(req):
        return httpx.Response(200, json={"scores": [0.1, 0.9, 0.5]})

    rr = HttpReranker("http://rr/v1/rerank", client=client_for(handler))
    assert [m.id for m in rr.rerank("q", MEMS)] == ["b", "c", "a"] and rr.stats["reranked"] == 1


def test_http_reranker_falls_back_to_host_order_and_stays_away_during_cooldown(monkeypatch):
    calls = []

    def handler(req):
        calls.append(1)
        raise httpx.ConnectError("down")

    rr = HttpReranker("http://rr/v1/rerank", cooldown=120, client=client_for(handler))
    assert [m.id for m in rr.rerank("q", MEMS)] == ["a", "b", "c"] and len(calls) == 1
    assert "ConnectError" in rr.stats["last_error"]
    rr.rerank("q", MEMS)
    assert len(calls) == 1 and rr.stats["fallback"] == 2  # cooldown: no second attempt
    now = rerank_mod.time.monotonic()
    monkeypatch.setattr(rerank_mod.time, "monotonic", lambda: now + 121)
    rr.rerank("q", MEMS)
    assert len(calls) == 2  # cooldown over: tries again


def test_http_reranker_treats_a_malformed_answer_as_failure():
    for bad in [{"scores": [1.0]}, {"nope": 1}, {"scores": [None, "x", 1]}, {"scores": {"0": 1, "1": 2, "2": 3}}]:
        rr = HttpReranker("http://rr", client=client_for(lambda req, bad=bad: httpx.Response(200, json=bad)))
        assert [m.id for m in rr.rerank("q", MEMS)] == ["a", "b", "c"] and rr.stats["fallback"] == 1


def test_http_reranker_times_out_into_fallback():
    def handler(req):
        raise httpx.ReadTimeout("slow")

    rr = HttpReranker("http://rr", timeout=0.01, client=client_for(handler))
    assert [m.id for m in rr.rerank("q", MEMS)] == ["a", "b", "c"]


def test_single_memory_needs_no_call():
    rr = HttpReranker("http://rr", client=client_for(lambda req: (_ for _ in ()).throw(AssertionError("called"))))
    assert rr.rerank("q", MEMS[:1]) == MEMS[:1] and NoRerank().rerank("q", MEMS) == MEMS


def test_serve_rerank_round_trip_through_a_real_http_server():
    """The server and the client speak the same contract; the scorer is a stand-in (no model needed)."""
    server = make_server(lambda q, docs: [float(len(d)) for d in docs], {"model": "fake"}, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        assert httpx.get(url + "/health").json() == {"ok": True, "model": "fake"}
        assert httpx.post(url + "/v1/rerank", json={"query": "q", "documents": []}).json()["scores"] == []
        assert httpx.post(url + "/nope", json={}).status_code == 404
        assert httpx.post(url + "/v1/rerank", content=b"not json").status_code == 400
        rr = HttpReranker(url + "/v1/rerank")
        short_first = [Memory("hi", id="s"), Memory("a much longer text", id="l")]
        assert [m.id for m in rr.rerank("q", short_first)] == ["l", "s"]  # the stand-in scorer prefers long texts
        assert rr.stats["reranked"] == 1
    finally:
        server.shutdown()


def test_cross_encoder_scorer_returns_scores_in_input_order_and_cleans_mps(monkeypatch):
    """No torch here: stand-ins for torch and sentence-transformers check the length-sorted batching and mapping."""
    import sys
    from types import SimpleNamespace

    calls = []

    class FakeCrossEncoder:
        def __init__(self, model, device=None, max_length=512):
            self.model = SimpleNamespace(device="mps:0", half=lambda: calls.append("half"))

        def predict(self, pairs, batch_size, show_progress_bar):
            calls.append((batch_size, [d for _, d in pairs]))
            return [float(len(d)) for _, d in pairs]  # score = length, in the (sorted) order it was given

    torch = SimpleNamespace(
        mps=SimpleNamespace(synchronize=lambda: calls.append("sync"), empty_cache=lambda: calls.append("empty"))
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=FakeCrossEncoder))
    scorer = rerank_mod.CrossEncoderScorer()
    assert scorer.score("q", ["ccc", "a", "bb"]) == [3.0, 1.0, 2.0]  # input order, though pairs ran shortest first
    assert calls == ["half", (8, ["a", "bb", "ccc"]), "sync", "empty"]
    local = rerank_mod.LocalCrossEncoder()
    assert [m.text for m in local.rerank("q", [Memory("a"), Memory("ccc"), Memory("bb")])] == ["ccc", "bb", "a"]


def test_a_missing_local_extra_says_how_to_install_it(monkeypatch):
    import sys

    import pytest

    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(ImportError, match=r"savestate\[local\]"):
        rerank_mod.CrossEncoderScorer()
