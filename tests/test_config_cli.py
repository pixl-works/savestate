import json

import pytest

from savestate import AnthropicLLM, HttpReranker, InMemory, Mem0Http, Memory, NoRerank, cli
from savestate.config import build_card, build_layer, build_llm, build_reranker, build_store, load_config
from savestate.testing import FakeLLM

TOML = """
[store]
type = "mem0"
url = "http://mem0:8888"
user_id = "me"
[card]
path = "{card}"
subject = "Sam"
[recall]
k = 4
[rerank]
url = "http://rr/v1/rerank"
"""


def test_load_config_reads_toml_and_environment_overrides_it(tmp_path):
    f = tmp_path / "c.toml"
    f.write_text(TOML.format(card=tmp_path / "card.json"))
    cfg = load_config(f, env={"SAVESTATE_USER_ID": "other", "SAVESTATE_K": "7", "SAVESTATE_OWNER_FILTER": "true"})
    assert cfg["store"] == {"type": "mem0", "url": "http://mem0:8888", "user_id": "other"}
    assert cfg["recall"] == {"k": 7, "owner_filter": True}


def test_load_config_from_environment_alone_and_missing_explicit_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert load_config(None, env={"SAVESTATE_STORE": "memos"}) == {"store": {"type": "memos"}}
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.toml", env={})


def test_builders_pick_the_right_parts(tmp_path):
    f = tmp_path / "c.toml"
    f.write_text(TOML.format(card=tmp_path / "card.json"))
    cfg = load_config(f, env={})
    assert isinstance(build_store(cfg), Mem0Http) and isinstance(build_reranker(cfg), HttpReranker)
    assert isinstance(build_llm(cfg), AnthropicLLM) and isinstance(build_reranker({}), NoRerank)
    layer = build_layer(cfg)
    assert layer.card.subject == "Sam" and layer.recall.k == 4
    with pytest.raises(ValueError, match="store"):
        build_store({})


def run(capsys, argv, monkeypatch, tmp_path, script=None):
    f = tmp_path / "c.toml"
    f.write_text(TOML.format(card=tmp_path / "card.json"))
    monkeypatch.setattr("savestate.config.build_llm", lambda cfg: FakeLLM(script or [{}]))
    assert cli.main(["--config", str(f), *argv]) == 0
    return capsys.readouterr().out


def test_cli_card_update_show_and_freshen(capsys, monkeypatch, tmp_path):
    ask = {"lines": [{"name": "residence", "text": "Sam's residence (as of 2026-03-01): Oslo.", "ids": ["m1"]}]}
    mem = tmp_path / "new.txt"
    mem.write_text('{"id": "7", "text": "moved to Oslo"}\n\nplain line\n')
    out = run(capsys, ["card", "update", "--date", "2026-03-01", "--file", str(mem)], monkeypatch, tmp_path, [ask])
    assert json.loads(out)[0]["ids"] == ["7"]
    shown = run(capsys, ["card", "show"], monkeypatch, tmp_path)
    assert shown.strip() == "STATE CARD:\n- Sam's residence (as of 2026-03-01): Oslo."
    assert json.loads(run(capsys, ["card", "freshen", "--today", "2026-04-01"], monkeypatch, tmp_path)) == []


def test_cli_context(capsys, monkeypatch, tmp_path):
    monkeypatch.setattr("savestate.config.build_store", lambda cfg: InMemory([Memory("lives in Oslo", "2026-03-01")]))
    out = run(capsys, ["context", "where in Oslo?", "--no-rules"], monkeypatch, tmp_path)
    assert "[2026-03-01] lives in Oslo" in out and "RULES" not in out


def test_cli_requires_a_command():
    with pytest.raises(SystemExit):
        cli.main([])


def test_config_refuses_a_key_over_plain_http_and_allow_http_opens_it(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("MEM0_API_KEY", "k")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    llm = {"llm": {"base_url": "http://llm.lan"}}
    with pytest.raises(ValueError, match="plain http"):
        build_llm(llm)
    with pytest.raises(ValueError, match="plain http"):
        build_llm({"llm": {"provider": "openai", "base_url": "http://llm.lan/v1"}})
    with pytest.raises(ValueError, match="plain http"):
        build_store({"store": {"type": "mem0", "url": "http://mem0.lan"}})
    assert isinstance(build_llm({"llm": {**llm["llm"], "allow_http": True}}), AnthropicLLM)
    assert build_llm({"llm": {"provider": "openai", "base_url": "http://llm.lan/v1", "allow_http": True}})
    assert isinstance(build_store({"store": {"type": "mem0", "url": "http://mem0.lan", "allow_http": True}}), Mem0Http)
    with pytest.raises(ValueError, match="plain http"):  # a quoted "true" is not the boolean true
        build_llm({"llm": {**llm["llm"], "allow_http": "true"}})


def test_allow_http_comes_from_toml_and_from_the_environment(tmp_path):
    f = tmp_path / "c.toml"
    f.write_text('[llm]\nbase_url = "http://llm.lan"\nallow_http = true\n[store]\ntype = "mem0"\n')
    cfg = load_config(f, env={"SAVESTATE_STORE_ALLOW_HTTP": "yes"})
    assert cfg["llm"]["allow_http"] is True and cfg["store"]["allow_http"] is True
    assert load_config(f, env={"SAVESTATE_LLM_ALLOW_HTTP": "false"})["llm"]["allow_http"] is False


def test_the_card_built_from_config_is_bound_to_the_configured_user(tmp_path):
    cfg = {"store": {"user_id": "me"}, "card": {"path": str(tmp_path / "card.json")}}
    assert build_card(cfg, FakeLLM([])).user_id == "me"
    assert build_card({"card": cfg["card"]}, FakeLLM([])).user_id is None
