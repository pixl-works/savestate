import json

import httpx
import pytest

from savestate import AnthropicLLM, OpenAICompatLLM
from savestate import llm as llm_mod
from savestate.llm import LLMError, parse_json, thinking_off


def test_thinking_is_off_per_model_family():
    assert thinking_off("claude-haiku-5-5") == {"thinking": {"type": "disabled"}}
    assert thinking_off("claude-sonnet-5-5") == {"thinking": {"type": "between_tools"}}
    assert thinking_off("claude-opus-5-5") == {"thinking": {"type": "between_tools"}}


def test_parse_json_takes_the_first_object_only():
    assert parse_json('noise {"a": 1} {"b": 2} trailing') == {"a": 1}
    assert parse_json("no json") == {} and parse_json('{"cut": ') == {} and parse_json("") == {}


def capture(reply, status=200):
    seen = []

    def handler(req):
        seen.append(req)
        return httpx.Response(status, json=reply)

    return seen, httpx.Client(transport=httpx.MockTransport(handler))


def test_anthropic_request_has_thinking_off_a_key_header_and_never_temperature():
    seen, client = capture({"stop_reason": "end_turn", "content": [{"type": "text", "text": 'x {"a": 1}'}]})
    llm = AnthropicLLM(
        "claude-haiku-5-5", api_key="k", base_url="http://proxy/anthropic/", client=client, allow_http=True
    )
    assert llm.complete_json("p") == {"a": 1}
    body = json.loads(seen[0].content)
    assert str(seen[0].url) == "http://proxy/anthropic/v1/messages" and seen[0].headers["x-api-key"] == "k"
    assert body["thinking"] == {"type": "disabled"} and "temperature" not in body


def test_anthropic_system_prompt_and_refusal():
    seen, client = capture({"stop_reason": "refusal", "content": []})
    llm = AnthropicLLM("claude-sonnet-5-5", api_key="k", client=client)
    assert llm.complete("p", system="rules") == "" and llm.complete_json("p") == {}
    assert json.loads(seen[0].content)["system"] == "rules"
    assert json.loads(seen[0].content)["thinking"] == {"type": "between_tools"}


def test_a_missing_key_fails_on_first_use_not_at_construction(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    llm = AnthropicLLM()
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        llm.complete("p")


def test_openai_compat_request_and_reply():
    seen, client = capture({"choices": [{"message": {"content": '{"ok": true}'}}]})
    llm = OpenAICompatLLM("m", base_url="http://x/v1", api_key="k", client=client, allow_http=True)
    assert llm.complete_json("p") == {"ok": True}
    body = json.loads(seen[0].content)
    assert str(seen[0].url) == "http://x/v1/chat/completions" and seen[0].headers["authorization"] == "Bearer k"
    assert "temperature" not in body and body["messages"] == [{"role": "user", "content": "p"}]


def test_transient_errors_are_retried_and_hard_errors_are_not(monkeypatch):
    monkeypatch.setattr(llm_mod.time, "sleep", lambda s: None)
    codes = iter([529, 429, 200])

    def flaky(req):
        c = next(codes)
        return httpx.Response(c, json={"content": [{"type": "text", "text": "{}"}]})

    llm = AnthropicLLM(api_key="k", client=httpx.Client(transport=httpx.MockTransport(flaky)))
    assert llm.complete_json("p") == {}
    seen, client = capture({"error": "bad"}, status=400)
    with pytest.raises(LLMError, match="400"):
        AnthropicLLM(api_key="k", client=client).complete("p")
    assert len(seen) == 1


# --- an API key is never sent in clear text to another machine ---------------------------------------------------


@pytest.mark.parametrize(
    "url",
    ["http://localhost:8890", "http://127.0.0.1:8890/x", "http://127.9.9.9", "http://[::1]:8890", "http://LOCALHOST"],
)
def test_loopback_http_with_a_key_is_allowed(url):
    AnthropicLLM(api_key="k", base_url=url)
    OpenAICompatLLM("m", base_url=url, api_key="k")


@pytest.mark.parametrize(
    "url", ["http://example.com", "http://10.0.0.5:8890", "http://127.0.0.1@evil.example/x", "HTTP://llm.lan/v1"]
)
def test_http_with_a_key_to_another_host_is_refused_by_every_llm_client(url, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="plain http"):
        AnthropicLLM(api_key="secret-key", base_url=url)
    with pytest.raises(ValueError, match="plain http") as e:
        OpenAICompatLLM("m", base_url=url, api_key="secret-key")
    assert "secret-key" not in str(e.value)
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")  # a key from the environment counts too
    with pytest.raises(ValueError, match="plain http"):
        OpenAICompatLLM("m", base_url=url)


def test_https_and_http_without_a_key_stay_allowed(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    AnthropicLLM(api_key="k", base_url="https://llm.example.com")
    AnthropicLLM(base_url="http://llm.lan")  # no key: nothing to leak (it fails on first use instead)
    OpenAICompatLLM("m", base_url="https://llm.example.com/v1", api_key="k")
    OpenAICompatLLM("m", base_url="http://ollama.lan:11434/v1")  # the placeholder key "none" is not a secret


def test_anthropic_key_from_the_environment_and_base_url_from_the_environment_are_checked(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://llm.lan")
    with pytest.raises(ValueError, match="plain http"):
        AnthropicLLM()


def test_allow_http_is_the_explicit_escape_hatch():
    AnthropicLLM(api_key="k", base_url="http://llm.lan", allow_http=True)
    OpenAICompatLLM("m", base_url="http://llm.lan/v1", api_key="k", allow_http=True)
