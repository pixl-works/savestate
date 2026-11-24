import json
from datetime import date

import pytest

from savestate import Card, Memory
from savestate.card import Line, dates_in, read_view, trim_history
from savestate.testing import FakeLLM


def line(name, text, ids=("m1",)):
    return {"name": name, "text": text, "ids": list(ids)}


def card_with(script, **kw):
    llm = FakeLLM(script)
    return Card(llm, **kw), llm


def test_update_adds_a_line_and_logs_the_source_memory_ids(tmp_path):
    card, llm = card_with(
        [
            {
                "lines": [
                    line(
                        "residence", "User's residence (as of 2026-03-01): Oslo. Before: Rome until 2026-03-01.", ["m2"]
                    )
                ]
            }
        ],
        path=tmp_path / "card.json",
    )
    changes = card.update([Memory("likes tea", id="a"), Memory("moved to Oslo from Rome", id="b")], "2026-03-01")
    assert card.lines["residence"].text.startswith("User's residence (as of 2026-03-01): Oslo")
    assert changes == [
        {"date": "2026-03-01", "name": "residence", "before": None, "after": card.lines["residence"].text, "ids": ["b"]}
    ]
    assert "[m2] moved to Oslo from Rome" in llm.prompts[0]
    logged = [json.loads(x) for x in (tmp_path / "card.log.jsonl").read_text().splitlines()]
    assert logged == changes


def test_card_persists_and_reloads(tmp_path):
    card, _ = card_with(
        [{"lines": [line("employer", "User's employer (as of 2026-02-01): Acme.")]}], path=tmp_path / "c.json"
    )
    card.update(["joined Acme"], date(2026, 2, 1))
    again = Card(FakeLLM(), tmp_path / "c.json")
    assert again.lines["employer"].as_of == "2026-02-01"
    assert (tmp_path / "c.json").stat().st_mode & 0o777 == 0o600


def test_fixed_facts_never_reach_the_card():
    card, _ = card_with(
        [
            {
                "lines": [
                    line("birthday", "User's birthday (as of 2026-01-01): 1 May."),
                    line("University", "User's university (as of 2026-01-01): MIT."),
                    line("degree", "User's degree (as of 2026-01-01): BSc."),
                ]
            }
        ]
    )
    changes = card.update(["I was born on 1 May"], "2026-01-01")
    assert card.lines == {}
    assert {c["reason"] for c in changes} == {"fixed fact"}


def test_blocklist_rejects_plumbing_by_name_or_text():
    card, _ = card_with(
        [
            {
                "lines": [
                    line("home", "User's home (as of 2026-01-01): the Echo Show in the kitchen is set up."),
                    line("residence", "User's residence (as of 2026-01-01): Oslo."),
                ]
            }
        ],
        blocklist=r"echo show|deploy",
    )
    card.update(["x"], "2026-01-01")
    assert list(card.lines) == ["residence"]


@pytest.mark.parametrize(
    "text,why",
    [
        ("User's residence: Oslo, no stamp at all here.", "missing (as of DATE) stamp"),
        ("User's residence (as of 2025-12-31): Oslo, the wrong day.", "missing (as of DATE) stamp"),
        ("Oslo is where she lives (as of 2026-01-01).", "does not start with User's"),
        ("User's residence (as of 2026-01-01): " + "Oslo, " * 80, "too long"),
    ],
)
def test_date_stamp_subject_and_length_guards(text, why):
    card, _ = card_with([{"lines": [line("residence", text)]}])
    changes = card.update(["x"], "2026-01-01")
    assert card.lines == {} and changes[0]["reason"] == why


def test_a_non_iso_date_is_refused():
    card, _ = card_with([{}])
    with pytest.raises(ValueError):
        card.update(["x"], "1 Jan 2026")


def test_subject_names_the_person_in_the_prompt_and_the_guard():
    card, llm = card_with([{"lines": [line("residence", "Sam's residence (as of 2026-01-01): Oslo.")]}], subject="Sam")
    card.update(["x"], "2026-01-01")
    assert "Sam's residence" in card.render() and "ONE person, Sam" in llm.prompts[0]


def test_custom_names_appear_in_the_prompt_with_meanings():
    card, llm = card_with([{}], names={"partner's health": "how the partner is doing", "kids": ""})
    card.update(["x"], "2026-01-01")
    assert "- partner's health: how the partner is doing\n- kids\n" in llm.prompts[0]


def test_history_is_capped_at_the_two_newest_before_values():
    long = (
        "User's residence (as of 2026-04-01): D. Before: C until 2026-04-01. Before that: B until 2026-03-01. "
        "Before that: A until 2026-02-01."
    )
    assert trim_history(long, 2).endswith("Before that: B until 2026-03-01.")
    card, _ = card_with([{"lines": [line("residence", long)]}])
    card.update(["x"], "2026-04-01")
    assert "A until" not in card.lines["residence"].text and "B until" in card.lines["residence"].text
    assert trim_history(long, 1).endswith("Before: C until 2026-04-01.")


def test_json_with_trailing_text_and_garbage_is_survived():
    class Chatty(FakeLLM):
        def complete_json(self, prompt):
            from savestate.llm import parse_json

            return parse_json(
                'Sure! {"lines": [{"name": "health", "text": "User\'s health (as of 2026-01-01): flu."}]} {"extra": 1}'
            )

    card = Card(Chatty())
    card.update(["x"], "2026-01-01")
    assert "health" in card.lines
    garbage = Card(FakeLLM([{"lines": "not a list"}]))
    assert "no JSON" in garbage.update(["x"], "2026-01-01")[0]["error"]


def test_a_prose_answer_is_logged_not_silently_swallowed(tmp_path):
    card, _ = card_with([{}], path=tmp_path / "card.json")  # what complete_json returns for a refusal or prose
    assert card.update(["x"], "2026-01-01")[0]["error"].startswith("the model returned no JSON")
    assert "error" in (tmp_path / "card.log.jsonl").read_text() and card.lines == {}
    quiet, _ = card_with([{"lines": []}])  # a real "nothing changed" is not an error
    assert quiet.update(["x"], "2026-01-01") == []


def test_repeating_the_current_line_is_not_a_change():
    same = "User's residence (as of 2026-01-01): Oslo."
    card, _ = card_with([{"lines": [line("residence", same)]}, {"lines": [line("residence", same)]}])
    card.update(["a"], "2026-01-01")
    assert card.update(["b"], "2026-01-01") == []


def test_merge_pass_runs_only_when_a_new_name_appears_and_folds_duplicates():
    script = [
        {"lines": [line("relationship", "User's relationship (as of 2026-01-01): single.")]},
        {"lines": [line("partner", "User's partner (as of 2026-02-01): Alex. Before: none until 2026-02-01.")]},
        {
            "merges": [
                {
                    "keep": "relationship",
                    "drop": ["partner"],
                    "text": "User's relationship (as of 2026-02-01): with Alex. Before: single until 2026-02-01.",
                }
            ]
        },
    ]
    card, llm = card_with(script)
    card.update(["a"], "2026-01-01")
    assert len(llm.prompts) == 1  # a first line is a new name, but there is nothing to merge it with
    changes = card.update(["b"], "2026-02-01")
    assert len(llm.prompts) == 3 and "STATE CARD about User" in llm.prompts[2]
    assert list(card.lines) == ["relationship"] and card.lines["relationship"].as_of == "2026-02-01"
    assert changes[-1]["merge"] == "relationship" and changes[-1]["drop"] == ["partner"]


def test_merge_that_breaks_a_guard_is_ignored():
    script = [
        {"lines": [line("a", "User's a (as of 2026-01-01): x.")]},
        {"lines": [line("b", "User's b (as of 2026-01-01): y.")]},
        {"merges": [{"keep": "a", "drop": ["b"], "text": "no stamp and wrong subject"}]},
    ]
    card, _ = card_with(script)
    card.update(["1"], "2026-01-01")
    card.update(["2"], "2026-01-01")
    assert set(card.lines) == {"a", "b"}


OLD = (
    "User's work status (as of 2026-11-01): on a business trip until 2026-11-10; flying to Lisbon on 2026-11-26. "
    "Before: busy until 2026-10-20."
)


def test_freshen_shortens_past_dates_but_never_removes_a_future_date():
    card = Card(
        FakeLLM(
            [
                {
                    "lines": [
                        {
                            "name": "work status",
                            "text": "User's work status (as of 2026-11-01): flying to Lisbon on 2026-11-26.",
                        },  # keeps future: ok
                    ]
                }
            ]
        )
    )
    card.lines = {"work status": Line(OLD, "2026-11-01")}
    changes = card.freshen("2026-11-20")
    assert card.lines["work status"].text == "User's work status (as of 2026-11-01): flying to Lisbon on 2026-11-26."
    assert changes[0]["freshen"] == "work status"


def test_freshen_refuses_to_strike_a_date_that_is_not_past_yet():
    card = Card(
        FakeLLM(
            [
                {
                    "lines": [
                        {"name": "work status", "text": "User's work status (as of 2026-11-01): busy until 2026-10-20."}
                    ],
                    "drop": [],
                }
            ]
        )
    )
    card.lines = {"work status": Line(OLD, "2026-11-01")}
    assert card.freshen("2026-11-20") == [] and card.lines["work status"].text == OLD


def test_freshen_cannot_reword_lengthen_or_drop_the_stamp():
    old = "User's health (as of 2026-11-01): a cold. Before: fine until 2026-11-01."
    for bad in [
        "User's health (as of 2026-11-01): a cold. Before: fine until 2026-11-01. And much more text.",
        "User's health: fine.",
    ]:
        card = Card(FakeLLM([{"lines": [{"name": "health", "text": bad}]}]))
        card.lines = {"health": Line(old, "2026-11-01")}
        assert card.freshen("2026-12-01") == [] and card.lines["health"].text == old


def test_freshen_drop_removes_expired_lines_but_not_lines_with_future_dates(tmp_path):
    card = Card(FakeLLM([{"lines": [], "drop": ["cold", "trip"]}]), tmp_path / "c.json")
    card.lines = {
        "cold": Line("User's cold (as of 2026-10-01): a cold.", "2026-10-01"),
        "trip": Line("User's trip (as of 2026-11-01): Lisbon on 2026-11-26.", "2026-11-01"),
    }
    changes = card.freshen("2026-11-20")
    assert list(card.lines) == ["trip"] and changes[0]["after"] is None
    assert "cold" in (tmp_path / "c.log.jsonl").read_text()


def test_render_shows_current_value_and_last_change_only():
    card = Card(FakeLLM())
    card.lines = {
        "residence": Line(
            "User's residence (as of 2026-04-01): D. Before: C until 2026-04-01. Before that: B until 2026-03-01.", "x"
        )
    }
    assert card.render() == "STATE CARD:\n- User's residence (as of 2026-04-01): D. Before: C until 2026-04-01."
    assert "Before that: B" in card.render(full=True)
    assert Card(FakeLLM()).render() == "STATE CARD:\n- (empty so far)"
    assert read_view("plain line") == "plain line"


def test_dates_in_reads_both_formats():
    assert dates_in("on 26 Nov 2026 and 2026-12-01, not Nov 2026") == {"2026-11-26", "2026-12-01"}


def test_large_updates_are_batched_and_saved_per_batch(tmp_path):
    seen = []

    def script(prompt):
        seen.append(prompt.count("\n[m"))
        return {}

    card = Card(FakeLLM(script), tmp_path / "c.json")
    card.update([f"fact {i}" for i in range(130)], "2026-01-01")
    assert seen == [60, 60, 10]


def test_a_model_answer_with_the_wrong_shape_does_not_crash_update_or_freshen():
    card, _ = card_with(
        [
            {
                "lines": [
                    {"name": "residence", "text": "User's residence (as of 2026-03-01): Oslo.", "ids": None},
                    "junk",
                ]
            },
            {
                "lines": [
                    {"name": "employer", "text": "User's employer (as of 2026-03-01): Acme.", "ids": [["m1"], 7, "m1"]}
                ]
            },
            {"lines": ["junk", 5]},
        ]
    )
    changes = card.update([Memory("moved to Oslo", id="a")], "2026-03-01") + card.update(
        ["works at Acme"], "2026-03-01"
    )
    assert [c["ids"] for c in changes if "ids" in c] == [[], []] and set(card.lines) == {"residence", "employer"}
    assert card.freshen("2026-04-01") == []


def test_one_memory_passed_alone_is_one_memory_not_a_list_of_characters():
    card, llm = card_with([{}])
    card.update("I moved to Oslo", "2026-03-01")
    card.update(Memory("I changed jobs"), "2026-03-01")
    assert "[m1] I moved to Oslo\n" in llm.prompts[0] + "\n" and "[m2]" not in llm.prompts[0]
    assert "[m1] I changed jobs" in llm.prompts[1]


# --- the card belongs to one person ----------------------------------------------------------------------------


def saved_card(path, **kw):
    ask = {"lines": [line("residence", "User's residence (as of 2026-03-01): Oslo.")]}
    card, _ = card_with([ask], path=path, **kw)
    card.update(["moved to Oslo"], "2026-03-01")
    return card


def test_the_card_file_stores_the_user_id_and_the_same_id_loads(tmp_path):
    p = tmp_path / "card.json"
    saved_card(p, user_id="u1")
    assert json.loads(p.read_text())["user_id"] == "u1"
    again = Card(FakeLLM([]), p, user_id="u1")
    assert again.user_id == "u1" and "residence" in again.lines


def test_a_card_file_of_another_user_raises_and_stays_byte_identical(tmp_path):
    p = tmp_path / "card.json"
    saved_card(p, user_id="u1")
    before = p.read_bytes()
    with pytest.raises(ValueError, match=r"'u1'.*'u2'"):
        Card(FakeLLM([]), p, user_id="u2")
    assert p.read_bytes() == before and [f.name for f in tmp_path.iterdir()] == ["card.json", "card.log.jsonl"]


def test_a_card_without_an_id_is_adopted_on_the_next_save(tmp_path):
    p = tmp_path / "card.json"
    saved_card(p)  # an older card: no id on disk
    assert "user_id" not in json.loads(p.read_text())
    ask = {"lines": [line("employer", "User's employer (as of 2026-04-01): Acme.")]}
    card = Card(FakeLLM([ask]), p, user_id="u1")
    assert "residence" in card.lines
    card.update(["works at Acme"], "2026-04-01")
    assert json.loads(p.read_text())["user_id"] == "u1"


def test_without_a_given_id_the_id_in_the_file_is_kept(tmp_path):
    p = tmp_path / "card.json"
    saved_card(p, user_id="u1")
    card = Card(FakeLLM([{"lines": [line("employer", "User's employer (as of 2026-04-01): Acme.")]}]), p)
    card.update(["works at Acme"], "2026-04-01")
    assert json.loads(p.read_text())["user_id"] == "u1"


def test_an_empty_user_id_is_refused():
    with pytest.raises(ValueError, match="user_id"):
        Card(FakeLLM([]), user_id=" ")
