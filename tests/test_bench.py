import json
import random
from pathlib import Path

from bench.memconflict import cli as bench_cli
from bench.memconflict.dataset import session_messages
from bench.memconflict.judge import judge_file
from bench.memconflict.report import compare, persona_scores, report
from bench.memconflict.run import run_persona

from savestate import InMemory, Layer
from savestate.testing import FakeLLM

# a hand-written persona in MemConflict's shape (the real data is never in this repo)
PERSONA = {
    "Full_Session_Chain": [
        {
            "Date": "2022-01-03",
            "Session_Dialogue": {
                "turn_2": [{"role": "assistant", "content": "ok"}],
                "turn_1": [{"role": "user", "content": "I live in Darwin"}],
            },
            "Session_Questions": [],
        },
        {
            "Date": "2022-02-03",
            "Session_Dialogue": {"turn_1": [{"role": "user", "content": "I moved to Perth"}]},
            "Session_Questions": [
                {
                    "question_id": "q1",
                    "question": "Where do I live?",
                    "answer": "Perth",
                    "conflict_type": "dynamic_conflict",
                    "difficulty": "easy",
                }
            ],
        },
    ]
}


class Scripted(FakeLLM):
    """complete_json maintains the card; complete answers and records which system prompt it was given."""

    def __init__(self):
        super().__init__(
            [
                {"lines": [{"name": "residence", "text": "User's residence (as of 2022-01-03): Darwin."}]},
                {
                    "lines": [
                        {
                            "name": "residence",
                            "text": "User's residence (as of 2022-02-03): Perth. Before: Darwin until 2022-02-03.",
                        }
                    ]
                },
            ]
        )
        self.systems: list[str] = []

    def complete(self, prompt, system="", max_tokens=400):
        self.systems.append(system)
        return "Perth" if "STATE CARD" in prompt else "Darwin"


def test_session_messages_orders_turns_numerically_and_skips_roleless_turns():
    s = {
        "Session_Dialogue": {
            "turn_10": [{"role": "user", "content": "c"}],
            "turn_2": [{"role": "user", "content": "b"}],
            "turn_1": [{"content": "no role"}, {"role": "assistant", "content": "a"}],
        }
    }
    assert [m["content"] for m in session_messages(s)] == ["a", "b", "c"]


def fake_clone(tmp_path, monkeypatch):
    ev = tmp_path / "MemConflict" / "Evaluation"
    ev.mkdir(parents=True)
    (ev / "eval_memzero.py").write_text('X = 1\nMEMZERO_ANSWER_SYSTEM_PROMPT = """Answer from memories."""\n')
    monkeypatch.setenv("MEMCONFLICT_REPO", str(tmp_path / "MemConflict"))


def test_host_prompt_is_read_from_the_memconflict_clone_never_shipped(tmp_path, monkeypatch):
    import pytest
    from bench.memconflict.run import host_system

    monkeypatch.delenv("MEMCONFLICT_REPO", raising=False)
    with pytest.raises(SystemExit):
        host_system()
    fake_clone(tmp_path, monkeypatch)
    assert host_system() == "Answer from memories."
    shipped = "".join(p.read_text() for p in Path(__file__).parents[1].glob("bench/**/*.py"))
    assert "You answer memory-evaluation questions" not in shipped


def test_run_persona_answers_each_arm_from_its_own_context(tmp_path, monkeypatch):
    fake_clone(tmp_path, monkeypatch)
    llm = Scripted()
    layer = Layer(InMemory(), llm, card_path=tmp_path / "card.json")
    info = run_persona(PERSONA, layer, llm, tmp_path, arms=("host", "layer", "layer-generic"), samples=2)
    recs = [json.loads(x) for x in (tmp_path / "predictions.jsonl").read_text().splitlines()]
    assert info == {"sessions": 2, "questions": 1, "card_lines": 1} and len(recs) == 6
    by_arm = {a: {r["answer"] for r in recs if r["arm"] == a} for a in ("host", "layer", "layer-generic")}
    assert by_arm == {
        "host": {"Darwin"},
        "layer": {"Perth"},
        "layer-generic": {"Perth"},
    }  # only layer arms see the card
    assert any("earliest recorded value" in s for s in llm.systems)  # the research arm keeps the benchmark rule...
    generic = [s for s in llm.systems if "STATE CARD" in s and "earliest recorded value" not in s]
    assert generic  # ...and layer-generic does not


def judged(path: Path, rows):
    path.mkdir(parents=True, exist_ok=True)
    (path / "judged.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def row(arm, qid, score, typ="dynamic_conflict", outdated=False):
    return {
        "arm": arm,
        "session": 1,
        "qid": qid,
        "type": typ,
        "judge": None if score is None else {"score": score, "outdated": outdated},
    }


def test_common_exclusion_drops_a_question_from_every_arm_when_any_arm_lacks_a_judgement():
    recs = [row("host", "a", 0), row("host", "b", 1), row("layer", "a", None), row("layer", "b", 1)]
    scores = persona_scores(recs, ["host", "layer"])
    assert (
        scores["host"]["dynamic_conflict"] == 1.0 and scores["layer"]["dynamic_conflict"] == 1.0
    )  # "a" excluded in both


def test_report_uses_half_credit_per_persona_deltas_and_labels_types(tmp_path):
    dirs = {}
    for p, (h, lay) in {"p0": (0.0, 1.0), "p1": (0.5, 1.0), "p2": (0.5, 0.5)}.items():
        dirs[p] = judged(
            tmp_path / p,
            [
                row("host", "a", h),
                row("layer", "a", lay, outdated=False),
                row("host", "s", 1, "static_conflict"),
                row("layer", "s", 0.5, "static_conflict"),
            ],
        )
    out = report(dirs, ["host", "layer"])
    assert out["accuracy_pct"]["host"]["changing"] == 33.3 and out["accuracy_pct"]["layer"]["changing"] == 83.3
    changing = out["vs_host"]["layer"]["changing"]
    assert changing["delta"] == 50.0 and changing["wins"] == 2 and changing["losses"] == 0 and changing["personas"] == 3
    assert out["vs_host"]["layer"]["never-change"]["delta"] == -50.0
    assert out["vs_host"]["layer"]["situational"] is None  # no such questions


def test_bootstrap_ci_brackets_the_mean_and_is_seeded():
    d = [0.2, 0.4, 0.3, 0.1, 0.5]
    a, b = compare(d, random.Random(0)), compare(d, random.Random(0))
    assert a == b and a["ci95"][0] <= a["delta"] <= a["ci95"][1] and a["wins"] == 5


def test_judge_file_marks_empty_and_unparseable_answers_as_unjudged(tmp_path):
    recs = [
        {"arm": "host", "session": 0, "qid": "a", "type": "dynamic_conflict", "question": "q", "gold": "g", "answer": x}
        for x in ("right", "", "garbage")
    ]
    (tmp_path / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    verdicts = {"right": {"score": 1, "outdated": False}, "garbage": {"nope": 1}}
    llm = FakeLLM(lambda prompt: next(v for k, v in verdicts.items() if f"answer: {k}" in prompt))
    assert judge_file(llm, tmp_path) == {"answers": 3, "judged": 1}
    out = [json.loads(x)["judge"] for x in (tmp_path / "judged.jsonl").read_text().splitlines()]
    assert out == [{"score": 1, "outdated": False}, None, None]


def test_a_rerun_starts_from_an_empty_card_not_the_previous_runs(tmp_path, monkeypatch):
    out = tmp_path / "memconflict" / "p00"
    out.mkdir(parents=True)
    (out / "card.json").write_text(
        json.dumps({"lines": {"employer": {"text": "User's employer (as of 2020-01-01): Old.", "as_of": "2020-01-01"}}})
    )
    seen = {}
    monkeypatch.setattr(bench_cli, "load_personas", lambda: {0: PERSONA})
    monkeypatch.setattr(bench_cli, "_llm", lambda a: FakeLLM())
    monkeypatch.setattr(bench_cli, "run_persona", lambda item, layer, *a: seen.update(lines=layer.card.lines) or {})
    bench_cli.main(["--out", str(tmp_path), "run", "--persona", "0", "--store", "memory"])
    assert seen["lines"] == {}
