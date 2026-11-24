"""LLM judge with half credit: 1 = states what the reference says, 0.5 = partly right or buried among wrong options,
0 = wrong, missing, "cannot confirm", or an outdated value given as the answer. Also flags outdated-value errors."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from savestate.models import LLM

JUDGE_PROMPT = """You grade an answer from an assistant's memory against the reference answer.

Question: {question}
Reference answer: {gold}
Assistant's answer: {answer}

Score 1 if the assistant's answer states what the reference says (paraphrase is fine; a hedge is fine if the
final answer matches). Score 0.5 if it is partly right or right but buried among wrong options. Score 0 if it is
wrong, missing, "cannot confirm", or gives an outdated or contradicted value as the answer.
Also say whether the main error (if any) is an OUTDATED value (a state that was true earlier).
Answer with JSON only: {{"score": 1, "outdated": false}}"""


def judge_file(llm: LLM, directory: Path, workers: int = 8) -> dict[str, Any]:
    """Judge directory/predictions.jsonl into judged.jsonl (judge = {"score", "outdated"} or None when unusable)."""
    recs = [json.loads(line) for line in (directory / "predictions.jsonl").read_text().splitlines() if line.strip()]

    def one(r: dict[str, Any]) -> dict[str, Any]:
        if not r["answer"].strip():  # a refusal or an empty answer is not a score of 0: it is excluded for every arm
            return {**r, "judge": None}
        verdict = llm.complete_json(JUDGE_PROMPT.format(question=r["question"], gold=r["gold"], answer=r["answer"]))
        return {**r, "judge": verdict if isinstance(verdict.get("score"), int | float) else None}

    with ThreadPoolExecutor(workers) as pool:
        out = list(pool.map(one, recs))
    (directory / "judged.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
    return {"answers": len(out), "judged": sum(r["judge"] is not None for r in out)}
