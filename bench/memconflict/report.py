"""The comparison, as fixed before the runs: half-credit judge score; common exclusion; per-persona deltas with a
bootstrap CI over personas.

* Common exclusion: a question (session, id) that has no usable judgement in ANY arm is dropped from ALL arms, so
  arms are always scored on identical questions.
* Per persona and conflict type, an arm's score is the mean over its questions and samples; arms are compared by the
  per-persona difference against the base arm.
* The confidence interval is a 10,000-draw bootstrap over personas (seed 0), the p-value a sign-flip permutation test.
  With few personas both are wide; with one persona they are meaningless, and the output says how many personas it used.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from bench.memconflict.dataset import TYPES


def load(directory: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (directory / "judged.jsonl").read_text().splitlines() if line.strip()]


def persona_scores(recs: list[dict[str, Any]], arms: list[str]) -> dict[str, dict[str, float]]:
    """arm -> conflict type -> mean score, over the questions every arm has a judgement for."""
    bad = {(r["session"], r["qid"]) for r in recs if r["arm"] in arms and r["judge"] is None}
    out: dict[str, dict[str, float]] = {}
    for arm in arms:
        by: dict[str, list[float]] = {}
        for r in recs:
            if r["arm"] == arm and (r["session"], r["qid"]) not in bad:
                by.setdefault(r["type"], []).append(r["judge"]["score"])
        out[arm] = {t: sum(v) / len(v) for t, v in by.items()}
        out[arm]["_outdated"] = sum(
            bool(r["judge"] and r["judge"].get("outdated"))
            for r in recs
            if r["arm"] == arm and (r["session"], r["qid"]) not in bad
        )
    return out


def compare(deltas: list[float], rng: random.Random, draws: int = 10000) -> dict[str, Any]:
    mean = sum(deltas) / len(deltas)
    boots = sorted(sum(rng.choice(deltas) for _ in deltas) / len(deltas) for _ in range(draws))
    flips = sum(
        abs(sum(x * rng.choice((1, -1)) for x in deltas) / len(deltas)) >= abs(mean) - 1e-12 for _ in range(draws)
    )
    return {
        "delta": round(100 * mean, 1),
        "ci95": [round(100 * boots[int(draws * 0.025)], 1), round(100 * boots[int(draws * 0.975) - 1], 1)],
        "p": round((flips + 1) / (draws + 1), 4),
        "personas": len(deltas),
        "wins": sum(x > 0 for x in deltas),
        "losses": sum(x < 0 for x in deltas),
    }


def report(run_dirs: dict[str, Path], arms: list[str], base: str = "host") -> dict[str, Any]:
    """run_dirs: persona label -> directory holding judged.jsonl (written by judge_file)."""
    per = {p: persona_scores(load(d), arms) for p, d in run_dirs.items()}
    rng = random.Random(0)
    accuracy = {
        a: {TYPES[t]: round(100 * sum(per[p][a].get(t, 0) for p in per) / len(per), 1) for t in TYPES} for a in arms
    }
    vs: dict[str, Any] = {}
    for arm in (a for a in arms if a != base):
        vs[arm] = {}
        for t, label in TYPES.items():
            d = [per[p][arm][t] - per[p][base][t] for p in per if t in per[p][arm] and t in per[p][base]]
            vs[arm][label] = compare(d, rng) if d else None
    return {
        "personas": sorted(per),
        "base": base,
        "accuracy_pct": accuracy,
        f"vs_{base}": vs,
        "outdated_value_errors": {a: sum(per[p][a]["_outdated"] for p in per) for a in arms},
    }
