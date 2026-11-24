"""The state card: one line per attribute that changes over time, each carrying its recent history.

    "User's residence (as of 2026-03-01): Melbourne. Before: Darwin until 2026-03-01."

The card is kept OUTSIDE the host memory store (a JSON file plus an append-only change log) and is only ever built
from memories the host already holds. Prompts ask for good behaviour; code enforces it, because prompt-only rules do
not hold on small models: fixed facts (`FIXED_ATTR`), a user blocklist, the `(as of DATE)` stamp, the history cap,
and a freshen pass that may only shorten and never strikes a date that is still ahead.
"""

from __future__ import annotations

import calendar
import contextlib
import json
import logging
import os
import re
import secrets
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import date as Date
from datetime import datetime
from pathlib import Path
from typing import Any

from savestate.models import LLM, Memory, check_user_id
from savestate.prompts import CARD_FRESHEN_PROMPT, CARD_MERGE_PROMPT, CARD_PROMPT

_log = logging.getLogger(__name__)

DEFAULT_NAMES: dict[str, str] = {
    "residence": "where the person lives",
    "employer": "who they work for",
    "job title": "their role",
    "employment status": "employed or not",
    "work status": "busy, free, on a business trip...",
    "income": "what they earn",
    "savings": "what they have saved",
    "relationship": "relationship status AND partner in one line",
    "health": "illness, treatment, how they are doing",
}

# Facts that cannot change are not states; a later (possibly false) claim about them must not overwrite the truth.
FIXED_ATTR = re.compile(
    r"birth|born|gender|\bsex\b|nationality|\bname\b|educat|degree|universit|college|major|"
    r"father|mother|parent|sibling|brother|sister|grew up|hometown",
    re.I,
)

MAX_BEFORE = 2  # earlier values a line keeps, newest first (the prompt asks for the same two)
MAX_LINE = 420  # a line that needs more is a paragraph, not a state; the prompt asks for under 300
BATCH = 60  # memories per model call
_HISTORY_SPLIT = re.compile(r"(?<=[.;])\s+(?=before\b)", re.I)
_AS_OF = re.compile(r"\(as of \d{4}-\d{2}-\d{2}\)")
_MONTHS = {
    m: i + 1 for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])
}


def iso_date(value: str | Date | datetime) -> str:
    """ISO date string. Lines are compared as strings, so anything not YYYY-MM-DD is refused up front."""
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, Date):
        return value.isoformat()
    return Date.fromisoformat(str(value)).isoformat()


def dates_in(text: str) -> set[str]:
    """Every full date in a line as ISO strings: 2026-11-26 and 26 Nov 2026."""
    out = set(re.findall(r"\b(20\d\d-\d\d-\d\d)\b", text))
    for d, m, y in re.findall(
        r"\b(\d{1,2})[.]? ?(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[.]? (20\d\d)\b", text, re.I
    ):
        out.add(f"{y}-{_MONTHS[m.lower()]:02d}-{int(d):02d}")
    return out


# Any token that carries a year: 2026-12-05, 2026/12/05, 12/05/2026, 5 Dec 2026, Dec 5, 2026, Dec 2026, 2027.
_YEAR_TOKEN = re.compile(
    rf"\b20\d\d[-/.]\d\d?[-/.]\d\d?\b|\b\d\d?[/.]\d\d?[/.]20\d\d\b"
    rf"|\b\d\d?(?:st|nd|rd|th)?[.]? ?(?:{'|'.join(_MONTHS)})[a-z]*[.]?,? 20\d\d\b"
    rf"|\b(?:{'|'.join(_MONTHS)})[a-z]*[.]? (?:\d\d?(?:st|nd|rd|th)?,? )?20\d\d\b|\b20\d\d\b",
    re.I,
)


def _last_day(token: str) -> Date | None:
    """The last day a date token can mean; None when it cannot be told ('12/05/2026' is 12 May or 5 Dec)."""
    parts = re.findall(r"[a-z]+|\d+", token.lower())
    nums = [int(p) for p in parts if p.isdigit()]
    month = next((_MONTHS[p[:3]] for p in parts if p[:3] in _MONTHS), None)
    try:
        if month:  # "5 Dec 2026", "Dec 5, 2026", "Dec 2026"
            year = nums[-1]
            return Date(year, month, nums[0] if len(nums) == 2 else calendar.monthrange(year, month)[1])
        if len(nums) == 1:  # a bare year
            return Date(nums[0], 12, 31)
        if nums[0] > 31:  # 2026-12-05
            return Date(*nums)
        a, b, year = nums  # 12/05/2026
        if a > 12 or a == b:
            return Date(year, b, a)
        return Date(year, a, b) if b > 12 else None
    except (ValueError, IndexError):
        return None


def removes_future(old: str, new: str, today: str) -> bool:
    """True if `new` drops a year-bearing token of `old` (any date format) that is not strictly before `today`."""
    gone = {t.lower() for t in _YEAR_TOKEN.findall(old)} - {t.lower() for t in _YEAR_TOKEN.findall(new)}
    return any((last := _last_day(t)) is None or last.isoformat() >= today for t in gone)


def write_private(path: Path, text: str) -> None:
    """Replace `path` with `text` atomically, readable by this user only (cards and caches are personal data).

    A fresh, exclusively created temp file in the same directory: a planted symlink or file is never followed.
    """
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def trim_history(text: str, max_before: int) -> str:
    """Keep the current value and only the `max_before` newest 'Before' parts."""
    head, *before = _HISTORY_SPLIT.split(text)
    return " ".join([head, *before[:max_before]])


def read_view(text: str) -> str:
    """A line as the answerer should see it: the current value and the last change; older history is cut."""
    return text.split(" Before that:")[0].rstrip(" .;") + "." if " Before that:" in text else text


def _norm(name: object) -> str:
    return " ".join(str(name).lower().split())[:40]


def _list(value: object) -> list[Any]:
    """Model output: a list or nothing (a wrong shape is ignored, as `_apply_lines` does)."""
    return value if isinstance(value, list) else []


@dataclass
class Line:
    text: str
    as_of: str


class Card:
    """A self-maintaining state card for one person.

    Args:
        llm: any LLM with `complete_json`.
        path: JSON file for the card (the change log goes next to it as `<name>.log.jsonl`); None keeps it in memory.
        subject: how lines name the person: lines read "<subject>'s residence (as of D): ...". Use the name your host's
            memories use for them ("User" if they say "the user"): when memories say "Sam moved" under a card about
            "User", the model treats Sam as someone else and skips the change.
        names: starting attribute names, a list or a {name: meaning} dict. The model may add names; a merge pass
            folds duplicates.
        blocklist: regex (str or compiled); a line whose name or text matches is rejected. Use it for things the
            model files as "life" but are not, such as your own tooling.
        user_id: whose card this is. It is saved in the file, and a file that names someone else raises ValueError
            (nothing is dropped or overwritten). A file without an id (an older card) is adopted on the next save;
            with no `user_id` given, an id already in the file is kept. `Layer` passes its store's id.
    """

    def __init__(
        self,
        llm: LLM,
        path: str | os.PathLike[str] | None = None,
        *,
        subject: str = "User",
        names: Mapping[str, str] | Sequence[str] = DEFAULT_NAMES,
        blocklist: str | re.Pattern[str] | None = None,
        user_id: str | None = None,
    ):
        self.llm, self.subject = llm, subject
        self.user_id = None if user_id is None else check_user_id(user_id)
        self.names = {
            _norm(k): v for k, v in (names.items() if isinstance(names, Mapping) else ((n, "") for n in names))
        }
        self.blocklist = re.compile(blocklist, re.I) if isinstance(blocklist, str) else blocklist
        self.path = Path(path).expanduser() if path else None
        self.log_path = self.path.with_name(self.path.stem + ".log.jsonl") if self.path else None
        self.lines: dict[str, Line] = {}
        if self.path and self.path.exists():
            text = self.path.read_text()
            raw = json.loads(text)
            owner = raw.get("user_id") if isinstance(raw, dict) else None
            if isinstance(owner, str) and owner:
                if self.user_id and owner != self.user_id:  # before anything is loaded, backed up or saved
                    raise ValueError(
                        f"card {self.path} belongs to user_id {owner!r}, not {self.user_id!r}; "
                        "use that person's card path (the file is untouched)"
                    )
                self.user_id = owner
            self.lines = self._load(raw)
            if self._dropped:  # keep what was dropped: the next save would otherwise destroy it for good
                write_private(
                    self.path.with_name(f"{self.path.stem}.dropped-{os.getpid()}-{secrets.token_hex(3)}.json"), text
                )

    def _load(self, raw: object) -> dict[str, Line]:
        """Lines from disk get the same guards as new ones; a tampered or stale line is dropped with a warning."""
        items = raw.get("lines") if isinstance(raw, dict) else None
        if not isinstance(items, dict):
            self._dropped = True
            _log.warning("card %s has no 'lines' object; starting empty", self.path)
            return {}
        out: dict[str, Line] = {}
        self._dropped = False
        for name, v in items.items():
            ok = isinstance(v, dict) and isinstance(v.get("text"), str) and isinstance(v.get("as_of"), str)
            why = "bad shape" if not ok or name != _norm(name) else self._reject(name, v["text"], v["as_of"])
            if why:
                _log.warning(
                    "card %s: dropped line %r (%s); the original file is kept as .dropped-*.json", self.path, name, why
                )
                self._dropped = True
            else:
                out[name] = Line(v["text"], v["as_of"])
        return out

    @contextlib.contextmanager
    def _atomic(self) -> Iterator[None]:
        """Whatever fails inside leaves `self.lines` as it was, so memory never differs from disk and log."""
        before = {k: replace(v) for k, v in self.lines.items()}
        try:
            yield
        except Exception:
            self.lines = before
            raise

    # --- updating -----------------------------------------------------------------------------------------------

    def update(
        self, memories: Iterable[Memory | str | dict[str, Any]], date: str | Date | datetime
    ) -> list[dict[str, Any]]:
        """Fold memories learned on `date` into the card. Returns the change-log entries this call made.

        The card is saved after every batch, so a crash loses at most one batch.
        """
        day = iso_date(date)
        if isinstance(memories, str | dict | Memory):  # one memory, not an iterable of characters
            memories = [memories]
        mems = [m for m in map(Memory.of, memories) if m.text.strip()]
        changes: list[dict[str, Any]] = []
        for i in range(0, len(mems), BATCH):
            chunk = mems[i : i + BATCH]
            handles = {f"m{j}": m for j, m in enumerate(chunk, 1)}
            with self._atomic():
                answer = self.llm.complete_json(
                    CARD_PROMPT.format(
                        subject=self.subject,
                        date=day,
                        names=self._names_block(),
                        card=self._card_text(),
                        new="\n".join(f"[{h}] {' '.join(m.text.split())}" for h, m in handles.items()),
                    )
                )
                if not isinstance(answer.get("lines"), list):  # a refusal or prose: say so instead of looking calm
                    batch = [{"date": day, "error": "the model returned no JSON with a 'lines' list; card unchanged"}]
                else:
                    batch = self._apply_lines(answer["lines"], handles, day)
                if any(c.get("name") and c["before"] is None for c in batch) and len(self.lines) > 1:
                    batch += self._merge(day)
                self._save(batch)
            changes += batch
        return changes

    def _apply_lines(self, lines: object, handles: dict[str, Memory], day: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for raw in _list(lines):
            if not isinstance(raw, dict):
                continue
            name = _norm(raw.get("name") or raw.get("attribute") or "")
            text = trim_history(" ".join(str(raw.get("text", "")).split()), MAX_BEFORE)
            why = self._reject(name, text, day)
            before = self.lines[name].text if name in self.lines else None
            if why or text == before:
                if why:
                    out.append({"date": day, "rejected": name, "reason": why, "text": text})
                continue
            self.lines[name] = Line(text, day)
            hs = raw["ids"] if isinstance(raw.get("ids"), list) else []  # model output: never trust its shape
            ids = [handles[h].id for h in hs if isinstance(h, str) and h in handles and handles[h].id]
            out.append({"date": day, "name": name, "before": before, "after": text, "ids": ids})
        return out

    def _reject(self, name: str, text: str, day: str) -> str | None:
        if not name:
            return "no name"
        head = text.removeprefix(f"{self.subject}'s").split("(as of", 1)[0]  # "<subject>'s <attribute> (as of ..."
        if FIXED_ATTR.search(name) or FIXED_ATTR.search(head):
            return "fixed fact"
        if self.blocklist and (self.blocklist.search(name) or self.blocklist.search(text)):
            return "blocklist"
        if f"(as of {day})" not in text:
            return "missing (as of DATE) stamp"
        if not text.startswith(f"{self.subject}'s"):
            return f"does not start with {self.subject}'s"
        if len(text) > MAX_LINE:
            return "too long"
        return None

    def _merge(self, day: str) -> list[dict[str, Any]]:
        """Fold lines that name the same attribute differently. Runs only when a new name has just appeared."""
        out: list[dict[str, Any]] = []
        answer = self.llm.complete_json(CARD_MERGE_PROMPT.format(subject=self.subject, card=self._card_text()))
        for m in _list(answer.get("merges")):
            if not isinstance(m, dict):
                continue
            keep = _norm(m.get("keep", ""))
            drop = [d for d in map(_norm, _list(m.get("drop"))) if d in self.lines and d != keep]
            text = trim_history(" ".join(str(m.get("text", "")).split()), MAX_BEFORE)
            if keep not in self.lines or not drop:
                continue
            newest = max(self.lines[n].as_of for n in [keep, *drop])  # the merged line is as fresh as its freshest part
            was = " ".join(self.lines[n].text for n in [keep, *drop])
            if not self._reject(keep, text, newest) and not removes_future(was, text, day):
                self.lines[keep] = Line(text, newest)
                for d in drop:
                    del self.lines[d]
                out.append({"date": day, "merge": keep, "drop": drop, "after": text})
        return out

    # --- freshening ---------------------------------------------------------------------------------------------

    def freshen(self, today: str | Date | datetime) -> list[dict[str, Any]]:
        """Strike what has expired. The model may only shorten or drop lines, and code refuses to remove any date
        that is `today` or later (unguarded, Haiku struck a future event as "passed")."""
        day = iso_date(today)
        if not self.lines:
            return []
        with self._atomic():
            answer = self.llm.complete_json(
                CARD_FRESHEN_PROMPT.format(subject=self.subject, today=day, card=self._card_text())
            )
            out: list[dict[str, Any]] = []
            for raw in _list(answer.get("lines")):
                if not isinstance(raw, dict):
                    continue
                name, text = _norm(raw.get("name", "")), " ".join(str(raw.get("text", "")).split())
                old = self.lines[name].text if name in self.lines else ""
                stamp = _AS_OF.search(old)
                if (
                    stamp
                    and stamp.group(0) in text
                    and len(text) < len(old)
                    and not self._reject(name, text, self.lines[name].as_of)
                    and not removes_future(old, text, day)
                ):
                    self.lines[name].text = text
                    out.append({"date": day, "freshen": name, "before": old, "after": text})
            for name in map(_norm, _list(answer.get("drop"))):
                if name in self.lines and not removes_future(self.lines[name].text, "", day):
                    out.append({"date": day, "freshen": name, "before": self.lines.pop(name).text, "after": None})
            self._save(out)
        return out

    # --- reading ------------------------------------------------------------------------------------------------

    def render(self, full: bool = False) -> str:
        """The block to pin at the top of a prompt: current value plus the last change per line (`full` keeps all
        the history the card holds)."""
        body = [f"- {ln.text if full else read_view(ln.text)}" for ln in self.lines.values()] or ["- (empty so far)"]
        return "STATE CARD:\n" + "\n".join(body)

    # --- plumbing -----------------------------------------------------------------------------------------------

    def _names_block(self) -> str:
        return "\n".join(f"- {n}" + (f": {meaning}" if meaning else "") for n, meaning in self.names.items())

    def _card_text(self) -> str:
        return "\n".join(f"[{k}] {v.text}" for k, v in self.lines.items()) or "(empty)"

    def _save(self, log: list[dict[str, Any]]) -> None:
        # shortcut: one writer per card file (a second process, such as a cron `card freshen`, overwrites this one's
        # saves with its older copy); upgrade to a lock or a reload-before-save if a second writer is needed.
        if not self.path or not self.log_path:
            return
        write_private(
            self.path,
            json.dumps(
                {
                    **({"user_id": self.user_id} if self.user_id else {}),
                    "lines": {k: asdict(v) for k, v in self.lines.items()},
                },
                indent=1,
                ensure_ascii=False,
            ),
        )
        if log:
            fd = os.open(self.log_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, "a") as f:
                os.fchmod(fd, 0o600)  # also for a log an older version made with the umask mode
                f.writelines(json.dumps(e, ensure_ascii=False) + "\n" for e in log)
