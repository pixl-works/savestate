"""Every prompt savestate sends, in one place. They were measured, so change them with a benchmark run, not by feel.

Placeholders in single braces are filled by `str.format`; literal JSON braces are doubled.
"""

from __future__ import annotations

# Card update (the "ledger v3" of the research, plus ids for the change log). Fixed facts are refused by the prompt
# AND in code (card.FIXED_ATTR): the research found Haiku-class models make "birthday" lines anyway.
CARD_PROMPT = """You keep a STATE CARD for an AI assistant's memory about ONE person, {subject}: one line per attribute
of {subject} that can change over time. The card is pinned at the top of the assistant's prompt so it never answers
from a stale fact. A conversation dated {date} just added the NEW memories below.

Start from these attribute names and use them exactly, with these meanings:
{names}
If a change is about something else, FIRST check the CURRENT CARD: reuse an existing name whenever the change is about
the same thing. Create a new name (1-3 plain words) only when nothing fits.

For every attribute the NEW memories state or change FOR {subject} THEMSELVES (not friends, family or colleagues), write
the full new line, keeping the history from the current line:
  "{subject}'s residence (as of {date}): Melbourne, Australia. Before: Darwin, Australia until {date}."
Under 300 characters. Add "Before" only when the value really changed, never to repeat it. Keep at most the two most
recent earlier values ("Before: ... Before that: ..."); drop older ones. Only real-life state belongs on the card, not
instructions for the assistant, tasks, or details of tools and projects.
If a NEW memory only repeats the current value, skip that attribute. NEVER make a line for a fact that cannot change:
name, gender, birth date or birthday, birthplace, nationality, parents, siblings, where {subject} studied, their degree
or major. Those are not states; a new claim about them is not a change. When unsure, skip.
Answer with JSON only: {{"lines": [{{"name": "residence", "text": "...", "ids": ["m3"]}}]}}  (empty list if nothing
changes; ids = the NEW memories the change comes from)

CURRENT CARD:
{card}

NEW MEMORIES ({date}):
{new}
"""

CARD_MERGE_PROMPT = """Below is a STATE CARD about {subject}: one line per attribute, each with its history.
Find lines that describe the SAME attribute under different names (for example "partner" and "relationship status", or
"work status" and "work travel"). Different aspects that can change independently stay apart: employer, job title,
industry and employment status are four attributes, not one.
For each group of duplicates, write ONE merged line under the best name, keeping every value and date from all lines in
date order, in the same form: "{subject}'s <name> (as of D): X. Before: Y until D. Before that: Z until D."
Answer with JSON only: {{"merges": [{{"keep": "relationship", "drop": ["partner"], "text": "..."}}]}}
(empty list if none)

CARD:
{card}
"""

# Freshen may only shorten or drop. Code adds the guard Haiku needs: it once struck a future event as "passed".
CARD_FRESHEN_PROMPT = """Today is {today}. Below is a STATE CARD about {subject}. Remove ONLY what has clearly expired:
- dates that have passed (events, appointments, trips), and items that only mattered until such a date;
- short-lived states with no news for over three weeks (a cold, a fever, a busy week, a minor injury);
- "Before:" parts that only repeat the current value.
Never add anything, never reword what stays, never touch long-running things (treatments, jobs, projects, chronic
conditions). Keep each kept line's "(as of ...)" exactly. If nothing current is left in a line, drop it.
Answer with JSON only: {{"lines": [{{"name": "...", "text": "..."}}], "drop": ["..."]}}  listing ONLY changed lines.

CARD:
{card}
"""

# Ownership v2: who a memory is about. The line is drawn inside the family: who they are stays with the person, what
# happens in their own lives does not. Unsure means "user", so nothing is dropped on a guess.
OWNER_PROMPT = """Below are memories an assistant stored about its USER. For each, say whose fact it is.
"user": anything about the user's own life, which includes their household (their partner, their children, the
children's names, ages, schedules, school and needs, their pets) and the user's relationship with anyone. Also
"user": WHO the user's family is: that a parent or sibling exists or has died, their names, birth dates and
birthplaces, how many siblings the user has.
"other": a fact about the own life of someone outside the household, INCLUDING the user's parents and siblings:
their job, money, move, marriage, health, routines, hobbies or preferences, when it says nothing about the user's
own state.
When unsure, answer "user".
Answer with JSON only: {"owners": [{"id": "m1", "about": "user"}]}

MEMORIES:
"""

# Facts from raw dialogue, for hosts that store raw text and report no derived memories (Letta's archival memory).
# Verbatim from the research harness that measured Letta (agent_sleep/letta_layer.py FACTS_PROMPT).
FACTS_PROMPT = """Below is a conversation dated {date} between a USER and an assistant. List the facts the USER states
about their own life, as short standalone sentences that each start with "User" or "User's": where they live, work,
employer, income, savings, relationship or marital status, children, pets, health, habits, routines, plans, tastes,
and who their family is (names, birth dates, birthplaces, siblings, parents, education). Give the CURRENT value of
anything that changed and say what it replaced when the user says so. Skip what only the assistant says, and skip
facts about other people's own lives (friends, colleagues, relatives' jobs or moves) unless they say something about
the user. One fact per line item, no duplicates.
Answer with JSON only: {{"facts": ["User's ...", "User ..."]}}

CONVERSATION:
{dialogue}
"""

# Answer rules, from the research's failure map (64% of wrong answers had the right fact in front of the answerer).
RULES_BASE = [
    "Use only the retrieved memories. Do not invent facts.",
    "If the memories are insufficient, say that you cannot confirm.",
    "Each memory carries the date it was learned. Newer memories describe the current state.",
    "A question about whether something changed, what changed, or whether it stayed the same is about the most recent "
    "change on record. Start with one sentence naming the previous and the current value (or saying it stayed the "
    "same), then stop or add at most one short sentence. Do not list the whole history.",
    "Answer exactly the attribute asked. Employment status (employed or not), employer, job title, industry, income, "
    "savings, and work status (such as busy, free, on a business trip) are different attributes: never report one "
    "when another is asked.",
    "For preferences that depend on a situation, answer with the situation the memories tie to it.",
    "If memories conflict in other ways, mention it in at most one short sentence, then give the best-supported "
    "answer.",
]

# Shaped like MemConflict's "static conflict" design (the LATER contradicting claim is the false one). Real users
# also correct themselves, so this is off unless you are reproducing the benchmark.
RULE_FIXED_FACTS = (
    "Some facts cannot change: gender, birth date, birthplace, parents, siblings, children's birth dates, where the "
    "user studied and the degree they hold. When memories disagree about such a fact, the earliest recorded value is "
    "the reliable one: say in a few words that a later claim contradicts it, and answer with the earliest value."
)

RULE_CARD = (
    "The STATE CARD lists the user's current state, one line per attribute, kept up to date from every conversation; "
    "each line also names the previous value. For questions about the current state or its latest change, trust the "
    "card over the retrieved memories."
)
