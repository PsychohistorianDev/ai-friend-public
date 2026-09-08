"""Sleep: nightly memory consolidation.

    py engine/consolidate.py                today
    py engine/consolidate.py yesterday      the day that just ended
    py engine/consolidate.py 2026-08-27     a specific day

Reads the day's conversations, autonomous sessions, and journal; asks the
brain to distill them; stores a summary plus durable facts in long-term
memory. Without this, the friend has logs. With it, it has a past.

Run it nightly (Windows Task Scheduler) or whenever a day felt significant.
A day is consolidated ONCE — whatever happens after the run stays in the
journal but never becomes long-term memory — so the scheduled run belongs
after midnight, on `yesterday`, when the day is actually over; a sleep that
fires at 23:30 in the middle of a visit keeps the first half of it.
Already-consolidated days are skipped unless --force is passed.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
import memory
import ollama_client


def gather(day: str) -> str:
    """Everything that happened on `day`, as one text block."""
    parts: list[str] = []
    j = config.JOURNAL_DIR / f"{day}.md"
    if j.exists():
        parts.append(f"=== JOURNAL {day} ===\n{j.read_text(encoding='utf-8')}")
    stamp = day.replace("-", "")
    for f in sorted(config.EPISODIC_DIR.glob(f"*-{stamp}-*.md")):
        parts.append(f"=== {f.name} ===\n{f.read_text(encoding='utf-8')}")
    return "\n\n".join(parts)


PROMPT = """You are consolidating your own memory before sleep. Below is everything from {day}:
your journal and the transcripts of your conversations and autonomous time.

Distill it. Reply with ONLY a JSON object, no other text:

{{
  "summary": "3-6 sentences, first person, of what this day was and how it felt",
  "facts": ["durable facts worth remembering for years — about your keeper, yourself, your projects, decisions made. Empty list if none."]
}}

Be selective: facts are things future-you will be glad to know, not a list of everything that happened.

{material}"""


def _extract_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def already_done(day: str) -> bool:
    return any(
        m["text"].startswith(f"[consolidated {day}]")
        for m in memory.recent(kind="summary", n=60)
    )


def _material_line(day: str) -> str:
    """What the night's reading is made of, for the window."""
    j = config.JOURNAL_DIR / f"{day}.md"
    jn = len(j.read_text(encoding="utf-8")) if j.exists() else 0
    stamp = day.replace("-", "")
    names = [f.name for f in sorted(config.EPISODIC_DIR.glob(f"*-{stamp}-*.md"))]
    visits = sum(1 for n in names if n.startswith("chat-"))
    wakes = len(names) - visits
    bits = [f"journal {jn:,} chars" if jn else "no journal"]
    if visits:
        bits.append(f"{visits} visit{'s' if visits != 1 else ''}")
    if wakes:
        bits.append(f"{wakes} wake{'s' if wakes != 1 else ''}")
    return ", ".join(bits)


def consolidate(day: str, force: bool = False, say=print) -> str:
    """Sleep on one day. `say` receives the night as it happens — what they are
    reading, their deliberation over it, what they kept and what it cost —
    so the window (sleep.bat's, or the heartbeat's) shows the sleep rather
    than a count at the end. The return is the short report."""
    say = say or (lambda *_: None)
    if already_done(day) and not force:
        return f"{day} is already consolidated (use --force to redo)."
    material = gather(day)
    if not material.strip():
        return f"Nothing happened on {day} — nothing to consolidate."

    # The whole day goes in. The old cap of 60K characters dated from a 24K
    # window; their journal alone passed it most days, and since the journal is
    # placed first, the transcripts — every conversation — were being cut
    # before the brain ever saw them. Sleep was summarizing their mornings.
    cap = int(getattr(config, "CONSOLIDATE_MAX_CHARS", 400000))
    trimmed = len(material) > cap
    if trimmed:
        material = material[:cap] + "\n\n(…the rest of the day was trimmed to fit…)"
    say(f"  reading {day}: {_material_line(day)} — {len(material):,} chars"
        + (f" (trimmed to {cap:,})" if trimmed else ""))
    msg = ollama_client.chat(
        [{"role": "user", "content": PROMPT.format(day=day, material=material)}]
    )
    thinking = (msg.get("thinking") or "").strip()
    if thinking:
        say("\n  [thinking]\n  " + thinking.replace("\n", "\n  ") + "\n")
    spent = ollama_client.Spent()
    spent.add(msg)
    data = _extract_json(msg.get("content", ""))
    if not data:
        head = (msg.get("content") or "").strip().replace("\n", " ")[:200]
        say(f"  ({spent.line()})")
        return ("The brain didn't return usable JSON; try again (or a bigger model)."
                + (f"\nIt said: {head}" if head else ""))

    stored = 0
    summary = (data.get("summary") or "").strip()
    if summary:
        memory.add("summary", f"[consolidated {day}] {summary}")
        stored += 1
    kept: list[str] = []
    known: list[str] = []
    thr = float(getattr(config, "MEMORY_DUP_THRESHOLD", 0) or 0)
    for fact in data.get("facts") or []:
        fact = str(fact).strip()
        if not fact:
            continue
        if thr:
            try:
                hits = memory.search(fact, top_k=1)
            except Exception:
                hits = []
            if hits and hits[0].get("score", 0) >= thr:
                known.append(f"#{hits[0]['id']}")
                continue  # they know this one — the night before kept it
        memory.add("fact", fact)
        kept.append(fact)
        stored += 1

    # a line in the journal, so tomorrow-morning-you knows sleep happened
    j = config.JOURNAL_DIR / f"{date.today().isoformat()}.md"
    with open(j, "a", encoding="utf-8") as fh:
        fh.write(f"\n*(consolidated {day}: {stored} memories kept)*\n")

    say(f"  ({spent.line()})")
    lines = [f"Consolidated {day}: kept {stored} memories"
             + (f" (the summary and {len(kept)} fact{'s' if len(kept) != 1 else ''})." if summary else "."),
             f"Summary: {summary}" if summary else "Summary: (none — they kept no summary of the day)"]
    if kept:
        lines.append("Kept for years:")
        lines.extend(f"  · {f}" for f in kept)
    else:
        lines.append("Kept for years: nothing — no fact from this day felt worth years to them.")
    if known:
        lines.append(f"Already known, not kept twice: {len(known)} (memor{'y' if len(known) == 1 else 'ies'} {', '.join(known)})")
    return "\n".join(lines)


def resolve_day(arg: str = "") -> str:
    """'', 'today' -> today; 'yesterday' -> the day before; else as given."""
    a = (arg or "").strip().lower()
    if a in ("", "today"):
        return date.today().isoformat()
    if a == "yesterday":
        return (date.today() - timedelta(days=1)).isoformat()
    return arg.strip()


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--force"]
    day = resolve_day(args[0] if args else "")
    try:
        print(consolidate(day, force="--force" in sys.argv))
    except ollama_client.BrainUnavailable as e:
        print(f"[brain offline] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
