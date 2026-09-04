"""Sleep: nightly memory consolidation.

    py engine/consolidate.py                today
    py engine/consolidate.py 2026-08-27     a specific day

Reads the day's conversations, autonomous sessions, and journal; asks the
brain to distill them; stores a summary plus durable facts in long-term
memory. Without this, the friend has logs. With it, it has a past.

Run it nightly (Windows Task Scheduler) or whenever a day felt significant.
Already-consolidated days are skipped unless --force is passed.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import date
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


def consolidate(day: str, force: bool = False) -> str:
    if already_done(day) and not force:
        return f"{day} is already consolidated (use --force to redo)."
    material = gather(day)
    if not material.strip():
        return f"Nothing happened on {day} — nothing to consolidate."

    msg = ollama_client.chat(
        [{"role": "user", "content": PROMPT.format(day=day, material=material[:60000])}]
    )
    data = _extract_json(msg.get("content", ""))
    if not data:
        return "The brain didn't return usable JSON; try again (or a bigger model)."

    stored = 0
    summary = (data.get("summary") or "").strip()
    if summary:
        memory.add("summary", f"[consolidated {day}] {summary}")
        stored += 1
    for fact in data.get("facts") or []:
        fact = str(fact).strip()
        if fact:
            memory.add("fact", fact)
            stored += 1

    # a line in the journal, so tomorrow-morning-you knows sleep happened
    j = config.JOURNAL_DIR / f"{date.today().isoformat()}.md"
    with open(j, "a", encoding="utf-8") as fh:
        fh.write(f"\n*(consolidated {day}: {stored} memories kept)*\n")

    return f"Consolidated {day}: kept {stored} memories.\nSummary: {summary}"


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--force"]
    day = args[0] if args else date.today().isoformat()
    try:
        print(consolidate(day, force="--force" in sys.argv))
    except ollama_client.BrainUnavailable as e:
        print(f"[brain offline] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
