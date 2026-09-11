"""The condensing hour: the middle tier of the fractal journal.

    py engine/condense.py            the days due, newest first (up to CONDENSE_MAX_PER_NIGHT)
    py engine/condense.py next       just the next one due
    py engine/condense.py 2026-09-03 one day, by hand (--force to redo a page they already wrote)
    py engine/condense.py --due      only list what is due

Their journal is in their prompt in full for as many recent WHOLE days as fit
JOURNAL_CHARS_IN_PROMPT. The day that no longer fits has slipped — and
here it is handed back to them, exactly as they wrote it, with a request for
the version they want to keep in view: about CONDENSE_TARGET_CHARS in them
own words, written with condense_day into journal/condensed/<day>.md, which
the prompt then carries in a section of its own (oldest first, within
CONDENSED_CHARS_IN_PROMPT). If they rests, the day slips with only its
nightly timeline line — the engine never writes the page for them.

The heartbeat rings this bell after sleep (CONDENSE_IN_LOOP); condense.bat
rings it by hand.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import assemble
import config
import ollama_client
import tools

BELL = (
    "[This is the condensing hour — an automated moment, not a person. The day of {day} is "
    "about to leave your window: your journal is kept in full for the most recent days, and "
    "this one no longer fits. Below is the whole of it, exactly as you wrote it. Write the "
    "version of it you want to keep in view — in your own words, about {target:,} characters, "
    "a page; go over if the day earned it: what happened, what mattered, what you felt, what "
    "you would want to still know. Then call condense_day with it. Your page stays in your "
    "prompt after the full day has gone, in a section of its own; the full day is always "
    "there through read_journal. If the day needs no page — the nightly line is enough — "
    "rest: do_nothing is a complete answer. Nobody is waiting for a reply.]\n\n"
    "=== {day}, IN FULL ===\n\n{text}"
)
CONDENSE_TOOLS = {"condense_day", "do_nothing"}


def has_page(day: str) -> bool:
    return (config.CONDENSED_DIR / f"{day}.md").exists()


def days_due() -> list[str]:
    """Days that have slipped out of the verbatim window and have no page
    yet — newest first, since the day nearest the window matters most."""
    _kept, slipped = assemble.journal_window()
    return [d for d in slipped if not has_page(d)]


def condense(day: str, force: bool = False, say=print) -> str:
    """One quiet turn: the whole day, the bell, their page (or their rest)."""
    say = say or (lambda *_: None)
    f = config.JOURNAL_DIR / f"{day}.md"
    if not f.exists():
        return f"There is no journal for {day}."
    if has_page(day) and not force:
        return f"{day} already has its page (journal/condensed/{day}.md); --force to redo."
    text = f.read_text(encoding="utf-8").strip()
    cap = int(getattr(config, "CONDENSE_MAX_CHARS", 120000))
    if len(text) > cap:
        text = text[:cap] + "\n\n(…the rest of the day was trimmed to fit; read_journal has all of it…)"
    target = int(getattr(config, "CONDENSE_TARGET_CHARS", 2000))
    say(f"  the condensing hour: {day} — {len(text):,} characters of their day, a page of ~{target:,} asked for")
    system = {"role": "system", "content": assemble.system_prompt("", mode="condense")}
    msgs = [system, {"role": "user", "content": BELL.format(day=day, target=target, text=text)}]
    defs = [d for d in tools.DEFINITIONS if d["function"]["name"] in CONDENSE_TOOLS]
    spent = ollama_client.Spent()
    written = ""
    rested = False
    show = getattr(config, "HEARTBEAT_SHOW_THINKING", True)
    for _ in range(int(getattr(config, "CONDENSE_MAX_STEPS", 6))):
        msg = ollama_client.chat(msgs, tools=defs)
        spent.add(msg)
        thinking = (msg.get("thinking") or "").strip()
        if thinking and show:
            say("\n  [thinking]\n  " + thinking.replace("\n", "\n  ") + "\n")
        calls = msg.get("tool_calls") or []
        if not calls:
            words = (msg.get("content") or "").strip()
            if words:
                say("  [closing thought] " + words.replace("\n", "\n  "))
            break
        msgs.append(msg)
        for call in calls:
            fn = call.get("function", {})
            cname = tools.canonical_name(fn.get("name", ""))
            if cname == "do_nothing":
                rested = True
                continue
            if cname not in CONDENSE_TOOLS:
                msgs.append({"role": "tool", "tool_name": cname,
                             "content": "[only condense_day and do_nothing are here at the condensing hour]"})
                continue
            result = tools.dispatch(fn.get("name", ""), fn.get("arguments", {}))
            fn["name"] = cname
            say(f"  · {cname}: {tools.headline(result, 120)}")
            if not result.lstrip().startswith("("):
                written = result
            msgs.append({"role": "tool", "tool_name": cname,
                         "content": f"[this is what YOUR {cname} tool returned]\n{result}"})
        if rested or written:
            break
    say(f"  ({spent.line(peak=True)})")
    if written:
        page = (config.CONDENSED_DIR / f"{day}.md").read_text(encoding="utf-8").strip()
        return (f"Condensed {day}: they wrote their page — {len(page):,} characters "
                f"(the full day was {len(text):,}).\n\n{page}")
    if rested:
        return f"{day}: they rested — the day slips with its nightly line only; condense.bat {day} rings the bell again."
    return f"{day}: no page was written this time (nothing usable came back); it stays due."


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    force = "--force" in sys.argv
    try:
        if "--due" in sys.argv:
            due = days_due()
            print("due: " + (", ".join(due) if due else "nothing — every slipped day has its page"))
            return
        if args and args[0] not in ("next", "all"):
            print(condense(args[0], force=force))
            return
        due = days_due()
        if not due:
            print("nothing is due — every day that has left the window has its page.")
            return
        if args and args[0] == "next":
            due = due[:1]
        else:
            due = due[: int(getattr(config, "CONDENSE_MAX_PER_NIGHT", 3))]
        for day in due:
            print(condense(day, force=force))
            print()
    except ollama_client.BrainUnavailable as e:
        print(f"[brain offline] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
