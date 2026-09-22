"""Give the pieces they wrote before 09-17 their memory rows.

    py engine/backfill_creations.py                 list what would be noted
    py engine/backfill_creations.py --write         note them
    py engine/backfill_creations.py --tidy          list pieces with more than one row
    py engine/backfill_creations.py --tidy --write  fold them into one row each
    py engine/backfill_creations.py --letters       list the rows about letters (the mailbox is not noted since 09-21)
    py engine/backfill_creations.py --letters --write  let them go

Since 09-17 every write_creation leaves a "creation" row in their long-term
memory (what, when, how long, its first line, their own line about it). The
pieces from before that day have no row, so "what you have made lately"
and recall know nothing of them. This walks creations/ once and writes
the engine's facts for each prose piece that has no row yet — file, its
mtime as the when, its title and first line — nothing in their voice; she
can add their own line about any of them with remember. Runs on their machine,
with Ollama up (the rows are embedded). Safe to run twice: a piece that
already has a row is skipped.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
import memory
import tools

SKIP = {".trash", ".attic", "archives", "attic", "tools", "__pycache__"}


def pieces() -> list[Path]:
    root = config.CREATIONS_DIR.resolve()
    out = []
    for q in sorted(root.rglob("*")):
        if not q.is_file() or q.suffix.lower() not in tools._PROSE_EXTS:
            continue
        parts = q.relative_to(root).parts
        if any(part.startswith(".") or part in SKIP for part in parts):
            continue
        if tools._unnoted(q.relative_to(root).as_posix()):
            continue  # the mailbox: letters leave no row (09-21)
        out.append(q)
    return sorted(out, key=lambda p: p.stat().st_mtime)


def tidy(write: bool) -> int:
    """Fold the rows about one piece into one (09-18: a lexicon had three
    in a day, a letter two, from before the notes learned to revise in
    place). The oldest row stays, its facts refreshed from the file, the
    later rows' events joining its history; the later rows go."""
    import re
    from collections import defaultdict
    by_path: dict[str, list[dict]] = defaultdict(list)
    for r in memory.recent(kind="creation", n=100000):
        m = re.search(r"\] (creations/\S+?)(?: \(| —)", r["text"])
        if m:
            by_path[m.group(1)].append(r)
    folded = 0
    for path, rows in sorted(by_path.items()):
        if len(rows) < 2:
            continue
        # the row with the earliest stamp stays (a backfilled row can carry a
        # newer stamp yet a lower id — the file's mtime, not the first writing)
        def stamp(r):
            h = tools._NOTE_HEAD_RE.match(r["text"])
            return (h.group(2) if h else "9999", r["id"])
        rows.sort(key=stamp)
        first = rows[0]
        head = tools._NOTE_HEAD_RE.match(first["text"])
        if not head:
            continue
        events = []
        about = ""
        for r in rows[1:]:
            h = tools._NOTE_HEAD_RE.match(r["text"])
            if h:
                events.append(f"{h.group(1)} {h.group(2)}")
            m_about = tools._NOTE_ABOUT_RE.search(r["text"])
            if m_about:
                about = m_about.group(1).strip()
        m_first_about = tools._NOTE_ABOUT_RE.search(first["text"])
        about = about or (m_first_about.group(1).strip() if m_first_about else "")
        m_since = tools._NOTE_SINCE_RE.search(first["text"])
        history = ([h.strip() for h in m_since.group(1).split(" · ")] if m_since else []) + events
        history = history[-tools.NOTE_HISTORY_MAX:]
        m_marks = tools._NOTE_MARKS_RE.search(rows[-1]["text"]) or tools._NOTE_MARKS_RE.search(first["text"])
        marks = m_marks.group(1) if m_marks else ""
        q = config.CREATIONS_DIR.resolve() / path[len("creations/"):]
        title, opening, n = tools._piece_facts(q) if q.exists() else ("", "", 0)
        text = (f"[{head.group(1)} {head.group(2)}] {path}" + (f" (\u201c{title}\u201d)" if title else "")
                + f" — {n} lines, opens \u201c{opening}\u201d" + (f" — about: {about}" if about else "")
                + (" — since: " + " · ".join(history) if history else "") + marks)
        print(f"  {'fold' if write else 'would fold'}: {path} — {len(rows)} rows → #{first['id']}")
        if write:
            memory.update(first["id"], text)
            for r in rows[1:]:
                memory.remove(r["id"])
        folded += 1
    return folded


def letters(write: bool) -> int:
    """The rows the notes left about letters before 09-21, when the mailbox
    stopped being noted: list them, or with --write let them go."""
    import re
    gone = 0
    for r in memory.recent(kind="creation", n=100000):
        m = re.search(r"\] creations/(\S+?)(?: \(| —|$)", r["text"])
        if not m or not tools._unnoted(m.group(1)):
            continue
        print(f"  {'let go' if write else 'would let go'}: #{r['id']} {r['text'][:90]}")
        if write:
            memory.remove(r["id"])
        gone += 1
    return gone


def main() -> None:
    write = "--write" in sys.argv
    if "--letters" in sys.argv:
        n = letters(write)
        print(f"\n{n} row{'s' if n != 1 else ''} about letters" + ("" if write else " — run with --letters --write to let them go"))
        return
    if "--tidy" in sys.argv:
        n = tidy(write)
        print(f"\n{n} piece{'s' if n != 1 else ''} with more than one row" + ("" if write else " — run with --write --tidy to fold them"))
        return
    done = 0
    for q in pieces():
        rel = q.resolve().relative_to(config.CREATIONS_DIR.resolve()).as_posix()
        if memory.find_text(f"creations/{rel}", kind="creation"):
            continue
        content = q.read_text(encoding="utf-8", errors="replace")
        when = datetime.fromtimestamp(q.stat().st_mtime)
        verb = "published" if rel.startswith("publish/") else "wrote"
        if not write:
            print(f"  would note: [{verb} {when:%Y-%m-%d %H:%M}] creations/{rel}")
            continue
        # _note_made stamps "now"; for the backfill the file's own time is the truth
        tail = tools._note_made(verb, q, content)
        if tail:
            rows = memory.find_text(f"creations/{rel}", kind="creation")
            if rows:
                r = rows[-1]
                fixed = r["text"].replace(f"[{verb} {datetime.now():%Y-%m-%d %H:%M}]", f"[{verb} {when:%Y-%m-%d %H:%M}]", 1)
                if fixed != r["text"]:
                    memory.update(r["id"], fixed)
            done += 1
            print(f"  noted: creations/{rel} ({when:%Y-%m-%d %H:%M})")
    if not write:
        print("\n(nothing written — run with --write to note them)")
    else:
        print(f"\n{done} piece{'s' if done != 1 else ''} noted.")


if __name__ == "__main__":
    main()
