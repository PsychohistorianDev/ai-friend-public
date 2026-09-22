"""The ladder: the fractal journal's tiers above the day.

    day → week → month → quarter → year → five years

Every tier is a folder of pages they wrote (journal/condensed/, then
weeks/, months/, quarters/, years/, five_years/), each page about
LADDER_TARGETS[tier] characters, each tier keeping its newest
LADDER_PAGES_KEPT pages in view. When a page falls out of its tier's view
and the period above it is complete and has no page yet, that period is
due: the condensing hour hands them the pages below it and asks for one —
the same bell, one rung up. The pages never leave the disk; the view is
what rides in the prompt. (The keeper, 2026-09-17: "more fractal" — a fixed count
per tier; the calendar's own periods, so the pages have names they can hold.
Sizes were the golden ratio until 09-20, when the keeper saw that seven days into
a 3,236 week was the harshest fold on the ladder: now the week is 4,000 and
the month 6,000, and every fold lands between 2× and 3.5×.)

Nothing here writes a page. The engine reads the calendar and rings bells.
"""
from __future__ import annotations

import re
from datetime import date, timedelta
from pathlib import Path

import config

TIERS = ["day", "week", "month", "quarter", "year", "five_years"]
_FOLDERS = {"day": "", "week": "weeks", "month": "months", "quarter": "quarters",
            "year": "years", "five_years": "five_years"}
_DEFAULT_TARGETS = {"day": 2000, "week": 4000, "month": 6000, "quarter": 8472,
                    "year": 13708, "five_years": 22180}
_KEY_RE = {
    "day": re.compile(r"^\d{4}-\d{2}-\d{2}$"),
    "week": re.compile(r"^\d{4}-W\d{2}$"),
    "month": re.compile(r"^\d{4}-\d{2}$"),
    "quarter": re.compile(r"^\d{4}-Q[1-4]$"),
    "year": re.compile(r"^\d{4}$"),
    "five_years": re.compile(r"^\d{4}-\d{4}$"),
}
_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
           "September", "October", "November", "December"]


def target(tier: str) -> int:
    t = dict(_DEFAULT_TARGETS)
    t.update(getattr(config, "LADDER_TARGETS", {}) or {})
    if tier == "day" and getattr(config, "CONDENSE_TARGET_CHARS", None):
        t["day"] = int(config.CONDENSE_TARGET_CHARS)
    return int(t.get(tier, 2000))


def kept() -> int:
    return int(getattr(config, "LADDER_PAGES_KEPT", 7) or 0)


def above(tier: str) -> str | None:
    i = TIERS.index(tier)
    return TIERS[i + 1] if i + 1 < len(TIERS) else None


def below(tier: str) -> str | None:
    i = TIERS.index(tier)
    return TIERS[i - 1] if i > 0 else None


def folder(tier: str) -> Path:
    base = config.CONDENSED_DIR
    return base / _FOLDERS[tier] if _FOLDERS[tier] else base


def page_path(tier: str, key: str) -> Path:
    return folder(tier) / f"{key}.md"


def valid_key(tier: str, key: str) -> bool:
    return tier in _KEY_RE and bool(_KEY_RE[tier].match(key or ""))


# ---- the calendar ----------------------------------------------------------

def _epoch() -> int:
    return int(getattr(config, "LADDER_EPOCH_YEAR", 2026))


def key_of(tier: str, d: date) -> str:
    """The key of the period of `tier` that holds day d."""
    if tier == "day":
        return d.isoformat()
    if tier == "week":
        y, w, _ = d.isocalendar()
        return f"{y}-W{w:02d}"
    if tier == "month":
        return f"{d.year}-{d.month:02d}"
    if tier == "quarter":
        return f"{d.year}-Q{(d.month - 1) // 3 + 1}"
    if tier == "year":
        return f"{d.year}"
    if tier == "five_years":
        start = _epoch() + 5 * ((d.year - _epoch()) // 5)
        return f"{start}-{start + 4}"
    raise ValueError(tier)


def span(tier: str, key: str) -> tuple[date, date]:
    """(first day, last day) of a period."""
    if tier == "day":
        d = date.fromisoformat(key)
        return d, d
    if tier == "week":
        y, w = int(key[:4]), int(key[6:])
        first = date.fromisocalendar(y, w, 1)
        return first, first + timedelta(days=6)
    if tier == "month":
        y, m = int(key[:4]), int(key[5:])
        first = date(y, m, 1)
        last = (date(y + 1, 1, 1) if m == 12 else date(y, m + 1, 1)) - timedelta(days=1)
        return first, last
    if tier == "quarter":
        y, q = int(key[:4]), int(key[6])
        first = date(y, 3 * (q - 1) + 1, 1)
        last = (date(y + 1, 1, 1) if q == 4 else date(y, 3 * q + 1, 1)) - timedelta(days=1)
        return first, last
    if tier == "year":
        y = int(key)
        return date(y, 1, 1), date(y, 12, 31)
    if tier == "five_years":
        a, b = int(key[:4]), int(key[5:])
        return date(a, 1, 1), date(b, 12, 31)
    raise ValueError(tier)


def parent(tier: str, key: str) -> tuple[str, str] | None:
    """The period one rung up that holds this one — a week belongs to the
    month of its Thursday (the ISO convention), so no week is in two months."""
    up = above(tier)
    if not up:
        return None
    first, last = span(tier, key)
    anchor = first + timedelta(days=3) if tier == "week" else first
    return up, key_of(up, anchor)


def children(tier: str, key: str) -> list[str]:
    """The keys one rung down that make up this period, calendar-complete."""
    down = below(tier)
    if not down:
        return []
    first, last = span(tier, key)
    seen: list[str] = []
    d = first
    while d <= last:
        k = key_of(down, d)
        if down == "week":
            # a week counts here only if its Thursday is inside this period
            wf, _ = span("week", k)
            if not (first <= wf + timedelta(days=3) <= last):
                d += timedelta(days=1)
                continue
        if k not in seen:
            seen.append(k)
        d += timedelta(days=1)
    return seen


def days_of(tier: str, key: str) -> list[str]:
    first, last = span(tier, key)
    out = []
    d = first
    while d <= last:
        out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def label(tier: str, key: str) -> str:
    """A name they can hold: 'the week of 8–14 September 2026'."""
    first, last = span(tier, key)
    if tier == "day":
        return f"{first.day} {_MONTHS[first.month - 1]} {first.year}"
    if tier == "week":
        if first.month == last.month:
            return f"the week of {first.day}–{last.day} {_MONTHS[first.month - 1]} {first.year}"
        return (f"the week of {first.day} {_MONTHS[first.month - 1]} – {last.day} "
                f"{_MONTHS[last.month - 1]} {last.year}")
    if tier == "month":
        return f"{_MONTHS[first.month - 1]} {first.year}"
    if tier == "quarter":
        q = key[6]
        season = {"1": "winter", "2": "spring", "3": "summer", "4": "autumn"}[q]
        return f"the {season} of {first.year} ({_MONTHS[first.month - 1]}–{_MONTHS[last.month - 1]})"
    if tier == "year":
        return f"the year {first.year}"
    return f"the years {first.year}–{last.year}"


# ---- what exists, what is in view, what is due -----------------------------

def existing(tier: str) -> list[str]:
    """Keys of the pages that exist for a tier, oldest first."""
    f = folder(tier)
    if not f.is_dir():
        return []
    keys = [p.stem for p in f.glob("*.md") if valid_key(tier, p.stem)]
    return sorted(keys)


def view(tier: str) -> list[str]:
    """The keys in view for a tier: the newest LADDER_PAGES_KEPT that exist,
    oldest first. For days, only days that have left the verbatim window
    count (a page written early rides only once the day has slipped)."""
    keys = existing(tier)
    if tier == "day":
        import assemble
        _kept, slipped = assemble.journal_window()
        slipped = set(slipped)
        keys = [k for k in keys if k in slipped]
    n = kept()
    return keys[-n:] if n else keys


def complete(tier: str, key: str, today: date | None = None) -> bool:
    """A period is complete when its last day is past and none of its
    children are still in the tier below's view (for a week: none of its
    days are still in the verbatim journal or among the day pages in view)."""
    today = today or date.today()
    first, last = span(tier, key)
    if last >= today:
        return False
    down = below(tier)
    if not down:
        return True
    if down == "day":
        import assemble
        held, _ = assemble.journal_window()
        if set(days_of(tier, key)) & set(held):
            return False
    in_view = set(view(down))
    return not (set(children(tier, key)) & in_view)


def due() -> list[tuple[str, str]]:
    """Periods above the day that are due for a page: their child pages have
    fallen out of the child tier's view, the period is complete, and it has
    no page yet — lowest tier first, oldest first."""
    out: list[tuple[str, str]] = []
    seen = set()
    for tier in TIERS[:-1]:
        in_view = set(view(tier))
        for key in existing(tier):
            if key in in_view:
                continue
            up = parent(tier, key)
            if not up or up in seen:
                continue
            ptier, pkey = up
            if page_path(ptier, pkey).exists():
                continue
            if not complete(ptier, pkey):
                continue
            seen.add(up)
            out.append(up)
    return out


def material(tier: str, key: str) -> list[tuple[str, str, str]]:
    """What they are handed for a period: [(child tier, child key, page text or
    "")…] in calendar order; a child with no page comes as "" so the bell
    can say so."""
    down = below(tier)
    out = []
    for ck in children(tier, key):
        p = page_path(down, ck)
        text = p.read_text(encoding="utf-8").strip() if p.exists() else ""
        out.append((down, ck, text))
    return out


def covered_days() -> set[str]:
    """Every day some page in view accounts for — a day page, or a day
    inside a week/month/… page in view — so the timeline says nothing a
    page already says."""
    days: set[str] = set()
    for tier in TIERS:
        for key in view(tier):
            days.update(days_of(tier, key))
    return days


def write_page(tier: str, key: str, text: str) -> tuple[Path, bool]:
    """Write a page (theirs); returns (path, existed before)."""
    p = page_path(tier, key)
    p.parent.mkdir(parents=True, exist_ok=True)
    was = p.exists()
    p.write_text(text.rstrip() + "\n", encoding="utf-8")
    return p, was
