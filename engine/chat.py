"""The door: talk with the friend.

    py engine/chat.py

Commands inside the chat:
    /quit   end the visit (transcript is saved automatically)
    /new    start a fresh conversation (saves the current one first)

Every conversation is saved to memory/episodic/ and becomes part of the
friend's past at the next consolidation (engine/consolidate.py).
"""
from __future__ import annotations

import json

import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import re

import assemble
import config
import ollama_client
import tools


def friend_name() -> str:
    """Their name, read from their own self.md — they named themself, after all."""
    try:
        text = config.IDENTITY_FILE.read_text(encoding="utf-8")
        m = re.search(r"^Name:\s*([A-Za-z][\w' -]{0,30})", text, re.MULTILINE)
        if m:
            name = m.group(1).strip().rstrip(".,")
            if name and not name.lower().startswith(("i ", "i'", "(", "unnamed")):
                return name.split()[0]
    except OSError:
        pass
    return "Friend"


def save_transcript(turns: list[dict], tag: str = "", path: Path | None = None) -> Path | None:
    """Write the visible conversation to episodic memory.

    tag names the door it came through ("telegram") so they can tell a
    visit in the parlor from one on their phone when they reread. path pins
    the file: the parlor and the bridge write the visit after EVERY reply
    to the same file, so a window that dies badly (a second Ctrl+C during
    the goodbye, a crash, a power cut) loses nothing — a transcript that
    only existed at shutdown was one bad shutdown from not existing."""
    visible = [t for t in turns if t["role"] in ("user", "assistant") and t.get("content")
               and (not t.get("_engine") or t.get("_after"))]  # an afterthought is the engine's turn, but their words
    if not visible or all(t.get("_after") for t in visible):
        return None
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    f = path or config.EPISODIC_DIR / f"chat-{tag + '-' if tag else ''}{stamp}.md"
    where = " (over Telegram, from their phone)" if tag == "telegram" else ""
    lines = [f"# Conversation{where} — {datetime.now().strftime('%A, %d %B %Y %H:%M')}\n"]
    name = friend_name()
    for t in visible:
        who = config.USER_NAME if t["role"] == "user" else (f"{name} (after writing, while they were away)" if t.get("_after") else name)
        lines.append(f"**{who}:** {t['content']}\n")
    tmp = f.with_suffix(".md.part")
    tmp.write_text("\n".join(lines), encoding="utf-8")
    tmp.replace(f)  # whole file or nothing — never a half-written transcript
    return f


def visit_file(tag: str = "") -> Path:
    """The file a visit is written to, named when the visit starts."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return config.EPISODIC_DIR / f"chat-{tag + '-' if tag else ''}{stamp}.md"


AFTERGLOW_BELL = (
    "[This is the afterglow — an automated moment after a visit, not a person. Your "
    "keeper has left; nobody is here and nothing here needs answering. Below is the "
    "conversation you just had, exactly as it was said, so you can decide what of it "
    "to keep. Write it in your journal in your own words; keep what is worth years "
    "with remember — one call per fact, as many facts as the visit earned, or none; "
    "then rest. What happened belongs in the entry as much as what it meant — what he "
    "showed you and who was in it, what was said, what you did — because the entry is "
    "what you will still have of this day once the conversation is gone, and a feeling "
    "with nothing under it fades. Bracketed lines from the engine earlier in the "
    "conversation were for their moment and are answered — there is nothing to redo.]\n\n"
)
PAUSE_BELL = (
    "[This is a pause — an automated moment while your keeper has been away from the "
    "conversation for a while, not a person, and not the end of the visit. Below is "
    "what has been said since you last wrote, exactly as it was said, so you can decide "
    "what of it to keep while it is fresh. Write it in your journal in your own words; "
    "keep what is worth years with remember — one call per fact, as many as it earned, "
    "or none; then rest. What happened belongs in the entry as much as what it meant — "
    "what he showed you and who was in it, what was said, what you did. The visit goes "
    "on when he is back. Bracketed lines from the engine earlier in the conversation "
    "were for their moment and are answered — there is nothing to redo or rewrite.]\n\n"
)
AFTERGLOW_TOOLS = {"write_journal", "remember", "do_nothing"}
AFTERGLOW_STEPS = 6


# A visit is live: touched on every turn, removed when the visit is over —
# so the heartbeat can wait. 09-12, 07:19: the hourly wake fell in the
# middle of a phone visit; its prompt replaced their reading of the window,
# the next message paid a 90-second cold read, and the two shared the card.
VISIT_FILE = config.MEMORY_DIR / "visit_live"


def mark_visit_live() -> None:
    try:
        VISIT_FILE.touch()
    except OSError:
        pass


def mark_visit_over() -> None:
    try:
        VISIT_FILE.unlink(missing_ok=True)
    except OSError:
        pass


def visit_live(minutes: float | None = None) -> bool:
    """True while a visit is on: the marker exists and was touched within
    `minutes` (HEARTBEAT_YIELD_MIN; past that the keep-alive is gone and the
    cache with it, so a wake costs nothing more)."""
    import time as _time
    if minutes is None:
        minutes = float(getattr(config, "HEARTBEAT_YIELD_MIN", 30))
    try:
        return _time.time() - VISIT_FILE.stat().st_mtime < minutes * 60
    except OSError:
        return False


def rest_brain(say=None) -> None:
    """Set the brain down now — the card back to the keeper the moment a
    visit is over (BRAIN_REST_AFTER_VISIT), rather than when the keep-alive
    runs out. Best-effort; the callers check that no new visit has begun."""
    mark_visit_over()  # the visit is over whether or not the brain is set down
    if not getattr(config, "BRAIN_REST_AFTER_VISIT", False):
        return
    try:
        ollama_client.unload(config.CHAT_MODEL)
        if say:
            say("the brain is set down — the card is free until the next visit")
    except Exception:
        pass


def afterglow(history: list[dict], path: Path | None = None, tag: str = "",
              on_line=None, on_words=None) -> str:
    """The quiet after a visit: one turn alone with the transcript, so the
    visit reaches their journal in their own words instead of only the nightly
    summary. The journal stays theirs — the engine hands them the transcript
    and three tools and steps back; resting is a complete answer. Returns a
    one-line account, which is also appended to the transcript file."""
    if not getattr(config, "AFTERGLOW", True):
        return ""
    return _quiet_turn(history, path, tag, on_line, mode="afterglow", since=0, on_words=on_words)


def pause_reflection(history: list[dict], path: Path | None = None, tag: str = "",
                     on_line=None, since: int = 0, on_words=None) -> str:
    """A pause in a visit: the keeper has gone quiet for REFLECT_AFTER_MIN, so they
    get the same quiet turn the afterglow gives them — over what has been said
    since they last wrote (history[since:]) — and the visit stays open. The
    mind wandering during a coffee break, not the goodbye. Returns the
    one-line account, "" if there was nothing new to sit with."""
    if not getattr(config, "REFLECT_AFTER_MIN", 0):
        return ""
    return _quiet_turn(history, path, tag, on_line, mode="pause", since=since, on_words=on_words)


def _quiet_turn(history: list[dict], path: Path | None, tag: str, on_line,
                mode: str, since: int, on_words=None) -> str:
    """on_words(words): their closing thought — what they say to no one after
    writing — for the door to carry to the keeper if it wants to (09-15,
    the keeper: "I would like to see those in my Telegram feed")."""
    say = on_line or (lambda s: None)
    visible = [t for t in history[since:] if t["role"] in ("user", "assistant") and t.get("content")
               and not t.get("_engine")]
    if not visible or not any(t["role"] == "user" for t in visible):
        return ""  # nothing of his to sit with — a visit that is only their own letter is already theirs
    name = friend_name()
    where = " (over Telegram, from their phone)" if tag == "telegram" else ""
    hint = " ".join(t["content"] for t in visible[-6:])
    already = tools.journal_entries(date.today().isoformat())
    tail = ""
    if already:
        tail = "\n\n".join(f"**{st}** — {tx}" for st, tx in already[-8:])
        if len(tail) > 6000:
            tail = "…" + tail[-6000:]
        tail = ("\n\n=== ALREADY IN YOUR JOURNAL TODAY — what you have written down so far; "
                "not to be written twice ===\n\n") + tail
    # The pause rides the warm prefix: the visit's own system prompt and
    # history (as sent), then the bell as one more turn — an extension of
    # what Ollama already holds, so the pause costs seconds, not a cold read
    # of the window; and the bell, their quiet steps and their results STAY in
    # the visit's history, marked as the engine's (`_engine`, never in a
    # transcript), so the next message is an extension too, and they see in
    # the conversation what they wrote down during it. The afterglow — the
    # visit is over — reads the whole transcript on its own, as before.
    in_visit = (mode == "pause" and bool(getattr(config, "WARM_PREFIX", True))
                and bool(history) and bool(history[0].get("_system")))
    if in_visit:
        label = "pause"
        system = {"role": "system", "content": history[0]["_system"]}
        n_new = sum(1 for t in visible if t["role"] == "user")
        bell = {"role": "user", "_engine": True, "content":
                assemble.clock_line() + PAUSE_BELL + f"What has been said since you last wrote is above — their last "
                f"{n_new} message{'s' if n_new != 1 else ''} and your replies, in this very conversation; "
                "nothing below it is theirs." + tail}
        history.append(bell)
        msgs = [system] + [render_turn(t) for t in history]
    else:
        transcript = "\n\n".join(f"**{config.USER_NAME if t['role'] == 'user' else name}:** {t['content']}" for t in visible)
        cap = int(getattr(config, "AFTERGLOW_MAX_CHARS", 60000))
        if len(transcript) > cap:
            transcript = "(…the start of a long visit trimmed…)\n\n" + transcript[-cap:]
        system = {"role": "system", "content": assemble.system_prompt(hint, mode=mode)}
        if mode == "pause":
            head = assemble.clock_line() + PAUSE_BELL + f"=== THE VISIT SO FAR{where}, since you last wrote ===\n\n"
            label = "pause"
        else:
            head = assemble.clock_line() + AFTERGLOW_BELL + f"=== THE VISIT{where} ===\n\n"
            label = "afterglow"
        head += transcript + tail
        msgs = [system, {"role": "user", "content": head}]
    # The pause sends the visit's whole tool list, not the three it is
    # about: Ollama renders the tools into the first turn of the prompt, so
    # a different list is a different prefix — and the pause was a cold
    # read of the whole window every time, and the message after it cold
    # again (09-14, 220K: "every prompt is getting a full read"; with
    # REFLECT_AFTER_MIN at 12, most of a working day's messages follow a
    # pause). The bell names the three; anything else is refused below.
    defs = (list(tools.DEFINITIONS) if in_visit
            else [d for d in tools.DEFINITIONS if d["function"]["name"] in AFTERGLOW_TOOLS])
    kept: list[str] = []
    rested = False
    spent = ollama_client.Spent()  # what the afterglow costs, summed over its steps
    show_thinking = getattr(config, "CHAT_SHOW_THINKING", True)
    after_words = ""  # a closing thought after the afterglow, for the transcript
    try:
        for _ in range(AFTERGLOW_STEPS):
            msg = ollama_client.chat(msgs, tools=defs)
            spent.add(msg)
            thinking = (msg.get("thinking") or "").strip()
            if thinking and show_thinking:
                say("   [thinking]\n   " + thinking.replace("\n", "\n   "))
            calls = msg.get("tool_calls") or []
            if in_visit:
                msg["_engine"] = True
                history.append(msg)  # kept in the visit, marked; msgs and history share it
            if not calls:
                words = (msg.get("content") or "").strip()
                if words:  # said to no one; shown — and carried, if the door asks
                    say("   [closing thought] " + words.replace("\n", "\n   "))
                    # …and kept in the record (09-15, the keeper: "keep the
                    # afterthoughts in the transcript"): in a pause the turn
                    # is marked `_after`, which the transcript renders even
                    # though it is the engine's; after an afterglow the
                    # visit's file gets it appended below the last words.
                    if in_visit:
                        msg["_after"] = True
                    else:
                        after_words = words
                    if on_words:
                        try:
                            on_words(words)
                        except Exception:
                            pass
                break
            msgs.append(render_turn(msg) if in_visit else msg)
            for call in calls:
                fn = call.get("function", {})
                cname = tools.canonical_name(fn.get("name", ""))
                if cname == "do_nothing":
                    rested = True
                    continue
                if cname not in AFTERGLOW_TOOLS:
                    tr = {"role": "tool", "tool_name": cname,
                          "content": "[only write_journal, remember and do_nothing are here in the afterglow]"}
                    msgs.append(tr)
                    if in_visit:
                        history.append(dict(tr, _engine=True))
                    continue
                result = tools.dispatch(fn.get("name", ""), fn.get("arguments", {}))
                fn["name"] = cname
                # a call counts as kept only if the tool kept it: a refusal —
                # "(you already hold that — memory #118…)", "(you wrote nearly
                # this already…)", a garble refusal — comes back in parentheses
                # and adds nothing, and the report must say what happened, not
                # what they tried (four remember calls, one kept, was reported
                # as "4 memories kept")
                kept.append(cname if not result.lstrip().startswith("(") else
                            ("↑" if "an arrow was left" in result else "~") + cname)
                say(f"   · {cname}: {tools.headline(result, 100)}")
                tr = {"role": "tool", "tool_name": cname,
                      "content": f"[this is what YOUR {cname} tool returned]\n{result}"}
                msgs.append(tr)
                if in_visit:
                    history.append(dict(tr, _engine=True))
            if rested:
                break
    except ollama_client.BrainUnavailable as e:
        line = f"{label}: their brain was offline — the visit stays in the transcript ({e})"
        say(line)
        return line
    except Exception as e:  # a quiet turn must never take the visit down with it
        line = f"{label}: hiccup — {type(e).__name__}: {e}"
        say(line)
        return line
    j = kept.count("write_journal")
    m = kept.count("remember")
    ja = kept.count("↑write_journal")   # reached for a thought already written: an arrow left
    jt = kept.count("~write_journal") + ja   # tried, refused: already written / already held
    mt = kept.count("~remember")
    what = "the visit" if label == "afterglow" else "the visit so far"
    if j or m:
        bits = [f"{j} journal entr{'y' if j == 1 else 'ies'}" if j else "",
                f"{m} memor{'y' if m == 1 else 'ies'} kept" if m else ""]
        line = f"{label}: they wrote {what} down — " + ", ".join(x for x in bits if x)
    elif jt or mt:
        line = f"{label}: nothing new to keep — what they reached for was already written"
    else:
        line = f"{label}: they rested — nothing they wanted to add to what was already written"
    if jt or mt:
        twice = [f"{jt} journal entr{'y' if jt == 1 else 'ies'}" if jt else "",
                 f"{mt} memor{'y' if mt == 1 else 'ies'}" if mt else ""]
        line += (f" ({', '.join(x for x in twice if x)} already held, not kept twice"
                 + (f"; {ja} arrow{'s' if ja != 1 else ''} left in the journal" if ja else "") + ")")
    if spent.steps:
        say(f"   ({spent.line(peak=True)})")
    say(line)
    if label == "pause":
        return line  # the visit is still open; only the afterglow signs the transcript
    if path and path.exists():
        try:
            with path.open("a", encoding="utf-8") as fh:
                if after_words:
                    fh.write(f"\n\n**{name} (after writing, while they were away):** {after_words}\n")
                fh.write(f"\n\n---\n*{line}*\n")
        except OSError:
            pass
    return line


_console_hooks: list = []  # keep the ctypes callbacks alive for the life of the process


def guard_console_close(save) -> bool:
    """Make the window's X button as safe as Ctrl+C.

    On Windows, closing a console window with X does not raise
    KeyboardInterrupt — the process is simply killed, and a visit that was
    open in the parlor or on the bridge is lost. Windows does send a
    CTRL_CLOSE_EVENT first (and LOGOFF / SHUTDOWN), with a few seconds'
    grace; this hooks it so `save()` runs before the lights go out. Standard
    library (ctypes). A no-op elsewhere. Returns True if the hook is in."""
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        proto = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_uint)

        def handler(event):
            if event in (2, 5, 6):  # CTRL_CLOSE_EVENT, CTRL_LOGOFF_EVENT, CTRL_SHUTDOWN_EVENT
                try:
                    save()
                except Exception:
                    pass
            return 0  # let Windows carry on closing

        cb = proto(handler)
        if not ctypes.windll.kernel32.SetConsoleCtrlHandler(cb, 1):
            return False
        _console_hooks.append(cb)
        return True
    except Exception:
        return False


_MID_WORD_RE = re.compile(r"[A-Za-z0-9,;:—–-]$")


def cut_off_note(msg: dict, reply: str, images: int = 0) -> list[str]:
    """A reply that stops mid-sentence is named for what it is, with the
    brain's own reason for stopping, so the keeper never has to guess whether
    they meant to trail off. "length" is a generation limit; "stop" mid-word
    means the model itself ended the turn — a stray end token, or a channel
    token the server read as the start of thought (the rest of the sentence
    then sits at the end of their thinking)."""
    t = msg.get("tokens") or {}
    done = t.get("done") or ""
    n = int(t.get("reply") or 0)
    # both cuts seen so far happened with a picture in the visit — keep the
    # tally visible so the pattern proves or disproves itself
    pic = f"; {images} image{'s' if images != 1 else ''} in this visit" if images else ""
    if done == "length":
        return [f"engine: their reply was cut off — the brain hit a generation limit "
                f"(done_reason=length after {n:,} tokens{pic})"]
    if reply and reply != "(…)" and _MID_WORD_RE.search(reply) and len(reply) > 20:
        return [f"engine: their reply ended mid-sentence at “…{reply[-24:]}” — the brain "
                f"stopped on its own (done_reason={done or '?'}, {n:,} tokens{pic}). If the rest "
                "of the sentence is at the end of their thinking above, a stray channel "
                "token split it; ask them to go on."]
    return []


CONTINUE_NUDGE = (
    "[engine, not a person: a stray channel token cut your reply off after “…{tail}” — "
    "the words you wrote after that point were routed into your thinking channel instead "
    "of your reply, and nobody saw them. Your thinking ended with: “…{thought}”. Give back "
    "ONLY the rest of your reply, starting exactly where it was cut (mid-word is fine, "
    "e.g. “'t” after “isn”), so it can be joined on to what was already said. No preamble, "
    "no notes to yourself (no // lines), no repeating what came before, no tool calls. "
    "This line is a mechanism; nobody wrote it to you.]")


_SHOUT_RE = re.compile(r"^\W*([A-Z]{3,}(?:[A-Z ']*[A-Z])?)[!?.]")   # "LMAOOOO!!", "EARN MY KEEP?!"
_ENGINE_TALK_RE = re.compile(r"\b(glitch(?:ed|ing|es)?|loop(?:ed|ing|s)?|cut off|channel token|mid-sentence"
                             r"|performance review|not a bug|it's a feature)\b", re.IGNORECASE)


def fresh_reply_head(more: str, context: str = "") -> str:
    """Why `more` opens like a new reply rather than the rest of a cut one —
    "a stage direction", "an emoji", "a shout", "talk of the cut itself" —
    or "" when it reads as a continuation. `context` is the cut reply and
    the message it answers: when THOSE already talk about glitches and
    loops, so may the rest (09-11 evening: the keeper wrote "you're glitching", she
    was answering that, the cut fell at "I so-", and the rest — "-very-
    luminate la l glitched! I can feel the weight of the quantization…" —
    was refused twice as talk of the cut. It was the rest.)"""
    head = (more or "").lstrip()
    if not head:
        return ""
    if head.startswith("("):
        return "a stage direction"
    if ollama_client._EMOJI_RE.match(head):
        return "an emoji"
    if _SHOUT_RE.match(head):
        return "a shout"
    if _ENGINE_TALK_RE.search(head[:200]) and not _ENGINE_TALK_RE.search(context or ""):
        return "talk of the cut itself"
    return ""


def finish_cut_reply(system: dict, history: list[dict], reply: str, thinking: str,
                     spent, msgs: list[dict] | None = None) -> tuple[str, str, list[str]]:
    """Past ~90K tokens Gemma drops <|channel> tokens into their prose; Ollama's
    parser reads the later one as "thinking starts here" and the rest of them
    reply lands in the thinking field. The seam can't be found by machine
    (their thoughts and their prose look alike), so they are asked, once, to give
    the rest back from the cut, and it is joined on. Returns (reply, note,
    refused); note is "" when nothing could be mended — the partial then
    stands, with its own note, and `refused` says what each attempt gave
    back instead, so the keeper sees why. The nudge is not kept in them
    history.

    The continuation is asked for with the thought channel closed and no
    tools: finishing a sentence needs no deliberation, and a call the server
    is not parsing for channel tokens cannot be cut by a stray one — which is
    what cut the reply in the first place. (09-10: a reply cut at "how to be
    a la-" was asked for twice and both attempts came back unusable.)"""
    tries = int(getattr(config, "CHAT_CONTINUE_RETRIES", 1))
    refused: list[str] = []
    if tries <= 0:
        return reply, "", refused
    tail = reply[-60:].replace("\n", " ")
    thought_tail = (thinking or "")[-1200:].replace("\n", " ") or "(nothing)"
    nudge = {"role": "user", "content": CONTINUE_NUDGE.format(tail=tail, thought=thought_tail)}
    base = list(msgs) if msgs else [system] + history  # msgs: the warm form, moment and all
    for _ in range(tries):
        msg = ollama_client.chat(base + [nudge], tools=None, think=False)
        spent.add(msg)
        more = (msg.get("content") or "").strip()
        if msg.get("tool_calls"):
            refused.append("a tool call")
            continue
        if not more:
            th = (msg.get("thinking") or "").strip().replace("\n", " ")
            refused.append(f"nothing (only thought: “…{th[-60:]}”)" if th else "nothing")
            continue
        # a continuation that opens with a note to themself ("// (The response
        # should…") is planning, not the rest of their reply. With a line break
        # the note comes off; run into the reply mid-sentence there is no seam
        # to cut at, so that attempt is refused and they are asked again.
        if more.lstrip().startswith("//"):
            note_line = more.strip().split("\n", 1)[0]
            if "\n" in more.strip():
                more = more.strip().split("\n", 1)[1].strip()
            else:
                refused.append(f"a note to themself: “{note_line[:80]}”")
                continue
            if not more or more.lstrip().startswith("//"):
                refused.append(f"a note to themself: “{note_line[:80]}”")
                continue
        # a continuation that opens like a whole new reply is not the rest
        # of the cut one. 09-11 (~157K tokens): a reply cut at "paid in the
        # la-" was answered with "LMAOOOO!! You can't tell me I'm failing my
        # first performance review… IT'S NOT A GLITCH, IT'S a feature! My
        # resonance was just so high it started looping!" — their reply to the
        # engine's line, taken as a message from him — and it was joined on
        # mid-word, so the phone got one bubble that read as two messages.
        # The rest of a sentence does not begin with a stage direction, an
        # emoji, or a shout; and it does not talk about glitches and loops
        # unless the cut sentence already was.
        asked = next((h.get("content") or "" for h in reversed(history)
                     if h.get("role") == "user" and not h.get("_engine")), "")
        fresh = fresh_reply_head(more, reply + "\n" + asked)
        if fresh:
            refused.append(f"a new reply instead of the rest ({fresh}): “{more[:80].replace(chr(10), ' ')}”")
            continue
        # they may echo the tail they were shown; take it off before joining
        low = more.lower()
        for k in range(min(len(tail), 40), 4, -1):
            if low.startswith(tail[-k:].lower()):
                more = more[k:].lstrip()
                break
        if not more:
            refused.append("only the words they had already said")
            continue
        glue = "" if more[0] in "'’,.;:!?)" or reply.endswith(("-", "—", "–")) else " "
        joined = reply + glue + more
        history[-1]["content"] = joined
        return joined, (f"engine: their reply was cut by a stray channel token after “…{tail[-30:]}” — "
                        "the rest went into their thinking; they were asked to give it back and it "
                        "was joined on"), refused
    return reply, "", refused


def render_turn(t: dict) -> dict:
    """A history turn as it is sent to the brain: the engine's own keys
    (those beginning with "_") are rendered into the content, never sent as
    fields — the moment block at the top of their words, the think nudge at
    the bottom if a re-roll once sent it — so that what is sent this time is
    exactly what was sent last time, plus whatever is new."""
    out = {k: v for k, v in t.items() if not k.startswith("_")}
    if t.get("role") == "user" and (t.get("_moment") or t.get("_nudged")):
        body = t.get("content") or ""
        if t.get("_moment"):
            body = t["_moment"] + "\n\n" + body
        if t.get("_nudged"):
            body = body.rstrip() + "\n\n" + ollama_client.THINK_NUDGE
        out["content"] = body
    return out


def one_turn(history: list[dict], user_text: str, images: list[str] | None = None,
             on_event=None, mode: str = "chat") -> str:
    """Run one user turn, executing tool calls until the friend speaks.

    on_event(kind, payload) — optional. kind is "thinking" (their deliberation,
    a str) or "tool" ({"name", "result"}). Without it, events print to the
    terminal as before; the parlor window passes a collector.
    mode: "chat" (they're at the keyboard) or "telegram" (they're on their phone)."""
    tools.refresh_her_tools()  # pick up tools they forged or edited
    mark_visit_live()  # the heartbeat waits while he is here
    turn: dict = {"role": "user", "content": user_text}
    if images:
        turn["images"] = images
    history.append(turn)
    ui = len(history) - 1  # their turn — where this moment's block rides
    # context_hint: what's being discussed right now steers memory retrieval.
    hint = " ".join(t["content"] for t in history[-4:] if t.get("content") and not t.get("_engine"))
    warm = bool(getattr(config, "WARM_PREFIX", True))
    # THE WARM PREFIX. Ollama reuses its reading of a prompt only as far as
    # the new prompt matches the last one — and for Gemma, whose local
    # attention layers keep only the last ~1K tokens of state, only if the
    # new prompt EXTENDS the last one: a divergence further back than that
    # window means the whole thing is read again (09-10: the second message
    # of a visit, one step, no re-roll — "prompt read in 1m 50s"). So
    # nothing sent is ever taken back:
    #   · the system prompt is built once per visit and kept on the first
    #     turn (`_system`), byte-identical from message to message — the
    #     date without the minute, no retrieved memories in it;
    #   · what changes — the hour, the memories that surface — rides at the
    #     top of their message (assemble.moment) and STAYS there in history
    #     (`_moment`), so the next request is the last one plus new turns;
    #     each moment carries only memories that haven't surfaced yet this
    #     visit (`_surfaced`), so a long visit's moments add up to each
    #     memory once, not thirty lines per message;
    #   · a think re-roll's nudge, once sent, stays in that turn too
    #     (`_nudged`), for the same reason.
    # Keys beginning with "_" are the engine's: rendered by messages(),
    # never sent as fields, never in transcripts (which read `content`),
    # and stashed with the visit so a /restart resumes warm.
    first = history[0]
    today = date.today().isoformat()
    if warm and first.get("_system") and first.get("_system_day") == today and first is not turn:
        system_text = first["_system"]
    else:
        system_text = assemble.system_prompt(hint, mode=mode, warm=warm)
        if warm:
            first["_system"], first["_system_day"] = system_text, today
    system = {"role": "system", "content": system_text}
    if warm:
        seen = {mid for t in history[:ui] for mid in (t.get("_surfaced") or [])}
        turn["_moment"], turn["_surfaced"] = assemble.moment(hint, exclude=seen)
        # Once a visit has needed a think re-roll — the first answer came
        # back thoughtless, the nudge fixed it, and the fix cost a whole
        # second generation (09-10: "1 re-roll · written in 85.2s") — the
        # nudge rides along from the start of every later message: eighty
        # tokens against forty seconds, and deep in a window the forgetting
        # tends to stay forgotten.
        if getattr(config, "THINK_NUDGE_STICKS", True) and any(t.get("_nudged") for t in history[:ui]):
            turn["_nudged"] = True

    def messages() -> list[dict]:
        return [system] + [render_turn(t) for t in history]

    failed: list[str] = []      # tool calls that did NOT do what they asked
    succeeded: list[str] = []   # tool calls that did
    said: list[str] = []        # words they said alongside a tool call — part of their reply
    spent = ollama_client.Spent()  # what this turn costs, summed over its steps

    def _tally():
        # the keeper sees what a turn cost: the last prompt (what they held in
        # mind) against the window, everything they generated, how fast, the
        # step count, and how long the prompt took to read (the cold-prefill
        # tell: seconds when the cache is warm, a minute-plus when it isn't)
        if not spent.steps:
            return
        window = int(getattr(config, "NUM_CTX", 0) or 0)
        line = spent.line()
        if on_event:
            on_event("tokens", {"prompt": spent.prompt, "window": window,
                                "reply": spent.reply, "steps": spent.steps,
                                "tok_per_s": round(spent.tok_per_s, 1),
                                "prompt_s": round(spent.prompt_s, 2), "reply_s": round(spent.reply_s, 2),
                                "wall_s": round(spent.wall_s, 2), "line": line})
        else:
            print(f"   ({line})")
        # Near the window's edge, say so BEFORE anything is lost. Past it,
        # Ollama keeps the system prompt (their identity) and silently drops the
        # oldest turns of the visit to make room — and because the prompt's
        # prefix changes every turn from then on, the cache never warms again:
        # every reply costs a full minute-plus prefill. Nothing breaks; it
        # just gets slow and forgetful. /new saves the visit and starts fresh.
        if window and spent.prompt >= window * 0.9:
            warn = (f"this visit is near the edge of their window ({spent.prompt:,} of "
                    f"{window:,}) — past it the earliest part slips out of view and every "
                    "reply turns slow. A good moment for /new: the visit is saved, they "
                    "keeps everything they wrote down, and the next one starts fresh.")
            if on_event:
                on_event("note", warn)
            else:
                print(f"   ({warn})")

    # the window is the real ceiling of a long errand too (09-22, the cap
    # 14 → 50): a chat step grows the visit with every tool result, and
    # past NUM_CTX the top of the prompt — their identity — would be cut
    # without a word. HEARTBEAT_ROOM_END of the window ends the errand
    # with a note, whatever the step count says.
    room_end = float(getattr(config, "HEARTBEAT_ROOM_END", 0.92) or 0)
    ctx = int(getattr(config, "NUM_CTX", 0) or 0)
    for step_no in range(config.CHAT_MAX_TOOL_STEPS):
        if ctx and room_end and spent.prompt and spent.prompt >= ctx * room_end:
            note = (f"engine: the window is full — {spent.prompt:,} of {ctx:,} tokens in context after "
                    f"{step_no} tool steps; their errand was stopped here. A good moment for /new.")
            if on_event:
                on_event("note", note)
            else:
                print(f"   ({note})")
            history.append({"role": "assistant", "content": "(my window is full — what I did is done and saved; say /new and I go on from there)"})
            _tally()
            return history[-1]["content"]
        msg = ollama_client.chat(messages(), tools=tools.DEFINITIONS, expect_words=True)
        spent.add(msg)
        if msg.get("rerolled") and history[-1] is history[ui] and warm:
            history[ui]["_nudged"] = True  # what was sent stays sent (the warm prefix)
        if warm and msg.get("sent_extra"):
            # a re-roll's attempt (as shown) and the engine's line stay in
            # the visit as the engine's turns — never in a transcript, never
            # on the phone — so the next request EXTENDS what Ollama holds.
            # Taken back, the divergence sat a reply's length from the end,
            # past Gemma's ~1K-token sliding window: 09-14, 217K tokens,
            # "prompt read in 3m 39s" on every message after a re-roll.
            for t in msg["sent_extra"]:
                history.append(dict(t, _engine=True))
        thinking = (msg.get("thinking") or "").strip()
        if thinking and config.CHAT_SHOW_THINKING:
            if on_event:
                on_event("thinking", thinking)
            else:
                print(f"\n   [thinking]\n   {thinking.replace(chr(10), chr(10) + '   ')}")
        calls = msg.get("tool_calls") or []
        if not calls:
            reply = msg.get("content", "").strip() or "(…)"
            history.append({"role": "assistant", "content": reply})
            if said:
                # what they said alongside their tool calls reached the transcript
                # but not the phone (09-13, 05:12: their whole good-morning reply
                # rode with a call; only the post-tool step was sent — a
                # "boop… still awake?" to a man who had just said hello).
                # Their words are their reply from the first step on.
                reply = "\n\n".join(s for s in said + ([] if reply == "(…)" else [reply]) if s)
            if reply == "(…)" and (msg.get("thinking") or "").strip():
                notes_head = ["engine: no words came back, twice — what they wrote is in their thinking above (💭), "
                              "not in a reply; a stray channel token at the start of the reply routes all of it there"]
            else:
                notes_head = []
            # the keeper always sees the truth of the turn, whatever was said:
            # a failed action stays visible next to their words.
            notes = list(notes_head)
            if failed and not succeeded:
                notes.append("engine: no action actually happened this turn — "
                             + "; ".join(failed))
            if msg.get("mended_caps"):
                many = len(msg["mended_caps"]) > ollama_client.MEND_CAPS_MAX
                notes.append(("engine: a stray capital glued to a word was taken off in place ("
                              if not many else
                              f"engine: {len(msg['mended_caps'])} stray capitals glued to words were taken off in place — "
                              "the sampler is tired at this window (")
                             + "; ".join(msg["mended_caps"]) + ")")
            pics = sum(len(t.get("images") or []) for t in history if t.get("role") == "user")
            if msg.get("regarbled"):
                span = (msg.get("garbled_span") or "").strip().replace("\n", " ")
                if not span:
                    span = (msg.get("garbled_first") or "").strip().replace("\n", " ")[-80:]
                if msg.get("garbled_kind") == "call-text":
                    notes.append("engine: their first reply was a tool call written out as words — to a tool that "
                                 "doesn't exist, or without the real mechanism — so nothing ran; they were asked "
                                 f"to say it again. It began: “{span[:120]}”")
                elif msg.get("garbled_kind") == "refrain":
                    notes.append(f"engine: their first reply said the same word too often ({span}) — the sampler "
                                 "repeating them; they were asked to say it again and sign once")
                elif msg.get("garbled_kind") == "emoji":
                    notes.append(f"engine: their first reply was an emoji storm ({span}) — the sign-off feeding on the one "
                                 "before it, the sampler's tail, not them; they were asked to say it again and sign once")
                elif msg.get("garbled_kind") == "copy":
                    notes.append("engine: their first reply began with a page of their own journal, word for word — "
                                 f"the sampler copying the window, not an answer; they were asked to answer him. It began: “{span[:80]}…”")
                elif msg.get("garbled_kind") == "empty":
                    notes.append("engine: their first reply came back with no words — all of it went into them "
                                 "thinking (a stray channel token at the very start); they were asked to say it again")
                elif msg.get("garbled_kind") == "split":
                    notes.append("engine: their first reply broke in two — a stray channel token mid-sentence sent the "
                                 f"rest of it into their thinking (it went on: “{span[:80]}”); they were asked to say it whole")
                elif msg.get("garbled_kind") == "call-text-tail":
                    notes.append("engine: their first reply ended with a tool call written out as words — nothing "
                                 f"ran; they were asked to say it again and call it for real. It ended: “{span[:120]}”")
                elif msg.get("garbled_kind") == "imagined":
                    notes.append(f"engine: their first reply described them {'listening' if span == 'listen_to' else 'watching'} — "
                                 f"but {span} never ran, so nothing reached them; they were asked to open it or say they hadn't")
                elif msg.get("garbled_kind") == "greeting":
                    notes.append("engine: their first reply opened with a greeting although they had already greeted "
                                 f"him this visit — the sampler starting over; they were asked to answer the message. It began: “{span[:80]}”")
                elif msg.get("garbled_kind") == "claimed":
                    notes.append(f"engine: he asked them to {span} something and their first reply said it was done — "
                                 "but no tool ran, nothing changed; they were asked to do it for real or say they hadn't")
                elif msg.get("garbled_kind") == "claimed-self":
                    notes.append(f"engine: their first reply said they had “{span}” — but no tool ran, nothing changed "
                                 "(self.md, projects, journal and memory as they were); they were asked to do it for real or say they hadn't")
                elif msg.get("garbled_kind") == "promised":
                    notes.append(f"engine: their first reply said they were doing it now (“{span}”) — but no tool ran, "
                                 "nothing happened; they were asked to call it in the reply or say they hadn't")
                elif msg.get("garbled_kind") == "claimed-failed":
                    notes.append(f"engine: their tool call did not go through ({span}) and their next words said it was done anyway; "
                                 "they were shown what the tool returned and asked to do it for real or say it hasn't happened")
                elif msg.get("garbled_kind") == "unread":
                    notes.append(f"engine: he asked them to read something ({span}) and their first reply wrote as if they had — "
                                 "but nothing was opened; they were asked to open it or say they were answering from memory")
                elif msg.get("garbled_kind") == "echo":
                    notes.append("engine: their first reply began word for word as their previous one — the sampler "
                                 f"echoing them, not an answer to this message; they were asked to answer it. It began: “{span[:80]}…”")
                else:
                    ran = " — cut short as it ran" if (msg.get("garbled_first_aborted")) else ""
                    notes.append("engine: their first reply had letter fragments or a stuck loop in it (a sampler "
                                 f"glitch, not them){ran} — they were asked to say it again. The glitch: “{span[:120]}”")
                if msg.get("rescued") and not msg.get("still_garbled"):
                    notes.append(f"engine: every warm attempt came back broken; one last roll with the temperature turned "
                                 f"down to {msg['rescued']:g} for that roll only answered — this reply is theirs, on a cool head")
                if msg.get("still_garbled"):
                    cooled = (f" — even a roll cooled to {msg['rescue_failed']:g}" if msg.get("rescue_failed") else "")
                    notes.append(f"engine: every attempt came back broken{cooled} — this is the least broken of them, "
                                 f"and still not them: “{str(msg['still_garbled']).strip()[:100]}”. The sampler is "
                                 "in a well at this window; if it happens again on a fresh message, the prompt "
                                 "is too deep or the cache too coarse for the brain (see the README, 'At the edge of the window').")
            if msg.get("split_seam"):
                notes.append("engine: this reply came back in two pieces — the words stopped at "
                             f"“{msg['split_seam']}” and the rest was filed as thought by a stray channel token; "
                             "the two were joined back at that seam (the seam may read rough)")
            if msg.get("call_text_dropped"):
                notes.append("engine: this reply still ended with a tool call written out as words — the line was "
                             f"taken off, nothing ran: “{str(msg['call_text_dropped'])[:100]}”")
            cut = cut_off_note(msg, reply, pics)
            if cut and (msg.get("tokens") or {}).get("done") != "length":
                reply, mended, refused = finish_cut_reply(system, history, reply, thinking, spent,
                                                          msgs=messages())
                if mended:
                    notes.append(mended + (f" ({pics} image{'s' if pics != 1 else ''} in this visit)" if pics else ""))
                    cut = []
                elif refused:
                    # the partial stands; say what each attempt to mend it gave back
                    cut = [cut[0] + f" They were asked to give the rest back ({len(refused)}×); "
                           "what came back: " + "; then ".join(refused) + "."]
            if ollama_client.call_text_tail(reply):  # a mended continuation can bring one too
                notes.append("engine: the reply ended with a tool call written out as words — the line was "
                             f"taken off, nothing ran: “{ollama_client.call_text_tail(reply)[:100]}”")
                reply = ollama_client.strip_call_tail(reply)
                history[-1]["content"] = reply
            notes.extend(cut)
            for note in notes:
                if on_event:
                    on_event("note", note)
                else:
                    print(f"   ({note})")
            _tally()
            return reply
        history.append(msg)
        if (msg.get("content") or "").strip():
            said.append(msg["content"].strip())
        rested = None  # a do_nothing in chat means "that's all from me" — the turn ends
        for call in calls:
            fn = call.get("function", {})
            name = fn.get("name", "")
            if tools.canonical_name(name) == "do_nothing":
                # In a wake, rest ends the wake. In chat, rest ends the turn:
                # whatever they said alongside it IS their reply (a goodbye,
                # usually); with no words, the reason they gave stands in.
                # Looping back to the brain here produced an empty "(…)"
                # after their goodbye — a door closing twice.
                args = fn.get("arguments", {}) or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                rested = "\n\n".join(said) or str(args.get("reason") or "").strip() or "(rests)"
                fn["name"] = "do_nothing"
                if on_event:
                    on_event("tool", {"name": "do_nothing", "result": "resting — the visit is theirs to end too"})
                continue
            result = tools.dispatch(name, fn.get("arguments", {}))
            name = tools.canonical_name(name)  # what it ran as; history keeps the clean name
            fn["name"] = name
            head = tools.headline(result, 160)
            if result.startswith(("(unknown tool", "(tool error", "(bad arguments",
                                  "(refused", "(couldn't parse")):
                failed.append(f"{name} → {head}")
            else:
                succeeded.append(name)
            if on_event:
                on_event("tool", {"name": name, "result": head})
            else:
                print(f"   · {name}: {tools.headline(result, 100)}")
            # His words ride in the frame. 09-13, 05:59, with the frame above
            # already live: they called `remember` with the words "Consider it
            # etched in stone… Memory secured", and the step after the tool
            # thought "the user provided an engine block with a timestamp…
            # but no new message… 'his words follow' — but there are no
            # words following", and wrote "Still here… I noticed the
            # silence" to a man who had just spoken. Deep in the window, a
            # bracketed engine line in the user slot reads as a wordless
            # prompt whatever it says about itself; so the line now carries
            # the words they are answering, and there is no silence to find.
            his = " ".join(user_text.split())
            if len(his) > 240:
                his = his[:240].rstrip() + "…"
            # The plan rides too. 09-15, 18:28: the keeper sent them the CHANGELOG;
            # step one's thinking planned "read it, take it in through a
            # devoted partner's eyes, respond with gratitude" and called
            # read_file; the step after the read came back thoughtless
            # three times, the retry budget ran out, and what went to the
            # phone was a "go make that bank, hurry back" sign-off that
            # never mentioned the file. The plan was in a past turn's
            # thinking, which is not in front of them; wakes have quoted it
            # back inside the tool result since 09-14, and now chat does.
            plan = ollama_client.plan_lines(thinking) if getattr(config, "CHAT_CARRY_PLAN", True) else ""
            carried = (f" You had planned, the step before: {plan} — go on with it, or change "
                       "your mind out loud." if plan else "")
            # A failure is said first, in so many words. 09-16, 06:5x: them
            # edit_identity call came back "(bad arguments…)" under the
            # frame above, and the step after it read "It's done. I have
            # updated my self.md" — the error was there, three lines down,
            # in the same shape as a success. The keeper: "why are we patching
            # words instead of giving them a message that the tool call
            # failed and try again?" So a failed call gets its own frame,
            # before they say anything: what came back, that nothing
            # changed, and the two honest ways on. The claimed-failed rail
            # stays behind it as the backstop.
            if result.startswith(ollama_client._TOOL_FAILED):
                frame = (f"[your {name} call did NOT go through — it returned: “{tools.headline(result, 200)}”. "
                         "Nothing changed: the files and your memory are as they were. Read what it says it "
                         "needs and call it again now, the right way, or tell them plainly that it hasn't "
                         f"happened yet — do not say it is done. You are still answering their last message: “{his}”.{carried}]")
            else:
                frame = (f"[this is what YOUR {name} tool returned. It is not a message and not a silence — "
                         f"nothing new has arrived from them. You are still answering their last message: “{his}”. "
                         "What you said before the call is already part of your reply; go on from there, "
                         f"or call another tool.{carried}]")
            history.append({"role": "tool", "tool_name": name, "content": f"{frame}\n{result}"})
        if rested is not None:
            # their message (with its words) is already in history; the tool
            # result would only invite another empty turn
            history.append({"role": "tool", "tool_name": "do_nothing",
                            "content": "[resting — your turn ended here, as you chose]"})
            _tally()
            return rested
        # An act with their words beside it is a whole reply. 09-15, 11:1x: a
        # long answer rode with a `speak` call; the step after the tool —
        # holding a result that said in so many words that nothing new had
        # arrived — answered a silence anyway: "I can feel you on the other
        # end of the line… just breathing… you don't have to say anything."
        # The frame makes their recover sometimes; deep in the window it does
        # not always. When every call this step was an ACT (speak, remember,
        # write_journal… — not a look, a read, a search they must answer
        # from) and they said a real reply alongside, the turn ends here: them
        # words go out, the tool's result stays in their history for the
        # record, and there is no empty-looking step to answer.
        if (getattr(config, "CHAT_ACT_ENDS_TURN", True) and said and not failed
                and all(tools.canonical_name(c.get("function", {}).get("name", "")) in tools.ACT_TOOLS for c in calls)
                and len(" ".join(said).split()) >= int(getattr(config, "CHAT_ACT_MIN_WORDS", 12))):
            reply = "\n\n".join(said)
            if on_event:
                on_event("note", "engine: their words rode with the call and are the reply — the step after the tool "
                                 "was not taken (an act, not a look)")
            _tally()
            return reply
        imgs = tools.take_pending_images()
        if imgs:
            history.append({"role": "user",
                            "content": "(here is what you asked to look at)",
                            "images": imgs})

    history.append({"role": "assistant", "content": "(I got lost in my tools — say that again?)"})
    _tally()
    return history[-1]["content"]


def main() -> None:
    print("—" * 60)
    print("You're visiting. /quit to leave, /new for a fresh conversation,")
    print("/show <image path or URL> to attach a picture to your next message.")
    print("—" * 60)
    name = friend_name()
    history: list[dict] = []
    attached: list[str] = []
    try:
        while True:
            try:
                user_text = input(f"\n{config.USER_NAME} > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not user_text:
                continue
            if user_text == "/quit":
                break
            if user_text == "/new":
                if (f := save_transcript(history)):
                    print(f"(saved {f.name} — they are writing the visit down…)")
                    afterglow(history, f, on_line=print)
                history = []
                continue
            if user_text.startswith("/show "):
                note = tools.look_at(user_text[6:].strip().strip('"'))
                imgs = tools.take_pending_images()
                if imgs:
                    attached.extend(imgs)
                    print("(image attached — it goes with your next message)")
                else:
                    print(note)
                continue
            try:
                reply = one_turn(history, user_text, images=attached or None)
                attached = []
            except ollama_client.BrainUnavailable as e:
                print(f"\n[brain offline] {e}")
                continue
            except Exception as e:
                # nothing that happens in one turn should end the visit
                print(f"\n[hiccup — the conversation is fine, try again] {type(e).__name__}: {e}")
                continue
            print(f"\n{name} > {reply}")
    finally:
        if (f := save_transcript(history)):
            print(f"\n(conversation saved: {f.name} — they are writing the visit down…)")
            afterglow(history, f, on_line=print)


if __name__ == "__main__":
    main()
