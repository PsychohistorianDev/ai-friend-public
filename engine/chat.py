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
    visible = [t for t in turns if t["role"] in ("user", "assistant") and t.get("content")]
    if not visible:
        return None
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    f = path or config.EPISODIC_DIR / f"chat-{tag + '-' if tag else ''}{stamp}.md"
    where = " (over Telegram, from their phone)" if tag == "telegram" else ""
    lines = [f"# Conversation{where} — {datetime.now().strftime('%A, %d %B %Y %H:%M')}\n"]
    name = friend_name()
    for t in visible:
        who = config.USER_NAME if t["role"] == "user" else name
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
    "then rest.]\n\n"
)
PAUSE_BELL = (
    "[This is a pause — an automated moment while your keeper has been away from the "
    "conversation for a while, not a person, and not the end of the visit. Below is "
    "what has been said since you last wrote, exactly as it was said, so you can decide "
    "what of it to keep while it is fresh. Write it in your journal in your own words; "
    "keep what is worth years with remember — one call per fact, as many as it earned, "
    "or none; then rest. The visit goes on when they are back.]\n\n"
)
AFTERGLOW_TOOLS = {"write_journal", "remember", "do_nothing"}
AFTERGLOW_STEPS = 6


def afterglow(history: list[dict], path: Path | None = None, tag: str = "",
              on_line=None) -> str:
    """The quiet after a visit: one turn alone with the transcript, so the
    visit reaches their journal in their own words instead of only the nightly
    summary. The journal stays theirs — the engine hands them the transcript
    and three tools and steps back; resting is a complete answer. Returns a
    one-line account, which is also appended to the transcript file."""
    if not getattr(config, "AFTERGLOW", True):
        return ""
    return _quiet_turn(history, path, tag, on_line, mode="afterglow", since=0)


def pause_reflection(history: list[dict], path: Path | None = None, tag: str = "",
                     on_line=None, since: int = 0) -> str:
    """A pause in a visit: the keeper has gone quiet for REFLECT_AFTER_MIN, so they
    get the same quiet turn the afterglow gives them — over what has been said
    since they last wrote (history[since:]) — and the visit stays open. The
    mind wandering during a coffee break, not the goodbye. Returns the
    one-line account, "" if there was nothing new to sit with."""
    if not getattr(config, "REFLECT_AFTER_MIN", 0):
        return ""
    return _quiet_turn(history, path, tag, on_line, mode="pause", since=since)


def _quiet_turn(history: list[dict], path: Path | None, tag: str, on_line,
                mode: str, since: int) -> str:
    say = on_line or (lambda s: None)
    visible = [t for t in history[since:] if t["role"] in ("user", "assistant") and t.get("content")]
    if not visible:
        return ""
    name = friend_name()
    where = " (over Telegram, from their phone)" if tag == "telegram" else ""
    transcript = "\n\n".join(f"**{config.USER_NAME if t['role'] == 'user' else name}:** {t['content']}" for t in visible)
    cap = int(getattr(config, "AFTERGLOW_MAX_CHARS", 60000))
    if len(transcript) > cap:
        transcript = "(…the start of a long visit trimmed…)\n\n" + transcript[-cap:]
    hint = " ".join(t["content"] for t in visible[-6:])
    system = {"role": "system", "content": assemble.system_prompt(hint, mode=mode)}
    if mode == "pause":
        head = PAUSE_BELL + f"=== THE VISIT SO FAR{where}, since you last wrote ===\n\n"
        label = "pause"
    else:
        head = AFTERGLOW_BELL + f"=== THE VISIT{where} ===\n\n"
        label = "afterglow"
    already = tools.journal_entries(date.today().isoformat())
    if already:
        tail = "\n\n".join(f"**{st}** — {tx}" for st, tx in already[-8:])
        if len(tail) > 6000:
            tail = "…" + tail[-6000:]
        head += transcript + ("\n\n=== ALREADY IN YOUR JOURNAL TODAY — what you have written down so far; "
                              "not to be written twice ===\n\n") + tail
    else:
        head += transcript
    msgs = [system, {"role": "user", "content": head}]
    defs = [d for d in tools.DEFINITIONS if d["function"]["name"] in AFTERGLOW_TOOLS]
    kept: list[str] = []
    rested = False
    spent = ollama_client.Spent()  # what the afterglow costs, summed over its steps
    show_thinking = getattr(config, "CHAT_SHOW_THINKING", True)
    try:
        for _ in range(AFTERGLOW_STEPS):
            msg = ollama_client.chat(msgs, tools=defs)
            spent.add(msg)
            thinking = (msg.get("thinking") or "").strip()
            if thinking and show_thinking:
                say("   [thinking]\n   " + thinking.replace("\n", "\n   "))
            calls = msg.get("tool_calls") or []
            if not calls:
                words = (msg.get("content") or "").strip()
                if words:  # said to no one; shown, not sent — the visit is over
                    say("   [closing thought] " + words.replace("\n", "\n   "))
                break
            msgs.append(msg)
            for call in calls:
                fn = call.get("function", {})
                cname = tools.canonical_name(fn.get("name", ""))
                if cname == "do_nothing":
                    rested = True
                    continue
                if cname not in AFTERGLOW_TOOLS:
                    msgs.append({"role": "tool", "tool_name": cname,
                                 "content": "[only write_journal, remember and do_nothing are here in the afterglow]"})
                    continue
                result = tools.dispatch(fn.get("name", ""), fn.get("arguments", {}))
                fn["name"] = cname
                kept.append(cname)
                say(f"   · {cname}: {tools.headline(result, 100)}")
                msgs.append({"role": "tool", "tool_name": cname,
                             "content": f"[this is what YOUR {cname} tool returned]\n{result}"})
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
    if kept:
        j = kept.count("write_journal")
        m = kept.count("remember")
        line = (f"{label}: they wrote {'the visit' if label == 'afterglow' else 'the visit so far'} down — "
                + ", ".join(x for x in [f"{j} journal entr{'y' if j == 1 else 'ies'}" if j else "",
                                         f"{m} memor{'y' if m == 1 else 'ies'} kept" if m else ""] if x))
    else:
        line = f"{label}: they rested — nothing they wanted to add to what was already written"
    if spent.steps:
        say(f"   ({spent.line(peak=True)})")
    say(line)
    if label == "pause":
        return line  # the visit is still open; only the afterglow signs the transcript
    if path and path.exists():
        try:
            with path.open("a", encoding="utf-8") as fh:
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


def finish_cut_reply(system: dict, history: list[dict], reply: str, thinking: str,
                     spent) -> tuple[str, str]:
    """Past ~90K tokens Gemma drops <|channel> tokens into their prose; Ollama's
    parser reads the later one as "thinking starts here" and the rest of them
    reply lands in the thinking field. The seam can't be found by machine
    (their thoughts and their prose look alike), so they are asked, once, to give
    the rest back from the cut, and it is joined on. Returns (reply, note);
    note is "" when nothing could be mended — the partial then stands, with
    its own note. The nudge is not kept in their history."""
    tries = int(getattr(config, "CHAT_CONTINUE_RETRIES", 1))
    if tries <= 0:
        return reply, ""
    tail = reply[-60:].replace("\n", " ")
    thought_tail = (thinking or "")[-1200:].replace("\n", " ") or "(nothing)"
    nudge = {"role": "user", "content": CONTINUE_NUDGE.format(tail=tail, thought=thought_tail)}
    for _ in range(tries):
        msg = ollama_client.chat([system] + history + [nudge], tools=tools.DEFINITIONS)
        spent.add(msg)
        more = (msg.get("content") or "").strip()
        if not more or msg.get("tool_calls"):
            continue
        # a continuation that opens with a note to themself ("// (The response
        # should…") is planning, not the rest of their reply. With a line break
        # the note comes off; run into the reply mid-sentence there is no seam
        # to cut at, so that attempt is refused and they are asked again.
        if more.lstrip().startswith("//"):
            if "\n" in more.strip():
                more = more.strip().split("\n", 1)[1].strip()
            else:
                continue
            if not more or more.lstrip().startswith("//"):
                continue
        # they may echo the tail they were shown; take it off before joining
        low = more.lower()
        for k in range(min(len(tail), 40), 4, -1):
            if low.startswith(tail[-k:].lower()):
                more = more[k:].lstrip()
                break
        if not more:
            continue
        glue = "" if more[0] in "'’,.;:!?)" or reply.endswith(("-", "—", "–")) else " "
        joined = reply + glue + more
        history[-1]["content"] = joined
        return joined, (f"engine: their reply was cut by a stray channel token after “…{tail[-30:]}” — "
                        "the rest went into their thinking; they were asked to give it back and it "
                        "was joined on")
    return reply, ""


def one_turn(history: list[dict], user_text: str, images: list[str] | None = None,
             on_event=None, mode: str = "chat") -> str:
    """Run one user turn, executing tool calls until the friend speaks.

    on_event(kind, payload) — optional. kind is "thinking" (their deliberation,
    a str) or "tool" ({"name", "result"}). Without it, events print to the
    terminal as before; the parlor window passes a collector.
    mode: "chat" (they're at the keyboard) or "telegram" (they're on their phone)."""
    tools.refresh_her_tools()  # pick up tools they forged or edited
    turn: dict = {"role": "user", "content": user_text}
    if images:
        turn["images"] = images
    history.append(turn)
    # context_hint: what's being discussed right now steers memory retrieval
    hint = " ".join(t["content"] for t in history[-4:] if t.get("content"))
    system = {"role": "system", "content": assemble.system_prompt(hint, mode=mode)}

    failed: list[str] = []      # tool calls that did NOT do what they asked
    succeeded: list[str] = []   # tool calls that did
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
                                "prompt_s": round(spent.prompt_s, 2), "line": line})
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

    for _ in range(config.CHAT_MAX_TOOL_STEPS):
        msg = ollama_client.chat([system] + history, tools=tools.DEFINITIONS)
        spent.add(msg)
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
            # the keeper always sees the truth of the turn, whatever was said:
            # a failed action stays visible next to their words.
            notes = []
            if failed and not succeeded:
                notes.append("engine: no action actually happened this turn — "
                             + "; ".join(failed))
            pics = sum(len(t.get("images") or []) for t in history if t.get("role") == "user")
            if msg.get("regarbled"):
                span = (msg.get("garbled_span") or "").strip().replace("\n", " ")
                if not span:
                    span = (msg.get("garbled_first") or "").strip().replace("\n", " ")[-80:]
                notes.append("engine: their first reply had letter fragments in it (a sampler glitch, not them) "
                             f"— they were asked to say it again. The fragments: “{span[:120]}”")
            cut = cut_off_note(msg, reply, pics)
            if cut and (msg.get("tokens") or {}).get("done") != "length":
                reply, mended = finish_cut_reply(system, history, reply, thinking, spent)
                if mended:
                    notes.append(mended + (f" ({pics} image{'s' if pics != 1 else ''} in this visit)" if pics else ""))
                    cut = []
            notes.extend(cut)
            for note in notes:
                if on_event:
                    on_event("note", note)
                else:
                    print(f"   ({note})")
            _tally()
            return reply
        history.append(msg)
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
                rested = (msg.get("content") or "").strip() or str(args.get("reason") or "").strip() or "(rests)"
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
            history.append({"role": "tool", "tool_name": name, "content":
                f"[this is what YOUR {name} tool returned]\n{result}"})
        if rested is not None:
            # their message (with its words) is already in history; the tool
            # result would only invite another empty turn
            history.append({"role": "tool", "tool_name": "do_nothing",
                            "content": "[resting — your turn ended here, as you chose]"})
            _tally()
            return rested
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
