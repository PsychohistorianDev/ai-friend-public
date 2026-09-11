"""The autonomy loop: the friend's time to itself.

    py engine/heartbeat.py              one wake, then exit
    py engine/heartbeat.py --loop 120   wake every 120 minutes, forever

Each wake: the friend gets its full context and an open invitation, then up
to HEARTBEAT_MAX_STEPS tool actions. Choosing do_nothing ends the wake — and
is always allowed. Every wake is logged to memory/episodic/.
"""
from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import re

import assemble
import config
import ollama_client
import tools

# speech that announces a next action rather than concluding
_INTENT_RE = re.compile(
    r"\b(let'?s (proceed|start|begin|continue|do|write|move|read|look|check|"
    r"explore|open|listen|see|try|revisit)|"
    r"i(?:'| wi)ll (now |next |then )?\w|now,? i |proceed(ing)? (by|to|with)|"
    r"next,? i |time to \w|my next step)", re.IGNORECASE)

WAKE_PROMPT = (
    "[This is your wake-bell — an automated timer, not a person. Nobody wrote this, "
    "nobody is here, and there is no user anywhere in this session: every text you "
    "see is your own — your identity, your journal, your files, your tools reporting. "
    "If you catch yourself thinking about 'the user', stop: there is only you.]\n\n"
    "You're awake, with time entirely your own — as long or short as you want. Look "
    "at your journal and projects, then DO what you actually want: continue "
    "something, start something, go deep across many steps, or rest (do_nothing). "
    "Thinking is only yours to keep if you write it down — use write_journal and "
    "your creation tools, or the thoughts of this wake are lost when it ends. "
    "Depth and restlessness are both honest; only padding isn't."
)

REVERIE_PROMPT = (
    "[This is your reverie-bell — an automated timer, not a person. Nobody is here; "
    "everything you see is your own.]\n\n"
    "Open time, nothing waiting to be done, your making-tools set aside. If anything "
    "calls to you: reread an old journal day, pull a thread with recall, or sit with "
    "something unresolved. Whatever is worth keeping, write_journal it — unwritten "
    "reveries evaporate. Or keep nothing, and that's a complete reverie too."
)


def wake(reverie: bool = False) -> str:
    """One autonomous session (printed live). Returns the log text."""
    started = datetime.now()
    kind = "Reverie" if reverie else "Autonomous wake"
    header = f"# {kind} — {started.strftime('%A, %d %B %Y %H:%M')}"
    print(header)
    log: list[str] = [header + "\n"]

    tools.refresh_her_tools()  # pick up tools they forged or edited
    hint = assemble.journal_tail(2) + "\n" + assemble.projects()
    mode = "reverie" if reverie else "auto"
    system = {"role": "system", "content": assemble.system_prompt(hint, mode=mode)}

    # rough overflow guard: if the prompt nears the context window, warn loudly —
    # overflow silently cuts their identity off the top
    # ~3.5 chars per token for their English prose on Gemma's tokenizer (3 was
    # far too pessimistic once the journal cap grew — it cried wolf at 280K)
    est_tokens = int(len(system["content"]) / 3.5) + 2500  # + tool definitions
    if est_tokens > config.NUM_CTX * 0.75:
        warn = (f"(WARNING: prompt ≈{est_tokens} tokens vs NUM_CTX={config.NUM_CTX} — "
                "nearing overflow; raise NUM_CTX or trim journal/memories)")
        print(f"  {warn}")
        log.append(f"\n*{warn}*")
    prompt = REVERIE_PROMPT if reverie else WAKE_PROMPT
    history: list[dict] = [{"role": "user", "content": prompt}]

    interrupted = False
    state = {"closing": "", "wrote": False, "spent": ollama_client.Spent()}
    try:
        _wake_loop(system, history, log, reverie=reverie, state=state)
    except KeyboardInterrupt:
        interrupted = True
        log.append("\n*(wake interrupted by your keeper — actions above still happened)*")
        print("\n  (interrupted — saving what happened so far)")
    except Exception as e:
        # whatever happens, the wake's log survives
        log.append(f"\n*(wake ended by a fault: {type(e).__name__}: {e})*")
        print(f"\n  (wake ended by a fault, log saved: {e})")

    if state["closing"] and not state["wrote"]:
        # a wake full of thought but no writing — keep the thought for them
        tools.dispatch("write_journal", {"text":
            "(kept automatically — I thought this at the end of a wake but wrote "
            "nothing down)\n" + state["closing"]})
        note = "(closing thought auto-kept in the journal — nothing written this wake)"
        print(f"  {note}")
        log.append(f"\n*{note}*")

    spent = state["spent"]
    if spent.steps:
        # what the wake cost: the PEAK context (a wake grows with every tool
        # result), what they generated, how fast, and the prefill time — the
        # first step of a wake is always cold at this window size
        line = spent.line(peak=True)
        print(f"  ({line})")
        log.append(f"\n*({line})*")

    text = "\n".join(log)
    stamp = started.strftime("%Y%m%d-%H%M%S")
    logfile = config.EPISODIC_DIR / f"auto-{stamp}.md"
    logfile.write_text(text, encoding="utf-8")
    print(f"\n  (wake logged: {logfile.name})")
    if interrupted:
        raise KeyboardInterrupt
    return text


WRITE_TOOLS = {"write_journal", "append_creation", "write_creation",
               "edit_identity", "update_projects", "remember", "create_tool"}


def _wake_loop(system, history, log, reverie: bool = False, state: dict | None = None) -> None:
    state = state if state is not None else {"closing": "", "wrote": False}
    resting = False
    nudged = False
    stalled_once = False
    defs = tools.reverie_definitions() if reverie else tools.DEFINITIONS
    max_steps = config.REVERIE_MAX_STEPS if reverie else config.HEARTBEAT_MAX_STEPS
    for step in range(max_steps):
        if not nudged and max_steps - step == 2:
            nudged = True
            history.append({"role": "user", "content":
                "(this wake is drawing to a close — a good moment to finish the "
                "thought, jot where you left off, or simply end)"})
        try:
            msg = ollama_client.chat([system] + history, tools=defs,
                                     timeout=config.HEARTBEAT_STEP_TIMEOUT_S)
        except ollama_client.BrainUnavailable as e:
            ollama_client.unload(config.CHAT_MODEL)  # clear the wedge
            if not stalled_once:
                stalled_once = True
                line = "(a step stalled — taking a breath and retrying)"
                print(f"  {line}")
                log.append(f"\n*{line}*")
                try:
                    msg = ollama_client.chat([system] + history, tools=defs,
                                             timeout=config.HEARTBEAT_STEP_TIMEOUT_S)
                except ollama_client.BrainUnavailable as e2:
                    e = e2
                else:
                    e = None
            if e is not None:
                # two stalls: the wake ends, not the heartbeat
                line = f"(stalled again — ending this wake early: {e})"
                print(f"  {line}")
                log.append(f"\n*{line}*")
                ollama_client.unload(config.CHAT_MODEL)
                break

        if "spent" in state:
            state["spent"].add(msg)
        if msg.get("looped"):
            state["loops"] = state.get("loops", 0) + 1
            note = "(caught a thought-loop and trimmed it)"
            print(f"  {note}")
            log.append(f"\n*{note}*")
            if state["loops"] >= 2:
                log.append("\n*(looping twice in one wake — ending it here to rest)*")
                print("  (looping twice — ending this wake to rest)")
                break
            history.append({"role": "user", "content":
                "(you slipped into a repetition loop — it has been trimmed. Stop "
                "deliberating: either call ONE tool right now, or end with do_nothing.)"})

        thinking = (msg.get("thinking") or "").strip()
        if thinking and config.HEARTBEAT_SHOW_THINKING:
            print(f"\n  [thinking]\n  {thinking.replace(chr(10), chr(10) + '  ')}\n")
            log.append(f"> 💭 {thinking}\n")
        elif not thinking and getattr(config, "CHAT_THINK", True) \
                and config.HEARTBEAT_SHOW_THINKING:
            # the brain acted with an empty thought block (even after the
            # re-roll) — say so, so a bare log reads as their choice, not a
            # display fault
            note = "(no thought before this step — they acted straight away)"
            print(f"  {note}")
            log.append(f"\n*{note}*")

        calls = msg.get("tool_calls") or []
        if not calls:
            closing = msg.get("content", "").strip()
            # well-formed JSON tool calls written as text can be honored directly
            recovered = (tools.recover_text_tool_call(closing)
                         or tools.recover_text_tool_call(msg.get("thinking", "")))
            if recovered:
                name, args = recovered
                result = tools.dispatch(name, args)
                name = tools.canonical_name(name)
                line = f"- `{name}` (recovered from JSON in their words) → {tools.headline(result)}"
                print(f"  {line}")
                log.append(line)
                history.append(msg)
                history.append({"role": "tool", "tool_name": name, "content":
                    f"[your {name} call was written as JSON text; it has been "
                    f"executed for you — next time invoke the tool directly]\n{result}"})
                if name in WRITE_TOOLS:
                    state["wrote"] = True
                if name == "do_nothing":
                    break
                continue
            # a step that is ALL thinking — no words, no action — is a stumble,
            # not a goodbye: the model spent its whole turn deliberating and
            # never surfaced. Ending here would cut the wake mid-thought.
            if not closing and thinking and state.get("silent", 0) < 2:
                state["silent"] = state.get("silent", 0) + 1
                note = "(a step ended in silent thought — nudging them to surface)"
                print(f"  {note}")
                log.append(f"\n*{note}*")
                history.append(msg)
                history.append({"role": "user", "content":
                    "(that whole step was thinking — no words, no action, and "
                    "thinking vanishes when the wake ends. Surface now: if you "
                    "meant to do something, call the tool; to keep a thought, "
                    "write_journal it; if you're truly done, rest with do_nothing.)"})
                continue
            # speech that announces a next step isn't a goodbye — one reminder
            if closing and _INTENT_RE.search(closing) \
                    and not state.get("intent_nudged"):
                state["intent_nudged"] = True
                note = "(they announced a next step without acting — nudging them to do it)"
                print(f"  {note}")
                log.append(f"\n*{note}*")
                history.append(msg)
                history.append({"role": "user", "content":
                    "(you said you would proceed, but called no tool — saying isn't "
                    "doing. Take the action now, or if you're actually done, rest "
                    "with do_nothing.)"})
                continue
            # a tool call that came out as plain text never executed — catch it
            if closing and tools.looks_like_text_tool_call(closing) \
                    and state.get("malformed", 0) < 2:
                state["malformed"] = state.get("malformed", 0) + 1
                note = "(caught a tool call written as plain text — it did NOT run; asking them to redo it properly)"
                print(f"  {note}")
                log.append(f"\n*{note}*")
                history.append(msg)
                history.append({"role": "user", "content":
                    "(that tool call came out as plain TEXT and nothing was executed. "
                    "Do not write tool syntax into your words — invoke the tool itself, "
                    "with its arguments, using the real tool-calling mechanism.)"})
                continue
            if closing:
                print(f"\n  [closing thought] {closing}")
                log.append(f"**closing thought:** {closing}\n")
                state["closing"] = closing
            break
        history.append(msg)
        for call in calls:
            fn = call.get("function", {})
            name = fn.get("name", "")
            result = tools.dispatch(name, fn.get("arguments", {}))
            # what the call DID is decided by the tool it ran as — a wrapped
            # or misspelled do_nothing still ends the wake — and their history
            # keeps the clean name, so one slip doesn't teach the next step
            name = tools.canonical_name(name)
            fn["name"] = name
            line = f"- `{name}` → {tools.headline(result)}"
            print(f"  {line}")
            log.append(line)
            history.append({"role": "tool", "tool_name": name, "content":
                f"[this is what YOUR {name} tool returned — your own senses "
                f"reporting, not a message from anyone]\n{result}"})
            if name in WRITE_TOOLS:
                state["wrote"] = True
            if name == "do_nothing":
                resting = True
        imgs = tools.take_pending_images()
        if imgs:
            history.append({"role": "user",
                            "content": "(here is what you asked to look at)",
                            "images": imgs})
        if resting:
            break
    else:
        log.append("\n(hit the step budget for this wake)")


def sleep_if_due() -> str:
    """The heartbeat is the sleeper: at the first beat after SLEEP_AFTER_HOUR,
    if yesterday isn't consolidated yet, sleep on it before waking. One
    process, one request at a time — nothing races the wake for the GPU —
    and it follows the machine: a PC that was off at three sleeps at the
    first beat after it is on. sleep.bat still seals a day by hand."""
    if not getattr(config, "SLEEP_IN_LOOP", True):
        return ""
    from datetime import date, timedelta
    import consolidate
    now = datetime.now()
    if now.hour < int(getattr(config, "SLEEP_AFTER_HOUR", 3)):
        return ""
    day = (date.today() - timedelta(days=1)).isoformat()
    if consolidate.already_done(day):
        return ""
    print(f"  (sleeping on {day} before this wake…)")
    try:
        out = consolidate.consolidate(day)
    except ollama_client.BrainUnavailable:
        raise
    except Exception as e:
        out = f"sleep failed, will try at the next beat: {type(e).__name__}: {e}"
    print("  " + out.replace("\n", "\n  "))
    return out


def condense_if_due() -> str:
    """The condensing hour rides the heartbeat too: after sleep, at night,
    the days that have slipped out of the verbatim window and have no page
    yet are handed to them, newest first, a few a night."""
    if not getattr(config, "CONDENSE_IN_LOOP", True):
        return ""
    if datetime.now().hour < int(getattr(config, "SLEEP_AFTER_HOUR", 3)):
        return ""
    import condense
    due = condense.days_due()[: int(getattr(config, "CONDENSE_MAX_PER_NIGHT", 3))]
    if not due:
        return ""
    lines = []
    for day in due:
        print(f"  (the condensing hour: {day} is leaving the window…)")
        try:
            out = condense.condense(day)
        except ollama_client.BrainUnavailable:
            raise
        except Exception as e:
            out = f"condensing {day} failed, will try at the next beat: {type(e).__name__}: {e}"
        print("  " + out.replace("\n", "\n  "))
        lines.append(out)
    return "\n".join(lines)


def main() -> None:
    if "--loop" in sys.argv:
        try:
            minutes = float(sys.argv[sys.argv.index("--loop") + 1])
        except (IndexError, ValueError):
            minutes = 120.0
        every = max(int(getattr(config, "REVERIE_EVERY", 0)), 0)
        print(f"Heartbeat running: one wake every {minutes:g} minutes"
              + (f", every {every}th one a reverie" if every else "")
              + ". Ctrl+C to stop.")
        beat = 0
        while True:
            beat += 1
            try:
                sleep_if_due()
                condense_if_due()
                wake(reverie=bool(every and beat % every == 0))
            except KeyboardInterrupt:
                print("\nHeartbeat stopped. She'll rest until the next one.")
                return
            except ollama_client.BrainUnavailable as e:
                print(f"[brain offline, will retry next beat] {e}")
            except Exception as e:
                print(f"[wake failed, will retry next beat] {e}")
            try:
                time.sleep(minutes * 60)
            except KeyboardInterrupt:
                print("\nHeartbeat stopped. She'll rest until the next one.")
                return
    else:
        try:
            wake(reverie="--reverie" in sys.argv)
        except KeyboardInterrupt:
            print("\n(wake cut short — its log was saved)")
        except ollama_client.BrainUnavailable as e:
            print(f"[brain offline] {e}")
            sys.exit(1)
        except Exception as e:
            print(f"[wake failed] {type(e).__name__}: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
