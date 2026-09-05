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
from datetime import datetime
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


def save_transcript(turns: list[dict]) -> Path | None:
    """Write the visible conversation to episodic memory."""
    visible = [t for t in turns if t["role"] in ("user", "assistant") and t.get("content")]
    if not visible:
        return None
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    f = config.EPISODIC_DIR / f"chat-{stamp}.md"
    lines = [f"# Conversation — {datetime.now().strftime('%A, %d %B %Y %H:%M')}\n"]
    name = friend_name()
    for t in visible:
        who = config.USER_NAME if t["role"] == "user" else name
        lines.append(f"**{who}:** {t['content']}\n")
    f.write_text("\n".join(lines), encoding="utf-8")
    return f


def one_turn(history: list[dict], user_text: str, images: list[str] | None = None,
             on_event=None) -> str:
    """Run one user turn, executing tool calls until the friend speaks.

    on_event(kind, payload) — optional. kind is "thinking" (their deliberation,
    a str) or "tool" ({"name", "result"}). Without it, events print to the
    terminal as before; the parlor window passes a collector."""
    tools.refresh_her_tools()  # pick up tools they forged or edited
    turn: dict = {"role": "user", "content": user_text}
    if images:
        turn["images"] = images
    history.append(turn)
    # context_hint: what's being discussed right now steers memory retrieval
    hint = " ".join(t["content"] for t in history[-4:] if t.get("content"))
    system = {"role": "system", "content": assemble.system_prompt(hint, mode="chat")}

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
                    "reply turns slow. A good moment for /new: the visit is saved, she "
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
            if failed and not succeeded:
                note = ("engine: no action actually happened this turn — "
                        + "; ".join(failed))
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
                    print(f"(saved {f.name})")
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
            print(f"\n(conversation saved: {f.name} — it becomes memory at the next consolidation)")


if __name__ == "__main__":
    main()
