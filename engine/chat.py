"""The door: talk with the friend.

    py engine/chat.py

Commands inside the chat:
    /quit   end the visit (transcript is saved automatically)
    /new    start a fresh conversation (saves the current one first)

Every conversation is saved to memory/episodic/ and becomes part of the
friend's past at the next consolidation (engine/consolidate.py).
"""
from __future__ import annotations

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
    for _ in range(config.CHAT_MAX_TOOL_STEPS):
        msg = ollama_client.chat([system] + history, tools=tools.DEFINITIONS)
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
            return reply
        history.append(msg)
        for call in calls:
            fn = call.get("function", {})
            name = fn.get("name", "")
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
        imgs = tools.take_pending_images()
        if imgs:
            history.append({"role": "user",
                            "content": "(here is what you asked to look at)",
                            "images": imgs})

    history.append({"role": "assistant", "content": "(I got lost in my tools — say that again?)"})
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
