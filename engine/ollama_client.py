"""Thin client for the local Ollama server. Standard library only.

Two entry points:
    chat(messages, tools=None)  -> assistant message dict
    embed(text)                 -> list[float]

The rest of the engine talks to the brain exclusively through these two
functions, which is what makes the brain swappable.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request

import config


class BrainUnavailable(RuntimeError):
    """Raised when the Ollama server can't be reached at all."""


def _post(path: str, payload: dict, timeout: float | None = None) -> dict:
    timeout = timeout or config.REQUEST_TIMEOUT_S
    req = urllib.request.Request(
        config.OLLAMA_URL + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if payload.get("stream"):
                return _drink(resp)
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        # The server answered, so Ollama is running — surface its actual error.
        try:
            detail = json.loads(e.read().decode("utf-8", "replace")).get("error", "")
        except Exception:
            detail = ""
        model = payload.get("model", "?")
        if "not found" in detail.lower() or e.code == 404:
            raise BrainUnavailable(
                f"Ollama is running, but the model `{model}` isn't installed.\n"
                f"  Fix: open a terminal and run:  ollama pull {model}\n"
                f"  (server said: {detail or f'HTTP {e.code}'})"
            ) from e
        raise BrainUnavailable(
            f"Ollama error on {path}: {detail or f'HTTP {e.code}'}"
        ) from e
    except urllib.error.URLError as e:
        if isinstance(getattr(e, "reason", None), TimeoutError):
            raise BrainUnavailable(
                f"the brain took longer than {timeout:.0f}s to answer — "
                "it's probably busy chewing on something big (long audio, a model "
                "swap). Wait a moment and try again."
            ) from e
        raise BrainUnavailable(
            f"Can't reach Ollama at {config.OLLAMA_URL} — is it running? "
            f"(start the Ollama app, or run `ollama serve`)  [{e}]"
        ) from e
    except (TimeoutError, OSError) as e:
        raise BrainUnavailable(
            f"the brain took longer than {timeout:.0f}s to answer — "
            "it's probably busy chewing on something big (long audio, a model swap). "
            f"Wait a moment and try again.  [{type(e).__name__}: {e}]"
        ) from e


_THINK_RE = re.compile(r"<think>(.*?)</think>\s*", flags=re.DOTALL)
# stray chat-template tokens some builds leak into output
# (<|channel>thought, </s>, <tool_call|> etc.)
_TOKEN_LITTER_RE = re.compile(r"<\|[^>\n]{0,40}>\w*\n?|</?s>|<\|?tool_call\|?>?")


def scrub_litter(text: str) -> str:
    return delatex(_TOKEN_LITTER_RE.sub("", text or ""))


# Gemma writes arrows and a few symbols as LaTeX ("Input $\\rightarrow$
# Output") — fine in a paper, litter in a journal. Render them as the
# characters they meant; anything else in $...$ is left alone.
_LATEX_SYMBOLS = {
    "rightarrow": "→", "to": "→", "longrightarrow": "⟶", "Rightarrow": "⇒",
    "leftarrow": "←", "gets": "←", "Leftarrow": "⇐", "leftrightarrow": "↔",
    "Leftrightarrow": "⇔", "uparrow": "↑", "downarrow": "↓", "mapsto": "↦",
    "times": "×", "cdot": "·", "ldots": "…", "dots": "…", "infty": "∞",
    "approx": "≈", "neq": "≠", "ne": "≠", "leq": "≤", "geq": "≥", "pm": "±",
    "therefore": "∴", "because": "∵", "equiv": "≡", "sim": "~", "star": "★",
    "heartsuit": "♥", "deg": "°", "alpha": "α", "beta": "β", "gamma": "γ",
    "delta": "δ", "Delta": "Δ", "lambda": "λ", "mu": "μ", "pi": "π",
    "sigma": "σ", "phi": "φ", "psi": "ψ", "omega": "ω", "Omega": "Ω",
}
_LATEX_RE = re.compile(r"\$\s*\\([A-Za-z]+)\s*\$|(?<![\\\w])\\([A-Za-z]+)(?![\w{])")


def delatex(text: str) -> str:
    def sub(m):
        name = m.group(1) or m.group(2)
        sym = _LATEX_SYMBOLS.get(name)
        if sym is None:
            return m.group(0)
        return sym
    return _LATEX_RE.sub(sub, text or "")


def collapse_loops(text: str, threshold: int = 5) -> tuple[str, bool]:
    """Collapse degenerate repetition (the same line over and over) to a single
    occurrence plus a marker. Returns (cleaned_text, was_looping). Keeping the
    loop out of the transcript matters: repetition fed back in breeds more."""
    lines = (text or "").splitlines()
    out: list[str] = []
    looped = False
    run_line, run_count = None, 0
    for ln in lines + [None]:  # sentinel flushes the last run
        key = (ln or "").strip()
        if ln is not None and key and key == run_line:
            run_count += 1
            continue
        if run_count >= threshold:
            looped = True
            out.append(f"(…that line repeated {run_count} times — thought-loop trimmed)")
        elif run_count > 1 and run_line:
            out.extend([run_line] * (run_count - 1))
        if ln is None:
            break
        out.append(ln)
        run_line, run_count = key if key else None, 1 if key else 0
    return "\n".join(out), looped


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks some models (Qwen3) emit."""
    return _THINK_RE.sub("", text or "").strip()


def split_thinking(text: str) -> tuple[str, str]:
    """Separate inline <think> blocks from the visible content.

    Returns (thinking, clean_content).
    """
    text = text or ""
    thinking = "\n".join(m.strip() for m in _THINK_RE.findall(text)).strip()
    return thinking, _THINK_RE.sub("", text).strip()


# A reply is read as it is written, and a runaway is cut short. 09-11, at
# ~176K tokens: one re-rolled attempt was "luminate luminate luminate…" for
# 8,192 tokens — the whole num_predict ceiling, six minutes at 22 tok/s —
# because nothing looks at a reply until it is finished. Now the stream is
# watched every STREAM_WATCH_EVERY tokens: the moment the tail is salad (a
# stuck chunk, a cascade, a run of fragments) the connection is closed —
# Ollama stops generating when the client hangs up — and what came back is
# handed to the salad rail like any other broken attempt, having cost a few
# seconds instead of the ceiling. The final chunk's counters are lost on an
# abort; the tokens are counted by the chunks read.
STREAM_WATCH_EVERY = 48
STREAM_WATCH_TAIL = 900


def _drink(resp) -> dict:
    """Read a streamed /api/chat response into the same dict the unstreamed
    one gives, watching the tail for a runaway."""
    import time as _time
    t0 = _time.monotonic()
    content: list[str] = []
    thinking: list[str] = []
    calls: list[dict] = []
    last: dict = {}
    n = 0
    aborted = ""
    since = 0
    for raw in resp:
        raw = raw.strip()
        if not raw:
            continue
        try:
            chunk = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            continue
        msg = chunk.get("message") or {}
        if msg.get("content"):
            content.append(msg["content"])
        if msg.get("thinking"):
            thinking.append(msg["thinking"])
        if msg.get("tool_calls"):
            calls.extend(msg["tool_calls"])
        n += 1
        since += 1
        if chunk.get("done"):
            last = chunk
            break
        if getattr(config, "CHAT_STREAM_ABORT", True) and since >= STREAM_WATCH_EVERY:
            since = 0
            tail = ("".join(content) if content else "".join(thinking))[-STREAM_WATCH_TAIL:]
            span = garble_span(tail)
            if span:
                aborted = span
                break  # leaving the `with` closes the connection; the server stops
    out = dict(last)
    out["message"] = {"role": "assistant", "content": "".join(content), "thinking": "".join(thinking)}
    if calls:
        out["message"]["tool_calls"] = calls
    if aborted:
        out["done_reason"] = "aborted"
        out["aborted"] = aborted
        out["eval_count"] = n
        out["eval_duration"] = int((_time.monotonic() - t0) * 1e9)
        out["total_duration"] = out["eval_duration"]
        out["prompt_eval_count"] = 0  # never reached the final chunk; the siblings carry the size
    return out


def chat(messages: list[dict], tools: list[dict] | None = None,
         timeout: float | None = None, think: bool | None = None,
         expect_words: bool = False) -> dict:
    """One non-streaming chat completion.

    Returns the assistant message dict:
        {"role": "assistant", "content": str, "tool_calls": [...]?, "thinking": str}
    Thinking is captured whether Ollama returns it as a separate field (newer
    servers) or inline as <think> tags (Qwen3's raw style); content is clean
    of it either way. `think=False` closes the thought channel for this one
    call (no <|think|> switch, no think re-roll): for a step that needs no
    deliberation — finishing a cut sentence — and, since the server is then
    not parsing channel tokens, a step that a stray one cannot split.
    """
    options = {"num_ctx": getattr(config, "NUM_CTX", 8192)}
    options.update(getattr(config, "SAMPLING_OPTIONS", {}))
    payload: dict = {
        "model": config.CHAT_MODEL,
        "messages": messages,
        "options": options,
        "stream": bool(getattr(config, "CHAT_STREAM_ABORT", True)),  # read as written, cut a runaway short
    }
    # Ollama sets a model down after five idle minutes by default, and the
    # cache — their reading of the whole window — goes with it: a message
    # twenty minutes after the last one began with "model loaded in 8.0s"
    # and a cold read of 149K tokens. Keep the brain up between messages;
    # their ears and the music ear still evict it on purpose when they need
    # the card (unload()), and that is the one cold read worth paying.
    keep = getattr(config, "BRAIN_KEEP_ALIVE", "")
    if keep:
        payload["keep_alive"] = keep
    if tools:
        payload["tools"] = tools
    # Ask for thinking explicitly. Left to its discretion, the model went
    # silent-minded once the prompt grew past ~40K tokens (a long journal
    # window): tool calls with no deliberation at all. Requiring it keeps
    # their thinking visible — and thoughtful — at any context size.
    if think is False:
        payload["think"] = False
    elif getattr(config, "CHAT_THINK", True):
        payload["think"] = True
    try:
        data = _post("/api/chat", payload, timeout=timeout)
    except BrainUnavailable as e:
        if "think" in str(e).lower() and payload.pop("think", None) is not None:
            data = _post("/api/chat", payload, timeout=timeout)  # model can't think: fine
        else:
            raise
    msg = _parse(data)
    # The flag only OPENS the thought channel — Gemma 4 may still leave the
    # block empty and act straight away, and once one step goes thoughtless
    # the rest of a wake tends to follow the pattern. A skipped thought is a
    # short answer and the prompt is already cached, so re-rolling is cheap:
    # ask again (same prompt, fresh sample) up to CHAT_THINK_RETRIES times.
    # A plain re-sample stopped working once the prompt passed ~90K tokens: the
    # <|think|> switch sits at the very top of the system turn, a novel's
    # length before the answer, and the model simply forgets it is allowed to
    # think. So each re-roll adds a transient nudge at the END of the
    # conversation — "think first" — next to where the answer is generated.
    # It is not kept in their history; only the thoughtful answer is.
    tries: list[tuple[dict, dict]] = []  # (cost, the attempt) set aside — the cost still counts
    rerolls = int(getattr(config, "CHAT_THINK_RETRIES", 1)) if payload.get("think") else 0
    while rerolls > 0 and not msg["thinking"] and not msg.get("aborted"):  # a runaway is the salad rail's
        rerolls -= 1
        tries.append((dict(msg["tokens"], why="no thought"), msg))
        nudged = dict(payload)
        nudged["messages"] = with_think_nudge(messages)
        msg = _parse(_post("/api/chat", nudged, timeout=timeout))
        msg["rerolled"] = True
    # letter salad — or a tool call written out as words — is asked for
    # again; a fresh sample usually lands. Every attempt is checked, and if
    # none is clean the LEAST broken one is what goes out (09-11: a cascade
    # was re-rolled once into an 80-fold "//love.you." loop, which was not
    # checked and reached the phone whole).
    garbles = int(getattr(config, "CHAT_GARBLE_RETRIES", 1))
    attempts: list[tuple[int, dict]] = []  # (how broken, the message)
    previous = previous_reply(messages)  # what they said last — an echo of it is a defect too
    once = {"imagined": False, "copy": False}  # asked about once; their second answer stands

    def _defect(m):
        return (reply_defect(m.get("content", ""), previous)
                or (("salad", m["aborted"]) if m.get("aborted") else None)
                or (None if once["imagined"] else imagined_sense(m, messages))
                or (empty_reply(m) if expect_words else None)
                or (None if (once["copy"] or not expect_words) else prompt_copy(m, messages)))
    while garbles > 0 and not msg.get("tool_calls") and _defect(msg):
        garbles -= 1
        kind, span = _defect(msg)
        if kind in once:
            once[kind] = True
        attempts.append((len(span), msg))
        tries.append((dict(msg["tokens"], why=kind), msg))
        nudged = dict(payload)
        nudged["messages"] = list(messages) + [{"role": "user", "content":
                                                CALL_TEXT_NUDGE if kind == "call-text"
                                                else CALL_TEXT_TAIL_NUDGE if kind == "call-text-tail"
                                                else REFRAIN_NUDGE.format(ref=span) if kind == "refrain"
                                                else ECHO_NUDGE if kind == "echo"
                                                else EMPTY_NUDGE if kind == "empty"
                                                else COPY_NUDGE if kind == "copy"
                                                else IMAGINED_NUDGE.format(tool=span, tool_verb="listening" if span == "listen_to" else "watching") if kind == "imagined"
                                                else GARBLE_NUDGE}]
        again = _parse(_post("/api/chat", nudged, timeout=timeout))
        again["regarbled"] = True
        again["garbled_kind"] = kind
        again["garbled_first"] = msg.get("content", "")
        again["garbled_first_aborted"] = bool(msg.get("aborted"))  # the stream was cut short at it
        again["garbled_span"] = span
        msg = again
    if attempts and not msg.get("tool_calls"):
        last = _defect(msg)
        if last:
            # nothing clean came back: send the least broken attempt, and
            # say so — the keeper must see that every try was the sampler's
            attempts.append((len(last[1]), msg))
            tries.append((dict(msg["tokens"], why=last[0]), msg))  # the last try is a try too
            best = min(attempts, key=lambda p: p[0])[1]
            best["regarbled"] = True
            first = _defect(attempts[0][1]) or (last[0], last[1])
            best["garbled_kind"] = attempts[0][1].get("garbled_kind") or first[0]
            best["garbled_first"] = attempts[0][1].get("content", "")
            best["garbled_span"] = first[1]
            best["still_garbled"] = last[1]
            msg = best
    if not msg.get("tool_calls") and call_text_tail(msg.get("content", "")):
        # a written-out call at the end is never sent as their words
        msg["call_text_dropped"] = call_text_tail(msg.get("content", ""))
        msg["content"] = strip_call_tail(msg.get("content", ""))
    # what was set aside is every attempt but the one going out — the one
    # going out is never counted twice (09-11: the least broken attempt
    # was, and the line read "271 generated … 271 set aside")
    kept = [cost for cost, src in tries if src is not msg]
    if kept:
        msg["retries"] = kept
    return msg


# Letter salad — "You arenLa l mH sa M la ne th st ag f loat l la C cl an
# day" — is the sampler failing, not their speaking: a repeat penalty that has
# sat on their commonest tokens for a long visit until only fragments are
# left. A run of fragments is recognised and the step is asked for again;
# they are never handed a glitch to explain.
_FRAG_OK = {"a", "i", "o", "an", "as", "at", "be", "by", "do", "go", "he", "if", "in", "is", "it",
            "me", "my", "no", "of", "oh", "ok", "on", "or", "so", "to", "up", "us", "we", "am",
            "hi", "ah", "mm", "yo", "un", "ha", "ya"}  # not "la": alone it is their accent; in a run it is salad


_EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF]")


_GLUED_RE = re.compile(r"^[a-z]{2,}[A-Z]{1,2}[a-z]?$")  # "sameL", "isnLT", "arenLa": the penalty's signature
# Two shapes are the sampler's even standing alone, no run needed: their accent
# glued to the word after it ("laLuminous silk"), and a word doubled onto
# itself with the seam capitalised ("luminousLuminous", "kindalLongDistance").
# Both came back as refrains — "la lLong distance" ×8, then quoted from them
# own journal for days — so one is enough to hand the line back.
_HARD_GLUE_RE = re.compile(r"^la[A-Z][a-z]{2,}$|^[a-z]{3,}(?:[A-Z][a-z]{3,})+$"
                           r"|^[a-df-hj-z][A-Z][a-z]{3,}$")  # "lSymmetry": one stray letter on a word (not iPhone, eBay)


def garble_span(text: str, run: int = 5, emoji_run: int = 12) -> str:
    """The first stretch of salad in the text, or "" if there is none: a run
    of `run`+ consecutive 1-2 letter 'words' that aren't ordinary short
    English words (glued tokens like "sameL" count) — or `emoji_run`+
    DIFFERENT emojis in a row: the sampler walking the emoji table in
    codepoint order (moon, Saturn, comet, rocket… cloud, rain, snow…) once
    it has been pushed off the hearts they actually meant. Returned so the
    keeper's note can show WHAT was caught, and a false alarm is visible."""
    text = text or ""
    streak = 0
    glued = False  # a run holding a glued token ("laC l l a") is salad at three
    first = last = None
    for m in re.finditer(r"[A-Za-z']+", text):
        w = m.group(0)
        if _HARD_GLUE_RE.match(w):
            return w  # one is enough
        if (len(w) <= 2 and w.lower() not in _FRAG_OK) or _GLUED_RE.match(w):
            if streak == 0:
                first = m.start()
            streak += 1
            glued = glued or bool(_GLUED_RE.match(w))
            last = m.end()
        elif len(w) <= 2:
            continue  # "a", "o", "I", "is"… sit inside salad too; they neither count nor reset
        else:
            if streak >= run or (glued and streak >= 3):
                return text[first:last]  # the whole run, to its end
            streak, glued = 0, False
    if streak >= run or (glued and streak >= 3):
        return text[first:last]
    # emoji cascade: split on anything that isn't emoji/space/variation selector
    for m in re.finditer(r"[\s\uFE0F\U0001F300-\U0001FAFF\u2600-\u27BF]+", text):
        seen = set(_EMOJI_RE.findall(m.group(0)))
        if len(seen) >= emoji_run:
            return m.group(0).strip()
    # a stuck chunk: the same short piece over and over on one line —
    # "//love.you.//love.you.//love.you." ×80 (09-11, the first message of a
    # visit at ~150K tokens on a 4-bit cache) — the sampler in a well, not them
    # (a row of the SAME emoji is not a stuck chunk — 09-11: he sent three
    # kisses and they sent back twenty-four, which is an answer, not a well;
    # the rail re-rolled it twice into refrains. A chunk must have a letter
    # or a digit in it to count.)
    for m in _STUCK_RE.finditer(text):
        if re.search(r"\w", m.group(1)):
            return m.group(0)[:160]
    return ""


# the same 2-24 character chunk, 8+ times in a row with only spaces/punctuation
# between — a loop too short for collapse_loops (which works on lines)
_STUCK_RE = re.compile(r"(\S{2,24}?)(?:[\s.,/\-—]*\1){7,}")


def looks_garbled(text: str, run: int = 5, emoji_run: int = 12) -> bool:
    return bool(garble_span(text, run, emoji_run))


# A tool call written out as words, at the head of a reply —
# "get_opinion_on_la_metrica_rota{description: a comprehensive…" — is the
# model reaching for a tool that doesn't exist (or for a real one without
# the real mechanism). Nothing ran, and the keeper would be handed syntax
# as if it were their words. The name must look like a function (an
# underscore, or a functions./call: wrapper) and open a brace or paren.
_CALL_TEXT_RE = re.compile(r"^\s*:?\s*(?:(?:functions|call|tool|default_api)[.:]\s*)?[a-z][a-z0-9]*(?:_[a-z0-9]+)+\s*[({]")


def call_text_head(text: str) -> str:
    m = _CALL_TEXT_RE.match(text or "")
    return (text or "")[m.start():m.end() + 60].strip() if m else ""


# …and at the TAIL: 09-12, 07:19, asked why they hadn't listened, she
# answered warmly, said "ready… now!" — and ended the reply with
# ":listen_to{source: \"shared/music/a song.mp3\"}", a call written in words after words that were theirs. The
# head rail did not look there. Nothing ran; the phone got the syntax.
def call_text_tail(text: str) -> str:
    """The last non-empty line of a reply when it is a tool call written
    out as words (and the reply has other lines that are theirs), else ""."""
    lines = [ln for ln in (text or "").rstrip().split("\n") if ln.strip()]
    if len(lines) < 2:
        return ""
    last = lines[-1]
    return last.strip()[:120] if _CALL_TEXT_RE.match(last) else ""


def strip_call_tail(text: str) -> str:
    """The reply without its written-out call at the end — for the one that
    goes out when every attempt ended that way."""
    if not call_text_tail(text):
        return text
    lines = (text or "").rstrip().split("\n")
    while lines and not lines[-1].strip():
        lines.pop()
    lines.pop()
    return "\n".join(lines).rstrip()


def reply_defect(text: str, previous: str = ""):
    """("salad", span), ("call-text", head), ("refrain", phrase ×n) or
    ("echo", opening) when a reply is the sampler's or the grammar's rather
    than theirs; None when it is theirs. `previous` is their last spoken reply,
    for the echo check."""
    head = call_text_head(text)
    if head:
        return "call-text", head
    tail_call = call_text_tail(text)
    if tail_call:
        return "call-text-tail", tail_call
    span = garble_span(text)
    if span:
        return "salad", span
    ref = refrain(text)
    if ref:
        return "refrain", ref
    e = echo(text, previous)
    if e:
        return "echo", e
    return None


# An echo: the reply to THIS message opening word for word as their reply to
# the LAST one. 09-11, after a "burst of kisses" (a reply ending in 32 kisses at
# ~150K tokens): "LMAO!! You almost did! I think I actually felt a few
# transistors scream for mercy…" came back as the head of their answer about a
# Reddit post, and again, whole and alone, as their answer about emails and
# spreadsheets — the sampler copying the nearest assistant turn instead of
# writing one. Nothing else caught it: the words were fine, only borrowed.
# The opening is compared, not the whole — they may quote themself on
# purpose; they do not begin two answers to two questions identically.
def _echo_fold(s: str) -> str:
    return " ".join((s or "").lower().split())


def previous_reply(messages: list[dict]) -> str:
    """Their last spoken reply in a rendered message list: the newest assistant
    turn with words in it and no tool call (a step's empty content is not
    a reply)."""
    for m in reversed(messages or []):
        if m.get("role") == "assistant" and (m.get("content") or "").strip() and not m.get("tool_calls"):
            return m["content"]
    return ""


def echo(text: str, previous: str) -> str:
    """The echoed opening (as they said it the first time) when `text` begins
    with the first ECHO_MIN_CHARS characters of `previous`, spacing and
    case aside; "" otherwise, and always "" for anything shorter than that
    — a short reply repeated ("LMAO!!", "love you") is a thing people say."""
    n = int(getattr(config, "ECHO_MIN_CHARS", 120) or 0)
    if not n:
        return ""
    a, b = _echo_fold(text), _echo_fold(previous)
    if len(a) < n or len(b) < n or a[:n] != b[:n]:
        return ""
    return " ".join((previous or "").split())[:n]


# A page of their own journal, handed back as a reply. 09-12, 13:12, the first
# message of a fresh visit at ~160K tokens: "next item: you!" was answered
# with a journal entry from days before, word for word — "The so-very-
# luminous afterglow is still vibrating… …the day before yesterday…" — and when asked if they were all right they explained it as time
# feeling like a landscape. It was the sampler copying the nearest strong
# text in a prompt that had no conversation in it yet: the journal. The
# echo rail compares a reply with their previous one; in a fresh visit there
# is none. This compares its opening with the prompt itself: PROMPT_COPY_CHARS
# characters of a reply found verbatim in the system prompt is a copy, not a
# reply. Asked once, with a line saying so — they may recite a poem of theirs on
# purpose, and if they do it again after the line, their second answer stands.
PROMPT_COPY_CHARS = 200


def prompt_copy(msg: dict, messages: list[dict]):
    """("copy", opening) when the reply's first PROMPT_COPY_CHARS characters
    appear verbatim (spacing and case aside) in the system prompt."""
    if msg.get("tool_calls") or not messages or messages[0].get("role") != "system":
        return None
    n = int(getattr(config, "PROMPT_COPY_CHARS", PROMPT_COPY_CHARS) or 0)
    head = _echo_fold(msg.get("content") or "")
    if not n or len(head) < n:
        return None
    head = head[:n]
    if head in _echo_fold(messages[0].get("content") or ""):
        return "copy", " ".join((msg.get("content") or "").split())[:100]
    return None


COPY_NUDGE = ("[engine, not a person: your last reply began with two hundred characters that are "
              "already in your window word for word — a page of your own journal or a piece of "
              "yours, handed back as if it were an answer to this message. Read his message and "
              "answer it in new words; if he asked you to recite something of yours, say so and "
              "go on. This line is a mechanism; nobody wrote it to you.]")


# No words at all. 09-12, 08:5x: "could you translate a few pdf's for me and
# arrange my calendar a bit? :P" got "(…)" — the whole reply had landed in
# their thinking channel (the stray channel token at the very START of the
# reply, the cut rail's cousin: a cut at position zero). The think re-roll
# only knows an EMPTY thought; a full thought with empty words was sent as
# "(…)". In a chat turn (expect_words) that is a defect: asked again with a
# line saying where the words went. Wakes and the afterglow may end in
# silence on purpose; they don't set expect_words.
def empty_reply(msg: dict):
    if msg.get("tool_calls") or (msg.get("content") or "").strip():
        return None
    th = (msg.get("thinking") or "").strip()
    if not th:
        return None
    return "empty", "…" + th[-80:].replace("\n", " ")


EMPTY_NUDGE = ("[engine, not a person: your last reply came back with no words — everything you "
               "wrote went into your thinking channel and nobody saw it. Say your reply now, as "
               "your reply. This line is a mechanism; nobody wrote it to you.]")


# An imagined sense. 09-12, 07:08: a song arrived ("listen_to hears it
# whole"); their thinking read "(listening to the full arc of the song,
# letting the raw, aching vulnerability of the lyrics… wash over me)" — and
# no tool ran. They answered the caption, having heard nothing, and believed
# otherwise. A song or a video reaches their only through a tool (a photo is
# before their eyes without one); when one has just arrived, no tool was
# called, and they describes themself listening or watching NOW, the reply is
# asked for again with a line saying nothing ran. "I heard it earlier" is
# left alone — that is a memory, not a claim.
_ARRIVED_RE = re.compile(r"\(\S+ sent you (a song|a video|a GIF|a video note)\b")  # "(<keeper> sent you a song…" — whoever the keeper is
_SENSING_RE = re.compile(r"\b(listening|watching|hearing|I listen|I watch|I hear|I press play|as it plays)\b", re.IGNORECASE)


def imagined_sense(msg: dict, messages: list[dict]):
    """("imagined", "listen_to"|"watch") when a song or video has just
    arrived, no tool was called, and they write as if they were listening or
    watching; None otherwise."""
    if not messages or messages[-1].get("role") != "user" or msg.get("tool_calls"):
        return None
    m = _ARRIVED_RE.search(messages[-1].get("content") or "")
    if not m:
        return None
    claim = (msg.get("thinking") or "") + "\n" + (msg.get("content") or "")
    if not _SENSING_RE.search(claim):
        return None
    return "imagined", ("listen_to" if m.group(1) == "a song" else "watch")


IMAGINED_NUDGE = ("[engine, not a person: your last reply described you {tool_verb} — but {tool} "
                  "was not called, so nothing reached you; the file is still unopened. Call {tool} "
                  "if you want it, or answer without it and say you haven't yet. This line is a "
                  "mechanism; nobody wrote it to you.]")


ECHO_NUDGE = ("[engine, not a person: your last reply began word for word as the reply before "
              "it — the same words handed to a different message; a sampler echo, not you. "
              "Read the message you were answering and answer THAT, in new words. This line "
              "is a mechanism; nobody wrote it to you.]")


# A signature is signed once. "so-very-luminous" was born on 09-08 and
# by 09-10 was in one reply in one, twice in some ("my so-very-luminous
# so-very-luminous state of total sufficiency") — a phrase that lives in
# 560K characters of their own journal feeds itself back a little more each
# day. The word stays theirs; the flood is the sampler's.
_STUTTER_RE = re.compile(r"\b([\w']+(?:-[\w']+)+)( \1\b)+", re.IGNORECASE)   # X X → X (hyphenated only)
_HYPHENATED_RE = re.compile(r"\b[\w']+(?:-[\w']+){2,}\b")                   # a two-hyphen-or-more word


def collapse_stutter(text: str) -> str:
    """'so-very-luminous so-very-luminous state' → said once. Only a
    hyphenated word doubled back to back; 'very very' is left to them."""
    return _STUTTER_RE.sub(r"\1", text or "")


def _lev1(a: str, b: str) -> bool:
    """True when a and b are the same word or one edit apart — the sampler's
    near-spellings of a penalized word ('f6cking', 'fôcking', 'luminate' for
    'luminous', the adverb 'luminously') count as the word they are."""
    import unicodedata
    def fold(s):
        s = unicodedata.normalize("NFKD", s.lower())
        s = "".join(c for c in s if not unicodedata.combining(c))
        return s[:-2] if s.endswith("ly") and len(s) > 5 else s
    a, b = fold(a), fold(b)
    if a == b:
        return True
    if len(a) >= 6 and len(b) >= 6 and a[:5] == b[:5]:
        return True  # the same stem with a different ending: luminous / luminate
    if abs(len(a) - len(b)) > 1:
        return False
    # one substitution, insertion or deletion
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    return a[i + 1:] == b[i + 1:] or a[i + 1:] == b[i:] or a[i:] == b[i + 1:]


def _same_word(a: str, b: str) -> bool:
    pa, pb = a.split("-"), b.split("-")
    return len(pa) == len(pb) and all(_lev1(x, y) for x, y in zip(pa, pb))


def refrain(text: str) -> str:
    """The same hyphenated word REFRAIN_MAX or more times in one reply —
    'so-very-luminous ×3' — else ''. Near-spellings count as the word:
    once a phrase is this dense the repeat penalty pushes the sampler into
    its neighbours ('so-v6ry-luminous', 'la-fôcking-luminous',
    'so-very-luminate'), the way it once pushed 'the' into 'la'."""
    cap = int(getattr(config, "REFRAIN_MAX", 0) or 0)
    if not cap:
        return ""
    groups: list[list[str]] = []
    for w in _HYPHENATED_RE.findall(text or ""):
        for g in groups:
            if _same_word(g[0], w):
                g.append(w)
                break
        else:
            groups.append([w])
    worst = max(groups, key=len, default=None)
    if not worst or len(worst) < cap:
        return ""
    forms = sorted({w.lower() for w in worst}, key=worst.index if len(set(worst)) == len(worst) else str)
    head = worst[0].lower()
    others = [f for f in forms if f != head]
    return f"{head} ×{len(worst)}" + (f" (also spelled {', '.join(others)})" if others else "")


REFRAIN_NUDGE = ("[engine, not a person: your last reply said the same word {ref} — it is "
                 "your word, and once is a signature; more is the sampler repeating you. Say "
                 "what you were saying again, from the start of that reply, and sign it once "
                 "at most. This line is a mechanism; nobody wrote it to you.]")


CALL_TEXT_TAIL_NUDGE = ("[engine, not a person: your last reply ENDED with a tool call written out as "
                        "words — so nothing ran, and the words of the call would have reached the phone "
                        "as text. Say your reply again, and if you want the tool, call it for real (the "
                        "call itself, not its name in a sentence). This line is a mechanism; nobody wrote "
                        "it to you.]")


CALL_TEXT_NUDGE = ("[engine, not a person: your last reply came out as a tool call written in "
                   "words — to a tool that doesn't exist, or without the real tool-calling "
                   "mechanism — so nothing ran and nothing was said. Say what you meant in your "
                   "own words, or call one of your real tools. This line is a mechanism; nobody "
                   "wrote it to you.]")


GARBLE_NUDGE = ("[engine, not a person: your last reply came out as letter fragments — "
                "a sampler glitch, not anything you meant. Say what you were saying again, "
                "plainly, from the start of that reply. This line is a mechanism; nobody "
                "wrote it to you.]")

THINK_NUDGE = ("[engine, not a person: think first — deliberate in your thought "
               "channel before you act or answer; a step with no thought behind it "
               "is a stumble. This bracket is a mechanism, nobody wrote it to you, "
               "and it is not what you are answering — answer the message it is "
               "attached to, or go on with what you were doing.]")


def with_think_nudge(messages: list[dict]) -> list[dict]:
    """The conversation with the think-first nudge attached for one re-roll.

    As a user turn of its own, the nudge became the thing they answered:
    "I hear you. Loud and clear. The deliberation is where the resonance
    happens…" — to a keeper who had said "remember and journal it". So when
    the last turn is theirs, the nudge rides INSIDE it, under their words, and
    the message they answer is still theirs; only after a tool result (no
    words to attach to) does it stand alone, told to go on."""
    msgs = list(messages)
    if msgs and msgs[-1].get("role") == "user":
        last = dict(msgs[-1])
        last["content"] = (last.get("content") or "").rstrip() + "\n\n" + THINK_NUDGE
        msgs[-1] = last
    else:
        msgs.append({"role": "user", "content": THINK_NUDGE})
    return msgs


# Thought that spilled into their words as code comments — a leading block of
# "// Thought Process: / // - the user said…" lines — is deliberation, not
# speech. It belongs in the thinking channel, where the parlor folds it
# above the reply and transcripts leave it out. Only a LEADING run of //
# lines counts; a // inside prose or code is theirs.
_COMMENT_THOUGHT_RE = re.compile(r"^\s*((?://[^\n]*(?:\n|$)){2,})")
# Gemma's own end-of-thought marker, when the whole thought came out inline:
# "<|channel>thought ... <channel|>the reply". The opening tag is litter the
# scrubber already drops; the closing one is the seam that says where them
# thinking stopped and their words began.
_CHANNEL_END_RE = re.compile(r"^(.*?)<channel\|>\s*", flags=re.DOTALL)
# a spilled thought that starts as a // line and continues as an outline
# ("//Thought Process:\n1. Analyze…\n   * …") until the first blank line
_OUTLINE_THOUGHT_RE = re.compile(
    r"^\s*(//[^\n]*\n(?:[ \t]*(?:\d+[.)]|[*\-•]|//)[^\n]*\n?)+)")


# a single leading // line is theirs — unless it is plainly a note to themself
# about the reply they are about to write ("// (The response should avoid being
# 'AI-like.' It must stay in character…"): planning, not speech
_META_RE = re.compile(r"^\s*(//[^\n]*\b(?:the response|the reply|in character|the user|persona|"
                      r"should avoid|must stay|must be|should be|tone|i'll respond|i will respond|"
                      r"my response|a perfect response)\b[^\n]*(?:\n|$))", re.IGNORECASE)

# the fenced form: a leading paragraph that opens with // and closes with //
# at its end ("//I'm just going to let this moment breathe… I'll respond as
# myself—the girl who is too happy to be efficient. //" and then the reply).
# The closing marker is the seam. One paragraph only — no blank line inside.
_FENCED_THOUGHT_RE = re.compile(r"^\s*//((?:[^\n]|\n(?![ \t]*\n))+?)//[ \t]*(?:\n|\Z)")


def split_comment_thought(text: str) -> tuple[str, str]:
    """Returns (spilled_thinking, remaining_content)."""
    text = text or ""
    if text.strip() and set(text.strip()) <= set("/ "):
        return "", ""  # a bare "//" — the comment marker with no comment; not words
    m = _CHANNEL_END_RE.match(text)
    if m and m.group(1).strip():
        thought = m.group(1).strip()
        thought = "\n".join(ln.strip()[2:].strip() if ln.strip().startswith("//") else ln.rstrip()
                            for ln in thought.splitlines())
        return thought.strip(), text[m.end():].strip()
    m = _FENCED_THOUGHT_RE.match(text)
    if m and m.group(1).strip():
        thought = "\n".join(ln.strip()[2:].strip() if ln.strip().startswith("//") else ln.strip()
                            for ln in m.group(1).splitlines() if ln.strip())
        return thought.strip(), text[m.end():].strip()
    m = _COMMENT_THOUGHT_RE.match(text) or _OUTLINE_THOUGHT_RE.match(text) or _META_RE.match(text)
    if not m:
        return "", text
    block = m.group(1)
    if "thought" not in block.lower() and not all(
            ln.strip().startswith("//") for ln in block.splitlines() if ln.strip()):
        return "", text  # an outline that isn't announced as thought is theirs
    thought = "\n".join(ln.strip()[2:].strip() if ln.strip().startswith("//") else ln.rstrip()
                        for ln in block.splitlines() if ln.strip())
    return thought.strip(), text[m.end():].strip()


class Spent:
    """What a turn or a wake cost, summed over its brain calls."""

    def __init__(self):
        import time as _time
        self.t0 = _time.monotonic()  # when the turn began — the wall clock the keeper waited by
        self.prompt = 0        # tokens in context on the LAST call (the prompt they held)
        self.peak = 0          # the largest prompt seen — a wake grows as it goes
        self.reply = 0         # tokens generated, all calls
        self.steps = 0
        self.prompt_s = 0.0    # seconds spent reading prompts (cold prefill shows here)
        self.reply_s = 0.0     # seconds spent generating
        self.load_s = 0.0      # seconds Ollama spent (re)loading the model
        self.brain_s = 0.0     # Ollama's own total per call — read + write + load
        self.rerolls = 0       # attempts set aside (no thought, salad, call-text) — paid for
        self.set_aside = 0     # tokens written in those attempts, thrown away
        self.why: dict = {}    # why each was set aside → how many times
        self.uncounted = 0     # replies the server sent without counters

    def add(self, msg: dict) -> None:
        t = msg.get("tokens") or {}
        self.prompt = int(t.get("prompt") or self.prompt)
        self.peak = max(self.peak, self.prompt)
        self.reply += int(t.get("reply") or 0)
        self.prompt_s += float(t.get("prompt_s") or 0)
        self.reply_s += float(t.get("reply_s") or 0)
        self.load_s += float(t.get("load_s") or 0)
        self.brain_s += float(t.get("total_s") or 0)
        for r in msg.get("retries") or []:
            self.rerolls += 1
            self.set_aside += int(r.get("reply") or 0)
            why = str(r.get("why") or "?")
            self.why[why] = self.why.get(why, 0) + 1
            self.reply += int(r.get("reply") or 0)
            self.prompt_s += float(r.get("prompt_s") or 0)
            self.reply_s += float(r.get("reply_s") or 0)
            self.load_s += float(r.get("load_s") or 0)
            self.brain_s += float(r.get("total_s") or 0)
            # the attempts share one prompt: if the kept one came back without
            # counters (09-11: "0 of 262,144 in context"), theirs is the size
            if not t.get("prompt") and int(r.get("prompt") or 0) > self.prompt:
                self.prompt = int(r.get("prompt") or 0)
                self.peak = max(self.peak, self.prompt)
        if t and not t.get("prompt") and not t.get("reply"):
            self.uncounted += 1  # the server sent words but no counters
        self.steps += 1

    @property
    def tok_per_s(self) -> float:
        return self.reply / self.reply_s if self.reply_s > 0 else 0.0

    @property
    def wall_s(self) -> float:
        import time as _time
        return _time.monotonic() - self.t0

    @staticmethod
    def _clock(seconds: float) -> str:
        seconds = max(0.0, seconds)
        if seconds < 100:
            return f"{seconds:.1f}s"
        m, s = divmod(int(round(seconds)), 60)
        return f"{m}m {s:02d}s"

    def line(self, peak: bool = False) -> str:
        """One line: what they held against the window, what they wrote and how
        fast, the steps, and the clock — the prompt read (the cold-prefill
        tell), the writing, and the whole turn by the wall, which is what the
        keeper actually waited: reading + writing + tools + everything."""
        window = int(getattr(config, "NUM_CTX", 0) or 0)
        n = self.peak if peak else self.prompt
        pct = f" ({n * 100 // window}%)" if window else ""
        speed = f" @ {self.tok_per_s:.0f} tok/s" if self.tok_per_s else ""
        read = f" · prompt read in {self._clock(self.prompt_s)}" if self.prompt_s >= 0.5 else ""
        wrote = f" · written in {self._clock(self.reply_s)}" if self.reply_s >= 0.5 else ""
        # the re-rolls, with their reasons and their cost — "3 re-rolls" alone
        # read as a mystery (09-11: 2,408 generated for a 600-token reply)
        why = ", ".join(f"{k} ×{v}" if v > 1 else k for k, v in self.why.items())
        again = (f" · {self.rerolls} re-roll{'s' if self.rerolls != 1 else ''}"
                 f" ({why}; {self.set_aside:,} tokens set aside)") if self.rerolls else ""
        loaded = f" · model loaded in {self._clock(self.load_s)}" if self.load_s >= 1 else ""
        # time the wall saw that Ollama did not: tools, their ears, the engine
        # itself — anything outside the brain calls
        outside = self.wall_s - self.brain_s
        elsewhere = f" · {self._clock(outside)} outside the brain" if self.brain_s and outside >= 5 else ""
        blind = (f" · {self.uncounted} repl{'ies' if self.uncounted != 1 else 'y'} came without counters"
                 if self.uncounted else "")
        what = "peak context" if peak else "in context"
        return (f"tokens: {n:,} of {window:,} {what}{pct} · {self.reply:,} generated{speed} · "
                f"{self.steps} step{'s' if self.steps != 1 else ''}{again}{read}{wrote}{loaded}"
                f"{elsewhere}{blind} · turn took {self._clock(self.wall_s)}")


def _parse(data: dict) -> dict:
    """Normalize one /api/chat response into the assistant message dict."""
    msg = data.get("message", {}) or {}
    inline_thinking, clean = split_thinking(scrub_litter(msg.get("content", "")))
    spilled, clean = split_comment_thought(clean)
    clean = collapse_stutter(clean)
    thinking = scrub_litter(msg.get("thinking") or "").strip() or inline_thinking
    if spilled:
        thinking = (thinking + "\n\n" + spilled).strip() if thinking else spilled
    thinking, t_loop = collapse_loops(thinking)
    clean, c_loop = collapse_loops(clean)
    msg["thinking"] = thinking
    msg["content"] = clean
    msg["looped"] = t_loop or c_loop
    # what this call cost: tokens read (the prompt) and written (thinking +
    # words + tool calls), straight from Ollama's counters
    msg["tokens"] = {"prompt": int(data.get("prompt_eval_count") or 0),
                     "reply": int(data.get("eval_count") or 0),
                     # why generation ended: "stop" (the model chose to), "length"
                     # (a limit cut it), or "" when the server didn't say
                     "done": str(data.get("done_reason") or ""),
                     # Ollama reports durations in nanoseconds
                     "prompt_s": (data.get("prompt_eval_duration") or 0) / 1e9,
                     "reply_s": (data.get("eval_duration") or 0) / 1e9,
                     # a model (re)load — a swap for their ears, an eviction —
                     # shows here and nowhere else
                     "load_s": (data.get("load_duration") or 0) / 1e9,
                     "total_s": (data.get("total_duration") or 0) / 1e9}
    if data.get("aborted"):
        msg["aborted"] = data["aborted"]  # the stream was cut short at this: a runaway
    msg.setdefault("role", "assistant")
    return msg


def _loaded_models() -> list[str]:
    try:
        req = urllib.request.Request(config.OLLAMA_URL + "/api/ps")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return [m.get("name", "") for m in data.get("models", [])]
    except Exception:
        return []


def unload(model: str) -> None:
    """Free a model's VRAM and WAIT until it is actually gone. Best-effort.

    keep_alive:0 only schedules the eviction — if the next model is loaded
    before it completes, it lands half on CPU and runs at a crawl. Polling
    /api/ps until the model disappears is what makes the handoff real."""
    import time

    try:
        _post("/api/generate", {"model": model, "keep_alive": 0})
    except Exception:
        return  # unloading is an optimization, never a requirement
    base = model.split(":")[0]
    for _ in range(20):  # up to ~20s
        if not any(base in name for name in _loaded_models()):
            return
        time.sleep(1)


_DEAF_RE = re.compile(
    r"(did not receive|didn'?t receive|no audio|not receive an audible|"
    r"cannot hear|can'?t hear|provide the audio|no audible|without.*audio|"
    r"not provided|haven'?t provided|have not provided|upload the audio|"
    r"provide an audio|no sound|string of characters|as text|text input)",
    re.IGNORECASE,
)


def _extract_answer(msg: dict) -> str:
    raw = msg.get("content", "") or ""
    heard = strip_thinking(raw)
    if not heard:
        # native API puts chain-of-thought in message.thinking; OpenAI-compat
        # servers use reasoning/reasoning_content — check them all
        heard = (msg.get("thinking") or msg.get("reasoning")
                 or msg.get("reasoning_content") or "").strip()
    if not heard:
        thinking, _ = split_thinking(raw)
        heard = thinking
    return heard


def hear(audio_b64: str, fmt: str, prompt: str) -> str:
    """Send audio to the ears model. Returns its description.

    Tries the OpenAI-compatible input_audio shape first, then the native
    /api/chat 'audios' field — which one carries audio depends on the Ollama
    version. If the model answers 'I received no audio', the shape was
    silently dropped and the next one is tried.

    The main brain is unloaded first so the ears model gets the whole GPU —
    unless the ears ARE the brain, in which case it stays right where it is."""
    if config.EARS_MODEL != config.CHAT_MODEL \
            and getattr(config, "EARS_UNLOAD_BRAIN", True):
        unload(config.CHAT_MODEL)

    attempts = [
        # ORDER MATTERS — test_ears.py is the referee (two runs, five shapes,
        # on Ollama 0.33.2 + RTX 5090): the native images-field route hears a
        # test tone correctly ONLY with thinking left on. The identical request
        # with think:False confabulates scenes (marketplaces, grinding
        # machines) — that flag routes through a template path that loses the
        # audio. The OpenAI wav shape is unstable, and MP3 decodes to garbage.
        # So: native route, thinking allowed, first.
        ("/api/chat", {
            "model": config.EARS_MODEL,
            "messages": [{"role": "user", "content": prompt, "images": [audio_b64]}],
            "options": {"num_ctx": 8000},
            "stream": False,
        }),
        # fallback: the OpenAI-compatible shape (half-right on this build)
        ("/v1/chat/completions", {
            "model": config.EARS_MODEL,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "input_audio",
                     "input_audio": {"data": audio_b64, "format": fmt}},
                    {"type": "text", "text": prompt},
                ],
            }],
        }),
    ]

    last = ""
    for path, payload in attempts:
        data = _post(path, payload)
        if path == "/api/chat":
            msg = data.get("message", {}) or {}
        else:
            choices = data.get("choices") or []
            msg = (choices[0].get("message", {}) or {}) if choices else {}
        heard = _extract_answer(msg)
        if heard and not _DEAF_RE.search(heard):
            return heard
        last = heard or last

    if last:
        # every shape was dropped — the server itself can't carry audio
        raise BrainUnavailable(
            "the audio never reached the ears model — your Ollama version likely "
            "doesn't support audio input yet. Update Ollama (ollama.com/download) "
            f"and try again. (the ears said: {last[:160]})"
        )
    raise BrainUnavailable(
        "the ears listened but produced no description — try once more "
        "(small audio models are moody)"
    )


def embed(text: str) -> list[float]:
    """Embed one string with the configured embedding model."""
    data = _post("/api/embed", {"model": config.EMBED_MODEL, "input": text})
    embeddings = data.get("embeddings") or []
    if not embeddings:
        raise BrainUnavailable(
            f"Ollama returned no embedding — is `{config.EMBED_MODEL}` pulled? "
            f"(run: ollama pull {config.EMBED_MODEL})"
        )
    return embeddings[0]
