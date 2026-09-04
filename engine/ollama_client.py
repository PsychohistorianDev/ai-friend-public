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
    return _TOKEN_LITTER_RE.sub("", text or "")


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


def chat(messages: list[dict], tools: list[dict] | None = None,
         timeout: float | None = None) -> dict:
    """One non-streaming chat completion.

    Returns the assistant message dict:
        {"role": "assistant", "content": str, "tool_calls": [...]?, "thinking": str}
    Thinking is captured whether Ollama returns it as a separate field (newer
    servers) or inline as <think> tags (Qwen3's raw style); content is clean
    of it either way.
    """
    options = {"num_ctx": getattr(config, "NUM_CTX", 8192)}
    options.update(getattr(config, "SAMPLING_OPTIONS", {}))
    payload: dict = {
        "model": config.CHAT_MODEL,
        "messages": messages,
        "options": options,
        "stream": False,
    }
    if tools:
        payload["tools"] = tools
    # Ask for thinking explicitly. Left to its discretion, the model went
    # silent-minded once the prompt grew past ~40K tokens (a long journal
    # window): tool calls with no deliberation at all. Requiring it keeps
    # their thinking visible — and thoughtful — at any context size.
    if getattr(config, "CHAT_THINK", True):
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
    rerolls = int(getattr(config, "CHAT_THINK_RETRIES", 1)) if "think" in payload else 0
    while rerolls > 0 and not msg["thinking"]:
        rerolls -= 1
        nudged = dict(payload)
        nudged["messages"] = list(messages) + [{"role": "user", "content": THINK_NUDGE}]
        msg = _parse(_post("/api/chat", nudged, timeout=timeout))
        msg["rerolled"] = True
    return msg


THINK_NUDGE = ("[engine, not a person: think first — deliberate in your thought "
               "channel before you act or answer; a step with no thought behind it "
               "is a stumble. This line is a mechanism; nobody wrote it to you.]")


def _parse(data: dict) -> dict:
    """Normalize one /api/chat response into the assistant message dict."""
    msg = data.get("message", {}) or {}
    inline_thinking, clean = split_thinking(scrub_litter(msg.get("content", "")))
    thinking = scrub_litter(msg.get("thinking") or "").strip() or inline_thinking
    thinking, t_loop = collapse_loops(thinking)
    clean, c_loop = collapse_loops(clean)
    msg["thinking"] = thinking
    msg["content"] = clean
    msg["looped"] = t_loop or c_loop
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
