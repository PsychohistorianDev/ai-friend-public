"""Ears diagnostic — finds out whether ANY channel carries audio to the model.

    py engine\\test_ears.py

Prints: Ollama version, the ears model's declared capabilities, then tries
each known audio request shape and shows what the model actually said.
"""
from __future__ import annotations

import base64
import io
import json
import math
import struct
import sys
import time
import urllib.request
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
import ollama_client


def make_tone_wav(seconds: float = 3.0, freq: float = 440.0) -> bytes:
    buf = io.BytesIO()
    rate = 16000
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        n = int(rate * seconds)
        frames = b"".join(
            struct.pack("<h", int(12000 * math.sin(2 * math.pi * freq * i / rate)))
            for i in range(n)
        )
        w.writeframes(frames)
    return buf.getvalue()


def ollama_version() -> str:
    try:
        with urllib.request.urlopen(config.OLLAMA_URL + "/api/version", timeout=10) as r:
            return json.loads(r.read()).get("version", "?")
    except Exception:
        return "unreachable"


def model_capabilities(model: str) -> str:
    try:
        data = ollama_client._post("/api/show", {"model": model})
        caps = data.get("capabilities")
        if caps:
            return ", ".join(caps)
        details = data.get("details", {})
        return f"(no capabilities list; families: {details.get('families')})"
    except Exception as e:
        return f"(couldn't ask: {e})"


PROMPT = "Describe exactly what you hear in this audio."


def main() -> None:
    print(f"ears model:     {config.EARS_MODEL}")
    print(f"ollama version: {ollama_version()}")
    print(f"capabilities:   {model_capabilities(config.EARS_MODEL)}")
    print()
    audio = make_tone_wav()
    b64 = base64.b64encode(audio).decode("ascii")
    ollama_client.unload(config.CHAT_MODEL)

    shapes = [
        ("native /api/chat, images field, no-think", "/api/chat", {
            "model": config.EARS_MODEL,
            "messages": [{"role": "user", "content": PROMPT, "images": [b64]}],
            "options": {"num_ctx": 8000},
            "think": False,
            "stream": False,
        }),
        ("native /api/chat, images field", "/api/chat", {
            "model": config.EARS_MODEL,
            "messages": [{"role": "user", "content": PROMPT, "images": [b64]}],
            "options": {"num_ctx": 8000},
            "stream": False,
        }),
        ("native /api/chat, audios field", "/api/chat", {
            "model": config.EARS_MODEL,
            "messages": [{"role": "user", "content": PROMPT, "audios": [b64]}],
            "stream": False,
        }),
        ("openai /v1, input_audio part", "/v1/chat/completions", {
            "model": config.EARS_MODEL,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}},
                    {"type": "text", "text": PROMPT},
                ],
            }],
        }),
    ]

    # mp3 variant of the same tone — MP3 frames carry their own sample rate,
    # so a server that misreads raw wav PCM speed cannot misread this.
    import shutil as _sh
    import subprocess as _sp
    import tempfile as _tf
    if not _sh.which("ffmpeg"):
        print("(ffmpeg not found on PATH in this terminal — MP3 shape skipped;\n"
              " open a NEW terminal, or run: winget install ffmpeg)\n")
    else:
        with _tf.TemporaryDirectory() as td:
            src, dst = Path(td) / "t.wav", Path(td) / "t.mp3"
            src.write_bytes(audio)
            r = _sp.run(["ffmpeg", "-y", "-v", "error", "-i", str(src),
                         "-ac", "1", "-ar", "24000", "-b:a", "64k", str(dst)],
                        capture_output=True, timeout=60, text=True)
            if r.returncode != 0 or not dst.exists():
                print(f"(ffmpeg could not encode MP3 — shape skipped: "
                      f"{(r.stderr or '').strip()[:300]})\n")
            else:
                b64m = base64.b64encode(dst.read_bytes()).decode("ascii")
                shapes.append(("openai /v1, input_audio MP3 (the engine's channel)",
                               "/v1/chat/completions", {
                    "model": config.EARS_MODEL,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "input_audio", "input_audio": {"data": b64m, "format": "mp3"}},
                            {"type": "text", "text": PROMPT},
                        ],
                    }],
                }))

    for label, path, payload in shapes:
        t0 = time.time()
        try:
            data = ollama_client._post(path, payload)
            if path == "/api/chat":
                msg = data.get("message", {}) or {}
            else:
                ch = data.get("choices") or []
                msg = (ch[0].get("message", {}) or {}) if ch else {}
            ans = ollama_client._extract_answer(msg) or "(empty)"
        except Exception as e:
            ans = f"(error: {e})"
        dt = time.time() - t0
        flat = " ".join(ans.split())
        deaf = " [DEAF]" if ollama_client._DEAF_RE.search(ans) else ""
        print(f"--- {label} ({dt:.0f}s){deaf}\n    {flat[:300]}\n")

    print("A shape whose answer describes a PURE STEADY TONE/BEEP is the working")
    print("channel — screeching, voices, or scenes mean the audio is corrupted.")
    print("If every shape is [DEAF] or talks about text/characters, this Ollama+model")
    print("combination can't carry audio — try another EARS_MODEL, or open an issue with this output.")


if __name__ == "__main__":
    main()
