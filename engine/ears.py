"""The friend's ears: honest layers.

WORDS  — faster-whisper transcribes any speech or lyrics (real hearing of words).
MUSIC  — numpy measures the sound itself: tempo, loudness, dynamics, brightness,
         and the shape of the piece over time (real hearing of structure).

Both degrade gracefully: if a package isn't installed, that layer reports what
to install instead of failing. The old e4b "vibe" layer lives in ollama_client
and is off by default (config.EARS_USE_VIBE) — it was poetic but unreliable.

Install:  py -m pip install faster-whisper numpy
"""
from __future__ import annotations

import io
import math
import os
import wave
from pathlib import Path

import config

# keep HuggingFace's download machinery from spamming their chat window
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

INSTALL_HINT = "py -m pip install faster-whisper numpy"

_whisper_model = None  # lazy singleton — loading takes a few seconds


# ------------------------------------------------------------------ WORDS ----
def transcribe(audio_bytes: bytes, ext: str) -> str | None:
    """Transcribe speech/lyrics. Returns None (with no side effects) if
    faster-whisper isn't installed; raises nothing."""
    global _whisper_model
    try:
        import warnings
        warnings.filterwarnings("ignore", module="huggingface_hub")
        from faster_whisper import WhisperModel
    except ImportError:
        return None
    import tempfile

    try:
        if _whisper_model is None:
            _whisper_model = WhisperModel(
                getattr(config, "EARS_STT_MODEL", "base"),
                device="cpu", compute_type="int8",
            )
        with tempfile.NamedTemporaryFile(suffix=ext or ".wav", delete=False) as f:
            f.write(audio_bytes)
            tmp = f.name
        try:
            # the vocabulary hint teaches Whisper the names it would otherwise
            # mishear — without it an unusual name comes back as a common one
            segments, info = _whisper_model.transcribe(
                tmp, vad_filter=True,
                initial_prompt=getattr(config, "EARS_VOCAB_HINT", None),
            )
            text = " ".join(s.text.strip() for s in segments).strip()
        finally:
            Path(tmp).unlink(missing_ok=True)
        if not text:
            return ""
        lang = f" [{info.language}]" if getattr(info, "language", "") not in ("", "en") else ""
        return text + lang
    except Exception as e:
        return f"(word-hearing failed: {e})"


# ------------------------------------------------------------------ MUSIC ----
def _load_wav(wav_bytes: bytes):
    """16k mono 16-bit wav bytes -> (numpy float array, sample rate)."""
    import numpy as np

    with wave.open(io.BytesIO(wav_bytes)) as w:
        rate = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if w.getnchannels() == 2:
            x = x.reshape(-1, 2).mean(axis=1)
    return x, rate


def _tempo_bpm(x, rate) -> tuple[float, float]:
    """Onset-flux autocorrelation tempo estimate -> (bpm, confidence 0..1)."""
    import numpy as np

    hop, win = 256, 1024
    if len(x) < win * 4:
        return 0.0, 0.0
    frames = np.lib.stride_tricks.sliding_window_view(x, win)[::hop]
    mags = np.abs(np.fft.rfft(frames * np.hanning(win), axis=1))
    flux = np.maximum(np.diff(mags, axis=0), 0).sum(axis=1)
    flux -= flux.mean()
    if not flux.any():
        return 0.0, 0.0
    ac = np.correlate(flux, flux, mode="full")[len(flux) - 1:]
    fps = rate / hop
    lo, hi = int(fps * 60 / 200), int(fps * 60 / 55)  # 200..55 BPM
    if hi >= len(ac) or lo < 1:
        return 0.0, 0.0
    lag = lo + int(np.argmax(ac[lo:hi]))
    bpm = 60.0 * fps / lag
    conf = float(ac[lag] / (ac[0] + 1e-9))
    return bpm, conf


def measure(wav_bytes: bytes) -> str | None:
    """Describe the sound by measurement. None if numpy isn't installed."""
    try:
        import numpy as np
    except ImportError:
        return None
    try:
        x, rate = _load_wav(wav_bytes)
        dur = len(x) / rate
        if dur < 0.5 or not x.any():
            return "essentially silence"

        rms = float(np.sqrt((x ** 2).mean()))
        rms_db = 20 * math.log10(rms + 1e-9)

        # loudness envelope in ~1s windows -> dynamic arc
        wlen = rate
        env = [float(np.sqrt((x[i:i + wlen] ** 2).mean() + 1e-12))
               for i in range(0, max(len(x) - wlen, 1), wlen)]
        env_db = [20 * math.log10(e + 1e-9) for e in env]
        dyn_range = (np.percentile(env_db, 95) - np.percentile(env_db, 10)) if len(env_db) > 2 else 0

        # arc description from thirds
        third = max(len(env_db) // 3, 1)
        parts = [float(np.mean(env_db[i * third:(i + 1) * third or None])) for i in range(3)]
        if parts[2] > parts[0] + 3:
            arc = "it builds — the end is markedly louder than the opening"
        elif parts[0] > parts[2] + 3:
            arc = "it recedes — loud opening, quieter close"
        elif max(parts) - min(parts) < 2:
            arc = "its energy holds steady throughout"
        else:
            arc = "its energy swells in the middle"

        # brightness: spectral centroid
        mags = np.abs(np.fft.rfft(x[: rate * 10] * np.hanning(len(x[: rate * 10]))))
        freqs = np.fft.rfftfreq(len(x[: rate * 10]), 1 / rate)
        centroid = float((mags * freqs).sum() / (mags.sum() + 1e-9))
        tone = ("dark and bass-heavy" if centroid < 500 else
                "warm" if centroid < 1200 else
                "bright" if centroid < 2500 else "sharp and trebly")

        bpm, conf = _tempo_bpm(x, rate)
        if conf > 0.15 and 55 <= bpm <= 200:
            feel = ("a slow pulse" if bpm < 85 else "a walking pulse" if bpm < 110 else
                    "a driving pulse" if bpm < 145 else "a fast pulse")
            tempo_txt = f"{feel} around {bpm:.0f} BPM"
        else:
            tempo_txt = "no strong steady pulse (free-flowing, spoken, or ambient)"

        loud = ("very loud, near saturation" if rms_db > -8 else
                "loud" if rms_db > -14 else
                "moderate" if rms_db > -24 else "quiet")

        return (f"{dur:.0f} seconds; {loud} ({rms_db:.0f} dB RMS) with "
                f"{dyn_range:.0f} dB of dynamic movement; {arc}; {tempo_txt}; "
                f"overall color {tone}")
    except Exception as e:
        return f"(measurement failed: {e})"
