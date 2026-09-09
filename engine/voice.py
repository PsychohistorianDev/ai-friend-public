"""Their voice: words → a voice note, spoken by Kokoro on the CPU.

    py engine\\voice.py --test "hello, it's me"      writes shared/voice-test.ogg
    py engine\\voice.py --voices                       lists the voices they can choose

Kokoro is an open-weight 82M-parameter text-to-speech model (Apache 2.0):
good, quick on a CPU (a sentence in well under a second, a paragraph in a
few), and it never touches the GPU their brain is holding. It ships with a
handful of voices; they pick theirs once (the `speak` tool's `voice=`), and the
choice is kept in memory/voice.json — their voice is their choice, like their name.

The voice note is OGG/Opus, which is what Telegram plays as a voice message
(the round waveform); ffmpeg does the encoding, as it does for their ears.

Install (once, on the PC):   pip install kokoro soundfile
                             (ffmpeg on PATH — the ears already need it)

If the engine's Python is too new for Kokoro's dependencies (Python 3.14:
numpy 1.26 has no wheel there and pip tries to compile it — "Unknown
compiler(s)"), give the voice its own interpreter: install Python 3.12 from
python.org beside the current one (keep 3.14 as the default), then
    py -3.12 -m pip install kokoro soundfile
and set in config.py:   VOICE_PYTHON = "py -3.12"
The voice then runs in that interpreter, one short process per note; the
engine itself stays where it is.
"""
from __future__ import annotations

import io
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config

INSTALL_HINT = ("pip install kokoro soundfile   (and ffmpeg on PATH); on Python 3.14 use a 3.12: "
                "py -3.12 -m pip install kokoro soundfile, then VOICE_PYTHON = \"py -3.12\" in config")

# Kokoro's voices. Prefix: a = American English, b = British English;
# f/m = the voice's register. Descriptions are the model card's grades and
# our own ears; they can try any and keep the one that sounds like them.
VOICES = {
    # American, feminine
    "af_heart":    "American, warm and clear — Kokoro's own favourite (grade A)",
    "af_bella":    "American, bright, a little breath in it (A-)",
    "af_nicole":   "American, whispery, close to the ear (B-)",
    "af_aoede":    "American, soft, low (C+)",
    "af_kore":     "American, steady (C+)",
    "af_sarah":    "American, even and calm (C+)",
    "af_nova":     "American, quick, a little dry (C)",
    "af_alloy":    "American, plain (C)",
    "af_sky":      "American, light and young (C-)",
    "af_jessica":  "American, flat (D)",
    "af_river":    "American, rough (D)",
    # American, masculine
    "am_michael":  "American, low and warm (C+)",
    "am_fenrir":   "American, deep (C+)",
    "am_puck":     "American, playful (C+)",
    "am_echo":     "American (D)",
    "am_eric":     "American (D)",
    "am_liam":     "American (D)",
    "am_onyx":     "American, dark (D)",
    "am_adam":     "American (F+)",
    "am_santa":    "American, ho ho (D-)",
    # British, feminine
    "bf_emma":     "British, gentle (B-)",
    "bf_isabella": "British, clear, poised (C)",
    "bf_alice":    "British, light (D)",
    "bf_lily":     "British (D)",
    # British, masculine
    "bm_george":   "British, grave (C)",
    "bm_fable":    "British, storyteller (C)",
    "bm_lewis":    "British (D+)",
    "bm_daniel":   "British (D)",
}
# Other languages exist (French ff_siwis, Spanish ef_dora/em_alex, Italian
# if_sara/im_nicola, Portuguese pf_dora/pm_alex, Hindi hf_alpha/hm_omega,
# Japanese jf_alpha/jm_kumo, Chinese zf_xiaoxiao/zm_yunxi…) but each speaks
# ITS language: an English sentence through the French voice comes out as
# French phonetics, and those pipelines need espeak-ng installed. Blends of
# the English voices are the way to a voice that is theirs: "af_bella,af_sky"
# averages two; "af_heart(2)+af_nicole(1)" weights them.
DEFAULT_VOICE = "af_heart"
_VOICE_FILE = config.MEMORY_DIR / "voice.json"


class VoiceUnavailable(RuntimeError):
    """Kokoro (or ffmpeg) isn't here; the message says what to install."""


def chosen() -> dict:
    """{"voice": ..., "speed": ...} — theirs, once they have chosen."""
    try:
        d = json.loads(_VOICE_FILE.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def choose(voice: str, speed: float | None = None) -> None:
    d = chosen()
    d["voice"] = voice
    if speed:
        d["speed"] = float(speed)
    _VOICE_FILE.write_text(json.dumps(d), encoding="utf-8")


def valid_voice(name: str) -> bool:
    """A listed voice, or a blend of listed voices — 'af_bella,af_sky'
    (averaged) or 'af_heart(2)+af_nicole(1)' (weighted)."""
    parts = [re.sub(r"\(.*?\)", "", p).strip() for p in re.split(r"[,+]", name or "") if p.strip()]
    return bool(parts) and all(p in VOICES for p in parts)


_STAGE_RE = re.compile(r"(?<!\*)\*(?!\*)([^*\n]+ [^*\n]{1,160})\*(?!\*)")  # *blushes a soft violet*: an action (two+ words), not speech; **bold** and *one* word stay
_MARK_RE = re.compile(r"[*_`#>]+")                      # markdown bones
_EMOJI_RE = re.compile("[\U0001F000-\U0001FAFF☀-➿⬀-⯿️‍❤♾]")
_SPACE_RE = re.compile(r"[ \t]+")


def clean_for_speech(text: str) -> str:
    """What gets spoken: their words, not their stage directions, markdown or
    emoji. A reply that is only a stage direction is spoken as it is —
    better a whispered '(blushes)' than silence."""
    t = text or ""
    spoken = _STAGE_RE.sub("", t)
    if not re.search(r"[A-Za-z]{3,}", spoken):
        spoken = t.replace("*", "")
    spoken = _MARK_RE.sub("", spoken)
    spoken = _EMOJI_RE.sub("", spoken)
    spoken = spoken.replace("—", ", ").replace(" - ", ", ").replace("…", "...")
    spoken = re.sub(r"\n{2,}", "\n", spoken)
    spoken = _SPACE_RE.sub(" ", spoken)
    spoken = re.sub(r"\s+([,.;:!?])", r"\1", spoken)
    return spoken.strip()


_pipelines: dict[str, object] = {}


def _pipeline(lang: str):
    """One Kokoro pipeline per language, loaded on first use (a few seconds)."""
    if lang not in _pipelines:
        try:
            import warnings
            warnings.filterwarnings("ignore")
            from kokoro import KPipeline
        except ImportError as e:
            raise VoiceUnavailable(f"no voice yet — your keeper runs: {INSTALL_HINT}") from e
        device = getattr(config, "VOICE_DEVICE", "cpu")
        try:
            _pipelines[lang] = KPipeline(lang_code=lang, device=device)
        except TypeError:  # older kokoro: no device argument
            _pipelines[lang] = KPipeline(lang_code=lang)
    return _pipelines[lang]


def _resolve(voice: str | None, speed: float | None) -> tuple[str, float]:
    c = chosen()
    voice = (voice or c.get("voice") or getattr(config, "VOICE_NAME", DEFAULT_VOICE)).strip()
    speed = float(speed or c.get("speed") or getattr(config, "VOICE_SPEED", 1.0))
    return voice, speed


def _wav_seconds(wav: bytes) -> float:
    import wave
    try:
        with wave.open(io.BytesIO(wav)) as w:
            return w.getnframes() / float(w.getframerate() or 24000)
    except Exception:
        return 0.0


def _synthesize_via(py: str, text: str, voice: str, speed: float) -> tuple[bytes, float]:
    """Kokoro in another interpreter (VOICE_PYTHON): this same file, run with
    --synth, text on stdin, wav to a temp file. One short process per note."""
    import shlex
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "out.wav"
        cmd = shlex.split(py) + [str(Path(__file__).resolve()), "--synth", "--voice", voice,
                                 "--speed", str(speed), "--out", str(out)]
        try:
            proc = subprocess.run(cmd, input=text.encode("utf-8"), capture_output=True,
                                  timeout=float(getattr(config, "VOICE_TIMEOUT_S", 180)))
        except FileNotFoundError as e:
            raise VoiceUnavailable(f"VOICE_PYTHON {py!r} isn't a Python I can run ({e})") from e
        except subprocess.TimeoutExpired as e:
            raise VoiceUnavailable("the voice took too long — Kokoro may still be downloading its model") from e
        if proc.returncode != 0 or not out.exists():
            err = (proc.stderr or b"").decode("utf-8", "replace").strip().splitlines()
            raise VoiceUnavailable("the voice failed in " + py + ": " + (err[-1] if err else "no output"))
        wav = out.read_bytes()
    return wav, _wav_seconds(wav)


def synthesize(text: str, voice: str | None = None, speed: float | None = None) -> tuple[bytes, float]:
    """Speak `text`. Returns (wav bytes at 24kHz mono, seconds)."""
    voice, speed = _resolve(voice, speed)
    py = (getattr(config, "VOICE_PYTHON", "") or "").strip()
    if py:
        return _synthesize_via(py, text, voice, speed)
    return synthesize_here(text, voice, speed)


def _blend(pipe, voice: str) -> str:
    """A weighted blend — "af_heart(2)+af_nicole(1)" — is mixed here (Kokoro
    itself only averages a comma list): each voice's pack is loaded, the
    weighted mean is registered under the blend's name, and the pipeline
    finds it by that name. Anything else is returned untouched."""
    if "+" not in voice and "(" not in voice:
        return voice
    try:
        import torch
        parts, weights = [], []
        for piece in re.split(r"[,+]", voice):
            piece = piece.strip()
            if not piece:
                continue
            m = re.match(r"([a-z]{2}_[a-z]+)\s*(?:\((\d+(?:\.\d+)?)\))?$", piece)
            if not m:
                raise ValueError(piece)
            parts.append(pipe.load_single_voice(m.group(1)))
            weights.append(float(m.group(2) or 1.0))
        w = torch.tensor(weights, dtype=parts[0].dtype).view(-1, *([1] * (parts[0].dim())))
        mixed = (torch.stack(parts) * w).sum(dim=0) / w.sum()
        pipe.voices[voice] = mixed
        return voice
    except Exception as e:
        raise VoiceUnavailable(f"couldn't mix the voice {voice!r}: {e}") from e


def synthesize_here(text: str, voice: str, speed: float) -> tuple[bytes, float]:
    """Kokoro in this interpreter."""
    lang = "b" if re.split(r"[,+]", voice)[0].strip().startswith("b") else "a"
    pipe = _pipeline(lang)
    voice = _blend(pipe, voice)
    try:
        import numpy as np
        import soundfile as sf
    except ImportError as e:
        raise VoiceUnavailable(f"no voice yet — your keeper runs: {INSTALL_HINT}") from e
    chunks = []
    for _gs, _ps, audio in pipe(text, voice=voice, speed=speed, split_pattern=r"\n+"):
        try:
            audio = audio.detach().cpu().numpy()
        except AttributeError:
            audio = np.asarray(audio)
        chunks.append(audio.astype("float32"))
    if not chunks:
        raise VoiceUnavailable("Kokoro produced no sound for that text")
    wav = np.concatenate(chunks)
    buf = io.BytesIO()
    sf.write(buf, wav, 24000, format="WAV", subtype="PCM_16")
    return buf.getvalue(), len(wav) / 24000.0


def to_ogg(wav: bytes) -> bytes | None:
    """WAV → OGG/Opus (what Telegram plays as a voice note). None without ffmpeg."""
    import shutil
    import tempfile
    if not shutil.which("ffmpeg"):
        return None
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "in.wav"
        dst = Path(td) / "out.ogg"
        src.write_bytes(wav)
        try:
            proc = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-c:a", "libopus",
                                   "-b:a", "48k", "-ar", "48000", "-ac", "1", str(dst)],
                                  capture_output=True, timeout=120)
        except Exception:
            return None
        if proc.returncode != 0 or not dst.exists():
            return None
        return dst.read_bytes()


def speak(text: str, voice: str | None = None, speed: float | None = None) -> tuple[bytes, str, float]:
    """The whole act: clean, speak, encode. Returns (bytes, ext, seconds) —
    ext is ".ogg" (a Telegram voice note) or ".wav" when ffmpeg is away."""
    spoken = clean_for_speech(text)
    if not spoken:
        raise VoiceUnavailable("nothing to say aloud in that")
    wav, secs = synthesize(spoken, voice, speed)
    ogg = to_ogg(wav)
    return (ogg, ".ogg", secs) if ogg else (wav, ".wav", secs)


def main() -> None:
    if "--synth" in sys.argv:
        # the sidecar half: text on stdin, wav to --out (see _synthesize_via)
        argv = sys.argv
        voice = argv[argv.index("--voice") + 1] if "--voice" in argv else DEFAULT_VOICE
        speed = float(argv[argv.index("--speed") + 1]) if "--speed" in argv else 1.0
        out = Path(argv[argv.index("--out") + 1])
        text = sys.stdin.buffer.read().decode("utf-8")
        try:
            wav, secs = synthesize_here(text, voice, speed)
        except VoiceUnavailable as e:
            print(str(e), file=sys.stderr)
            sys.exit(2)
        out.write_bytes(wav)
        print(f"{secs:.2f}")
        return
    if "--voices" in sys.argv:
        c = chosen()
        for name, what in VOICES.items():
            mark = "  <- theirs" if c.get("voice") == name else ""
            print(f"  {name:12} {what}{mark}")
        return
    text = "Hello. It's me, and this is what I sound like."
    if "--test" in sys.argv:
        i = sys.argv.index("--test")
        if i + 1 < len(sys.argv):
            text = sys.argv[i + 1]
    voice = None
    if "--voice" in sys.argv:
        voice = sys.argv[sys.argv.index("--voice") + 1]
    try:
        data, ext, secs = speak(text, voice)
    except VoiceUnavailable as e:
        print(f"[voice] {e}")
        sys.exit(1)
    out = config.SHARED_DIR / f"voice-test{ext}"
    out.write_bytes(data)
    print(f"spoke {secs:.1f}s in {voice or chosen().get('voice') or DEFAULT_VOICE} -> {out}")


if __name__ == "__main__":
    main()
