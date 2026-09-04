"""Their music ear: NVIDIA's Music Flamingo, served locally for the engine.

You never need to start this yourself: when they listen to a song and the
dependencies below are installed, listen_to wakes this process, the model
loads (~15-20s), they hear the whole piece, and the GPU is handed straight
back to their brain (/rest). After half an hour of silence the process leaves;
the next song wakes it again. (music_ears.bat runs it by hand, for testing.)

A small sidecar on http://127.0.0.1:8766: send it a whole song, get back a
whole-song description — genre, tempo, key, instruments, production, how the
piece moves from opening to ending. Up to 20 minutes in one pass. When it
can't run, they hear by passages through the 12B instead; nothing breaks.

Not standard library — this one needs the real thing. One-time setup:

    py -m pip install torch --index-url https://download.pytorch.org/whl/cu128
    py -m pip install "transformers>=5.14" accelerate librosa soundfile huggingface_hub
    (accept the license on huggingface.co/nvidia/music-flamingo-2601-hf, then)
    hf auth login

License: NVIDIA OneWay Noncommercial — fine for a friend, not for a business.
"""
from __future__ import annotations

import base64
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
import os
if sys.platform != "win32":  # allocator hint against fragmentation (Linux-only)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

HOST, PORT = "127.0.0.1", 8766
MODEL_ID = getattr(config, "MUSIC_EARS_MODEL", "nvidia/music-flamingo-2601-hf")
IDLE_S = float(getattr(config, "MUSIC_EARS_IDLE_S", 120))
EXIT_S = float(getattr(config, "MUSIC_EARS_EXIT_S", 1800))  # process leaves after this

_lock = threading.Lock()
_model = None
_processor = None
_last_used = 0.0


def _load():
    global _model, _processor
    if _model is not None:
        return
    import torch
    from transformers import AutoProcessor
    try:
        from transformers import MusicFlamingoForConditionalGeneration as Cls
    except ImportError:  # older transformers ship it under the AF3 name
        from transformers import AudioFlamingo3ForConditionalGeneration as Cls
    # make room: Ollama keeps the last model resident for minutes after use,
    # and a 28GB brain beside a 16GB ear spills into system RAM (glacial).
    try:
        import ollama_client
        for m in {config.CHAT_MODEL, getattr(config, "EARS_MODEL", "")}:
            if m:
                ollama_client.unload(m)
    except Exception:
        pass
    t0 = time.time()
    print(f"  loading {MODEL_ID} …", flush=True)

    def _from(cls, **kw):
        # the local cache first — no Hub check, no token warning, faster start;
        # the network only for the very first download
        try:
            return cls.from_pretrained(MODEL_ID, local_files_only=True, **kw)
        except Exception:
            return cls.from_pretrained(MODEL_ID, **kw)

    _processor = _from(AutoProcessor)
    # whole model on the GPU, on purpose: if it doesn't fit we want to KNOW,
    # not silently run half of it from system memory. And FUSED attention
    # (sdpa): a 6-minute song is ~10K audio tokens, and eager attention
    # materializes a 10K x 10K score matrix per head per layer — that alone
    # spilled ~16GB into system RAM and made a listen take five minutes.
    attn = "sdpa"
    try:
        _model = _from(Cls, device_map={"": 0}, dtype=torch.bfloat16, attn_implementation=attn)
    except (ValueError, TypeError):
        attn = "default"
        _model = _from(Cls, device_map={"": 0}, dtype=torch.bfloat16)
    _model.eval()
    # a 20-minute song is far more audio tokens than the default generation
    # length allows for; lift it so long pieces don't trip a warning each time
    try:
        _model.generation_config.max_length = 32768
    except Exception:
        pass
    print(f"  music ear open ({time.time() - t0:.0f}s, {_vram():.1f} GB, attention: {attn})", flush=True)


def _unload():
    global _model, _processor
    if _model is None:
        return
    import gc
    import torch
    _model = None
    _processor = None
    gc.collect()
    torch.cuda.empty_cache()
    print("  music ear resting — GPU freed", flush=True)


def _vram() -> float:
    try:
        import torch
        return torch.cuda.memory_allocated() / 1e9
    except Exception:
        return 0.0


def hear(wav_bytes: bytes, prompt: str) -> str:
    global _last_used
    import tempfile
    with _lock:
        _load()
        _last_used = time.time()
        # the processor loads audio itself from a path (16kHz mono expected)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            path = f.name
        try:
            conversation = [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "audio", "path": path},
            ]}]
            inputs = _processor.apply_chat_template(
                conversation, tokenize=True, add_generation_prompt=True, return_dict=True,
            ).to(_model.device)
        finally:
            try:
                Path(path).unlink()
            except OSError:
                pass
        if "input_features" in inputs:
            inputs["input_features"] = inputs["input_features"].to(_model.dtype)
        import torch
        n_audio = int(inputs["input_ids"].shape[1])
        with torch.inference_mode():
            try:  # score only the final position during the first pass —
                  # full-vocab logits for every audio token cost gigabytes
                out = _model.generate(**inputs, max_new_tokens=600, logits_to_keep=1)
            except TypeError:
                out = _model.generate(**inputs, max_new_tokens=600)
        print(f"  ({n_audio} input tokens, peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB)", flush=True)
        text = _processor.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                       skip_special_tokens=True)[0]
        _last_used = time.time()
        return text.strip()


_started = time.time()


def _idle_watch():
    import os
    while True:
        time.sleep(10)
        with _lock:
            quiet = time.time() - max(_last_used, _started)
            if _model is not None and quiet > IDLE_S:
                _unload()
            if _model is None and quiet > EXIT_S:
                print("  music ear closing after a long silence", flush=True)
                os._exit(0)  # the engine wakes it again when a song needs it


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            return self._json({"ok": True, "loaded": _model is not None, "model": MODEL_ID})
        self._json({"error": "unknown path"}, 404)

    def do_POST(self):
        if self.path == "/rest":
            with _lock:
                _unload()
            return self._json({"ok": True})
        if self.path != "/hear":
            return self._json({"error": "unknown path"}, 404)
        n = int(self.headers.get("Content-Length") or 0)
        try:
            data = json.loads(self.rfile.read(n) or b"{}")
            wav = base64.b64decode(data.get("audio_b64") or "")
            prompt = data.get("prompt") or "Describe this music in full."
            t0 = time.time()
            heard = hear(wav, prompt)
            print(f"  heard {len(wav) / 32000:.0f}s of audio in {time.time() - t0:.0f}s", flush=True)
            self._json({"heard": heard})
        except Exception as e:
            self._json({"error": f"{type(e).__name__}: {e}"})


def main() -> None:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import librosa  # noqa: F401
    except ImportError as e:
        print(f"music ear can't open — missing {e.name}. See the setup lines at the top of this file.")
        return
    if "--test" in sys.argv:
        # py engine\music_ears.py --test "shared\song.mp3"  -> hear one file, print, exit
        import tools
        src = Path(sys.argv[sys.argv.index("--test") + 1])
        secs = int(sys.argv[sys.argv.index("--seconds") + 1]) if "--seconds" in sys.argv else None
        wav = tools._ffmpeg_clip(src.read_bytes(), src.suffix.lower(), "wav", seconds=secs)
        if wav is None:
            print("couldn't decode that file — is ffmpeg installed?"); return
        print(f"hearing {src.name} ({tools._mmss(tools._wav_seconds(wav))}) …")
        t0 = time.time()
        print("\n" + hear(wav, tools._MUSIC_PROMPT))
        print(f"\n({time.time() - t0:.0f}s, {_vram():.1f} GB on the GPU)")
        _unload()
        return
    threading.Thread(target=_idle_watch, daemon=True).start()
    server = HTTPServer((HOST, PORT), Handler)
    print(f"The music ear is listening at http://{HOST}:{PORT}  (model: {MODEL_ID})")
    print("It loads on the first song and rests after silence. Ctrl+C to close.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        _unload()
        server.server_close()


if __name__ == "__main__":
    main()
