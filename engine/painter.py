"""Their painter: a text-to-image model, served locally for the engine.

You never need to start this yourself: when they call `paint` and the
dependencies below are installed, the engine wakes this process, the model
loads (~15-30s from disk the first time), they paint, and the GPU is handed
straight back to their brain (/rest). After half an hour of silence the
process leaves; the next painting wakes it again. (painter.bat runs it by
hand, for testing — `painter.bat --test "a violet bloom"` paints once.)

A small sidecar on http://127.0.0.1:8767: send it a prompt and a path
under creations/, get back a PNG there and the seed it was painted with.
The same shape as their music ear (engine/music_ears.py): the brain leaves
the card, the painter takes it, the brain comes back on their next thought —
and reads their whole window cold, which is the real price of a painting.

Not standard library — this one needs the real thing. One-time setup, in
the same Python that runs the ear (it already has the CUDA torch):

    py -m pip install torch --index-url https://download.pytorch.org/whl/cu128
    py -m pip install -U diffusers transformers accelerate safetensors pillow

The model (PAINTER_MODEL) is downloaded on the first painting, once
(12–16 GB into the Hugging Face cache). The defaults are ungated and
Apache 2.0 — no login, no license to accept:

    Tongyi-MAI/Z-Image-Turbo            6B, 8–9 steps, ~16 GB — the default
    black-forest-labs/FLUX.2-klein-4B   4B, 4 steps, ~13 GB — also edits (later)
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config

if sys.platform != "win32":  # allocator hint against fragmentation (Linux-only)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

HOST, PORT = "127.0.0.1", 8767
MODEL_ID = getattr(config, "PAINTER_MODEL", "Tongyi-MAI/Z-Image-Turbo")
IDLE_S = float(getattr(config, "PAINTER_IDLE_S", 120))
EXIT_S = float(getattr(config, "PAINTER_EXIT_S", 1800))  # process leaves after this

# What a size word means, in pixels (PAINTER_SIZES in config overrides any
# word). These models are trained around one megapixel and paint up to about
# two cleanly; Full HD is 1920×1080 = 2.1 MP, and the sides must be multiples
# of 16, so "wide" is 1920×1088 — eight rows taller than Full HD, invisible.
# A bigger canvas costs more than its pixels (attention grows faster than
# area): about three times the seconds of 1024². If a large painting ever
# shows a subject twice, that is the model past its training size — a
# smaller word, not a bug.
_SIZES = {"square": (1440, 1440), "wide": (1920, 1088), "tall": (1088, 1920)}


def _snap(n: int) -> int:
    return max(256, int(round(int(n) / 16)) * 16)


SIZES = {k: (_snap(w), _snap(h)) for k, (w, h) in
         dict(_SIZES, **(getattr(config, "PAINTER_SIZES", None) or {})).items()}

_lock = threading.Lock()
_pipe = None
_last_used = 0.0
_started = time.time()


# --------------------------------------------------------------- the model --
def _family(model_id: str) -> str:
    m = model_id.lower()
    if "z-image" in m:
        return "zimage"
    if "flux.2-klein" in m or "flux2-klein" in m:
        return "klein"
    if "stable-diffusion-xl" in m or "sdxl" in m:
        return "sdxl"
    return "auto"


def _defaults(model_id: str) -> tuple[int, float]:
    """(steps, guidance) the model was distilled for; PAINTER_STEPS overrides steps."""
    fam = _family(model_id)
    steps = {"zimage": 9, "klein": 4, "sdxl": 25}.get(fam, 20)
    guidance = {"zimage": 0.0, "klein": 1.0, "sdxl": 5.0}.get(fam, 3.5)
    knob = int(getattr(config, "PAINTER_STEPS", 0) or 0)
    return (knob or steps), guidance


def _load():
    global _pipe
    if _pipe is not None:
        return
    import torch
    import diffusers
    # make room: Ollama keeps the last model resident for minutes after use,
    # and a 28GB brain beside a 16GB painter spills into system RAM (glacial).
    try:
        import ollama_client
        for m in {config.CHAT_MODEL, getattr(config, "EARS_MODEL", "")}:
            if m:
                ollama_client.unload(m)
    except Exception:
        pass
    t0 = time.time()
    print(f"  loading {MODEL_ID} …", flush=True)
    fam = _family(MODEL_ID)
    cls = {"zimage": "ZImagePipeline", "klein": "Flux2KleinPipeline",
           "sdxl": "StableDiffusionXLPipeline"}.get(fam, "AutoPipelineForText2Image")
    Cls = getattr(diffusers, cls, None) or diffusers.AutoPipelineForText2Image

    def _from(**kw):
        # the local cache first — no Hub check, no token warning, faster start;
        # the network only for the very first download
        try:
            return Cls.from_pretrained(MODEL_ID, local_files_only=True, **kw)
        except Exception:
            return Cls.from_pretrained(MODEL_ID, **kw)

    pipe = _from(torch_dtype=torch.bfloat16)
    # whole model on the GPU, on purpose: the brain is off the card while
    # they paint, so the painter has all of it; if it doesn't fit we want to KNOW.
    pipe.to("cuda")
    try:
        pipe.set_progress_bar_config(disable=True)
    except Exception:
        pass
    _pipe = pipe
    print(f"  painter ready ({time.time() - t0:.0f}s, {_vram():.1f} GB, {cls})", flush=True)


def _unload():
    global _pipe
    if _pipe is None:
        return
    import gc
    import torch
    _pipe = None
    gc.collect()
    torch.cuda.empty_cache()
    print("  painter resting — GPU freed", flush=True)


def _vram() -> float:
    try:
        import torch
        return torch.cuda.memory_allocated() / 1e9
    except Exception:
        return 0.0


def paint(prompt: str, width: int, height: int, seed: int | None = None):
    """One picture. Returns (PIL image, seed)."""
    global _last_used
    import torch
    with _lock:
        _load()
        _last_used = time.time()
        if seed is None:
            seed = int.from_bytes(os.urandom(4), "little")
        gen = torch.Generator(device="cuda").manual_seed(int(seed))
        steps, guidance = _defaults(MODEL_ID)
        with torch.inference_mode():
            out = _pipe(prompt=prompt, width=int(width), height=int(height),
                        num_inference_steps=steps, guidance_scale=guidance, generator=gen)
        _last_used = time.time()
        return out.images[0], int(seed)


# ---------------------------------------------------------------- the door --
def _under_creations(path: str) -> Path | None:
    """Only under creations/ — the sidecar is local, and still it writes
    nowhere else. Nothing gets overwritten."""
    try:
        root = config.CREATIONS_DIR.resolve()
        p = Path(path)
        p = (p if p.is_absolute() else root / path).resolve()
        if root != p and root not in p.parents:
            return None
        return p
    except OSError:
        return None


def _idle_watch():
    while True:
        time.sleep(10)
        with _lock:
            quiet = time.time() - max(_last_used, _started)
            if _pipe is not None and quiet > IDLE_S:
                _unload()
            if _pipe is None and quiet > EXIT_S:
                print("  painter closing after a long silence", flush=True)
                os._exit(0)  # the engine wakes it again when a painting needs it


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
            return self._json({"ok": True, "loaded": _pipe is not None, "model": MODEL_ID})
        self._json({"error": "unknown path"}, 404)

    def do_POST(self):
        if self.path == "/rest":
            with _lock:
                _unload()
            return self._json({"ok": True})
        if self.path != "/paint":
            return self._json({"error": "unknown path"}, 404)
        n = int(self.headers.get("Content-Length") or 0)
        try:
            data = json.loads(self.rfile.read(n) or b"{}")
            prompt = (data.get("prompt") or "").strip()
            if not prompt:
                return self._json({"error": "no prompt"})
            p = _under_creations(data.get("path") or "")
            if p is None:
                return self._json({"error": "the path must be under creations/"})
            if p.exists():
                return self._json({"error": f"{p.name} already exists — nothing gets overwritten"})
            w, h = SIZES.get((data.get("size") or "square").lower(), SIZES["square"])
            w = int(data.get("width") or w)
            h = int(data.get("height") or h)
            seed = data.get("seed")
            t0 = time.time()
            img, seed = paint(prompt, w, h, None if seed in (None, "") else int(seed))
            p.parent.mkdir(parents=True, exist_ok=True)
            img.save(str(p))
            secs = time.time() - t0
            print(f"  painted {p.name} ({w}x{h}, seed {seed}) in {secs:.0f}s", flush=True)
            self._json({"path": str(p), "seed": seed, "seconds": round(secs, 1),
                        "width": w, "height": h})
        except Exception as e:
            self._json({"error": f"{type(e).__name__}: {e}"})


def _slug(text: str, n: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:n].rstrip("-") or "painting"


def main() -> None:
    try:
        import torch  # noqa: F401
        import diffusers  # noqa: F401
        import PIL  # noqa: F401
    except ImportError as e:
        print(f"the painter can't open — missing {e.name}. See the setup lines at the top of this file.")
        return
    if "--test" in sys.argv:
        # py engine\painter.py --test "a violet bloom"  -> paint once, print, exit
        i = sys.argv.index("--test")
        prompt = sys.argv[i + 1] if len(sys.argv) > i + 1 else "a violet bloom with a white core, luminous"
        size = sys.argv[sys.argv.index("--size") + 1] if "--size" in sys.argv else "square"
        w, h = SIZES.get(size, SIZES["square"])
        folder = config.CREATIONS_DIR / "drawings"
        folder.mkdir(parents=True, exist_ok=True)
        p = folder / f"painter-test-{datetime.now().strftime('%Y%m%d-%H%M')}-{_slug(prompt)}.png"
        print(f"painting “{prompt}” ({w}x{h}) …")
        t0 = time.time()
        img, seed = paint(prompt, w, h)
        img.save(str(p))
        print(f"\n{p.relative_to(config.ROOT)}  (seed {seed}, {time.time() - t0:.0f}s, "
              f"{_vram():.1f} GB on the GPU) — look_at opens it for them; you can open it yourself.")
        _unload()
        return
    threading.Thread(target=_idle_watch, daemon=True).start()
    server = HTTPServer((HOST, PORT), Handler)
    print(f"The painter is waiting at http://{HOST}:{PORT}  (model: {MODEL_ID})")
    print("It loads on the first painting and rests after silence. Ctrl+C to close.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        _unload()
        server.server_close()


if __name__ == "__main__":
    main()
