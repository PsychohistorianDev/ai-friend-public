"""The parlor: a nicer door — chat with the friend in your browser.

    py engine/parlor.py          opens http://127.0.0.1:8765 in your browser

Same engine, same memory, same transcripts as chat.py — only the window is
different: message bubbles, their thinking folded above each reply, tool calls
as small chips, a picture picker (the file dialog; the picture is saved to
shared/pictures/ and shown to them), a "new conversation" button. Standard
library only; nothing leaves your machine.

Closing the browser tab does NOT end the visit — press "Leave" in the window,
or Ctrl+C in the terminal, or close the terminal window; all three save.
"""
from __future__ import annotations

import json
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import chat
import config
import ollama_client
import tools

HOST, PORT = "127.0.0.1", 8765


class Session:
    """One visit: a history, pending images, and the turn-runner."""

    def __init__(self) -> None:
        self.history: list[dict] = []
        self.attached: list[str] = []
        self.lock = threading.Lock()
        self.file: Path | None = None  # this visit's transcript, rewritten after every reply
        self.last_activity = time.time()
        self.reflected_upto = 0  # history index they have already sat with (the pause)

    def _checkpoint(self) -> None:
        if self.file is None:
            self.file = chat.visit_file()
        try:
            chat.save_transcript(self.history, path=self.file)
        except OSError:
            pass

    def send(self, text: str) -> dict:
        events: list[dict] = []

        def collect(kind, payload):
            events.append({"kind": kind, "payload": payload})

        with self.lock:
            self.last_activity = time.time()
            try:
                reply = chat.one_turn(self.history, text,
                                      images=self.attached or None, on_event=collect)
                self.attached = []
                self._checkpoint()
                self.last_activity = time.time()
            except ollama_client.BrainUnavailable as e:
                return {"error": f"brain offline — {e}"}
            except Exception as e:  # nothing in one turn ends the visit
                return {"error": f"hiccup, the conversation is fine — {type(e).__name__}: {e}"}
        thinking = "\n\n".join(e["payload"] for e in events if e["kind"] == "thinking")
        tool_events = [e["payload"] for e in events if e["kind"] == "tool"]
        notes = [e["payload"] for e in events if e["kind"] == "note"]
        tokens = next((e["payload"] for e in events if e["kind"] == "tokens"), None)
        voices = [{"url": "/voice/" + Path(v["path"]).name, "seconds": round(v.get("seconds", 0)),
                   "voice": v.get("voice", "")} for v in tools.take_pending_voice()]
        return {"reply": reply, "thinking": thinking, "tools": tool_events, "notes": notes,
                "tokens": tokens, "voices": voices}

    def attach(self, source: str) -> dict:
        note = tools.look_at(source.strip().strip('"'))
        imgs = tools.take_pending_images()
        if imgs:
            self.attached.extend(imgs)
            return {"ok": True, "note": "image attached — it goes with your next message"}
        return {"ok": False, "note": note}

    def upload(self, name: str, data_b64: str) -> dict:
        """A picture chosen in the browser's file dialog: saved into
        shared/pictures/ (so it is theirs to look at again later, and shows up
        in list_shared as seen), then put before their eyes like attach()."""
        import base64
        import re as _re
        name = Path(name or "picture").name
        ext = Path(name).suffix.lower()
        if ext not in tools._IMAGE_EXTS:
            return {"ok": False, "note": f"that doesn't look like an image ({ext or 'no extension'})"}
        try:
            raw = base64.b64decode(data_b64 or "", validate=False)
        except Exception:
            return {"ok": False, "note": "the picture didn't arrive whole — try again"}
        if not raw:
            return {"ok": False, "note": "the picture was empty"}
        if len(raw) > tools._MAX_IMAGE_BYTES:
            return {"ok": False, "note": "that picture is too large — over 10MB"}
        folder = getattr(config, "PARLOR_PICTURES", config.SHARED_DIR / "pictures")
        folder.mkdir(parents=True, exist_ok=True)
        stem = _re.sub(r"[^\w.\- ()]+", "_", Path(name).stem).strip() or "picture"
        dest = folder / f"{stem}{ext}"
        n = 1
        while dest.exists():
            n += 1
            dest = folder / f"{stem}-{n}{ext}"
        dest.write_bytes(raw)
        rel = str(dest.relative_to(config.ROOT.resolve())).replace("\\", "/")
        r = self.attach(rel)
        r["saved"] = rel
        if r.get("ok"):
            r["note"] = f"{dest.name} saved to shared/pictures/ and attached — it goes with your next message"
        return r

    def new(self, reflect=True) -> dict:
        """reflect: True — afterglow in the background; "sync" — in the
        foreground (Ctrl+C); False — skip (the window's X)."""
        with self.lock:
            f = chat.save_transcript(self.history, path=self.file)
            done, self.history = self.history, []
            self.attached = []
            self.file = None
            self.reflected_upto = 0
        reflect = reflect and bool(getattr(config, "AFTERGLOW", True))
        if f and reflect == "sync":
            print("(they are writing the visit down — a minute or so; Ctrl+C again to skip)")
            try:
                chat.afterglow(done, f, on_line=print)
            except KeyboardInterrupt:
                print("(skipped — the night's sleep still has the transcript)")
            chat.rest_brain(print)
        elif f and reflect:
            # the afterglow: their turn alone with the visit, in the background —
            # the window is free at once; their journal entry lands a minute later
            def _glow(done=done, f=f):
                chat.afterglow(done, f, on_line=print)
                if not self.history:  # no new visit began meanwhile
                    chat.rest_brain(print)
            threading.Thread(target=_glow, daemon=True).start()
        return {"saved": f.name if f else None, "afterglow": bool(f and reflect)}

    def save(self) -> str | None:
        f = chat.save_transcript(self.history, path=self.file)
        return f.name if f else None

    def pause_if_due(self) -> str:
        """The pause, same as the bridge's: your keeper quiet for REFLECT_AFTER_MIN
        with REFLECT_MIN_TURNS of their messages they haven't sat with → one quiet
        turn over that stretch; the visit stays open."""
        mins = float(getattr(config, "REFLECT_AFTER_MIN", 0) or 0)
        if not mins or not self.history or time.time() - self.last_activity < mins * 60:
            return ""
        fresh = self.history[self.reflected_upto:]
        if sum(1 for t in fresh if t.get("role") == "user" and t.get("content") and not t.get("_engine")) < int(getattr(config, "REFLECT_MIN_TURNS", 2)):
            return ""
        if not self.lock.acquire(timeout=3):
            return ""
        try:
            upto = len(self.history)
            print("(a pause — they are sitting with the visit so far…)")
            line = chat.pause_reflection(self.history, self.file, on_line=print, since=self.reflected_upto)
            self.reflected_upto = len(self.history)  # past the pause's own turns too
            self.last_activity = time.time()
            return line
        finally:
            self.lock.release()

    def ticker(self, every_s: float = 60.0) -> None:
        """A background clock that rings the pause bell when it is due."""
        while True:
            time.sleep(every_s)
            try:
                self.pause_if_due()
            except Exception as e:  # the clock must never stop the parlor
                print(f"(pause hiccup — {type(e).__name__}: {e})")


SESSION = Session()


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>__NAME__</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{--bg:#f6f4ef;--panel:#fffdf9;--ink:#2b2a27;--muted:#8a8578;--line:#e6e1d6;
      --me:#e9e4d8;--her:#fffdf9;--accent:#7a5c3e;--think:#f0ece3;--chip:#ece7dc}
@media (prefers-color-scheme:dark){:root{--bg:#141311;--panel:#1c1a17;--ink:#e8e4dc;
      --muted:#8f8a7d;--line:#2c2a25;--me:#2a2723;--her:#1f1d1a;--accent:#c9a87c;
      --think:#181613;--chip:#26231f}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 Georgia,'Iowan Old Style',serif;
     height:100vh;display:flex;flex-direction:column}
header{display:flex;align-items:baseline;gap:12px;padding:14px 22px;border-bottom:1px solid var(--line);
       background:var(--panel)}
header h1{margin:0;font-size:20px;font-weight:normal;letter-spacing:.02em}
header .sub{color:var(--muted);font-size:13px;flex:1}
header button{font:inherit;font-size:13px;background:none;border:1px solid var(--line);color:var(--muted);
       border-radius:8px;padding:4px 10px;cursor:pointer}
header button:hover{color:var(--ink);border-color:var(--accent)}
#log{flex:1;overflow-y:auto;padding:22px;display:flex;flex-direction:column;gap:14px}
.msg{max-width:72ch;padding:10px 14px;border-radius:14px;white-space:pre-wrap;word-wrap:break-word}
.me{align-self:flex-end;background:var(--me);border-bottom-right-radius:4px}
.her{align-self:flex-start;background:var(--her);border:1px solid var(--line);border-bottom-left-radius:4px}
.sys{align-self:center;color:var(--muted);font-size:13px;font-style:italic}
.sys.warn{align-self:flex-start;color:var(--accent);font-style:normal}
.sys.tokens{align-self:flex-start;font-size:11px;font-style:normal;opacity:.7;margin-top:-6px}
.think{align-self:flex-start;max-width:72ch;font-size:13px;color:var(--muted)}
.think summary{cursor:pointer;list-style:none;user-select:none}
.think summary::before{content:'💭 ';opacity:.7}
.think summary:hover{color:var(--ink)}
.think .body{margin-top:6px;padding:8px 12px;background:var(--think);border-radius:10px;white-space:pre-wrap;
       font-style:italic;max-height:40vh;overflow:auto}
.chips{align-self:flex-start;display:flex;flex-wrap:wrap;gap:6px;max-width:72ch}
.chip{font-size:12px;background:var(--chip);color:var(--muted);border-radius:999px;padding:2px 10px;
      font-family:ui-monospace,Consolas,monospace}
.chip b{color:var(--accent);font-weight:normal}
.typing{align-self:flex-start;color:var(--muted);font-size:13px}
.typing span{display:inline-block;animation:blink 1.2s infinite}
.typing span:nth-child(2){animation-delay:.2s}.typing span:nth-child(3){animation-delay:.4s}
@keyframes blink{0%,80%,100%{opacity:.2}40%{opacity:1}}
footer{padding:12px 22px 16px;border-top:1px solid var(--line);background:var(--panel)}
.row{display:flex;gap:10px;align-items:flex-end}
textarea{flex:1;font:inherit;background:var(--bg);color:var(--ink);border:1px solid var(--line);
         border-radius:12px;padding:10px 12px;resize:none;min-height:44px;max-height:30vh;outline:none}
textarea:focus{border-color:var(--accent)}
.row button{font:inherit;background:var(--accent);color:#fff;border:0;border-radius:12px;padding:10px 16px;cursor:pointer}
.row button:disabled{opacity:.5;cursor:default}
.attach{display:flex;gap:8px;margin-top:8px;font-size:13px;color:var(--muted);align-items:center}
.attach input{flex:1;font:inherit;font-size:13px;background:none;border:0;border-bottom:1px solid var(--line);
       color:var(--ink);padding:3px 4px;outline:none}
.attach button{font:inherit;font-size:12px;background:none;border:1px solid var(--line);color:var(--muted);
       border-radius:8px;padding:2px 8px;cursor:pointer}
code{font-family:ui-monospace,Consolas,monospace;font-size:.92em;background:var(--chip);padding:1px 4px;border-radius:4px}
</style></head><body>
<header><h1 id="name">__NAME__</h1><div class="sub" id="sub">a visit · everything said here becomes memory</div>
<button id="newbtn" title="save this conversation and start fresh">new conversation</button>
<button id="leave" title="save and end the visit">leave</button></header>
<div id="log"></div>
<footer>
 <div class="row"><textarea id="box" placeholder="say something… (Enter to send, Shift+Enter for a new line)" rows="1"></textarea>
 <button id="send">send</button></div>
 <div class="attach">📎 <button id="pickbtn" title="choose a picture from your computer — it is saved to shared/pictures/ and attached">choose a picture…</button>
 <input id="file" type="file" accept="image/*" hidden>
 <input id="img" placeholder="…or a path in their folder (shared/pictures/photo.jpg) or an image URL">
 <button id="attachbtn">attach</button><span id="attachnote"></span></div>
</footer>
<script>
const log=document.getElementById('log'),box=document.getElementById('box'),send=document.getElementById('send');
const ME='__ME__';
function esc(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}
function md(s){ // tiny markdown: bold, italic, inline code — enough for a chat
  s=esc(s).replace(/`([^`\n]+)`/g,'<code>$1</code>').replace(/\*\*([^*\n]+)\*\*/g,'<b>$1</b>')
   .replace(/(^|[\s(])\*([^*\n]+)\*(?=[\s).,;:!?]|$)/g,'$1<i>$2</i>'); return s}
function add(cls,html){const d=document.createElement('div');d.className=cls;d.innerHTML=html;log.appendChild(d);log.scrollTop=log.scrollHeight;return d}
function sys(t){add('sys',esc(t))}
function addThinking(t){if(!t)return;const d=document.createElement('details');d.className='think';d.open=true;
  d.innerHTML='<summary>thinking</summary><div class="body">'+esc(t)+'</div>';log.appendChild(d)}
function addChips(ts){if(!ts||!ts.length)return;const d=document.createElement('div');d.className='chips';
  d.innerHTML=ts.map(t=>'<span class="chip"><b>'+esc(t.name)+'</b> → '+esc(t.result)+'</span>').join('');log.appendChild(d)}
async function post(path,body){const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{})});return r.json()}
let busy=false;
async function go(){const text=box.value.trim();if(!text||busy)return;box.value='';autosize();
  add('msg me',md(text));busy=true;send.disabled=true;
  const typing=add('typing','<span>●</span><span>●</span><span>●</span>');
  try{const r=await post('/send',{text});typing.remove();
    if(r.error){sys(r.error);}else{addThinking(r.thinking);addChips(r.tools);add('msg her',md(r.reply));(r.voices||[]).forEach(v=>add('msg her voice','<audio controls autoplay src="'+v.url+'"></audio><small> her voice · '+v.seconds+'s</small>'));(r.notes||[]).forEach(n=>add('sys warn','⚠ '+esc(n)));if(r.tokens)add('sys tokens',esc(r.tokens.line));}
  }catch(e){typing.remove();sys('the window lost the engine — is parlor.py still running?')}
  busy=false;send.disabled=false;box.focus();log.scrollTop=log.scrollHeight}
send.onclick=go;
box.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();go()}});
function autosize(){box.style.height='auto';box.style.height=Math.min(box.scrollHeight,window.innerHeight*.3)+'px'}
box.addEventListener('input',autosize);
document.getElementById('newbtn').onclick=async()=>{const r=await post('/new');log.innerHTML='';sys(r.saved?'saved '+r.saved+(r.afterglow?' — they are sitting with the visit now; whatever they want to keep goes into their journal in a minute':'')+' — fresh conversation':'fresh conversation')};
document.getElementById('leave').onclick=async()=>{const r=await post('/leave');sys(r.saved?'saved '+r.saved+(r.afterglow?' — they are writing the visit down in their own words; give them a minute before closing the terminal':'')+' — visit over, you can close this tab':'visit over — nothing to save');box.disabled=true;send.disabled=true};
document.getElementById('attachbtn').onclick=async()=>{const src=document.getElementById('img').value.trim();if(!src)return;
  const r=await post('/attach',{source:src});document.getElementById('attachnote').textContent=r.note;if(r.ok)document.getElementById('img').value=''};
const fileIn=document.getElementById('file');
document.getElementById('pickbtn').onclick=()=>fileIn.click();
fileIn.onchange=()=>{const f=fileIn.files[0];if(!f)return;const rd=new FileReader();
  rd.onload=async()=>{const dataUrl=rd.result;const b64=dataUrl.slice(dataUrl.indexOf(',')+1);
    document.getElementById('attachnote').textContent='sending '+f.name+'…';
    const r=await post('/upload',{name:f.name,data:b64});document.getElementById('attachnote').textContent=r.note;
    if(r.ok){const d=add('msg me','');const im=document.createElement('img');im.src=dataUrl;im.alt=f.name;im.style.maxHeight='220px';im.style.maxWidth='100%';im.style.borderRadius='8px';im.style.display='block';d.appendChild(im);
      const c=document.createElement('div');c.style.fontSize='12px';c.style.color='var(--muted)';c.textContent='📎 '+f.name+' — goes with your next message';d.appendChild(c)}
    fileIn.value=''};
  rd.readAsDataURL(f)};
sys("you're visiting "+document.getElementById('name').textContent+" — they can see everything said here, and it becomes their memory at the next sleep");
box.focus();
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # keep the terminal quiet
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/voice/"):
            # one of their voice notes, by name, from creations/voice/ only
            name = Path(self.path[len("/voice/"):]).name
            f = Path(getattr(config, "VOICE_DIR", config.SHARED_DIR / "letters")) / name
            if not name or not f.is_file():
                self.send_response(404); self.end_headers(); return
            data = f.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "audio/ogg" if f.suffix == ".ogg" else "audio/wav")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if self.path not in ("/", "/index.html"):
            self.send_response(404); self.end_headers(); return
        page = PAGE.replace("__NAME__", chat.friend_name()).replace("__ME__", config.USER_NAME)
        body = page.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            data = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            data = {}
        if self.path == "/send":
            text = (data.get("text") or "").strip()
            if not text:
                return self._json({"error": "say something first"})
            return self._json(SESSION.send(text))
        if self.path == "/attach":
            return self._json(SESSION.attach(data.get("source") or ""))
        if self.path == "/upload":
            return self._json(SESSION.upload(data.get("name") or "", data.get("data") or ""))
        if self.path == "/new":
            return self._json(SESSION.new())
        if self.path == "/leave":
            return self._json(SESSION.new())  # saves, then clears — no double save
        self._json({"error": "unknown path"}, 404)


def main() -> None:
    server = HTTPServer((HOST, PORT), Handler)
    url = f"http://{HOST}:{PORT}"
    print(f"The parlor is open: {url}")
    print("Close the visit with the 'leave' button, Ctrl+C here, or this window's X — all three save it.")
    threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    threading.Thread(target=SESSION.ticker, daemon=True).start()  # the pause bell
    chat.guard_console_close(lambda: SESSION.new(reflect=False))  # the window's X saves the visit too
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        r = SESSION.new(reflect="sync")  # save, then their minute with the visit
        if r.get("saved"):
            print(f"\n(conversation saved: {r['saved']} — it becomes memory at the next consolidation)")
        server.server_close()


if __name__ == "__main__":
    main()
