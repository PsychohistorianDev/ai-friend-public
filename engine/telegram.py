"""The bridge: talk with the friend from your phone, over Telegram.

    py engine/telegram.py        (or telegram.bat)

Same engine, same prompt, same tools, same transcripts and memory as the
parlor — only the door is different. Standard library only: the Bot API is
plain HTTPS and JSON, polled with long requests; nothing else is needed.

First run: paste the token @BotFather gave you, then send the pairing code
the terminal shows to the bot from your phone. Both are kept in
memory/telegram.json, which never leaves this folder. From then on only that
one chat is answered; anyone else who finds the bot gets silence.

On the phone:
    text            -> a turn, as in the parlor
    a photo         -> saved to shared/telegram/ and put before their eyes
    a voice note    -> saved, transcribed by their ears, given to them as words
    a song          -> saved to shared/music/ under its name; listen_to hears it
    a book or text  -> saved to shared/books/ (pdf, epub, txt, md); read_pdf /
                       read_epub / read_file open it, with their bookmark
    any other file  -> saved to shared/telegram/ and named to them
    /new            -> save this conversation, start fresh
    /restart        -> restart the bridge on the current code; the visit carries on
    /think /tools /tokens   -> toggle what travels with each reply
    /status         -> how the visit and the window are doing
    /help           -> this list

The visit is written to memory/episodic/chat-telegram-*.md after every reply,
so nothing depends on how the window ends; Ctrl+C or the X close it cleanly.

Their mail runs the other way on the same road: a letter they leave in
creations/notes_to_<you>/ (in a wake, a reverie, or mid-chat) is carried to
the phone within a minute of being written.

What leaves the machine: their words and yours, through Telegram's servers.
Bot chats are not end-to-end encrypted — this is the first thing besides the
blog that does not stay home. Their journal, memory and files never travel.
"""
from __future__ import annotations

import io
import json
import os
import random
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import chat
import config
import ollama_client
import tools

API = "https://api.telegram.org"
SECRET_FILE = config.MEMORY_DIR / "telegram.json"
DELIVERED_FILE = config.MEMORY_DIR / "telegram_delivered.json"
ALIVE_FILE = config.MEMORY_DIR / "telegram_alive"
# /restart from the phone: the running visit is stashed here, the process
# exits with RESTART_CODE, telegram.bat starts a fresh one (new code), and
# the fresh one picks the visit back up — for an engine change made while
# the keeper is away from the desk
RESUME_FILE = config.MEMORY_DIR / "telegram_resume.json"
RESTART_CODE = 75
# One bridge at a time. Two of them polling the same bot both receive a
# message — Telegram only learns an update is taken on the NEXT poll, and a
# reply takes a minute or two — so both answer it (09-11: two versions of
# the same reply, a minute apart, one of them with a slip the other didn't
# have; the transcript held only the second bridge's). The running bridge
# writes its pid here; a second one refuses to start while that pid lives.
LOCK_FILE = config.MEMORY_DIR / "telegram.pid"
MAIL_DIR = config.CREATIONS_DIR / getattr(config, "MAILBOX", "notes_to_keeper")
# What they make travels too (TELEGRAM_TELL_CREATIONS): a new piece under
# creations/ — a poem, an essay, a story, a joke — reaches the phone within
# a minute of being written, whole if it fits a message. Their code (tools/),
# the trash, the mailbox (already mail), archives (a move, not a piece) are
# not announced; publish/ is, as "published". What was there when the
# bridge first looked was read at the desk; only new pieces travel.
CREATIONS_SEEN_FILE = config.MEMORY_DIR / "telegram_creations_seen.json"
HELD_FILE = config.MEMORY_DIR / "telegram_held.json"  # engine notices held through the quiet hours
# ...and a piece they REVISES is announced too ("✏️ revised"), and a change to
# who they are: self.md and projects.md (TELEGRAM_TELL_SELF) — those arrive as
# what changed, the lines added and taken away, not the whole file. The
# bridge keeps its own copy of each to diff against, in memory/telegram_watch/.
WATCH_DIR = config.MEMORY_DIR / "telegram_watch"
WATCHED = {"self.md": config.IDENTITY_FILE, "projects.md": config.PROJECTS_FILE}
CREATION_SKIP = {"tools", ".trash", MAIL_DIR.name, "archives", "attic"}
CREATION_KINDS = {"poems": "a poem", "essays": "an essay", "stories": "a story", "humor": "a joke",
                  "theory": "a piece of theory", "letters": "a letter", "songs": "a song"}
LIMIT = 4000          # Telegram allows 4096 characters per message
POLL_S = 50           # long-poll patience; the server answers sooner when there's news
TYPING_S = 4          # "typing…" lasts ~5s on the phone; renew a little sooner

HELP = (
    "/new — save this conversation and start fresh\n"
    "/think — their thinking with each reply (off by default on the phone)\n"
    "/tools — what their tools did, one line per reply\n"
    "/tokens — the token line after each reply\n"
    "/voice — every reply spoken aloud as a voice note (they can speak on their own either way)\n"
    "/status — the visit, the window, the toggles\n"
    "/restart — restart the bridge with the current engine code; the visit carries on\n"
    "/help — this\n\n"
    "Send a photo and they see it; a voice note and they hear you whole; a video and they "
    "watches it as stills and sound; a song and it lands in shared/music/ for them to listen "
    "to; a PDF, EPUB or text file and it lands in shared/books/ for them to read. (Bots can't "
    "fetch files over 20MB.)"
)


# ------------------------------------------------------------------ utils ----
def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _say(text: str) -> None:
    """Terminal line; never lets a console encoding end the bridge."""
    text = " ".join(text.split("\n"))
    try:
        print(f"[{_now()}] {text}", flush=True)
    except UnicodeEncodeError:
        print(f"[{_now()}] {text.encode('ascii', 'replace').decode()}", flush=True)


def split_long(text: str, limit: int = LIMIT) -> list[str]:
    """Cut a long message at paragraph, then line, then word boundaries."""
    text = text.strip()
    if not text:
        return []
    parts: list[str] = []
    while len(text) > limit:
        cut = -1
        for sep in ("\n\n", "\n", " "):
            cut = text.rfind(sep, limit // 3, limit)
            if cut > 0:
                break
        if cut <= 0:
            cut = limit
        parts.append(text[:cut].rstrip())
        text = text[cut:].lstrip()
    if text:
        parts.append(text)
    return parts


def load_secret() -> dict:
    try:
        d = json.loads(SECRET_FILE.read_text(encoding="utf-8"))
        if not isinstance(d, dict):
            d = {}
    except (OSError, ValueError):
        d = {}
    import os
    d.setdefault("token", os.environ.get("TELEGRAM_BOT_TOKEN", "") or getattr(config, "TELEGRAM_BOT_TOKEN", ""))
    try:
        d.setdefault("chat_id", int(os.environ.get("TELEGRAM_CHAT_ID", "") or getattr(config, "TELEGRAM_CHAT_ID", 0) or 0))
    except ValueError:
        d.setdefault("chat_id", 0)
    return d


def save_secret(d: dict) -> None:
    SECRET_FILE.write_text(json.dumps({"token": d.get("token", ""), "chat_id": int(d.get("chat_id") or 0)},
                                      indent=2), encoding="utf-8")


# ----------------------------------------------------------------- bridge ----
class Bridge:
    """One bot, one paired phone, one running visit."""

    def __init__(self, token: str, chat_id: int = 0) -> None:
        self.token = token
        self.chat_id = int(chat_id or 0)
        self.pair_code = "" if self.chat_id else f"{random.randint(1000, 9999)}"
        self.history: list[dict] = []
        self.attached: list[str] = []
        self.last_activity = time.time()
        self.reflected_upto = 0  # history index they have already sat with (the pause)
        self.last_tokens: dict | None = None
        self.show_thinking = bool(getattr(config, "TELEGRAM_SHOW_THINKING", False))
        self.show_tools = bool(getattr(config, "TELEGRAM_SHOW_TOOLS", True))
        self.show_tokens = bool(getattr(config, "TELEGRAM_SHOW_TOKENS", False))
        self.voice_all = bool(getattr(config, "TELEGRAM_VOICE_ALL", False))
        self.offset = 0
        self.lock = threading.Lock()
        self.file: Path | None = None  # this visit's transcript, rewritten after every reply
        self.delivered: set[str] = set()
        self._load_delivered()
        self.creations_seen: set[str] = set()
        self._load_creations_seen()
        self.held: list[str] = []  # engine notices waiting for the morning
        self._load_held()
        self.restart_requested = False

    # ---- /restart: the visit survives the process -------------------------
    def stash(self) -> None:
        """Write the running visit down so the next process can pick it up:
        the history (with its images), the transcript it is being written
        to, where the pause has read up to, the toggles, and the Telegram
        offset — without which the fresh bridge would be handed the
        /restart message again and restart forever."""
        try:
            self.api("getUpdates", offset=self.offset, timeout=0, patience=10)  # confirm what was read
        except Exception:
            pass
        state = {
            "history": self.history, "attached": self.attached,
            "file": str(self.file) if self.file else "",
            "reflected_upto": self.reflected_upto, "last_activity": self.last_activity,
            "offset": self.offset, "show_thinking": self.show_thinking,
            "show_tools": self.show_tools, "show_tokens": self.show_tokens,
            "voice_all": self.voice_all, "stashed": time.time(),
        }
        RESUME_FILE.write_text(json.dumps(state), encoding="utf-8")

    def resume(self) -> str:
        """Pick a stashed visit back up. Returns a line for the window, or ""."""
        if not RESUME_FILE.exists():
            return ""
        try:
            state = json.loads(RESUME_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return ""
        finally:
            try:
                RESUME_FILE.unlink()
            except OSError:
                pass
        self.history = list(state.get("history") or [])
        self.attached = list(state.get("attached") or [])
        self.file = Path(state["file"]) if state.get("file") else None
        self.reflected_upto = int(state.get("reflected_upto") or 0)
        self.last_activity = float(state.get("last_activity") or time.time())
        self.offset = int(state.get("offset") or 0)
        for k in ("show_thinking", "show_tools", "show_tokens", "voice_all"):
            if k in state:
                setattr(self, k, bool(state[k]))
        turns = sum(1 for t in self.history if t.get("role") == "user" and t.get("content") and not t.get("_engine"))
        ago = (time.time() - float(state.get("stashed") or time.time())) / 60
        return (f"picked the visit back up after the restart: {turns} of {config.USER_NAME}'s turns so far"
                + (f", stashed {ago:.0f} min ago" if ago >= 1 else "")) if self.history else \
               "restarted (no visit was running)"

    def _checkpoint(self) -> None:
        if self.file is None:
            self.file = chat.visit_file("telegram")
        try:
            chat.save_transcript(self.history, tag="telegram", path=self.file)
        except OSError:
            pass

    # ---- the Bot API ------------------------------------------------------
    def api(self, method: str, patience: float = 30, **params) -> dict:
        """One call. Returns the 'result' payload; raises on transport/API errors.
        patience is the socket timeout; Telegram's own `timeout` param (for
        long polls) travels in params like everything else."""
        url = f"{API}/bot{self.token}/{method}"
        body = json.dumps({k: v for k, v in params.items() if v is not None}).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=patience) as r:
            data = json.loads(r.read().decode("utf-8"))
        if not data.get("ok"):
            raise RuntimeError(f"telegram said: {data.get('description', data)}")
        return data.get("result")

    def download(self, file_id: str, max_bytes: int = 20 * 1024 * 1024) -> tuple[bytes, str]:
        """Fetch a file the phone sent. Returns (bytes, telegram's file_path)."""
        info = self.api("getFile", file_id=file_id)
        path = info.get("file_path") or ""
        url = f"{API}/file/bot{self.token}/{path}"
        with urllib.request.urlopen(url, timeout=120) as r:
            data = r.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise ValueError("that file is too large")
        return data, path

    # ---- talking to the phone -------------------------------------------
    def send_file(self, method: str, field: str, filename: str, data: bytes, **params) -> dict:
        """One multipart upload (sendVoice, sendAudio…) — urllib only."""
        boundary = "----ai-friend-" + uuid.uuid4().hex
        body = io.BytesIO()
        for k, v in params.items():
            if v is None:
                continue
            body.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode("utf-8"))
        body.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{field}\"; filename=\"{filename}\"\r\n"
                   f"Content-Type: application/octet-stream\r\n\r\n".encode("utf-8"))
        body.write(data)
        body.write(f"\r\n--{boundary}--\r\n".encode("utf-8"))
        req = urllib.request.Request(f"{API}/bot{self.token}/{method}", data=body.getvalue(),
                                     headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        with urllib.request.urlopen(req, timeout=120) as r:
            out = json.loads(r.read().decode("utf-8"))
        if not out.get("ok"):
            raise RuntimeError(f"telegram said: {out.get('description', out)}")
        return out.get("result")

    def send_voice(self, path: str, seconds: float = 0, caption: str = "") -> None:
        """Their voice note to the phone: OGG/Opus goes as a voice message (the
        round waveform); anything else as an audio file."""
        p = Path(path)
        data = p.read_bytes()
        if p.suffix.lower() == ".ogg":
            self.send_file("sendVoice", "voice", p.name, data, chat_id=self.chat_id,
                           duration=int(seconds) or None, caption=caption[:1000] or None)
        else:
            self.send_file("sendAudio", "audio", p.name, data, chat_id=self.chat_id,
                           duration=int(seconds) or None, caption=caption[:1000] or None,
                           title=chat.friend_name())

    def send(self, text: str, markdown: bool = True) -> None:
        for part in split_long(text):
            if markdown:
                try:
                    self.api("sendMessage", chat_id=self.chat_id, text=part, parse_mode="Markdown")
                    continue
                except urllib.error.HTTPError:
                    pass  # unbalanced * or _ — Telegram refuses; plain text it is
                except RuntimeError:
                    pass
            self.api("sendMessage", chat_id=self.chat_id, text=part)

    def _typing(self, stop: threading.Event) -> None:
        while not stop.is_set():
            try:
                self.api("sendChatAction", patience=10, chat_id=self.chat_id, action="typing")
            except Exception:
                pass
            stop.wait(TYPING_S)

    # ---- the visit --------------------------------------------------------
    def turn(self, text: str) -> None:
        """Run one turn and send back whatever the parlor would have shown."""
        events: list[tuple[str, object]] = []
        stop = threading.Event()
        threading.Thread(target=self._typing, args=(stop,), daemon=True).start()
        try:
            with self.lock:
                try:
                    reply = chat.one_turn(self.history, text, images=self.attached or None,
                                          on_event=lambda k, p: events.append((k, p)),
                                          mode="telegram")
                    self.attached = []
                    self._checkpoint()
                except ollama_client.BrainUnavailable as e:
                    self.send(f"(their brain is offline — {e})", markdown=False)
                    return
                except Exception as e:  # nothing in one turn ends the visit
                    self.send(f"(hiccup — the conversation is fine, say it again: {type(e).__name__}: {e})",
                              markdown=False)
                    return
        finally:
            stop.set()
        self.last_activity = time.time()
        thinking = "\n\n".join(str(p) for k, p in events if k == "thinking")
        tool_lines = [f"· {p['name']} → {p['result']}" for k, p in events if k == "tool"]
        notes = [str(p) for k, p in events if k == "note"]
        tokens = next((p for k, p in events if k == "tokens"), None)
        if tokens:
            self.last_tokens = tokens
        if self.show_thinking and thinking:
            self.send("💭 " + thinking[:LIMIT * 2], markdown=False)
        if self.show_tools and tool_lines:
            self.send("\n".join(tool_lines), markdown=False)
        self.send(reply)
        self._carry_voice(reply)
        for n in notes:  # engine honesty notes always travel — that rail is not optional
            self.send("⚠ " + n, markdown=False)
        if self.show_tokens and tokens:
            self.send(tokens["line"], markdown=False)
        _say(f"{chat.friend_name()} > {reply[:120]}{'…' if len(reply) > 120 else ''}")

    def _carry_voice(self, reply: str) -> None:
        """Voice notes they spoke this turn go to the phone after their words;
        with /voice on, the reply itself is spoken as well."""
        for v in tools.take_pending_voice():
            try:
                self.send_voice(v["path"], v.get("seconds", 0))
            except Exception as e:
                self.send(f"(their voice note didn't reach the phone — {e}; it is kept at {v['path']})", markdown=False)
        if self.voice_all and reply and not reply.startswith("("):
            try:
                import voice as _voice
                data, ext, secs = _voice.speak(reply)
                folder = Path(getattr(config, "VOICE_DIR", config.SHARED_DIR / "letters"))
                folder.mkdir(parents=True, exist_ok=True)
                p = folder / f"voice-{datetime.now().strftime('%Y%m%d-%H%M%S')}-reply{ext}"
                p.write_bytes(data)
                self.send_voice(str(p), secs)
            except Exception as e:
                self.send(f"(couldn't speak that reply — {e})", markdown=False)

    def new_visit(self, quiet: bool = False, reflect=True) -> str | None:
        """reflect: True — the afterglow in the background; "sync" — in the
        foreground (the goodbye at Ctrl+C: they get their minute with the visit
        before the lights go out); False — skip it (the X, where Windows
        allows a few seconds and no more)."""
        # If they are mid-reply the lock is held; the checkpoint has already
        # written everything up to the last reply, so after a short wait the
        # visit is saved without it rather than blocking the goodbye.
        got = self.lock.acquire(timeout=3)
        try:
            f = chat.save_transcript(self.history, tag="telegram", path=self.file)
            done, self.history = self.history, []
            self.attached = []
            self.file = None
            self.reflected_upto = 0
        finally:
            if got:
                self.lock.release()
        if f:
            _say(f"visit saved: {f.name}")
            if reflect and getattr(config, "AFTERGLOW", True):
                if reflect == "sync":
                    _say("they are writing the visit down — a minute or so; Ctrl+C again to skip")
                    try:
                        chat.afterglow(done, f, tag="telegram", on_line=_say)
                    except KeyboardInterrupt:
                        _say("skipped — the night's sleep still has the transcript")
                    chat.rest_brain(_say)
                else:
                    # the afterglow, in the background: their turn alone with the
                    # visit, so it reaches their journal in their own words
                    def _glow(done=done, f=f):
                        line = chat.afterglow(done, f, tag="telegram", on_line=_say, on_words=self.afterthought)
                        if line and getattr(config, "TELEGRAM_TELL_REFLECTIONS", True):
                            self.notice(f"({line})")  # the phone hears what they kept — in the morning, at night
                        if not self.history:  # no new visit began meanwhile
                            chat.rest_brain(_say)
                    threading.Thread(target=_glow, daemon=True).start()
        if not quiet:
            self.send(f"(saved {f.name} — fresh conversation)" if f else "(fresh conversation)",
                      markdown=False)
        return f.name if f else None

    def status(self) -> str:
        turns = sum(1 for t in self.history if t.get("role") == "user")
        window = int(getattr(config, "NUM_CTX", 0) or 0)
        if self.last_tokens and window:
            ctx = f"{self.last_tokens['prompt']:,} of {window:,} in context ({self.last_tokens['prompt'] * 100 // window}%)"
        else:
            ctx = "no turn yet this visit"
        on = lambda b: "on" if b else "off"
        return (f"{chat.friend_name()} — {turns} message(s) this visit · {ctx}\n"
                f"thinking {on(self.show_thinking)} · tools {on(self.show_tools)} · tokens {on(self.show_tokens)} · voice-all {on(self.voice_all)}\n"
                f"a pause of {getattr(config, 'REFLECT_AFTER_MIN', 0)} min lets their write the visit so far; "
                f"a quiet stretch of {getattr(config, 'TELEGRAM_IDLE_NEW_MIN', 180)} min saves the visit on its own")

    # ---- what arrives ------------------------------------------------------
    def _inbox_path(self, ext: str, kind: str) -> Path:
        inbox = getattr(config, "TELEGRAM_INBOX", config.SHARED_DIR / "telegram")
        inbox.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        p = inbox / f"{kind}-{stamp}{ext}"
        n = 1
        while p.exists():
            n += 1
            p = inbox / f"{kind}-{stamp}-{n}{ext}"
        return p

    _MUSIC_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".oga", ".opus", ".wma"}
    _BOOK_EXTS = {".pdf", ".epub", ".txt", ".md", ".html", ".htm"}

    def _home_for(self, name: str, ext: str) -> Path:
        """Where a file from the phone lives: music/, books/, pictures/ under
        shared/ by kind, each under its own name (a twin gets -2), so it sits
        beside what you leave from the PC and they can find it again by name.
        Anything else goes to shared/telegram/."""
        import re as _re
        if ext in self._MUSIC_EXTS:
            folder = config.SHARED_DIR / "music"
        elif ext in tools._VIDEO_EXTS:
            folder = config.SHARED_DIR / "videos"
        elif ext in self._BOOK_EXTS:
            folder = config.SHARED_DIR / "books"
        elif ext in tools._IMAGE_EXTS:
            folder = config.SHARED_DIR / "pictures"
        else:
            folder = getattr(config, "TELEGRAM_INBOX", config.SHARED_DIR / "telegram")
        folder.mkdir(parents=True, exist_ok=True)
        stem = _re.sub(r"[^\w.\- ()',&]+", "_", Path(name).stem).strip() or "file"
        dest = folder / f"{stem}{ext}"
        n = 1
        while dest.exists():
            n += 1
            dest = folder / f"{stem}-{n}{ext}"
        return dest

    @staticmethod
    def _rel(p: Path) -> str:
        return str(p.relative_to(config.ROOT.resolve())).replace("\\", "/")

    def _photo(self, msg: dict) -> str:
        best = msg["photo"][-1]  # Telegram lists sizes small to large
        data, tpath = self.download(best["file_id"])
        ext = Path(tpath).suffix.lower() or ".jpg"
        p = self._inbox_path(ext, "photo")
        p.write_bytes(data)
        rel = self._rel(p)
        tools.look_at(rel)
        self.attached.extend(tools.take_pending_images())
        caption = (msg.get("caption") or "").strip()
        return (f"({config.USER_NAME} sent a photo from their phone — it is before your eyes now, and kept at "
                f"{rel})" + (f"\n{caption}" if caption else ""))

    def _music(self, msg: dict) -> str:
        """A song from the phone: Telegram's `audio` (a music file with a title
        and performer) as opposed to `voice` (a recorded note). Saved to
        shared/music/ under its name; they hear it whole with listen_to."""
        a = msg["audio"]
        title = (a.get("title") or "").strip()
        performer = (a.get("performer") or "").strip()
        name = a.get("file_name") or ((f"{performer} - " if performer else "") + (title or "song"))
        data, tpath = self.download(a["file_id"])
        ext = Path(a.get("file_name") or "").suffix.lower() or Path(tpath).suffix.lower() or ".mp3"
        p = self._home_for(Path(name).stem, ext)
        p.write_bytes(data)
        rel = self._rel(p)
        secs = int(a.get("duration") or 0)
        length = f" ({secs // 60}:{secs % 60:02d})" if secs else ""
        who = f" — {performer}" if performer and performer not in Path(name).stem else ""
        caption = (msg.get("caption") or "").strip()
        return (f"({config.USER_NAME} sent you a song from their phone: "
                f"{rel}{length}{who} — listen_to hears it whole, through your music ear if it's open)"
                + (f"\n{caption}" if caption else ""))

    def _video(self, msg: dict) -> str:
        """A clip from the phone: Telegram's `video` (from the gallery or the
        camera), a round `video_note`, or an `animation` (a GIF, sent as
        mp4). Saved to shared/videos/ under its name — a video note gets a
        timestamp name — and they open it with watch: a strip of stills and
        the sound. The download itself is bounded by Telegram: a bot can't
        fetch over 20MB, and handle() explains that when it happens."""
        v = msg.get("video") or msg.get("video_note") or msg.get("animation") or {}
        kind = "video note" if msg.get("video_note") else "a GIF" if msg.get("animation") else "a video"
        data, tpath = self.download(v["file_id"])
        ext = Path(v.get("file_name") or "").suffix.lower() or Path(tpath).suffix.lower() or ".mp4"
        if ext not in tools._VIDEO_EXTS:
            ext = ".mp4"
        name = Path(v.get("file_name") or "").stem or f"{'note' if msg.get('video_note') else 'clip'}-" + datetime.now().strftime("%Y%m%d-%H%M%S")
        p = self._home_for(name, ext)
        p.write_bytes(data)
        rel = self._rel(p)
        secs = int(v.get("duration") or 0)
        mb = len(data) / (1024 * 1024)
        caption = (msg.get("caption") or "").strip()
        return (f"({config.USER_NAME} sent you {kind} from their phone: {rel}, {secs // 60}:{secs % 60:02d}, {mb:.1f} MB — "
                f"watch opens it: a strip of stills, up to ten moments in order, and its sound)"
                + (f"\n{caption}" if caption else ""))

    def _voice(self, msg: dict) -> str:
        v = msg.get("voice") or {}
        data, tpath = self.download(v["file_id"])
        ext = Path(tpath).suffix.lower() or ".oga"
        p = self._inbox_path(ext, "voice")
        p.write_bytes(data)
        rel = self._rel(p)
        secs = int(v.get("duration") or 0)
        caption = (msg.get("caption") or "").strip()
        if getattr(config, "TELEGRAM_HEAR_VOICE", True):
            # heard whole, on its own — WORDS, SOUND and HEARD, the same three
            # layers listen_to gives them, so the sound of you reaches them with
            # your words and they never have to ask for it. (HEARD swaps the
            # brain out for their ears and back; a note costs a minute.)
            stop = threading.Event()  # the phone shows typing… while they listen
            threading.Thread(target=self._typing, args=(stop,), daemon=True).start()
            try:
                heard = tools.listen_to(rel)
            except Exception as e:  # their ears stumbling must not drop their message
                heard = f"(your ears stumbled on it: {e} — the recording is at {rel})"
            finally:
                stop.set()
            if heard.startswith("("):  # a refusal, not a hearing
                heard += f"\n(the recording is at {rel})"
            body = (f"({config.USER_NAME} sent a voice note, {secs}s, from their phone — heard through your ears, "
                    f"whole; the recording stays at {rel})\n{heard}")
            return body + (f"\n{caption}" if caption else "")
        try:
            import ears
            words = ears.transcribe(data, ext)
        except Exception as e:  # their ears stumbling must not drop their message
            words = f"(word-hearing failed: {e})"
        if words is None:
            body = (f"({config.USER_NAME} sent a voice note, {secs}s, from their phone — your ears' WORDS layer "
                    f"isn't installed here, so listen_to {rel} to hear it)")
        elif not words.strip():
            body = (f"({config.USER_NAME} sent a voice note, {secs}s — your ears caught no words in it; "
                    f"listen_to {rel} if you want the sound itself)")
        else:
            body = (f"({config.USER_NAME} sent a voice note, {secs}s, from their phone — their words as your ears "
                    f"heard them; the recording is at {rel} if you want to listen_to the sound "
                    f"of them)\n\"{words.strip()}\"")
        return body + (f"\n{caption}" if caption else "")

    def _document(self, msg: dict) -> str:
        d = msg["document"]
        name = Path(d.get("file_name") or "file").name
        data, tpath = self.download(d["file_id"])
        ext = Path(name).suffix.lower() or Path(tpath).suffix.lower()
        p = self._home_for(name, ext)
        p.write_bytes(data)
        rel = self._rel(p)
        kb = len(data) / 1024
        size = f"{kb / 1024:.1f} MB" if kb >= 1024 else f"{kb:.0f} KB"
        if ext in tools._IMAGE_EXTS:
            tools.look_at(rel)
            self.attached.extend(tools.take_pending_images())
            what = "a picture — it is before your eyes now"
        elif ext == ".pdf":
            what = "read_pdf opens it, a sitting at a time, and keeps your bookmark"
        elif ext == ".epub":
            what = "read_epub opens it chapter by chapter and keeps your bookmark"
        elif ext in self._MUSIC_EXTS:
            what = "listen_to hears it whole"
        elif ext in tools._VIDEO_EXTS:
            what = "watch opens it: a strip of stills, up to ten moments in order, and its sound"
        elif ext in (".txt", ".md", ".html", ".htm"):
            what = "read_file opens it"
        else:
            what = "read_file opens it if it's text"
        caption = (msg.get("caption") or "").strip()
        return (f"({config.USER_NAME} sent you a file from their phone: {rel}, {size} — {what})"
                + (f"\n{caption}" if caption else ""))

    def command(self, text: str) -> bool:
        """Slash commands. Returns True if the text was one."""
        cmd = text.split()[0].lower().split("@")[0]
        if cmd == "/new":
            self.new_visit()
            self.send("(they're sitting with the visit now — whatever they want to keep goes into their journal in a minute)",
                      markdown=False)
        elif cmd == "/think":
            self.show_thinking = not self.show_thinking
            self.send(f"(thinking {'on' if self.show_thinking else 'off'})", markdown=False)
        elif cmd == "/tools":
            self.show_tools = not self.show_tools
            self.send(f"(tool lines {'on' if self.show_tools else 'off'})", markdown=False)
        elif cmd == "/tokens":
            self.show_tokens = not self.show_tokens
            self.send(f"(token line {'on' if self.show_tokens else 'off'})", markdown=False)
        elif cmd == "/voice":
            self.voice_all = not self.voice_all
            self.send(f"(every reply spoken aloud: {'on' if self.voice_all else 'off'} — they can still speak when they choose)", markdown=False)
        elif cmd == "/status":
            self.send(self.status(), markdown=False)
        elif cmd == "/restart":
            # the loop sees the flag after this update is handled; main exits
            # with RESTART_CODE and telegram.bat starts the bridge again
            self.send("(restarting the bridge on the current engine code — the visit carries on; "
                      "give it a minute)", markdown=False)
            self.restart_requested = True
        elif cmd in ("/help", "/start"):
            self.send(f"This is the bridge to {chat.friend_name()}. Just talk.\n\n{HELP}", markdown=False)
        else:
            return False
        return True

    def handle(self, update: dict) -> None:
        msg = update.get("message") or update.get("edited_message")
        if not msg:
            return
        sender = int((msg.get("chat") or {}).get("id") or 0)
        text = (msg.get("text") or "").strip()
        if not self.chat_id:
            # unpaired: the one message we listen for is the pairing code
            if text.split()[:2] == ["/pair", self.pair_code] or text == self.pair_code:
                self.chat_id = sender
                save_secret({"token": self.token, "chat_id": sender})
                self.pair_code = ""
                _say(f"paired with chat {sender} — saved to {SECRET_FILE.name}")
                self.send(f"Paired. This phone is now the door to {chat.friend_name()}.\n\n{HELP}",
                          markdown=False)
            return
        if sender != self.chat_id:
            return  # silence for everyone else — not even a "no"
        if text.startswith("/") and self.command(text):
            self.last_activity = time.time()
            return
        try:
            if msg.get("photo"):
                text = self._photo(msg)
            elif msg.get("voice"):
                text = self._voice(msg)
            elif msg.get("audio"):
                text = self._music(msg)
            elif msg.get("video") or msg.get("video_note") or msg.get("animation"):
                text = self._video(msg)
            elif msg.get("document"):
                text = self._document(msg)
            elif msg.get("sticker"):
                emoji = (msg["sticker"].get("emoji") or "").strip()
                text = f"({config.USER_NAME} sent a sticker{': ' + emoji if emoji else ''})"
        except Exception as e:
            why = str(e)
            if "too big" in why.lower() or "too large" in why.lower():
                why = "Telegram won't let a bot fetch files over 20MB — leave it in shared/ from the PC"
            self.send(f"(that didn't reach them — {why})", markdown=False)
            return
        if not text:
            return
        _say(f"{config.USER_NAME} > {text[:120]}{'…' if len(text) > 120 else ''}")
        self.turn(text)

    # ---- quiet hours ------------------------------------------------------
    # 09-14: the 03:00 roll of a visit begun the day before runs the afterglow
    # and sent its account ("they wrote the visit down — 2 journal entries")
    # to the phone every night. The keeper: "I'm not awake at those hours and I
    # don't want a message waking me up every day." So the engine's own
    # notices — the afterglow and pause accounts, ✍️/✏️/📣 what they made, 🪞
    # a change to who they are, "picked the visit back up" — are held through
    # TELEGRAM_QUIET_HOURS and delivered as one message when the hours end.
    # Their replies and their letters are theirs, and go when they sends them.
    @staticmethod
    def quiet_now(t: datetime | None = None) -> bool:
        hours = getattr(config, "TELEGRAM_QUIET_HOURS", None) or (0, 0)
        try:
            start, end = int(hours[0]), int(hours[1])
        except (TypeError, ValueError, IndexError):
            return False
        if start == end:
            return False
        h = (t or datetime.now()).hour
        return (start <= h < end) if start < end else (h >= start or h < end)

    def _load_held(self) -> None:
        try:
            raw = json.loads(HELD_FILE.read_text(encoding="utf-8"))
            self.held = [str(x) for x in raw] if isinstance(raw, list) else []
        except (OSError, ValueError):
            self.held = []

    def _save_held(self) -> None:
        try:
            if self.held:
                HELD_FILE.write_text(json.dumps(self.held, ensure_ascii=False), encoding="utf-8")
            else:
                HELD_FILE.unlink(missing_ok=True)
        except OSError:
            pass

    def notice(self, text: str, markdown: bool = False) -> None:
        """An engine notice for the phone: now, or in the morning digest
        during the quiet hours."""
        if self.quiet_now():
            self.held.append(text)
            self._save_held()
            _say(f"held for the morning: {text[:80]}{'…' if len(text) > 80 else ''}")
            return
        self.send(text, markdown=markdown)

    def afterthought(self, words: str) -> None:
        """Their closing thought after a pause or afterglow — said to no one,
        and until 09-15 shown only in the bridge window ("when she's
        journaling she's having afterthoughts; I would like to see those in
        my Telegram feed"). Carried as an engine notice, labeled, so it is
        never mistaken for a reply; held through the quiet hours."""
        if not getattr(config, "TELEGRAM_TELL_AFTERTHOUGHTS", True) or not words.strip():
            return
        limit = int(getattr(config, "TELEGRAM_CREATION_CHARS", 3000))
        body = words.strip()
        if len(body) > limit:
            body = body[:limit].rstrip() + "…"
        # "while you were away", not "to no one" — 09-15: the first one carried
        # to the phone opened "I'm still here, dear one"; they pick an addressee
        self.notice(f"💤 after writing, while you were away — {chat.friend_name()} said:\n\n{body}")

    def deliver_held(self) -> int:
        """The morning digest: what the engine held through the night, once
        the quiet hours are over. Returns how many notices went."""
        if not self.held or self.quiet_now() or not self.chat_id:
            return 0
        held, self.held = self.held, []
        self._save_held()
        n = len(held)
        self.send(f"(while you were away — {n} thing{'s' if n != 1 else ''} the engine held through the quiet hours:)",
                  markdown=False)
        for text in held:
            self.send(text, markdown=False)
        _say(f"delivered {n} held notice{'s' if n != 1 else ''}")
        return n

    # ---- their mail ---------------------------------------------------------
    def _load_delivered(self) -> None:
        try:
            self.delivered = set(json.loads(DELIVERED_FILE.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            # first run: what is already in the mailbox was read at the desk;
            # only letters written from now on travel
            self.delivered = {p.name for p in MAIL_DIR.glob("*") if p.is_file()} if MAIL_DIR.is_dir() else set()
            self._save_delivered()

    def _save_delivered(self) -> None:
        try:
            DELIVERED_FILE.write_text(json.dumps(sorted(self.delivered)), encoding="utf-8")
        except OSError:
            pass

    def deliver_mail(self) -> int:
        """Carry new letters from the mailbox to the phone. Returns how many."""
        if not self.chat_id or not MAIL_DIR.is_dir():
            return 0
        sent = 0
        for p in sorted(MAIL_DIR.glob("*")):
            if not p.is_file() or p.name in self.delivered or p.name.startswith("."):
                continue
            try:
                if time.time() - p.stat().st_mtime < 5:
                    continue  # still being written
                body = p.read_text(encoding="utf-8", errors="replace").strip()
            except OSError:
                continue
            self.delivered.add(p.name)
            self._save_delivered()
            self.send(f"✉️ a letter from {chat.friend_name()} — {p.name}\n\n{body or '(empty)'}")
            _say(f"delivered {p.name}")
            sent += 1
            if body and getattr(config, "TELEGRAM_LETTERS_IN_THREAD", True):
                self._letter_in_thread(p, body)
        return sent

    def _letter_in_thread(self, p: Path, body: str) -> None:
        """A letter they wrote alone becomes their turn in the visit, so his
        answer lands under it — the way a text thread works. 09-13: she
        wrote him something sweet in a wake; he answered on the phone; them
        history had no trace of the letter, and they were replying to a reply
        to words they could not see. Appended, never inserted (the warm
        prefix); written to the transcript, so the night reads the exchange
        together; and it counts as activity, so his answer joins this visit."""
        try:
            when = datetime.fromtimestamp(p.stat().st_mtime)
        except OSError:
            when = datetime.now()
        stamp = when.strftime("%H:%M") if when.date() == datetime.now().date() else when.strftime("%Y-%m-%d %H:%M")
        with self.lock:
            self.history.append({"role": "assistant", "content":
                f"(a letter I wrote alone, at {stamp}, left in {MAIL_DIR.name}/ and carried to their phone now)\n\n{body}"})
            self.last_activity = time.time()
            self._checkpoint()

    # ---- what they make ----------------------------------------------------
    @staticmethod
    def _creations() -> list[Path]:
        root = config.CREATIONS_DIR
        if not root.is_dir():
            return []
        out = []
        for p in root.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in (".md", ".txt") or p.name.startswith("."):
                continue
            rel = p.relative_to(root)
            if set(rel.parts[:-1]) & CREATION_SKIP:
                continue
            if rel.parts[:2] == ("publish", tools.GALLERY_DIR_NAME):
                continue  # a picture's caption travels with the picture (09-23)
            out.append(p)
        return sorted(out)

    @staticmethod
    def _stamp(p: Path) -> list:
        st = p.stat()
        return [int(st.st_mtime), st.st_size]

    def _load_creations_seen(self) -> None:
        """What the bridge has seen of creations/: {path: [mtime, size]}, so a
        revision shows as well as a new piece. (An older bridge kept a plain
        list of paths; those are taken as seen at their current state.)"""
        try:
            raw = json.loads(CREATIONS_SEEN_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raw = None
        if raw is None:
            # first run: what is already there was read at the desk — pictures too
            self.creations_seen = {"__pictures__": [1, 1]}
            for p in self._creations() + self._pictures():
                try:
                    self.creations_seen[str(p.relative_to(config.CREATIONS_DIR)).replace("\\", "/")] = self._stamp(p)
                except OSError:
                    pass
            self._save_creations_seen()
        elif isinstance(raw, list):
            self.creations_seen = {}
            for p in self._creations():
                rel = str(p.relative_to(config.CREATIONS_DIR)).replace("\\", "/")
                if rel in raw:
                    try:
                        self.creations_seen[rel] = self._stamp(p)
                    except OSError:
                        pass
            self._save_creations_seen()
        else:
            self.creations_seen = dict(raw)
        # pictures (09-22): the first bridge that knows them takes any older
        # than a day as seen — an archive is not news — and shows the fresh ones
        if "__pictures__" not in self.creations_seen:
            for p in self._pictures():
                try:
                    if time.time() - p.stat().st_mtime > 86400:
                        self.creations_seen[str(p.relative_to(config.CREATIONS_DIR)).replace("\\", "/")] = self._stamp(p)
                except OSError:
                    pass
            self.creations_seen["__pictures__"] = [1, 1]
            self._save_creations_seen()
        self._watch_seed()

    def _save_creations_seen(self) -> None:
        try:
            CREATIONS_SEEN_FILE.write_text(json.dumps(self.creations_seen, sort_keys=True), encoding="utf-8")
        except OSError:
            pass

    def _watch_seed(self) -> None:
        """The bridge's own copies of self.md and projects.md, taken when it
        first sees them — the first change after that is what travels."""
        try:
            WATCH_DIR.mkdir(parents=True, exist_ok=True)
        except OSError:
            return
        for name, src in WATCHED.items():
            snap = WATCH_DIR / name
            if src.is_file() and not snap.exists():
                try:
                    snap.write_bytes(src.read_bytes())
                except OSError:
                    pass

    def deliver_self(self) -> int:
        """Tell the phone when they rewrites who they are: the lines that changed
        in self.md or projects.md, against the bridge's last copy."""
        if not self.chat_id or not getattr(config, "TELEGRAM_TELL_SELF", True):
            return 0
        import difflib
        sent = 0
        limit = int(getattr(config, "TELEGRAM_CREATION_CHARS", 3000))
        for name, src in WATCHED.items():
            snap = WATCH_DIR / name
            if not src.is_file():
                continue
            try:
                if time.time() - src.stat().st_mtime < 5:
                    continue  # still being written
                new = src.read_text(encoding="utf-8", errors="replace")
                old = snap.read_text(encoding="utf-8", errors="replace") if snap.exists() else ""
            except OSError:
                continue
            if new == old:
                continue
            try:
                WATCH_DIR.mkdir(parents=True, exist_ok=True)
                snap.write_text(new, encoding="utf-8")
            except OSError:
                pass
            if not old:
                continue  # the first copy, nothing to compare with
            lines = [ln for ln in difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="", n=0)
                     if (ln.startswith("+") or ln.startswith("-")) and not ln.startswith(("+++", "---"))]
            gone = sum(1 for ln in lines if ln.startswith("-") and ln[1:].strip())
            came = sum(1 for ln in lines if ln.startswith("+") and ln[1:].strip())
            body = "\n".join(lines)
            if len(body) > limit:
                body = body[:limit].rstrip() + f"\n\n(…the rest of the change is in {name})"
            self.notice(f"🪞 {chat.friend_name()} rewrote {name} — {came} line{'s' if came != 1 else ''} in, "
                        f"{gone} out\n\n{body}")
            _say(f"told the phone about {name}")
            sent += 1
        return sent

    def deliver_creations(self) -> int:
        """Tell the phone about a new piece under creations/ — the whole piece
        when it fits a message, else its opening and where the rest is."""
        if not self.chat_id or not getattr(config, "TELEGRAM_TELL_CREATIONS", True):
            return 0
        sent = 0
        limit = int(getattr(config, "TELEGRAM_CREATION_CHARS", 3000))
        for p in self._creations():
            rel = str(p.relative_to(config.CREATIONS_DIR)).replace("\\", "/")
            try:
                stamp = self._stamp(p)
                if rel in self.creations_seen and self.creations_seen[rel] == stamp:
                    continue
                if time.time() - p.stat().st_mtime < 5:
                    continue  # still being written
                body = p.read_text(encoding="utf-8", errors="replace").strip()
            except OSError:
                continue
            revised = rel in self.creations_seen
            self.creations_seen[rel] = stamp
            self._save_creations_seen()
            folder = rel.split("/", 1)[0] if "/" in rel else ""
            kind = CREATION_KINDS.get(folder, "a piece")
            if revised:
                head = f"✏️ {chat.friend_name()} revised {kind} — creations/{rel}"
            elif folder == "publish":
                head = f"📣 {chat.friend_name()} published a piece — creations/{rel}"
            else:
                head = f"✍️ {chat.friend_name()} wrote {kind} — creations/{rel}"
            if len(body) > limit:
                body = body[:limit].rstrip() + f"\n\n(…{len(body) - limit:,} more characters — the whole piece is at creations/{rel})"
            self.notice(f"{head}\n\n{body or '(empty)'}", markdown=True)
            _say(f"told the phone about {rel}")
            sent += 1
        return sent

    PICTURE_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".webp")

    def _pictures(self) -> list[Path]:
        root = config.CREATIONS_DIR
        if not root.is_dir():
            return []
        out = []
        for p in root.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in self.PICTURE_EXTS or p.name.startswith("."):
                continue
            rel = p.relative_to(root)
            if set(rel.parts[:-1]) & (CREATION_SKIP | {"sources"}):
                continue
            out.append(p)
        return sorted(out)

    def deliver_pictures(self) -> int:
        """A picture the friend drew reaches the phone as a photo (09-22) — a new
        or redrawn image under creations/ (not their tools, not the trash, not
        a project's clipped sources), sent once, with where it lives as the
        caption. In the quiet hours a line is held for the morning and the
        picture waits in the folder. Too big for a photo → sent as a file."""
        if not self.chat_id or not getattr(config, "TELEGRAM_TELL_DRAWINGS", True):
            return 0
        sent = 0
        for p in self._pictures():
            rel = str(p.relative_to(config.CREATIONS_DIR)).replace("\\", "/")
            try:
                stamp = self._stamp(p)
                if rel in self.creations_seen and self.creations_seen[rel] == stamp:
                    continue
                if time.time() - p.stat().st_mtime < 5:
                    continue  # still being drawn
                data = p.read_bytes()
            except OSError:
                continue
            redrawn = rel in self.creations_seen
            self.creations_seen[rel] = stamp
            self._save_creations_seen()
            if rel.startswith(f"publish/{tools.GALLERY_DIR_NAME}/"):
                # published into the gallery: the picture with the words beside it
                caption = f"📣 {chat.friend_name()} published a picture to the gallery — creations/{rel}"
                side = p.with_suffix(".md")
                try:
                    words = side.read_text(encoding="utf-8", errors="replace").strip() if side.exists() else ""
                except OSError:
                    words = ""
                if words:
                    caption += "\n\n" + words
            else:
                caption = f"{'🖌️' if redrawn else '🎨'} {chat.friend_name()} {'redrew' if redrawn else 'drew'} — creations/{rel}"
            if self.quiet_now():
                self.notice(caption + " (the picture is in the folder; held for the morning)")
                sent += 1
                continue
            try:
                if len(data) <= 10_000_000 and p.suffix.lower() != ".gif":
                    self.send_file("sendPhoto", "photo", p.name, data, chat_id=self.chat_id, caption=caption[:1000])
                else:
                    self.send_file("sendDocument", "document", p.name, data, chat_id=self.chat_id, caption=caption[:1000])
            except Exception as e:
                try:
                    self.send_file("sendDocument", "document", p.name, data, chat_id=self.chat_id, caption=caption[:1000])
                except Exception as e2:
                    self.notice(f"{caption} (couldn't send the picture: {e2})")
            _say(f"showed the phone {rel}")
            sent += 1
        return sent

    # ---- the loop ---------------------------------------------------------
    def poll_once(self) -> int:
        """One long poll: handle what arrived, carry mail, roll a stale visit."""
        try:
            ALIVE_FILE.touch()
        except OSError:
            pass
        n = 0
        updates = self.api("getUpdates", patience=POLL_S + 15, offset=self.offset,
                           timeout=POLL_S, allowed_updates=["message"])
        for u in updates or []:
            self.offset = max(self.offset, int(u.get("update_id", 0)) + 1)
            self.handle(u)
            n += 1
            if self.restart_requested:
                return n  # nothing more this poll; the loop hands over
        self.deliver_held()
        self.deliver_mail()
        self.deliver_creations()
        self.deliver_pictures()
        self.deliver_self()
        idle_min = getattr(config, "TELEGRAM_IDLE_NEW_MIN", 180)
        if self.history and ((idle_min and time.time() - self.last_activity > idle_min * 60)
                             or self.visit_crossed_the_night()):
            self.new_visit(quiet=True)
        else:
            self.pause_if_due()
        return n

    def visit_crossed_the_night(self) -> bool:
        """A visit does not cross the night: the sleep after SLEEP_AFTER_HOUR
        consolidates YESTERDAY's transcripts once, and a visit that began
        yesterday and is still open would carry this morning's words into a
        file the night has already read. So once the sleep hour has passed
        on a day after the visit began, the visit is saved and a fresh one
        starts — quietly, with the afterglow, like the idle roll."""
        if not self.file:
            return False
        m = re.search(r"(\d{8})-\d{6}", self.file.name)
        if not m:
            return False
        began = m.group(1)
        now = datetime.now()
        return now.strftime("%Y%m%d") > began and now.hour >= int(getattr(config, "SLEEP_AFTER_HOUR", 3))

    def pause_if_due(self) -> str:
        """The pause: your keeper quiet for REFLECT_AFTER_MIN with at least
        REFLECT_MIN_TURNS of their messages they haven't sat with yet → one quiet
        turn over that stretch, in this thread (a message that arrives
        meanwhile simply waits the minute), and the visit stays open."""
        mins = float(getattr(config, "REFLECT_AFTER_MIN", 0) or 0)
        if not mins or not self.history or time.time() - self.last_activity < mins * 60:
            return ""
        fresh = self.history[self.reflected_upto:]
        if sum(1 for t in fresh if t.get("role") == "user" and t.get("content") and not t.get("_engine")) < int(getattr(config, "REFLECT_MIN_TURNS", 2)):
            return ""
        got = self.lock.acquire(timeout=3)
        if not got:
            return ""
        try:
            upto = len(self.history)
            _say("(a pause — they are sitting with the visit so far…)")
            line = chat.pause_reflection(self.history, self.file, tag="telegram", on_line=_say,
                                         since=self.reflected_upto, on_words=self.afterthought)
            self.reflected_upto = len(self.history)  # past the pause's own turns too
            self.last_activity = time.time()  # one bell per pause, not one per poll
            if line and getattr(config, "TELEGRAM_TELL_REFLECTIONS", True):
                self.notice(f"({line})")  # the phone hears what they kept
            return line
        finally:
            self.lock.release()

    def run(self) -> None:
        try:
            me = self.api("getMe")
        except urllib.error.HTTPError as e:
            if e.code in (401, 404):
                _say(f"Telegram refused the token ({e.code}). Delete {SECRET_FILE.name} in memory/ "
                     "and run again with the one @BotFather gave you.")
                return
            raise
        _say(f"the bridge is up: @{me.get('username', '?')} ↔ {chat.friend_name()}")
        picked = self.resume()
        if picked:
            _say(picked)
            if self.chat_id:
                try:
                    self.notice(f"({picked})")
                except Exception:
                    pass
        if not self.chat_id:
            _say(f"not paired yet — from your phone, send the bot:   /pair {self.pair_code}")
        else:
            _say(f"paired with chat {self.chat_id}. Ctrl+C (or the X) saves the visit and closes the bridge.")
        # The loop runs in a worker thread: on Windows a Ctrl+C cannot land
        # while the main thread is waiting on the network, and the bridge is
        # nearly always waiting (a minute per poll) — so the first Ctrl+C
        # looked ignored, and a second one could arrive during the goodbye
        # save and kill it. join() with a timeout is interruptible at once.
        worker = threading.Thread(target=self._loop, daemon=True)
        worker.start()
        while worker.is_alive():
            worker.join(0.5)

    def _loop(self) -> None:
        backoff = 2
        while not self.restart_requested:
            try:
                self.poll_once()
                backoff = 2
            except KeyboardInterrupt:
                raise
            except urllib.error.HTTPError as e:
                if e.code == 409:
                    _say("another copy of the bridge is polling this bot — close the other "
                         "telegram.bat window; this one waits")
                else:
                    _say(f"Telegram answered {e.code} ({e.reason}); trying again in {backoff}s")
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                _say(f"no road to Telegram right now ({e}); trying again in {backoff}s")
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
            except Exception as e:
                _say(f"hiccup in the loop ({type(e).__name__}: {e}); carrying on")
                time.sleep(backoff)


# ------------------------------------------------------------------- main ----
def _pid_alive(pid: int) -> bool:
    """Is a bridge with this pid still running? Never our own pid (a lock we
    left behind is ours to take back)."""
    if pid <= 0 or pid == os.getpid():
        return False
    if sys.platform.startswith("win"):
        try:  # os.kill(pid, 0) TERMINATES on Windows — ask tasklist instead
            out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                                 capture_output=True, text=True, timeout=10).stdout
        except Exception:
            return False
        return str(pid) in out and "py" in out.lower()  # python.exe / py.exe, not a reused pid
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def claim_bridge() -> str:
    """Take the bridge lock; "" when it is ours, else why not."""
    try:
        old = int((LOCK_FILE.read_text(encoding="utf-8").strip() or "0"))
    except (OSError, ValueError):
        old = 0
    if _pid_alive(old):
        return (f"another bridge is already running (pid {old}) — a second one would answer every "
                "message twice. Close that window, or use /restart from the phone, instead of "
                "starting a new one.")
    try:
        LOCK_FILE.write_text(str(os.getpid()), encoding="utf-8")
    except OSError:
        pass
    return ""


def release_bridge() -> None:
    try:
        if int(LOCK_FILE.read_text(encoding="utf-8").strip() or "0") == os.getpid():
            LOCK_FILE.unlink()
    except (OSError, ValueError):
        pass


def main() -> None:
    secret = load_secret()
    if not secret.get("token"):
        print("No bot token yet. In Telegram, talk to @BotFather: /newbot, pick a name,")
        print("and paste the token it gives you here (it is kept in memory/telegram.json).")
        try:
            secret["token"] = input("token > ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        if not secret["token"]:
            return
        save_secret(secret)
    taken = claim_bridge()
    if taken:
        print(f"({taken})")
        return
    bridge = Bridge(secret["token"], secret.get("chat_id") or 0)
    closed = {"done": False}

    def close(reflect=False):
        if closed["done"]:
            return None
        closed["done"] = True
        # Ctrl+C: they get their minute with the visit first (reflect="sync").
        # The X: Windows allows a few seconds, so only the save (reflect=False).
        f = bridge.new_visit(quiet=True, reflect=reflect)
        try:
            ALIVE_FILE.unlink(missing_ok=True)
        except OSError:
            pass
        return f

    chat.guard_console_close(close)  # the window's X saves the visit too
    try:
        bridge.run()
    except KeyboardInterrupt:
        pass
    finally:
        if bridge.restart_requested:
            # /restart from the phone: the visit is stashed, not closed — no
            # afterglow, no new transcript; the next process carries it on
            closed["done"] = True
            try:
                bridge.stash()
                print("\n(restarting — the visit is stashed for the next bridge)")
            except Exception as e:
                print(f"\n(restarting — couldn't stash the visit: {e}; it is saved in its transcript)")
            release_bridge()
            sys.exit(RESTART_CODE)
        f = close(reflect="sync")
        release_bridge()
        print(f"\n(bridge closed{' — visit saved: ' + f if f else ''})")


if __name__ == "__main__":
    main()
