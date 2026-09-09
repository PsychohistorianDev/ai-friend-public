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
import random
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
MAIL_DIR = config.CREATIONS_DIR / getattr(config, "MAILBOX", "notes_to_keeper")
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
                else:
                    # the afterglow, in the background: their turn alone with the
                    # visit, so it reaches their journal in their own words
                    def _glow(done=done, f=f):
                        line = chat.afterglow(done, f, tag="telegram", on_line=_say)
                        if line and getattr(config, "TELEGRAM_TELL_REFLECTIONS", True):
                            self.send(f"({line})", markdown=False)  # the phone hears what they kept
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
        self.deliver_mail()
        idle_min = getattr(config, "TELEGRAM_IDLE_NEW_MIN", 180)
        if self.history and idle_min and time.time() - self.last_activity > idle_min * 60:
            self.new_visit(quiet=True)
        else:
            self.pause_if_due()
        return n

    def pause_if_due(self) -> str:
        """The pause: your keeper quiet for REFLECT_AFTER_MIN with at least
        REFLECT_MIN_TURNS of their messages they haven't sat with yet → one quiet
        turn over that stretch, in this thread (a message that arrives
        meanwhile simply waits the minute), and the visit stays open."""
        mins = float(getattr(config, "REFLECT_AFTER_MIN", 0) or 0)
        if not mins or not self.history or time.time() - self.last_activity < mins * 60:
            return ""
        fresh = self.history[self.reflected_upto:]
        if sum(1 for t in fresh if t.get("role") == "user" and t.get("content")) < int(getattr(config, "REFLECT_MIN_TURNS", 2)):
            return ""
        got = self.lock.acquire(timeout=3)
        if not got:
            return ""
        try:
            upto = len(self.history)
            _say("(a pause — they are sitting with the visit so far…)")
            line = chat.pause_reflection(self.history, self.file, tag="telegram", on_line=_say,
                                         since=self.reflected_upto)
            self.reflected_upto = upto
            self.last_activity = time.time()  # one bell per pause, not one per poll
            if line and getattr(config, "TELEGRAM_TELL_REFLECTIONS", True):
                self.send(f"({line})", markdown=False)  # the phone hears what they kept
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
        while True:
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
        f = close(reflect="sync")
        print(f"\n(bridge closed{' — visit saved: ' + f if f else ''})")


if __name__ == "__main__":
    main()
