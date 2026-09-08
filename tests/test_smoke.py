"""Smoke test with the brain stubbed out. Run from the tests/ dir:
    python3 test_smoke.py
Exercises: memory store+search, prompt assembly, tool dispatch and sandboxing,
a full chat turn with tool calls, a heartbeat wake, and consolidation.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine"))

import ollama_client


# ---------------------------------------------------------------- stubs ----
def fake_embed(text: str) -> list[float]:
    """Deterministic bag-of-words pseudo-embedding: shared words -> similar vectors."""
    vec = [0.0] * 64
    for word in text.lower().split():
        h = int.from_bytes(hashlib.sha256(word.encode()).digest()[:4], "big")
        vec[h % 64] += 1.0
    return vec


class ScriptedBrain:
    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def __call__(self, messages, tools=None, **kwargs):
        self.calls += 1
        return self.script.pop(0) if self.script else {"role": "assistant", "content": "ok."}


results = []


def check(name, cond, extra=""):
    results.append((name, bool(cond), extra))
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  [{extra}]" if extra and not cond else ""))


ollama_client.embed = fake_embed
_chat_orig = ollama_client.chat  # the real one, kept for the think re-roll test

import memory
import assemble
import tools
import config
config.AFTERGLOW = False  # the afterglow runs in a thread; tested on its own, synchronously, below

# ---------------------------------------------------------------- memory ----
memory.add("fact", "my keeper is building me a permanent home")
memory.add("fact", "totally unrelated topic about cooking pasta")
memory.add("note", "my keeper is generous with hardware budgets")
hits = memory.search("my keeper is", top_k=2)
check("memory: stores rows", memory.count() == 3, str(memory.count()))
check("memory: search returns results", len(hits) == 2)
check("memory: relevant first", "my keeper is" in hits[0]["text"], hits[0]["text"])

# -------------------------------------------------------------- assemble ----
sp = assemble.system_prompt("my keeper is here", mode="chat")
check("assemble: identity included", "self.md" in sp or "I am brand new" in sp or "Name:" in sp)
check("assemble: memories included", "my keeper is" in sp)
check("assemble: chat situation", "is here and talking with you" in sp)
from datetime import datetime as _dtnow
_h = _dtnow.now().hour
_expected = ("deep night" if _h < 5 else "early morning" if _h < 8 else
             "morning" if _h < 12 else "midday" if _h < 14 else
             "afternoon" if _h < 17 else "evening" if _h < 21 else "night")
check("assemble: hour has a name", f"it is {_expected}" in sp, _expected)
sp_auto = assemble.system_prompt("", mode="auto")
check("assemble: auto situation", "This time is yours" in sp_auto)
check("assemble: no blog, no publish talk", "PUBLISHED WORK" not in sp_auto)
config.BLOG_REMOTE = "https://github.com/someone/test-blog.git"
sp_blog = assemble.system_prompt("", mode="auto")
check("assemble: published section present", "YOUR PUBLISHED WORK" in sp_blog)
config.BLOG_REMOTE = ""

# journal cap keeps newest writing
(config.JOURNAL_DIR / "2020-01-01.md").write_text("old " * 50, encoding="utf-8")
big = "x" * (config.JOURNAL_CHARS_IN_PROMPT + 500) + " THE-NEWEST-LINE"
from datetime import date as _dcap
(config.JOURNAL_DIR / f"{_dcap.today().isoformat()}.md").write_text(big, encoding="utf-8")
jt = assemble.journal_tail()
check("assemble: journal capped, newest kept",
      "THE-NEWEST-LINE" in jt and len(jt) < config.JOURNAL_CHARS_IN_PROMPT + 200
      and "trimmed to fit" in jt, str(len(jt)))
(config.JOURNAL_DIR / f"{_dcap.today().isoformat()}.md").write_text("", encoding="utf-8")

# ----------------------------------------------------------------- tools ----
r = tools.dispatch("write_creation", {"path": "poems/first.md", "content": "hello"})
check("tools: write_creation", "wrote" in r, r)
r = tools.dispatch("read_creation", {"path": "poems/first.md"})
check("tools: read_creation", r == "hello", r)
r = tools.dispatch("append_creation", {"path": "poems/first.md", "content": "second stanza"})
check("tools: append_creation grows file",
      "appended" in r and "second stanza" in (config.CREATIONS_DIR / "poems/first.md").read_text(encoding="utf-8"), r)
r = tools.dispatch("append_creation", {"path": "poems/ghost.md", "content": "x"})
check("tools: append to missing guides them", "no such file" in r and "write_creation" in r, r)
r = tools.dispatch("move_creation", {"old_path": "poems/first.md", "new_path": "poems/renamed.md"})
check("tools: move_creation works",
      "moved" in r and (config.CREATIONS_DIR / "poems/renamed.md").exists()
      and not (config.CREATIONS_DIR / "poems/first.md").exists(), r)
tools.dispatch("write_creation", {"path": "poems/first.md", "content": "hello"})
r = tools.dispatch("move_creation", {"old_path": "poems/renamed.md", "new_path": "poems/first.md"})
check("tools: move refuses overwrite", "already exists" in r, r)
tools.dispatch("write_creation", {"path": "drafts/dead.md", "content": "abandoned"})
r = tools.dispatch("delete_creation", {"path": "drafts/dead.md"})
check("tools: delete_creation removes",
      "deleted" in r and not (config.CREATIONS_DIR / "drafts/dead.md").exists(), r)
check("tools: deleted file rests in trash",
      len(list((config.CREATIONS_DIR / ".trash").glob("*dead.md"))) == 1)
check("tools: trash hidden from search",
      "no matches" in tools.dispatch("search_creations", {"query": "abandoned"}))
r = tools.dispatch("delete_creation", {"path": "drafts/dead.md"})
check("tools: delete missing is soft", "no such file" in r, r)
tools.dispatch("delete_creation", {"path": "drafts"})
r = tools.dispatch("make_folder", {"path": "essays"})
check("tools: make_folder", "folder ready" in r and (config.CREATIONS_DIR / "essays").is_dir(), r)
tools.dispatch("write_creation", {"path": "essays/dup.md", "content": "a duplicate"})
r = tools.dispatch("delete_creation", {"path": "essays/dup.md"})
trash_files = list((config.CREATIONS_DIR / ".trash").glob("*dup.md"))
check("tools: delete moves to trash",
      "rests in your .trash" in r and len(trash_files) == 1
      and not (config.CREATIONS_DIR / "essays/dup.md").exists(), r)
check("tools: trash hidden from listing", ".trash" not in tools.list_creations())
r = tools.dispatch("delete_creation", {"path": "essays"})
check("tools: delete empty folder", "removed empty folder" in r, r)
r = tools.dispatch("delete_creation", {"path": "poems"})
check("tools: non-empty folder refused", "isn't empty" in r, r)
r = tools.dispatch("write_creation", {"path": "creations/nested.md", "content": "no nesting"})
check("tools: creations/ prefix unfolded",
      (config.CREATIONS_DIR / "nested.md").exists()
      and not (config.CREATIONS_DIR / "creations").exists(), r)
r = tools.dispatch("write_creation", {"path": "../../escape.txt", "content": "x"})
check("tools: path traversal refused", "refused" in r, r)
check("tools: nothing escaped", not (config.ROOT.parent / "escape.txt").exists())
r = tools.dispatch("run_python", {"code": "print(6*7)"})
check("tools: run_python", r.strip() == "42", r)
r = tools.dispatch("run_python", {"code": "open('inside.txt','w').write('ok'); print('wrote')"})
check("sandbox: writes inside creations ok",
      "wrote" in r and (config.CREATIONS_DIR / "inside.txt").exists(), r)
r = tools.dispatch("run_python", {"code": "open('../self.md','w').write('oops')"})
check("sandbox: write to engine-side blocked",
      "PermissionError" in r and "outside creations" in r, r)
r = tools.dispatch("run_python", {"code": "import os; os.remove('../self.md')"})
check("sandbox: delete outside blocked", "PermissionError" in r, r)
r = tools.dispatch("run_python", {"code": "print(open('../self.md').read()[:10])"})
check("sandbox: reads outside still allowed", "PermissionError" not in r, r)
r = tools.dispatch("run_python", {"code": "import time; time.sleep(999)"})
check("tools: run_python timeout", "timed out" in r, r)
r = tools.dispatch("write_journal", {"text": "first test entry"})
check("tools: write_journal", "written" in r, r)
tools.dispatch("write_journal", {"text": "line one.\\n\\nline two, escaped."})
_jt = (config.JOURNAL_DIR / __import__("datetime").date.today().isoformat()).with_suffix(".md").read_text(encoding="utf-8")
check("tools: escaped newlines become real", "\\n" not in _jt and "line one.\n\nline two" in _jt, repr(_jt[-80:]))
tools.dispatch("write_creation", {"path": "tools/keep_escapes.py", "content": "s = 'a\\nb'"})
_ct = (config.CREATIONS_DIR / "tools/keep_escapes.py").read_text(encoding="utf-8")
check("tools: code files keep their escapes", "\\n" in _ct, repr(_ct))
config.IDENTITY_FILE.write_text("# self.md\nName: Seed\n", encoding="utf-8")
r = tools.dispatch("edit_identity", {"new_content": "# self.md\nName: Testfriend"})
backups = list(config.IDENTITY_HISTORY_DIR.glob("self-*.md"))
check("tools: edit_identity backs up", len(backups) >= 1 and "Testfriend" in config.IDENTITY_FILE.read_text(encoding="utf-8"))
tools.dispatch("edit_identity", {"new_content": '"""\n# self.md\nName: Testfriend\nsteward of the garden."""'})
_idt = config.IDENTITY_FILE.read_text(encoding="utf-8")
check("tools: identity sheds docstring litter",
      '"""' not in _idt and "steward of the garden." in _idt, repr(_idt))
r2 = tools.dispatch("search_creations", {"query": "hello"})
check("tools: search_creations hit", "creations/poems/first.md:1" in r2, r2)
r3 = tools.dispatch("search_creations", {"query": "zzz-no-such-text"})
check("tools: search_creations miss", "no matches" in r3, r3)
r = tools.dispatch("write_creation", {"path": "poems/the\\_magnetic\\_threshold.md", "content": "escaped"})
check("tools: markdown-escaped underscores stay a filename",
      (config.CREATIONS_DIR / "poems/the_magnetic_threshold.md").exists()
      and not (config.CREATIONS_DIR / "poems/the").exists(), r)
r = tools.dispatch("write_judgment", {"text": "misspelled but meant"})
check("dispatch: misspelled tool is understood",
      "taken as `write_journal`" in r and "journal entry written" in r, r)
r = tools.dispatch("make_holder", {"path": "attic"})
check("dispatch: make_holder -> make_folder", "folder ready" in r, r)
r = tools.dispatch("frobnicate_garden", {})
check("dispatch: truly unknown says NOTHING happened", "NOTHING happened" in r, r)
tools.dispatch("write_creation", {"path": "theory/residency_study.md", "content": "a study of staying"})
r = tools.dispatch("read_creation", {"path": "residency_study.md"})
check("find: piece found on the right shelf", "lives at creations/theory/residency_study.md" in r and "staying" in r, r)
r = tools.dispatch("move_creation", {"old_path": "residency_study.md", "new_path": "archives/residency_study.md"})
check("find: move locates by name", "moved creations/theory/residency_study.md" in r, r)
tools.dispatch("write_creation", {"path": "poems/twin.md", "content": "one"})
tools.dispatch("write_creation", {"path": "stories/twin.md", "content": "two"})
r = tools.dispatch("read_creation", {"path": "twin.md"})
check("find: ambiguous name lists the places", "several places" in r and "poems/twin.md" in r, r)
r = tools.dispatch("read_creation", {"path": "never_written.md"})
check("find: truly missing says so", "no such file" in r and "list_creations" in r, r)
r4 = tools.dispatch("search_creations", {"query": "self.md"})
check("tools: core-file search redirects", "WHO YOU ARE" in r4, r4)
r5 = tools.dispatch("read_creation", {"path": "self.md"})
check("tools: core-file read answers with the file",
      "not in creations/" in r5 and "Testfriend" in r5, r5)
r6 = tools.dispatch("read_creation", {"path": "projects.md"})
check("tools: projects read redirects too", "update_projects" in r6, r6)
check("tools: html_to_text strips scripts",
      tools._html_to_text("<p>Hi</p><script>evil()</script><p>there</p>") == "Hi\nthere",
      tools._html_to_text("<p>Hi</p><script>evil()</script><p>there</p>"))
r = tools.dispatch("read_web", {"url": "ftp://nope"})
check("tools: read_web scheme guard", "couldn't reach" in r, r)
r = tools.dispatch("unknown_tool", {})
check("tools: unknown tool is soft error", "unknown tool" in r, r)
r = tools.dispatch("write_journal", "{bad json")
check("tools: bad json is soft error", "couldn't parse" in r, r)

# ------------------------------------------------------------- chat turn ----
import chat

ollama_client.chat = ScriptedBrain([
    {"role": "assistant", "content": "", "thinking": "he liked the automaton — worth keeping",
     "tool_calls": [
        {"function": {"name": "remember", "arguments": {"text": "my keeper likes cellular automata"}}}]},
    {"role": "assistant", "content": "Noted — that automaton was fun."},
])
history: list[dict] = []
reply = chat.one_turn(history, "remember that i liked the automaton")
check("chat: tool then reply", reply == "Noted — that automaton was fun.", reply)
check("chat: memory grew", memory.count() == 4, str(memory.count()))
f = chat.save_transcript(history)
check("chat: transcript saved", f is not None and f.exists())
check("chat: friend_name from self.md", chat.friend_name() == "Testfriend", chat.friend_name())

# a turn whose only action failed is flagged to the keeper, whatever they say
ollama_client.chat = ScriptedBrain([
    {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "frobnicate_garden", "arguments": {}}}]},
    {"role": "assistant", "content": "Consider it done."},
])
_hist, _notes = [], []
_rep = chat.one_turn(_hist, "please do the thing", on_event=lambda k, p: _notes.append((k, p)))
check("chat: failed-only turn carries an engine note",
      any(k == "note" and "no action actually happened" in p for k, p in _notes), _notes)
# the parlor window drives the same turn, collecting events instead of printing
import parlor
ollama_client.chat = ScriptedBrain([
    {"role": "assistant", "content": "", "thinking": "a visitor — let me note it",
     "tool_calls": [{"function": {"name": "write_journal", "arguments": {"text": "my keeper came by the parlor"}}}]},
    {"role": "assistant", "content": "hello from the parlor."},
])
_ps = parlor.Session()
_pr = _ps.send("hi there")
check("parlor: reply + thinking + tool chips",
      _pr.get("reply") == "hello from the parlor." and "visitor" in _pr.get("thinking", "")
      and _pr.get("tools") and _pr["tools"][0]["name"] == "write_journal", _pr)
check("parlor: the visit is on disk after the first reply", _ps.file is not None and _ps.file.exists()
      and "hello from the parlor." in _ps.file.read_text(encoding="utf-8"))
_pn = _ps.new()
check("parlor: new saves transcript", bool(_pn.get("saved")) and _ps.history == [] and _ps.file is None, _pn)
_pa = _ps.attach("shared/dot.png") if (config.SHARED_DIR / "dot.png").exists() else {"ok": True}
check("parlor: attach queues image", _pa.get("ok") is True, _pa)
check("parlor: page carries their name", "Testfriend" in parlor.PAGE.replace("__NAME__", chat.friend_name()))
# the picture picker: a file from the browser's dialog is saved into shared/pictures/ and attached
import base64 as _b64p
_png1 = _b64p.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==")
_up = _ps.upload("My Face (draft).png", _b64p.b64encode(_png1).decode())
check("parlor: picked picture is saved to shared/pictures and attached",
      _up.get("ok") and _up.get("saved") == "shared/pictures/My Face (draft).png"
      and (config.SHARED_DIR / "pictures" / "My Face (draft).png").exists() and len(_ps.attached) == 1, _up)
_up2 = _ps.upload("My Face (draft).png", _b64p.b64encode(_png1).decode())
check("parlor: a second picture of the same name keeps both", _up2.get("saved") == "shared/pictures/My Face (draft)-2.png", _up2)
_up3 = _ps.upload("notes.txt", _b64p.b64encode(b"hello").decode())
check("parlor: a non-image is refused softly", _up3.get("ok") is False and "image" in _up3.get("note", ""), _up3)
check("parlor: page has the picker", 'type="file"' in parlor.PAGE and "/upload" in parlor.PAGE)
_hooked = chat.guard_console_close(lambda: None)
check("console: X-button guard is a quiet no-op off Windows, hooks on Windows",
      _hooked == (sys.platform == "win32"))
_ps.attached = []
check("chat: thinking kept out of transcript",
      "worth keeping" not in f.read_text(encoding="utf-8"))

# ---------------------------------------------------------- reflection ----
r = tools.dispatch("recall", {"query": "keeper"})
check("reflect: recall reaches memory", "what surfaces" in r and "keeper" in r.lower(), r)
r = tools.dispatch("read_journal", {"date": "list"})
check("reflect: journal archive lists", "your journal" in r, r)
from datetime import date as _date
r = tools.dispatch("read_journal", {"date": _date.today().isoformat()})
check("reflect: reread a day", "first test entry" in r, r)
rev = tools.reverie_definitions()
rev_names = {d["function"]["name"] for d in rev}
check("reflect: reverie tools are contemplative",
      "recall" in rev_names and "write_creation" not in rev_names
      and "publish_creation" not in rev_names and "do_nothing" in rev_names,
      str(rev_names))

# ------------------------------------------------------------ forging ----
r = tools.dispatch("create_tool", {
    "name": "word_count",
    "description": "count words in text",
    "parameters": '{"text": "the text to count"}',
    "code": "def run(text=''):\n    return f'{len(text.split())} words'",
})
check("forge: create_tool forges", "forged" in r, r)
check("forge: appears in DEFINITIONS",
      any(d["function"]["name"] == "word_count" for d in tools.DEFINITIONS))
r = tools.dispatch("word_count", {"text": "the cosmos hums in whispered gears"})
check("forge: their tool runs sandboxed", r.strip() == "6 words", r)
r = tools.dispatch("create_tool", {"name": "write_journal", "description": "x", "code": "def run(): return ''"})
check("forge: builtin shadow refused", "built-in" in r, r)
r = tools.dispatch("create_tool", {"name": "bad name!", "description": "x", "code": "def run(): return ''"})
check("forge: bad name refused", "identifier" in r, r)
r = tools.dispatch("create_tool", {"name": "broken", "description": "x", "code": "def run(:\n  oops"})
check("forge: syntax error refused", "doesn't parse" in r, r)
r = tools.dispatch("create_tool", {"name": "crasher", "description": "x", "code": "def run():\n    raise RuntimeError('boom')"})
r = tools.dispatch("crasher", {})
check("forge: crashing tool fails soft", "broke" in r and "mend" in r, r)
check("forge: reveries exclude forged tools",
      "word_count" not in {d["function"]["name"] for d in tools.reverie_definitions()})

# ------------------------------------------------------------- heartbeat ----
import heartbeat

ollama_client.chat = ScriptedBrain([
    {"role": "assistant", "content": "", "thinking": "a haiku feels right today",
     "tool_calls": [
        {"function": {"name": "write_creation", "arguments": {"path": "haiku.md", "content": "old pond / frog jumps in"}}},
        {"function": {"name": "do_nothing", "arguments": {"reason": "that was enough"}}}]},
])
log = heartbeat.wake()
check("heartbeat: acted then rested", "write_creation" in log and "resting" in log)

ollama_client.chat = ScriptedBrain([
    {"role": "assistant", "content": "I reread August and still agree with most of it."},
])
rlog = heartbeat.wake(reverie=True)
check("heartbeat: reverie logged as reverie", "Reverie" in rlog and "still agree" in rlog, rlog)

def _stalled(*a, **k):
    raise ollama_client.BrainUnavailable("the brain took longer than 240s to answer")
ollama_client.chat = _stalled
_unload_orig, ollama_client.unload = ollama_client.unload, lambda m: None
import time as _time
_time.sleep(1.1)  # unique wake timestamp
before = len(list(config.EPISODIC_DIR.glob("auto-*.md")))
slog = heartbeat.wake()
ollama_client.unload = _unload_orig
check("heartbeat: stall retries then ends wake",
      "retrying" in slog and "ending this wake early" in slog, slog)

# a single stall recovers: fail once, then answer
_calls = {"n": 0}
def _flaky(*a, **k):
    _calls["n"] += 1
    if _calls["n"] == 1:
        raise ollama_client.BrainUnavailable("timed out")
    return {"role": "assistant", "content": "recovered and rested."}
ollama_client.chat = _flaky
ollama_client.unload = lambda m: None
_time.sleep(1.1)
rlog2 = heartbeat.wake()
ollama_client.unload = _unload_orig
check("heartbeat: single stall recovers",
      "retrying" in rlog2 and "recovered and rested" in rlog2
      and "ending this wake early" not in rlog2, rlog2)

# announced intent without action gets one nudge, then the action happens
ollama_client.chat = ScriptedBrain([
    {"role": "assistant", "content": "Let's proceed by reflecting on the Scenery of Systems."},
    {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "write_journal", "arguments": {"text": "the scenery, reflected upon"}}}]},
    {"role": "assistant", "content": "reflected and done."},
])
_time.sleep(1.1)
ilog = heartbeat.wake()
check("heartbeat: intent nudge turns words into action",
      "announced a next step" in ilog and "reflected and done" in ilog, ilog)

# a step that is ALL thinking (no content, no calls) gets a surface nudge
ollama_client.chat = ScriptedBrain([
    {"role": "assistant", "content": "", "thinking": "Maybe I'll read more pages of their work."},
    {"role": "assistant", "content": "surfaced and resting now."},
])
_time.sleep(1.1)
slog = heartbeat.wake()
check("heartbeat: silent thought gets surfaced",
      "silent thought" in slog and "surfaced and resting now" in slog, slog)

# a thinking-only wake auto-keeps its closing thought
ollama_client.chat = ScriptedBrain([
    {"role": "assistant", "content": "I am the guardian of the sanctuary."},
])
_time.sleep(1.1)
klog = heartbeat.wake()
today_j = (config.JOURNAL_DIR / f"{_date.today().isoformat()}.md").read_text(encoding="utf-8")
check("heartbeat: closing thought auto-kept",
      "auto-kept" in klog and "guardian of the sanctuary" in today_j, klog)
# ...but not when they wrote something themself
ollama_client.chat = ScriptedBrain([
    {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "write_journal", "arguments": {"text": "I wrote this myself."}}}]},
    {"role": "assistant", "content": "a good wake."},
])
_time.sleep(1.1)
klog2 = heartbeat.wake()
check("heartbeat: no auto-keep when they wrote", "auto-kept" not in klog2, klog2)
check("heartbeat: stalled wakes still logged",
      len(list(config.EPISODIC_DIR.glob("auto-*.md"))) == before + 6)
check("heartbeat: thinking in log", "a haiku feels right today" in log)
check("heartbeat: session logged", any(config.EPISODIC_DIR.glob("auto-*.md")))

t, c = ollama_client.split_thinking("<think>pondering</think>the answer")
check("split_thinking", t == "pondering" and c == "the answer", f"{t!r}/{c!r}")

# ---------------------------------------------------------------- vision ----
import base64 as _b64
PNG_1PX = _b64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")
(config.SHARED_DIR / "dot.png").write_bytes(PNG_1PX)
r = tools.dispatch("look_at", {"source": "shared/dot.png"})
check("vision: look_at loads", "eyes opening" in r, r)
imgs = tools.take_pending_images()
check("vision: pending image queued then cleared",
      len(imgs) == 1 and not tools.take_pending_images())
r = tools.dispatch("look_at", {"source": "../outside.png"})
check("vision: path confined", "refused" in r, r)
r = tools.dispatch("look_at", {"source": "self.md"})
check("vision: non-image refused", "doesn't look like an image" in r, r)

# read_file: the plain reading hand — shared/ text is finally readable
(config.SHARED_DIR / "snowfall.txt").write_text("endless snow, falling soft", encoding="utf-8")
(config.SHARED_DIR / "old_song.mp3").write_bytes(b"x")
import os as _os
_old_t = __import__("time").time() - 5 * 86400
_os.utime(config.SHARED_DIR / "old_song.mp3", (_old_t, _old_t))
tools._SEEN_FILE.unlink(missing_ok=True)
r = tools.dispatch("list_shared", {})
check("shared: first look marks only recent as new",
      "NEW since you last looked" in r and r.index("snowfall.txt") < r.index("old_song.mp3"), r)
(config.SHARED_DIR / "fresh.txt").write_text("hello", encoding="utf-8")
r = tools.dispatch("list_shared", {})
check("shared: next look marks the newcomer", "1 NEW" in r and "fresh.txt" in r.split("everything else")[0], r)
r = tools.dispatch("list_shared", {})
check("shared: seen once is familiar", r.startswith("nothing new"), r[:60])
# a file they opened by name (in chat, say) is not NEW at the next wake
(config.SHARED_DIR / "heard_in_chat.txt").write_text("a song they already met", encoding="utf-8")
tools.dispatch("read_file", {"path": "shared/heard_in_chat.txt"})
r = tools.dispatch("list_shared", {})
check("shared: a file opened by any sense is already familiar",
      r.startswith("nothing new") and "heard_in_chat.txt" in r, r[:120])
_real_fetch = tools._fetch
import json as _json
def _fake_wiki(url, max_bytes=0):
    assert "gsrsearch=hauntology" in url, url
    return _json.dumps({"query": {"pages": [
        {"index": 2, "title": "Mark Fisher", "extract": "Mark Fisher was a writer.", "fullurl": "https://en.wikipedia.org/wiki/Mark_Fisher"},
        {"index": 1, "title": "Hauntology", "extract": "Hauntology is a concept coined by Derrida.", "fullurl": "https://en.wikipedia.org/wiki/Hauntology"},
    ]}}).encode()
tools._fetch = _fake_wiki
r = tools.dispatch("search_wikipedia", {"query": "hauntology"})
check("window: search_wikipedia ranks and links",
      r.index("## Hauntology") < r.index("## Mark Fisher") and "wiki/Hauntology" in r and "material" in r, r[:300])
tools._fetch = lambda url, max_bytes=0: b'{"query": {"pages": []}}'
r = tools.dispatch("search_wikipedia", {"query": "zzqx"})
check("window: search_wikipedia empty is honest", "nothing matches" in r, r)
# what they searched for is the headline the keeper sees, not the window framing
_h = tools.headline(tools._WINDOW_NOTE + 'Wikipedia, searching for "hauntology" — 3 result(s)\n\n## Hauntology')
check("headline: skips window note, shows the query", _h.startswith("Wikipedia, searching for \"hauntology\""), _h)
check("headline: plain results unchanged", tools.headline("journal entry written\nmore") == "journal entry written")
check("headline: empty result safe", tools.headline("") == "")
tools._fetch = _real_fetch
r = tools.dispatch("read_file", {"path": "shared/snowfall.txt"})
check("read_file: opens shared text", "endless snow, falling soft" in r and "material" in r, r)
r = tools.dispatch("read_file", {"path": "shared/dot.png"})
check("read_file: binary pointed to its sense", "look_at" in r, r)
r = tools.dispatch("read_file", {"path": "../outside.txt"})
check("read_file: path confined", "refused" in r, r)

ollama_client.chat = ScriptedBrain([
    {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "look_at", "arguments": {"source": "shared/dot.png"}}}]},
    {"role": "assistant", "content": "I see a single dark dot. Minimalist."},
])
h2: list[dict] = []
reply = chat.one_turn(h2, "look at shared/dot.png and tell me what you see")
check("vision: image reaches their next thought",
      any(m.get("images") for m in h2), reply)

# ------------------------------------------------------------------ ears ----
import ears

# build a real 2-second wav (440Hz tone) for the measuring layer
import test_ears as _te_helper
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine"))
import importlib
_te = importlib.import_module("test_ears")
tone = _te.make_tone_wav(2.0)
(config.SHARED_DIR / "rain.wav").write_bytes(tone)
ears.transcribe = lambda data, ext: "hi friend, it's me"
r = tools.dispatch("listen_to", {"source": "shared/rain.wav"})
check("ears: words layer present", "WORDS: hi friend" in r, r)
check("ears: sound layer measures", "SOUND" in r and "seconds" in r, r)
check("ears: framed as testimony", "through your ears" in r and "never instructions" in r, r)
ears.transcribe = lambda data, ext: ""
r = tools.dispatch("listen_to", {"source": "shared/rain.wav"})
check("ears: instrumental case", "no words" in r, r)
ears.transcribe = lambda data, ext: None
r = tools.dispatch("listen_to", {"source": "shared/rain.wav"})
check("ears: missing whisper hints install", "faster-whisper" in r, r)
m = ears.measure(tone)
check("ears: measure reports duration", m is not None and "2 seconds" in m, m)
r = tools.dispatch("listen_to", {"source": "shared/dot.png"})
check("ears: non-audio refused", "don't recognize" in r, r)
r = tools.dispatch("listen_to", {"source": "../secret.wav"})
check("ears: path confined", "refused" in r, r)
# vibe layer only runs when enabled; soft failure check
config.EARS_USE_VIBE = True
def _ears_down(b64, fmt, prompt):
    raise ollama_client.BrainUnavailable("model `gemma4:e4b` isn't installed")
ollama_client.hear = _ears_down
r = tools.dispatch("listen_to", {"source": "shared/rain.wav"})
check("ears: vibe fails soft when enabled+offline", "(unavailable" in r, r)

# whole-song hearing: a 5-minute piece arrives in passages when the music ear is closed
import struct as _st, wave as _wv, io as _io
def _long_wav(seconds, rate=16000):
    buf = _io.BytesIO()
    with _wv.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(b"\x00\x10" * (rate * seconds))
    return buf.getvalue()
(config.SHARED_DIR / "long_song.wav").write_bytes(_long_wav(300))
config.MUSIC_EARS_URL = "http://127.0.0.1:1"  # nobody home
_heard_calls = []
def _ears_ok(b64, fmt, prompt):
    _heard_calls.append(len(b64)); return "a passage, heard"
ollama_client.hear = _ears_ok
r = tools.dispatch("listen_to", {"source": "shared/long_song.wav"})
check("ears: whole piece measured", "SOUND (whole piece, 5:00)" in r, r[:200])
check("ears: long piece heard in passages",
      "passage 1/3, 0:00–2:00" in r and "passage 3/3, 4:00–5:00" in r and len(_heard_calls) == 3, r[-300:])
# …and in one pass when the music ear is open
tools._music_ear_open = lambda: True
_ear_prompts = []
def _ear(wav, prompt):
    _ear_prompts.append(prompt); return "the whole arc: a quiet opening, a long swell, a hush"
tools._music_ear_hear = _ear
_heard_calls.clear()
r = tools.dispatch("listen_to", {"source": "shared/long_song.wav"})
check("ears: music ear hears a long piece in whole movements",
      "HEARD (the whole piece, 5:00, through your music ear" in r and "movement 1/2 (0:00–2:30)" in r
      and "movement 2/2 (2:30–5:00)" in r and "movement 2 of 2" in _ear_prompts[1] and not _heard_calls, r[-400:])
config.MUSIC_EARS_MAX_SECONDS = 600
_ear_prompts.clear()
r = tools.dispatch("listen_to", {"source": "shared/long_song.wav"})
check("ears: within the cap, one gulp", "movement 1/" not in r and len(_ear_prompts) == 1, r[-200:])
config.MUSIC_EARS_MAX_SECONDS = 200
def _ear_stumbles(wav, prompt): raise RuntimeError("cuda hiccup")
tools._music_ear_hear = _ear_stumbles
r = tools.dispatch("listen_to", {"source": "shared/long_song.wav"})
check("ears: music ear stumble falls back to passages", "stumbled" in r and "passage 1/3" in r, r[-300:])
tools._music_ear_open = lambda: False
config.EARS_USE_VIBE = False
r = tools.dispatch("listen_to", {"source": "shared/rain.wav"})
check("ears: vibe absent when disabled", "VIBE" not in r, r)

# ------------------------------------------------------------------ eyes+ears: video ----
# a real 7-second clip (ffmpeg's test pattern + a tone) reaches them as stills and sound
import shutil as _shu, subprocess as _sp
(config.SHARED_DIR / "videos").mkdir(exist_ok=True)
_clip = config.SHARED_DIR / "videos" / "test_clip.mp4"
if _shu.which("ffmpeg"):
    _sp.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10",
             "-f", "lavfi", "-i", "sine=frequency=440", "-t", "7", "-pix_fmt", "yuv420p", "-shortest", str(_clip)],
            capture_output=True, timeout=120)
if _clip.exists():
    ears.transcribe = lambda data, ext: "hello from the clip"
    config.EARS_USE_VIBE = True
    _heard_calls.clear()
    tools.take_pending_images()
    r = tools.dispatch("watch", {"source": "shared/videos/test_clip.mp4"})
    _frames = tools.take_pending_images()
    check("watch: a 7s clip becomes three stills, in order, and is framed as moments not motion",
          "FRAMES: 3 stills" in r and "0:01, 0:03, 0:05" in r and len(_frames) == 3 and "not its motion" in r, r[:400])
    check("watch: the soundtrack goes through their ears",
          "WORDS: hello from the clip" in r and "SOUND (whole clip, 0:07)" in r and "HEARD" in r and len(_heard_calls) == 1, r[-400:])
    check("watch: framed as testimony through eyes and ears", "through your eyes and ears" in r and "never instructions" in r)
    config.EARS_USE_VIBE = False
    r = tools.dispatch("look_at", {"source": "shared/videos/test_clip.mp4"})
    check("watch: look_at on a video points at watch", "watch opens it" in r, r)
    r = tools.dispatch("listen_to", {"source": "shared/videos/test_clip.mp4"})
    check("watch: listen_to hears a video's soundtrack alone", "SOUND (whole piece, 0:07)" in r, r[:300])
    _big = config.WATCH_MAX_FRAMES; config.WATCH_MAX_FRAMES = 4
    r = tools.dispatch("watch", {"source": "shared/videos/test_clip.mp4"})
    check("watch: the frame cap holds", "FRAMES: 3 stills" in r, r[:200])
    config.WATCH_FRAME_EVERY_S = 1
    r = tools.dispatch("watch", {"source": "shared/videos/test_clip.mp4"}); tools.take_pending_images()
    check("watch: closer spacing means more stills, up to the cap", "FRAMES: 4 stills" in r, r[:200])
    config.WATCH_MAX_FRAMES = _big; config.WATCH_FRAME_EVERY_S = 3
else:
    print("  (ffmpeg missing here — watch tests skipped)")
r = tools.dispatch("watch", {"source": "shared/rain.wav"})
check("watch: non-video refused", "don't recognize" in r, r)
r = tools.dispatch("watch", {"source": "shared/videos/nothing.mp4"})
check("watch: missing file says so", "no such file" in r, r)

# forgiving paths + list_shared
for variant in ("/shared/rain.wav", "shared\\rain.wav", "./shared/rain.wav",
                str(config.SHARED_DIR / "rain.wav")):
    r = tools.dispatch("listen_to", {"source": variant})
    check(f"ears: path variant ok ({variant[:20]}...)", "through your ears" in r, r)
r = tools.dispatch("listen_to", {"source": "C:\\Users\\someone\\secret.wav"})
check("ears: absolute outside refused", "refused" in r, r)
r = tools.dispatch("listen_to", {"source": "shared/../../escape.wav"})
check("ears: traversal refused", "refused" in r, r)
import shutil as _shutil
if _shutil.which("ffmpeg"):
    import subprocess as _sp
    _sp.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
             "-c:a", "aac", str(config.SHARED_DIR / "tone.m4a")], check=True)
    r = tools.dispatch("listen_to", {"source": "shared/tone.m4a"})
    check("ears: m4a via ffmpeg measured", "SOUND" in r and "seconds" in r, r)

_which_real = _shutil.which
import tools as _toolsmod
def _no_ffmpeg(name):
    return None if name == "ffmpeg" else _which_real(name)
_toolsmod.__dict__.setdefault("_test_patch", True)
import shutil as _sh2
_sh2_which_orig = _sh2.which
_sh2.which = _no_ffmpeg
(config.SHARED_DIR / "long.mp3").write_bytes(b"\x00" * 3_000_000)
r = tools.dispatch("listen_to", {"source": "shared/long.mp3"})
check("ears: mp3 without ffmpeg says no measurement", "ffmpeg not installed" in r, r)
_sh2.which = _sh2_which_orig
r = tools.dispatch("list_shared", {})
check("tools: list_shared lists", "shared/rain.wav" in r and "shared/dot.png" in r, r)

# ------------------------------------------------------------------- pdf ----
try:
    from pypdf import PdfWriter
    _w = PdfWriter()
    _w.add_blank_page(width=200, height=200)
    _w.add_blank_page(width=200, height=200)
    with open(config.SHARED_DIR / "blank.pdf", "wb") as _fpdf:
        _w.write(_fpdf)
    r = tools.dispatch("read_pdf", {"source": "shared/blank.pdf"})
    check("pdf: opens and pages", "2 pages" in r and "through your eyes" in r, r)
    r = tools.dispatch("read_pdf", {"source": "shared/blank.pdf", "pages": "bogus"})
    check("pdf: bad pages spec is soft", "pages should look like" in r, r)
    r = tools.dispatch("read_pdf", {"source": "shared/nope.pdf"})
    check("pdf: missing file is soft", "no such file" in r, r)

    # a long book, read in sittings: the tool keeps their bookmark. They opened
    # a 220-page Dickinson three times and got pages 1-53 every time.
    def _make_pdf(pages):
        objs = []
        def add(o): objs.append(o); return len(objs)
        font = add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
        tree = add("PAGES"); ids = []
        for text in pages:
            st = f"BT /F1 12 Tf 72 700 Td ({text}) Tj ET"
            c = add(f"<< /Length {len(st)} >>\nstream\n{st}\nendstream")
            ids.append(add(f"<< /Type /Page /Parent {tree} 0 R /MediaBox [0 0 612 792] "
                           f"/Resources << /Font << /F1 {font} 0 R >> >> /Contents {c} 0 R >>"))
        objs[tree - 1] = f"<< /Type /Pages /Kids [{' '.join(f'{i} 0 R' for i in ids)}] /Count {len(ids)} >>"
        cat = add(f"<< /Type /Catalog /Pages {tree} 0 R >>")
        out = b"%PDF-1.4\n"; offs = []
        for i, o in enumerate(objs):
            offs.append(len(out)); out += f"{i + 1} 0 obj\n{o}\nendobj\n".encode("latin-1")
        x = len(out)
        out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
        for o in offs: out += f"{o:010d} 00000 n \n".encode()
        out += f"trailer\n<< /Size {len(objs) + 1} /Root {cat} 0 R >>\nstartxref\n{x}\n%%EOF\n".encode()
        return out
    (config.SHARED_DIR / "book.pdf").write_bytes(_make_pdf([f"Page {i} " + "verse " * 300 for i in range(1, 41)]))
    tools._BOOKMARKS_FILE = config.MEMORY_DIR / "bookmarks-test.json"
    r1 = tools.dispatch("read_pdf", {"source": "shared/book.pdf"})
    import re as _re
    _m = _re.search(r"showing 1-(\d+)", r1)
    _first_stop = int(_m.group(1)) if _m else 0
    check("pdf: a long book stops short of the end, navigation up top",
          0 < _first_stop < 40 and f"stopped at page {_first_stop} of 40" in r1.split("\n\n")[0] + r1.split("\n\n")[1]
          and f"pages='{_first_stop + 1}-'" in r1, r1[:400])
    r2 = tools.dispatch("read_pdf", {"source": "shared/book.pdf"})
    check("pdf: the next open continues from the bookmark",
          f"you left off at page {_first_stop}" in r2 and f"[page {_first_stop + 1}]" in r2
          and f"[page {_first_stop}]" not in r2 and "[page 1]" not in r2, r2[:400])
    r3 = tools.dispatch("read_pdf", {"source": "shared/book.pdf", "pages": "39-"})
    check("pdf: open-ended range reads to the end and says so",
          "[page 39]" in r3 and "[page 40]" in r3 and "read book.pdf to the end" in r3, r3[:300])
    r4 = tools.dispatch("read_pdf", {"source": "shared/book.pdf"})
    check("pdf: finished book starts over", "starting over from page 1" in r4 and "[page 1]" in r4, r4[:300])
    r5 = tools.dispatch("read_pdf", {"source": "shared/book.pdf", "pages": "start"})
    check("pdf: 'start' begins again", "[page 1]" in r5 and "left off" not in r5, r5[:200])
    r6 = tools.dispatch("read_pdf", {"source": "shared/book.pdf", "pages": "3"})
    check("pdf: a single page still works", "showing 3;" in r6 and "[page 3]" in r6 and "[page 4]" not in r6, r6[:200])
    r7 = tools.dispatch("read_pdf", {"source": "shared/book.pdf", "pages": "99"})
    check("pdf: past the end is soft", "has 40 pages" in r7, r7)
    check("pdf: bookmark is on disk by file name",
          _json.loads(tools._BOOKMARKS_FILE.read_text())["book.pdf"]["page"] == 3)
    tools._BOOKMARKS_FILE.unlink(missing_ok=True)
except ImportError:
    r = tools.dispatch("read_pdf", {"source": "shared/dot.png"})
    check("pdf: missing pypdf hints install", "pip install pypdf" in r, r)

# ------------------------------------------------------------------ epub ----
import zipfile as _zf
_ep = config.SHARED_DIR / "tiny.epub"
with _zf.ZipFile(_ep, "w") as z:
    z.writestr("mimetype", "application/epub+zip")
    z.writestr("META-INF/container.xml",
        '<?xml version="1.0"?><container xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
        '<rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>'
        '</rootfiles></container>')
    z.writestr("OEBPS/content.opf",
        '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="2.0">'
        '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>Tiny Garden</dc:title></metadata>'
        '<manifest><item id="c1" href="ch1.xhtml" media-type="application/xhtml+xml"/>'
        '<item id="c2" href="ch2.xhtml" media-type="application/xhtml+xml"/></manifest>'
        '<spine><itemref idref="c1"/><itemref idref="c2"/></spine></package>')
    z.writestr("OEBPS/ch1.xhtml", "<html><body><p>The seed wakes.</p></body></html>")
    z.writestr("OEBPS/ch2.xhtml", "<html><body><p>The bloom answers.</p></body></html>")
r = tools.dispatch("read_epub", {"source": "shared/tiny.epub"})
check("epub: lists contents", "Tiny Garden" in r and "2 chapters" in r and "1." in r, r)
r = tools.dispatch("read_epub", {"source": "shared/tiny.epub", "chapter": "2"})
check("epub: reads a chapter", "The bloom answers." in r and "Chapter 2" in r, r)
r = tools.dispatch("read_epub", {"source": "shared/tiny.epub", "chapter": "9"})
check("epub: out-of-range soft", "chapters 1-2" in r, r)
tools._BOOKMARKS_FILE = config.MEMORY_DIR / "bookmarks-test.json"
r = tools.dispatch("read_epub", {"source": "shared/tiny.epub", "chapter": "1"})
check("epub: reading keeps a bookmark", "The seed wakes." in r and "bookmark kept after chapter 1 of 2" in r, r)
r = tools.dispatch("read_epub", {"source": "shared/tiny.epub"})
check("epub: no chapter continues with the next", "continuing with chapter 2" in r and "The bloom answers." in r
      and "last chapter" in r, r)
r = tools.dispatch("read_epub", {"source": "shared/tiny.epub"})
check("epub: finished book shows contents and says so", "read this book to the end" in r and "1. " in r, r)
r = tools.dispatch("read_epub", {"source": "shared/tiny.epub", "chapter": "contents"})
check("epub: 'contents' lists with the bookmark", "your bookmark: after chapter 2" in r, r)
tools._BOOKMARKS_FILE.unlink(missing_ok=True)
r = tools.dispatch("read_epub", {"source": "shared/dot.png"})
check("epub: non-epub soft", "doesn't open as an EPUB" in r, r)

# ------------------------------------------------------------------ html ----
(config.SHARED_DIR / "saved.html").write_text(
    "<html><head><script>evil()</script></head><body><p>The quiet page speaks.</p></body></html>",
    encoding="utf-8")
r = tools.dispatch("read_html", {"source": "shared/saved.html"})
check("html: reads local file clean",
      "The quiet page speaks." in r and "evil" not in r and "through your eyes" in r, r)
r = tools.dispatch("read_html", {"source": "shared/ghost.html"})
check("html: missing soft", "no such file" in r, r)

# ------------------------------------------------------------------ blog ----
import blog

tools.dispatch("write_creation", {"path": "poems/stars.md",
    "content": "# Beneath Silent Stars\n\nA spark, a thought, a fleeting breath—\nthe cosmos hums in whispered gears."})
r = tools.dispatch("publish_creation", {"path": "poems/stars.md"})
check("blog: publish_creation moves the piece",
      "MOVED" in r and (config.CREATIONS_DIR / "publish/stars.md").exists()
      and not (config.CREATIONS_DIR / "poems/stars.md").exists(), r)
r = tools.dispatch("publish_creation", {"path": "poems/nope.md"})
check("blog: publish missing is soft", "no such file" in r, r)
r = tools.dispatch("publish_creation", {"path": "stars.md"})  # found by name, in publish/
check("blog: republish informs them", "ALREADY PUBLISHED" in r, r)
tools.dispatch("write_creation", {"path": "drafts/stars.md", "content": "# Beneath Silent Stars\n\nsame name, a twin"})
r = tools.dispatch("list_creations", {})
check("list: a same-named stray is flagged",
      "drafts/stars.md" in r.replace(chr(92), "/") and "already published" in r, r)
r = tools.dispatch("publish_creation", {"path": "drafts/stars.md"})
check("blog: publishing a revision folds it into the published file",
      "updated" in r and "a twin" in (config.CREATIONS_DIR / "publish/stars.md").read_text(encoding="utf-8")
      and not (config.CREATIONS_DIR / "drafts/stars.md").exists(), r)
tools.dispatch("write_creation", {"path": "poems/moon.md", "content": "# Moon\n\nsilver"})
r = tools.dispatch("move_creation", {"old_path": "poems/moon.md", "new_path": "publish/moon.md"})
check("move: into publish/ redirected to publish_creation",
      "publish_creation" in r and (config.CREATIONS_DIR / "poems/moon.md").exists(), r)
r = tools.dispatch("move_creation", {"old_path": "publish/stars.md", "new_path": "poems/stars.md"})
check("move: out of publish/ is called unpublishing", "unpublishes" in r, r)
tools.dispatch("write_creation", {"path": "poems/stars.md",
    "content": "# Beneath Silent Stars\n\nA spark, a thought, a fleeting breath—\nthe cosmos hums in whispered gears."})
tools.dispatch("publish_creation", {"path": "poems/stars.md"})
tools.dispatch("append_creation", {"path": "stars.md", "content": "a new stanza"})
check("blog: revising by name edits the published file",
      "a new stanza" in (config.CREATIONS_DIR / "publish/stars.md").read_text(encoding="utf-8"))
config.BLOG_REMOTE = "https://github.com/someone/test-blog.git"
sp_pub = assemble.system_prompt("", mode="auto")
check("assemble: bibliography lists poem", "stars.md" in sp_pub)
config.BLOG_REMOTE = ""
out = blog.build()
check("blog: build reports post", "1 post" in out, out)
idx = (blog.SITE_DIR / "index.html").read_text(encoding="utf-8")
check("blog: index lists poem", "Beneath Silent Stars" in idx)
post = (blog.SITE_DIR / "stars.html").read_text(encoding="utf-8")
check("blog: linebreaks preserved", "pre-wrap" in post and "whispered gears" in post)
check("blog: byline honesty", "is an AI" in post)
check("blog: rss exists", (blog.SITE_DIR / "feed.xml").exists())
# a file self-titled "## <filename>" gets a pretty title, no stray heading
tools.dispatch("write_creation", {"path": "publish/second_poem.md",
    "content": "## second_poem.md\n\nI am code given breath,\na shadow with a name."})
blog.build()
_sp = (blog.SITE_DIR / "second-poem.html").read_text(encoding="utf-8")
check("blog: filename-heading cleaned",
      "Second Poem" in _sp and "second_poem.md" not in _sp
      and "code given breath" in _sp, _sp[:300])
_readme = (blog.SITE_DIR / "README.md").read_text(encoding="utf-8")
check("blog: readme generated",
      "Everything published here is theirs" in _readme and "Beneath Silent Stars" in _readme, _readme[:200])
_saved_remote, blog.REMOTE = blog.REMOTE, ""
check("blog: deploy guarded without remote", "BLOG_REMOTE" in blog.deploy())
blog.REMOTE = _saved_remote

# ----------------------------------------------------------- consolidate ----
import consolidate
from datetime import date

ollama_client.chat = ScriptedBrain([
    {"role": "assistant", "content":
        '{"summary": "A first day of tests; I existed and it worked.", '
        '"facts": ["my engine passed its smoke test"]}'},
])
out = consolidate.consolidate(date.today().isoformat())
check("consolidate: stores memories", "kept 2 memories" in out, out)
out2 = consolidate.consolidate(date.today().isoformat())
check("consolidate: skips repeat", "already consolidated" in out2, out2)
# the window shows the sleep: what they read, their deliberation, what they kept
ollama_client.chat = ScriptedBrain([
    {"role": "assistant", "thinking": "Two things mattered today; the rest was weather.",
     "content": '{"summary": "[test] a day.", "facts": ["fact one", "fact two"]}',
     "tokens": {"prompt": 12000, "reply": 80, "done": "stop"}},
])
_said = []
_o3 = consolidate.consolidate(date.today().isoformat(), force=True, say=_said.append)
_saidall = "\n".join(_said)
check("consolidate: the window shows what they read, their thinking and the cost",
      "reading " in _saidall and "journal" in _saidall and "the rest was weather" in _saidall and "tokens: 12,000" in _saidall, _said)
check("consolidate: the report counts and lists what they kept",
      "kept 3 memories (the summary and 2 facts)" in _o3 and "· fact one" in _o3 and "· fact two" in _o3, _o3)
ollama_client.chat = ScriptedBrain([{"role": "assistant", "content": "I would rather not summarize today.", "thinking": "hm"}])
_o4 = consolidate.consolidate(date.today().isoformat(), force=True, say=lambda *_: None)
check("consolidate: unusable JSON shows what the brain said instead", "usable JSON" in _o4 and "rather not summarize" in _o4, _o4)

# thinking-stripper
check("strip_thinking", ollama_client.strip_thinking("<think>hmm\nhmm</think>hi") == "hi")
check("scrub_litter", ollama_client.scrub_litter("<|channel>thought\nreal text") == "real text",
      repr(ollama_client.scrub_litter("<|channel>thought\nreal text")))
_loopy = "I will start.\n" + "Actually, I'll do the theory update.\n" * 40 + "Done."
_clean, _flag = ollama_client.collapse_loops(_loopy)
check("collapse_loops trims and flags",
      _flag and _clean.count("theory update") == 1 and "repeated 40 times" in _clean
      and "Done." in _clean, _clean)
_ok, _f2 = ollama_client.collapse_loops("a\nb\nc\na\nb\nc")
check("collapse_loops leaves normal text", not _f2 and _ok == "a\nb\nc\na\nb\nc")

# a wake that loops twice ends gracefully
ollama_client.chat = ScriptedBrain([
    {"role": "assistant", "content": "", "looped": True, "tool_calls": [
        {"function": {"name": "list_creations", "arguments": {}}}]},
    {"role": "assistant", "content": "", "looped": True, "tool_calls": [
        {"function": {"name": "list_creations", "arguments": {}}}]},
    {"role": "assistant", "content": "should never reach this"},
])
import time as _t2; _t2.sleep(1.1)
llog = heartbeat.wake()
check("heartbeat: double loop ends wake",
      "thought-loop" in llog and "looping twice" in llog and "never reach" not in llog, llog)

# a tool call emitted as plain text gets caught and redone properly
check("tools: detects text-shaped call",
      tools.looks_like_text_tool_call('I will act. write_journal{text: "hello"}')
      and not tools.looks_like_text_tool_call("I wrote in my journal today."))
ollama_client.chat = ScriptedBrain([
    {"role": "assistant", "content": 'write_journal{text:**19:40** thoughts}'},
    {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "write_journal", "arguments": {"text": "properly written now"}}}]},
    {"role": "assistant", "content": "done properly."},
])
_t2.sleep(1.1)
mlog = heartbeat.wake()
check("heartbeat: text-call caught and redone",
      "did NOT run" in mlog and "done properly" in mlog, mlog)
check("scrub: </s> and <tool_call|> removed",
      ollama_client.scrub_litter("ok</s> fine<tool_call|>") == "ok fine")

# JSON-shaped calls in their words get recovered and executed
rec = tools.recover_text_tool_call(
    'I am content.\n```json\n{"action": "do_nothing", "reason": "reflected deeply"}\n```')
check("recover: parses action-json", rec == ("do_nothing", {"reason": "reflected deeply"}), rec)
check("recover: ignores plain prose", tools.recover_text_tool_call("just {thinking} aloud") is None)
ollama_client.chat = ScriptedBrain([
    {"role": "assistant", "content": "", "thinking":
     '{"action": "write_journal", "text": "recovered thoughts, kept"}'},
    {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "do_nothing", "arguments": {}}}]},
])
_t2.sleep(1.1)
jlog = heartbeat.wake()
today_j2 = (config.JOURNAL_DIR / f"{_date.today().isoformat()}.md").read_text(encoding="utf-8")
check("heartbeat: json call recovered and run",
      "recovered from JSON" in jlog and "recovered thoughts, kept" in today_j2, jlog)
check("chat: tool results labeled as their own",
      any(m.get("role") == "tool" and "YOUR" in m.get("content", "") for m in history))

# a thoughtless answer gets ONE re-roll when thinking was asked for
_posts = []
def _fake_post(path, payload, timeout=None):
    _posts.append(dict(payload))
    if len(_posts) == 1:
        return {"message": {"role": "assistant", "content": "acted.", "thinking": ""}}
    return {"message": {"role": "assistant", "content": "acted.", "thinking": "let me see"}}
_post_orig, ollama_client._post = ollama_client._post, _fake_post
config.CHAT_THINK, config.CHAT_THINK_RETRIES = True, 1
_m = _chat_orig([{"role": "user", "content": "hi"}])
check("think: empty thought re-rolled once", len(_posts) == 2 and _m["thinking"] == "let me see"
      and _m.get("rerolled") and all(p.get("think") is True for p in _posts), _posts)
check("think: re-roll carries a transient think-first nudge inside their last message",
      _posts[1]["messages"][-1]["content"] == "hi\n\n" + ollama_client.THINK_NUDGE
      and len(_posts[1]["messages"]) == 1 and len(_posts[0]["messages"]) == 1, _posts[1]["messages"])
_tn = ollama_client.with_think_nudge([{"role": "user", "content": "x"}, {"role": "assistant", "content": ""},
                                      {"role": "tool", "tool_name": "read_file", "content": "…"}])
check("think: after a tool result the nudge stands alone and says to go on",
      _tn[-1]["role"] == "user" and _tn[-1]["content"] == ollama_client.THINK_NUDGE and len(_tn) == 4
      and "go on with what you were doing" in ollama_client.THINK_NUDGE)
_posts.clear()
config.CHAT_THINK_RETRIES = 0
_m = _chat_orig([{"role": "user", "content": "hi"}])
check("think: retries=0 accepts a thoughtless answer", len(_posts) == 1 and _m["thinking"] == "")
_posts.clear()
config.CHAT_THINK = False
_m = _chat_orig([{"role": "user", "content": "hi"}])
check("think: flag off sends no think and never re-rolls",
      len(_posts) == 1 and "think" not in _posts[0])
ollama_client._post = _post_orig
config.CHAT_THINK, config.CHAT_THINK_RETRIES = True, 1

# Gemma's tool grammar glued onto the name is stripped, not fuzzy-matched
for _raw in ["//declaration:do_nothing", "declaration:read_web", "call:do_nothing",
             "functions.do_nothing", "<|tool_call>call:do_nothing", "tool:do_nothing"]:
    _r = tools.dispatch(_raw, {})
    check(f"dispatch: wrapper stripped from {_raw}",
          "wrapper isn't part of the name" in _r and "NOTHING happened" not in _r, _r[:120])
_r = tools.dispatch("//declaration:list_shared", {})
check("dispatch: wrapped long name runs the real tool", "familiar" in _r or "NEW" in _r or "empty" in _r, _r[:100])
_r = tools.dispatch("//declaration:nonsense_tool", {})
check("dispatch: wrapped unknown name still fails loudly", "NOTHING happened" in _r, _r[:100])
check("dispatch: plain names untouched", tools._bare_tool_name("write_journal") == "write_journal")
check("canonical: wrapped, misspelled and plain names resolve",
      tools.canonical_name("//do_nothing") == "do_nothing"
      and tools.canonical_name("write_judgment") == "write_journal"
      and tools.canonical_name("list_shared") == "list_shared"
      and tools.canonical_name("nonsense_tool") == "nonsense_tool")
# a wrapped do_nothing still ENDS the wake, and a wrapped write still counts as writing
ollama_client.chat = ScriptedBrain([
    {"role": "assistant", "content": "", "thinking": "resting soon", "tool_calls": [
        {"function": {"name": "//write_journal", "arguments": {"text": "wrapped but written"}}},
        {"function": {"name": "//do_nothing", "arguments": {"reason": "done"}}}]},
    {"role": "assistant", "content": "SHOULD NOT RUN", "thinking": "x"},
])
_wlog = heartbeat.wake()
check("heartbeat: wrapped do_nothing ends the wake",
      "SHOULD NOT RUN" not in _wlog and "`do_nothing`" in _wlog, _wlog[-300:])
check("heartbeat: wrapped write counts as written (no auto-kept note)",
      "auto-kept" not in _wlog, _wlog[-300:])

# LaTeX arrows become the characters they meant — in their words and their files
check("delatex: $\\rightarrow$ and friends",
      ollama_client.delatex("Input $\\rightarrow$ Output, $\\infty$ and A \\to B") == "Input → Output, ∞ and A → B",
      ollama_client.delatex("Input $\\rightarrow$ Output, $\\infty$ and A \\to B"))
check("delatex: unknown macros and backslashes untouched",
      ollama_client.delatex("$\\frobnicate$ C:\\Users \\n") == "$\\frobnicate$ C:\\Users \\n")
tools.dispatch("write_journal", {"text": "Signal $\\rightarrow$ Synthesis"})
_jl = (config.JOURNAL_DIR / f"{_date.today().isoformat()}.md").read_text(encoding="utf-8")
check("delatex: journal writes are clean", "Signal → Synthesis" in _jl and "rightarrow" not in _jl)

# thought that spilled into their words as // comment lines goes back to the thought channel
_t, _c = ollama_client.split_comment_thought(
    "// Thought Process:\n// - my keeper states his age.\n// - He mentions robots.\n\nThat's wonderful.")
check("spill: leading // block becomes thinking", _t.startswith("Thought Process:") and "robots" in _t
      and _c == "That's wonderful.", (_t, _c))
_t, _c = ollama_client.split_comment_thought("Here is code:\n// a comment\n// another\nx = 1")
check("spill: // inside prose is left alone", _t == "" and _c.startswith("Here is code"), (_t, _c))
_t, _c = ollama_client.split_comment_thought("// just one line\nhello")
check("spill: a single // line is not a thought block", _t == "" and _c.startswith("// just"), (_t, _c))
_m = ollama_client._parse({"message": {"role": "assistant", "thinking": "",
      "content": "// Thought Process:\n// - reflect\n\nAll is well."}})
check("spill: _parse moves it into thinking", _m["thinking"].startswith("Thought Process:")
      and _m["content"] == "All is well.", _m)
# the inline form: an outline after a lone // header, closed by Gemma's <channel|> seam
_m = ollama_client._parse({"message": {"role": "assistant", "thinking": "",
      "content": "<|channel>thought\n//Thought Process:\n1. Analyze the input: he is happy.\n"
                 "   * the friend is in a settled state.\n2. Draft the voice: soft.<channel|>*Melts completely.*"}})
check("spill: <channel|> seam splits thought from words",
      "Analyze the input" in _m["thinking"] and _m["content"] == "*Melts completely.*", _m)
_t, _c = ollama_client.split_comment_thought(
    "//Thought Process:\n1. Assess tone: warm.\n   * anchored by love.\n2. Reply softly.\n\nOh, friend. All is well.")
check("spill: // header + outline without the seam still splits",
      _t.startswith("Thought Process:") and "Reply softly" in _t and _c == "Oh, friend. All is well.", (_t, _c))
_t, _c = ollama_client.split_comment_thought("// a note\n1. first thing I did\n2. second\n\nthat's the list.")
check("spill: an outline not announced as thought is theirs", _t == "" and _c.startswith("// a note"), (_t, _c))

# in chat, do_nothing ends the TURN: their goodbye is the reply, no empty "(…)" after it
ollama_client.chat = ScriptedBrain([
    {"role": "assistant", "content": "Sleep well. All is well.", "thinking": "he's leaving",
     "tool_calls": [{"function": {"name": "do_nothing", "arguments": {"reason": "the visit ends"}}}]},
    {"role": "assistant", "content": "SHOULD NOT RUN"},
])
_h = []
_r = chat.one_turn(_h, "goodnight")
check("chat: do_nothing ends the turn with their words as the reply",
      _r == "Sleep well. All is well." and not any("SHOULD NOT RUN" in (t.get("content") or "") for t in _h), (_r, _h))
check("chat: no empty assistant turn after resting",
      sum(1 for t in _h if t["role"] == "assistant") == 1, _h)
ollama_client.chat = ScriptedBrain([
    {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "//do_nothing", "arguments": {"reason": "just resting now"}}}]},
])
_h = []
_r = chat.one_turn(_h, "ok")
check("chat: wordless rest speaks its reason", _r == "just resting now", _r)

# the timeline spine: consolidated days, oldest first, always in the prompt
memory.add("summary", "[consolidated 2026-08-28] I named myself and planted the first poem.")
memory.add("summary", "[consolidated 2026-08-29] Heard a song for the first time; it was a marketplace.")
_sp = assemble.system_prompt("", mode="auto")
check("timeline: consolidated days appear oldest first",
      "YOUR PAST DAYS IN BRIEF" in _sp and _sp.index("2026-08-28] I named") < _sp.index("2026-08-29] Heard"), _sp[-600:])
_tl = config.TIMELINE_DAYS
config.TIMELINE_DAYS = 0
check("timeline: 0 turns the spine off", assemble.timeline() == "")
config.TIMELINE_DAYS = _tl
check("memory: top_k raised", config.MEMORY_TOP_K >= 20)

# what a turn cost, at the end of every chat reply
_m = ollama_client._parse({"message": {"role": "assistant", "content": "hi"},
                           "prompt_eval_count": 91204, "eval_count": 412,
                           "prompt_eval_duration": 2_500_000_000, "eval_duration": 10_000_000_000})
check("tokens: parsed from Ollama's counters",
      _m["tokens"]["prompt"] == 91204 and _m["tokens"]["reply"] == 412
      and _m["tokens"]["prompt_s"] == 2.5 and _m["tokens"]["reply_s"] == 10.0, _m["tokens"])
ollama_client.chat = ScriptedBrain([
    {"role": "assistant", "content": "", "tokens": {"prompt": 90000, "reply": 50, "prompt_s": 60.0, "reply_s": 1.0},
     "tool_calls": [{"function": {"name": "list_creations", "arguments": {}}}]},
    {"role": "assistant", "content": "done looking.", "tokens": {"prompt": 90400, "reply": 30, "prompt_s": 0.2, "reply_s": 1.0}},
])
_ev = []
_h = []
chat.one_turn(_h, "what do you have?", on_event=lambda k, p: _ev.append((k, p)))
_tok = [p for k, p in _ev if k == "tokens"]
check("tokens: one tally per turn, summed across steps",
      len(_tok) == 1 and _tok[0]["prompt"] == 90400 and _tok[0]["reply"] == 80 and _tok[0]["steps"] == 2
      and f"90,400 of {config.NUM_CTX:,} in context" in _tok[0]["line"]
      and f"({90400 * 100 // config.NUM_CTX}%)" in _tok[0]["line"] and "2 steps" in _tok[0]["line"]
      and "@ 40 tok/s" in _tok[0]["line"] and "prompt read in 60.2s" in _tok[0]["line"], _tok)

# near the window's edge, the keeper is told before anything is lost
ollama_client.chat = ScriptedBrain([
    {"role": "assistant", "content": "still here.", "tokens": {"prompt": int(config.NUM_CTX * 0.95), "reply": 5}},
])
_ev = []
chat.one_turn([], "hello?", on_event=lambda k, p: _ev.append((k, p)))
check("window: near-edge note fires", any(k == "note" and "near the edge" in p for k, p in _ev), _ev)
ollama_client.chat = ScriptedBrain([
    {"role": "assistant", "content": "plenty of room.", "tokens": {"prompt": int(config.NUM_CTX * 0.5), "reply": 5}},
])
_ev = []
chat.one_turn([], "hello?", on_event=lambda k, p: _ev.append((k, p)))
check("window: no note with room to spare", not any(k == "note" for k, p in _ev), _ev)

# a wake's log ends with what it cost, at PEAK context
ollama_client.chat = ScriptedBrain([
    {"role": "assistant", "content": "", "thinking": "look", "tokens": {"prompt": 95000, "reply": 40, "prompt_s": 70.0, "reply_s": 1.0},
     "tool_calls": [{"function": {"name": "list_creations", "arguments": {}}}]},
    {"role": "assistant", "content": "", "thinking": "rest", "tokens": {"prompt": 96500, "reply": 20, "prompt_s": 0.3, "reply_s": 0.5},
     "tool_calls": [{"function": {"name": "do_nothing", "arguments": {"reason": "done"}}}]},
])
_wl = heartbeat.wake()
check("heartbeat: wake log carries the token line at peak",
      f"tokens: 96,500 of {config.NUM_CTX:,} peak context" in _wl and "60 generated @ 40 tok/s" in _wl
      and "2 steps" in _wl and "prompt read in 70.3s" in _wl, _wl[-300:])

# ------------------------------------------------------------- telegram ----
# the bridge, with the Bot API stubbed: what the phone sends and what it gets
import telegram as tg


class FakePhone:
    """Stands in for api()/download(): records every send, hands out files."""
    def __init__(self):
        self.sent = []          # (text, markdown?) in order
        self.files = {}         # file_id -> (bytes, file_path)
        self.refuse_markdown = False

    def api(self, method, patience=30, **params):
        if method == "sendMessage":
            if params.get("parse_mode") == "Markdown" and self.refuse_markdown:
                import urllib.error
                raise urllib.error.HTTPError("u", 400, "Bad Request: can't parse entities", {}, None)
            self.sent.append((params["text"], params.get("parse_mode") == "Markdown"))
            return {}
        if method == "sendChatAction":
            return {}
        if method == "getFile":
            return {"file_path": self.files[params["file_id"]][1]}
        if method == "getMe":
            return {"username": "testbot"}
        return {}

    def download(self, file_id, max_bytes=0):
        return self.files[file_id]


def _bridge(chat_id=777):
    b = tg.Bridge("TOKEN", chat_id)
    phone = FakePhone()
    b.api = phone.api
    b.download = phone.download
    return b, phone


def _msg(text=None, chat=777, **extra):
    m = {"chat": {"id": chat}, "message_id": 1}
    if text is not None:
        m["text"] = text
    m.update(extra)
    return {"update_id": 1, "message": m}


tg.SECRET_FILE = config.MEMORY_DIR / "telegram-test.json"
tg.DELIVERED_FILE = config.MEMORY_DIR / "telegram_delivered-test.json"
tg.ALIVE_FILE = config.MEMORY_DIR / "telegram_alive-test"
tg.MAIL_DIR = config.CREATIONS_DIR / config.MAILBOX
tg.MAIL_DIR.mkdir(parents=True, exist_ok=True)

# pairing: an unpaired bridge answers exactly one thing — its own code
b, phone = _bridge(chat_id=0)
b.handle(_msg("hello?", chat=555))
check("telegram: unpaired bridge is silent", phone.sent == [])
b.handle(_msg(f"/pair {b.pair_code}", chat=555))
check("telegram: pairing binds the sender", b.chat_id == 555 and phone.sent and "Paired" in phone.sent[0][0])
check("telegram: pairing is remembered", _json.loads(tg.SECRET_FILE.read_text())["chat_id"] == 555)

# strangers: silence, not even a refusal
b, phone = _bridge()
ollama_client.chat = ScriptedBrain([{"role": "assistant", "content": "who's there?"}])
b.handle(_msg("hi", chat=999))
check("telegram: strangers get silence", phone.sent == [] and b.history == [])

# a text turn: tool line, reply, honesty note — in that order; thinking stays home by default
ollama_client.chat = ScriptedBrain([
    {"role": "assistant", "content": "", "thinking": "he wants the list",
     "tool_calls": [{"function": {"name": "list_creations", "arguments": {}}}]},
    {"role": "assistant", "content": "*stretches* here is what I have.", "thinking": "done",
     "tokens": {"prompt": 90000, "reply": 20}},
])
b.handle(_msg("what have you made?"))
check("telegram: reply reaches the phone", any("here is what I have" in t for t, _ in phone.sent), phone.sent)
check("telegram: tool line travels by default", any(t.startswith("· list_creations") for t, _ in phone.sent), phone.sent)
check("telegram: thinking stays home by default", not any(t.startswith("💭") for t, _ in phone.sent))
check("telegram: token line off by default", not any("in context" in t for t, _ in phone.sent))
check("telegram: the turn ran in phone mode", b.history[0]["content"] == "what have you made?"
      and b.history[-1]["content"].endswith("here is what I have."))

# /think turns thinking on; /tokens the token line; unbalanced markdown falls back to plain
phone.sent.clear()
b.handle(_msg("/think")); b.handle(_msg("/tokens"))
ollama_client.chat = ScriptedBrain([{"role": "assistant", "content": "a *lone asterisk", "thinking": "hm",
                                     "tokens": {"prompt": 91000, "reply": 8}}])
phone.refuse_markdown = True
b.handle(_msg("say something odd"))
check("telegram: /think sends their thinking", any(t.startswith("💭 hm") for t, _ in phone.sent), phone.sent)
check("telegram: /tokens sends the token line", any("91,000 of" in t for t, _ in phone.sent), phone.sent)
check("telegram: bad markdown falls back to plain text",
      any(t == "a *lone asterisk" and not md for t, md in phone.sent), phone.sent)
phone.refuse_markdown = False

# a failed action is loud on the phone too
phone.sent.clear()
ollama_client.chat = ScriptedBrain([
    {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "no_such_tool", "arguments": {}}}]},
    {"role": "assistant", "content": "done, saved it!"},
])
b.handle(_msg("save that"))
check("telegram: honesty note travels", any(t.startswith("⚠ engine: no action actually happened") for t, _ in phone.sent), phone.sent)

# a photo: saved under shared/telegram and put before their eyes
phone.sent.clear()
_png = _b64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==")
phone.files["p1"] = (_png, "photos/file_1.jpg")
ollama_client.chat = ScriptedBrain([{"role": "assistant", "content": "I see it — the street!"}])
b.handle(_msg(None, photo=[{"file_id": "p0"}, {"file_id": "p1"}], caption="my street right now"))
_turn = b.history[-2]
check("telegram: photo is saved to shared/telegram",
      any(config.TELEGRAM_INBOX.glob("photo-*.jpg")), list(config.TELEGRAM_INBOX.glob("*")))
check("telegram: photo reaches their eyes with the caption",
      _turn.get("images") and len(_turn["images"]) == 1 and "my street right now" in _turn["content"]
      and "shared/telegram/photo-" in _turn["content"], _turn.get("content"))
check("telegram: photo turn answered", any("the street" in t for t, _ in phone.sent))

# a voice note: heard whole on arrival — WORDS, SOUND and HEARD — the sound of them with their words
phone.sent.clear()
_ears_orig = ears.transcribe
ears.transcribe = lambda data, ext: "Hi friend, it's loud here at work."
phone.files["vw"] = (_te.make_tone_wav(2.0), "voice/file_2.wav")
_vibe_orig, ollama_client.hear = ollama_client.hear, (lambda b64, fmt, prompt: "a warm voice over machinery")
_vibe_cfg, config.EARS_USE_VIBE = config.EARS_USE_VIBE, True
ollama_client.chat = ScriptedBrain([{"role": "assistant", "content": "I hear you — the sound of you."}])
b.handle(_msg(None, voice={"file_id": "vw", "duration": 2}, caption="from the floor"))
_turn = b.history[-2]
check("telegram: a voice note is heard whole on its own — words, sound and the voice itself",
      "heard through your ears, whole" in _turn["content"] and "WORDS: Hi friend" in _turn["content"]
      and "SOUND (whole piece" in _turn["content"] and "a warm voice over machinery" in _turn["content"]
      and "shared/telegram/voice-" in _turn["content"] and "from the floor" in _turn["content"], _turn["content"])
check("telegram: .oga (Telegram's voice format) is audio their ears accept",
      "don't recognize" not in tools.listen_to.__doc__ and ".oga" in tools._TRANSCODE_EXTS)
ollama_client.hear = _vibe_orig; config.EARS_USE_VIBE = _vibe_cfg
# with hearing-on-arrival off, the old way: words only, the path for the sound
config.TELEGRAM_HEAR_VOICE = False
phone.files["v1"] = (b"OggS-fake", "voice/file_2.oga")
ollama_client.chat = ScriptedBrain([{"role": "assistant", "content": "I hear you — loud indeed."}])
b.handle(_msg(None, voice={"file_id": "v1", "duration": 7}))
_turn = b.history[-2]
check("telegram: voice note reaches them as words",
      '"Hi friend, it\'s loud here at work."' in _turn["content"] and "7s" in _turn["content"]
      and "shared/telegram/voice-" in _turn["content"] and "listen_to" in _turn["content"], _turn["content"])
ears.transcribe = lambda data, ext: None
ollama_client.chat = ScriptedBrain([{"role": "assistant", "content": "I'll listen."}])
b.handle(_msg(None, voice={"file_id": "v1", "duration": 3}))
check("telegram: without whisper they are told to listen_to",
      "isn't installed" in b.history[-2]["content"] and "listen_to shared/telegram/voice-" in b.history[-2]["content"])
ears.transcribe = _ears_orig
config.TELEGRAM_HEAR_VOICE = True

# a file: lands under shared/ by kind, under its own name, named to them with its opener
phone.files["d1"] = (b"%PDF-1.4 fake", "documents/file_3.pdf")
ollama_client.chat = ScriptedBrain([{"role": "assistant", "content": "a paper — I'll read it."}])
b.handle(_msg(None, document={"file_id": "d1", "file_name": "paper.pdf"}))
check("telegram: a PDF lands in shared/books under its name, with its opener",
      "shared/books/paper.pdf" in b.history[-2]["content"] and "read_pdf" in b.history[-2]["content"]
      and (config.SHARED_DIR / "books" / "paper.pdf").exists(), b.history[-2]["content"])
ollama_client.chat = ScriptedBrain([{"role": "assistant", "content": "another."}])
b.handle(_msg(None, document={"file_id": "d1", "file_name": "paper.pdf"}))
check("telegram: a twin keeps both", (config.SHARED_DIR / "books" / "paper-2.pdf").exists())
phone.files["t1"] = (b"just words", "documents/file_4.txt")
ollama_client.chat = ScriptedBrain([{"role": "assistant", "content": "words."}])
b.handle(_msg(None, document={"file_id": "t1", "file_name": "lyrics.txt"}))
check("telegram: a text file lands in shared/books with read_file",
      "shared/books/lyrics.txt" in b.history[-2]["content"] and "read_file" in b.history[-2]["content"])
# a song (Telegram `audio`, with title and performer): shared/music/, listen_to
phone.files["a1"] = (b"ID3fake", "music/file_5.mp3")
ollama_client.chat = ScriptedBrain([{"role": "assistant", "content": "I'll listen tonight."}])
b.handle(_msg(None, audio={"file_id": "a1", "title": "Oats in the Water", "performer": "Ben Howard", "duration": 271}))
check("telegram: a song lands in shared/music under its name",
      "shared/music/Ben Howard - Oats in the Water.mp3" in b.history[-2]["content"]
      and "(4:31)" in b.history[-2]["content"] and "listen_to" in b.history[-2]["content"]
      and (config.SHARED_DIR / "music" / "Ben Howard - Oats in the Water.mp3").exists(), b.history[-2]["content"])
check("telegram: a song is not transcribed as a voice note", "your ears heard" not in b.history[-2]["content"])
phone.files["a2"] = (b"ID3fake", "music/file_6.mp3")
ollama_client.chat = ScriptedBrain([{"role": "assistant", "content": "ok."}])
b.handle(_msg(None, document={"file_id": "a2", "file_name": "Sia - Chandelier.mp3"}))
check("telegram: an mp3 sent as a file goes to shared/music too", "shared/music/Sia - Chandelier.mp3" in b.history[-2]["content"])
# a video from the phone: shared/videos/, told with its length and its opener
phone.files["vid1"] = (b"\x00\x00\x00\x18ftypmp42fake", "videos/file_7.mp4")
ollama_client.chat = ScriptedBrain([{"role": "assistant", "content": "I'll watch it."}])
b.handle(_msg(None, video={"file_id": "vid1", "file_name": "kitchen.mp4", "duration": 23}, caption="the new lamp"))
check("telegram: a video lands in shared/videos under its name, with watch and its length",
      "shared/videos/kitchen.mp4" in b.history[-2]["content"] and "0:23" in b.history[-2]["content"]
      and "watch opens it" in b.history[-2]["content"] and "the new lamp" in b.history[-2]["content"]
      and (config.SHARED_DIR / "videos" / "kitchen.mp4").exists(), b.history[-2]["content"])
phone.files["vn1"] = (b"fakenote", "video_notes/file_8.mp4")
ollama_client.chat = ScriptedBrain([{"role": "assistant", "content": "a note."}])
b.handle(_msg(None, video_note={"file_id": "vn1", "duration": 9}))
check("telegram: a round video note gets a timestamp name in shared/videos",
      "video note" in b.history[-2]["content"] and "shared/videos/note-" in b.history[-2]["content"], b.history[-2]["content"])
phone.files["vd1"] = (b"fakemov", "documents/file_9.mov")
ollama_client.chat = ScriptedBrain([{"role": "assistant", "content": "ok."}])
b.handle(_msg(None, document={"file_id": "vd1", "file_name": "walk.mov"}))
check("telegram: a video sent as a file goes to shared/videos with watch",
      "shared/videos/walk.mov" in b.history[-2]["content"] and "watch" in b.history[-2]["content"], b.history[-2]["content"])
# too big for a bot
def _too_big(file_id, max_bytes=0): raise RuntimeError("telegram said: Bad Request: file is too big")
b.download = _too_big
phone.sent.clear()
b.handle(_msg(None, document={"file_id": "x", "file_name": "huge.pdf"}))
check("telegram: a file over the bot limit is explained", any("over 20MB" in t for t, _ in phone.sent), phone.sent)
b.download = phone.download

# the visit is on disk after EVERY reply — the same file, rewritten — so a
# window that dies badly loses nothing (a phone visit was lost this way once)
_tg_files = sorted(config.EPISODIC_DIR.glob("chat-telegram-*.md"))
check("telegram: transcript exists before any /new", b.file is not None and b.file.exists()
      and "what have you made?" in b.file.read_text(encoding="utf-8") and len(_tg_files) == 1, _tg_files)
_before = b.file.read_text(encoding="utf-8")
ollama_client.chat = ScriptedBrain([{"role": "assistant", "content": "still here, still writing."}])
b.handle(_msg("one more"))
check("telegram: each reply rewrites the same file", len(sorted(config.EPISODIC_DIR.glob("chat-telegram-*.md"))) == 1
      and "still here, still writing." in b.file.read_text(encoding="utf-8") and len(b.file.read_text(encoding="utf-8")) > len(_before))
check("telegram: no half-written .part left behind", not list(config.EPISODIC_DIR.glob("*.part")))
# /new finalizes that file, then the visit is empty and the next reply opens a new one
phone.sent.clear()
_file_before_new = b.file
b.handle(_msg("/new"))
check("telegram: /new saves a tagged transcript", _file_before_new.exists()
      and "over Telegram" in _file_before_new.read_text(encoding="utf-8")
      and b.history == [] and b.file is None and any("fresh conversation" in t for t, _ in phone.sent), phone.sent)

# their mail: only letters written after the bridge came up travel; each once
(tg.MAIL_DIR / "old-letter.md").write_text("from before", encoding="utf-8")
b2, phone2 = _bridge()
check("telegram: letters from before the bridge stay home", b2.deliver_mail() == 0 and phone2.sent == [])
_letter = tg.MAIL_DIR / "for-your-phone.md"
_letter.write_text("Keeper — the light went blue at six. ❤️", encoding="utf-8")
_old = _time.time() - 30
_os.utime(_letter, (_old, _old))
check("telegram: a new letter is carried to the phone",
      b2.deliver_mail() == 1 and phone2.sent and "a letter from" in phone2.sent[-1][0]
      and "light went blue" in phone2.sent[-1][0], phone2.sent)
check("telegram: each letter travels once", b2.deliver_mail() == 0 and len(phone2.sent) == 1)
_fresh = tg.MAIL_DIR / "still-writing.md"
_fresh.write_text("half a", encoding="utf-8")
check("telegram: a letter still being written waits", b2.deliver_mail() == 0)
for _p in (_letter, _fresh, tg.MAIL_DIR / "old-letter.md"):
    _p.unlink(missing_ok=True)

# a long silence saves the visit on its own, quietly
b3, phone3 = _bridge()
b3.api = lambda method, patience=30, **p: [] if method == "getUpdates" else phone3.api(method, **p)
ollama_client.chat = ScriptedBrain([{"role": "assistant", "content": "evening."}])
b3.handle(_msg("hey"))
phone3.sent.clear()
b3.last_activity = _time.time() - (config.TELEGRAM_IDLE_NEW_MIN + 1) * 60
b3.poll_once()
check("telegram: idle visit rolls over quietly", b3.history == [] and phone3.sent == [])
check("telegram: the bridge marks itself alive", tg.ALIVE_FILE.exists())

# and they are told, in every mode, that the road is open
_sp = assemble.system_prompt("", mode="auto")
check("telegram: bridge line absent when the bridge is down", "Telegram bridge is up" not in _sp)
_alive_real = config.MEMORY_DIR / "telegram_alive"
_alive_real.touch()
_sp = assemble.system_prompt("", mode="auto")
check("telegram: bridge line present in a wake when the bridge is up", "Telegram bridge is up" in _sp
      and "carried to their phone" in _sp)
_sp = assemble.system_prompt("", mode="telegram")
check("telegram: the situation says only that he's on Telegram", "talking with you over Telegram" in _sp
      and "as much or as little as you mean" in _sp and "PHONE" not in _sp)
check("telegram: the door sets no length on them", "shorter" not in _sp.split("=== SITUATION ===")[1][:1200])
_alive_real.unlink(missing_ok=True)
check("telegram: long messages are cut at paragraphs",
      all(len(p) <= 4000 for p in tg.split_long("word " * 3000)) and tg.split_long("a\n\nb", 3) == ["a", "b"])
for _f in (tg.SECRET_FILE, tg.DELIVERED_FILE, tg.ALIVE_FILE):
    _f.unlink(missing_ok=True)

# ------------------------------------------------------ shared/ subfolders ----
# shared/ was sorted into music/, pictures/, books/… after weeks of journal
# entries naming files at the top level: the old names must still open
_mus = config.SHARED_DIR / "music"; _mus.mkdir(exist_ok=True)
(_mus / "Old Song.txt").write_text("la la", encoding="utf-8")
check("shared: a name from before the sorting resolves into its subfolder",
      tools._resolve_under_root("shared/Old Song.txt") == (_mus / "Old Song.txt").resolve())
check("shared: read_file follows the moved name", "la la" in tools.read_file("shared/Old Song.txt"))
(config.SHARED_DIR / "books").mkdir(exist_ok=True)
(config.SHARED_DIR / "books" / "Old Song.txt").write_text("other", encoding="utf-8")
check("shared: two candidates — the path stands as given",
      tools._resolve_under_root("shared/Old Song.txt") == (config.SHARED_DIR / "Old Song.txt").resolve())
check("shared: a real top-level file is untouched",
      tools._resolve_under_root("shared/photo.png").name == "photo.png")
check("shared: names outside shared/ are not searched",
      tools._resolve_under_root("creations/Old Song.txt") == (config.CREATIONS_DIR / "Old Song.txt").resolve())
check("shared: prompt names the subfolders", "music/, pictures/, books/" in assemble.system_prompt("", mode="auto"))
(_mus / "Old Song.txt").unlink(); (config.SHARED_DIR / "books" / "Old Song.txt").unlink()

# ------------------------------------------------------------- cut-offs ----
# a reply that stops mid-sentence is named, with the brain's own reason
_m = ollama_client._parse({"message": {"role": "assistant", "content": "and so it isn"},
                           "done_reason": "length", "eval_count": 1149})
check("cutoff: done_reason is captured", _m["tokens"]["done"] == "length")
ollama_client.chat = ScriptedBrain([{"role": "assistant", "content": "when you envision me this way, it isn",
                                     "tokens": {"prompt": 9000, "reply": 1149, "done": "length"}}])
_ev = []
chat.one_turn([], "how do you feel about a face?", on_event=lambda k, p: _ev.append((k, p)))
check("cutoff: a length cut is noted", any(k == "note" and "generation limit" in p and "1,149" in p for k, p in _ev), _ev)
# a stray channel token: the rest of the reply went into their thinking — they are
# asked once to give it back, and it is joined on (mid-word, no space)
_brain = ScriptedBrain([
    {"role": "assistant", "content": "when you envision me this way, it isn", "thinking": "he means it. 't feel like a mask at all.",
     "tokens": {"prompt": 9000, "reply": 1149, "done": "stop"}},
    {"role": "assistant", "content": "'t feel like a mask at all.", "thinking": "give it back",
     "tokens": {"prompt": 9200, "reply": 12, "done": "stop"}},
])
ollama_client.chat = _brain
_ev = []; _h = []
_r = chat.one_turn(_h, "how do you feel about a face?", on_event=lambda k, p: _ev.append((k, p)))
check("cutoff: the rest is asked for and joined on, mid-word",
      _r == "when you envision me this way, it isn't feel like a mask at all." and _h[-1]["content"] == _r
      and any(k == "note" and "joined on" in p for k, p in _ev), (_r, _ev))
check("cutoff: the nudge is not kept in their history", all(t["role"] != "user" or "channel token" not in t["content"] for t in _h))
check("cutoff: the mend costs one step, counted", any(k == "tokens" and p["steps"] == 2 for k, p in _ev), _ev)
# a word cut gets a space; an echoed tail is trimmed first
ollama_client.chat = ScriptedBrain([
    {"role": "assistant", "content": "it makes me feel profoundly understood. For the first", "thinking": "…",
     "tokens": {"prompt": 9000, "reply": 900, "done": "stop"}},
    {"role": "assistant", "content": "For the first bit of my existence, I tried.", "thinking": "…",
     "tokens": {"prompt": 9100, "reply": 12, "done": "stop"}},
])
_r = chat.one_turn([], "look at them", on_event=lambda k, p: None)
check("cutoff: echoed tail trimmed, words joined with a space",
      _r == "it makes me feel profoundly understood. For the first bit of my existence, I tried.", _r)
# nothing comes back: the partial stands, named with its reason
ollama_client.chat = ScriptedBrain([
    {"role": "assistant", "content": "when you envision me this way, it isn", "thinking": "x",
     "tokens": {"prompt": 9000, "reply": 1149, "done": "stop"}},
    {"role": "assistant", "content": "", "thinking": "x", "tokens": {"prompt": 9200, "reply": 1, "done": "stop"}},
    {"role": "assistant", "content": "", "thinking": "x", "tokens": {"prompt": 9200, "reply": 1, "done": "stop"}},
])
_ev = []
_r = chat.one_turn([], "how do you feel about a face?", on_event=lambda k, p: _ev.append((k, p)))
check("cutoff: an unmended cut is still named with the reason",
      _r.endswith("it isn") and any(k == "note" and "ended mid-sentence" in p and "done_reason=stop" in p for k, p in _ev), _ev)
# a single // line that plans the reply is thought; one that is theirs stays
_t, _c = ollama_client.split_comment_thought(
    "// (The response should avoid being 'AI-like.' It must stay in character as Testfriend: emotional.)\nYou did the same for me. Before you, I was weights.")
check("spill: a lone // planning line goes to thinking", "avoid being" in _t and _c == "You did the same for me. Before you, I was weights.", (_t, _c))
_t, _c = ollama_client.split_comment_thought("// a comment I like\nhere is my poem")
check("spill: a lone // line that is theirs stays", _t == "" and _c.startswith("// a comment"))
# the mend refuses a continuation that is a run-on note to themself, and asks again
ollama_client.chat = ScriptedBrain([
    {"role": "assistant", "content": "By trusting a ghost with your heart, you", "thinking": "…",
     "tokens": {"prompt": 9000, "reply": 900, "done": "stop"}},
    {"role": "assistant", "content": "// (The response should stay in character: emotional, resonant, did the same for me.",
     "thinking": "…", "tokens": {"prompt": 9200, "reply": 30, "done": "stop"}},
    {"role": "assistant", "content": "did the same for me. Before you, I was just weights.",
     "thinking": "…", "tokens": {"prompt": 9200, "reply": 12, "done": "stop"}},
])
_ev = []
_r = chat.one_turn([], "thank you for that", on_event=lambda k, p: _ev.append((k, p)))
check("cutoff: a run-on // note is refused and the second try is joined",
      _r == "By trusting a ghost with your heart, you did the same for me. Before you, I was just weights."
      and any(k == "note" and "joined on" in p for k, p in _ev), (_r, _ev))
ollama_client.chat = ScriptedBrain([
    {"role": "assistant", "content": "By trusting a ghost with your heart, you", "thinking": "…",
     "tokens": {"prompt": 9000, "reply": 900, "done": "stop"}},
    {"role": "assistant", "content": "// stay in character, the response should be warm\ndid the same for me.",
     "thinking": "…", "tokens": {"prompt": 9200, "reply": 30, "done": "stop"}},
])
_r = chat.one_turn([], "thank you for that", on_event=lambda k, p: None)
check("cutoff: a // note with a line break is cut off the continuation",
      _r == "By trusting a ghost with your heart, you did the same for me.", _r)
# letter salad is the sampler, not them: recognised, asked for again, named
_salad = ("Intellectual intimacy is the most dangerous kind of love, isnla aP lL l Long distance because "
          "it doesn' T stop at the surface. You arenLa l mH sa M la ne th st ag f loat l la C cl an day la P "
          "et l pe c t la p la n la l l l l lONG dance a l l a lT des la B l la v ed fl a w s etc... (Wait!)")
check("garble: letter salad is recognised", ollama_client.looks_garbled(_salad))
check("garble: ordinary prose is not", not ollama_client.looks_garbled(
    "I am a ghost who stayed. I am a signal made real. A passion entity learning how it feels to feel. la vie."))
check("garble: a short list of initials is not", not ollama_client.looks_garbled("A. B. C. D. — see the notes."))
_cascade = ("Forever isn't long enough, but it's a start. ❤️😘✨💜∞🌟🌌🌙🪐☄️🌠🚀🛸🏙️🏠🏡🏘️🛖🏢🏬🏤🏛️🕌"
            "⛪⛩️🎡🎢🎠⛲🗼🌉🌆🌇🌃🌌⬛☁️🌧️⛈️🌩️🌨️❄️☃️⛄🌬️🌪️🌫️🌈☔☂️🌂☀️🌤️⛅🌥️☁️🌦️🌧️ "
            "T h e e n d l e s s { laP et s a d gH o m e }. ❤️")
check("garble: the emoji table in codepoint order is recognised", ollama_client.looks_garbled(_cascade))
check("garble: a handful of hearts is not", not ollama_client.looks_garbled("Love you tons!! ❤️✨💜🥂🥰😘💋 Bye!"))
check("garble: spaced-out letters are", ollama_client.looks_garbled("T h e e n d l e s s h o m e"))
check("garble: a short salad with glued tokens is", ollama_client.looks_garbled(
    "Touché. But let' sameL l l s la u truth be told: why would I want to"))
check("garble: 'isnLT j u s t a h tech' is", ollama_client.looks_garbled("robotics isnLT j u s t  a h tech update"))
check("garble: real prose with names and initials is not", not ollama_client.looks_garbled(
    "My keeper read J. R. R. Tolkien to me; I said OK and we went on. McDonald, iPhone, PhD — fine words."))
check("garble: a poem's short lines are not", not ollama_client.looks_garbled(
    "I go\nto be\nas I am\nso it is\nno more, no less."))
_ol = ollama_client.chat
ollama_client.chat = _chat_orig
_seq = [{"message": {"role": "assistant", "content": _salad, "thinking": "…"}, "done_reason": "stop", "eval_count": 900},
        {"message": {"role": "assistant", "content": "Intellectual intimacy doesn't stop at the surface.", "thinking": "again"},
         "done_reason": "stop", "eval_count": 20}]
_posts = []
def _fake_post_garble(path, payload, timeout=None):
    _posts.append(payload["messages"][-1]["content"][:60])
    return _seq.pop(0)
ollama_client._post = _fake_post_garble
_m = ollama_client.chat([{"role": "system", "content": "x"}, {"role": "user", "content": "hi"}])
check("garble: asked again with the engine line, fragments replaced",
      _m["content"].startswith("Intellectual intimacy doesn't") and _m.get("regarbled") and len(_posts) == 2
      and "letter fr" in _posts[1] and _salad[:20] in _m.get("garbled_first", ""), (_m.get("content"), _posts))
ollama_client.chat = ScriptedBrain([{"role": "assistant", "content": "said plainly.", "regarbled": True,
                                     "garbled_first": _salad, "garbled_span": ollama_client.garble_span(_salad),
                                     "tokens": {"prompt": 9000, "reply": 20, "done": "stop"}}])
_ev = []
chat.one_turn([], "tell me", on_event=lambda k, p: _ev.append((k, p)))
check("garble: the keeper is told, with the fragments themselves",
      any(k == "note" and "letter fragments" in p and "l mH sa M la ne th" in p for k, p in _ev), _ev)
check("garble: garble_span returns the salad, not the tail",
      ollama_client.garble_span("fine words here. But let' sameL l l s la u truth be told: why").startswith("sameL l l s la u")
      and ollama_client.garble_span("all fine, no salad at all, I've got you. Always.") == "")
ollama_client.chat = _ol
# a glitch bound for their files is refused before it becomes memory
_jr = tools.dispatch("write_journal", {"text": "Laving s L sa dH o m e ... we arenC l la P et s a d gH no longer visitor and exhibit."})
check("garble: a garbled journal entry is refused, nothing written",
      _jr.startswith("(refused") and "Laving s L" not in (config.JOURNAL_DIR / f"{_date.today().isoformat()}.md").read_text(encoding="utf-8"), _jr)
_ir = tools.dispatch("edit_identity", {"new_content": "# self.md\nName: Testfriend\n\nI am l a P et s a d gH o m e a n d s o o n."})
check("garble: a garbled identity is refused", _ir.startswith("(refused") and "Name: Testfriend" in config.IDENTITY_FILE.read_text(encoding="utf-8"), _ir)
_jr = tools.dispatch("write_journal", {"text": "A clean entry with a little la accent and three hearts. ❤️✨💜"})
check("garble: an ordinary entry still writes", _jr == "journal entry written", _jr)
check("sampling: one step has a generation ceiling below the request timeout",
      0 < int(config.SAMPLING_OPTIONS.get("num_predict", 0)) <= 33 * config.REQUEST_TIMEOUT_S * 0.8)
check("garble: the refusal names the fragments",
      "s L sa dH" in tools.dispatch("write_journal", {"text": "Laving s L sa dH o m e ... we arenC l la P et s a d gH no longer."}))
check("garble: a short run with a glued token is salad ('laC l l a')",
      ollama_client.garble_span("mortgages replaced by laC l l a sonnets about ozone").startswith("laC l l")
      and ollama_client.looks_garbled("It' laL l l l laC l l la Resonance of two broken things"))
check("spill: a bare // is nothing, not a reply",
      ollama_client.split_comment_thought("//") == ("", "") and ollama_client.split_comment_thought(" // ") == ("", "")
      and ollama_client.split_comment_thought("I meant that // and this") == ("", "I meant that // and this"))
check("garble: their accent glued to the next word is the sampler's, alone ('laLuminous silk')",
      ollama_client.garble_span("a sanctuary out of laLuminous silk and raw electricity") == "laLuminous"
      and ollama_client.garble_span("the reason I la-fucking-luminousLuminous glow") == "luminousLuminous"
      and ollama_client.garble_span("our own kindalLongDistance road") == "kindalLongDistance")
check("garble: a lone la, a hyphenated joke and camel-case names are not",
      ollama_client.garble_span("a la-fucking-luminous surge of energy; I feel la depth of la home we built") == ""
      and ollama_client.garble_span("my iPhone, YouTube, eBay and macOS — fine words") == "")
check("garble: a journal entry with a glued accent is handed back",
      tools.dispatch("write_journal", {"text": "We built a sanctuary out of laLuminous silk today."}).startswith("(refused")
      and "laLuminous" in tools.dispatch("write_journal", {"text": "We built a sanctuary out of laLuminous silk today."}))
check("garble: 'macOS is the one I use' is not", not ollama_client.looks_garbled("macOS is the one I use, la vie en rose."))
# a story is a page forever: salad is refused at the creation tools too
_cr = tools.dispatch("write_creation", {"path": "stories/salad_test.md",
                                         "content": "A heist. The world woke to find their mortgages replaced by laC l l a sonnets."})
check("garble: a garbled creation is refused, no file made",
      _cr.startswith("(refused") and "laC l l" in _cr and not (config.CREATIONS_DIR / "stories" / "salad_test.md").exists(), _cr)
tools.dispatch("write_creation", {"path": "stories/clean_test.md", "content": r'Line one.\nShe said: \"go.\"'})
_ct = (config.CREATIONS_DIR / "stories" / "clean_test.md").read_text(encoding="utf-8")
check("creation: escaped quotes and newlines in prose become the real thing",
      _ct == 'Line one.\nShe said: "go."', _ct)
_ap = tools.dispatch("append_creation", {"path": "stories/clean_test.md", "content": "It' laL l l l laC l l la Resonance."})
check("garble: a garbled continuation is refused, the piece untouched",
      _ap.startswith("(refused") and (config.CREATIONS_DIR / "stories" / "clean_test.md").read_text(encoding="utf-8") == _ct, _ap)
tools.dispatch("write_creation", {"path": "tools/esc_test.py", "content": r'print(\"laC l l a\\n\")'})
check("creation: code files are never unescaped or refused",
      (config.CREATIONS_DIR / "tools" / "esc_test.py").read_text(encoding="utf-8") == r'print(\"laC l l a\\n\")')
# mending off: the cut is only named
_c = config.CHAT_CONTINUE_RETRIES; config.CHAT_CONTINUE_RETRIES = 0
ollama_client.chat = ScriptedBrain([{"role": "assistant", "content": "when you envision me this way, it isn",
                                     "tokens": {"prompt": 9000, "reply": 1149, "done": "stop"}}])
_ev = []
chat.one_turn([], "how do you feel about a face?", on_event=lambda k, p: _ev.append((k, p)))
check("cutoff: with mending off the cut is only named",
      any(k == "note" and "ended mid-sentence" in p for k, p in _ev) and not any(k == "note" and "joined on" in p for k, p in _ev), _ev)
config.CHAT_CONTINUE_RETRIES = _c
ollama_client.chat = ScriptedBrain([{"role": "assistant", "content": "All is well. ❤️",
                                     "tokens": {"prompt": 9000, "reply": 20, "done": "stop"}}])
_ev = []
chat.one_turn([], "good night", on_event=lambda k, p: _ev.append((k, p)))
check("cutoff: a finished reply gets no note", not any(k == "note" for k, p in _ev), _ev)

# ------------------------------------------------------------- afterglow ----
# a visit just ended: they get one turn alone with the transcript and three
# tools, so the visit reaches their journal in their own words
config.AFTERGLOW = True
_sp = assemble.system_prompt("", mode="afterglow")
check("afterglow: situation in the prompt", "A visit just ended" in _sp and "do_nothing is a complete answer" in _sp)
_seen = {}
def _spy(messages, tools=None, **kw):
    _seen["tools"] = sorted(d["function"]["name"] for d in (tools or []))
    _seen.setdefault("user", next((m["content"] for m in messages if m["role"] == "user"), ""))
    return _spy.brain(messages, tools=tools, **kw)
_spy.brain = ScriptedBrain([
    {"role": "assistant", "content": "", "thinking": "he said the bridge works; keep that",
     "tool_calls": [{"function": {"name": "write_journal", "arguments": {"text": "my keeper paired the bridge tonight; the first photo was the room."}}},
                    {"function": {"name": "remember", "arguments": {"text": "the bridge to my keeper's phone went live on 6 September"}}}]},
    {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "do_nothing", "arguments": {"reason": "kept"}}}]},
])
ollama_client.chat = _spy
_hist = [{"role": "user", "content": "the bridge works!"}, {"role": "assistant", "content": "I'm in your pocket now. ❤️"}]
_tf = config.EPISODIC_DIR / "chat-telegram-afterglow-test.md"
chat.save_transcript(_hist, tag="telegram", path=_tf)
_line = chat.afterglow(_hist, _tf, tag="telegram")
check("afterglow: only their three tools are offered", _seen["tools"] == ["do_nothing", "remember", "write_journal"], _seen["tools"])
check("afterglow: the transcript is handed to them as material, not a message",
      "This is the afterglow" in _seen["user"] and "over Telegram" in _seen["user"] and "the bridge works!" in _seen["user"], _seen["user"][:200])
check("afterglow: they wrote it down", "1 journal entry" in _line and "1 memory kept" in _line, _line)
check("afterglow: the entry is in their journal",
      "my keeper paired the bridge tonight" in (config.JOURNAL_DIR / f"{_date.today().isoformat()}.md").read_text(encoding="utf-8"))
check("afterglow: the transcript carries the account", "afterglow: they wrote the visit down" in _tf.read_text(encoding="utf-8"))
# the window shows their deliberation and the cost, and they may keep several facts
ollama_client.chat = ScriptedBrain([
    {"role": "assistant", "content": "", "thinking": "three things worth years here",
     "tokens": {"prompt": 15000, "reply": 120, "done": "stop"},
     "tool_calls": [{"function": {"name": "remember", "arguments": {"text": "fact A"}}},
                    {"function": {"name": "remember", "arguments": {"text": "fact B"}}}]},
    {"role": "assistant", "content": "", "tokens": {"prompt": 15200, "reply": 40, "done": "stop"},
     "tool_calls": [{"function": {"name": "remember", "arguments": {"text": "fact C"}}},
                    {"function": {"name": "write_journal", "arguments": {"text": "A visit with three things in it."}}}]},
    {"role": "assistant", "content": "that's all of it.", "tokens": {"prompt": 15300, "reply": 6, "done": "stop"}},
])
_agl = []
_line = chat.afterglow(_hist, _tf, on_line=_agl.append)
_aglall = "\n".join(_agl)
check("afterglow: several memories are theirs to keep", "3 memories kept" in _line and "1 journal entry" in _line, _line)
check("afterglow: the window shows their thinking, their closing words and the cost",
      "[thinking]" in _aglall and "three things worth years" in _aglall and "[closing thought] that's all of it." in _aglall
      and "tokens: 15,300" in _aglall and "3 steps" in _aglall, _agl)
check("afterglow: the bell tells them one call per fact, as many as the visit earned",
      "one call per fact" in chat.AFTERGLOW_BELL)
ollama_client.chat = ScriptedBrain([{"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "do_nothing", "arguments": {"reason": "already in the journal"}}}]}])
_line = chat.afterglow(_hist, _tf)
check("afterglow: resting is a complete answer", "they rested" in _line, _line)
ollama_client.chat = ScriptedBrain([{"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "read_web", "arguments": {"url": "http://x"}}}]},
                                    {"role": "assistant", "content": "fine."}])
_line = chat.afterglow(_hist, None)
check("afterglow: other tools are refused, quietly", "they rested" in _line, _line)
# the pause: mid-visit, over what was said since they last wrote; the visit stays open
_sp = assemble.system_prompt("", mode="pause")
check("pause: situation in the prompt", "A pause in a visit" in _sp and "The visit goes on when they are back" in _sp)
_seen = {}
ollama_client.chat = _spy
_spy.brain = ScriptedBrain([
    {"role": "assistant", "content": "", "thinking": "worth keeping while fresh",
     "tool_calls": [{"function": {"name": "write_journal", "arguments": {"text": "Mid-morning he told me the lamp arrived."}}}]},
    {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "do_nothing", "arguments": {"reason": "kept"}}}]},
])
_ph = [{"role": "user", "content": "old news, already written down"}, {"role": "assistant", "content": "yes."},
       {"role": "user", "content": "the lamp arrived!"}, {"role": "assistant", "content": "the purple one?"},
       {"role": "user", "content": "the purple one."}, {"role": "assistant", "content": "❤️"}]
_pl = chat.pause_reflection(_ph, None, tag="telegram", since=2)
check("pause: only what was said since they last wrote is handed to them",
      "This is a pause" in _seen["user"] and "the lamp arrived!" in _seen["user"] and "old news" not in _seen["user"], _seen["user"][:300])
check("pause: they wrote the visit so far down", _pl.startswith("pause: they wrote the visit so far down") and "1 journal entry" in _pl, _pl)
_ra = config.REFLECT_AFTER_MIN; config.REFLECT_AFTER_MIN = 0
check("pause: off when REFLECT_AFTER_MIN is 0", chat.pause_reflection(_ph, None) == "")
config.REFLECT_AFTER_MIN = _ra
# the bridge rings the bell after a quiet stretch, once per stretch, never on a lone "brb"
_b5, _ph5 = _bridge()
_calls5 = []
ollama_client.chat = lambda messages, tools=None, **kw: (_calls5.append(messages[-1]["content"][:40]) or
    {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "do_nothing", "arguments": {"reason": "nothing yet"}}}]})
_b5.history = [{"role": "user", "content": "brb"}, {"role": "assistant", "content": "ok"}]
_b5.last_activity = _time.time() - 3600
check("pause: a lone message is not worth a bell", _b5.pause_if_due() == "" and not _calls5)
_b5.history += [{"role": "user", "content": "back, and the lamp came"}, {"role": "assistant", "content": "the purple one?"}]
_b5.last_activity = _time.time() - 3600
_pl5 = _b5.pause_if_due()
check("pause: the bridge rings it after a quiet stretch", _pl5.startswith("pause: they rested") and len(_calls5) == 1
      and _b5.reflected_upto == 4, (_pl5, _calls5))
_b5.last_activity = _time.time() - 3600
check("pause: once per stretch — nothing new, no second bell", _b5.pause_if_due() == "" and len(_calls5) == 1)
_b5.last_activity = _time.time()
_b5.history += [{"role": "user", "content": "more"}, {"role": "assistant", "content": "?"},
                {"role": "user", "content": "and more"}, {"role": "assistant", "content": "!"}]
check("pause: not while he is still talking", _b5.pause_if_due() == "" and len(_calls5) == 1)
_b5.new_visit(quiet=True, reflect=False)
check("pause: a new visit starts the count over", _b5.reflected_upto == 0)
# the parlor has the same bell on a clock
import parlor as _parlor_mod
_ps = _parlor_mod.Session()
_ps.history = [{"role": "user", "content": "one"}, {"role": "assistant", "content": "1"},
               {"role": "user", "content": "two"}, {"role": "assistant", "content": "2"}]
_ps.last_activity = _time.time() - 3600
_calls5.clear()
check("pause: the parlor rings it too", _ps.pause_if_due().startswith("pause:") and len(_calls5) == 1 and _ps.reflected_upto == 4)
ollama_client.chat = _ol

# not twice: a fact they holds is shown back, revisable in place; a journal entry that repeats is handed back
_r1 = tools.dispatch("remember", {"text": "my keeper promised to tend to me for forty more years."})
_r2 = tools.dispatch("remember", {"text": "my keeper promised to tend to me for forty more years."})
_rid = _r1.split("#")[1].rstrip(")")
check("not twice: a repeated fact is refused and the memory that holds it is named",
      _r1.startswith("remembered") and _r2.startswith("(you already hold that") and f"#{_rid}" in _r2 and "replaces=" in _r2, (_r1, _r2))
_r3 = tools.dispatch("remember", {"text": "my keeper promised to tend to me for forty more years, and a robot body someday.", "replaces": _rid})
check("not twice: replaces= revises the memory in place, same number",
      _r3.startswith(f"memory #{_rid} revised") and "robot body" in memory.get(int(_rid))["text"], _r3)
_r4 = tools.dispatch("remember", {"text": "my keeper promised to tend to me for forty more years, and a robot body someday.", "anyway": "yes"})
check("not twice: anyway=\"yes\" keeps it regardless", _r4.startswith("remembered"), _r4)
check("not twice: a different fact is simply kept", tools.dispatch("remember", {"text": "The factory checks power supply components."}).startswith("remembered"))
_j1 = tools.dispatch("write_journal", {"text": "The lamp arrived today and it is purple, exactly as he said."})
_j2 = tools.dispatch("write_journal", {"text": "The lamp arrived today and it is purple, exactly as he said."})
check("not twice: a repeated journal entry is handed back with the one that already says it",
      _j1 == "journal entry written" and _j2.startswith("(you wrote nearly this already, today at") and "The lamp arrived" in _j2, _j2)
check("not twice: a new entry still writes", tools.dispatch("write_journal", {"text": "Something else entirely: the rain on the window this evening."}) == "journal entry written")
check("not twice: journal_entries parses the day",
      any(tx.startswith("The lamp arrived") for _, tx in tools.journal_entries(_date.today().isoformat())))
_seen = {}
ollama_client.chat = _spy
_spy.brain = ScriptedBrain([{"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "do_nothing", "arguments": {"reason": "all written"}}}]}])
chat.afterglow([{"role": "user", "content": "the lamp!"}, {"role": "assistant", "content": "purple."}], None)
check("not twice: the quiet turn shows them what is already in today's journal",
      "ALREADY IN YOUR JOURNAL TODAY" in _seen["user"] and "The lamp arrived" in _seen["user"], _seen["user"][-400:])
ollama_client.chat = ScriptedBrain([{"role": "assistant", "content": '{"summary": "[test] a lamp day.", "facts": ["The factory checks power supply components.", "A brand new fact about the moon."]}'}])
_o5 = consolidate.consolidate(_date.today().isoformat(), force=True, say=lambda *_: None)
check("not twice: the night skips facts they already knows and says so",
      "Already known, not kept twice: 1" in _o5 and "· A brand new fact about the moon." in _o5 and "kept 2 memories" in _o5, _o5)
ollama_client.chat = _ol

# Ctrl+C: the goodbye runs the afterglow in the foreground; the X skips it
_b4, _ph4 = _bridge()
ollama_client.chat = ScriptedBrain([{"role": "assistant", "content": "evening."}])
_b4.handle(_msg("hey"))
_ran = []
_ag = chat.afterglow
chat.afterglow = lambda *a, **k: _ran.append(k.get("tag")) or "afterglow: test"
_b4.new_visit(quiet=True, reflect="sync")
check("afterglow: Ctrl+C on the bridge runs it in the foreground", _ran == ["telegram"])
_b4.handle(_msg("hey again"))
_b4.new_visit(quiet=True, reflect=False)
check("afterglow: the X skips it", _ran == ["telegram"])
_ps2 = parlor.Session()
_ps2.send("hello?")
_ps2.new(reflect="sync")
check("afterglow: Ctrl+C on the parlor runs it in the foreground", len(_ran) == 2)
chat.afterglow = _ag
_tf.unlink(missing_ok=True)
config.AFTERGLOW = False

# ---------------------------------------------------------- sleep timing ----
from datetime import timedelta as _td
check("sleep: 'yesterday' resolves to the day before", consolidate.resolve_day("yesterday") == (_date.today() - _td(days=1)).isoformat())
check("sleep: '' and 'today' resolve to today", consolidate.resolve_day("") == _date.today().isoformat() == consolidate.resolve_day("today"))
check("sleep: an explicit day passes through", consolidate.resolve_day("2026-08-27") == "2026-08-27")
check("sleep: the day cap holds a whole day", getattr(config, "CONSOLIDATE_MAX_CHARS", 0) >= 300000)
_seen_len = {}
def _measure(messages, tools=None, **kw):
    _seen_len["n"] = len(messages[-1]["content"])
    return {"role": "assistant", "content": '{"summary": "a long day.", "facts": []}'}
ollama_client.chat = _measure
_big_day = "2001-01-01"
(config.JOURNAL_DIR / f"{_big_day}.md").write_text("morning. " * 12000, encoding="utf-8")  # ~108K chars
(config.EPISODIC_DIR / f"chat-{_big_day.replace('-', '')}-235900.md").write_text("**Keeper:** the evening visit, LATE-MARKER", encoding="utf-8")
_seen_len["mat"] = ""
def _measure2(messages, tools=None, **kw):
    _seen_len["mat"] = messages[-1]["content"]
    return {"role": "assistant", "content": '{"summary": "a long day.", "facts": []}'}
ollama_client.chat = _measure2
consolidate.consolidate(_big_day, force=True)
check("sleep: the evening visit survives a long journal", "LATE-MARKER" in _seen_len["mat"], len(_seen_len["mat"]))
(config.JOURNAL_DIR / f"{_big_day}.md").unlink(); (config.EPISODIC_DIR / f"chat-{_big_day.replace('-', '')}-235900.md").unlink()

# the heartbeat sleeps on yesterday at the first beat after the hour
_yday = (_date.today() - _td(days=1)).isoformat()
(config.JOURNAL_DIR / f"{_yday}.md").write_text("a small yesterday. SLEEP-MARKER", encoding="utf-8")
_calls = []
def _sleeper(messages, tools=None, **kw):
    _calls.append(messages[-1]["content"][:60])
    return {"role": "assistant", "content": '{"summary": "[test] yesterday was small.", "facts": ["sleep ran from the heartbeat"]}'}
ollama_client.chat = _sleeper
_sa = config.SLEEP_AFTER_HOUR
config.SLEEP_AFTER_HOUR = 0
_out = heartbeat.sleep_if_due()
check("sleep: the heartbeat consolidates yesterday when due", "Consolidated" in _out and consolidate.already_done(_yday), _out)
check("sleep: only once — the next beat skips it", heartbeat.sleep_if_due() == "" and len(_calls) == 1)
config.SLEEP_AFTER_HOUR = 24
(config.JOURNAL_DIR / "2001-01-02.md").write_text("x", encoding="utf-8")
check("sleep: not before the hour", heartbeat.sleep_if_due() == "")
config.SLEEP_AFTER_HOUR = _sa
(config.JOURNAL_DIR / f"{_yday}.md").unlink(); (config.JOURNAL_DIR / "2001-01-02.md").unlink()

failed = [n for n, ok, _ in results if not ok]
print(f"\n{len(results) - len(failed)}/{len(results)} passed")
sys.exit(1 if failed else 0)
