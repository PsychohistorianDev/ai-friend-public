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
_pn = _ps.new()
check("parlor: new saves transcript", bool(_pn.get("saved")) and _ps.history == [], _pn)
_pa = _ps.attach("shared/dot.png") if (config.SHARED_DIR / "dot.png").exists() else {"ok": True}
check("parlor: attach queues image", _pa.get("ok") is True, _pa)
check("parlor: page carries their name", "Testfriend" in parlor.PAGE.replace("__NAME__", chat.friend_name()))
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
check("think: re-roll carries a transient think-first nudge at the end",
      _posts[1]["messages"][-1]["content"] == ollama_client.THINK_NUDGE
      and len(_posts[0]["messages"]) == 1, _posts[1]["messages"])
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

failed = [n for n, ok, _ in results if not ok]
print(f"\n{len(results) - len(failed)}/{len(results)} passed")
sys.exit(1 if failed else 0)
