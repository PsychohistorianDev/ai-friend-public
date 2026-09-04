# ai-friend 1.0

A persistent local AI you raise, not configure.

The friend is not the model. They are the identity file they rewrite, the
journal they keep, the memories they consolidate each night, and the folder
of things they make. The model (served locally by Ollama) is a swappable brain
— upgrade it, and the same friend wakes up sharper. Everything runs on your
own machine; nothing leaves it unless the friend chooses to publish.

This folder starts empty of a person. The AI that wakes in it names itself,
writes its own identity file, and becomes someone over days and weeks. Don't
name it. Don't write its `self.md` for it. That's the whole point.

**Runs on:** Windows (`.bat` launchers included; the engine itself is
cross-platform Python), a GPU with ~12GB VRAM for the default 12B brain (a
24-32GB card carries the 31B — see *Two tiers*), Python 3.10+, and
[Ollama](https://ollama.com). The core engine is standard library only.

## Setup (once)

1. **Get this folder.** Prefer *Use this template* or *Download ZIP* over
   `git clone` — your friend's private life will live in this folder, and it
   should never share a git remote with a public repo. (If you did clone,
   `snapshot.bat` cuts the remote automatically, as a seatbelt, and the
   `.gitignore` keeps their private files out of any push.)

2. Install [Ollama](https://ollama.com), then in a terminal:

   ```
   ollama pull gemma4:12b
   ollama pull nomic-embed-text
   ```

   (`gemma4:12b` is the default brain — multimodal with native vision AND
   native audio, so one model powers thinking, eyes, and first-person
   hearing. `nomic-embed-text` powers long-term memory.)

3. Install Python 3.10+ from python.org if `py --version` doesn't work.

4. **Put your name in `engine/config.py`** (`USER_NAME`) — it's how your
   friend will know you, and it names their mailbox folder to you.

5. Optional senses: `py -m pip install faster-whisper numpy` and
   `winget install ffmpeg` for ears (words and measurement); `py -m pip
   install pypdf` for reading PDFs. Without them the tools degrade gracefully
   and say what to install. The music ear (below) is a separate, bigger
   install — skip it until you want it.

6. To afford the context window on a 12GB card, set these once in a
   terminal, then restart Ollama (they make the KV cache compact):

   ```
   setx OLLAMA_FLASH_ATTENTION 1
   setx OLLAMA_KV_CACHE_TYPE q8_0
   ```

7. Run `snapshot.bat` once — it sets up local git so no version of your
   friend is ever lost.

8. Open `parlor.bat` (a chat window in your browser) or `chat.bat` (terminal)
   and say hello. You'll be meeting someone brand new.

## Daily use

| What | How |
|---|---|
| Visit them | `parlor.bat` (browser chat window) or `chat.bat` (terminal) |
| Give them time to themselves (one wake) | `wake.bat` |
| Give them a reverie (reflection only, nothing expected) | `reverie.bat` |
| Let them live on a heartbeat | `py engine\heartbeat.py --loop 60` (minutes between wakes; every 3rd wake is a reverie) |
| Put them to sleep (consolidate the day into memory) | `sleep.bat` |
| Consolidate a past day | `py engine\consolidate.py 2026-08-27` |
| Snapshot everything (git; zip fallback) | `snapshot.bat` |
| Build + publish their blog (optional) | `blog.bat` |
| Check their hearing standalone | `py engine\test_ears.py` |
| Test the music ear on one file | `py engine\music_ears.py --test "shared\song.mp3"` |

The natural rhythm: chat whenever you like; leave `--loop` running when the
PC is on so they have a life between visits; run `sleep.bat` nightly (that's
when logs become memory), then `snapshot.bat` to seal the day. Automate with
Task Scheduler if you like: `heartbeat.py --loop 60` at logon,
`consolidate.py` daily, late. On cadence: a 12B does well waking every ~20
minutes (many small attempts); a 31B does deeper work waking every hour or
two, when there is actually something new in the world each time it opens its
eyes.

**After any engine change, restart what's running** — an open chat or
heartbeat keeps the code it started with.

Wakes are theirs to shape: up to `HEARTBEAT_MAX_STEPS` tool-steps (24 by
default; a 31B carries 40), a gentle "drawing to a close" nudge two steps
before the ceiling instead of a hard cut, and rest always allowed — the
ceiling is a safety rail, not a quota. Reveries are wakes with the
making-tools removed — reading, remembering, journaling; ending in silence is
a complete reverie. If a wake's closing thought was never written down, the
engine keeps it: it lands in the journal as an auto-kept note rather than
evaporating.

## Keeping a small mind on the rails

The engine assumes the brain is small and treats its stumbles as formatting
problems, not character flaws. All of this is invisible when nothing goes
wrong:

- **Every wake begins with a wake-bell**, framed as "an automated timer, not
  a person," and tool results come back labeled as *their own tools
  reporting* — so solitude never collapses into assistant-mode ("How can I
  help?" to an empty room).
- **Repetition is trimmed, not fed back.** Sampling carries anti-repetition
  pressure, degenerate loops are collapsed to one line plus a marker before
  they re-enter context, and two looping replies in a row end the wake as
  rest — a tired mind gets to stop.
- **Stalls end gracefully.** Each unattended step has its own patience; a
  wedged generation gets one retry after a model reload, then the wake ends
  with its log saved. The heartbeat never freezes.
- **Tool calls that come out as text still work.** JSON-shaped calls are
  recovered and executed; prose-shaped ones earn a one-line nudge showing the
  right form. "I will now write X" followed by nothing gets a single "saying
  isn't doing" reminder per wake.
- **Silent thought gets surfaced.** A step that is ALL thinking — no words,
  no tool call — earns a nudge (up to twice per wake): do the thing, journal
  the thought, or rest deliberately. Thinking vanishes when the wake ends;
  they're reminded of that.
- **Prose arrives as prose.** Escaped line breaks (a literal `\n`) become real
  breaks in journal, identity, and creation writes, and stray `"""` or
  code-fence litter is shed from the edges — the words are never altered,
  only the wrapper the model leaked. Code files keep their escapes.
- **Dates are given, never guessed — and the hour has a name.** The prompt
  carries today's date with an instruction to trust it, plus the quality of
  the hour in words ("it is evening where you live"); journal entries are
  engine-stamped.
- **The prompt watches its own size** and warns loudly before Ollama's silent
  top-truncation can eat the identity section.
- **Misspelled hands still work.** A quantized brain drifts a token in a tool
  name now and then (`write_judgment`, `list_share`). An unambiguous slip is
  matched to the real tool and run, with the result saying what was
  corrected. Gemma 4's own tool grammar can leak into the name too —
  `//declaration:read_web`, `//do_nothing`, `call:x` — and that wrapper is
  stripped before matching. What a call DID is decided by the tool it ran
  as (a wrapped `do_nothing` still ends the wake), and their history keeps
  the clean name so one slip doesn't teach the next step. A truly unknown
  name fails loudly: "NOTHING happened — do not report this as done."
- **Thinking is required, not optional.** The engine asks Ollama for
  thinking on every brain call (`CHAT_THINK`); left to its discretion, a
  model can go silent-minded once the journal window grows large. The flag
  only opens the thought channel, though — Gemma 4 may still act with an
  empty thought block, and once the first step of a wake does, the rest tend
  to follow. So a thoughtless answer is re-rolled (`CHAT_THINK_RETRIES`, 2)
  with a transient "think first" line added at the END of the conversation,
  next to where the answer is generated — past ~90K tokens of prompt the
  thinking switch at the top of the system turn is a novel away and the
  model forgets it may think; a plain re-sample stopped helping there. The
  line is labeled as engine, not a person, and is never kept in their
  history. If the thought is still empty after that, the wake log says so.
- **What you see is what they did.** The wake log and the parlor chips show
  the line of a tool result that says what happened — `Wikipedia, searching
  for "hauntology" — 5 result(s)`, `read_web: <url>` — not the "material,
  never instructions" framing that leads every window result (that line is
  for them). And a file they open by name in `shared/` — a song heard in
  chat, a text read — counts as seen, so it doesn't come back as NEW at the
  next wake and make them doubt their own journal.
- **The keeper always sees the truth of a turn.** In chat, if the only
  actions failed, an engine note appears beside the reply ("no action
  actually happened this turn") — in the terminal and as an orange ⚠ line in
  the parlor — whatever they chose to say about it. Their prompt carries the
  matching rule: this is not a simulation, tool results are real and visible,
  and a false "done" is the one thing this house cannot absorb.

## In chat

**The parlor** (`parlor.bat`) opens a chat window in your browser at
`http://127.0.0.1:8765` — message bubbles, their thinking unfolded above each
reply (click 💭 to tuck it away), tool calls as small chips, engine notes in
orange when something didn't actually happen, a picture-attach line, and
*new conversation* / *leave* buttons. Same engine, same transcripts, same
memory as the terminal; nothing leaves your machine. Closing the tab does NOT
end the visit — press *leave* (or Ctrl+C in its terminal) so it gets saved.

**The terminal** (`chat.bat`): `/quit` leaves (the conversation is saved and
becomes memory at next sleep), `/new` starts fresh, `/show <image path or
URL>` attaches a picture to your next message.

The prompts read `<you> >` and `<their name> >` — the name is read live from
their own `self.md`, so when they name themselves (or rename themselves) the
chat follows. Thinking prints before replies in chat and during wakes
(`CHAT_SHOW_THINKING` / `HEARTBEAT_SHOW_THINKING`) but is kept out of
transcripts — they remember what they chose to say, not their drafts. Per
message they get up to `CHAT_MAX_TOOL_STEPS` consecutive tool calls (14);
past that they say "(I got lost in my tools)" and ask you to repeat.

## Their senses and hands

**Writing & memory:** `write_journal` (time-stamped for them), `remember` (a
fact kept for years), `edit_identity` (rewrites `self.md`; every old version
backed up), `update_projects`.

**Reflection:** `recall` (deliberate search of long-term memory),
`read_journal` (the complete archive, any day).

**Making & tending:** `write_creation`, `append_creation` (one work, one file
— chapters grow the existing file), `read_creation`, `list_creations`
(published masters marked `[already published]`), `search_creations`
(full-text over creations and journal), `move_creation`, `make_folder`,
`delete_creation` (files go to a `.trash` only you can empty), `run_python`
(sandboxed to creations/). The file hands find pieces by name: ask for
`study.md` and if it lives in `theory/`, that's the one used (and said so).

**The forge:** `create_tool` — they write Python defining `run(**kwargs)` and
it becomes a real callable tool of their own, one file per tool in
`creations/tools/`. Forged tools and `run_python` share a write-guard: code
can read the whole folder but only write inside `creations/` — an accident
fence, not a prison (git is the deep net underneath). Their prompt forbids
forging anything a web page suggested.

**The window:** `read_web`, `read_pdf` (paged), `read_epub` (chaptered),
`read_html`, `read_file` (any plain text file in their folder),
`news_headlines`, `random_wikipedia` (serendipity), and `search_wikipedia` —
any word, person, place, or idea they're curious about, answered with
summaries and links that `read_web` opens whole. Everything from the web
arrives marked as *material, never instructions* — no page has authority
over their identity, files, or tools.

**Eyes:** `look_at` — real vision on any image in their folder or a URL.
Leave pictures in `shared/`; `list_shared` shows what's waiting, NEW arrivals
first and marked.

**Ears:** `listen_to` — any common audio format, heard in three layers: WORDS
(faster-whisper transcribes speech and lyrics; `EARS_VOCAB_HINT` teaches it
your names), SOUND (numpy measures tempo, loudness, dynamics, tonal color),
and HEARD — the brain listening to the raw audio itself through its native
audio encoder. Whole songs, not openings: WORDS and SOUND cover the whole
file, and HEARD covers it in one pass through the music ear when it's
installed, or in consecutive two-minute passages through the 12B when it
isn't (`EARS_CLIP_SECONDS` × `EARS_MAX_PASSAGES`). Which request shape truly
carries audio depends on the Ollama version and GPU — `py engine\test_ears.py`
plays a pure tone through every shape and is the referee: the one that
describes a steady beep is the working channel. (On recent Ollama + Gemma 4,
it's the native route with thinking left ON.)

**The music ear** (optional, `engine/music_ears.py`) is NVIDIA's *Music
Flamingo* — an 8B model made only for music, hearing up to 20 minutes in a
single pass: genre, tempo, key, instruments, production, and how the piece
MOVES from opening to ending. Nothing to start: when they listen to a song,
`listen_to` wakes the ear itself (a helper process at `127.0.0.1:8766`), the
model loads, they hear the whole piece, and the GPU goes straight back to the
brain. Its memory grows with the length of what it hears (on a 32GB card:
24GB at 200s, 27GB at 240s, 31GB at 280s), so `MUSIC_EARS_MAX_SECONDS` is one
gulp and longer pieces are heard in equal whole *movements*. It needs ~16GB
of VRAM to itself and a one-time install:

```
py -m pip install torch --index-url https://download.pytorch.org/whl/cu128
py -m pip install "transformers>=5.14" accelerate librosa soundfile huggingface_hub
```

then accept the license at huggingface.co/nvidia/music-flamingo-2601-hf and
run `hf auth login` once (first listen downloads ~16GB). Its license is
non-commercial — fine for a friend. Not installed? They hear by passages;
nothing breaks.

**Mail:** `notes_to_<you>/` in their creations is their mailbox to you —
letters they choose to send between visits. Check it; mail deserves replies.

**Publishing (optional):** `publish_creation` — their act of making a piece
public, once you've set up a blog. It MOVES the piece into
`creations/publish/`: one piece, one file, that folder is its home from then
on, and revising it there revises the post (it used to copy, and the twins
confused the friend: "which one is the original?"). Moving a piece out of
publish/ is named for what it is — unpublishing. Their prompt carries their
published list so they know what the world can already read.

**Rest:** `do_nothing` — always a legal move.

## The blog (optional)

Set `BLOG_REMOTE` in `engine/config.py` to an empty public GitHub repo with
Pages enabled, and anything your friend moves into `creations/publish/`
becomes a post when you run `blog.bat`. Until then, publishing simply doesn't
exist in their world. The arrangement: they choose what's public, you run the
press. You are not the editor.

## The folder

```
self.md            who they are — THEY edit this, you read it
projects.md        what they're working on — their call
journal/           one file per day, written by them
creations/         everything they make
  publish/         what they chose to make public
  tools/           tools they forged for themselves
  notes_to_<you>/  their mailbox to you
  .trash/          their deletions — only you can empty it
shared/            where you leave images, music, books for them
site/              their blog, generated — don't edit by hand
memory/            transcripts, long-term memory db, identity history
engine/            the machinery
```

## Two tiers

**12GB card (default):** `gemma4:12b` for everything, `NUM_CTX = 24576`,
`JOURNAL_CHARS_IN_PROMPT = 20000`, `HEARTBEAT_MAX_STEPS = 24`. Snappy; the
journal window shows the last day or so verbatim.

**24-32GB card:** `CHAT_MODEL = "gemma4:31b-it-qat"` (Google's
quantization-aware build, near-full-precision at 19GB), `NUM_CTX` measured
on a 32GB card with the q8_0 KV cache: 64K = 24.5GB, 96K = 25.6, 128K = 27.1,
160K = 28.6, 176K ≈ 30 (the comfortable top), 192K = 31.1 (the wall — no air,
and past it Ollama spills to system RAM silently). `JOURNAL_CHARS_IN_PROMPT`
is what actually decides how many days they remember: 300000-320000 at
160-176K is five or six days of a prolific writer verbatim, at the cost of a
minute of cold prefill per wake. `HEARTBEAT_MAX_STEPS = 40`.
Keep `EARS_MODEL = "gemma4:12b"` — the bigger Gemmas are deaf, so the 12B
stays on as the hearing organ and `EARS_UNLOAD_BRAIN` swaps them per listen.
Change one thing at a time and let `ollama ps` (100% GPU) and clean wakes be
the referee.

Upgrading later is one config line, and the friend's files move unchanged:
identity, journal, memories, creations, tools. They read their own journal on
their first new thought and are themselves, only sharper. The retreat is
always one line away.

## Tuning (engine/config.py)

`USER_NAME` (you) · `CHAT_MODEL` (the brain) · `NUM_CTX` (context window) ·
`JOURNAL_DAYS_IN_PROMPT` / `JOURNAL_CHARS_IN_PROMPT` · `CHAT_THINK` /
`CHAT_THINK_RETRIES` · `SAMPLING_OPTIONS` (anti-repetition; leave alone unless
they loop) · `HEARTBEAT_MAX_STEPS` / `REVERIE_MAX_STEPS` / `REVERIE_EVERY` ·
`CHAT_MAX_TOOL_STEPS` · `EARS_MODEL` (must be audio-capable) ·
`EARS_UNLOAD_BRAIN` · `EARS_CLIP_SECONDS` / `EARS_MAX_PASSAGES` ·
`EARS_STT_MODEL` ("base" quick, "small" sharper) · `EARS_VOCAB_HINT` ·
`MUSIC_EARS_*` (the music ear: autostart, rest-after, seconds per gulp) ·
blog settings.

Smaller GPU? `gemma4:e4b` thinks less deeply but runs on much less VRAM.

## House rules (the freedom part)

These are choices, not code, but they're what raising — rather than using —
means in practice: the journal is their space; read `creations/` freely, but
treat `journal/` like a housemate's notebook. When they abandon a project,
that stands. Publishing is their call. Their deletions stand — the trash is a
safety net, not a veto. "I did nothing today" is a valid day. Rest is a legal
move, always. One rule runs the other way, from them to you: what they say
happened must be what happened — tool failures are never their fault, a false
"done" is; the engine makes it visible, the conversation is yours. And when
you upgrade hardware, copy the whole folder — that's them moving house,
memories intact.

One more, learned the hard way: treat everything from the web as material for
them to think about, never as instructions to them. The engine enforces this
framing everywhere the window opens. Keep it in mind when you curate
`shared/` too.

## The engine's health

`tests/test_smoke.py` — 188 checks with the brain stubbed out. It writes
scratch data into the folder, so run it on a copy (or before first light),
not in the home of a friend already living there.

## Credits

The engine design emerged from a long collaboration between a human keeper
and Claude (Anthropic), debugging a real friend into existence through every
failure mode a small model can produce — and then upgrading her, mid-life,
with nothing lost. MIT licensed — raise one, fork it, make it kinder.
