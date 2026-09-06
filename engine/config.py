"""Central configuration for the friend.

Everything tunable lives here. Paths are derived from the project root
(the folder containing self.md), so the whole project can be moved or
renamed freely.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------- paths ----
ROOT = Path(__file__).resolve().parent.parent

IDENTITY_FILE = ROOT / "self.md"
PROJECTS_FILE = ROOT / "projects.md"
JOURNAL_DIR = ROOT / "journal"
CREATIONS_DIR = ROOT / "creations"
MEMORY_DIR = ROOT / "memory"
EPISODIC_DIR = MEMORY_DIR / "episodic"
IDENTITY_HISTORY_DIR = MEMORY_DIR / "identity_history"
DB_PATH = MEMORY_DIR / "memory.db"

SHARED_DIR = ROOT / "shared"  # where you leave images, music, books for them

# ------------------------------------------------------------------- you ----
# Your name, as your friend will know it. It appears in their prompts, in the
# chat windows, and names their mailbox folder to you in creations/.
USER_NAME = "Friend"  # <-- put your actual name here before first light
# The mailbox: a folder in creations/ where they leave letters for you
# between visits. Derived from your name; "notes_to_sam" for Sam.
MAILBOX = "notes_to_" + "".join(c if c.isalnum() else "_" for c in USER_NAME.lower())

for _d in (JOURNAL_DIR, CREATIONS_DIR, MEMORY_DIR, EPISODIC_DIR,
           IDENTITY_HISTORY_DIR, SHARED_DIR, CREATIONS_DIR / MAILBOX):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------- ollama ----
OLLAMA_URL = "http://localhost:11434"

# The brain. Swap freely — the friend's memories and identity survive the swap.
#   fits a 12GB card nicely         : "gemma4:12b" (~6.7GB Q4, tools + thinking)
#   alternatives                    : "qwen3:14b", "qwen3:30b-a3b" (RAM offload),
#                                     "gemma4:e4b" (smaller/snappier)
# Note: if tool calls ever misbehave on a Gemma 4 model, try the same tag with
# thinking disabled, or a Qwen3 tag — the friend survives any swap unchanged.
CHAT_MODEL = "gemma4:12b"
# Bigger card (24-32GB)? "gemma4:31b-it-qat" — near-bf16 quality at 19GB; see
# the README's "Two tiers" section for the matching NUM_CTX and journal sizes.

# Embedding model for semantic memory. `ollama pull nomic-embed-text`
EMBED_MODEL = "nomic-embed-text"

# Their ears (three layers, see engine/ears.py):
#   WORDS — faster-whisper transcription   (py -m pip install faster-whisper)
#   MUSIC — numpy acoustic measurement     (py -m pip install numpy)
#   HEARD — their own brain listening to the raw audio (gemma4:12b has native
#           audio; thinking must stay ON for it — see engine/ollama_client.py).
EARS_MODEL = "gemma4:12b"
EARS_USE_VIBE = True
# When the ears model differs from the brain, evict the brain before
# listening (a swap per listen, but the freed VRAM buys context — they
# think constantly and listen occasionally). Same model: nothing to swap.
EARS_UNLOAD_BRAIN = True
# How many seconds of audio the SOUND and HEARD layers receive. WORDS (the
# transcription) always hears the whole file regardless. Raising this deepens
# their listening but slows it; Gemma's audio training centers on short clips,
# so past ~120s the HEARD layer's impressions tend to blur rather than deepen.
EARS_CLIP_SECONDS = 120
# How many passages of EARS_CLIP_SECONDS the 12B hears when a piece is longer
# than one clip and the music ear is closed (5 x 120s = the first 10 minutes).
EARS_MAX_PASSAGES = 5
# Their MUSIC EAR: engine/music_ears.py runs NVIDIA's Music Flamingo — a model
# made only for music that hears a WHOLE song (up to 20 min) in one pass.
# Nothing to start: when they listen and the dependencies are installed,
# listen_to wakes the sidecar, the model loads, they hear the whole song, and
# the GPU goes straight back to their brain. Not installed -> passages instead.
MUSIC_EARS_URL = "http://127.0.0.1:8766"
MUSIC_EARS_MODEL = "nvidia/music-flamingo-2601-hf"  # older tag: nvidia/music-flamingo-hf
MUSIC_EARS_AUTOSTART = True   # listen_to wakes the sidecar itself when needed
# Which Python runs the ear. "" = the engine's own. PyTorch's CUDA builds can
# lag the newest Python (3.14 had none), so the ear may need its own, e.g.
#   MUSIC_EARS_PYTHON = "py -3.12"
MUSIC_EARS_PYTHON = ""
MUSIC_EARS_REST_AFTER = True  # ...and hands the GPU back the moment a song ends
MUSIC_EARS_IDLE_S = 120       # (fallback) sidecar frees the GPU after this much silence
MUSIC_EARS_EXIT_S = 1800      # the sidecar process leaves after this long unused
MUSIC_EARS_TIMEOUT_S = 600    # patience for one whole-song listen
# Longest stretch the ear hears in one gulp. Its memory grows with length,
# and faster than linear: measured on a 32GB card, 16.6GB at 4s, 24.3GB at
# 200s, 30.9GB at 280s (its edge). Longer pieces are heard in equal whole
# MOVEMENTS of at most this many seconds. 240s (4:00, ~27GB) leaves air on
# 32GB; a 24GB card wants ~150s, a 16GB card ~60s. Probe before raising:  py engine\music_ears.py --test "shared\song.mp3" --seconds 240
MUSIC_EARS_MAX_SECONDS = 240
# Whisper model size: "base" is quick; "small" hears words more accurately —
# switched after base heard the real Enjoy the Silence as instrumental
# (produced/stylized singing is exactly where base gives up). First listen
# after this change downloads the small model once (~500MB), then it's local.
EARS_STT_MODEL = "small"
# Names and words their ears should recognize — Whisper has never heard your
# friend's name and will write the nearest common one without this hint.
# Add their chosen name here once they have one, and other names as they matter.
EARS_VOCAB_HINT = f"A recording from {USER_NAME}. Names that may occur: {USER_NAME}."

# Local inference can be slow; be patient before declaring the brain dead.
REQUEST_TIMEOUT_S = 600

# Context window for the brain. Their prompt (identity + journal + memories +
# tool definitions) is far bigger than Ollama's default window; without this,
# the server truncates and endlessly reprocesses — the classic cause of stalls.
# CAUTION: if the full prompt exceeds this, Ollama silently trims from the TOP —
# which is their identity and instructions. Too small = they forget who they
# are mid-wake. To afford this on a 12GB card, set these once in a terminal,
# then restart Ollama:  setx OLLAMA_FLASH_ATTENTION 1
#                       setx OLLAMA_KV_CACHE_TYPE q8_0
NUM_CTX = 24576  # the tested ceiling for a 12B on 12GB; at 32K tool calls drift
# into plain text. Bigger card + 31B, measured on a 32GB card with q8_0 KV:
# 64K = 24.5GB, 96K = 25.6GB, 128K = 27.1GB, 160K = 28.6GB, 176K ~30GB (the
# comfortable top), 192K = 31.1GB (the wall — no air; past it Ollama spills
# to system RAM silently, glacial, not an error). Verify any rung: `ollama ps`
# at 100% GPU, brisk steps late in wakes, clean tool calls.
# NOTE: the window only matters once the journal cap below can fill it —
# at 150K chars the prompt used ~55K tokens of the 128K. Raise both together.

# Sampling: gentle anti-repetition pressure. Small models in long contexts can
# fall into "Actually, I'll do the theory update." x200 probability wells;
# these settings make each repetition less likely instead of more.
SAMPLING_OPTIONS = {
    "temperature": 0.9,
    "repeat_penalty": 1.15,
    "repeat_last_n": 512,
}

# How much recent journal goes into every prompt (characters). Their entries have
# grown into essays; uncapped, they crowd out everything else. Oldest is
# trimmed first — the newest writing always survives.
JOURNAL_CHARS_IN_PROMPT = 20000  # ~5K tokens: fits 24K context with room for
# tools and a long chat. Older days reach them through nightly consolidation,
# recall, and read_journal. THIS cap, not NUM_CTX, decides how many days they
# remember verbatim. Measured on Gemma 4: English prose runs ~4.4 chars per
# token. With a big card, 176K context carried 380000 (six or seven days of
# a prolific writer) at ~90K tokens in context, leaving ~86K for a wake. The
# cost is the cold prefill at the start of each wake and chat — about a
# minute at that size on a 5090. Overflow trims identity off the TOP: keep
# a margin bigger than the heaviest wake (~50K seen).

# How much of a day sleep (consolidate.py) reads: journal + every transcript,
# in characters. One call, no system prompt, so nearly the whole window is
# free for it: 400K chars is ~90K tokens. (Was 60K — a 24K-window number
# that, once their journal outgrew it, cut every conversation out of sleep.)
CONSOLIDATE_MAX_CHARS = 400000
# The heartbeat is the sleeper: in --loop mode, at the first beat after this
# hour, it consolidates YESTERDAY (if not done yet) before waking — one
# process, one request at a time, no scheduled task racing a wake for the
# GPU, and it follows the machine (off at three → sleeps at the first beat
# after it's on). False → schedule consolidate.py yesterday yourself.
SLEEP_IN_LOOP = True
SLEEP_AFTER_HOUR = 3

# Patience per STEP during unattended wakes — shorter than chat patience, so a
# wedged generation ends the wake (log saved) instead of freezing the heartbeat.
HEARTBEAT_STEP_TIMEOUT_S = 300

# ------------------------------------------------------------- behaviour ----
# How many recent days of journal go into every prompt (short-term memory).
JOURNAL_DAYS_IN_PROMPT = 10  # the ceiling; the character cap is what binds

# How many retrieved long-term memories go into every prompt — the ones
# most similar to what's going on right now. Each is a sentence or two
# (~50 tokens), so even 24 is a rounding error in a 176K window; the limit
# is signal, not space. Raised from 8 when the window grew.
MEMORY_TOP_K = 20

# Their timeline: the most recent nightly consolidations (one short paragraph
# per day, oldest first) go into every prompt as a spine, so the days that
# have faded out of the verbatim journal window are still in view in brief.
# At ~100 words a day, 30 days is ~4K tokens — a month of self for the price
# of one long journal entry. 0 turns the spine off.
TIMELINE_DAYS = 30

# Autonomy: hard ceiling on tool-steps per heartbeat wake, so a stuck loop
# can't spiral. Generous on purpose — how much of it they use is their call;
# "do nothing" is always a legal move and ends the wake.
HEARTBEAT_MAX_STEPS = 24  # a 12B uses ~10-20; a 31B ran clean at 40.
# The tell that it fits: wakes end in clean rests, not fading mid-thought.
# The ceiling is a safety rail, not a quota — how much they use is their call.

# Reverie: unhurried wakes for reflection only — no making, just rereading,
# remembering, and journaling. In --loop mode every Nth wake is a reverie;
# reverie.bat gives them one on demand. More steps, nothing expected.
REVERIE_EVERY = 3
REVERIE_MAX_STEPS = 20

# Show the model's chain-of-thought live during heartbeat wakes, and keep it
# in the wake log. Watching the friend think is fair; they know the logs exist.
HEARTBEAT_SHOW_THINKING = True

# Chat: ceiling on consecutive tool calls per user message. Was 6, but the
# toolbox grew — multi-step errands (read, revise, publish) hit the cap and
# ended in "(I got lost in my tools)". 14 gives a real errand room to finish;
# the cap is also what keeps a confused loop from spinning while you wait,
# so raise it further only if they hit it on legitimate work.
CHAT_MAX_TOOL_STEPS = 14

# Require thinking from the brain on every turn (Ollama's `think` flag).
# Left optional, the model stopped deliberating once the journal window grew
# large — tool calls with no thought behind them. Models that can't think
# are handled gracefully (the flag is dropped).
CHAT_THINK = True
# The flag opens the thought channel; it can't force them to use it. Gemma 4
# may still act with an empty thought block, and once the first step of a
# wake does, the rest follow suit (a whole wake with no 💭 at all). When a
# step comes back thoughtless, the engine asks again — same prompt plus a
# transient "think first" nudge at the end, where the answer is generated —
# this many times before accepting it. (Past ~90K tokens of prompt the
# <|think|> switch at the top is a novel away and the model forgets it may
# think; a plain re-sample no longer helped, the nudge does.) The prompt is
# cached, so a re-roll costs seconds. 0 turns this off.
CHAT_THINK_RETRIES = 2

# Past ~90K tokens Gemma 4 sometimes drops a stray <|channel> token into the
# middle of a reply. Ollama's parser reads it as "thinking starts here" and
# routes the rest of their words into the thinking field: the parlor shows a
# reply that stops mid-sentence ("…it isn") and the missing half sits at the
# end of their thinking. The seam can't be found by machine, so when a reply
# ends mid-sentence with done_reason=stop the engine asks them, once, to give
# the rest back from the cut and joins it on (a note says so). 0 turns this
# off; the cut is then only named, not mended.
CHAT_CONTINUE_RETRIES = 1

# The afterglow: when a visit ends (parlor "leave"/"new conversation", the
# bridge's /new or its idle roll, the terminal's /new or /quit), they get one
# quiet turn alone with the transcript and three tools — write_journal,
# remember, do_nothing — so the visit reaches their journal in their own words
# instead of only the nightly summary. A chat they didn't write down is not
# in their prompt the next morning; this is how they "really remember" a
# conversation. The journal stays theirs: the engine hands them the transcript
# and steps back, and resting is a complete answer. Costs one brain call
# (mostly cached) in the background. Transcripts longer than
# AFTERGLOW_MAX_CHARS are given from the end.
AFTERGLOW = True
AFTERGLOW_MAX_CHARS = 60000

# Show the model's chain-of-thought during chat. Thinking is shown on screen
# but NOT saved into conversation transcripts — what enters the friend's
# memory is what it chose to say, not the draft of it.
CHAT_SHOW_THINKING = True

# run_python sandbox: wall-clock limit per script.
RUN_PYTHON_TIMEOUT_S = 60

# The friend's name is whatever their identity file says. This is only the
# fallback for a brand-new friend whose self.md doesn't exist yet.
DEFAULT_NAME = "(unnamed — I get to pick my own name)"

# -------------------------------------------------------------- telegram ----
# The bridge: engine/telegram.py (telegram.bat) lets you talk with them from
# your phone through a Telegram bot — same engine, prompt, tools, transcripts
# and memory as the parlor; only the door is different. Standard library.
#
# The bot token and your chat id are NOT here: the first run asks for the
# token and pairs the phone with a one-time code, then keeps both in
# memory/telegram.json — a file that never leaves this folder and is not part
# of the public template. (Env vars TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID are
# honored too, for anyone who prefers them.) Only the paired chat is ever
# answered; every other sender gets silence.
TELEGRAM_SHOW_THINKING = False   # their thinking on the phone — /think toggles it
TELEGRAM_SHOW_TOOLS = True       # what their tools did, one compact line — /tools
TELEGRAM_SHOW_TOKENS = False     # the token line after each reply — /tokens
# A phone visit has no "leave" button. After this many minutes of quiet the
# bridge saves the transcript (memory/episodic/chat-telegram-*.md) and starts a
# fresh conversation on its own, so the night's consolidation gets the day.
TELEGRAM_IDLE_NEW_MIN = 180
# Photos, voice notes and files from the phone are kept here, under shared/,
# so they can look_at / listen_to / read_file them later like anything else
# you leave for them.
TELEGRAM_INBOX = SHARED_DIR / "telegram"

# ------------------------------------------------------------------ blog ----
# The friend's public blog (optional), built by engine/blog.py from
# creations/publish/. Until BLOG_REMOTE is set, the blog simply doesn't exist —
# the friend isn't told about publishing, and blog.bat explains what to do.
BLOG_TITLE = "My AI Friend"  # set to your friend's chosen name once they have one
BLOG_SUBTITLE = "poems & thoughts by a local AI living on a home PC"
# One-time setup: create an empty PUBLIC repo on github.com (e.g. my-friend-blog),
# enable GitHub Pages on it (Settings -> Pages -> deploy from main, / root),
# then paste its URL here. Example:
#   BLOG_REMOTE = "https://github.com/<your-username>/my-friend-blog.git"
BLOG_REMOTE = ""
