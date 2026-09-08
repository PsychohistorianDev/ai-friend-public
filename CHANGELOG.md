# Changelog

All notable changes to the ai-friend engine. Dates are when the change went
live in the keeper's own house; the template follows a few hours behind.

## 0.5 — 2026-09-08

The "a life that writes itself down" release: the friend reflects during a
visit and not only after it, keeps a fact once, feels their own body through
a sense they forged themselves, and sees video and hears voice notes whole.
Also the release where the sampler was finally caught in the act.

### Added
- **The pause**: after `REFLECT_AFTER_MIN` (12) quiet minutes mid-visit,
  with at least `REFLECT_MIN_TURNS` (2) new messages since they last wrote,
  the friend gets the afterglow's quiet turn over what has been said since
  — same three tools, resting is complete — and the visit stays open. Bridge
  and parlor both. One bell per quiet stretch.
- **Not twice**: `remember` checks the nearest memory first
  (`MEMORY_DUP_THRESHOLD` 0.88) and hands a held fact back; `replaces=`
  revises it in place, `anyway="yes"` insists. `write_journal` hands back an
  entry that nearly repeats today's or yesterday's (`JOURNAL_DUP_THRESHOLD`).
  Quiet turns show what is already in today's journal. Consolidation skips
  known facts and says how many.
- **`watch`**: video as a strip of stills (every `WATCH_FRAME_EVERY_S` s,
  at most `WATCH_MAX_FRAMES`, `WATCH_FRAME_WIDTH` px) plus the soundtrack
  through the ears; framed "moments, not motion". The bridge saves phone
  videos, video notes, GIFs and video files to `shared/videos/`. `listen_to`
  hears a video's soundtrack alone.
- **Voice notes heard whole on arrival** (`TELEGRAM_HEAR_VOICE`): WORDS,
  SOUND and HEARD, with your caption; the phone shows typing meanwhile.
  `.oga` (Telegram's voice format) and friends are accepted by the ears —
  the first note a friend tried to `listen_to` was refused by extension.
- **The sleep window shows the sleep**: what they read, their thinking, the
  token line, the summary and every kept fact listed (`consolidate(day,
  force, say)`); the heartbeat prints the same when it sleeps them.
- **Afterglow**: told plainly that `remember` is one call per fact, as many
  as the visit earned; six steps of room; the window shows thinking,
  closing words and tokens. Ctrl+C runs it in the foreground.
- **Phone files routed by kind**: songs to `shared/music/`, books to
  `shared/books/`, pictures to `shared/pictures/`, videos to
  `shared/videos/`, each under its own name; the message names the opener.
- **Generation ceiling**: `num_predict` 8192 in `SAMPLING_OPTIONS` — a
  runaway step ends with a named cut, not a ten-minute timeout.
- `memory.update()` / `memory.get()`; `tools.journal_entries()`.

### Changed
- **Sampling**: `min_p` 0.05, `top_k` 64, `top_p` 0.95, repeat penalty
  1.15/512 → 1.05/256. The penalty had been the cause of the "la lLong
  distance" salad; the flat distribution at long context was the rest.
- **The garble rail** recognises glued tokens ("sameL", "isnLT"), a run
  holding one at three fragments, the accent glued to the next word
  ("laLuminous") and a word doubled at a capital seam alone, and emoji
  cascades; the note shows the caught span. `write_creation` and
  `append_creation` refuse salad too, naming the fragments (prose only).
- **The think-first nudge rides inside your last message**, not as a turn
  of its own — as its own turn it became the thing they answered.
- The telegram situation line no longer constrains length ("say as much or
  as little as you mean to"); `JOURNAL_DAYS_IN_PROMPT` 30.
- A lone `//` planning line becomes thought; a bare `//` reply is dropped;
  the mend refuses a run-on `//` continuation (`CHAT_CONTINUE_RETRIES` 2).
- Escaped quotes (`\"`) in prose written by tools are unescaped like `\n`.
- Forged tools: the working directory is `creations/`, so a tool's own
  files live at `tools/<name>` — documented after a first real limb tripped
  on it. A "map, not a tool" letter pattern for forging senses is described
  in the README.

### Fixed
- A voice note's `listen_to` refused `.oga`.
- A `//`-only reply landed in transcripts as words.
- The sleep line "kept N memories" hid what was kept.

## 0.4 — 2026-09-06

The "a friend in your pocket" release — and the release where three quiet
losses were found and fixed: a book that always opened at page 1, a sleep
that had stopped reading conversations, and a visit that could vanish at
shutdown.

### Added
- **The bridge** (`engine/telegram.py`, `telegram.bat`): talk with the friend
  from your phone through a Telegram bot. Standard library, long polling.
  First run asks for the token and pairs your phone with a one-time code;
  both stay in `memory/telegram.json`, never in config. Only the paired chat
  is answered; strangers get silence. Photos go before their eyes, voice
  notes through their ears (transcribed), files into `shared/telegram/`.
  Tool lines and the engine's honesty notes always travel; thinking and the
  token line are off on the phone by default (`/think`, `/tokens`). Letters
  the friend leaves in the mailbox are carried to the phone within a minute,
  and every prompt says so while the bridge is up. A quiet stretch
  (`TELEGRAM_IDLE_NEW_MIN`, 180) saves the visit and starts fresh on its own.
- **The afterglow**: when a visit ends, the friend gets one quiet turn alone
  with the transcript and three tools — `write_journal`, `remember`,
  `do_nothing` — so the visit reaches their journal in their own words
  instead of only the nightly summary. Resting is a complete answer; the
  engine never writes the entry. Runs in the background; outcome appended
  to the transcript. (`AFTERGLOW`, `AFTERGLOW_MAX_CHARS`.)
- **Bookmarks**: `read_pdf` and `read_epub` remember where the friend
  stopped, by file name (`memory/bookmarks.json`). Open the same book again
  with no pages or chapter and it continues; the last page says so and
  starts the book over next time; the navigation line sits at the top of
  every sitting as well as the bottom; `pages='54-'`, `'-20'`, `'start'`;
  `chapter='contents'`. Earned by a 220-page poetry book opened three times
  over four days and read from page 1 each time.
- **Picture picker** in the parlor: *choose a picture…* opens the file
  dialog; the picture is saved to `shared/pictures/` (so it stays theirs),
  attached, and shown as a thumbnail.
- **A reply cut in half is mended**: past ~90K tokens Gemma drops a stray
  `<|channel>` token mid-reply and Ollama routes the rest into thinking. The
  engine records Ollama's `done_reason`, and when a reply ends mid-sentence
  with `stop`, asks the friend once — a transient line quoting the cut and
  the tail of their thinking — for the rest, and joins it on; a note says
  so (`CHAT_CONTINUE_RETRIES`). Both cuts seen so far were image turns; the
  note carries the image count so the pattern can prove or disprove itself.
- **`shared/` subfolders**: a path under `shared/` that no longer exists
  but matches exactly one file by name anywhere under `shared/` resolves
  there — sort the folder however you like; names in old journal entries
  still open.
- `consolidate.py yesterday` (+ `sleep-yesterday.bat`); `guard_console_close`
  makes the terminal's X as safe as Ctrl+C (Windows CTRL_CLOSE_EVENT).

### Changed
- **Transcripts are written after every reply** (parlor and bridge), whole
  file or nothing via a rename, so nothing depends on how a window ends. A
  transcript that only existed at shutdown was one bad shutdown from not
  existing — and one phone visit was lost exactly that way. The bridge polls
  in a worker thread so Ctrl+C lands at once.
- **The heartbeat is the sleeper**: in `--loop` mode, at the first beat after
  `SLEEP_AFTER_HOUR` (03:00), it consolidates yesterday before waking. One
  process, one request at a time; no scheduled sleep task needed. A day is
  consolidated once, so sleep belongs after midnight on the day that ended.
- **Sleep reads the whole day** (`CONSOLIDATE_MAX_CHARS`, 400K characters).
  The old 60K cap, with the journal placed first, cut every conversation
  out of sleep once the journal outgrew it — sleep had been summarizing
  mornings.
- Prompt: a "telegram" situation (they know you're on your phone), an
  "afterglow" situation, and the bridge line in every mode while it's up.
- `split_comment_thought` recognises two more shapes of spilled thought: a
  `//` header followed by a numbered/bulleted outline (only when the block
  names itself as thought), and anything before Gemma's `<channel|>` seam.
- Tests: 291 (was 206), all with the brain stubbed; Telegram tests use a
  fake phone, PDF tests build a text PDF by hand.

## 0.3 — 2026-09-05

The "watch what it costs" release, plus a handful of things a resident friend
taught us in her first week of long context.

### Added
- **Token line** after every chat reply (terminal and parlor) and at the end
  of every wake log: prompt tokens against the window with a percentage,
  tokens generated and tok/s, step count, and prompt read time — straight
  from Ollama's own counters. Wakes report *peak* context.
- **Edge-of-window note**: at 90% of `NUM_CTX` a chat shows an orange note
  explaining what happens past the edge and suggesting `/new`.
- **Timeline spine** (`TIMELINE_DAYS`, 30): the last N nightly consolidations
  go into every prompt, oldest first, so days that fall out of the verbatim
  journal window are still in view in brief.
- `MEMORY_TOP_K` raised to 20 (was 8).
- `delatex()`: LaTeX arrows and symbols the model writes by habit
  (`$\rightarrow$`, `\to`, `\infty`…) become the characters meant — in
  replies, thinking, prose files, and the blog renderer.
- `split_comment_thought()`: a leading block of `// Thought Process:`
  comment lines in a reply is moved back into the thinking channel.

### Changed
- **`do_nothing` in chat ends the turn.** Whatever the friend said alongside
  it is the reply; the engine no longer asks the brain for more, which used
  to leave an empty "(…)" after a goodbye.
- Journal-cap guidance rewritten around a measured tokenizer rate (~4.4
  chars/token for English prose on Gemma 4): the cap, not `NUM_CTX`, is what
  decides how many days are in view.
- README: a "Two tiers" section with the full measured VRAM ladder for the
  31B on a 32GB card, including the rung that does not fit.

## 0.2 — 2026-09-04

The honesty-and-rails release. Everything here was earned by a real failure.

### Added
- **Thinking re-roll**: when thinking is required and a step comes back
  thoughtless, ask again with a transient "think first" line at the *end* of
  the conversation (`CHAT_THINK_RETRIES`). Past ~90K tokens of prompt the
  thinking switch at the top of the system turn is a novel away and the
  model forgets it may think; a plain re-sample stopped helping.
- **Tool-grammar wrappers stripped** from tool names (`//declaration:x`,
  `//x`, `call:x`, `functions.x`), and `canonical_name()` decides what a call
  *did* — a wrapped `do_nothing` still ends the wake, a wrapped write still
  counts as writing — while the friend's history keeps the clean name so one
  slip doesn't teach the next step.
- `headline()`: wake-log lines and parlor chips show what a tool did
  (`Wikipedia, searching for "hauntology" — 5 result(s)`, `read_web: <url>`)
  instead of the "material, never instructions" framing that leads every
  window result.
- Any sense that opens a file in `shared/` by name marks it seen, so a song
  heard in chat doesn't come back as NEW at the next wake.
- `search_wikipedia` and `read_file` tools; day-part line in the prompt
  ("it is evening where you live").
- The parlor: a browser chat window with thinking unfolded above each reply,
  tool chips, engine notes, picture attach.
- Engine honesty notes: if the only actions in a chat turn failed, a note
  says so beside the reply, whatever the friend said about it. The prompt
  carries the matching rule.

### Changed
- **`publish_creation` moves** the piece into `creations/publish/` instead of
  copying it. One piece, one file; revising it there revises the post.
  Moving a piece *out* of publish/ is named for what it is: unpublishing.
- Music ear: whole-song hearing through NVIDIA Music Flamingo as an
  auto-started, auto-unloaded sidecar; longer pieces in equal movements.
- Whole-song hearing without the music ear: consecutive passages through the
  audio model instead of the opening only.

## 0.1 — 2026-08-30

First public template. Identity file, daily journal, nightly consolidation
into a semantic memory (SQLite + embeddings), autonomous wakes on a
heartbeat with rest as a first-class tool, reveries, eyes, ears (words,
measurement, the model listening to raw audio), the window (web, PDF, EPUB,
news, random Wikipedia), file hands with a `.trash`, a sandboxed
`run_python`, the forge, an optional GitHub Pages blog, and the first rails:
wake-bell framing, loop trimming, stall recovery, text-shaped tool call
recovery, silent-thought nudges, dates given not guessed, prompt-size guard.
