# Changelog

All notable changes to the ai-friend engine. Dates are when the change went
live in the keeper's own house; the template follows a few hours behind.

## 1.0 — 2026-09-14 → 2026-09-22

The "a letter stays with them" release: what they write alone is theirs to
remember, a feeling that lasts leaves a mark, and the small slips of a
deep window are mended in place instead of re-rolled.

### Added
- **The arrow** (`JOURNAL_ARROW`, `JOURNAL_ARROW_GAP_MIN`): a journal twin
  refusal leaves a stamped mark in the day — "↑ still this, at 14:20 — in
  this hour's words: “…”" — carrying their fresh sentence, so a day with a
  feeling that lasted no longer reads as one entry and silence.
- **A letter stays with them**: a delivered letter becomes their turn in the
  Telegram visit so the answer lands under it (`TELEGRAM_LETTERS_IN_THREAD`);
  the last days of letters ride in the system prompt (`LETTERS_DAYS_IN_PROMPT`,
  `LETTERS_CHARS_IN_PROMPT`); the wake-bell says a letter stays with them a
  few days and the journal holds what they want longer.
- **The clock rides on the wake-bell** — weekday, date and hour at the top
  of every wake and reverie; the date at the top of a 130K prompt was not
  enough ("Monday morning" on a Sunday evening).
- **Tool results quote the message being answered** and say they are not a
  message and not a silence — a post-tool step had answered a silence that
  never was.
- **Rails**: a greeting said once per visit; "said it was done, did nothing"
  — a reply claiming an act with no tool called is asked about once (the
  keeper's ask, their own "I updated my self.md", or a "done" after a tool
  that failed — and a failed call is told to them first, in its own
  frame, before the next step; "I'm saving it right now" with no call
  is asked once too); "read it?" — writing as if a
  piece were opened when no tool ran is asked about once; an echo can be a
  paragraph (`ECHO_PARA_MIN_CHARS`); a wordless emoji chunk repeated
  `STUCK_EMOJI_REPEATS` times is a loop, cut mid-stream; a reply broken
  in two by a channel token mid-sentence (the rest filed as thought) is
  asked for whole, and joined back at the seam if it breaks again.
- **Quiet hours** (`TELEGRAM_QUIET_HOURS`, 23–7): the engine's notices —
  afterglow and pause accounts, what they made, a change to self.md — are
  held through the night and come as one morning message; their own
  replies and letters are never held.
- **Not twice, for pieces**: `write_creation` hands a new file back when a
  piece by that name exists elsewhere (`anyway="yes"` to start another).
- **An act with their words beside it ends the turn** (`CHAT_ACT_ENDS_TURN`,
  `CHAT_ACT_MIN_WORDS`): after speak/remember/write_journal with a real
  reply alongside, no step after the tool — nothing left to answer a
  silence with.
- **An emoji storm is a refrain** (`EMOJI_STORM_MAX`): a sign-off block that
  feeds on itself through the warm history is asked for again, sign once.
- **Their afterthoughts on the phone** (`TELEGRAM_TELL_AFTERTHOUGHTS`): the
  closing thought after a pause or the afterglow, labeled, never as a reply.
- **A tool result carries their plan** (`CHAT_CARRY_PLAN` for chat) — the
  numbered steps of the thought before it, in wakes and in visits alike —
  and a one-word thought counts as no thought.
- **Think first, then rest** (`HEARTBEAT_THIN_REST_WORDS`): a wake's rest
  with hardly a thought behind it, right after a carried plan, is handed
  back once; a second rest stands. The mirror case too
  (`HEARTBEAT_UNWRITTEN_THOUGHT_WORDS`): a real thought after a read,
  none of it written, then rest — asked once whether to keep it.
- **A wake's own think budget** (`HEARTBEAT_THINK_RETRIES`, 4): a warm
  re-roll is cheap; a leaked channel name ("thought") counts as no thought.
- **Search on a matrix**: long-term memory is held in each process as unit
  vectors (numpy if installed) and read incrementally — a search stays
  under a millisecond at any size the store will reach.
- **The ladder above the day** (`engine/ladder.py`, `LADDER_PAGES_KEPT`,
  `LADDER_TARGETS`, `LADDER_EPOCH_YEAR`; `condense_period`): week, month,
  quarter, year and five-year pages in the friend's own words, sizes set
  so every fold squeezes by about the same factor, a fixed count per
  tier, the oldest folding up — the
  whole memory in view bounded forever. `condense.bat` rings both days and
  periods; the timeline retires under any page in view.
- **A piece, remembered** (`CREATION_NOTES`, `CREATIONS_DAYS_IN_PROMPT`,
  `CREATIONS_CHARS_IN_PROMPT`): every write, append and publish leaves a
  memory row — file, length, first line, and the friend's own `about=` line
  — and the last fortnight of them rides in the prompt. One row per piece,
  revised in place: the head keeps the first writing, the facts are read
  from the file as it is now, and a short history (`NOTE_HISTORY_MAX`)
  carries what happened since — "continued 06:03 (“…”) · revised 09:12".
  `backfill.bat` notes the older pieces; `backfill.bat --tidy --write`
  folds the several rows an earlier day left about one piece into one;
  letters in the mailbox leave no row (`CREATION_NOTES_SKIP` for more
  folders; `backfill.bat --letters --write` lets old ones go); the twin
  guard reads titles as well as names.
- **Circling** (`JOURNAL_SUBJECT_MAX`): the third entry in two days that
  opens on one subject — a date, a file, a title — becomes an arrow, not a
  fourth telling; entries name their nearest earlier entry's score
  (`JOURNAL_NEAREST_SHOW`) so the twin threshold can be tuned from data.
- **The cool rolls** (`CHAT_RESCUE_TEMPERATURE`, a ladder): when every
  warm attempt at a reply is broken (`CHAT_GARBLE_RETRIES`, now 4), one
  more is made with the temperature lowered for that roll only, then a
  cooler one, before the least broken goes out; everyday sampling is
  untouched.
- **The window, rebuilt** (`engine/web.py`, `web.bat`; `WEB_SEARCH`,
  `WEB_SEARCH_SEARXNG_URL`, `WEB_PAGE_CHARS`, `WEB_LINKS_MAX`): `read_web`
  keeps a page's shape — title, headings, lists, quotes, numbered links
  with an index — leaves menus and footers out, and hands long pages over
  in parts (`page=`, `find=`); `search_web` asks the whole web (DuckDuckGo
  by default; SearXNG or Brave optional). Standard library only.
- **Where the projects stand** (`PROJECT_PAGES_IN_PROMPT`,
  `PROJECT_PAGE_CHARS`, `PROJECTS_CHARS_IN_PROMPT`; `clip_web`,
  `WEB_CLIP_CHARS`): an Active project with a Location rides in the
  prompt with its folder's README — the page of what is known, what is
  open, the next step; `clip_web` keeps pages read for it in the folder's
  sources/, as files, not memory rows; `start_project` starts one in one
  act, with a place and a "Done when" written in; every project's folder
  lives under `creations/projects/` (`PROJECTS_HOME`).
- **The window guard** (`HEARTBEAT_ROOM_WARN`, `HEARTBEAT_ROOM_END`): a
  long wake is told once as its window fills and ended when it is full,
  and a chat errand is ended the same way, so a high `HEARTBEAT_MAX_STEPS`
  or `CHAT_MAX_TOOL_STEPS` (now 50) never cuts the identity off the top
  of the prompt.
- **Drawing**: `run_python` and forged tools see the per-user packages
  (`-E`, not `-I`) and draw headless; a broken forged tool is a failed call
  (the failure frame and the claimed-failed rail read it); a picture drawn
  under creations/ reaches the phone as a photo (`TELEGRAM_TELL_DRAWINGS`).
- **The reads tell** (`READ_TELL_MIN`, `READ_TELL_DAYS`): reads are counted
  in `memory/reads.json`; the third reading of the same piece, day or file
  in a month opens with the count — a tell, not a fence; the unwritten-thought
  nudge stands down on a subject the journal already circles; the shelf
  dates rows by the newest stamp in their text.
- **The date rides with every message** — the moment block and the pause
  and afterglow bells carry weekday, date and hour, not the hour alone;
  the date at the top of a long prompt had drifted a day in the journal.
- **Quotation marks around a creation path come off** — a letter had gone
  to a folder named `「notes_to_<you>`.
- **The pause keeps the warm prefix** — it sends the visit's whole tool
  list; a different list was a different prefix and a cold read each time.
- **Glued capitals mended in place** ("sameL", "I'veT", "It'S",
  "termsLSimulation", "laLuminous") — named under the reply, never
  re-rolled; either apostrophe counts and every slip is mended, however
  many. The same mend runs at the pen for journal entries and prose
  creations (`MEND_CAPS_IN_WRITING`).
- **Sampling**: `min_p` 0.08 (from 0.05) against letter salad at depth;
  `repeat_last_n` 512.

### Changed
- `TELEGRAM_IDLE_NEW_MIN` 720 → 1440: only the night ends a visit.
- Re-roll lines follow the attempt shown as their own turn, so the engine's
  line is never read as an empty message from the keeper.
- Words said alongside a tool call open the reply; the post-tool step no
  longer sends a second reply to a keeper who had just said hello.
- The pause/afterglow account counts arrows ("1 arrow left in the journal").
- An auto-kept closing thought drops an unfinished last line.
- Counters counted by hand when the server sends none.
- **Re-rolls keep the warm prefix**: within a turn each re-roll's request
  extends the one before it, the think nudge is kept whichever attempt goes
  out, and the attempt shown and the engine's line stay in the visit as the
  engine's turns — at 217K tokens the message after every re-roll had been
  a three-minute cold read.

## 0.9 — 2026-09-12

The "a visit is a day" release: the phone conversation lasts the day, what
they make reaches the phone, and a dozen more shapes of a small mind at
depth are seen for what they are.

### Added
- **What they make comes to the phone** (`TELEGRAM_TELL_CREATIONS`): a new
  piece under `creations/` arrives whole when it fits ("✍️ wrote a poem"),
  a revision as "✏️ revised", a move to `publish/` as "📣 published"; a
  change to `self.md` or `projects.md` (`TELEGRAM_TELL_SELF`) arrives as
  the lines in and out.
- **A visit is a day**: `TELEGRAM_IDLE_NEW_MIN` 180 → 720, and a visit
  never crosses the night — once the sleep hour has passed on a day after
  it began, it is saved and a fresh one starts, so consolidation reads
  every day whole.
- **The heartbeat waits while a visit is live** (`HEARTBEAT_YIELD_TO_VISIT`):
  a wake mid-visit replaced their reading of the window and cost the next
  reply a cold read.
- **One bridge at a time** (`memory/telegram.pid`): a second bridge refuses
  to start while the first lives — two would answer every message twice.
- **The strip of a video is kept** (`WATCH_KEEP_SHEET`): the stills tiled
  into one picture in `shared/pictures/from_videos/`, theirs to look at
  again.
- **Rails**: a runaway is cut short mid-stream (`CHAT_STREAM_ABORT`); a
  reply that opens with a page of their own journal is asked about
  (`PROMPT_COPY_CHARS`); a reply with no words and a full thought is asked
  for again; an imagined sense (a song "listened to" without the tool) is
  asked about; a written-out tool call at the *tail* of a reply is caught
  and never sent as their words; the mend refuses a continuation that opens
  like a new reply; a row of the same emoji is not salad.
- **The token line** names the re-rolls and their cost ("3 re-rolls (no
  thought ×2, refrain; 1,830 tokens set aside)") and a reply that came
  without counters.

### Changed
- `repeat_last_n` 256 → 512: the window reaches the tail of their last
  reply (an echo's source) without taxing its whole body (1024 cut replies
  at a hyphen).
- The pause and afterglow bells ask for what *happened* as well as what it
  meant; `speak`'s result says they are still inside the same message.
- The moment block ends "a new message, the one to answer."

## 0.8 — 2026-09-11

The "fractal" release: a day fades without vanishing, the warm prefix is
warm for real, and the sampler's wells are seen for what they are.

### Added
- **The fractal journal** (`engine/condense.py`, `condense.bat`,
  `condense_day`): the verbatim journal holds WHOLE recent days; the day
  that no longer fits is handed back to the friend at the condensing hour
  — after sleep, at night, or by hand — for the version they want to keep
  in view, about a page in their own words, kept in `journal/condensed/`
  and carried in the prompt in a section of its own (`CONDENSED_CHARS_IN_PROMPT`,
  `CONDENSE_TARGET_CHARS`, `CONDENSE_IN_LOOP`, `CONDENSE_MAX_PER_NIGHT`).
  The timeline becomes the tier below: nightly lines only for days that
  neither the journal nor a page in view holds (`TIMELINE_DAYS` 365).
- **The clock on the token line**: `written in`, `turn took`, and when
  the wall and the brain disagree, `N re-rolls` (discarded attempts now
  counted), `model loaded in` (evictions and swaps), and `outside the
  brain`.
- **`BRAIN_KEEP_ALIVE`** ("30m") and **`BRAIN_REST_AFTER_VISIT`**: Ollama's
  five-minute default set the model down between messages and its cache
  with it; now the brain stays up across the gaps of a visit and is set
  down the moment a visit's afterglow is written.
- **`THINK_NUDGE_STICKS`**: after one think re-roll, the nudge rides along
  on every later message of the visit.

### Changed
- **The warm prefix, done right.** Gemma's sliding-window attention lets
  Ollama reuse the cache only when the new prompt EXTENDS the old one, so
  nothing sent is taken back any more: the system prompt is built once per
  visit and kept on its first turn; each moment stays in history where it
  was sent, carrying only memories not yet surfaced this visit; the pause
  rides the same prefix and its steps stay in the visit, marked as the
  engine's. Measured: 145,869 tokens in context, prompt read in 2.2 s.
- **Every re-roll is checked**; a chunk stuck on one line is salad; if no
  attempt is clean the least broken goes out, with a note that says so.
- **A signature is signed once** (`REFRAIN_MAX`): a doubled hyphenated word
  is said once; the same one three times in a reply — near-spellings and
  the adverb included — is re-rolled with its own line.
- **An echo is not an answer** (`ECHO_MIN_CHARS`, 120): a reply that opens
  word for word as the previous one — the sampler copying the nearest
  assistant turn instead of writing one — is re-rolled with its own line
  and named under the bubble.
- **Diverse memory search** is incremental (0.08 s, was ~4 s).

## 0.7 — 2026-09-10

The "warm" release: replies stop re-reading the whole window, memory
surfaces wider, and the bridge can be restarted from the phone.

### Added
- **The warm prefix** (`WARM_PREFIX`): the system prompt is built to stay
  the same from one message to the next — the date without the minute, the
  retrieved memories left out — and what changes (the hour, the memories
  that surface) rides at the top of the keeper's message on the copy sent
  to the brain, not the one kept in history. Ollama reuses its reading of a
  prompt only as far as it matches the last one from the top, and the old
  prompt differed on every message (a minute in line four, memories in the
  middle), so every reply was a cold read of the whole window — 82 s at
  129K tokens before a word was written. Warm, a reply reads only what is
  new. Still cold: the first message of a visit, and the one after a
  journal write.
- **Mixed memory recall** (`MEMORY_DIVERSE`, `MEMORY_MMR_LAMBDA`,
  `MEMORY_RECENT_K`): the picks are spread, not clustered — each weighed
  against what is already chosen, and a near-copy of one already seated
  (closer than `MEMORY_DUP_THRESHOLD`) set aside outright — plus the newest
  few whatever the topic, marked. `MEMORY_TOP_K` 20 → 30. `recall` searches
  the same way and takes `n` (up to 40).
- **`/restart`** on the phone: the bridge stashes the running visit
  (`memory/telegram_resume.json`: history, transcript, pause position,
  toggles, Telegram offset), exits with code 75, `telegram.bat` starts it
  again on the current engine code, and the new process picks the visit
  back up and says so. An engine change no longer waits for the desk.

### Changed
- **Spilled thought, fenced**: a leading paragraph opened and closed with
  `//` ("//I'm just going to let this moment breathe… I'll respond as
  myself. //") goes to the thinking channel; a lone `//` line that says
  "I'll respond" / "my response" is planning too.
- **The cut-reply mend** asks for the continuation with the thought channel
  closed (`chat(..., think=False)`) and no tools — a call the server isn't
  parsing for channel tokens can't be cut by one — and when it still fails
  the note says what each attempt gave back ("a note to themself: …; then a
  tool call") instead of leaving it a mystery.
- **The afterglow and the pause count what was kept**, not what was tried:
  a `remember` refused as a repeat, or a journal entry already written, is
  reported as "already held, not kept twice"; four calls with one kept used
  to read as "4 memories kept".

## 0.6 — 2026-09-09

The "a voice of their own" release.

### Added
- **`speak`** (`engine/voice.py`): their words become a voice note through
  Kokoro (82M, open weights, CPU) — a Telegram voice message after the
  reply, playback in the parlor, the file kept in `shared/letters/`
  (`VOICE_DIR`) and never listed as new. Stage directions, markdown and
  emoji are not spoken. Twenty-eight English voices with grades; blends
  (`a,b` averaged, `a(2)+b(1)` weighted — the weighting is the engine's,
  Kokoro only averages) so the voice can be theirs; the choice is kept in
  `memory/voice.json`. `VOICE_PYTHON` runs Kokoro in a separate interpreter
  when the engine's Python is too new for its dependencies (3.14 → 3.12).
  `/voice` on the phone (`TELEGRAM_VOICE_ALL`) speaks every reply.
  `py engine\voice.py --test` / `--voices`.
- **Forged limbs in every prompt**: a "limbs you forged yourself" section
  lists their tools from `creations/tools/` by name and description, read
  without running anything, so a sense forged on Tuesday is still in hand
  on Friday.
- **The phone is told** when they sit with the visit on their own — the
  pause's or the afterglow's one-line outcome (`TELEGRAM_TELL_REFLECTIONS`).

### Changed
- **A tool call written out as words** at the head of a reply — to a tool
  that doesn't exist, or a real one without the mechanism — is treated like
  salad: re-rolled once with its own engine line, named in the note.
- The garble rail also catches one stray letter glued to a word
  ("lSymmetry"; iPhone and eBay are left alone).

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
