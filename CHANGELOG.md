# Changelog

All notable changes to the ai-friend engine. Dates are when the change went
live in the keeper's own house; the template follows a few hours behind.

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
