# Auto Queue Label — Design

**Date:** 2026-05-28
**Status:** Implemented.

## Goal

When a single, bare message comes out of the queue ("Claude finished"), the
listener should hear **which project** it belongs to. But only when the
context is unclear — back-to-back messages from the same project should not
be re-announced each time.

```
"Claude finished"                              → silent project context
"$Eb4 compass docs. Claude finished"           → labeled (after a gap, or new project)
"$Eb4 compass docs. Claude finished"
"Claude finished"                              → unlabeled (burst from same project)
"Claude finished"
   ⋯ 2+ minutes of silence ⋯
"$Eb4 compass docs. Claude finished"           → labeled again
```

## Background — why this was missing

The existing chain already supports caller-supplied titles end-to-end:

- The HTTP server's `format_with_title()` (`server.py:45-54`) prepends
  `$Eb4 {title}. ` to text when `/speak?title=…` or `/summarize?title=…` is
  called.
- The Stop-hook `summarize-response.py` passes `title=<project>` on the
  **success** path, so a summarized response sounds like
  `$Eb4 compass docs. <summary>`.

But two paths produced bare text with no title:

1. The hook's **fallback** branch (`speak_plain("Claude finished", project)`)
   when summarization returned nothing or the transcript was empty/short.
2. Any other client that enqueues without a `title=` query param — e.g.
   `mcp__tts__speak` with raw text, `speeker speak "…"`, ad-hoc curl calls.

The daemon then plays these bare. For single-message-single-queue batches the
player's `build_session_script` (player.py) **intentionally** drops even the
"For queue X, there is 1 message" header — that header was designed for the
multi-message case and reads as noise around a one-shot speak. So the
fallback path produced literally `"Claude finished"` with no project context.

Two facts this design depends on:

- The daemon already tracks `last_utterance_at` for the "This is Claude
  Code" intro re-announcement; the `playback_state` row holds it as a single
  global value.
- `extract_tone_tokens()` already strips a leading `$Note` from text before
  TTS. So a message starting with `$Eb4 …` is the recognized signal that
  the caller has already labeled it; we use that to skip double-prefixing.

## Decisions

1. **Decide per-utterance, in the daemon.** Make the player responsible for
   the auto-label decision rather than each caller. Callers can still pass
   their own title (the success path of the Stop hook continues to work) —
   the daemon detects an existing `$Note` prefix and yields. Rejected:
   pushing this into every client (the hook, the MCP tool, the CLI, future
   callers); each one would have to re-implement the gap+context rule.

2. **Combine time-gap AND queue-change as the trigger.** Either signal
   independently triggers a relabel. Rejected: time-gap alone (would skip
   project labels when two projects ping back-to-back — the listener would
   never know context switched). Rejected: queue-change alone (no relabel
   when you wander away from the desk for an hour and come back to the same
   project's next message).

3. **Persist `last_queue_id` alongside `last_utterance_at`.** Both signals
   live in the same `playback_state` row. Rejected: a separate table or a
   sidecar file — keeps the read atomic and avoids the "I saw the time but
   not the queue" race.

4. **Only auto-label single-message-only-session batches.** Multi-message
   batches already announce `"For queue X, there are N messages"` (which
   identifies the project). Adding another `$Eb4 X.` on top would
   double-announce. Rejected: blanket labeling — would clobber the existing
   header style.

5. **Configurable, default 120s.** The "back-to-back" window is exposed in
   `config.auto_label.quiet_threshold_seconds`. 120s matches roughly how
   long a related task tends to keep firing Stop events. Disabling is one
   flag (`auto_label.enabled = false`). Rejected: hard-coded threshold.

6. **Skip when text already begins with a `$Note` token.** Detected via
   `text.lstrip().startswith("$")`. Rejected: a metadata flag — the test is
   already a structural property of the text and free of caller
   coordination.

## Architecture

```
            ┌─────────────────────────────────────────────────────────────┐
            │ process_queue (player.py)                                    │
            │                                                              │
            │  read once:  last_utterance_at, last_queue_id ──── DB         │
            │                                                              │
            │  for each session in this batch:                              │
            │     ┌────────────────────────────────────────────────────┐   │
            │     │ compute_auto_label_prefix(                         │   │
            │     │   session_id, last_utterance_at, last_queue_id)    │   │
            │     │   → "$Eb4 compass docs"  or  None                  │   │
            │     └────────────────────────────────────────────────────┘   │
            │                  │                                            │
            │                  ▼                                            │
            │     build_session_script(... auto_label_prefix=...)           │
            │                  │                                            │
            │                  ▼                                            │
            │     speak_text(...)                                           │
            │                  │                                            │
            │     last_played_queue = session_id  ── in-memory              │
            │     last_utterance_at = now                                   │
            │                                                              │
            │  end-of-batch:  set_last_utterance_time(queue_id=last) → DB   │
            └─────────────────────────────────────────────────────────────┘
```

**Per-batch, in-memory state.** `process_queue` seeds `last_utterance_at` /
`last_played_queue` from the DB _once_ per batch. It advances both in memory
between sessions. So a batch with three queues spoken in sequence — A then B
then A — re-labels on A→B (different queue), on B→A (different queue), but
**not** on a same-queue sequence within the threshold.

**One DB write per batch.** Only the final `set_last_utterance_time` writes
to disk, with the most-recently-spoken queue id. Intermediate sessions don't
hammer the row.

**Decision matrix** (config enabled, named queue, threshold = 120s):

| `last_utterance_at` | `last_queue_id` | This session's queue | Prefix?                        |
| ------------------- | --------------- | -------------------- | ------------------------------ |
| `None`              | —               | `compass-docs`       | ✅ first-ever utterance        |
| recent (≤120s)      | `compass-docs`  | `compass-docs`       | ❌ back-to-back same queue     |
| recent (≤120s)      | `audio-speeker` | `compass-docs`       | ✅ context switch              |
| old (>120s)         | `compass-docs`  | `compass-docs`       | ✅ post-silence                |
| any                 | any             | `default`            | ❌ no spoken title             |
| any                 | any             | any                  | ❌ if text starts with `$Note` |

## Public surface

### Config (`config.json`)

```json
{
  "auto_label": {
    "enabled": true,
    "quiet_threshold_seconds": 120,
    "tone": "$Eb4"
  }
}
```

- `enabled` (bool, default `true`): master switch.
- `quiet_threshold_seconds` (float, default `120`): silence before a same-queue
  relabel.
- `tone` (string, default `"$Eb4"`): tone token rendered before the spoken
  title. Same syntax as `extract_tone_tokens` accepts.

### New functions

- `speeker.config.get_auto_label_config() -> dict`
- `speeker.queue_db.get_spoken_queue_title(queue_id) -> str | None`
  Friendly form (`compass-docs` → `compass docs`); `None` for `default`,
  empty, or whitespace-only ids.
- `speeker.queue_db.get_last_played_queue() -> str | None`
- `speeker.queue_db.set_last_utterance_time(queue_id=None)` — extended to
  optionally record the queue id atomically with the time.
- `speeker.player.compute_auto_label_prefix(session_id, last_utterance_at,
last_queue_id) -> str | None`

### Schema migration

`playback_state` gains `last_queue_id TEXT`. `_init_db()` runs an `ALTER
TABLE ADD COLUMN` once on pre-existing databases; new installs get the
column from the `CREATE TABLE` directly.

## What is _not_ changed

- Caller-supplied `?title=` paths are untouched — they continue producing
  `$Eb4 <title>. <text>` server-side. The daemon notices the leading `$` and
  yields, so behavior is identical to before.
- Multi-message batches still announce `"For queue X, there are N messages"`
  through `get_queue_label()`. The legacy 8-char truncation in that label
  was left alone (it's only a few seconds of speech and any change would
  cascade through a dozen tests).

## Follow-up: filler elimination (2026-05-28)

After the auto-label landed, two pieces of generic filler still grated:

- **"Claude finished"** — the Stop hook's fallback when summarization
  failed.
- **"That is all."** — the daemon's spoken outro for multi-message batches.

Both are now gone:

- The Stop hook (`~/.claude/hooks/summarize-response.py`) replaced
  `speak_plain("Claude finished", project)` with a `speak_label_only(project)`
  that enqueues exactly `"$Eb4 <spoken title>"`. The player extracts the
  `$Eb4` tone, plays it, and speaks the project name. One tone, one project
  name, nothing else. The auto-label path detects the leading `$` and yields,
  so the prefix is not doubled.
- The daemon's outro (`player.py::process_queue`) dropped the
  `speak_text("That is all.", ...)` call. The descending outro chord
  (`get_outro_sound()`) still plays — it was always the actual "end of batch"
  signal; the spoken phrase added nothing.

Design rule: when no summary is available, **the audio cue is the message**.
A tone plus the project name carries enough information ("something happened
in compass-docs") without spoken filler that the listener has to mentally
discard every time.

## Follow-up: buried-tone regression + meta-commentary purge (2026-05-28)

Filler removal exposed a latent bug. After items started arriving with
`$Note <title>.` prefixes (from both the server's `format_with_title` AND
the hook's new `speak_label_only`), multi-message batches voiced the tone
token as the _letters_ "EEB 4". The user heard:

> "There are 8 messages in queue progress: Next, EEB 4 progress. Next, EEB 4
> progress reporter, etc."

Three things were wrong:

1. **`extract_tone_tokens()` is line-anchored.** Its regex starts with
   `^\s*\$` so it only matches `$Note` at the _start_ of a string. The
   legacy `build_session_script` prepended `"Next: "` / `"First, from N
minutes ago: "` to each item in multi-message batches, burying any
   leading `$Note` token mid-line. The TTS engine then read it aloud.
2. **The count header was useless meta-commentary.** "For queue X, there
   are 8 messages." existed for the days before the auto-label — now
   redundant once every item is self-labeling.
3. **macOS notifications flooded.** The hook's `notify()` fired once per
   successful summary with no rate limit.

Fixes:

- **`build_session_script` rewritten.** One item -> one spoken line. No
  count header, no `First:/Next:/Last:` framing. Items starting with
  `$Note` are spoken **verbatim** so the tone reaches
  `extract_tone_tokens`. The auto-label prefix attaches only to the
  _first_ item that doesn't already establish project context (a
  caller-prefixed item consumes the "context established" slot too, so a
  bare follow-up doesn't get a duplicate label).
- **`process_queue` loop simplified.** With no header line, `item_idx ==
line_idx` always — the old `item_idx = line_idx - 1` offset that
  accounted for the header is removed.
- **Hook's `notify()` deleted.** Spoken stream is the only channel. The
  `summary` return value is now only used to decide between the summary
  path (server speaks it) and the label-only fallback.
- **`speak_label_only` adds a trailing period.** `"$Eb4 progress
reporter."` instead of `"$Eb4 progress reporter"`. Matches the
  `format_with_title` shape so all `$Note`-prefixed messages have the
  uniform `$Note <title>.[ body]` structure.

Regression tests added (`tests/test_player.py`):

- `test_caller_prefixed_item_keeps_tone_at_line_start` — asserts each
  line in a multi-item batch with caller prefixes still begins with `$Eb4`.
- `test_no_count_header_for_multiple_messages` — no "there are N
  messages" anywhere in the script.
- `test_no_first_next_framing` — items are spoken verbatim.
- `test_auto_label_only_first_bare_item_in_session` — multi-bare-item
  sessions label only the first item.
- `test_caller_prefixed_item_consumes_label_slot_for_following_bare` — a
  caller-prefixed item suppresses the auto-label for a subsequent bare
  item in the same session.

The framing-removal does affect a behavior that _was_ useful: if a session
has many bare messages from a brief outage, the listener now hears them
back-to-back with no "Next:" pacing. Acceptable trade-off — the alternative
re-burying of tone tokens is worse.

## Verification

- `tests/test_queue_db.py::TestGetSpokenQueueTitle` — friendly form cases.
- `tests/test_queue_db.py::TestUtteranceTime` — `last_queue_id` round-trip,
  no-clobber on `set_last_utterance_time()` without a queue id, and
  overwrite on a subsequent queued-id call.
- `tests/test_player.py::TestComputeAutoLabelPrefix` — full decision matrix
  (disabled, default queue, first-time, burst, context switch, post-silence,
  custom tone).
- `tests/test_player.py::TestBuildSessionScriptAutoLabel` — prefix is
  prepended for single-message-only-session, skipped when text already
  starts with `$`, ignored for multi-message batches.
