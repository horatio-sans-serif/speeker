# Interpretation Cues (SUCCESS / ERROR / custom) — Design

**Date:** 2026-05-27
**Status:** Implemented (commit `4dfc383`). Written retroactively from the
brainstorming session; the cues were tuned interactively by ear.

## Goal

Let an utterance carry an **interpretation** — a named outcome such as
`SUCCESS`, `ERROR`, or any custom label — and play a short cue before the
speech to signal that outcome. Make it generic: configuration maps each
interpretation name to an **indication** that is either a sequence of notes or
a sound file. Expose it on the CLI, the HTTP server, and the MCP tool so Claude
Code can mark how a task turned out.

```bash
speeker speak --interpretation SUCCESS "All tests passed"
speeker speak --interpretation ERROR   "Build failed: 3 errors"
```

```python
# MCP
speak(text="Deploy complete", interpretation="SUCCESS")
```

## Background — current architecture

Two generation paths existed, and the relevant cue logic lived in only one:

- **Live path (server + MCP):** `POST /speak` → `enqueue(text, metadata)` into
  the SQLite queue → the **player daemon** (`process_queue`) generates TTS at
  playback time and plays audio. `$Note` tone tokens are handled here
  (`extract_tone_tokens`, `generate_combined_tones_from_tokens`).
- **CLI direct path:** `speeker speak` generated audio synchronously and wrote
  the WAV path to a file-based queue (`data_dir()/queue`) via
  `queue_for_playback`. **Nothing in the current daemon consumed that file** —
  only `cmd_status` read it (to count length). So `speeker speak` in default
  mode produced no sound through the daemon.

Two facts this design depends on:

- Note synthesis used a single shared `duration` for every note. The SUCCESS
  cue (a short note stepping up to a ringing one) needs **per-note durations**.
- `metadata` is stored as JSON and already read by `process_queue` (engine,
  voice, polly_engine, ssml). Adding `interpretation` is purely additive — no
  schema or migration change — and rides the MCP → HTTP → queue → daemon chain
  for free.

## Decisions

Three choices were resolved up front (they change the implementation):

1. **Playback path:** Route the CLI's default `speeker speak` through the live
   SQLite queue (the same path the server uses), with the interpretation stored
   in metadata and the daemon playing the cue. This unifies the surfaces, makes
   `--interpretation` work everywhere, and fixes the orphaned file queue.
   Rejected: in-process CLI playback (leaves the orphaned queue broken; splits
   playback across two mechanisms).
2. **Duration model:** Explicit **seconds** per note (`{pitch, seconds}`). No
   tempo/BPM concept.
3. **Scope:** CLI **and** HTTP server **and** MCP. Interpretation in queue
   metadata makes the server/MCP support nearly free.

Cue placement: the cue plays first, blocks until it finishes, pauses
(`pause_after_seconds`), then the utterance speaks. Per item, after any queue
header line.

## The cues (built-in defaults)

Tuned by ear during the session:

- **SUCCESS** — `Eb3` (0.15s) → `G#3` (0.9s ring). A quick step up to a held
  major-third-ish note; reads as bright/affirmative. (Original Brahms-Lullaby
  shape `Eb4 Eb4 Eb4 Gb4` was rejected for sounding like a known tune.)
- **ERROR** — `Eb4` (0.3s) → `D4` (0.2s) → `Bb2` (0.2s) → `Bb2` (0.2s). A
  half-step down then a two-octave drop to a doubled low Bb; lands like a thud.

## Architecture

### `config.py`

New `interpretations` section in `DEFAULT_CONFIG`:

```json
{
  "interpretations": {
    "pause_after_seconds": 0.3,
    "map": {
      "SUCCESS": {
        "type": "notes",
        "notes": [
          { "pitch": "Eb3", "seconds": 0.15 },
          { "pitch": "G#3", "seconds": 0.9 }
        ]
      },
      "ERROR": {
        "type": "notes",
        "notes": [
          { "pitch": "Eb4", "seconds": 0.3 },
          { "pitch": "D4", "seconds": 0.2 },
          { "pitch": "Bb2", "seconds": 0.2 },
          { "pitch": "Bb2", "seconds": 0.2 }
        ]
      }
    }
  }
}
```

Plus `get_interpretations_config()`. A `sound_file` indication is
`{"type": "sound_file", "path": "/abs/path"}` (`~` expanded).

### `interpretations.py` (new, pure — no audio I/O)

- `BUILTIN_INTERPRETATIONS` for `SUCCESS`/`ERROR`, layered **under** the config
  `map` (`effective_map = {**BUILTIN, **config_map}`). Because the config merge
  is shallow (one level), a user who defines `map` would otherwise erase the
  defaults; keeping built-ins in code guarantees they remain resolvable, while a
  same-named config entry overrides them.
- `resolve_interpretation(name)` — exact match, then case-insensitive.
- `interpretation_names()`, `is_valid_interpretation(name)`,
  `pause_after_seconds()`.
- `parse_pitch("Eb3") -> ("eb", 3)` and
  `notes_to_spec(indication) -> [(note, octave, seconds), ...]` (skips
  malformed pitches; defaults seconds to 0.2).

### `player.py` (daemon)

- `synthesize_note_cue(name, spec)` — per-note durations via the `tones` mixer
  (same track config as `$Note` synthesis: SINE_WAVE, vibrato 5.5, attack 0.01,
  decay 0.3), cached by `name + md5(spec)` so editing the config regenerates.
- `render_interpretation_cue(name)` — notes → synthesized WAV; sound_file →
  validated path. Unknown name, unknown type, or missing file → `None` + warning
  (a misconfigured cue never aborts playback).
- `play_interpretation_cue(name)` — render, `play_audio` (blocks), then
  `time.sleep(pause_after_seconds())`.
- `process_queue` reads `meta.get("interpretation")` and calls
  `play_interpretation_cue` before the item's `speak_text`.

### Surfaces

- **CLI:** `speeker speak --interpretation NAME`. Validated in `cmd_speak`
  (`_validate_interpretation` lists known names on error). `speak_text` default
  mode now `enqueue`s text + metadata (`engine`, `voice`, `polly_engine`,
  `ssml`, `interpretation`) and starts the player; `--stdout`/`--no-play` stay
  synchronous and ignore interpretation. `cmd_status` reports the SQLite
  pending count. Daemon preprocessing parity confirmed (same
  `preprocess_for_tts`/`prepare_payload`), so routing through the queue does not
  change pronunciation.
- **HTTP `/speak`:** top-level `interpretation` field (or `metadata`),
  validated → 400 + known names on unknown. `HTTPException` is re-raised so it
  is not flattened into a 200 error body by the handler's `except Exception`.
- **MCP `speak`:** `interpretation` argument forwarded as queue metadata.

## Error handling

- Unknown name at CLI / HTTP / MCP edge → reject with the list of valid names.
- Missing sound file or unknown indication type at play time → warn, skip cue,
  still speak.
- Enqueue failure on the CLI → reported, returns non-zero.

## Testing

- `test_interpretations.py` — built-ins present and correct; config overlay
  (custom entry keeps built-ins; same-named entry overrides); case-insensitive
  resolution; `pause_after_seconds` default/override; pitch parsing;
  `notes_to_spec` skipping/defaults.
- `test_player.py` — render unknown → None; sound_file present/missing; cue
  plays then pauses; no-op when unresolved.
- `test_cli.py` — default speak enqueues with metadata; interpretation rides in
  metadata; absent when not requested; enqueue error handled; `cmd_status`
  reports pending count.
- `test_server.py` — interpretation field flows into metadata; unknown → 400,
  no enqueue.
- `test_mcp.py` — interpretation forwarded; omitted when not given.

Full suite: 769 passed, 3 skipped, ruff clean. End-to-end verified that a
`SUCCESS`-tagged queue item fires its cue immediately before the speech in
`process_queue`.
