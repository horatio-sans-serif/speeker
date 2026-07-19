# Background music per (queue, intention) + ducking

Date: 2026-06-17

## Context

Speeker speaks queued messages aloud. The user wants an ambient background-music
bed that plays under speech and **ducks** (drops in volume) while TTS is
actually speaking, with the track chosen **per (queue, intention)** — e.g. calm
music under SUCCESS messages, tense music under ERROR — resolved like the
existing `tone_rules`.

Constraint: TTS is played by spawning `afplay <file>` (blocking, sequential, no
live volume control). Ducking needs a music source whose volume can change
_while it plays_, so music runs through `mpv` (live volume via its JSON IPC),
not afplay. The OS mixes mpv (music) with afplay (speech).

## Decisions (confirmed with user)

- **Ducking: gate** — music dips to a fixed level while TTS speaks, ramps back
  when it stops. (Sidechain/level-following is a later upgrade needing an
  in-process mixer; out of scope.)
- **Bed lifetime: per-message, crossfade on change** — resolve the track per
  message; loop under that message (ducked); crossfade when the next message
  resolves to a different track; fade out at batch end.
- **Resolution: by (queue, intention)** like `tone_rules` (queue +2, interp +1,
  most specific wins; fallback = no music).
- **Source: one looping file per rule.**
- **Engine: mpv** (`brew install mpv`), optional. Off by default.

## Components

### `music.py` — resolution + config

- `resolve_music_track(queue, interpretation) -> Path | None`: iterate
  `config.music_rules`, score with the same logic as
  `tone_rules.resolve_tone_notes` (reuse `tone_rules._queue_match` /
  `_interp_match`); return the highest-scoring rule's `track` (expanded `~`) if
  the file exists, else `None`. A rule whose track is missing is treated as
  non-matching (a half-typed row never kills the bed).
- `get_music_config()` / `get_music_rules()` accessors.

### `music_engine.py` — mpv control (lives in the daemon)

A `MusicEngine` driving a pool of **two** mpv instances (A/B) for crossfades.
Each launched: `mpv --no-video --idle=yes --loop-file=inf --really-quiet
--input-ipc-server=<sock>`. Volume is set via IPC `{"command":["set_property",
"volume", <0..100>]}`.

- `available() -> bool`: `shutil.which("mpv")` is not None and `music.enabled`.
- `set_track(path | None)`: if same as current, no-op. Else load `path` in the
  idle instance at volume 0 (`loadfile`), then **crossfade**: ramp new up to the
  effective volume and old down to 0 over `crossfade_ms`, then stop the old.
  `None` fades the current out.
- `duck(on: bool)`: set the ducking flag; effective volume =
  `volume × (duck_level if on else 1.0) × 100`, applied to the active
  instance(s) with a short ramp (`fade_ms`).
- `stop(fade=True)`: fade both out; `shutdown()`: terminate mpv processes + remove
  sockets.
- **Transport is injectable** (`send=callable(instance, command_dict)`), defaulting
  to a real Unix-socket writer, so tests assert the volume/loadfile command
  sequence without a real mpv. Ramps are computed as N discrete volume steps.
- **Best-effort:** every public call is wrapped so an mpv/socket failure logs and
  returns — it must never block or break TTS. If `available()` is false, all
  calls are no-ops.

### `player.py` — integration

- Lazy-init one `MusicEngine` in `run_daemon` when `music.enabled` and mpv
  present; `shutdown()` in the daemon's `finally`.
- In `process_queue`, per utterance (where `line_interpretation` is already
  resolved): `engine.set_track(resolve_music_track(session_id, line_interp))`,
  then `engine.duck(True)` immediately before the `speak_text(...)` call and
  `engine.duck(False)` after (covers cue + speech + trailing tone).
- At batch end (after the loop): `engine.stop(fade=True)`.
- All engine calls guarded so they never raise into the playback path.

### `config.py`

```python
"music": {
    "enabled": False,       # master on/off
    "volume": 0.6,          # base music level 0..1
    "duck_level": 0.4,      # music level while speech plays 0..1
    "fade_ms": 400,         # duck/in/out ramp
    "crossfade_ms": 600,    # between tracks
},
"music_rules": [
    # {queue?: str, queue_regex?: bool, interpretation?: str, track: "/path"}
],
```

Accessors `get_music_config()` / `get_music_rules()`.

### Web UI — Settings → Music

- `GET/PUT /api/music` (enable + volume/duck/fade/crossfade).
- `GET/PUT /api/music-rules` (replace the rules list; validate track non-empty).
- `POST /api/music/try` (optional): preview a track for N seconds via the daemon.
- React `MusicSection`: global toggle + sliders, and a rules table
  (queue / interpretation / track / Try / Remove), mirroring the tone-rules
  editor. Added to the settings sub-tabs.

## Data flow

```
utterance ->
  track = resolve_music_track(queue, interpretation)
  engine.set_track(track)         # crossfade only if changed
  engine.duck(True)
  speak_text(...)                 # afplay; OS mixes with ducked mpv
  engine.duck(False)
batch end -> engine.stop(fade=True)
```

## Error handling

mpv absent / `music.enabled` false / track file missing / IPC error -> silent
no-op; TTS is never blocked. Engine failures are logged to the daemon log only.

## Testing

- `test_music.py`: `resolve_music_track` scoring (queue-only, interp-only,
  queue+interp, regex, missing-track-ignored, no-match -> None) — pure, mirrors
  the tone-rules tests.
- `test_music_engine.py`: with a fake transport, assert (a) `duck(True/False)`
  emits the expected volume ramp to the active instance; (b) `set_track` to a new
  path emits loadfile + a crossfade (new up / old down); (c) `available()` false
  -> no commands; (d) failures in the transport don't raise.
- `test_config`: music defaults. `test_web`: `/api/music` + `/api/music-rules`
  round-trip.
- Manual: with `mpv` installed and a rule configured, confirm the bed plays,
  ducks under speech, crossfades on intention change, and fades at batch end.

## Out of scope (v1)

Sidechain/level-following ducking (needs a sounddevice mixer with TTS routed
through it), folders/playlists per rule, beat-sync, non-message continuous beds.

## Verification

1. `uv run pytest` (new + existing).
2. `brew install mpv`; set `music.enabled`, add a rule with a local track;
   enqueue SUCCESS and ERROR messages for a queue and listen: music plays,
   ducks during speech, crossfades on intention change, fades at end.
