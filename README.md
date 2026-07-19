# Speeker

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: PolyForm Noncommercial](https://img.shields.io/badge/License-PolyForm%20NC-blue.svg)](LICENSE)
[![Tests](https://github.com/horatio-sans-serif/speeker/actions/workflows/test.yml/badge.svg)](https://github.com/horatio-sans-serif/speeker/actions/workflows/test.yml)
[![Coverage](https://img.shields.io/badge/coverage-89%25-brightgreen.svg)](https://github.com/horatio-sans-serif/speeker)

A text-to-speech system with HTTP API, web UI, and CLI. Queue text for playback with metadata, search history, and configurable voices.

## Features

- **HTTP API**: Queue text via REST endpoints with metadata support
- **Web UI**: View queue history, play audio, search messages
- **Multiple TTS engines**: pocket-tts (fast), kokoro (higher quality), and Amazon Polly (cloud, SSML-native)
- **SSML support**: native on Polly; best-effort emulation on local engines
- **Daemon mode**: Low-latency playback with warm TTS model
- **Metadata**: Attach arbitrary key-value data to messages
- **Search**: Fuzzy text search or semantic search with embeddings
- **Per-session settings**: Speed, voice, intro/outro sounds
- **Background music**: A music bed per (queue, interpretation) that ducks under speech (needs `mpv`)

## Installation

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv tool install speeker
```

For semantic search support:

```bash
uv tool install speeker[semantic]
```

## Quick Start

```bash
# Start the server
speeker-server

# Queue text via API
curl -X POST http://127.0.0.1:7849/speak \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, world!"}'

# View web UI
open http://127.0.0.1:7849/
```

## HTTP API

### POST /speak

Queue text for TTS playback.

```bash
# Simple text
curl -X POST http://127.0.0.1:7849/speak \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, world!"}'

# With metadata via JSON body
curl -X POST http://127.0.0.1:7849/speak \
  -H "Content-Type: application/json" \
  -d '{"text": "Task complete", "metadata": {"source": "claude", "project": "myapp"}}'

# With metadata via query params (! prefix, URL-encode ! as %21)
curl -X POST 'http://127.0.0.1:7849/speak?%21source=claude&%21project=myapp' \
  -H "Content-Type: application/json" \
  -d '{"text": "Task complete"}'
```

### POST /summarize

Summarize text and queue for playback (requires LLM backend).

```bash
curl -X POST http://127.0.0.1:7849/summarize \
  -H "Content-Type: application/json" \
  -d '{"text": "Long text to summarize..."}'
```

### GET /

Web UI showing queue history with search.

### GET /settings

Settings page for global or per-session configuration.

### GET /api/items

JSON endpoint for real-time updates (used by web UI polling).

```json
{"hash": "abc123", "items": [...]}
```

### GET /health

Health check endpoint.

## Web UI

Access at `http://127.0.0.1:7849/`

- **Real-time updates**: Auto-refreshes every 2 seconds when items are added/played
- **Card layout**: Full-width responsive grid of message cards
- **Search**: Type to search (debounced, real-time updates paused during search)
- **Play**: Click play icon; playing card shows animated border, others dim
- **Metadata**: Displayed inline in small monospace text (scrollable)

## CLI

### speeker-server

```bash
speeker-server              # Start on default port 7849
speeker-server -p 8080      # Custom port
speeker-server -H 0.0.0.0   # Bind to all interfaces
```

### speeker-player

```bash
speeker-player              # Process queue once
speeker-player --daemon     # Run as daemon (low latency)
speeker-player --cleanup 7  # Delete audio older than 7 days
```

The daemon uses a lock file to prevent multiple instances from running simultaneously.

### speeker

```bash
speeker speak "Hello"           # Generate and queue audio
speeker speak -s                # Stream mode (sentence by sentence)
speeker speak -e kokoro         # Use kokoro engine
speeker speak -v bf_emma        # Use specific voice
speeker voices                  # List available voices
speeker status                  # Show queue status
```

## Tone Tokens

Prefix text with `$Note` tokens to play musical tones before speech:

```bash
# Play two Eb3 tones then speak
curl -X POST http://127.0.0.1:7849/speak \
  -H "Content-Type: application/json" \
  -d '{"text": "$Eb3 $Eb3 Alert: build failed"}'

# Just play tones (no speech)
curl -X POST http://127.0.0.1:7849/speak \
  -H "Content-Type: application/json" \
  -d '{"text": "$C4 $E4 $G4"}'
```

Note format: `$[A-G][b/#]?[0-8]` (e.g., `$C4`, `$Eb3`, `$F#5`)

## Interpretations (outcome cues)

An _interpretation_ tags an utterance with an outcome — `SUCCESS`, `ERROR`, or
any custom name — and plays a short cue before the speech. The cue is either a
sequence of notes or a sound file, configured in the `interpretations` map.
These are built in:

- `SUCCESS` — a quick Eb3 stepping up to a ringing G#3.
- `ERROR` / `FAILURE` — Eb4, D4, then a doubled low Bb2 ("something went wrong").
- `USER_PROMPT` — a rising Bb3→Eb4 "ding-dong?" that reads as a question; use it
  when the turn paused to ask the listener something.
- `INFO` — a single Eb4 chime. `WARNING` — a doubled Eb4 attention signal.

```bash
# CLI
speeker speak --interpretation SUCCESS "All tests passed"
speeker speak --interpretation ERROR "Build failed: 3 errors"

# HTTP (top-level field or metadata)
curl -X POST http://127.0.0.1:7849/speak \
  -H "Content-Type: application/json" \
  -d '{"text": "Deploy complete", "interpretation": "SUCCESS"}'
```

From the MCP `speak` tool, pass `interpretation="SUCCESS"` (or `"ERROR"`) so
Claude can signal how a task turned out. Unknown names are rejected with the
list of valid interpretations. The cue plays, finishes, pauses
(`pause_after_seconds`), then the utterance speaks. Cues apply only when
queued for playback (not with the CLI's `--stdout`/`--no-play`).

### Auto-assessment (`/summarize` with `assess`)

`POST /summarize` accepts `"assess": true`. Instead of just summarizing, speeker
classifies the turn — `SUCCESS`, `FAILURE`, `USER_PROMPT`, or neutral — in a
single LLM call, attaches the matching cue, and styles the summary to the
outcome (a `USER_PROMPT` is phrased as a question, e.g. "I have a question
about whether to delete the old records."). A neutral turn gets no cue. The
chosen interpretation is returned in the `interpretation` field. The Claude Code
Stop hook (`summarize-response.py`) uses this so each finished turn is announced
with how it went. Falls back to a keyword heuristic when no LLM is configured.

## Background music & ducking

An optional ambient music bed plays under speech and **ducks** (drops in volume)
while a message is actually being spoken, then returns. The track is chosen per
**(queue, interpretation)** using the same most-specific-wins scoring as tone
rules — e.g. a calm bed under `SUCCESS`, a tense one under `ERROR`. Off by
default; requires `mpv` (`brew install mpv`) and is a silent no-op without it.

How it behaves:

- The bed is resolved **per message**. It loops under that message (ducked),
  **crossfades** when the next message resolves to a different track, and fades
  out at the end of the batch.
- Ducking is a **gate**: music dips to `duck_level` while TTS speaks and ramps
  back when it stops (smooth, `fade_ms`). Music runs in `mpv`; TTS still plays
  via the normal path and the OS mixes the two.

Configure in **Settings → Music** (enable, volume, duck level, fade/crossfade
times) and add rules in the table (queue / interpretation / track path). Or in
`config.json`:

```jsonc
"music": { "enabled": true, "volume": 0.6, "duck_level": 0.4, "fade_ms": 400, "crossfade_ms": 600 },
"music_rules": [
  { "queue": "deploy", "interpretation": "SUCCESS", "track": "~/Music/calm.mp3" },
  { "queue": "deploy", "interpretation": "ERROR",   "track": "~/Music/tense.mp3" }
]
```

A rule with no `queue`/`interpretation` matches anything (a default bed); a
missing track file is treated as no match. API: `GET/PUT /api/music` and
`GET/PUT /api/music-rules`.

## Configuration

### Server Config

Config file location (macOS: `~/Library/Application Support/speeker/config.json`):

```json
{
  "semantic_search": {
    "enabled": false,
    "model": "all-MiniLM-L6-v2",
    "cache_dir": null
  },
  "player": {
    "model_idle_timeout_minutes": 0
  }
}
```

| Setting                      | Default | Description                                         |
| ---------------------------- | ------- | --------------------------------------------------- |
| `model_idle_timeout_minutes` | 0       | Minutes idle before unloading TTS model (0 = never) |

When `model_idle_timeout_minutes` is 0 (default), the daemon preloads the TTS model at startup and keeps it in memory. Set to a positive value (e.g., 5) to unload the model after that many minutes of inactivity -- the model reloads automatically on the next request.

#### `polly` section

| Setting   | Default  | Description                                                   |
| --------- | -------- | ------------------------------------------------------------- |
| `region`  | null     | AWS region (null = boto3 default from profile/env)            |
| `profile` | null     | AWS profile name (null = default credential chain)            |
| `engine`  | "neural" | Polly engine variant: standard, neural, long-form, generative |
| `voice`   | "Joanna" | Polly VoiceId                                                 |

#### `ssml` section

| Setting             | Default | Description                                                                                   |
| ------------------- | ------- | --------------------------------------------------------------------------------------------- |
| `emulate_for_local` | false   | Approximate SSML on local engines (spell acronyms, pauses, casing)                            |
| `acronyms_file`     | null    | Path to file of extra acronyms to spell out (whitespace, comma, pipe, or semicolon separated) |

#### `interpretations` section

Outcome cues played before an utterance (see [Interpretations](#interpretations-outcome-cues)).

| Setting               | Default | Description                                   |
| --------------------- | ------- | --------------------------------------------- |
| `pause_after_seconds` | 0.3     | Pause after a cue finishes, before the speech |
| `map`                 | (below) | Maps an interpretation name to its indication |

Each `map` entry is one of two indication types:

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
      },
      "DEPLOY": { "type": "sound_file", "path": "/abs/path/to/chime.wav" }
    }
  }
}
```

- `notes` — a list of `{pitch, seconds}`; pitch is `[A-G][b/#]?[0-8]` (e.g. `Eb3`, `G#3`).
- `sound_file` — an absolute path (`~` is expanded); the file plays to completion, then the pause.

`SUCCESS`, `ERROR`, `FAILURE`, `USER_PROMPT`, `INFO`, and `WARNING` are built
in, so they work even if you define a custom `map`; an entry of the same name
overrides the built-in.

#### `auto_label` section

When a single bare message comes through the queue (e.g., `"Claude finished"`
enqueued without a `title=` prefix), the daemon prepends the queue's spoken
title — but only after a quiet period or when the queue context just changed.
A back-to-back burst from the same queue is **not** re-announced.

| Setting                   | Default | Description                                                               |
| ------------------------- | ------- | ------------------------------------------------------------------------- |
| `enabled`                 | true    | Master switch                                                             |
| `quiet_threshold_seconds` | 120     | Silence (seconds) before a same-queue message gets relabeled              |
| `tone`                    | `$Eb4`  | Tone token spoken before the title (same syntax as inline `$Note` tokens) |

Trigger matrix (with `enabled: true`, named queue, threshold = 120s):

| Last utterance | Last queue      | This queue     | Relabel?                                         |
| -------------- | --------------- | -------------- | ------------------------------------------------ |
| never          | —               | `compass-docs` | yes                                              |
| ≤ 120s ago     | `compass-docs`  | `compass-docs` | no                                               |
| ≤ 120s ago     | `audio-speeker` | `compass-docs` | yes (context switch)                             |
| > 120s ago     | `compass-docs`  | `compass-docs` | yes (after silence)                              |
| any            | any             | `default`      | no (no spoken title)                             |
| any            | any             | any            | no (if text already starts with a `$Note` token) |

A queue id like `compass-docs` is spoken as `compass docs`
(hyphens/underscores → spaces). The `default` queue has no spoken title and
is never auto-labeled. The auto-label only applies to single-message
batches; multi-message batches still use the `"For queue X, there are N
messages."` header.

### Settings (via Web UI or API)

Settings are hierarchical: global defaults with per-session overrides.

| Setting       | Default      | Description                          |
| ------------- | ------------ | ------------------------------------ |
| `intro_sound` | true         | Play tone before/after batches       |
| `speed`       | 1.0          | Playback speed (0.5 - 2.0)           |
| `engine`      | "pocket-tts" | TTS engine (pocket-tts/kokoro/polly) |
| `voice`       | "azelma"     | TTS voice name                       |

## Voices

### pocket-tts (default, fast)

| Voice    | Description                        |
| -------- | ---------------------------------- |
| azelma\* | Female, natural and conversational |
| alba     | Female, soft and warm              |
| javert   | Male, deep and authoritative       |
| marius   | Male, clear and articulate         |

### kokoro (higher quality)

| Voice     | Description                           |
| --------- | ------------------------------------- |
| am_liam\* | American male, clear and professional |
| af_bella  | American female, warm and friendly    |
| bf_emma   | British female, refined and elegant   |
| bm_george | British male, classic and articulate  |

\* = default voice for engine

### Amazon Polly (cloud, SSML-native)

Requires the optional dependency and AWS credentials:

```bash
uv sync --extra polly   # installs boto3
```

Credentials come from the standard AWS chain (`~/.aws/credentials`, `AWS_PROFILE`,
environment, or instance role). Configure region/profile/voice in `config.json`:

```json
{
  "polly": {
    "region": "us-east-1",
    "profile": null,
    "engine": "neural",
    "voice": "Joanna"
  }
}
```

Selecting a profile (in order of precedence):

- `speeker speak --aws-profile NAME ...` (CLI; sets `AWS_PROFILE` for that run)
- `polly.profile` in `config.json`
- the `AWS_PROFILE` environment variable (use this for the server/daemon, e.g.
  `AWS_PROFILE=personal speeker-server`); `AWS_DEFAULT_REGION` selects the region

Engine variants (`--polly-engine`): `standard` (cheapest), `neural` (natural,
widest voice selection), `long-form` (narration), `generative` (most human).

```bash
speeker speak -e polly --polly-engine neural --polly-voice Matthew --aws-profile personal "Hello there."
speeker speak -e polly --polly-engine long-form --polly-voice Danielle < chapter.txt
```

## SSML

Mark input as SSML with `--ssml`, or it is auto-detected when the text starts
with `<speak>`:

```bash
speeker speak -e polly --ssml '<speak>Hello <break time="500ms"/> world.</speak>'
```

- **Polly** renders SSML natively.
- **Local engines** (pocket-tts, kokoro) do not support SSML. By default the tags
  are stripped to plain text. With `--best-effort-ssml-emulation` (or
  `ssml.emulate_for_local: true` in config), Speeker approximates SSML: spelling
  acronyms (`<say-as interpret-as="characters">PHI</say-as>` → "P-H-I"), turning
  `<break>` into punctuation, and normalizing ALL-CAPS so the engine does not shout.

Extra acronyms to spell out can be listed in a file referenced by
`ssml.acronyms_file` (tokens separated by whitespace, commas, pipes, or semicolons);
they are merged with the built-in set.

### Generating SSML

`speeker ssml` reads plain text on stdin and writes purpose-tuned SSML to stdout.
It uses the configured LLM backend when available and falls back to a rule-based
generator; output is always sanitized to Polly-safe tags.

```bash
speeker ssml --purpose audiobook < chapter.txt | speeker speak --ssml -e polly --polly-engine long-form
```

Purposes: `audiobook` (default), `article` (alias `news`), `announcement`,
`conversational`, `technical`, `plain`. Run `speeker ssml --help` for descriptions.

Also available over HTTP (`POST /ssml` with `{"text": "...", "purpose": "..."}`)
and as the MCP `generate_ssml` tool.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ HTTP API    │────▶│ SQLite Queue │────▶│ Player      │
│ /speak      │     │ + Metadata   │     │ Daemon      │
│ /summarize  │     │ + Embeddings │     │             │
└─────────────┘     └──────────────┘     └──────┬──────┘
       │                                        │
       ▼                                        ▼
┌─────────────┐                          ┌─────────────┐
│ Web UI      │                          │ TTS Engine  │
│ Search      │                          │ Audio Out   │
│ History     │                          │             │
└─────────────┘                          └─────────────┘
```

### Storage

Speeker uses OS-appropriate directories via [platformdirs](https://github.com/platformdirs/platformdirs). On macOS:

```
~/Library/Application Support/speeker/   (config + data)
├── config.json                          # Server configuration
├── voice-prefs.json                     # Voice preference rankings
├── queue.db                             # SQLite database
├── audio/
│   └── 2024-01-15/
│       ├── 123.wav                      # Audio files by queue ID
│       └── 124.wav
└── voices/                              # Cloned voice references
    └── manifest.json

~/Library/Caches/speeker/                (cache)
├── tones/
│   └── tone_311.13_0.045.wav            # Cached tone files
├── voice-samples/                       # Voice preview audio
├── tone_intro.wav
└── tone_outro.wav

/var/folders/.../T/speeker/              (runtime)
└── player.lock                          # Daemon lock file (PID)
```

Set `SPEEKER_DIR` to override all paths (all subdirectories under it):

```bash
SPEEKER_DIR=/tmp/speeker-test speeker status
```

Auto-migration from the legacy `~/.speeker/` and `~/.config/speeker/` layout runs once on first launch.

### Database Schema

**queue** - Message history

- `id`, `session_id`, `text`, `audio_path`, `metadata` (JSON)
- `created_at`, `played_at`

**embeddings** - Semantic search vectors

- `queue_id`, `embedding` (BLOB)

**settings** - Per-session settings

- `session_id`, `intro_sound`, `speed`, `voice`, `engine`

## Development

```bash
# Clone and install
git clone https://github.com/horatio-sans-serif/speeker
cd speeker
uv sync

# Run from source
uv run speeker-server

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=speeker --cov-report=term-missing

# Format
uv run ruff format src/
```

## License

[PolyForm Noncommercial 1.0.0](LICENSE) - Free to use and modify for non-commercial purposes.
