# Speeker

A text-to-speech system with HTTP API, web UI, and CLI. Queue text for playback with metadata, search history, and configurable voices.

## Features

- **HTTP API**: Queue text via REST endpoints with metadata support
- **Web UI**: View queue history, play audio, search messages
- **Multiple TTS engines**: pocket-tts (fast) and kokoro (higher quality)
- **Daemon mode**: Low-latency playback with warm TTS model
- **Metadata**: Attach arbitrary key-value data to messages
- **Search**: Fuzzy text search or semantic search with embeddings
- **Per-session settings**: Speed, voice, intro/outro sounds

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

The daemon uses a lock file (`~/.speeker/.player.lock`) to prevent multiple instances from running simultaneously.

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

## Configuration

### Server Config

`~/.config/speeker/config.json`:

```json
{
    "semantic_search": {
        "enabled": false,
        "model": "all-MiniLM-L6-v2",
        "cache_dir": null
    }
}
```

### Settings (via Web UI or API)

Settings are hierarchical: global defaults with per-session overrides.

| Setting       | Default      | Description                    |
| ------------- | ------------ | ------------------------------ |
| `intro_sound` | true         | Play tone before/after batches |
| `speed`       | 1.0          | Playback speed (0.5 - 2.0)     |
| `engine`      | "pocket-tts" | TTS engine (pocket-tts/kokoro) |
| `voice`       | "azelma"     | TTS voice name                 |

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

```
~/.speeker/
├── queue.db                    # SQLite database
├── .player.lock                # Daemon lock file (PID)
├── audio/
│   └── 2024-01-15/
│       ├── 123.wav             # Audio files by queue ID
│       └── 124.wav
└── tones/
    └── tone_311.13_0.045.wav   # Cached tone files

~/.config/speeker/
└── config.json                 # Server configuration
```

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

# Format
uv run ruff format src/
```

## License

MIT
