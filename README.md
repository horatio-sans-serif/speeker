# Speeker

A text-to-speech CLI with multiple engines and voice options. Generate natural-sounding speech from text using local TTS models.

## Features

- **Multiple TTS engines**: pocket-tts (lightweight) and kokoro (higher quality)
- **18 voice options**: Various male/female voices with different accents and styles
- **Streaming support**: Speak text as it arrives via stdin (sentence-by-sentence)
- **Background playback**: Audio is queued and played asynchronously
- **Cross-platform**: macOS, Linux, and Windows support
- **Automatic MP3 conversion**: Smaller files when ffmpeg is available

## Installation

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
cd ~/projects/speeker
uv sync
```

For global installation:

```bash
uv tool install .
```

## Usage

### Basic Speech Generation

```bash
# Speak text directly
speeker speak "Hello, world!"

# Read from stdin
echo "Hello from stdin" | speeker speak

# Use a different engine and voice
speeker speak -e kokoro -v bf_emma "British accent"

# Generate audio without playing
speeker speak --no-play "Save this for later"
```

### Streaming Mode

Process text as it arrives, speaking sentence-by-sentence:

```bash
# Stream from a command (use -s or --stream)
some-command | speeker speak -s

# Stream from an LLM or chatbot
llm "Tell me a story" | speeker speak --stream

# With custom voice
cat long-document.txt | speeker speak -s -e kokoro -v am_liam
```

Streaming mode detects sentence boundaries (`.`, `!`, `?`) and speaks each sentence as soon as it's complete. This provides near-real-time speech for streaming text sources like LLMs.

### List Available Voices

```bash
speeker voices

# Filter by engine
speeker voices -e pocket-tts
```

### Check Status

```bash
speeker status
```

Shows base directory, player status, queue length, and total audio files.

### Player Management

```bash
# Start the player manually (usually automatic)
speeker play

# Run player with verbose output
speeker-player -v

# Clean up old audio files (older than 7 days)
speeker-player --cleanup 7
```

## Voices

### pocket-tts (default engine, lightweight)

| Voice    | Description                        |
| -------- | ---------------------------------- |
| azelma\* | Female, natural and conversational |
| alba     | Female, soft and warm              |
| cosette  | Female, young and bright           |
| eponine  | Female, spirited and dynamic       |
| fantine  | Female, emotional and melodic      |
| javert   | Male, deep and authoritative       |
| jean     | Male, gentle and expressive        |
| marius   | Male, clear and articulate         |

### kokoro (higher quality)

| Voice       | Description                             |
| ----------- | --------------------------------------- |
| am_liam\*   | American male, clear and professional   |
| af_bella    | American female, warm and friendly      |
| af_nicole   | American female, bright and energetic   |
| af_sarah    | American female, calm and soothing      |
| am_adam     | American male, deep and resonant        |
| am_michael  | American male, natural and casual       |
| bf_emma     | British female, refined and elegant     |
| bf_isabella | British female, warm and expressive     |
| bm_george   | British male, classic and articulate    |
| bm_lewis    | British male, modern and conversational |

\* = default voice for engine

## How It Works

### Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ speeker CLI │────▶│ TTS Engine   │────▶│ Audio File  │
│             │     │ (pocket-tts/ │     │ (.mp3/.wav) │
│             │     │  kokoro)     │     │             │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                │
                                                ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ System      │◀────│ speeker-     │◀────│ Queue File  │
│ Audio       │     │ player       │     │             │
└─────────────┘     └──────────────┘     └─────────────┘
```

1. **Text input**: The CLI receives text from arguments or stdin
2. **TTS generation**: Text is converted to audio using the selected engine
3. **File storage**: Audio is saved to `~/.speeker/YYYY-MM-DD/` with a timestamp
4. **Queue**: The audio file path is appended to a queue file
5. **Playback**: The background player processes the queue and plays audio
6. **Cleanup**: Old files can be removed with `speeker-player --cleanup DAYS`

### File Structure

```
~/.speeker/
├── 2024-01-15/
│   ├── 2024-01-15-10-30-45.mp3    # Audio file
│   ├── 2024-01-15-10-30-45.txt    # Original text
│   └── ...
├── queue                           # Pending audio files
├── queue.processing                # Currently playing (temporary)
└── .tone.wav                       # Transition sound between files
```

### Streaming Mode

In streaming mode (`--stream`), text is processed incrementally:

1. Characters are buffered as they arrive from stdin
2. When a sentence boundary is detected (`. `, `! `, `? `, or newline after punctuation), the sentence is spoken
3. Multiple sentences can be queued while earlier ones are still playing
4. Remaining text is spoken when stdin closes

This enables near-real-time speech for streaming sources like LLMs.

## Configuration

### Environment Variables

| Variable      | Default      | Description                              |
| ------------- | ------------ | ---------------------------------------- |
| `SPEEKER_DIR` | `~/.speeker` | Base directory for audio files and queue |

### Audio Players

Speeker automatically detects and uses the best available audio player:

- **macOS**: `afplay` (built-in)
- **Linux**: `paplay` (PulseAudio), `aplay` (ALSA), or `ffplay` (FFmpeg)
- **Windows**: PowerShell `Media.SoundPlayer`

### MP3 Conversion

If `ffmpeg` is installed, audio is automatically converted from WAV to MP3 (64kbps) for smaller file sizes. Otherwise, WAV files are used.

## Dependencies

Core dependencies (installed automatically):

- `pocket-tts` - Lightweight TTS engine
- `kokoro` - High-quality TTS engine
- `scipy` - Audio file handling
- `numpy` - Audio processing
- `torch` - Neural network backend

## Development

```bash
# Run from source
uv run speeker speak "test"

# Run tests
uv run pytest

# Format code
uv run ruff format src/
```

## License

MIT
