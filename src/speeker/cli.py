#!/usr/bin/env python3
"""Speeker - Text-to-speech CLI with multiple engines and voice options."""

import argparse
import fcntl
import re
import select
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.io import wavfile

from .paths import data_dir, audio_dir as _audio_dir, ensure_dir
from .preprocessing import preprocess_for_tts
from .voices import (
    DEFAULT_ENGINE,
    get_default_voice,
    get_pocket_tts_voice_path,
    get_voices,
    validate_voice,
)
from .voice_prefs import (
    run_voice_prefs_server,
    get_preferred_voice,
    get_preferred_engine,
    ensure_all_samples,
    get_voice_prefs,
    BUNDLED_PREFS_FILE,
)

if TYPE_CHECKING:
    from kokoro import KPipeline
    from pocket_tts import TTSModel

# Lazy-loaded TTS models (expensive to initialize)
_pocket_tts_model: "TTSModel | None" = None
_pocket_tts_voice_states: dict[str, Any] = {}
_kokoro_pipeline: "KPipeline | None" = None


def get_queue_file() -> Path:
    """Get the path to the queue file."""
    return data_dir() / "queue"


def ensure_output_dir() -> Path:
    """Ensure the output directory for today exists."""
    today = datetime.now().strftime("%Y-%m-%d")
    return ensure_dir(_audio_dir() / today)


def get_pocket_tts_model() -> "TTSModel":
    """Lazy-load the pocket-tts model."""
    global _pocket_tts_model
    if _pocket_tts_model is None:
        from pocket_tts import TTSModel

        _pocket_tts_model = TTSModel.load_model()
    return _pocket_tts_model


def get_pocket_tts_voice_state(voice: str) -> Any:
    """Get or create voice state for pocket-tts."""
    global _pocket_tts_voice_states
    if voice not in _pocket_tts_voice_states:
        model = get_pocket_tts_model()
        voice_path = get_pocket_tts_voice_path(voice)
        _pocket_tts_voice_states[voice] = model.get_state_for_audio_prompt(voice_path)
    return _pocket_tts_voice_states[voice]


def get_kokoro_pipeline() -> "KPipeline":
    """Lazy-load the kokoro pipeline."""
    global _kokoro_pipeline
    if _kokoro_pipeline is None:
        from kokoro import KPipeline

        _kokoro_pipeline = KPipeline(lang_code="a")
    return _kokoro_pipeline


def generate_pocket_tts(text: str, voice: str) -> tuple[np.ndarray, int]:
    """Generate audio using pocket-tts."""
    model = get_pocket_tts_model()
    voice_state = get_pocket_tts_voice_state(voice)
    audio = model.generate_audio(voice_state, text)
    return audio.numpy(), model.sample_rate


def generate_kokoro(text: str, voice: str) -> tuple[np.ndarray, int]:
    """Generate audio using kokoro."""
    pipeline = get_kokoro_pipeline()
    generator = pipeline(text, voice=voice)

    audio_chunks = []
    for _, _, audio in generator:
        audio_chunks.append(audio)

    if not audio_chunks:
        raise ValueError("Kokoro generated no audio")

    audio = np.concatenate(audio_chunks)
    sample_rate = 24000
    return audio, sample_rate


def save_audio(audio: np.ndarray, sample_rate: int, text: str) -> Path:
    """Save audio to file (MP3 if ffmpeg available, otherwise WAV) and text to TXT file."""
    output_dir = ensure_output_dir()
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    # Generate unique filename if collision
    base_name = timestamp
    counter = 0
    while (output_dir / f"{base_name}.wav").exists() or (output_dir / f"{base_name}.mp3").exists():
        counter += 1
        base_name = f"{timestamp}-{counter}"

    wav_path = output_dir / f"{base_name}.wav"
    mp3_path = output_dir / f"{base_name}.mp3"
    txt_path = output_dir / f"{base_name}.txt"

    # Normalize audio to 16-bit PCM range
    audio_normalized = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio_normalized * 32767).astype(np.int16)

    # Save WAV file
    wavfile.write(wav_path, sample_rate, audio_int16)

    # Convert to MP3 if ffmpeg is available
    final_path = wav_path
    if shutil.which("ffmpeg"):
        try:
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(wav_path),
                    "-b:a",
                    "64k",
                    "-loglevel",
                    "error",
                    str(mp3_path),
                ],
                capture_output=True,
                timeout=30,
            )
            if result.returncode == 0 and mp3_path.exists():
                wav_path.unlink()
                final_path = mp3_path
        except (subprocess.TimeoutExpired, OSError):
            pass

    # Save text file
    txt_path.write_text(text, encoding="utf-8")

    return final_path


def is_player_running() -> bool:
    """Check if the speeker player is already running (excludes zombie processes)."""
    try:
        # Get PIDs matching speeker-player
        result = subprocess.run(
            ["pgrep", "-f", "speeker-player"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return False

        # Check each PID to see if it's actually running (not zombie)
        for pid in result.stdout.strip().split('\n'):
            if not pid:
                continue
            # Check process state via ps
            ps_result = subprocess.run(
                ["ps", "-o", "state=", "-p", pid],
                capture_output=True,
                text=True,
            )
            state = ps_result.stdout.strip()
            # Z = zombie, skip those
            if state and state[0] != 'Z':
                return True

        return False
    except OSError:
        return False


def start_player() -> None:
    """Start the speeker player in the background."""
    if is_player_running():
        return

    # Find speeker-player - check PATH first, then common locations
    player_cmd = shutil.which("speeker-player")
    if not player_cmd:
        # Fallback to common install locations
        for path in [
            Path.home() / ".local/bin/speeker-player",
            Path("/usr/local/bin/speeker-player"),
        ]:
            if path.exists():
                player_cmd = str(path)
                break

    if not player_cmd:
        return  # Can't find player

    try:
        subprocess.Popen(
            [player_cmd, "--daemon"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError:
        pass


def queue_for_playback(audio_path: Path) -> None:
    """Add audio file to the playback queue and ensure player is running."""
    ensure_dir(data_dir())
    queue_file = get_queue_file()

    with open(queue_file, "a") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(f"{audio_path}\n")
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    start_player()


def speak_text(
    text: str, engine: str, voice: str, no_play: bool, quiet: bool, stdout: bool
) -> bool:
    """Generate and optionally queue speech for a piece of text. Returns True on success."""
    if not text or not text.strip():
        return True  # Empty text is not an error

    # Preprocess text for better TTS output
    processed_text = preprocess_for_tts(text)

    try:
        if engine == "pocket-tts":
            audio, sample_rate = generate_pocket_tts(processed_text, voice)
        else:
            audio, sample_rate = generate_kokoro(processed_text, voice)

        if stdout:
            audio_normalized = np.clip(audio, -1.0, 1.0)
            audio_int16 = (audio_normalized * 32767).astype(np.int16)
            wavfile.write(sys.stdout.buffer, sample_rate, audio_int16)
        elif no_play:
            audio_path = save_audio(audio, sample_rate, text)
            print(audio_path)
        else:
            audio_path = save_audio(audio, sample_rate, text)
            queue_for_playback(audio_path)
            if not quiet:
                print(f"Queued: {audio_path}", file=sys.stderr)

        return True

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return False


# Sentence boundary pattern: ends with .!? followed by space, newline, or end of string
SENTENCE_END_PATTERN = re.compile(r"[.!?](?:\s|$)")


def stream_sentences_from_stdin():
    """Generator that yields complete sentences as they arrive from stdin.

    Buffers input and yields when sentence boundaries are detected.
    Sentence boundaries are: . ! ? followed by whitespace or end of input.
    """
    buffer = ""

    # Check if stdin is a tty (interactive) - if so, don't use streaming
    if sys.stdin.isatty():
        yield sys.stdin.read()
        return

    while True:
        # Use select to check if data is available (with timeout for responsiveness)
        if sys.platform != "win32":
            readable, _, _ = select.select([sys.stdin], [], [], 0.1)
            if not readable:
                # No data available, check if we should yield buffered content
                continue

        # Read available data
        chunk = sys.stdin.read(1)
        if not chunk:
            # EOF - yield remaining buffer
            if buffer.strip():
                yield buffer.strip()
            break

        buffer += chunk

        # Check for sentence boundaries
        match = SENTENCE_END_PATTERN.search(buffer)
        if match:
            # Find the position after the sentence-ending punctuation and whitespace
            end_pos = match.end()
            sentence = buffer[:end_pos].strip()
            buffer = buffer[end_pos:]

            if sentence:
                yield sentence


def cmd_speak_stream(args: argparse.Namespace) -> int:
    """Handle streaming speak mode - process sentences as they arrive."""
    engine = args.engine or DEFAULT_ENGINE
    voice = args.voice or get_default_voice(engine)

    if engine not in ("pocket-tts", "kokoro"):
        print(f"Error: Unknown engine '{engine}'. Use 'pocket-tts' or 'kokoro'.", file=sys.stderr)
        return 1

    if not validate_voice(engine, voice):
        available = list(get_voices(engine).get(engine, {}).keys())
        print(f"Error: Unknown voice '{voice}' for engine '{engine}'.", file=sys.stderr)
        print(f"Available voices: {', '.join(available)}", file=sys.stderr)
        return 1

    if not args.quiet:
        print(f"Streaming with {engine}/{voice}...", file=sys.stderr)

    sentence_count = 0
    error_count = 0

    for sentence in stream_sentences_from_stdin():
        if not speak_text(sentence, engine, voice, args.no_play, args.quiet, args.stdout):
            error_count += 1
        else:
            sentence_count += 1

    if not args.quiet:
        print(f"Streamed {sentence_count} sentence(s)", file=sys.stderr)

    return 1 if error_count > 0 and sentence_count == 0 else 0


def cmd_speak(args: argparse.Namespace) -> int:
    """Handle the speak command."""
    # If streaming mode, delegate to streaming handler
    if args.stream:
        return cmd_speak_stream(args)

    text = args.text
    if not text:
        # Read from stdin if no text provided
        text = sys.stdin.read()

    if not text or not text.strip():
        print("Error: No text provided", file=sys.stderr)
        return 1

    engine = args.engine or DEFAULT_ENGINE
    voice = args.voice or get_default_voice(engine)

    if engine not in ("pocket-tts", "kokoro"):
        print(f"Error: Unknown engine '{engine}'. Use 'pocket-tts' or 'kokoro'.", file=sys.stderr)
        return 1

    if not validate_voice(engine, voice):
        available = list(get_voices(engine).get(engine, {}).keys())
        print(f"Error: Unknown voice '{voice}' for engine '{engine}'.", file=sys.stderr)
        print(f"Available voices: {', '.join(available)}", file=sys.stderr)
        return 1

    if not args.quiet:
        print(f"Generating speech with {engine}/{voice}...", file=sys.stderr)

    if not speak_text(text, engine, voice, args.no_play, args.quiet, args.stdout):
        return 1

    return 0


def cmd_voices(args: argparse.Namespace) -> int:
    """Handle the voices command."""
    voices = get_voices(args.engine)

    for engine_name, voice_dict in voices.items():
        default_voice = get_default_voice(engine_name)
        is_default_engine = engine_name == DEFAULT_ENGINE

        engine_label = engine_name
        if is_default_engine:
            engine_label += " (default engine)"
        print(f"\n{engine_label}:")
        print("-" * 40)

        for name, description in voice_dict.items():
            marker = " *" if name == default_voice else ""
            print(f"  {name:<15} {description}{marker}")

    print("\n* = default voice for engine")
    return 0


def cmd_play(args: argparse.Namespace) -> int:
    """Handle the play command - start the player."""
    if is_player_running():
        print("Player is already running", file=sys.stderr)
        return 0

    start_player()
    print("Player started", file=sys.stderr)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Handle the status command."""
    audio = _audio_dir()
    queue_file = get_queue_file()

    print(f"Data directory: {data_dir()}")
    print(f"Player running: {'yes' if is_player_running() else 'no'}")

    if queue_file.exists():
        with open(queue_file) as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
        print(f"Queue length: {len(lines)}")
    else:
        print("Queue length: 0")

    # Count audio files
    total_files = 0
    total_size = 0
    if audio.exists():
        for day_dir in audio.iterdir():
            if day_dir.is_dir() and day_dir.name[0].isdigit():
                for f in day_dir.iterdir():
                    if f.suffix in (".wav", ".mp3"):
                        total_files += 1
                        total_size += f.stat().st_size

    print(f"Audio files: {total_files}")
    print(f"Total size: {total_size / (1024 * 1024):.1f} MB")

    return 0


def cmd_voice_prefs(args: argparse.Namespace) -> int:
    """Handle the voice-prefs command."""
    run_voice_prefs_server(quiet=args.quiet)
    return 0


def cmd_generate_samples(args: argparse.Namespace) -> int:
    """Handle the generate-samples command."""
    print("Generating voice samples for all voices...", file=sys.stderr)
    samples = ensure_all_samples(quiet=args.quiet)

    total = sum(len(v) for v in samples.values())
    print(f"Generated {total} voice samples.", file=sys.stderr)
    return 0


def cmd_bundle_prefs(args: argparse.Namespace) -> int:
    """Handle the bundle-prefs command - copy user prefs to bundled defaults."""
    prefs = get_voice_prefs()
    if not prefs.get("pocket-tts") and not prefs.get("kokoro"):
        print("No voice preferences found. Run 'speeker voice-prefs' first.", file=sys.stderr)
        return 1

    # Write to the bundled defaults file
    import json
    with open(BUNDLED_PREFS_FILE, "w") as f:
        json.dump(prefs, f, indent=2)

    print(f"Bundled preferences saved to: {BUNDLED_PREFS_FILE}", file=sys.stderr)
    return 0


def cmd_voice_clone(args: argparse.Namespace) -> int:
    """Handle the voice-clone command."""
    from .voice_clone import clone_voice, get_custom_voices, delete_custom_voice

    if args.list:
        voices = get_custom_voices()
        if not voices:
            print("No custom voices found.", file=sys.stderr)
            return 0
        print(f"\nCustom voices ({len(voices)}):")
        print("-" * 50)
        for name, entry in voices.items():
            desc = entry.get("description", "")
            created = entry.get("created_at", "unknown")[:10]
            print(f"  {name:<25} {desc}  ({created})")
        print()
        return 0

    if args.delete:
        if delete_custom_voice(args.delete):
            print(f"Deleted voice: {args.delete}", file=sys.stderr)
            return 0
        else:
            print(f"Voice not found: {args.delete}", file=sys.stderr)
            return 1

    if not args.name or not args.sources:
        print("Error: NAME and at least one SOURCE are required", file=sys.stderr)
        print("Usage: speeker voice-clone NAME SOURCE [SOURCE...] [--start N] [--duration N]", file=sys.stderr)
        return 1

    try:
        path = clone_voice(
            name=args.name,
            sources=args.sources,
            start_secs=args.start,
            duration_secs=args.duration,
            description=args.description,
        )
        print(f"Voice cloned: {args.name}", file=sys.stderr)
        print(f"Audio saved to: {path}", file=sys.stderr)
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def main() -> int:
    """Main entry point."""
    from .migrate import migrate
    migrate()

    parser = argparse.ArgumentParser(
        prog="speeker",
        description="Text-to-speech CLI with multiple engines and voice options",
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # speak command
    speak_parser = subparsers.add_parser("speak", help="Generate and play speech from text")
    speak_parser.add_argument(
        "text", nargs="?", help="Text to speak (reads from stdin if not provided)"
    )
    speak_parser.add_argument("-e", "--engine", choices=["pocket-tts", "kokoro"], help="TTS engine")
    speak_parser.add_argument("-v", "--voice", help="Voice to use")
    speak_parser.add_argument(
        "-q", "--quiet", action="store_true", help="Suppress progress messages"
    )
    speak_parser.add_argument(
        "--no-play", action="store_true", help="Generate audio but don't queue for playback"
    )
    speak_parser.add_argument(
        "--stdout", action="store_true", help="Write WAV audio to stdout instead of file"
    )
    speak_parser.add_argument(
        "-s",
        "--stream",
        action="store_true",
        help="Stream mode: speak sentences as they arrive from stdin",
    )
    speak_parser.set_defaults(func=cmd_speak)

    # voices command
    voices_parser = subparsers.add_parser("voices", help="List available voices")
    voices_parser.add_argument(
        "-e", "--engine", choices=["pocket-tts", "kokoro"], help="Filter by engine"
    )
    voices_parser.set_defaults(func=cmd_voices)

    # play command
    play_parser = subparsers.add_parser("play", help="Start the audio player")
    play_parser.set_defaults(func=cmd_play)

    # status command
    status_parser = subparsers.add_parser("status", help="Show speeker status")
    status_parser.set_defaults(func=cmd_status)

    # voice-prefs command
    voice_prefs_parser = subparsers.add_parser(
        "voice-prefs",
        help="Open voice preferences UI to rank voices by preference"
    )
    voice_prefs_parser.add_argument(
        "-q", "--quiet", action="store_true", help="Suppress progress messages"
    )
    voice_prefs_parser.set_defaults(func=cmd_voice_prefs)

    # generate-samples command
    generate_samples_parser = subparsers.add_parser(
        "generate-samples",
        help="Generate voice samples for all voices (used by voice-prefs)"
    )
    generate_samples_parser.add_argument(
        "-q", "--quiet", action="store_true", help="Suppress progress messages"
    )
    generate_samples_parser.set_defaults(func=cmd_generate_samples)

    # bundle-prefs command (developer use)
    bundle_prefs_parser = subparsers.add_parser(
        "bundle-prefs",
        help="Copy your voice preferences to bundled defaults (developer use)"
    )
    bundle_prefs_parser.set_defaults(func=cmd_bundle_prefs)

    # voice-clone command
    voice_clone_parser = subparsers.add_parser(
        "voice-clone",
        help="Clone a voice from audio/video files or URLs",
    )
    voice_clone_parser.add_argument("name", nargs="?", help="Name for the cloned voice")
    voice_clone_parser.add_argument(
        "sources", nargs="*", help="Audio/video file paths or URLs (YouTube supported)"
    )
    voice_clone_parser.add_argument(
        "--start", type=float, default=0, help="Start time in seconds (default: 0)"
    )
    voice_clone_parser.add_argument(
        "--duration", type=float, default=30, help="Duration in seconds (default: 30)"
    )
    voice_clone_parser.add_argument(
        "--description", help="Description of the voice"
    )
    voice_clone_parser.add_argument(
        "--list", action="store_true", help="List custom voices"
    )
    voice_clone_parser.add_argument(
        "--delete", metavar="NAME", help="Delete a custom voice"
    )
    voice_clone_parser.set_defaults(func=cmd_voice_clone)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
