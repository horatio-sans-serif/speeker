#!/usr/bin/env python3
"""Speeker playback daemon - watches SQLite queue and plays TTS immediately.

Keeps TTS model warm for low-latency speech generation.
"""

import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pocket_tts import TTSModel

from .queue_db import (
    get_connection,
    get_last_utterance_time,
    get_pending_count,
    get_pending_for_session,
    get_sessions_with_pending,
    get_settings,
    mark_played,
    relative_time,
    get_session_label,
    set_last_utterance_time,
    cleanup_old_entries,
)

# Default base directory
DEFAULT_BASE_DIR = Path.home() / ".speeker"

# Timing
PAUSE_BETWEEN_MESSAGES = 0.3
PAUSE_BETWEEN_SESSIONS = 0.5
POLL_INTERVAL = 0.5  # Check queue every 500ms
IDLE_TIMEOUT = 300  # Exit after 5 minutes of no activity

# How long before we re-announce "This is Claude Code"
ANNOUNCE_THRESHOLD_MINUTES = 30

# Lazy-loaded TTS model (expensive to initialize, kept warm)
_tts_model: "TTSModel | None" = None
_voice_states: dict[str, object] = {}

# Cached sound files
_intro_sound_path: Path | None = None
_outro_sound_path: Path | None = None
_tone_cache: dict[str, Path] = {}

# Musical note parsing for tone tokens
import re

NOTE_PATTERN = re.compile(r"^\s*\$([A-Ga-g])([b#]?)([0-8])")

def parse_note_token(token: str) -> tuple[str, int] | None:
    """Parse a note token like 'Eb4' into (note_name, octave)."""
    match = re.match(r"([A-Ga-g])([b#]?)([0-8])", token)
    if not match:
        return None
    note = match.group(1).lower()
    accidental = match.group(2)
    octave = int(match.group(3))
    # tones uses '#' for sharp, 'b' for flat in note name
    if accidental:
        note = note + accidental
    return note, octave

def extract_tone_tokens(text: str) -> tuple[list[str], str]:
    """Extract $Note tokens from text and return (tokens, clean_text)."""
    tokens = []
    remaining = text
    while True:
        match = NOTE_PATTERN.match(remaining)
        if not match:
            break
        full_note = match.group(1) + match.group(2) + match.group(3)
        tokens.append(full_note)
        remaining = remaining[match.end():]
    return tokens, remaining.strip()


def get_base_dir() -> Path:
    """Get the base directory for speeker output."""
    return Path(os.environ.get("SPEEKER_DIR", DEFAULT_BASE_DIR))


def get_audio_player() -> list[str] | None:
    """Get the appropriate audio player command for this platform."""
    system = platform.system()

    if system == "Darwin":
        if shutil.which("afplay"):
            return ["afplay"]
    elif system == "Linux":
        if shutil.which("paplay"):
            return ["paplay"]
        if shutil.which("aplay"):
            return ["aplay", "-q"]
        if shutil.which("ffplay"):
            return ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"]
    elif system == "Windows":
        if shutil.which("powershell"):
            return None

    if shutil.which("ffplay"):
        return ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"]

    return None


AUDIO_PLAYER = get_audio_player()


def get_tts_model() -> "TTSModel":
    """Get the TTS model, loading it if needed (kept warm for fast responses)."""
    global _tts_model
    if _tts_model is None:
        from pocket_tts import TTSModel
        _tts_model = TTSModel.load_model()
    return _tts_model


def get_voice_state(voice: str) -> object:
    """Get or create voice state for the given voice."""
    global _voice_states
    if voice not in _voice_states:
        from .voices import get_pocket_tts_voice_path
        model = get_tts_model()
        voice_path = get_pocket_tts_voice_path(voice)
        _voice_states[voice] = model.get_state_for_audio_prompt(voice_path)
    return _voice_states[voice]


def generate_tone(frequencies: list[int], rising: bool = True) -> Path:
    """Generate a multi-note tone.

    Args:
        frequencies: List of frequencies to play in sequence
        rising: If True, play in order (intro). If False, reverse (outro).
    """
    import math
    import struct
    import wave

    base_dir = get_base_dir()
    suffix = "intro" if rising else "outro"
    tone_path = base_dir / f".tone_{suffix}.wav"

    if tone_path.exists():
        return tone_path

    if not rising:
        frequencies = list(reversed(frequencies))

    sample_rate = 44100
    amplitude = 0.3
    note_duration = 0.12
    gap_duration = 0.03

    samples = []
    for note_idx, frequency in enumerate(frequencies):
        n_samples = int(sample_rate * note_duration)
        fade_samples = int(sample_rate * 0.015)

        for i in range(n_samples):
            t = i / sample_rate
            if i < fade_samples:
                envelope = i / fade_samples
            elif i > n_samples - fade_samples:
                envelope = (n_samples - i) / fade_samples
            else:
                envelope = 1.0
            value = amplitude * envelope * math.sin(2 * math.pi * frequency * t)
            samples.append(int(value * 32767))

        if note_idx < len(frequencies) - 1:
            gap_samples = int(sample_rate * gap_duration)
            samples.extend([0] * gap_samples)

    base_dir.mkdir(parents=True, exist_ok=True)
    with wave.open(str(tone_path), "w") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(struct.pack(f"{len(samples)}h", *samples))

    return tone_path


def get_intro_sound() -> Path:
    """Get path to intro sound (rising pitch)."""
    global _intro_sound_path
    if _intro_sound_path is None:
        # E4 -> G4 -> C5 (rising major chord)
        _intro_sound_path = generate_tone([330, 392, 523], rising=True)
    return _intro_sound_path


def get_outro_sound() -> Path:
    """Get path to outro sound (falling pitch)."""
    global _outro_sound_path
    if _outro_sound_path is None:
        # C5 -> G4 -> E4 (falling major chord)
        _outro_sound_path = generate_tone([523, 392, 330], rising=False)
    return _outro_sound_path


def play_audio(audio_path: Path, verbose: bool = False) -> bool:
    """Play an audio file. Returns True on success."""
    if not audio_path.exists():
        if verbose:
            print(f"[WARN] Audio file not found: {audio_path}", file=sys.stderr)
        return False

    if AUDIO_PLAYER is None:
        if platform.system() == "Windows":
            try:
                ps_cmd = f'(New-Object Media.SoundPlayer "{audio_path}").PlaySync()'
                subprocess.run(["powershell", "-c", ps_cmd], check=True)
                return True
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                if verbose:
                    print(f"[ERROR] PowerShell audio failed: {e}", file=sys.stderr)
                return False
        else:
            print("[ERROR] No audio player found.", file=sys.stderr)
            return False

    if verbose:
        print(f"[PLAY] {audio_path}", file=sys.stderr)

    try:
        subprocess.run([*AUDIO_PLAYER, str(audio_path)], check=True)
        return True
    except subprocess.CalledProcessError as e:
        if verbose:
            print(f"[ERROR] Audio playback failed: {e}", file=sys.stderr)
        return False
    except FileNotFoundError:
        print(f"[ERROR] Audio player not found: {AUDIO_PLAYER[0]}", file=sys.stderr)
        return False


def generate_tts(
    text: str,
    voice: str | None = None,
    speed: float = 1.0,
    save_path: Path | None = None,
    verbose: bool = False,
) -> Path | None:
    """Generate TTS audio for text.

    Args:
        text: Text to speak
        voice: Voice to use (default: azelma)
        speed: Playback speed multiplier (default: 1.0)
        save_path: If provided, save to this path instead of temp file
        verbose: Print debug info

    Returns:
        Path to audio file, or None on failure
    """
    try:
        import numpy as np
        from scipy.io import wavfile
        from .voices import get_default_voice
        from .preprocessing import preprocess_for_tts

        if verbose:
            print(f"[TTS] {text[:60]}{'...' if len(text) > 60 else ''}", file=sys.stderr)

        # Preprocess text
        processed = preprocess_for_tts(text)

        # Generate audio (model is kept warm)
        model = get_tts_model()
        voice = voice or get_default_voice("pocket-tts")
        voice_state = get_voice_state(voice)
        audio = model.generate_audio(voice_state, processed)

        # Apply speed adjustment by resampling
        audio_np = audio.numpy()
        if speed != 1.0 and speed > 0:
            from scipy import signal
            # Resample to change speed (higher speed = fewer samples)
            new_length = int(len(audio_np) / speed)
            audio_np = signal.resample(audio_np, new_length)

        # Normalize and convert to int16
        audio_normalized = np.clip(audio_np, -1.0, 1.0)
        audio_int16 = (audio_normalized * 32767).astype(np.int16)

        # Save to file
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            wavfile.write(str(save_path), model.sample_rate, audio_int16)
            return save_path
        else:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                wavfile.write(f.name, model.sample_rate, audio_int16)
                return Path(f.name)

    except Exception as e:
        if verbose:
            print(f"[ERROR] TTS failed: {e}", file=sys.stderr)
        return None


def generate_combined_tones_from_tokens(tokens: list[str], duration: float = 0.8) -> Path:
    """Generate a single WAV with tones using the tones library."""
    from tones import SINE_WAVE
    from tones.mixer import Mixer

    # Cache key
    cache_key = "_".join(tokens) + f"_{duration}_v6"
    if cache_key in _tone_cache and _tone_cache[cache_key].exists():
        return _tone_cache[cache_key]

    base_dir = get_base_dir()
    tone_dir = base_dir / "tones"
    tone_dir.mkdir(parents=True, exist_ok=True)
    tone_path = tone_dir / f"combined_{cache_key}.wav"

    if tone_path.exists():
        _tone_cache[cache_key] = tone_path
        return tone_path

    # Use tones library for synthesis
    mixer = Mixer(44100, 0.5)
    mixer.create_track(0, SINE_WAVE, vibrato_frequency=5.5, vibrato_variance=0.02, attack=0.01, decay=0.3)

    for token in tokens:
        parsed = parse_note_token(token)
        if parsed:
            note, octave = parsed
            mixer.add_note(0, note=note, octave=octave, duration=duration)

    mixer.write_wav(str(tone_path))
    _tone_cache[cache_key] = tone_path
    return tone_path


def play_tone_tokens(tokens: list[str], verbose: bool = False) -> None:
    """Play a sequence of tone tokens (e.g., ["Eb4"]) as a single audio file."""
    if not tokens:
        return

    if verbose:
        print(f"[TONE] Playing tokens: {tokens}", file=sys.stderr)

    tone_path = generate_combined_tones_from_tokens(tokens)
    play_audio(tone_path, verbose)


def speak_text(
    text: str,
    voice: str | None = None,
    speed: float = 1.0,
    save_path: Path | None = None,
    verbose: bool = False,
) -> Path | None:
    """Generate and play TTS for text.

    Handles tone tokens like $Eb3 at the start of text - plays them as
    musical tones before speaking the remaining text.

    Returns path to saved audio file if save_path provided, else None.
    """
    # Extract and play any leading tone tokens
    tone_tokens, clean_text = extract_tone_tokens(text)
    if tone_tokens:
        play_tone_tokens(tone_tokens, verbose)
        # No pause - start speaking immediately after tone

    # If only tones, nothing to speak
    if not clean_text:
        return save_path

    audio_path = generate_tts(clean_text, voice=voice, speed=speed, save_path=save_path, verbose=verbose)
    if audio_path is None:
        return None

    try:
        play_audio(audio_path, verbose)
        return save_path  # Return saved path if we saved it
    finally:
        # Clean up temp files (not saved files)
        if save_path is None and audio_path:
            try:
                audio_path.unlink()
            except OSError:
                pass


def should_announce_intro() -> bool:
    """Check if we should say 'This is Claude Code'."""
    last_time = get_last_utterance_time()
    if last_time is None:
        return True
    threshold = datetime.now(timezone.utc) - timedelta(minutes=ANNOUNCE_THRESHOLD_MINUTES)
    return last_time < threshold


def build_session_script(session_id: str, items: list[dict], is_only_session: bool) -> list[str]:
    """Build the speech script for a session's messages."""
    lines = []
    count = len(items)
    session_label = get_session_label(session_id)

    # Session header (skip if single message in single session)
    if not (count == 1 and is_only_session):
        if count == 1:
            lines.append(f"For {session_label}, there is 1 message.")
        else:
            lines.append(f"For {session_label}, there are {count} messages.")

    # Each message
    for i, item in enumerate(items):
        time_ago = relative_time(item["created_at"])
        text = item["text"]

        if count == 1 and is_only_session:
            # Single message total - just say the text
            if time_ago:
                lines.append(f"From {time_ago}: {text}")
            else:
                lines.append(text)
        else:
            # Multiple messages
            if i == 0 and count > 1:
                prefix = "First"
            elif i == count - 1 and count > 1:
                prefix = "Last"
            elif count == 1:
                prefix = "It"
            else:
                prefix = "Next"

            if time_ago:
                lines.append(f"{prefix}, from {time_ago}: {text}")
            else:
                lines.append(f"{prefix}: {text}")

    return lines


def update_audio_path(item_id: int, audio_path: Path) -> None:
    """Update the audio_path for a queue item."""
    with get_connection() as conn:
        conn.execute(
            "UPDATE queue SET audio_path = ? WHERE id = ?",
            (str(audio_path), item_id)
        )
        conn.commit()


def get_audio_save_path(item_id: int) -> Path:
    """Get the path where audio for this item should be saved."""
    from datetime import datetime
    base_dir = get_base_dir()
    today = datetime.now().strftime("%Y-%m-%d")
    audio_dir = base_dir / "audio" / today
    audio_dir.mkdir(parents=True, exist_ok=True)
    return audio_dir / f"{item_id}.wav"


def process_queue(verbose: bool = False) -> int:
    """Process all pending messages. Returns count of messages played."""
    sessions = get_sessions_with_pending()
    if not sessions:
        return 0

    # Count total messages
    total_messages = sum(len(get_pending_for_session(s)) for s in sessions)
    is_single_message = total_messages == 1

    total_played = 0

    # Get global settings for intro
    global_settings = get_settings()

    # Intro sound if enabled and not single message
    if should_announce_intro() and not is_single_message and global_settings["intro_sound"]:
        if verbose:
            print("[INFO] Playing intro sound", file=sys.stderr)
        play_audio(get_intro_sound(), verbose)
        time.sleep(0.2)
        speak_text("This is Claude Code.", verbose=verbose)
        time.sleep(PAUSE_BETWEEN_SESSIONS)

    # Process each session
    is_only_session = len(sessions) == 1
    for session_idx, session_id in enumerate(sessions):
        items = get_pending_for_session(session_id)
        if not items:
            continue

        # Get settings for this session
        settings = get_settings(session_id)
        voice = settings["voice"]
        speed = settings["speed"]

        if session_idx > 0:
            time.sleep(PAUSE_BETWEEN_SESSIONS)

        script_lines = build_session_script(session_id, items, is_only_session)

        for line_idx, line in enumerate(script_lines):
            if line_idx > 0:
                time.sleep(PAUSE_BETWEEN_MESSAGES)

            # Determine which item this line corresponds to (for saving audio)
            # Script: [header], [msg1], [msg2], ...
            item_idx = line_idx - 1 if not (len(items) == 1 and is_only_session) else line_idx
            save_path = None
            if 0 <= item_idx < len(items):
                save_path = get_audio_save_path(items[item_idx]["id"])

            result = speak_text(line, voice=voice, speed=speed, save_path=save_path, verbose=verbose)

            if result is not None or save_path is None:
                total_played += 1
                # Update audio path in database
                if save_path and 0 <= item_idx < len(items):
                    update_audio_path(items[item_idx]["id"], save_path)

        # Mark items as played
        for item in items:
            mark_played(item["id"])

    # Outro (skip if single message)
    if total_played > 0 and not is_single_message and global_settings["intro_sound"]:
        time.sleep(PAUSE_BETWEEN_SESSIONS)
        speak_text("That is all.", verbose=verbose)
        time.sleep(0.2)
        play_audio(get_outro_sound(), verbose)

    if total_played > 0:
        set_last_utterance_time()

    return total_played


def acquire_lock() -> Path | None:
    """Try to acquire a lock file. Returns lock path if acquired, None if already running."""
    lock_path = get_base_dir() / ".player.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if lock exists and if process is still running
    if lock_path.exists():
        try:
            pid = int(lock_path.read_text().strip())
            # Check if process is still alive
            os.kill(pid, 0)
            return None  # Process is still running
        except (ValueError, OSError, ProcessLookupError):
            # Lock file is stale, remove it
            lock_path.unlink(missing_ok=True)

    # Create lock file with our PID
    lock_path.write_text(str(os.getpid()))
    return lock_path


def release_lock(lock_path: Path) -> None:
    """Release the lock file."""
    try:
        lock_path.unlink(missing_ok=True)
    except OSError:
        pass


def run_daemon(verbose: bool = False) -> None:
    """Run as a daemon - watch queue and process items immediately."""
    lock_path = acquire_lock()
    if lock_path is None:
        print("[ERROR] Another speeker-player daemon is already running", file=sys.stderr)
        sys.exit(1)

    if verbose:
        print("[INFO] Speeker player daemon starting...", file=sys.stderr)

    # Pre-warm the TTS model
    if verbose:
        print("[INFO] Warming up TTS model...", file=sys.stderr)
    get_tts_model()
    get_voice_state("azelma")  # Default voice
    if verbose:
        print("[INFO] TTS model ready!", file=sys.stderr)

    last_activity = time.time()

    try:
        while True:
            pending = get_pending_count()

            if pending > 0:
                if verbose:
                    print(f"[INFO] Processing {pending} pending item(s)", file=sys.stderr)
                process_queue(verbose)
                last_activity = time.time()
            else:
                # Check for idle timeout
                if time.time() - last_activity > IDLE_TIMEOUT:
                    if verbose:
                        print("[INFO] Idle timeout, exiting", file=sys.stderr)
                    break

            time.sleep(POLL_INTERVAL)
    finally:
        release_lock(lock_path)


def run_once(verbose: bool = False) -> None:
    """Run once - process queue and exit."""
    if verbose:
        print("[INFO] Speeker player (one-shot mode)", file=sys.stderr)

    played = process_queue(verbose)

    if verbose:
        print(f"[INFO] Done. Played {played} utterance(s)", file=sys.stderr)


def cleanup_old_files(days: int, verbose: bool = False) -> int:
    """Remove old database entries."""
    removed = cleanup_old_entries(days)
    if verbose:
        print(f"[CLEANUP] Removed {removed} database entries", file=sys.stderr)
    return removed


def main() -> int:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="speeker-player",
        description="Speeker TTS playback daemon",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--daemon", "-d", action="store_true",
                        help="Run as daemon (watch queue, keep model warm)")
    parser.add_argument("--cleanup", type=int, metavar="DAYS",
                        help="Remove entries older than DAYS days and exit")

    args = parser.parse_args()
    get_base_dir().mkdir(parents=True, exist_ok=True)

    if args.cleanup is not None:
        removed = cleanup_old_files(args.cleanup, args.verbose)
        print(f"Removed {removed} item(s)", file=sys.stderr)
        return 0

    if args.daemon:
        run_daemon(args.verbose)
    else:
        run_once(args.verbose)

    return 0


if __name__ == "__main__":
    sys.exit(main())
