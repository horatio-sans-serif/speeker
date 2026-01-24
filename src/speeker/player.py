#!/usr/bin/env python3
"""Speeker playback - processes queue and plays audio, then exits.

Supports macOS (afplay), Linux (paplay/aplay/ffplay), and Windows (PowerShell).
"""

import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Default base directory (can be overridden with SPEEKER_DIR env var)
DEFAULT_BASE_DIR = Path.home() / ".speeker"

PAUSE_BETWEEN = 0.5  # seconds between audio files


def get_base_dir() -> Path:
    """Get the base directory for speeker output."""
    return Path(os.environ.get("SPEEKER_DIR", DEFAULT_BASE_DIR))


def get_queue_file() -> Path:
    """Get the path to the global queue file."""
    return get_base_dir() / "queue"


def get_queue_processing() -> Path:
    """Get the path to the processing queue file."""
    return get_base_dir() / "queue.processing"


def get_all_queue_files() -> list[Path]:
    """Get all queue files (global + per-session), oldest first."""
    base_dir = get_base_dir()
    queues = []

    # Global queue
    global_queue = base_dir / "queue"
    if global_queue.exists():
        queues.append(global_queue)

    # Per-session queues
    sessions_dir = base_dir / "sessions"
    if sessions_dir.exists():
        for session_dir in sorted(sessions_dir.iterdir()):
            if session_dir.is_dir():
                session_queue = session_dir / "queue"
                if session_queue.exists():
                    queues.append(session_queue)

    return queues


def get_audio_player() -> list[str] | None:
    """Get the appropriate audio player command for this platform."""
    system = platform.system()

    if system == "Darwin":  # macOS
        if shutil.which("afplay"):
            return ["afplay"]
    elif system == "Linux":
        if shutil.which("paplay"):  # PulseAudio
            return ["paplay"]
        if shutil.which("aplay"):  # ALSA
            return ["aplay", "-q"]
        if shutil.which("ffplay"):  # FFmpeg
            return ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"]
    elif system == "Windows":
        if shutil.which("powershell"):
            return None  # Special handling for Windows

    # Fallback: try ffplay (cross-platform via FFmpeg)
    if shutil.which("ffplay"):
        return ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"]

    return None


AUDIO_PLAYER = get_audio_player()


def generate_tone() -> Path:
    """Generate a two-note tone (G2 E2) for transitions."""
    import math
    import struct
    import wave

    base_dir = get_base_dir()
    tone_path = base_dir / ".tone.wav"
    if tone_path.exists():
        return tone_path

    sample_rate = 44100
    amplitude = 0.3
    note_duration = 0.15
    gap_duration = 0.05

    notes = [98, 82]  # G2 = 98 Hz, E2 = 82 Hz
    samples = []

    for note_idx, frequency in enumerate(notes):
        n_samples = int(sample_rate * note_duration)
        fade_samples = int(sample_rate * 0.02)

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

        if note_idx < len(notes) - 1:
            gap_samples = int(sample_rate * gap_duration)
            samples.extend([0] * gap_samples)

    base_dir.mkdir(parents=True, exist_ok=True)
    with wave.open(str(tone_path), "w") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(struct.pack(f"{len(samples)}h", *samples))

    return tone_path


def play_audio(audio_path: Path, verbose: bool = False) -> bool:
    """Play an audio file using the system audio player. Returns True on success."""
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
            print(
                "[ERROR] No audio player found. Install ffmpeg, pulseaudio, or alsa-utils.",
                file=sys.stderr,
            )
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


def atomic_take_queue(queue_file: Path | None = None) -> list[str]:
    """Atomically take ownership of a queue file.

    Renames queue -> queue.processing, reads it, deletes it.
    Returns list of audio file paths.
    """
    if queue_file is None:
        queue_file = get_queue_file()
    queue_processing = queue_file.parent / f"{queue_file.name}.processing"

    if not queue_file.exists():
        return []

    try:
        os.rename(queue_file, queue_processing)
    except FileNotFoundError:
        return []
    except OSError as e:
        print(f"[ERROR] Failed to rename queue: {e}", file=sys.stderr)
        return []

    entries = []
    try:
        with open(queue_processing, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    entries.append(line)
    except OSError as e:
        print(f"[ERROR] Failed to read queue: {e}", file=sys.stderr)

    try:
        queue_processing.unlink()
    except OSError:
        pass

    return entries


def process_queue(verbose: bool = False) -> int:
    """Process all items in the queue. Returns number of items played."""
    entries = atomic_take_queue()
    if not entries:
        return 0

    tone_path = generate_tone()
    played = 0

    for i, entry in enumerate(entries):
        audio_path = Path(entry)

        if i > 0:
            time.sleep(PAUSE_BETWEEN)
            play_audio(tone_path, verbose=False)
            time.sleep(0.2)

        if play_audio(audio_path, verbose):
            played += 1

    return played


def process_session_queue(queue_file: Path, played_before: bool = False, verbose: bool = False) -> int:
    """Process all items in a specific queue. Returns number of items played."""
    entries = atomic_take_queue(queue_file)
    if not entries:
        return 0

    tone_path = generate_tone()
    played = 0

    for i, entry in enumerate(entries):
        audio_path = Path(entry)

        # Play tone before each item except the very first one ever
        if i > 0 or played_before:
            time.sleep(PAUSE_BETWEEN)
            play_audio(tone_path, verbose=False)
            time.sleep(0.2)

        if play_audio(audio_path, verbose):
            played += 1

    # Clean up empty session directory
    if queue_file.parent.name != ".speeker":
        try:
            if not any(queue_file.parent.iterdir()):
                queue_file.parent.rmdir()
        except OSError:
            pass

    return played


def run_player(verbose: bool = False) -> None:
    """Run the player loop - process all queues (global + per-session) until empty, then exit."""
    if verbose:
        print("[INFO] Speeker player started", file=sys.stderr)

    total_played = 0

    while True:
        queues = get_all_queue_files()
        if not queues:
            break

        played_this_round = 0
        for queue_file in queues:
            if verbose:
                print(f"[INFO] Processing queue: {queue_file}", file=sys.stderr)
            # Pass whether we've played anything before (for tone between batches)
            played = process_session_queue(queue_file, played_before=(total_played > 0), verbose=verbose)
            played_this_round += played
            total_played += played

        if played_this_round == 0:
            # Double-check no new queues appeared
            if not get_all_queue_files():
                break

    if verbose:
        print(f"[INFO] Speeker player done. Played {total_played} file(s)", file=sys.stderr)


def cleanup_old_files(days: int, verbose: bool = False) -> int:
    """Remove audio files older than N days. Returns number of files removed."""
    base_dir = get_base_dir()
    cutoff = datetime.now() - timedelta(days=days)
    removed = 0

    if not base_dir.exists():
        return 0

    for day_dir in base_dir.iterdir():
        if not day_dir.is_dir():
            continue
        try:
            dir_date = datetime.strptime(day_dir.name, "%Y-%m-%d")
        except ValueError:
            continue

        if dir_date < cutoff:
            if verbose:
                print(f"[CLEANUP] Removing {day_dir}", file=sys.stderr)
            for f in day_dir.iterdir():
                f.unlink()
                removed += 1
            day_dir.rmdir()

    return removed


def main() -> int:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="speeker-player",
        description="Speeker playback - processes queue and exits when empty",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print verbose output",
    )
    parser.add_argument(
        "--cleanup",
        type=int,
        metavar="DAYS",
        help="Remove audio files older than DAYS days and exit",
    )

    args = parser.parse_args()

    get_base_dir().mkdir(parents=True, exist_ok=True)

    if args.cleanup is not None:
        removed = cleanup_old_files(args.cleanup, args.verbose)
        print(f"[INFO] Removed {removed} file(s)", file=sys.stderr)
        return 0

    run_player(args.verbose)
    return 0


if __name__ == "__main__":
    sys.exit(main())
