#!/usr/bin/env python3
"""Speeker playback - processes SQLite queue and plays audio with session announcements.

Supports macOS (afplay), Linux (paplay/aplay/ffplay), and Windows (PowerShell).
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

from .queue_db import (
    get_last_utterance_time,
    get_pending_for_session,
    get_sessions_with_pending,
    mark_played,
    relative_time,
    get_session_label,
    set_last_utterance_time,
    cleanup_old_entries,
)

# Default base directory
DEFAULT_BASE_DIR = Path.home() / ".speeker"

# Pause between messages
PAUSE_BETWEEN_MESSAGES = 0.3
PAUSE_BETWEEN_SESSIONS = 0.5

# How long before we re-announce "This is Claude Code"
ANNOUNCE_THRESHOLD_MINUTES = 30


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
            return None  # Special handling

    if shutil.which("ffplay"):
        return ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"]

    return None


AUDIO_PLAYER = get_audio_player()


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


def generate_tts(text: str, verbose: bool = False) -> Path | None:
    """Generate TTS audio for text. Returns path to audio file or None on failure."""
    try:
        # Import TTS components
        from .cli import get_pocket_tts_model, get_pocket_tts_voice_state
        from .voices import get_default_voice
        import numpy as np
        from scipy.io import wavfile

        if verbose:
            print(f"[TTS] Generating: {text[:50]}...", file=sys.stderr)

        # Generate audio
        model = get_pocket_tts_model()
        voice = get_default_voice("pocket-tts")
        voice_state = get_pocket_tts_voice_state(voice)
        audio = model.generate_audio(voice_state, text)

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            audio_normalized = np.clip(audio.numpy(), -1.0, 1.0)
            audio_int16 = (audio_normalized * 32767).astype(np.int16)
            wavfile.write(f.name, model.sample_rate, audio_int16)
            return Path(f.name)

    except Exception as e:
        if verbose:
            print(f"[ERROR] TTS generation failed: {e}", file=sys.stderr)
        return None


def speak_text(text: str, verbose: bool = False) -> bool:
    """Generate and play TTS for text. Returns True on success."""
    audio_path = generate_tts(text, verbose)
    if audio_path is None:
        return False

    try:
        success = play_audio(audio_path, verbose)
        return success
    finally:
        # Clean up temp file
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


def build_session_script(session_id: str, items: list[dict]) -> list[str]:
    """Build the speech script for a session's messages."""
    lines = []
    count = len(items)
    session_label = get_session_label(session_id)

    # Session header
    if count == 1:
        lines.append(f"For {session_label}, there is 1 message.")
    else:
        lines.append(f"For {session_label}, there are {count} messages.")

    # Each message
    for i, item in enumerate(items):
        time_ago = relative_time(item["created_at"])
        text = item["text"]

        if i == 0 and count > 1:
            prefix = "The first message"
        elif i == count - 1 and count > 1:
            prefix = "The last message"
        elif count == 1:
            prefix = "It"
        else:
            prefix = "The next message"

        lines.append(f"{prefix} was created {time_ago} and says: {text}")

    return lines


def run_player(verbose: bool = False) -> None:
    """Run the player - process all pending messages with announcements."""
    if verbose:
        print("[INFO] Speeker player started", file=sys.stderr)

    sessions = get_sessions_with_pending()
    if not sessions:
        if verbose:
            print("[INFO] No pending messages", file=sys.stderr)
        return

    total_played = 0

    # Intro announcement if needed
    if should_announce_intro():
        if verbose:
            print("[INFO] Announcing intro", file=sys.stderr)
        speak_text("This is Claude Code.", verbose)
        time.sleep(PAUSE_BETWEEN_SESSIONS)

    # Process each session
    for session_idx, session_id in enumerate(sessions):
        items = get_pending_for_session(session_id)
        if not items:
            continue

        if session_idx > 0:
            time.sleep(PAUSE_BETWEEN_SESSIONS)

        # Build and speak the script for this session
        script_lines = build_session_script(session_id, items)

        for line_idx, line in enumerate(script_lines):
            if line_idx > 0:
                time.sleep(PAUSE_BETWEEN_MESSAGES)

            if speak_text(line, verbose):
                total_played += 1

        # Mark all items as played
        for item in items:
            mark_played(item["id"])

    # Outro
    if total_played > 0:
        time.sleep(PAUSE_BETWEEN_SESSIONS)
        speak_text("That is all.", verbose)
        set_last_utterance_time()

    if verbose:
        print(f"[INFO] Speeker player done. Played {total_played} utterance(s)", file=sys.stderr)


def cleanup_old_files(days: int, verbose: bool = False) -> int:
    """Remove old audio files and database entries."""
    base_dir = get_base_dir()
    removed = 0

    # Clean database
    db_removed = cleanup_old_entries(days)
    if verbose and db_removed:
        print(f"[CLEANUP] Removed {db_removed} database entries", file=sys.stderr)

    # Clean audio files
    if base_dir.exists():
        from datetime import datetime as dt
        cutoff = dt.now() - timedelta(days=days)

        for day_dir in base_dir.iterdir():
            if not day_dir.is_dir():
                continue
            try:
                dir_date = dt.strptime(day_dir.name, "%Y-%m-%d")
            except ValueError:
                continue

            if dir_date < cutoff:
                if verbose:
                    print(f"[CLEANUP] Removing {day_dir}", file=sys.stderr)
                for f in day_dir.iterdir():
                    f.unlink()
                    removed += 1
                day_dir.rmdir()

    return removed + db_removed


def main() -> int:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="speeker-player",
        description="Speeker playback - processes queue with session announcements",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--cleanup",
        type=int,
        metavar="DAYS",
        help="Remove entries older than DAYS days and exit",
    )

    args = parser.parse_args()
    get_base_dir().mkdir(parents=True, exist_ok=True)

    if args.cleanup is not None:
        removed = cleanup_old_files(args.cleanup, args.verbose)
        print(f"[INFO] Removed {removed} item(s)", file=sys.stderr)
        return 0

    run_player(args.verbose)
    return 0


if __name__ == "__main__":
    sys.exit(main())
