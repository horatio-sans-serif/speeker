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
    get_last_utterance_time,
    get_pending_count,
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
    """Generate TTS audio for text. Returns path to temp audio file."""
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
        voice = get_default_voice("pocket-tts")
        voice_state = get_voice_state(voice)
        audio = model.generate_audio(voice_state, processed)

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            audio_normalized = np.clip(audio.numpy(), -1.0, 1.0)
            audio_int16 = (audio_normalized * 32767).astype(np.int16)
            wavfile.write(f.name, model.sample_rate, audio_int16)
            return Path(f.name)

    except Exception as e:
        if verbose:
            print(f"[ERROR] TTS failed: {e}", file=sys.stderr)
        return None


def speak_text(text: str, verbose: bool = False) -> bool:
    """Generate and play TTS for text. Returns True on success."""
    audio_path = generate_tts(text, verbose)
    if audio_path is None:
        return False

    try:
        return play_audio(audio_path, verbose)
    finally:
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


def process_queue(verbose: bool = False) -> int:
    """Process all pending messages. Returns count of messages played."""
    sessions = get_sessions_with_pending()
    if not sessions:
        return 0

    # Count total messages
    total_messages = sum(len(get_pending_for_session(s)) for s in sessions)
    is_single_message = total_messages == 1

    total_played = 0

    # Intro announcement if needed
    if should_announce_intro() and not is_single_message:
        if verbose:
            print("[INFO] Announcing intro", file=sys.stderr)
        speak_text("This is Claude Code.", verbose)
        time.sleep(PAUSE_BETWEEN_SESSIONS)

    # Process each session
    is_only_session = len(sessions) == 1
    for session_idx, session_id in enumerate(sessions):
        items = get_pending_for_session(session_id)
        if not items:
            continue

        if session_idx > 0:
            time.sleep(PAUSE_BETWEEN_SESSIONS)

        script_lines = build_session_script(session_id, items, is_only_session)

        for line_idx, line in enumerate(script_lines):
            if line_idx > 0:
                time.sleep(PAUSE_BETWEEN_MESSAGES)

            if speak_text(line, verbose):
                total_played += 1

        # Mark items as played
        for item in items:
            mark_played(item["id"])

    # Outro (skip if single message)
    if total_played > 0 and not is_single_message:
        time.sleep(PAUSE_BETWEEN_SESSIONS)
        speak_text("That is all.", verbose)

    if total_played > 0:
        set_last_utterance_time()

    return total_played


def run_daemon(verbose: bool = False) -> None:
    """Run as a daemon - watch queue and process items immediately."""
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
