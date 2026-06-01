#!/usr/bin/env -S uv run python
"""End-to-end audio + STT test for speeker.

Exercises the real installed daemon (the one launchd runs, not the dev venv)
by:

  1. POSTing a known phrase to ``/speak`` on the running server.
  2. Polling the queue.db until the player marks it played AND records an
     audio_path -- this is the load-bearing check, because every "Polly has
     no credentials" / "boto3 not installed" failure has the player silently
     swallow the error and leave audio_path empty.
  3. Reading the generated WAV and running DICTATOR's Voxtral (Mistral)
     STT against it. Whisper is excluded by user policy ("No OpenAI
     products" -- see ``MEMORY.md``).
  4. Comparing the STT transcript to the expected phrase with a
     normalize-and-substring check.

Why this exists: until now the project shipped with only unit-level
verification (mocked engines). The tool-venv vs dev-venv split, the AWS
credentials chain, and the boto3 optional dep all broke the production
audio chain in ways that no unit test could catch. This script is the
gate that runs before any "Done" claim involving audible behavior.

Usage:
    uv run scripts/e2e_stt_test.py                  # default phrase
    uv run scripts/e2e_stt_test.py "compass north"  # custom phrase
    uv run scripts/e2e_stt_test.py --strict         # exact transcript match

Exit code: 0 on pass, 1 on any failure (HTTP, timeout, missing audio,
transcript mismatch).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

SPEEKER_SERVER = "http://127.0.0.1:7849"
QUEUE_DB = Path.home() / "Library" / "Application Support" / "speeker" / "queue.db"
DICTATOR_ROOT = Path.home() / "projects" / "audio" / "dictator"
WAIT_FOR_AUDIO_TIMEOUT = 45.0  # seconds; cold Polly call can take a few sec
POLL_INTERVAL = 0.3


def post_speak(text: str, queue: str) -> int:
    """POST to /speak. Returns the queue row id; raises on HTTP failure."""
    body = json.dumps({"text": text, "metadata": {"queue": queue}}).encode("utf-8")
    req = urllib.request.Request(
        f"{SPEEKER_SERVER}/speak",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        payload = json.loads(resp.read())
    if payload.get("status") != "success":
        raise RuntimeError(f"/speak returned non-success: {payload}")
    return int(payload["queue_id"])


def wait_for_audio(queue_id: int, timeout: float) -> Path:
    """Poll the queue.db until the row has a non-empty audio_path. Returns it.

    Raises ``TimeoutError`` if the daemon doesn't write an audio path within
    *timeout* seconds. Empty audio_path after played_at is the canonical
    "silent failure" signature (Polly threw and ``generate_tts`` swallowed
    the exception).
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with sqlite3.connect(str(QUEUE_DB)) as conn:
            cur = conn.execute(
                "SELECT played_at, audio_path FROM queue WHERE id = ?", (queue_id,)
            )
            row = cur.fetchone()
        if row is not None:
            played_at, audio_path = row
            if played_at and audio_path:
                return Path(audio_path)
            if played_at and not audio_path:
                raise RuntimeError(
                    f"Queue row {queue_id} was marked played but no audio_path "
                    f"was written. This is the classic 'engine silently failed' "
                    f"signature -- check /tmp/speeker-player.err and confirm "
                    f"the engine venv has its dependencies (boto3 for polly)."
                )
        time.sleep(POLL_INTERVAL)
    raise TimeoutError(
        f"Daemon did not produce audio for queue id {queue_id} within {timeout}s. "
        f"Is the player daemon running? `ps aux | grep speeker-player`."
    )


def transcribe(audio_path: Path) -> str:
    """Run DICTATOR's Voxtral (Mistral) STT on the WAV. Returns transcript text.

    Uses ``--engine voxtral`` because the user's policy excludes OpenAI's
    Whisper. Requires ``MISTRAL_API_KEY`` in the environment, which the
    user already has set globally.
    """
    if "MISTRAL_API_KEY" not in os.environ or not os.environ["MISTRAL_API_KEY"]:
        raise RuntimeError("MISTRAL_API_KEY not set; cannot run Voxtral STT.")

    # Run dictator in transcribe-only mode so the curses UI doesn't open and
    # the diarized transcript streams to stdout. Run from the dictator project
    # root so its uv environment is used (it has the voxtral client installed).
    result = subprocess.run(
        ["uv", "run", "dictator", "-T", "--engine", "voxtral", "--no-sentiment",
         "--no-sidecar", str(audio_path)],
        cwd=str(DICTATOR_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"dictator failed (exit {result.returncode}):\n"
            f"--- stderr ---\n{result.stderr}\n"
            f"--- stdout ---\n{result.stdout}"
        )
    # Voxtral's diarized output uses two formats depending on version:
    #   "SPEAKER 1 (00:00-00:07): hello world"
    #   "speaker_00: hello world"
    # Strip whichever shape it produced and keep just the words.
    transcript_lines = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        # Bracketed timestamps anywhere at the start: "[00:00]"
        line = re.sub(r"^\[[^\]]+\]\s*", "", line)
        # "SPEAKER 1 (00:00-00:07):" or "speaker_00:" (case-insensitive)
        line = re.sub(
            r"^speaker[\s_]*\d+\s*(?:\([^)]*\))?\s*:\s*",
            "",
            line,
            flags=re.IGNORECASE,
        )
        if line:
            transcript_lines.append(line)
    return " ".join(transcript_lines)


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace. STT-vs-input compare."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def words(text: str) -> list[str]:
    return normalize(text).split()


def assert_transcript_matches(expected: str, actual: str, strict: bool) -> None:
    """Assert STT transcript contains the expected phrase. Raises on mismatch.

    In non-strict mode, every word of the expected phrase must appear (in
    order) in the actual transcript -- this tolerates STT inserting
    punctuation/filler and gives a sane signal even when Voxtral capitalizes
    differently. Strict mode requires exact normalized equality.
    """
    expected_norm = normalize(expected)
    actual_norm = normalize(actual)

    if strict:
        if expected_norm != actual_norm:
            raise AssertionError(
                f"Transcript mismatch (strict):\n"
                f"  expected: {expected_norm!r}\n"
                f"  actual:   {actual_norm!r}"
            )
        return

    # Order-preserving subsequence match on words.
    expected_words = words(expected)
    actual_words = words(actual)
    i = 0
    for w in actual_words:
        if i < len(expected_words) and w == expected_words[i]:
            i += 1
    if i != len(expected_words):
        missing = expected_words[i:]
        raise AssertionError(
            f"Transcript missing words {missing!r}:\n"
            f"  expected: {expected_norm!r}\n"
            f"  actual:   {actual_norm!r}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "phrase", nargs="?",
        # Use a phrase that consists only of real English words. The project
        # name "speeker" is intentionally a misspelling of "speaker", which
        # STT will always normalize -- including it in the default test
        # phrase produced a false-negative "speeker"-vs-"speaker" mismatch.
        default="The compass shows progress on the project today.",
        help="Phrase to synthesize, transcribe, and verify.",
    )
    parser.add_argument("--strict", action="store_true",
                        help="Require exact normalized transcript match.")
    parser.add_argument("--keep-audio", action="store_true",
                        help="Leave the generated WAV in place after the test.")
    args = parser.parse_args()

    # Use the "default" queue so the auto-label path doesn't prepend a
    # spoken project name to the phrase under test -- that would pollute
    # the transcript and turn the assertion into a test of the auto-label
    # path rather than of the phrase itself.
    queue = "default"
    print(f"[1/4] Posting to /speak  queue={queue!r}  text={args.phrase!r}")
    try:
        queue_id = post_speak(args.phrase, queue)
    except Exception as e:
        print(f"  FAIL: {e}", file=sys.stderr)
        return 1
    print(f"      queue_id={queue_id}")

    print(f"[2/4] Waiting for daemon to write audio_path (timeout={WAIT_FOR_AUDIO_TIMEOUT}s)...")
    try:
        audio_path = wait_for_audio(queue_id, WAIT_FOR_AUDIO_TIMEOUT)
    except Exception as e:
        print(f"  FAIL: {e}", file=sys.stderr)
        return 1
    size = audio_path.stat().st_size
    print(f"      audio_path={audio_path}  ({size} bytes)")
    if size < 1000:
        print(f"  FAIL: audio file is suspiciously small ({size} bytes)", file=sys.stderr)
        return 1

    print(f"[3/4] Transcribing via DICTATOR --engine voxtral ...")
    try:
        transcript = transcribe(audio_path)
    except Exception as e:
        print(f"  FAIL: {e}", file=sys.stderr)
        return 1
    print(f"      transcript={transcript!r}")

    print(f"[4/4] Verifying transcript matches expected phrase ({'strict' if args.strict else 'subsequence'})...")
    try:
        assert_transcript_matches(args.phrase, transcript, args.strict)
    except AssertionError as e:
        print(f"  FAIL: {e}", file=sys.stderr)
        return 1
    print("      PASS")

    if not args.keep_audio:
        # Leave it on disk -- speeker's audio_dir is the source of truth for
        # the history UI. Removing the file would break that link.
        pass

    print()
    print("E2E PASS.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
