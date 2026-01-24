#!/usr/bin/env python3
"""Test preprocessing by comparing TTS output via STT.

Generates audio from preprocessed text, transcribes it with Whisper,
and checks for key words/phrases that should appear.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from speeker.preprocessing import preprocess_for_tts
from speeker.cli import generate_pocket_tts


# Test cases: (input_text, required_words, forbidden_words)
# required_words: at least 80% of these should appear in transcription
# forbidden_words: none of these should appear in transcription
TEST_CASES = [
    # Arrows - check that arrow symbols are converted
    ("Navigate home → settings", ["navigate", "home", "to", "settings"], ["→"]),
    ("Go ← back", ["go", "from", "back"], ["←"]),

    # Math symbols
    ("Value ≈ 95%", ["value", "approximately", "95"], ["≈", "%"]),
    ("x ≠ y", ["not", "equal"], ["≠"]),
    ("72°F", ["72", "degrees"], ["°"]),

    # Abbreviations
    ("e.g. this works", ["for", "example", "this", "works"], []),
    ("i.e. correct", ["that", "is", "correct"], []),
    ("vs. that", ["versus", "that"], []),

    # File paths
    ("Check ~/projects", ["check", "home", "slash", "projects"], ["~", "/"]),
    ("Run ../script", ["run", "dot", "slash", "script"], [".."]),

    # Operators
    ("a == b", ["equals"], ["=="]),
    ("a != b", ["not", "equals"], ["!="]),
    ("combine with && logic", ["combine", "with", "and", "logic"], ["&&"]),

    # Version numbers
    ("version 3.11", ["version", "3", "11"], []),  # "point" may or may not appear

    # Mixed real-world example
    ("Updated config → done. Status ≈ 100%",
     ["updated", "config", "to", "done", "status", "approximately", "100"],
     ["→", "≈", "%"]),
]


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    import re
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_words(text: str) -> set[str]:
    """Get set of words from text."""
    return set(normalize_text(text).split())


def test_case(
    model: WhisperModel,
    input_text: str,
    required_words: list[str],
    forbidden_words: list[str],
    verbose: bool = True
) -> dict:
    """Test a single case."""
    # Preprocess
    preprocessed = preprocess_for_tts(input_text)

    # Generate audio
    audio, sample_rate = generate_pocket_tts(preprocessed, "azelma")

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        audio_normalized = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio_normalized * 32767).astype(np.int16)
        wavfile.write(f.name, sample_rate, audio_int16)
        temp_path = f.name

    # Transcribe
    segments, _ = model.transcribe(temp_path, language="en")
    transcription = " ".join(seg.text for seg in segments)

    # Clean up
    Path(temp_path).unlink()

    # Check words
    transcribed_words = get_words(transcription)
    required_set = set(w.lower() for w in required_words)
    forbidden_set = set(w.lower() for w in forbidden_words)

    found_required = required_set & transcribed_words
    missing_required = required_set - transcribed_words
    found_forbidden = forbidden_set & transcribed_words

    # Pass if 80%+ required words found and no forbidden words
    required_ratio = len(found_required) / len(required_set) if required_set else 1.0
    passed = required_ratio >= 0.8 and len(found_forbidden) == 0

    result = {
        "input": input_text,
        "preprocessed": preprocessed,
        "transcription": transcription.strip(),
        "required_words": required_words,
        "found_required": list(found_required),
        "missing_required": list(missing_required),
        "found_forbidden": list(found_forbidden),
        "required_ratio": required_ratio,
        "passed": passed,
    }

    if verbose:
        status = "✓" if passed else "✗"
        print(f"\n{status} Input: {input_text}")
        print(f"  Preprocessed:  {preprocessed}")
        print(f"  Transcribed:   {transcription.strip()}")
        print(f"  Required ({len(found_required)}/{len(required_set)}): {', '.join(sorted(found_required)) or 'none'}")
        if missing_required:
            print(f"  Missing:       {', '.join(sorted(missing_required))}")
        if found_forbidden:
            print(f"  Forbidden:     {', '.join(sorted(found_forbidden))}")

    return result


def main():
    print("Loading Whisper model...")
    model = WhisperModel("base.en", device="cpu", compute_type="int8")

    print(f"\nTesting {len(TEST_CASES)} cases...\n")

    results = []
    for input_text, required, forbidden in TEST_CASES:
        result = test_case(model, input_text, required, forbidden)
        results.append(result)

    # Summary
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    avg_ratio = sum(r["required_ratio"] for r in results) / total

    print(f"\n{'='*60}")
    print(f"Results: {passed}/{total} passed ({passed/total:.0%})")
    print(f"Average word coverage: {avg_ratio:.0%}")

    if passed < total:
        print(f"\nFailed cases need investigation:")
        for r in results:
            if not r["passed"]:
                print(f"  - {r['input']}")
                if r["missing_required"]:
                    print(f"    Missing: {r['missing_required']}")
                if r["found_forbidden"]:
                    print(f"    Forbidden: {r['found_forbidden']}")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
