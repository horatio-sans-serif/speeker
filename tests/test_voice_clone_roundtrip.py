#!/usr/bin/env python3
"""Round-trip test: clone voice -> TTS -> STT -> verify output.

Generates a synthetic reference audio, clones it as a custom voice,
generates TTS with that voice, transcribes the output with faster-whisper,
and checks that the spoken text matches the input.

Requires: pocket-tts (with gated weights), faster-whisper
"""

import shutil
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from speeker import voice_clone
from speeker.voices import get_pocket_tts_voice_path


@pytest.fixture
def voices_dir(tmp_path, monkeypatch):
    """Isolated voices directory."""
    monkeypatch.setenv("SPEEKER_DIR", str(tmp_path))
    vdir = tmp_path / "data" / "voices"
    vdir.mkdir(parents=True)
    return vdir


def _generate_reference_audio(path: Path) -> Path:
    """Generate a reference audio clip using a built-in voice.

    We use pocket-tts with a built-in voice to create a clean reference
    clip, then use that as the cloning source. This tests the cloning
    pipeline without needing external audio files.
    """
    from speeker.cli import generate_pocket_tts

    text = "The quick brown fox jumps over the lazy dog."
    audio, sample_rate = generate_pocket_tts(text, "azelma")

    audio_normalized = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio_normalized * 32767).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(path), sample_rate, audio_int16)
    return path


@pytest.mark.slow
def test_cloned_voice_tts_roundtrip(voices_dir, tmp_path):
    """Clone a voice from generated audio, use it for TTS, verify via STT."""
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")

    # Step 1: Generate a reference audio clip
    ref_audio = tmp_path / "reference.wav"
    _generate_reference_audio(ref_audio)
    assert ref_audio.exists()

    # Step 2: Clone it as a custom voice
    voice_clone.clone_voice(
        name="Test Clone",
        sources=[str(ref_audio)],
        start_secs=0,
        duration_secs=10,
        description="Test cloned voice for round-trip",
    )

    # Verify the voice is registered
    assert voice_clone.get_custom_voice_path("Test Clone") is not None
    voice_path = get_pocket_tts_voice_path("Test Clone")
    assert isinstance(voice_path, Path)

    # Step 3: Generate TTS with the cloned voice
    from pocket_tts import TTSModel
    from speeker.preprocessing import preprocess_for_tts

    model = TTSModel.load_model()
    voice_state = model.get_state_for_audio_prompt(voice_path)

    test_text = "Hello, this is a test of voice cloning."
    processed = preprocess_for_tts(test_text)
    audio = model.generate_audio(voice_state, processed)

    # Save TTS output
    tts_output = tmp_path / "tts_output.wav"
    audio_np = audio.numpy()
    audio_normalized = np.clip(audio_np, -1.0, 1.0)
    audio_int16 = (audio_normalized * 32767).astype(np.int16)
    wavfile.write(str(tts_output), model.sample_rate, audio_int16)

    assert tts_output.exists()
    assert tts_output.stat().st_size > 1000  # Should have actual audio content

    # Step 4: Transcribe with faster-whisper
    from faster_whisper import WhisperModel

    whisper = WhisperModel("base.en", device="cpu", compute_type="int8")
    segments, _ = whisper.transcribe(str(tts_output), language="en")
    transcription = " ".join(seg.text for seg in segments).strip().lower()

    # Step 5: Verify key words appear in transcription
    # We don't expect exact match -- TTS + STT introduces errors.
    # But key content words should survive the round-trip.
    expected_words = {"hello", "test", "voice", "cloning"}
    transcribed_words = set(transcription.split())

    found = expected_words & transcribed_words
    coverage = len(found) / len(expected_words)

    print(f"\nTranscription: {transcription}")
    print(f"Expected words found: {found} ({coverage:.0%})")

    # At least 50% of key words should survive the round-trip
    assert coverage >= 0.5, (
        f"Only {coverage:.0%} of expected words found. "
        f"Transcription: '{transcription}', Missing: {expected_words - transcribed_words}"
    )
