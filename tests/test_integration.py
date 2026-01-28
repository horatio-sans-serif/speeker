#!/usr/bin/env python3
"""Integration tests for speeker TTS generation and playback.

These tests require the actual TTS models to be available.
They will be skipped if models cannot be loaded.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch
import pytest
import numpy as np

# Check if TTS models are available
def _check_pocket_tts():
    try:
        from pocket_tts import TTSModel
        return True
    except ImportError:
        return False

def _check_kokoro():
    try:
        from kokoro import KPipeline
        return True
    except ImportError:
        return False

HAS_POCKET_TTS = _check_pocket_tts()
HAS_KOKORO = _check_kokoro()

skip_no_pocket_tts = pytest.mark.skipif(
    not HAS_POCKET_TTS,
    reason="pocket-tts not available"
)

skip_no_kokoro = pytest.mark.skipif(
    not HAS_KOKORO,
    reason="kokoro not available"
)


@pytest.fixture
def temp_speeker_dir(tmp_path):
    """Create a temporary speeker directory."""
    with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
        yield tmp_path


class TestPocketTTSGeneration:
    """Integration tests for pocket-tts generation."""

    @skip_no_pocket_tts
    def test_generate_pocket_tts_returns_audio(self):
        """Test pocket-tts generates audio array."""
        from speeker.cli import generate_pocket_tts

        audio, sample_rate = generate_pocket_tts("Hello world", "azelma")

        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0
        assert sample_rate > 0

    @skip_no_pocket_tts
    def test_generate_pocket_tts_different_voices(self):
        """Test pocket-tts works with different voices."""
        from speeker.cli import generate_pocket_tts
        from speeker.voices import POCKET_TTS_VOICES

        # Test first available voice
        voice = list(POCKET_TTS_VOICES.keys())[0]
        audio, sample_rate = generate_pocket_tts("Test", voice)

        assert len(audio) > 0

    @skip_no_pocket_tts
    def test_generate_pocket_tts_longer_text(self):
        """Test pocket-tts with longer text."""
        from speeker.cli import generate_pocket_tts

        text = "This is a longer piece of text that should generate more audio samples."
        audio, sample_rate = generate_pocket_tts(text, "azelma")

        # Longer text should produce more audio
        short_audio, _ = generate_pocket_tts("Hi", "azelma")
        assert len(audio) > len(short_audio)


class TestKokoroGeneration:
    """Integration tests for kokoro generation.

    Note: These tests require kokoro and its spacy model to be fully set up.
    They may be skipped if the environment is not ready.
    """

    @pytest.mark.skip(reason="Kokoro requires spacy model download - run manually")
    def test_generate_kokoro_returns_audio(self):
        """Test kokoro generates audio array."""
        from speeker.cli import generate_kokoro

        audio, sample_rate = generate_kokoro("Hello world", "af_bella")

        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0
        assert sample_rate == 24000  # Kokoro uses 24kHz

    @pytest.mark.skip(reason="Kokoro requires spacy model download - run manually")
    def test_generate_kokoro_different_voices(self):
        """Test kokoro works with different voices."""
        from speeker.cli import generate_kokoro
        from speeker.voices import KOKORO_VOICES

        # Test first available voice
        voice = list(KOKORO_VOICES.keys())[0]
        audio, sample_rate = generate_kokoro("Test", voice)

        assert len(audio) > 0


class TestAudioSaving:
    """Integration tests for audio file saving."""

    @skip_no_pocket_tts
    def test_save_audio_creates_file(self, temp_speeker_dir):
        """Test save_audio creates an audio file."""
        from speeker.cli import generate_pocket_tts, save_audio

        audio, sample_rate = generate_pocket_tts("Test message", "azelma")
        audio_path = save_audio(audio, sample_rate, "Test message")

        assert audio_path.exists()
        assert audio_path.suffix in [".wav", ".mp3"]

    @skip_no_pocket_tts
    def test_save_audio_creates_txt_file(self, temp_speeker_dir):
        """Test save_audio creates accompanying text file."""
        from speeker.cli import generate_pocket_tts, save_audio

        audio, sample_rate = generate_pocket_tts("Test message", "azelma")
        audio_path = save_audio(audio, sample_rate, "Test message")

        txt_path = audio_path.with_suffix(".txt")
        assert txt_path.exists()
        assert txt_path.read_text() == "Test message"

    @skip_no_pocket_tts
    def test_save_audio_unique_filenames(self, temp_speeker_dir):
        """Test save_audio creates unique filenames."""
        from speeker.cli import generate_pocket_tts, save_audio

        audio, sample_rate = generate_pocket_tts("Test", "azelma")

        path1 = save_audio(audio, sample_rate, "Test 1")
        path2 = save_audio(audio, sample_rate, "Test 2")

        assert path1 != path2
        assert path1.exists()
        assert path2.exists()


class TestToneGeneration:
    """Integration tests for tone generation in player."""

    def test_generate_tone_creates_file(self, temp_speeker_dir):
        """Test generate_tone creates a wav file."""
        from speeker.player import generate_tone

        tone_path = generate_tone([440, 554, 659], rising=True)

        assert tone_path.exists()
        assert tone_path.suffix == ".wav"

    def test_generate_tone_intro_outro(self, temp_speeker_dir):
        """Test generate_tone creates intro and outro files."""
        from speeker.player import generate_tone

        intro_path = generate_tone([440, 554, 659], rising=True)
        outro_path = generate_tone([440, 554, 659], rising=False)

        assert "intro" in str(intro_path)
        assert "outro" in str(outro_path)

    def test_generate_tone_caches_file(self, temp_speeker_dir):
        """Test generate_tone caches the file."""
        from speeker.player import generate_tone

        path1 = generate_tone([440], rising=True)
        path2 = generate_tone([440], rising=True)

        # Should return same path (cached)
        assert path1 == path2

    def test_generate_tone_valid_wav(self, temp_speeker_dir):
        """Test generated tone is a valid WAV file."""
        from speeker.player import generate_tone
        from scipy.io import wavfile

        tone_path = generate_tone([440, 880], rising=True)

        # Should be readable as WAV
        sample_rate, audio = wavfile.read(tone_path)
        assert sample_rate == 44100
        assert len(audio) > 0


class TestCombinedToneGeneration:
    """Integration tests for combined tone generation."""

    def test_generate_combined_tones(self, temp_speeker_dir):
        """Test generating combined tones from tokens."""
        from speeker.player import generate_combined_tones_from_tokens

        tone_path = generate_combined_tones_from_tokens(["C4", "E4", "G4"])

        assert tone_path.exists()
        assert tone_path.suffix == ".wav"

    def test_generate_combined_tones_single(self, temp_speeker_dir):
        """Test generating tone from single token."""
        from speeker.player import generate_combined_tones_from_tokens

        tone_path = generate_combined_tones_from_tokens(["A4"])

        assert tone_path.exists()

    def test_generate_combined_tones_with_accidentals(self, temp_speeker_dir):
        """Test generating tones with sharps and flats."""
        from speeker.player import generate_combined_tones_from_tokens

        tone_path = generate_combined_tones_from_tokens(["F#4", "Bb3"])

        assert tone_path.exists()


class TestPreprocessingIntegration:
    """Integration tests for preprocessing with TTS."""

    @skip_no_pocket_tts
    def test_preprocessed_text_generates_audio(self):
        """Test preprocessed text can be spoken."""
        from speeker.cli import generate_pocket_tts
        from speeker.preprocessing import preprocess_for_tts

        original = "Check the file at ~/projects/test.py"
        preprocessed = preprocess_for_tts(original)

        # Should not crash and should produce audio
        audio, sample_rate = generate_pocket_tts(preprocessed, "azelma")
        assert len(audio) > 0

    @skip_no_pocket_tts
    def test_special_chars_handled(self):
        """Test special characters are handled in TTS."""
        from speeker.cli import generate_pocket_tts
        from speeker.preprocessing import preprocess_for_tts

        texts = [
            "Navigate home → settings",
            "Value ≈ 95%",
            "Error in /src/main.py",
            "Use && to chain commands",
        ]

        for text in texts:
            preprocessed = preprocess_for_tts(text)
            audio, _ = generate_pocket_tts(preprocessed, "azelma")
            assert len(audio) > 0


class TestQueueIntegration:
    """Integration tests for queue operations with TTS."""

    def test_enqueue_and_retrieve(self, temp_speeker_dir):
        """Test enqueueing and retrieving items."""
        from speeker.queue_db import enqueue, get_pending_for_session, mark_played

        # Enqueue a message with session_id
        queue_id = enqueue("Test message", session_id="test")

        assert queue_id > 0

        # Retrieve pending items
        pending = get_pending_for_session("test")
        assert len(pending) == 1
        assert pending[0]["text"] == "Test message"

        # Mark as played
        mark_played(queue_id)

        pending = get_pending_for_session("test")
        assert len(pending) == 0

    @skip_no_pocket_tts
    def test_full_tts_pipeline(self, temp_speeker_dir):
        """Test full TTS pipeline: enqueue, generate, save, mark played."""
        from speeker.queue_db import enqueue, get_pending_for_session, mark_played
        from speeker.cli import generate_pocket_tts, save_audio

        # Enqueue
        text = "Integration test message"
        queue_id = enqueue(text, session_id="integration")

        # Get pending
        pending = get_pending_for_session("integration")
        assert len(pending) == 1

        # Generate audio
        audio, sample_rate = generate_pocket_tts(pending[0]["text"], "azelma")

        # Save audio
        audio_path = save_audio(audio, sample_rate, text)
        assert audio_path.exists()

        # Mark played
        mark_played(queue_id)

        # Verify complete
        pending = get_pending_for_session("integration")
        assert len(pending) == 0


class TestModelCaching:
    """Integration tests for TTS model caching."""

    @skip_no_pocket_tts
    def test_pocket_tts_model_cached(self):
        """Test pocket-tts model is cached between calls."""
        from speeker.cli import get_pocket_tts_model

        model1 = get_pocket_tts_model()
        model2 = get_pocket_tts_model()

        # Should be same object (cached)
        assert model1 is model2

    @skip_no_pocket_tts
    def test_voice_state_cached(self):
        """Test voice state is cached between calls."""
        from speeker.cli import get_pocket_tts_voice_state

        state1 = get_pocket_tts_voice_state("azelma")
        state2 = get_pocket_tts_voice_state("azelma")

        # Should be same object (cached)
        assert state1 is state2

    @pytest.mark.skip(reason="Kokoro requires spacy model download - run manually")
    def test_kokoro_pipeline_cached(self):
        """Test kokoro pipeline is cached between calls."""
        from speeker.cli import get_kokoro_pipeline

        pipeline1 = get_kokoro_pipeline()
        pipeline2 = get_kokoro_pipeline()

        # Should be same object (cached)
        assert pipeline1 is pipeline2
