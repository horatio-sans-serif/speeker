#!/usr/bin/env python3
"""Unit tests for voices.py functions."""

from speeker.voices import (
    POCKET_TTS_VOICES,
    KOKORO_VOICES,
    DEFAULT_ENGINE,
    DEFAULT_POCKET_TTS_VOICE,
    DEFAULT_KOKORO_VOICE,
    get_voices,
    get_default_voice,
    validate_voice,
    get_pocket_tts_voice_path,
)


class TestVoiceConstants:
    """Tests for voice constant definitions."""

    def test_pocket_tts_voices_not_empty(self):
        """Test pocket-tts voices dict is populated."""
        assert len(POCKET_TTS_VOICES) > 0

    def test_kokoro_voices_not_empty(self):
        """Test kokoro voices dict is populated."""
        assert len(KOKORO_VOICES) > 0

    def test_default_engine_is_valid(self):
        """Test default engine is a known engine."""
        assert DEFAULT_ENGINE in ["pocket-tts", "kokoro"]

    def test_default_pocket_tts_voice_is_valid(self):
        """Test default pocket-tts voice exists in voice list."""
        assert DEFAULT_POCKET_TTS_VOICE in POCKET_TTS_VOICES

    def test_default_kokoro_voice_is_valid(self):
        """Test default kokoro voice exists in voice list."""
        assert DEFAULT_KOKORO_VOICE in KOKORO_VOICES

    def test_all_pocket_tts_voices_have_descriptions(self):
        """Test all pocket-tts voices have non-empty descriptions."""
        for voice, desc in POCKET_TTS_VOICES.items():
            assert isinstance(voice, str) and len(voice) > 0
            assert isinstance(desc, str) and len(desc) > 0

    def test_all_kokoro_voices_have_descriptions(self):
        """Test all kokoro voices have non-empty descriptions."""
        for voice, desc in KOKORO_VOICES.items():
            assert isinstance(voice, str) and len(voice) > 0
            assert isinstance(desc, str) and len(desc) > 0


class TestGetVoices:
    """Tests for get_voices function."""

    def test_get_voices_no_filter_returns_both_engines(self):
        """Test no engine filter returns both engines."""
        result = get_voices()
        assert "pocket-tts" in result
        assert "kokoro" in result

    def test_get_voices_none_filter_returns_both(self):
        """Test None engine filter returns both engines."""
        result = get_voices(engine=None)
        assert "pocket-tts" in result
        assert "kokoro" in result

    def test_get_voices_pocket_tts_filter(self):
        """Test pocket-tts filter returns only pocket-tts."""
        result = get_voices(engine="pocket-tts")
        assert "pocket-tts" in result
        assert "kokoro" not in result
        assert result["pocket-tts"] == POCKET_TTS_VOICES

    def test_get_voices_kokoro_filter(self):
        """Test kokoro filter returns only kokoro."""
        result = get_voices(engine="kokoro")
        assert "kokoro" in result
        assert "pocket-tts" not in result
        assert result["kokoro"] == KOKORO_VOICES

    def test_get_voices_unknown_engine_returns_empty(self):
        """Test unknown engine returns empty dict."""
        result = get_voices(engine="unknown-engine")
        assert result == {}

    def test_get_voices_returns_dict_of_dicts(self):
        """Test return type is dict of dicts."""
        result = get_voices()
        assert isinstance(result, dict)
        for engine, voices in result.items():
            assert isinstance(engine, str)
            assert isinstance(voices, dict)


class TestGetDefaultVoice:
    """Tests for get_default_voice function."""

    def test_get_default_voice_pocket_tts(self):
        """Test default voice for pocket-tts."""
        result = get_default_voice("pocket-tts")
        assert result == DEFAULT_POCKET_TTS_VOICE

    def test_get_default_voice_kokoro(self):
        """Test default voice for kokoro."""
        result = get_default_voice("kokoro")
        assert result == DEFAULT_KOKORO_VOICE

    def test_get_default_voice_unknown_returns_pocket_tts_default(self):
        """Test unknown engine returns pocket-tts default."""
        result = get_default_voice("unknown")
        assert result == DEFAULT_POCKET_TTS_VOICE

    def test_get_default_voice_empty_string_returns_pocket_tts_default(self):
        """Test empty string returns pocket-tts default."""
        result = get_default_voice("")
        assert result == DEFAULT_POCKET_TTS_VOICE


class TestValidateVoice:
    """Tests for validate_voice function."""

    def test_validate_voice_valid_pocket_tts(self):
        """Test valid pocket-tts voice returns True."""
        assert validate_voice("pocket-tts", "azelma") is True
        assert validate_voice("pocket-tts", "alba") is True

    def test_validate_voice_valid_kokoro(self):
        """Test valid kokoro voice returns True."""
        assert validate_voice("kokoro", "am_liam") is True
        assert validate_voice("kokoro", "af_bella") is True

    def test_validate_voice_invalid_pocket_tts(self):
        """Test invalid pocket-tts voice returns False."""
        assert validate_voice("pocket-tts", "nonexistent") is False
        assert validate_voice("pocket-tts", "am_liam") is False  # kokoro voice

    def test_validate_voice_invalid_kokoro(self):
        """Test invalid kokoro voice returns False."""
        assert validate_voice("kokoro", "nonexistent") is False
        assert validate_voice("kokoro", "azelma") is False  # pocket-tts voice

    def test_validate_voice_unknown_engine_returns_false(self):
        """Test unknown engine always returns False."""
        assert validate_voice("unknown", "azelma") is False
        assert validate_voice("unknown", "am_liam") is False

    def test_validate_voice_empty_voice_returns_false(self):
        """Test empty voice string returns False."""
        assert validate_voice("pocket-tts", "") is False
        assert validate_voice("kokoro", "") is False

    def test_validate_voice_case_sensitive(self):
        """Test voice names are case-sensitive."""
        assert validate_voice("pocket-tts", "Azelma") is False
        assert validate_voice("pocket-tts", "AZELMA") is False


class TestGetPocketTtsVoicePath:
    """Tests for get_pocket_tts_voice_path function."""

    def test_get_pocket_tts_voice_path_valid_voice(self):
        """Test valid voice returns the voice name."""
        assert get_pocket_tts_voice_path("azelma") == "azelma"
        assert get_pocket_tts_voice_path("alba") == "alba"
        assert get_pocket_tts_voice_path("marius") == "marius"

    def test_get_pocket_tts_voice_path_invalid_voice_returns_default(self):
        """Test invalid voice returns default."""
        assert get_pocket_tts_voice_path("nonexistent") == DEFAULT_POCKET_TTS_VOICE

    def test_get_pocket_tts_voice_path_empty_string_returns_default(self):
        """Test empty string returns default."""
        assert get_pocket_tts_voice_path("") == DEFAULT_POCKET_TTS_VOICE

    def test_get_pocket_tts_voice_path_kokoro_voice_returns_default(self):
        """Test kokoro voice returns default (not valid for pocket-tts)."""
        assert get_pocket_tts_voice_path("am_liam") == DEFAULT_POCKET_TTS_VOICE

    def test_get_pocket_tts_voice_path_case_sensitive(self):
        """Test voice path lookup is case-sensitive."""
        assert get_pocket_tts_voice_path("Azelma") == DEFAULT_POCKET_TTS_VOICE
