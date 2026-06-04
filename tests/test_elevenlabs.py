#!/usr/bin/env python3
"""Unit tests for the ElevenLabs engine and config (no network calls)."""

import numpy as np
import pytest

from speeker import voice_clone
from speeker.engines import ElevenLabsEngine


@pytest.fixture
def voices_dir(tmp_path, monkeypatch):
    """Isolated SPEEKER_DIR with an empty voices dir."""
    monkeypatch.setenv("SPEEKER_DIR", str(tmp_path))
    vdir = tmp_path / "data" / "voices"
    vdir.mkdir(parents=True)
    return vdir


def _pcm_bytes(values: list[int]) -> bytes:
    return np.array(values, dtype=np.int16).tobytes()


class TestElevenLabsConfig:
    def test_env_overrides_api_key(self, voices_dir, monkeypatch):
        from speeker.config import get_elevenlabs_config

        monkeypatch.setenv("ELEVENLABS_API_KEY", "sk-from-env")
        cfg = get_elevenlabs_config()
        assert cfg["api_key"] == "sk-from-env"
        assert cfg["model"] == "eleven_multilingual_v2"
        assert cfg["output_format"] == "pcm_24000"


class TestElevenLabsEngine:
    def test_validate_voice(self):
        eng = ElevenLabsEngine()
        assert eng.validate_voice("some-voice-id") is True
        assert eng.validate_voice("") is False

    def test_generate_parses_pcm(self, voices_dir, monkeypatch):
        from speeker import elevenlabs_api

        samples = [0, 16384, -16384, 32767, -32768]
        monkeypatch.setattr(
            elevenlabs_api, "synthesize", lambda *a, **k: _pcm_bytes(samples)
        )

        eng = ElevenLabsEngine()
        audio, sr = eng.generate("hello", "raw_voice_id")

        assert sr == 24000
        assert audio.dtype == np.float32
        expected = np.array(samples, dtype=np.float32) / 32768.0
        assert np.allclose(audio, expected)

    def test_generate_resolves_cloned_name_to_voice_id(self, voices_dir, monkeypatch):
        from speeker import elevenlabs_api

        voice_clone._save_manifest({
            "Narrator": {
                "audio_path": str(voices_dir / "ref.wav"),
                "provider": "elevenlabs",
                "voice_id": "el_abc123",
                "description": "d",
                "created_at": "",
            }
        })

        captured = {}

        def fake_synth(voice_id, text, model_id, output_format):
            captured["voice_id"] = voice_id
            captured["model_id"] = model_id
            captured["output_format"] = output_format
            return _pcm_bytes([0, 1, 2])

        monkeypatch.setattr(elevenlabs_api, "synthesize", fake_synth)

        eng = ElevenLabsEngine()
        eng.generate("hi", "Narrator")

        assert captured["voice_id"] == "el_abc123"
        assert captured["model_id"] == "eleven_multilingual_v2"
        assert captured["output_format"] == "pcm_24000"

    def test_generate_passes_through_raw_voice_id(self, voices_dir, monkeypatch):
        from speeker import elevenlabs_api

        captured = {}

        def fake_synth(voice_id, *a, **k):
            captured["voice_id"] = voice_id
            return _pcm_bytes([0])

        monkeypatch.setattr(elevenlabs_api, "synthesize", fake_synth)

        ElevenLabsEngine().generate("hi", "unknown_name_or_raw_id")
        assert captured["voice_id"] == "unknown_name_or_raw_id"

    def test_generate_rejects_non_pcm_format(self, voices_dir, monkeypatch):
        from speeker import config

        monkeypatch.setattr(
            config,
            "get_elevenlabs_config",
            lambda: {"model": "m", "output_format": "mp3_44100_128", "voice": None},
        )
        with pytest.raises(ValueError, match="pcm_"):
            ElevenLabsEngine().generate("hi", "vid")
