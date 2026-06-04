#!/usr/bin/env python3
"""Unit tests for voice_clone.py."""

import math
import struct
import wave
from pathlib import Path

import pytest

from speeker import voice_clone


def make_test_wav(path: Path, duration_secs: float = 2.0, freq: float = 440.0) -> Path:
    """Create a synthetic 24kHz mono WAV for testing."""
    sample_rate = 24000
    n_samples = int(sample_rate * duration_secs)
    samples = []
    for i in range(n_samples):
        t = i / sample_rate
        value = int(0.5 * 32767 * math.sin(2 * math.pi * freq * t))
        samples.append(value)

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "w") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(struct.pack(f"{len(samples)}h", *samples))
    return path


@pytest.fixture
def voices_dir(tmp_path, monkeypatch):
    """Set up an isolated voices directory for testing."""
    monkeypatch.setenv("SPEEKER_DIR", str(tmp_path))
    # voices_dir() returns SPEEKER_DIR/data/voices
    vdir = tmp_path / "data" / "voices"
    vdir.mkdir(parents=True)
    return vdir


@pytest.fixture
def sample_wav(tmp_path):
    """Create a sample WAV file for testing."""
    return make_test_wav(tmp_path / "sample.wav", duration_secs=3.0)


class TestManifest:
    """Tests for manifest CRUD operations."""

    def test_empty_manifest(self, voices_dir):
        assert voice_clone.get_custom_voices() == {}

    def test_get_nonexistent_voice(self, voices_dir):
        assert voice_clone.get_custom_voice_path("nonexistent") is None

    def test_save_and_load_manifest(self, voices_dir):
        manifest = {
            "test voice": {
                "audio_path": str(voices_dir / "test.wav"),
                "description": "A test voice",
                "created_at": "2025-01-01T00:00:00+00:00",
            }
        }
        voice_clone._save_manifest(manifest)
        loaded = voice_clone._load_manifest()
        assert loaded == manifest

    def test_delete_nonexistent_voice(self, voices_dir):
        assert voice_clone.delete_custom_voice("nonexistent") is False

    def test_delete_existing_voice(self, voices_dir, sample_wav):
        # Create a voice entry with real audio file in a subdirectory
        voice_subdir = voices_dir / "test_voice"
        voice_subdir.mkdir()
        audio_path = voice_subdir / "reference.wav"
        import shutil
        shutil.copy2(sample_wav, audio_path)

        manifest = {
            "Test Voice": {
                "voice_dir": str(voice_subdir),
                "audio_path": str(audio_path),
                "description": "test",
                "created_at": "2025-01-01T00:00:00+00:00",
            }
        }
        voice_clone._save_manifest(manifest)

        assert voice_clone.delete_custom_voice("Test Voice") is True
        assert voice_clone.get_custom_voices() == {}
        assert not voice_subdir.exists()

    def test_get_custom_voice_path_falls_back_to_canonical(self, voices_dir, sample_wav):
        # Stored audio_path is stale (data dir moved), but reference.wav exists
        # at the canonical current location. Resolution must still find it.
        canonical_dir = voices_dir / voice_clone._safe_filename("Moved Voice")
        canonical_dir.mkdir()
        canonical_ref = canonical_dir / "reference.wav"
        import shutil
        shutil.copy2(sample_wav, canonical_ref)

        voice_clone._save_manifest({
            "Moved Voice": {
                "voice_dir": "/old/gone/moved_voice",
                "audio_path": "/old/gone/moved_voice/reference.wav",
                "provider": "local",
                "description": "t",
                "created_at": "",
            }
        })

        assert voice_clone.get_custom_voice_path("Moved Voice") == canonical_ref

    def test_get_custom_voice_path_returns_path(self, voices_dir, sample_wav):
        audio_path = voices_dir / "my_voice.wav"
        import shutil
        shutil.copy2(sample_wav, audio_path)

        manifest = {
            "My Voice": {
                "audio_path": str(audio_path),
                "description": "test",
                "created_at": "2025-01-01T00:00:00+00:00",
            }
        }
        voice_clone._save_manifest(manifest)

        result = voice_clone.get_custom_voice_path("My Voice")
        assert result is not None
        assert isinstance(result, Path)
        assert result == audio_path


class TestExtractAudio:
    """Tests for audio extraction (requires ffmpeg)."""

    @pytest.fixture(autouse=True)
    def _check_ffmpeg(self):
        import shutil
        if not shutil.which("ffmpeg"):
            pytest.skip("ffmpeg not installed")

    def test_extract_audio_from_wav(self, tmp_path, sample_wav):
        output = tmp_path / "extracted.wav"
        result = voice_clone.extract_audio(sample_wav, output)
        assert result.exists()
        assert result == output

        # Verify output is 24kHz mono
        with wave.open(str(output)) as wav:
            assert wav.getnchannels() == 1
            assert wav.getframerate() == 24000

    def test_extract_audio_creates_parent_dirs(self, tmp_path, sample_wav):
        output = tmp_path / "deep" / "nested" / "dir" / "extracted.wav"
        result = voice_clone.extract_audio(sample_wav, output)
        assert result.exists()


class TestTrimAudio:
    """Tests for audio trimming (requires ffmpeg)."""

    @pytest.fixture(autouse=True)
    def _check_ffmpeg(self):
        import shutil
        if not shutil.which("ffmpeg"):
            pytest.skip("ffmpeg not installed")

    def test_trim_audio(self, tmp_path, sample_wav):
        output = tmp_path / "trimmed.wav"
        result = voice_clone.trim_audio(sample_wav, output, start_secs=0, duration_secs=1)
        assert result.exists()

        # Trimmed file should be smaller than original
        assert output.stat().st_size < sample_wav.stat().st_size

    def test_trim_with_offset(self, tmp_path):
        # Create a longer WAV for offset testing
        source = make_test_wav(tmp_path / "long.wav", duration_secs=5.0)
        output = tmp_path / "trimmed.wav"
        result = voice_clone.trim_audio(source, output, start_secs=2, duration_secs=1)
        assert result.exists()


class TestCloneVoice:
    """Tests for the full clone_voice pipeline (requires ffmpeg)."""

    @pytest.fixture(autouse=True)
    def _check_ffmpeg(self):
        import shutil
        if not shutil.which("ffmpeg"):
            pytest.skip("ffmpeg not installed")

    def test_clone_from_local_file(self, voices_dir, sample_wav):
        result = voice_clone.clone_voice(
            name="Test Voice",
            sources=[str(sample_wav)],
            start_secs=0,
            duration_secs=2,
            description="A test cloned voice",
        )

        assert result.exists()
        assert result.suffix == ".wav"

        # Verify manifest was updated
        voices = voice_clone.get_custom_voices()
        assert "Test Voice" in voices
        assert voices["Test Voice"]["description"] == "A test cloned voice"

        # Verify lookup works
        path = voice_clone.get_custom_voice_path("Test Voice")
        assert path == result

    def test_clone_overwrites_existing(self, voices_dir, sample_wav):
        voice_clone.clone_voice("Dup", sources=[str(sample_wav)])
        voice_clone.clone_voice("Dup", sources=[str(sample_wav)])

        voices = voice_clone.get_custom_voices()
        assert len(voices) == 1

    def test_clone_source_not_found(self, voices_dir):
        with pytest.raises(FileNotFoundError, match="Source not found"):
            voice_clone.clone_voice("Bad", sources=["/nonexistent/file.wav"])

    def test_safe_filename(self):
        assert voice_clone._safe_filename("David Attenborough") == "david_attenborough"
        assert voice_clone._safe_filename("Morgan Freeman!") == "morgan_freeman"
        assert voice_clone._safe_filename("test/bad\\chars") == "testbadchars"
        assert voice_clone._safe_filename("") == "voice"


class TestVoicesIntegration:
    """Test that voices.py correctly integrates with custom voices."""

    def test_is_custom_voice(self, voices_dir, sample_wav):
        from speeker.voices import is_custom_voice

        assert is_custom_voice("nonexistent") is False

        # Create a custom voice
        audio_path = voices_dir / "test.wav"
        import shutil
        shutil.copy2(sample_wav, audio_path)
        voice_clone._save_manifest({
            "Test": {"audio_path": str(audio_path), "description": "t", "created_at": ""},
        })

        assert is_custom_voice("Test") is True

    def test_validate_voice_accepts_custom(self, voices_dir, sample_wav):
        from speeker.voices import validate_voice

        audio_path = voices_dir / "custom.wav"
        import shutil
        shutil.copy2(sample_wav, audio_path)
        voice_clone._save_manifest({
            "Custom": {"audio_path": str(audio_path), "description": "t", "created_at": ""},
        })

        assert validate_voice("pocket-tts", "Custom") is True

    def test_get_pocket_tts_voice_path_returns_path_for_custom(self, voices_dir, sample_wav):
        from speeker.voices import get_pocket_tts_voice_path

        audio_path = voices_dir / "my_voice.wav"
        import shutil
        shutil.copy2(sample_wav, audio_path)
        voice_clone._save_manifest({
            "My Voice": {"audio_path": str(audio_path), "description": "t", "created_at": ""},
        })

        result = get_pocket_tts_voice_path("My Voice")
        assert isinstance(result, Path)
        assert result == audio_path

    def test_get_voices_includes_custom(self, voices_dir, sample_wav):
        from speeker.voices import get_voices

        audio_path = voices_dir / "v.wav"
        import shutil
        shutil.copy2(sample_wav, audio_path)
        voice_clone._save_manifest({
            "V": {"audio_path": str(audio_path), "description": "desc", "created_at": ""},
        })

        voices = get_voices()
        assert "custom" in voices
        assert "V" in voices["custom"]
        assert voices["custom"]["V"] == "desc"


class TestProviders:
    """Provider tracking on manifest entries."""

    def test_provider_defaults_to_local_for_legacy_entry(self, voices_dir):
        # Entry written before providers existed (no "provider" key).
        voice_clone._save_manifest({
            "Old": {"audio_path": "x", "description": "d", "created_at": ""},
        })
        assert voice_clone.get_custom_voice_provider("Old") == "local"

    def test_provider_unknown_voice_returns_none(self, voices_dir):
        assert voice_clone.get_custom_voice_provider("nope") is None

    def test_get_elevenlabs_voice_id(self, voices_dir):
        voice_clone._save_manifest({
            "El": {
                "audio_path": "x", "provider": "elevenlabs",
                "voice_id": "vid_9", "description": "d", "created_at": "",
            },
            "Loc": {
                "audio_path": "x", "provider": "local",
                "description": "d", "created_at": "",
            },
        })
        assert voice_clone.get_elevenlabs_voice_id("El") == "vid_9"
        assert voice_clone.get_elevenlabs_voice_id("Loc") is None
        assert voice_clone.get_elevenlabs_voice_id("missing") is None

    def test_elevenlabs_voices_surface_under_engine(self, voices_dir):
        from speeker.voices import get_voices

        voice_clone._save_manifest({
            "El": {
                "audio_path": "x", "provider": "elevenlabs",
                "voice_id": "vid", "description": "cloud voice", "created_at": "",
            },
        })
        assert get_voices("elevenlabs")["elevenlabs"]["El"] == "cloud voice"

    def test_clone_invalid_provider_rejected(self, voices_dir, sample_wav):
        with pytest.raises(ValueError, match="Unknown provider"):
            voice_clone.clone_voice("X", sources=[str(sample_wav)], provider="bogus")

    def test_delete_elevenlabs_calls_server_delete(self, voices_dir, monkeypatch):
        from speeker import elevenlabs_api

        voice_subdir = voices_dir / "el"
        voice_subdir.mkdir()
        voice_clone._save_manifest({
            "El": {
                "voice_dir": str(voice_subdir),
                "audio_path": str(voice_subdir / "reference.wav"),
                "provider": "elevenlabs", "voice_id": "vid_42",
                "description": "d", "created_at": "",
            },
        })

        deleted = {}
        monkeypatch.setattr(
            elevenlabs_api, "delete_voice", lambda vid: deleted.setdefault("vid", vid)
        )

        assert voice_clone.delete_custom_voice("El") is True
        assert deleted["vid"] == "vid_42"
        assert voice_clone.get_custom_voices() == {}


class TestCloneElevenLabs:
    """clone_voice(provider="elevenlabs") uploads and records voice_id (ffmpeg)."""

    @pytest.fixture(autouse=True)
    def _check_ffmpeg(self):
        import shutil
        if not shutil.which("ffmpeg"):
            pytest.skip("ffmpeg not installed")

    def test_clone_elevenlabs_records_voice_id(self, voices_dir, sample_wav, monkeypatch):
        from speeker import elevenlabs_api

        calls = {}

        def fake_create(name, description, files):
            calls["name"] = name
            calls["files"] = list(files)
            return "vid_created"

        monkeypatch.setattr(elevenlabs_api, "create_ivc_voice", fake_create)

        voice_clone.clone_voice(
            name="Cloud", sources=[str(sample_wav)],
            start_secs=0, duration_secs=2, provider="elevenlabs",
        )

        entry = voice_clone.get_custom_voices()["Cloud"]
        assert entry["provider"] == "elevenlabs"
        assert entry["voice_id"] == "vid_created"
        assert voice_clone.get_elevenlabs_voice_id("Cloud") == "vid_created"
        assert calls["name"] == "Cloud"
        assert len(calls["files"]) == 1
