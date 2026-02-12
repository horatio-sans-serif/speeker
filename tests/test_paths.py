#!/usr/bin/env python3
"""Unit tests for paths.py - centralized path management."""

import os
from pathlib import Path
from unittest.mock import patch

from speeker import paths


class TestOverrideRoot:
    """Tests for _override_root (SPEEKER_DIR env var)."""

    def test_override_root_not_set(self):
        """Test returns None when SPEEKER_DIR is not set."""
        with patch.dict(os.environ, {}, clear=True):
            assert paths._override_root() is None

    def test_override_root_set(self, tmp_path):
        """Test returns Path when SPEEKER_DIR is set."""
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            result = paths._override_root()
            assert result == tmp_path
            assert isinstance(result, Path)

    def test_override_root_empty_string(self):
        """Test returns None when SPEEKER_DIR is empty."""
        with patch.dict(os.environ, {"SPEEKER_DIR": ""}):
            assert paths._override_root() is None


class TestConfigPaths:
    """Tests for config path functions."""

    def test_config_dir_with_override(self, tmp_path):
        """Test config_dir uses SPEEKER_DIR/config."""
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            assert paths.config_dir() == tmp_path / "config"

    def test_config_dir_without_override(self):
        """Test config_dir returns platformdirs path."""
        with patch.dict(os.environ, {}, clear=True):
            result = paths.config_dir()
            assert isinstance(result, Path)
            assert "speeker" in str(result)

    def test_config_file(self, tmp_path):
        """Test config_file is config_dir/config.json."""
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            assert paths.config_file() == tmp_path / "config" / "config.json"

    def test_voice_prefs_file(self, tmp_path):
        """Test voice_prefs_file is config_dir/voice-prefs.json."""
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            assert paths.voice_prefs_file() == tmp_path / "config" / "voice-prefs.json"


class TestDataPaths:
    """Tests for data path functions."""

    def test_data_dir_with_override(self, tmp_path):
        """Test data_dir uses SPEEKER_DIR/data."""
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            assert paths.data_dir() == tmp_path / "data"

    def test_data_dir_without_override(self):
        """Test data_dir returns platformdirs path."""
        with patch.dict(os.environ, {}, clear=True):
            result = paths.data_dir()
            assert isinstance(result, Path)
            assert "speeker" in str(result)

    def test_db_path(self, tmp_path):
        """Test db_path is data_dir/queue.db."""
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            assert paths.db_path() == tmp_path / "data" / "queue.db"

    def test_audio_dir(self, tmp_path):
        """Test audio_dir is data_dir/audio."""
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            assert paths.audio_dir() == tmp_path / "data" / "audio"

    def test_voices_dir(self, tmp_path):
        """Test voices_dir is data_dir/voices."""
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            assert paths.voices_dir() == tmp_path / "data" / "voices"

    def test_voices_manifest(self, tmp_path):
        """Test voices_manifest is voices_dir/manifest.json."""
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            assert paths.voices_manifest() == tmp_path / "data" / "voices" / "manifest.json"


class TestCachePaths:
    """Tests for cache path functions."""

    def test_cache_dir_with_override(self, tmp_path):
        """Test cache_dir uses SPEEKER_DIR/cache."""
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            assert paths.cache_dir() == tmp_path / "cache"

    def test_cache_dir_without_override(self):
        """Test cache_dir returns platformdirs path."""
        with patch.dict(os.environ, {}, clear=True):
            result = paths.cache_dir()
            assert isinstance(result, Path)
            assert "speeker" in str(result).lower()

    def test_tones_dir(self, tmp_path):
        """Test tones_dir is cache_dir/tones."""
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            assert paths.tones_dir() == tmp_path / "cache" / "tones"

    def test_voice_samples_dir(self, tmp_path):
        """Test voice_samples_dir is cache_dir/voice-samples."""
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            assert paths.voice_samples_dir() == tmp_path / "cache" / "voice-samples"

    def test_tone_intro_path(self, tmp_path):
        """Test tone_intro_path is cache_dir/tone_intro.wav."""
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            assert paths.tone_intro_path() == tmp_path / "cache" / "tone_intro.wav"

    def test_tone_outro_path(self, tmp_path):
        """Test tone_outro_path is cache_dir/tone_outro.wav."""
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            assert paths.tone_outro_path() == tmp_path / "cache" / "tone_outro.wav"


class TestRuntimePaths:
    """Tests for runtime path functions."""

    def test_runtime_dir_with_override(self, tmp_path):
        """Test runtime_dir uses SPEEKER_DIR directly (flat)."""
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            assert paths.runtime_dir() == tmp_path

    def test_runtime_dir_without_override(self):
        """Test runtime_dir returns platformdirs path."""
        with patch.dict(os.environ, {}, clear=True):
            result = paths.runtime_dir()
            assert isinstance(result, Path)

    def test_player_lock_path(self, tmp_path):
        """Test player_lock_path is runtime_dir/player.lock."""
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            assert paths.player_lock_path() == tmp_path / "player.lock"


class TestEnsureDir:
    """Tests for ensure_dir helper."""

    def test_ensure_dir_creates(self, tmp_path):
        """Test ensure_dir creates directory."""
        target = tmp_path / "a" / "b" / "c"
        result = paths.ensure_dir(target)
        assert target.exists()
        assert target.is_dir()
        assert result == target

    def test_ensure_dir_idempotent(self, tmp_path):
        """Test ensure_dir is safe to call on existing dir."""
        target = tmp_path / "existing"
        target.mkdir()
        result = paths.ensure_dir(target)
        assert result == target

    def test_ensure_dir_returns_path(self, tmp_path):
        """Test ensure_dir returns the path."""
        target = tmp_path / "new"
        result = paths.ensure_dir(target)
        assert isinstance(result, Path)


class TestPathConsistency:
    """Tests for path hierarchy consistency."""

    def test_config_file_under_config_dir(self, tmp_path):
        """Test config_file is under config_dir."""
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            assert paths.config_file().parent == paths.config_dir()

    def test_voice_prefs_under_config_dir(self, tmp_path):
        """Test voice_prefs_file is under config_dir."""
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            assert paths.voice_prefs_file().parent == paths.config_dir()

    def test_db_path_under_data_dir(self, tmp_path):
        """Test db_path is under data_dir."""
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            assert paths.db_path().parent == paths.data_dir()

    def test_audio_dir_under_data_dir(self, tmp_path):
        """Test audio_dir is under data_dir."""
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            assert paths.audio_dir().parent == paths.data_dir()

    def test_tones_dir_under_cache_dir(self, tmp_path):
        """Test tones_dir is under cache_dir."""
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            assert paths.tones_dir().parent == paths.cache_dir()

    def test_all_paths_return_path_type(self, tmp_path):
        """Test all path functions return Path objects."""
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            funcs = [
                paths.config_dir, paths.config_file, paths.voice_prefs_file,
                paths.data_dir, paths.db_path, paths.audio_dir,
                paths.voices_dir, paths.voices_manifest,
                paths.cache_dir, paths.tones_dir, paths.voice_samples_dir,
                paths.tone_intro_path, paths.tone_outro_path,
                paths.runtime_dir, paths.player_lock_path,
            ]
            for func in funcs:
                result = func()
                assert isinstance(result, Path), f"{func.__name__} returned {type(result)}"
