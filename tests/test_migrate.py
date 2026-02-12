#!/usr/bin/env python3
"""Unit tests for migrate.py - auto-migration from legacy paths."""

import os
from pathlib import Path
from unittest.mock import patch

from speeker.migrate import migrate, _needs_migration, _move, _marker_path


class TestNeedsMigration:
    """Tests for _needs_migration function."""

    def test_no_migration_when_speeker_dir_set(self, tmp_path):
        """Test migration is skipped when SPEEKER_DIR is set."""
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            assert _needs_migration() is False

    def test_no_migration_when_marker_exists(self, tmp_path):
        """Test migration is skipped when marker file exists."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("speeker.migrate._marker_path") as mock_marker:
                mock_marker.return_value = tmp_path / ".migrated_v2"
                (tmp_path / ".migrated_v2").write_text("migrated")
                with patch("speeker.migrate._LEGACY_BASE", tmp_path / "nonexistent"):
                    with patch("speeker.migrate._LEGACY_CONFIG", tmp_path / "nonexistent2"):
                        assert _needs_migration() is False

    def test_needs_migration_when_legacy_base_exists(self, tmp_path):
        """Test migration needed when legacy ~/.speeker exists."""
        legacy_base = tmp_path / ".speeker"
        legacy_base.mkdir()

        with patch.dict(os.environ, {}, clear=True):
            with patch("speeker.migrate._marker_path") as mock_marker:
                mock_marker.return_value = tmp_path / ".migrated_v2"
                with patch("speeker.migrate._LEGACY_BASE", legacy_base):
                    with patch("speeker.migrate._LEGACY_CONFIG", tmp_path / "nonexistent"):
                        assert _needs_migration() is True

    def test_needs_migration_when_legacy_config_exists(self, tmp_path):
        """Test migration needed when legacy ~/.config/speeker exists."""
        legacy_config = tmp_path / ".config" / "speeker"
        legacy_config.mkdir(parents=True)

        with patch.dict(os.environ, {}, clear=True):
            with patch("speeker.migrate._marker_path") as mock_marker:
                mock_marker.return_value = tmp_path / ".migrated_v2"
                with patch("speeker.migrate._LEGACY_BASE", tmp_path / "nonexistent"):
                    with patch("speeker.migrate._LEGACY_CONFIG", legacy_config):
                        assert _needs_migration() is True


class TestMove:
    """Tests for _move helper function."""

    def test_move_file(self, tmp_path):
        """Test moving a file."""
        src = tmp_path / "src.txt"
        src.write_text("hello")
        dst = tmp_path / "dst.txt"

        _move(src, dst)

        assert not src.exists()
        assert dst.read_text() == "hello"

    def test_move_nonexistent_source(self, tmp_path):
        """Test moving nonexistent source is a no-op."""
        src = tmp_path / "nonexistent"
        dst = tmp_path / "dst.txt"

        _move(src, dst)

        assert not dst.exists()

    def test_move_does_not_overwrite(self, tmp_path):
        """Test move does not overwrite existing destination."""
        src = tmp_path / "src.txt"
        src.write_text("new")
        dst = tmp_path / "dst.txt"
        dst.write_text("existing")

        _move(src, dst)

        assert dst.read_text() == "existing"
        assert src.exists()

    def test_move_creates_parent_dirs(self, tmp_path):
        """Test move creates parent directories for destination."""
        src = tmp_path / "src.txt"
        src.write_text("data")
        dst = tmp_path / "a" / "b" / "c" / "dst.txt"

        _move(src, dst)

        assert dst.exists()
        assert dst.read_text() == "data"

    def test_move_directory(self, tmp_path):
        """Test moving a directory."""
        src = tmp_path / "src_dir"
        src.mkdir()
        (src / "file.txt").write_text("content")
        dst = tmp_path / "dst_dir"

        _move(src, dst)

        assert not src.exists()
        assert dst.is_dir()
        assert (dst / "file.txt").read_text() == "content"


class TestMigrate:
    """Tests for full migrate function."""

    def test_migrate_skipped_when_not_needed(self, tmp_path):
        """Test migrate does nothing when SPEEKER_DIR is set."""
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            migrate()
            # No marker file created when SPEEKER_DIR is set
            assert not (tmp_path / "data" / ".migrated_v2").exists()

    def test_migrate_moves_queue_db(self, tmp_path):
        """Test migrate moves queue.db from legacy location."""
        legacy_base = tmp_path / "legacy"
        legacy_base.mkdir()
        (legacy_base / "queue.db").write_text("db")

        legacy_config = tmp_path / "legacy_config"

        with patch.dict(os.environ, {}, clear=True):
            with patch("speeker.migrate._LEGACY_BASE", legacy_base):
                with patch("speeker.migrate._LEGACY_CONFIG", legacy_config):
                    with patch("speeker.migrate._marker_path") as mock_marker:
                        marker = tmp_path / "new_data" / ".migrated_v2"
                        mock_marker.return_value = marker
                        with patch("speeker.migrate.paths.data_dir", return_value=tmp_path / "new_data"):
                            with patch("speeker.migrate.paths.audio_dir", return_value=tmp_path / "new_data" / "audio"):
                                with patch("speeker.migrate.paths.voices_dir", return_value=tmp_path / "new_data" / "voices"):
                                    with patch("speeker.migrate.paths.tones_dir", return_value=tmp_path / "new_tones"):
                                        with patch("speeker.migrate.paths.tone_intro_path", return_value=tmp_path / "new_cache" / "tone_intro.wav"):
                                            with patch("speeker.migrate.paths.tone_outro_path", return_value=tmp_path / "new_cache" / "tone_outro.wav"):
                                                with patch("speeker.migrate.paths.voice_samples_dir", return_value=tmp_path / "new_cache" / "voice-samples"):
                                                    with patch("speeker.migrate.paths.config_dir", return_value=tmp_path / "new_config"):
                                                        migrate()

                        assert (tmp_path / "new_data" / "queue.db").read_text() == "db"
                        assert marker.exists()

    def test_migrate_moves_config_files(self, tmp_path):
        """Test migrate moves config.json and voice-prefs.json."""
        legacy_base = tmp_path / "legacy"
        legacy_config = tmp_path / "legacy_config"
        legacy_config.mkdir(parents=True)
        (legacy_config / "config.json").write_text('{"key": "val"}')
        (legacy_config / "voice-prefs.json").write_text('{"prefs": true}')

        new_config = tmp_path / "new_config"

        with patch.dict(os.environ, {}, clear=True):
            with patch("speeker.migrate._LEGACY_BASE", legacy_base):
                with patch("speeker.migrate._LEGACY_CONFIG", legacy_config):
                    with patch("speeker.migrate._marker_path") as mock_marker:
                        marker = tmp_path / "new_data" / ".migrated_v2"
                        mock_marker.return_value = marker
                        with patch("speeker.migrate.paths.data_dir", return_value=tmp_path / "new_data"):
                            with patch("speeker.migrate.paths.audio_dir", return_value=tmp_path / "new_data" / "audio"):
                                with patch("speeker.migrate.paths.voices_dir", return_value=tmp_path / "new_data" / "voices"):
                                    with patch("speeker.migrate.paths.tones_dir", return_value=tmp_path / "new_tones"):
                                        with patch("speeker.migrate.paths.tone_intro_path", return_value=tmp_path / "new_cache" / "tone_intro.wav"):
                                            with patch("speeker.migrate.paths.tone_outro_path", return_value=tmp_path / "new_cache" / "tone_outro.wav"):
                                                with patch("speeker.migrate.paths.voice_samples_dir", return_value=tmp_path / "new_cache" / "voice-samples"):
                                                    with patch("speeker.migrate.paths.config_dir", return_value=new_config):
                                                        migrate()

                        assert (new_config / "config.json").read_text() == '{"key": "val"}'
                        assert (new_config / "voice-prefs.json").read_text() == '{"prefs": true}'

    def test_migrate_moves_date_dirs(self, tmp_path):
        """Test migrate moves audio date directories."""
        legacy_base = tmp_path / "legacy"
        legacy_base.mkdir()
        date_dir = legacy_base / "2024-01-15"
        date_dir.mkdir()
        (date_dir / "audio.wav").write_bytes(b"wav")

        legacy_config = tmp_path / "legacy_config"
        new_audio = tmp_path / "new_data" / "audio"

        with patch.dict(os.environ, {}, clear=True):
            with patch("speeker.migrate._LEGACY_BASE", legacy_base):
                with patch("speeker.migrate._LEGACY_CONFIG", legacy_config):
                    with patch("speeker.migrate._marker_path") as mock_marker:
                        marker = tmp_path / "new_data" / ".migrated_v2"
                        mock_marker.return_value = marker
                        with patch("speeker.migrate.paths.data_dir", return_value=tmp_path / "new_data"):
                            with patch("speeker.migrate.paths.audio_dir", return_value=new_audio):
                                with patch("speeker.migrate.paths.voices_dir", return_value=tmp_path / "new_data" / "voices"):
                                    with patch("speeker.migrate.paths.tones_dir", return_value=tmp_path / "new_tones"):
                                        with patch("speeker.migrate.paths.tone_intro_path", return_value=tmp_path / "new_cache" / "tone_intro.wav"):
                                            with patch("speeker.migrate.paths.tone_outro_path", return_value=tmp_path / "new_cache" / "tone_outro.wav"):
                                                with patch("speeker.migrate.paths.voice_samples_dir", return_value=tmp_path / "new_cache" / "voice-samples"):
                                                    with patch("speeker.migrate.paths.config_dir", return_value=tmp_path / "new_config"):
                                                        migrate()

                        assert (new_audio / "2024-01-15" / "audio.wav").exists()

    def test_migrate_removes_stale_lock(self, tmp_path):
        """Test migrate removes legacy .player.lock file."""
        legacy_base = tmp_path / "legacy"
        legacy_base.mkdir()
        lock = legacy_base / ".player.lock"
        lock.write_text("12345")

        legacy_config = tmp_path / "legacy_config"

        with patch.dict(os.environ, {}, clear=True):
            with patch("speeker.migrate._LEGACY_BASE", legacy_base):
                with patch("speeker.migrate._LEGACY_CONFIG", legacy_config):
                    with patch("speeker.migrate._marker_path") as mock_marker:
                        marker = tmp_path / "new_data" / ".migrated_v2"
                        mock_marker.return_value = marker
                        with patch("speeker.migrate.paths.data_dir", return_value=tmp_path / "new_data"):
                            with patch("speeker.migrate.paths.audio_dir", return_value=tmp_path / "new_data" / "audio"):
                                with patch("speeker.migrate.paths.voices_dir", return_value=tmp_path / "new_data" / "voices"):
                                    with patch("speeker.migrate.paths.tones_dir", return_value=tmp_path / "new_tones"):
                                        with patch("speeker.migrate.paths.tone_intro_path", return_value=tmp_path / "new_cache" / "tone_intro.wav"):
                                            with patch("speeker.migrate.paths.tone_outro_path", return_value=tmp_path / "new_cache" / "tone_outro.wav"):
                                                with patch("speeker.migrate.paths.voice_samples_dir", return_value=tmp_path / "new_cache" / "voice-samples"):
                                                    with patch("speeker.migrate.paths.config_dir", return_value=tmp_path / "new_config"):
                                                        migrate()

                        assert not lock.exists()

    def test_migrate_writes_marker(self, tmp_path):
        """Test migrate writes .migrated_v2 marker on success."""
        legacy_base = tmp_path / "legacy"
        legacy_base.mkdir()

        legacy_config = tmp_path / "legacy_config"

        with patch.dict(os.environ, {}, clear=True):
            with patch("speeker.migrate._LEGACY_BASE", legacy_base):
                with patch("speeker.migrate._LEGACY_CONFIG", legacy_config):
                    with patch("speeker.migrate._marker_path") as mock_marker:
                        marker = tmp_path / "new_data" / ".migrated_v2"
                        mock_marker.return_value = marker
                        with patch("speeker.migrate.paths.data_dir", return_value=tmp_path / "new_data"):
                            with patch("speeker.migrate.paths.audio_dir", return_value=tmp_path / "new_data" / "audio"):
                                with patch("speeker.migrate.paths.voices_dir", return_value=tmp_path / "new_data" / "voices"):
                                    with patch("speeker.migrate.paths.tones_dir", return_value=tmp_path / "new_tones"):
                                        with patch("speeker.migrate.paths.tone_intro_path", return_value=tmp_path / "new_cache" / "tone_intro.wav"):
                                            with patch("speeker.migrate.paths.tone_outro_path", return_value=tmp_path / "new_cache" / "tone_outro.wav"):
                                                with patch("speeker.migrate.paths.voice_samples_dir", return_value=tmp_path / "new_cache" / "voice-samples"):
                                                    with patch("speeker.migrate.paths.config_dir", return_value=tmp_path / "new_config"):
                                                        migrate()

                        assert marker.exists()
                        assert marker.read_text() == "migrated"
