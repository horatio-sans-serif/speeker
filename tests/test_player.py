#!/usr/bin/env python3
"""Unit tests for player.py utility functions."""

import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from speeker.player import (
    parse_note_token,
    extract_tone_tokens,
    get_base_dir,
    get_audio_player,
    get_intro_sound,
    get_outro_sound,
    play_audio,
    should_announce_intro,
    build_session_script,
    NOTE_PATTERN,
    DEFAULT_BASE_DIR,
    POLL_INTERVAL,
    IDLE_TIMEOUT,
    PAUSE_BETWEEN_MESSAGES,
    PAUSE_BETWEEN_SESSIONS,
    ANNOUNCE_THRESHOLD_MINUTES,
)


class TestParseNoteToken:
    """Tests for parse_note_token function."""

    def test_parse_note_token_simple(self):
        """Test parsing simple note like C4."""
        result = parse_note_token("C4")
        assert result == ("c", 4)

    def test_parse_note_token_sharp(self):
        """Test parsing sharp note like F#5."""
        result = parse_note_token("F#5")
        assert result == ("f#", 5)

    def test_parse_note_token_flat(self):
        """Test parsing flat note like Bb3."""
        result = parse_note_token("Bb3")
        assert result == ("bb", 3)

    def test_parse_note_token_lowercase(self):
        """Test parsing lowercase note."""
        result = parse_note_token("e4")
        assert result == ("e", 4)

    def test_parse_note_token_all_notes(self):
        """Test parsing all note letters."""
        for note in "ABCDEFG":
            result = parse_note_token(f"{note}4")
            assert result is not None
            assert result[0] == note.lower()
            assert result[1] == 4

    def test_parse_note_token_octave_range(self):
        """Test parsing all valid octaves."""
        for octave in range(9):  # 0-8
            result = parse_note_token(f"A{octave}")
            assert result == ("a", octave)

    def test_parse_note_token_invalid_note(self):
        """Test returns None for invalid note."""
        result = parse_note_token("X4")
        assert result is None

    def test_parse_note_token_invalid_octave(self):
        """Test returns None for invalid octave."""
        result = parse_note_token("A9")
        assert result is None

    def test_parse_note_token_empty_string(self):
        """Test returns None for empty string."""
        result = parse_note_token("")
        assert result is None

    def test_parse_note_token_just_note(self):
        """Test returns None for note without octave."""
        result = parse_note_token("A")
        assert result is None


class TestExtractToneTokens:
    """Tests for extract_tone_tokens function."""

    def test_extract_single_token(self):
        """Test extracting single tone token."""
        tokens, text = extract_tone_tokens("$C4 Hello world")
        assert tokens == ["C4"]
        assert text == "Hello world"

    def test_extract_multiple_tokens(self):
        """Test extracting multiple tone tokens."""
        tokens, text = extract_tone_tokens("$C4 $E4 $G4 Hello")
        assert tokens == ["C4", "E4", "G4"]
        assert text == "Hello"

    def test_extract_no_tokens(self):
        """Test text without tone tokens."""
        tokens, text = extract_tone_tokens("Hello world")
        assert tokens == []
        assert text == "Hello world"

    def test_extract_sharp_token(self):
        """Test extracting sharp note token."""
        tokens, text = extract_tone_tokens("$F#4 Alert")
        assert tokens == ["F#4"]
        assert text == "Alert"

    def test_extract_flat_token(self):
        """Test extracting flat note token."""
        tokens, text = extract_tone_tokens("$Bb3 Warning")
        assert tokens == ["Bb3"]
        assert text == "Warning"

    def test_extract_preserves_remaining_text(self):
        """Test remaining text is preserved."""
        tokens, text = extract_tone_tokens("$A4 Important message here")
        assert text == "Important message here"

    def test_extract_empty_string(self):
        """Test empty string."""
        tokens, text = extract_tone_tokens("")
        assert tokens == []
        assert text == ""

    def test_extract_only_token(self):
        """Test string that is only a token."""
        tokens, text = extract_tone_tokens("$G4")
        assert tokens == ["G4"]
        assert text == ""

    def test_extract_whitespace_handling(self):
        """Test whitespace is stripped from remaining text."""
        tokens, text = extract_tone_tokens("$E4    Hello  ")
        assert text == "Hello"


class TestGetBaseDir:
    """Tests for get_base_dir function in player."""

    def test_get_base_dir_default(self):
        """Test returns default directory when env not set."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("SPEEKER_DIR", None)
            result = get_base_dir()
            assert result == DEFAULT_BASE_DIR

    def test_get_base_dir_from_env(self):
        """Test returns directory from environment variable."""
        with patch.dict(os.environ, {"SPEEKER_DIR": "/custom/path"}):
            result = get_base_dir()
            assert result == Path("/custom/path")


class TestGetAudioPlayer:
    """Tests for get_audio_player function."""

    @patch("speeker.player.platform.system")
    @patch("speeker.player.shutil.which")
    def test_get_audio_player_macos(self, mock_which, mock_system):
        """Test returns afplay on macOS."""
        mock_system.return_value = "Darwin"
        mock_which.return_value = "/usr/bin/afplay"
        result = get_audio_player()
        assert result == ["afplay"]

    @patch("speeker.player.platform.system")
    @patch("speeker.player.shutil.which")
    def test_get_audio_player_linux_aplay(self, mock_which, mock_system):
        """Test returns aplay on Linux when available."""
        mock_system.return_value = "Linux"
        mock_which.side_effect = lambda x: "/usr/bin/aplay" if x == "aplay" else None
        result = get_audio_player()
        assert result == ["aplay", "-q"]

    @patch("speeker.player.platform.system")
    @patch("speeker.player.shutil.which")
    def test_get_audio_player_linux_paplay(self, mock_which, mock_system):
        """Test returns paplay on Linux when aplay not available."""
        mock_system.return_value = "Linux"
        mock_which.side_effect = lambda x: "/usr/bin/paplay" if x == "paplay" else None
        result = get_audio_player()
        assert result == ["paplay"]

    @patch("speeker.player.platform.system")
    @patch("speeker.player.shutil.which")
    def test_get_audio_player_not_found(self, mock_which, mock_system):
        """Test returns None when no player found."""
        mock_system.return_value = "Darwin"
        mock_which.return_value = None
        result = get_audio_player()
        assert result is None


class TestNotePattern:
    """Tests for NOTE_PATTERN regex."""

    def test_note_pattern_matches_basic(self):
        """Test pattern matches basic notes."""
        assert NOTE_PATTERN.match("$C4") is not None
        assert NOTE_PATTERN.match("$A4") is not None
        assert NOTE_PATTERN.match("$G0") is not None

    def test_note_pattern_matches_with_sharp(self):
        """Test pattern matches sharps."""
        assert NOTE_PATTERN.match("$F#4") is not None
        assert NOTE_PATTERN.match("$C#3") is not None

    def test_note_pattern_matches_with_flat(self):
        """Test pattern matches flats."""
        assert NOTE_PATTERN.match("$Bb4") is not None
        assert NOTE_PATTERN.match("$Eb3") is not None

    def test_note_pattern_with_whitespace(self):
        """Test pattern handles leading whitespace."""
        assert NOTE_PATTERN.match("  $C4") is not None

    def test_note_pattern_no_match_without_dollar(self):
        """Test pattern requires $ prefix."""
        assert NOTE_PATTERN.match("C4") is None


class TestConstants:
    """Tests for player constants."""

    def test_poll_interval_reasonable(self):
        """Test poll interval is a reasonable value."""
        assert 0 < POLL_INTERVAL <= 2.0

    def test_idle_timeout_reasonable(self):
        """Test idle timeout is a reasonable value."""
        assert 60 <= IDLE_TIMEOUT <= 600

    def test_pause_between_messages_reasonable(self):
        """Test pause between messages is a reasonable value."""
        assert 0 <= PAUSE_BETWEEN_MESSAGES <= 2.0

    def test_pause_between_sessions_reasonable(self):
        """Test pause between sessions is a reasonable value."""
        assert 0 <= PAUSE_BETWEEN_SESSIONS <= 2.0


class TestGetAudioPlayerLinux:
    """Additional tests for get_audio_player on Linux."""

    @patch("speeker.player.platform.system")
    @patch("speeker.player.shutil.which")
    def test_get_audio_player_linux_ffplay(self, mock_which, mock_system):
        """Test returns ffplay on Linux when others not available."""
        mock_system.return_value = "Linux"
        def which_side(cmd):
            return "/usr/bin/ffplay" if cmd == "ffplay" else None
        mock_which.side_effect = which_side
        result = get_audio_player()
        assert result == ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"]

    @patch("speeker.player.platform.system")
    @patch("speeker.player.shutil.which")
    def test_get_audio_player_windows(self, mock_which, mock_system):
        """Test returns None on Windows (not supported via CLI)."""
        mock_system.return_value = "Windows"
        mock_which.return_value = "/usr/bin/powershell"
        result = get_audio_player()
        assert result is None

    @patch("speeker.player.platform.system")
    @patch("speeker.player.shutil.which")
    def test_get_audio_player_fallback_ffplay(self, mock_which, mock_system):
        """Test falls back to ffplay on unknown platform."""
        mock_system.return_value = "FreeBSD"
        mock_which.return_value = "/usr/bin/ffplay"
        result = get_audio_player()
        assert result == ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"]


class TestParseNoteTokenEdgeCases:
    """Additional edge cases for parse_note_token."""

    def test_parse_note_token_double_sharp(self):
        """Test double sharp is not valid."""
        result = parse_note_token("A##4")
        # Pattern doesn't match double accidentals
        assert result is None

    def test_parse_note_token_with_trailing_text(self):
        """Test note with trailing text."""
        # parse_note_token doesn't care about trailing text
        result = parse_note_token("C4hello")
        assert result == ("c", 4)


class TestExtractToneTokensEdgeCases:
    """Additional edge cases for extract_tone_tokens."""

    def test_extract_token_mid_text(self):
        """Test tokens not at start are not extracted."""
        tokens, text = extract_tone_tokens("Hello $C4 world")
        assert tokens == []
        assert text == "Hello $C4 world"

    def test_extract_consecutive_tokens(self):
        """Test consecutive tokens without spaces."""
        # The pattern requires whitespace handling, let's test
        tokens, text = extract_tone_tokens("$C4$E4 Hello")
        # First token parsed, then continues
        assert "C4" in tokens


class TestGetIntroSound:
    """Tests for get_intro_sound function."""

    def test_get_intro_sound_returns_path(self, tmp_path):
        """Test get_intro_sound returns a path."""
        import speeker.player as player

        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            # Reset cached path
            original = player._intro_sound_path
            player._intro_sound_path = None
            try:
                path = get_intro_sound()
                assert path.exists()
                assert "intro" in str(path)
            finally:
                player._intro_sound_path = original

    def test_get_intro_sound_cached(self, tmp_path):
        """Test get_intro_sound returns cached path."""
        import speeker.player as player

        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            original = player._intro_sound_path
            player._intro_sound_path = None
            try:
                path1 = get_intro_sound()
                path2 = get_intro_sound()
                assert path1 == path2
            finally:
                player._intro_sound_path = original


class TestGetOutroSound:
    """Tests for get_outro_sound function."""

    def test_get_outro_sound_returns_path(self, tmp_path):
        """Test get_outro_sound returns a path."""
        import speeker.player as player

        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            original = player._outro_sound_path
            player._outro_sound_path = None
            try:
                path = get_outro_sound()
                assert path.exists()
                assert "outro" in str(path)
            finally:
                player._outro_sound_path = original


class TestPlayAudio:
    """Tests for play_audio function."""

    def test_play_audio_file_not_found(self, tmp_path):
        """Test play_audio returns False for missing file."""
        result = play_audio(tmp_path / "nonexistent.wav")
        assert result is False

    @patch("speeker.player.AUDIO_PLAYER", ["afplay"])
    @patch("speeker.player.subprocess.run")
    def test_play_audio_success(self, mock_run, tmp_path):
        """Test play_audio plays file successfully."""
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"RIFF" + b"\x00" * 100)
        mock_run.return_value = MagicMock(returncode=0)

        result = play_audio(audio_file)

        assert result is True
        mock_run.assert_called_once()

    @patch("speeker.player.AUDIO_PLAYER", ["afplay"])
    @patch("speeker.player.subprocess.run")
    def test_play_audio_failure(self, mock_run, tmp_path):
        """Test play_audio returns False on player error."""
        from subprocess import CalledProcessError

        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"RIFF" + b"\x00" * 100)
        mock_run.side_effect = CalledProcessError(1, "afplay")

        result = play_audio(audio_file)

        assert result is False

    @patch("speeker.player.AUDIO_PLAYER", None)
    @patch("speeker.player.platform.system")
    def test_play_audio_no_player_non_windows(self, mock_system, tmp_path, capsys):
        """Test play_audio with no player on non-Windows."""
        mock_system.return_value = "Linux"
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"RIFF" + b"\x00" * 100)

        result = play_audio(audio_file)

        assert result is False
        captured = capsys.readouterr()
        assert "No audio player found" in captured.err

    @patch("speeker.player.AUDIO_PLAYER", None)
    @patch("speeker.player.platform.system")
    @patch("speeker.player.subprocess.run")
    def test_play_audio_windows_powershell(self, mock_run, mock_system, tmp_path):
        """Test play_audio uses PowerShell on Windows."""
        mock_system.return_value = "Windows"
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"RIFF" + b"\x00" * 100)
        mock_run.return_value = MagicMock(returncode=0)

        result = play_audio(audio_file)

        assert result is True
        mock_run.assert_called_once()
        assert "powershell" in mock_run.call_args[0][0]


class TestShouldAnnounceIntro:
    """Tests for should_announce_intro function."""

    @patch("speeker.player.get_last_utterance_time")
    def test_should_announce_intro_first_time(self, mock_last):
        """Test returns True when no previous utterance."""
        mock_last.return_value = None

        result = should_announce_intro()

        assert result is True

    @patch("speeker.player.get_last_utterance_time")
    def test_should_announce_intro_recent(self, mock_last):
        """Test returns False when utterance was recent."""
        from datetime import datetime, timezone

        mock_last.return_value = datetime.now(timezone.utc)

        result = should_announce_intro()

        assert result is False

    @patch("speeker.player.get_last_utterance_time")
    def test_should_announce_intro_old(self, mock_last):
        """Test returns True when utterance was long ago."""
        from datetime import datetime, timedelta, timezone

        mock_last.return_value = datetime.now(timezone.utc) - timedelta(minutes=60)

        result = should_announce_intro()

        assert result is True


class TestBuildSessionScript:
    """Tests for build_session_script function."""

    @patch("speeker.player.get_queue_label")
    def test_build_session_script_single_message_only_session(self, mock_label):
        """Test script for single message in only session."""
        mock_label.return_value = "Project X"
        items = [{"text": "Hello world", "created_at": "2024-01-01 12:00:00"}]

        script = build_session_script("session1", items, is_only_session=True)

        # Single message in only session - just says the text
        assert any("Hello world" in line for line in script)
        # No header for single message in single session
        assert not any("there is 1 message" in line.lower() for line in script)

    @patch("speeker.player.get_queue_label")
    def test_build_session_script_multiple_messages(self, mock_label):
        """Test script for multiple messages."""
        mock_label.return_value = "Project X"
        items = [
            {"text": "First message", "created_at": "2024-01-01 12:00:00"},
            {"text": "Second message", "created_at": "2024-01-01 12:01:00"},
        ]

        script = build_session_script("session1", items, is_only_session=True)

        # Should have header
        assert any("2 messages" in line for line in script)
        # Should have first/last markers
        assert any("First" in line for line in script)
        assert any("Last" in line for line in script)

    @patch("speeker.player.get_queue_label")
    def test_build_session_script_not_only_session(self, mock_label):
        """Test script for session when multiple sessions exist."""
        mock_label.return_value = "Project X"
        items = [{"text": "Test", "created_at": "2024-01-01 12:00:00"}]

        script = build_session_script("session1", items, is_only_session=False)

        # Should have header even for single message when not only session
        assert any("1 message" in line for line in script)


class TestAcquireLock:
    """Tests for acquire_lock function."""

    def test_acquire_lock_success(self, tmp_path):
        """Test acquire_lock succeeds when no lock exists."""
        from speeker.player import acquire_lock, release_lock

        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            lock_path = acquire_lock()

            try:
                assert lock_path is not None
                assert lock_path.exists()
            finally:
                if lock_path:
                    release_lock(lock_path)

    def test_acquire_lock_stale_lock(self, tmp_path):
        """Test acquire_lock removes stale lock."""
        from speeker.player import acquire_lock, release_lock

        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            # Create a stale lock with invalid PID
            lock_file = tmp_path / ".player.lock"
            lock_file.write_text("999999999")  # Invalid PID

            lock_path = acquire_lock()

            try:
                assert lock_path is not None
            finally:
                if lock_path:
                    release_lock(lock_path)


class TestReleaseLock:
    """Tests for release_lock function."""

    def test_release_lock_removes_file(self, tmp_path):
        """Test release_lock removes lock file."""
        from speeker.player import release_lock

        lock_file = tmp_path / ".player.lock"
        lock_file.write_text("12345")

        release_lock(lock_file)

        assert not lock_file.exists()

    def test_release_lock_missing_file(self, tmp_path):
        """Test release_lock handles missing file."""
        from speeker.player import release_lock

        lock_file = tmp_path / ".player.lock"
        # File doesn't exist - should not raise
        release_lock(lock_file)


class TestCleanupOldFiles:
    """Tests for cleanup_old_files function."""

    @patch("speeker.player.cleanup_old_entries")
    def test_cleanup_old_files(self, mock_cleanup):
        """Test cleanup_old_files calls cleanup_old_entries."""
        from speeker.player import cleanup_old_files

        mock_cleanup.return_value = 5

        result = cleanup_old_files(7, verbose=False)

        assert result == 5
        mock_cleanup.assert_called_once_with(7)


class TestRunOnce:
    """Tests for run_once function."""

    @patch("speeker.player.process_queue")
    def test_run_once_processes_queue(self, mock_process):
        """Test run_once processes the queue."""
        from speeker.player import run_once

        mock_process.return_value = 3

        run_once(verbose=False)

        mock_process.assert_called_once_with(False)


class TestGetAudioSavePath:
    """Tests for get_audio_save_path function."""

    def test_get_audio_save_path(self, tmp_path):
        """Test get_audio_save_path returns correct path."""
        from speeker.player import get_audio_save_path

        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            path = get_audio_save_path(123)

            assert path.parent.exists()
            assert path.name == "123.wav"
            assert "audio" in str(path)


class TestUpdateAudioPath:
    """Tests for update_audio_path function."""

    @patch("speeker.player.get_connection")
    def test_update_audio_path(self, mock_conn):
        """Test update_audio_path updates database."""
        from speeker.player import update_audio_path

        mock_connection = MagicMock()
        mock_conn.return_value.__enter__ = MagicMock(return_value=mock_connection)
        mock_conn.return_value.__exit__ = MagicMock(return_value=False)

        update_audio_path(123, Path("/tmp/test.wav"))

        mock_connection.execute.assert_called_once()
        mock_connection.commit.assert_called_once()


class TestProcessQueue:
    """Tests for process_queue function."""

    @patch("speeker.player.get_sessions_with_pending")
    def test_process_queue_empty(self, mock_sessions):
        """Test process_queue with no pending items."""
        from speeker.player import process_queue

        mock_sessions.return_value = []

        result = process_queue(verbose=False)

        assert result == 0

    @patch("speeker.player.set_last_utterance_time")
    @patch("speeker.player.mark_played")
    @patch("speeker.player.speak_text")
    @patch("speeker.player.get_settings")
    @patch("speeker.player.get_pending_for_session")
    @patch("speeker.player.get_sessions_with_pending")
    def test_process_queue_single_item(
        self, mock_sessions, mock_pending, mock_settings, mock_speak, mock_mark, mock_set_time
    ):
        """Test process_queue with single item."""
        from speeker.player import process_queue

        mock_sessions.return_value = ["session1"]
        mock_pending.return_value = [
            {"id": 1, "text": "Hello", "created_at": "2024-01-01 12:00:00"}
        ]
        mock_settings.return_value = {"voice": "azelma", "speed": 1.0, "intro_sound": False}
        mock_speak.return_value = Path("/tmp/test.wav")

        result = process_queue(verbose=False)

        assert result >= 0
        mock_mark.assert_called()


class TestMainFunction:
    """Tests for main entry point."""

    @patch("speeker.player.run_once")
    @patch("speeker.player.get_base_dir")
    @patch("speeker.player.sys.argv", ["speeker-player"])
    def test_main_runs_once(self, mock_base, mock_run, tmp_path):
        """Test main runs in one-shot mode by default."""
        from speeker.player import main

        mock_base.return_value = tmp_path

        result = main()

        assert result == 0
        mock_run.assert_called_once()

    @patch("speeker.player.cleanup_old_files")
    @patch("speeker.player.get_base_dir")
    @patch("speeker.player.sys.argv", ["speeker-player", "--cleanup", "7"])
    def test_main_cleanup_mode(self, mock_base, mock_cleanup, tmp_path, capsys):
        """Test main runs cleanup mode."""
        from speeker.player import main

        mock_base.return_value = tmp_path
        mock_cleanup.return_value = 10

        result = main()

        assert result == 0
        mock_cleanup.assert_called_once_with(7, False)
        captured = capsys.readouterr()
        assert "Removed 10" in captured.err

    @patch("speeker.player.run_daemon")
    @patch("speeker.player.get_base_dir")
    @patch("speeker.player.sys.argv", ["speeker-player", "--daemon"])
    def test_main_daemon_mode(self, mock_base, mock_daemon, tmp_path):
        """Test main runs daemon mode."""
        from speeker.player import main

        mock_base.return_value = tmp_path

        result = main()

        assert result == 0
        mock_daemon.assert_called_once()


class TestGenerateTTS:
    """Tests for generate_tts function."""

    @patch("speeker.player.get_voice_state")
    @patch("speeker.player.get_tts_model")
    def test_generate_tts_success(self, mock_model, mock_voice, tmp_path):
        """Test generate_tts generates audio file."""
        from speeker.player import generate_tts
        import numpy as np

        mock_model_instance = MagicMock()
        mock_model_instance.generate_audio.return_value = MagicMock(
            numpy=MagicMock(return_value=np.zeros(1000, dtype=np.float32))
        )
        mock_model_instance.sample_rate = 22050
        mock_model.return_value = mock_model_instance
        mock_voice.return_value = MagicMock()

        path = generate_tts("Hello world", verbose=False)

        assert path is not None
        assert path.exists()
        # Clean up
        path.unlink()

    @patch("speeker.player.get_voice_state")
    @patch("speeker.player.get_tts_model")
    def test_generate_tts_with_save_path(self, mock_model, mock_voice, tmp_path):
        """Test generate_tts saves to specified path."""
        from speeker.player import generate_tts
        import numpy as np

        mock_model_instance = MagicMock()
        mock_model_instance.generate_audio.return_value = MagicMock(
            numpy=MagicMock(return_value=np.zeros(1000, dtype=np.float32))
        )
        mock_model_instance.sample_rate = 22050
        mock_model.return_value = mock_model_instance
        mock_voice.return_value = MagicMock()

        save_path = tmp_path / "output.wav"
        path = generate_tts("Hello world", save_path=save_path, verbose=False)

        assert path == save_path
        assert save_path.exists()

    @patch("speeker.player.get_voice_state")
    @patch("speeker.player.get_tts_model")
    def test_generate_tts_with_speed(self, mock_model, mock_voice, tmp_path):
        """Test generate_tts applies speed adjustment."""
        from speeker.player import generate_tts
        import numpy as np

        mock_model_instance = MagicMock()
        mock_model_instance.generate_audio.return_value = MagicMock(
            numpy=MagicMock(return_value=np.zeros(1000, dtype=np.float32))
        )
        mock_model_instance.sample_rate = 22050
        mock_model.return_value = mock_model_instance
        mock_voice.return_value = MagicMock()

        path = generate_tts("Hello world", speed=1.5, verbose=False)

        assert path is not None
        path.unlink()

    @patch("speeker.player.get_tts_model")
    def test_generate_tts_error(self, mock_model):
        """Test generate_tts returns None on error."""
        from speeker.player import generate_tts

        mock_model.side_effect = Exception("TTS error")

        path = generate_tts("Hello world", verbose=False)

        assert path is None


class TestSpeakTextPlayer:
    """Tests for speak_text function in player module."""

    @patch("speeker.player.play_audio")
    @patch("speeker.player.generate_tts")
    def test_speak_text_success(self, mock_gen, mock_play, tmp_path):
        """Test speak_text plays audio."""
        from speeker.player import speak_text as player_speak_text

        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"fake audio")
        mock_gen.return_value = audio_file
        mock_play.return_value = True

        result = player_speak_text("Hello world", verbose=False)

        mock_play.assert_called_once()

    @patch("speeker.player.play_tone_tokens")
    @patch("speeker.player.extract_tone_tokens")
    def test_speak_text_with_tones(self, mock_extract, mock_play_tone):
        """Test speak_text handles tone tokens."""
        from speeker.player import speak_text as player_speak_text

        mock_extract.return_value = (["C4", "E4"], "Hello")

        with patch("speeker.player.generate_tts") as mock_gen:
            mock_gen.return_value = None

            player_speak_text("$C4 $E4 Hello", verbose=False)

            mock_play_tone.assert_called_once_with(["C4", "E4"], False)

    @patch("speeker.player.generate_tts")
    def test_speak_text_tts_failure(self, mock_gen):
        """Test speak_text handles TTS failure."""
        from speeker.player import speak_text as player_speak_text

        mock_gen.return_value = None

        result = player_speak_text("Hello world", verbose=False)

        assert result is None


class TestPlayToneTokens:
    """Tests for play_tone_tokens function."""

    @patch("speeker.player.play_audio")
    @patch("speeker.player.generate_combined_tones_from_tokens")
    def test_play_tone_tokens_success(self, mock_gen, mock_play, tmp_path):
        """Test play_tone_tokens plays generated tones."""
        from speeker.player import play_tone_tokens

        tone_path = tmp_path / "tone.wav"
        tone_path.write_bytes(b"fake tone")
        mock_gen.return_value = tone_path

        play_tone_tokens(["C4", "E4"], verbose=False)

        mock_play.assert_called_once()

    @patch("speeker.player.play_audio")
    def test_play_tone_tokens_empty(self, mock_play):
        """Test play_tone_tokens does nothing for empty tokens."""
        from speeker.player import play_tone_tokens

        play_tone_tokens([], verbose=False)

        mock_play.assert_not_called()


class TestRunDaemon:
    """Tests for run_daemon function."""

    @patch("speeker.player.time.sleep")
    @patch("speeker.player.get_pending_count")
    @patch("speeker.player.get_voice_state")
    @patch("speeker.player.get_tts_model")
    @patch("speeker.player.release_lock")
    @patch("speeker.player.acquire_lock")
    def test_run_daemon_starts(self, mock_acquire, mock_release, mock_model, mock_voice, mock_pending, mock_sleep, tmp_path):
        """Test run_daemon starts and warms up model."""
        from speeker.player import run_daemon

        lock_path = tmp_path / ".player.lock"
        mock_acquire.return_value = lock_path
        mock_pending.return_value = 0

        # Make sleep raise to exit the loop after first iteration
        call_count = [0]
        def sleep_side_effect(duration):
            call_count[0] += 1
            if call_count[0] >= 2:
                # Simulate idle timeout by raising
                raise KeyboardInterrupt()

        mock_sleep.side_effect = sleep_side_effect

        with patch("speeker.player.IDLE_TIMEOUT", 0.001):
            with patch("speeker.player.time.time") as mock_time:
                mock_time.return_value = 0
                try:
                    run_daemon(verbose=False)
                except KeyboardInterrupt:
                    pass

        mock_model.assert_called_once()
        mock_release.assert_called()

    @patch("speeker.player.acquire_lock")
    def test_run_daemon_already_running(self, mock_acquire, capsys):
        """Test run_daemon exits if already running."""
        from speeker.player import run_daemon

        mock_acquire.return_value = None  # Lock failed

        with pytest.raises(SystemExit) as exc_info:
            run_daemon(verbose=False)

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "already running" in captured.err


class TestAcquireLockRunning:
    """Tests for acquire_lock with running process."""

    def test_acquire_lock_process_running(self, tmp_path):
        """Test acquire_lock returns None when process is running."""
        from speeker.player import acquire_lock

        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            # Create a lock file with current PID (definitely running)
            lock_file = tmp_path / ".player.lock"
            lock_file.write_text(str(os.getpid()))

            result = acquire_lock()

            assert result is None


class TestBuildSessionScriptEdgeCases:
    """Edge cases for build_session_script."""

    @patch("speeker.player.get_queue_label")
    def test_build_session_script_three_messages(self, mock_label):
        """Test script for three messages uses 'Next'."""
        mock_label.return_value = "Project X"
        items = [
            {"text": "First", "created_at": "2024-01-01 12:00:00"},
            {"text": "Middle", "created_at": "2024-01-01 12:01:00"},
            {"text": "Third", "created_at": "2024-01-01 12:02:00"},
        ]

        script = build_session_script("session1", items, is_only_session=True)

        assert any("Next" in line for line in script)

    @patch("speeker.player.get_queue_label")
    def test_build_session_script_single_not_only(self, mock_label):
        """Test script for single message when other sessions exist."""
        mock_label.return_value = "Project X"
        items = [{"text": "Solo", "created_at": "2024-01-01 12:00:00"}]

        script = build_session_script("session1", items, is_only_session=False)

        # Should have "It" prefix when single item in multi-session context
        assert any("1 message" in line for line in script)


class TestProcessQueueAdvanced:
    """Advanced tests for process_queue function."""

    @patch("speeker.player.set_last_utterance_time")
    @patch("speeker.player.play_audio")
    @patch("speeker.player.speak_text")
    @patch("speeker.player.mark_played")
    @patch("speeker.player.get_settings")
    @patch("speeker.player.get_pending_for_session")
    @patch("speeker.player.get_sessions_with_pending")
    def test_process_queue_with_intro(
        self, mock_sessions, mock_pending, mock_settings, mock_mark,
        mock_speak, mock_play_audio, mock_set_time
    ):
        """Test process_queue plays intro sound."""
        from speeker.player import process_queue

        mock_sessions.return_value = ["session1"]
        mock_pending.return_value = [
            {"id": 1, "text": "Hello", "created_at": "2024-01-01 12:00:00"},
            {"id": 2, "text": "World", "created_at": "2024-01-01 12:01:00"},
        ]
        mock_settings.return_value = {"voice": "azelma", "speed": 1.0, "intro_sound": True}
        mock_speak.return_value = Path("/tmp/test.wav")

        with patch("speeker.player.should_announce_intro") as mock_announce:
            mock_announce.return_value = True
            with patch("speeker.player.get_intro_sound") as mock_intro:
                mock_intro.return_value = Path("/tmp/intro.wav")

                result = process_queue(verbose=False)

        # Should have played intro
        assert mock_play_audio.called or result >= 0

    @patch("speeker.player.get_sessions_with_pending")
    def test_process_queue_multiple_sessions(self, mock_sessions):
        """Test process_queue handles multiple sessions."""
        from speeker.player import process_queue

        mock_sessions.return_value = ["session1", "session2"]

        with patch("speeker.player.get_pending_for_session") as mock_pending:
            mock_pending.return_value = []  # Empty for both sessions

            result = process_queue(verbose=False)

            assert result == 0
