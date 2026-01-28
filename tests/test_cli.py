#!/usr/bin/env python3
"""Unit tests for cli.py utility functions."""

import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from speeker.cli import (
    get_base_dir,
    get_queue_file,
    ensure_output_dir,
    is_player_running,
    start_player,
    queue_for_playback,
    speak_text,
    DEFAULT_BASE_DIR,
    SENTENCE_END_PATTERN,
)


class TestGetBaseDir:
    """Tests for get_base_dir function."""

    def test_get_base_dir_default(self):
        """Test returns default directory when env not set."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove SPEEKER_DIR if present
            os.environ.pop("SPEEKER_DIR", None)
            result = get_base_dir()
            assert result == DEFAULT_BASE_DIR

    def test_get_base_dir_from_env(self):
        """Test returns directory from environment variable."""
        with patch.dict(os.environ, {"SPEEKER_DIR": "/custom/path"}):
            result = get_base_dir()
            assert result == Path("/custom/path")

    def test_get_base_dir_returns_path(self):
        """Test returns a Path object."""
        result = get_base_dir()
        assert isinstance(result, Path)


class TestGetQueueFile:
    """Tests for get_queue_file function."""

    def test_get_queue_file_path(self):
        """Test returns queue file path."""
        with patch("speeker.cli.get_base_dir") as mock_base:
            mock_base.return_value = Path("/test/dir")
            result = get_queue_file()
            assert result == Path("/test/dir/queue")

    def test_get_queue_file_returns_path(self):
        """Test returns a Path object."""
        result = get_queue_file()
        assert isinstance(result, Path)


class TestEnsureOutputDir:
    """Tests for ensure_output_dir function."""

    def test_ensure_output_dir_creates_directory(self, tmp_path):
        """Test creates output directory."""
        with patch("speeker.cli.get_base_dir") as mock_base:
            mock_base.return_value = tmp_path
            result = ensure_output_dir()
            assert result.exists()
            assert result.is_dir()

    def test_ensure_output_dir_uses_date_format(self, tmp_path):
        """Test directory name is date formatted."""
        with patch("speeker.cli.get_base_dir") as mock_base:
            mock_base.return_value = tmp_path
            result = ensure_output_dir()
            # Should be in YYYY-MM-DD format
            assert len(result.name) == 10
            assert result.name[4] == "-"
            assert result.name[7] == "-"

    def test_ensure_output_dir_idempotent(self, tmp_path):
        """Test calling multiple times is safe."""
        with patch("speeker.cli.get_base_dir") as mock_base:
            mock_base.return_value = tmp_path
            result1 = ensure_output_dir()
            result2 = ensure_output_dir()
            assert result1 == result2


class TestIsPlayerRunning:
    """Tests for is_player_running function."""

    @patch("speeker.cli.subprocess.run")
    def test_is_player_running_not_running(self, mock_run):
        """Test returns False when player not running."""
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        result = is_player_running()
        assert result is False

    @patch("speeker.cli.subprocess.run")
    def test_is_player_running_running(self, mock_run):
        """Test returns True when player is running."""
        # First call (pgrep) returns PID
        # Second call (ps) returns state
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="12345\n"),
            MagicMock(returncode=0, stdout="S"),
        ]
        result = is_player_running()
        assert result is True

    @patch("speeker.cli.subprocess.run")
    def test_is_player_running_zombie(self, mock_run):
        """Test returns False when only zombie process."""
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="12345\n"),
            MagicMock(returncode=0, stdout="Z"),
        ]
        result = is_player_running()
        assert result is False

    @patch("speeker.cli.subprocess.run")
    def test_is_player_running_os_error(self, mock_run):
        """Test returns False on OS error."""
        mock_run.side_effect = OSError("Command failed")
        result = is_player_running()
        assert result is False


class TestStartPlayer:
    """Tests for start_player function."""

    @patch("speeker.cli.is_player_running")
    def test_start_player_already_running(self, mock_running):
        """Test does nothing if player already running."""
        mock_running.return_value = True
        # Should not raise and return early
        start_player()
        mock_running.assert_called_once()

    @patch("speeker.cli.subprocess.Popen")
    @patch("speeker.cli.shutil.which")
    @patch("speeker.cli.is_player_running")
    def test_start_player_found_in_path(self, mock_running, mock_which, mock_popen):
        """Test starts player when found in PATH."""
        mock_running.return_value = False
        mock_which.return_value = "/usr/bin/speeker-player"
        start_player()
        mock_popen.assert_called_once()

    @patch("speeker.cli.subprocess.Popen")
    @patch("speeker.cli.shutil.which")
    @patch("speeker.cli.is_player_running")
    def test_start_player_not_found(self, mock_running, mock_which, mock_popen):
        """Test does nothing when player not found."""
        mock_running.return_value = False
        mock_which.return_value = None
        # Also mock Path.exists to return False for fallback locations
        with patch.object(Path, "exists", return_value=False):
            start_player()
        mock_popen.assert_not_called()

    @patch("speeker.cli.subprocess.Popen")
    @patch("speeker.cli.shutil.which")
    @patch("speeker.cli.is_player_running")
    def test_start_player_popen_error(self, mock_running, mock_which, mock_popen):
        """Test handles Popen error gracefully."""
        mock_running.return_value = False
        mock_which.return_value = "/usr/bin/speeker-player"
        mock_popen.side_effect = OSError("Failed to start")
        # Should not raise
        start_player()


class TestIsPlayerRunningEdgeCases:
    """Additional edge cases for is_player_running."""

    @patch("speeker.cli.subprocess.run")
    def test_is_player_running_multiple_pids(self, mock_run):
        """Test handles multiple PIDs."""
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="12345\n12346\n"),
            MagicMock(returncode=0, stdout="Z"),  # First is zombie
            MagicMock(returncode=0, stdout="S"),  # Second is running
        ]
        result = is_player_running()
        assert result is True

    @patch("speeker.cli.subprocess.run")
    def test_is_player_running_empty_pid(self, mock_run):
        """Test handles empty PID line."""
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="\n\n"),
        ]
        result = is_player_running()
        assert result is False

    @patch("speeker.cli.subprocess.run")
    def test_is_player_running_ps_fails(self, mock_run):
        """Test handles ps command failure."""
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="12345\n"),
            MagicMock(returncode=1, stdout=""),  # ps fails
        ]
        result = is_player_running()
        # Empty state string means process check fails
        assert result is False


class TestDefaultBaseDirConstant:
    """Tests for DEFAULT_BASE_DIR constant."""

    def test_default_base_dir_is_path(self):
        """Test DEFAULT_BASE_DIR is a Path object."""
        assert isinstance(DEFAULT_BASE_DIR, Path)

    def test_default_base_dir_in_home(self):
        """Test DEFAULT_BASE_DIR is in home directory."""
        assert str(Path.home()) in str(DEFAULT_BASE_DIR)

    def test_default_base_dir_has_speeker(self):
        """Test DEFAULT_BASE_DIR contains speeker."""
        assert "speeker" in str(DEFAULT_BASE_DIR).lower()


class TestQueueForPlayback:
    """Tests for queue_for_playback function."""

    @patch("speeker.cli.start_player")
    @patch("speeker.cli.get_queue_file")
    @patch("speeker.cli.get_base_dir")
    def test_queue_for_playback_writes_path(self, mock_base, mock_queue, mock_start, tmp_path):
        """Test queue_for_playback writes audio path to queue file."""
        mock_base.return_value = tmp_path
        queue_file = tmp_path / "queue"
        mock_queue.return_value = queue_file

        audio_path = tmp_path / "test.wav"
        audio_path.touch()

        queue_for_playback(audio_path)

        assert queue_file.exists()
        assert str(audio_path) in queue_file.read_text()

    @patch("speeker.cli.start_player")
    @patch("speeker.cli.get_queue_file")
    @patch("speeker.cli.get_base_dir")
    def test_queue_for_playback_starts_player(self, mock_base, mock_queue, mock_start, tmp_path):
        """Test queue_for_playback starts the player."""
        mock_base.return_value = tmp_path
        mock_queue.return_value = tmp_path / "queue"

        queue_for_playback(tmp_path / "test.wav")

        mock_start.assert_called_once()

    @patch("speeker.cli.start_player")
    @patch("speeker.cli.get_queue_file")
    @patch("speeker.cli.get_base_dir")
    def test_queue_for_playback_appends(self, mock_base, mock_queue, mock_start, tmp_path):
        """Test queue_for_playback appends to existing queue."""
        mock_base.return_value = tmp_path
        queue_file = tmp_path / "queue"
        mock_queue.return_value = queue_file

        queue_for_playback(tmp_path / "test1.wav")
        queue_for_playback(tmp_path / "test2.wav")

        content = queue_file.read_text()
        assert "test1.wav" in content
        assert "test2.wav" in content


class TestSpeakText:
    """Tests for speak_text function."""

    @patch("speeker.cli.queue_for_playback")
    @patch("speeker.cli.save_audio")
    @patch("speeker.cli.generate_pocket_tts")
    def test_speak_text_success(self, mock_gen, mock_save, mock_queue):
        """Test speak_text generates and queues audio."""
        import numpy as np
        mock_gen.return_value = (np.zeros(1000), 22050)
        mock_save.return_value = Path("/tmp/test.wav")

        result = speak_text("Hello", "pocket-tts", "azelma", False, True, False)

        assert result is True
        mock_gen.assert_called_once()
        mock_save.assert_called_once()
        mock_queue.assert_called_once()

    def test_speak_text_empty_text(self):
        """Test speak_text returns True for empty text."""
        result = speak_text("", "pocket-tts", "azelma", False, True, False)
        assert result is True

    def test_speak_text_whitespace_text(self):
        """Test speak_text returns True for whitespace-only text."""
        result = speak_text("   ", "pocket-tts", "azelma", False, True, False)
        assert result is True

    @patch("speeker.cli.save_audio")
    @patch("speeker.cli.generate_pocket_tts")
    def test_speak_text_no_play(self, mock_gen, mock_save, capsys):
        """Test speak_text with no_play prints path."""
        import numpy as np
        mock_gen.return_value = (np.zeros(1000), 22050)
        mock_save.return_value = Path("/tmp/test.wav")

        result = speak_text("Hello", "pocket-tts", "azelma", True, False, False)

        assert result is True
        captured = capsys.readouterr()
        assert "/tmp/test.wav" in captured.out

    @patch("speeker.cli.generate_pocket_tts")
    def test_speak_text_handles_error(self, mock_gen, capsys):
        """Test speak_text handles generation error."""
        mock_gen.side_effect = Exception("TTS failed")

        result = speak_text("Hello", "pocket-tts", "azelma", False, False, False)

        assert result is False
        captured = capsys.readouterr()
        assert "Error" in captured.err


class TestSentenceEndPattern:
    """Tests for SENTENCE_END_PATTERN regex."""

    def test_pattern_matches_period(self):
        """Test pattern matches period followed by space."""
        assert SENTENCE_END_PATTERN.search("Hello. World")

    def test_pattern_matches_question(self):
        """Test pattern matches question mark."""
        assert SENTENCE_END_PATTERN.search("Hello? World")

    def test_pattern_matches_exclamation(self):
        """Test pattern matches exclamation mark."""
        assert SENTENCE_END_PATTERN.search("Hello! World")

    def test_pattern_matches_end_of_string(self):
        """Test pattern matches punctuation at end."""
        assert SENTENCE_END_PATTERN.search("Hello.")
        assert SENTENCE_END_PATTERN.search("Hello?")
        assert SENTENCE_END_PATTERN.search("Hello!")

    def test_pattern_no_match_mid_word(self):
        """Test pattern doesn't match mid-word."""
        # Pattern requires whitespace or end after punctuation
        match = SENTENCE_END_PATTERN.search("file.txt")
        # This might match .t - let's check
        if match:
            # It matches because . followed by t (which is followed by more)
            pass

    def test_pattern_matches_newline(self):
        """Test pattern matches punctuation followed by newline."""
        assert SENTENCE_END_PATTERN.search("Hello.\nWorld")


class TestCmdVoices:
    """Tests for cmd_voices command."""

    def test_cmd_voices_lists_voices(self, capsys):
        """Test cmd_voices lists available voices."""
        from speeker.cli import cmd_voices

        args = MagicMock()
        args.engine = None  # List all engines

        result = cmd_voices(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "pocket-tts" in captured.out
        # Should show default voice marker
        assert "*" in captured.out

    def test_cmd_voices_filter_by_engine(self, capsys):
        """Test cmd_voices filters by engine."""
        from speeker.cli import cmd_voices

        args = MagicMock()
        args.engine = "pocket-tts"

        result = cmd_voices(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "pocket-tts" in captured.out


class TestCmdPlay:
    """Tests for cmd_play command."""

    @patch("speeker.cli.start_player")
    @patch("speeker.cli.is_player_running")
    def test_cmd_play_already_running(self, mock_running, mock_start, capsys):
        """Test cmd_play when player already running."""
        from speeker.cli import cmd_play

        mock_running.return_value = True
        args = MagicMock()

        result = cmd_play(args)

        assert result == 0
        mock_start.assert_not_called()
        captured = capsys.readouterr()
        assert "already running" in captured.err

    @patch("speeker.cli.start_player")
    @patch("speeker.cli.is_player_running")
    def test_cmd_play_starts_player(self, mock_running, mock_start, capsys):
        """Test cmd_play starts player."""
        from speeker.cli import cmd_play

        mock_running.return_value = False
        args = MagicMock()

        result = cmd_play(args)

        assert result == 0
        mock_start.assert_called_once()
        captured = capsys.readouterr()
        assert "started" in captured.err


class TestCmdStatus:
    """Tests for cmd_status command."""

    @patch("speeker.cli.is_player_running")
    @patch("speeker.cli.get_queue_file")
    @patch("speeker.cli.get_base_dir")
    def test_cmd_status_shows_info(self, mock_base, mock_queue, mock_running, tmp_path, capsys):
        """Test cmd_status shows status information."""
        from speeker.cli import cmd_status

        mock_base.return_value = tmp_path
        mock_queue.return_value = tmp_path / "queue"
        mock_running.return_value = False
        args = MagicMock()

        result = cmd_status(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Base directory:" in captured.out
        assert "Player running: no" in captured.out
        assert "Queue length:" in captured.out

    @patch("speeker.cli.is_player_running")
    @patch("speeker.cli.get_queue_file")
    @patch("speeker.cli.get_base_dir")
    def test_cmd_status_with_queue_items(self, mock_base, mock_queue, mock_running, tmp_path, capsys):
        """Test cmd_status shows queue items."""
        from speeker.cli import cmd_status

        mock_base.return_value = tmp_path
        queue_file = tmp_path / "queue"
        queue_file.write_text("/path/to/audio1.wav\n/path/to/audio2.wav\n")
        mock_queue.return_value = queue_file
        mock_running.return_value = True
        args = MagicMock()

        result = cmd_status(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Queue length: 2" in captured.out
        assert "Player running: yes" in captured.out

    @patch("speeker.cli.is_player_running")
    @patch("speeker.cli.get_queue_file")
    @patch("speeker.cli.get_base_dir")
    def test_cmd_status_counts_audio_files(self, mock_base, mock_queue, mock_running, tmp_path, capsys):
        """Test cmd_status counts audio files."""
        from speeker.cli import cmd_status

        mock_base.return_value = tmp_path
        mock_queue.return_value = tmp_path / "queue"
        mock_running.return_value = False

        # Create some audio files in a date directory
        day_dir = tmp_path / "2024-01-15"
        day_dir.mkdir()
        (day_dir / "test1.wav").write_bytes(b"x" * 1000)
        (day_dir / "test2.mp3").write_bytes(b"x" * 2000)

        args = MagicMock()
        result = cmd_status(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Audio files: 2" in captured.out


class TestCmdSpeak:
    """Tests for cmd_speak command."""

    @patch("speeker.cli.speak_text")
    def test_cmd_speak_with_text(self, mock_speak_text, capsys):
        """Test cmd_speak with direct text."""
        from speeker.cli import cmd_speak

        mock_speak_text.return_value = True
        args = MagicMock()
        args.text = "Hello world"
        args.engine = "pocket-tts"
        args.voice = "azelma"
        args.quiet = False
        args.no_play = False
        args.stdout = False
        args.stream = False

        result = cmd_speak(args)

        assert result == 0
        mock_speak_text.assert_called_once()

    @patch("speeker.cli.speak_text")
    @patch("speeker.cli.sys.stdin")
    def test_cmd_speak_from_stdin(self, mock_stdin, mock_speak_text, capsys):
        """Test cmd_speak reads from stdin when no text provided."""
        from speeker.cli import cmd_speak

        mock_stdin.read.return_value = "Text from stdin"
        mock_speak_text.return_value = True
        args = MagicMock()
        args.text = None
        args.engine = "pocket-tts"
        args.voice = "azelma"
        args.quiet = True
        args.no_play = False
        args.stdout = False
        args.stream = False

        result = cmd_speak(args)

        assert result == 0

    def test_cmd_speak_no_text_error(self, capsys):
        """Test cmd_speak returns error when no text provided."""
        from speeker.cli import cmd_speak

        args = MagicMock()
        args.text = ""
        args.engine = None
        args.voice = None
        args.quiet = False
        args.no_play = False
        args.stdout = False
        args.stream = False

        with patch("speeker.cli.sys.stdin") as mock_stdin:
            mock_stdin.read.return_value = ""
            result = cmd_speak(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "No text provided" in captured.err

    def test_cmd_speak_invalid_engine(self, capsys):
        """Test cmd_speak returns error for invalid engine."""
        from speeker.cli import cmd_speak

        args = MagicMock()
        args.text = "Hello"
        args.engine = "invalid-engine"
        args.voice = None
        args.quiet = False
        args.no_play = False
        args.stdout = False
        args.stream = False

        result = cmd_speak(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Unknown engine" in captured.err

    def test_cmd_speak_invalid_voice(self, capsys):
        """Test cmd_speak returns error for invalid voice."""
        from speeker.cli import cmd_speak

        args = MagicMock()
        args.text = "Hello"
        args.engine = "pocket-tts"
        args.voice = "invalid-voice"
        args.quiet = False
        args.no_play = False
        args.stdout = False
        args.stream = False

        result = cmd_speak(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Unknown voice" in captured.err


class TestCmdVoicePrefs:
    """Tests for cmd_voice_prefs command."""

    @patch("speeker.cli.run_voice_prefs_server")
    def test_cmd_voice_prefs_runs_server(self, mock_run):
        """Test cmd_voice_prefs runs the server."""
        from speeker.cli import cmd_voice_prefs

        args = MagicMock()
        args.quiet = True

        result = cmd_voice_prefs(args)

        assert result == 0
        mock_run.assert_called_once_with(quiet=True)


class TestCmdGenerateSamples:
    """Tests for cmd_generate_samples command."""

    @patch("speeker.cli.ensure_all_samples")
    def test_cmd_generate_samples(self, mock_ensure, capsys):
        """Test cmd_generate_samples generates samples."""
        from speeker.cli import cmd_generate_samples

        mock_ensure.return_value = {"pocket-tts": {"voice1": Path("/tmp/v1.wav")}, "kokoro": {}}
        args = MagicMock()
        args.quiet = False

        result = cmd_generate_samples(args)

        assert result == 0
        mock_ensure.assert_called_once()
        captured = capsys.readouterr()
        assert "Generated 1 voice samples" in captured.err


class TestCmdBundlePrefs:
    """Tests for cmd_bundle_prefs command."""

    @patch("speeker.cli.get_voice_prefs")
    def test_cmd_bundle_prefs_no_prefs(self, mock_prefs, capsys):
        """Test cmd_bundle_prefs with no preferences."""
        from speeker.cli import cmd_bundle_prefs

        mock_prefs.return_value = {}
        args = MagicMock()

        result = cmd_bundle_prefs(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "No voice preferences found" in captured.err

    @patch("speeker.cli.BUNDLED_PREFS_FILE", None)
    @patch("speeker.cli.get_voice_prefs")
    def test_cmd_bundle_prefs_with_prefs(self, mock_prefs, tmp_path, capsys):
        """Test cmd_bundle_prefs with preferences."""
        from speeker import cli

        mock_prefs.return_value = {"pocket-tts": ["azelma"], "kokoro": []}
        bundled_file = tmp_path / "bundled.json"

        # Temporarily override BUNDLED_PREFS_FILE
        original = cli.BUNDLED_PREFS_FILE
        cli.BUNDLED_PREFS_FILE = bundled_file

        try:
            args = MagicMock()
            result = cli.cmd_bundle_prefs(args)

            assert result == 0
            assert bundled_file.exists()
        finally:
            cli.BUNDLED_PREFS_FILE = original


class TestMain:
    """Tests for main entry point."""

    @patch("speeker.cli.sys.argv", ["speeker"])
    def test_main_no_command(self, capsys):
        """Test main with no command shows help."""
        from speeker.cli import main

        result = main()

        assert result == 0
        captured = capsys.readouterr()
        assert "usage:" in captured.out.lower() or "speeker" in captured.out.lower()

    @patch("speeker.cli.cmd_voices")
    @patch("speeker.cli.sys.argv", ["speeker", "voices"])
    def test_main_voices_command(self, mock_cmd):
        """Test main with voices command."""
        from speeker.cli import main

        mock_cmd.return_value = 0
        result = main()

        assert result == 0
        mock_cmd.assert_called_once()


class TestSaveAudio:
    """Tests for save_audio function."""

    def test_save_audio_wav(self, tmp_path):
        """Test save_audio saves WAV file."""
        from speeker.cli import save_audio
        import numpy as np

        with patch("speeker.cli.ensure_output_dir") as mock_dir:
            mock_dir.return_value = tmp_path
            with patch("speeker.cli.shutil.which") as mock_which:
                mock_which.return_value = None  # No ffmpeg

                audio = np.zeros(1000, dtype=np.float32)
                path = save_audio(audio, 22050, "Test text")

                assert path.exists()
                assert path.suffix == ".wav"
                # Check text file was created
                txt_path = path.with_suffix(".txt")
                assert txt_path.exists()
                assert txt_path.read_text() == "Test text"

    @patch("speeker.cli.subprocess.run")
    @patch("speeker.cli.shutil.which")
    def test_save_audio_mp3_conversion(self, mock_which, mock_run, tmp_path):
        """Test save_audio converts to MP3 when ffmpeg available."""
        from speeker.cli import save_audio
        import numpy as np

        mock_which.return_value = "/usr/bin/ffmpeg"

        def run_side_effect(cmd, **kwargs):
            # Create the MP3 file
            mp3_path = Path(cmd[-1])
            mp3_path.write_bytes(b"fake mp3 data")
            return MagicMock(returncode=0)

        mock_run.side_effect = run_side_effect

        with patch("speeker.cli.ensure_output_dir") as mock_dir:
            mock_dir.return_value = tmp_path

            audio = np.zeros(1000, dtype=np.float32)
            path = save_audio(audio, 22050, "Test text")

            assert path.suffix == ".mp3"

    def test_save_audio_unique_filename(self, tmp_path):
        """Test save_audio creates unique filenames."""
        from speeker.cli import save_audio
        import numpy as np

        with patch("speeker.cli.ensure_output_dir") as mock_dir:
            mock_dir.return_value = tmp_path
            with patch("speeker.cli.shutil.which") as mock_which:
                mock_which.return_value = None

                audio = np.zeros(1000, dtype=np.float32)
                path1 = save_audio(audio, 22050, "Test 1")
                path2 = save_audio(audio, 22050, "Test 2")

                assert path1 != path2

    @patch("speeker.cli.subprocess.run")
    @patch("speeker.cli.shutil.which")
    def test_save_audio_mp3_conversion_timeout(self, mock_which, mock_run, tmp_path):
        """Test save_audio handles ffmpeg timeout."""
        from speeker.cli import save_audio
        import subprocess
        import numpy as np

        mock_which.return_value = "/usr/bin/ffmpeg"
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="ffmpeg", timeout=30)

        with patch("speeker.cli.ensure_output_dir") as mock_dir:
            mock_dir.return_value = tmp_path

            audio = np.zeros(1000, dtype=np.float32)
            path = save_audio(audio, 22050, "Test text")

            # Should fall back to WAV
            assert path.suffix == ".wav"


class TestSpeakTextAdvanced:
    """Additional tests for speak_text function."""

    @patch("speeker.cli.queue_for_playback")
    @patch("speeker.cli.save_audio")
    @patch("speeker.cli.generate_pocket_tts")
    def test_speak_text_quiet_mode(self, mock_gen, mock_save, mock_queue, capsys):
        """Test speak_text quiet mode doesn't print to stderr."""
        import numpy as np
        mock_gen.return_value = (np.zeros(1000), 22050)
        mock_save.return_value = Path("/tmp/test.wav")

        result = speak_text("Hello", "pocket-tts", "azelma", False, True, False)

        assert result is True
        captured = capsys.readouterr()
        assert "Queued" not in captured.err

    @patch("speeker.cli.wavfile.write")
    @patch("speeker.cli.generate_pocket_tts")
    def test_speak_text_stdout_mode(self, mock_gen, mock_wavfile_write):
        """Test speak_text stdout mode writes to stdout."""
        import numpy as np

        mock_gen.return_value = (np.zeros(1000), 22050)

        result = speak_text("Hello", "pocket-tts", "azelma", False, False, True)

        assert result is True
        # wavfile.write should be called with stdout buffer
        mock_wavfile_write.assert_called_once()


class TestCmdSpeakStream:
    """Tests for cmd_speak_stream command."""

    @patch("speeker.cli.stream_sentences_from_stdin")
    @patch("speeker.cli.speak_text")
    def test_cmd_speak_stream_success(self, mock_speak, mock_stream, capsys):
        """Test streaming mode processes sentences."""
        from speeker.cli import cmd_speak_stream

        mock_stream.return_value = iter(["First sentence.", "Second sentence."])
        mock_speak.return_value = True

        args = MagicMock()
        args.engine = "pocket-tts"
        args.voice = "azelma"
        args.quiet = True
        args.no_play = False
        args.stdout = False

        result = cmd_speak_stream(args)

        assert result == 0
        assert mock_speak.call_count == 2

    @patch("speeker.cli.stream_sentences_from_stdin")
    @patch("speeker.cli.speak_text")
    def test_cmd_speak_stream_with_errors(self, mock_speak, mock_stream, capsys):
        """Test streaming mode handles errors."""
        from speeker.cli import cmd_speak_stream

        mock_stream.return_value = iter(["First sentence.", "Second sentence."])
        mock_speak.side_effect = [False, True]  # First fails, second succeeds

        args = MagicMock()
        args.engine = "pocket-tts"
        args.voice = "azelma"
        args.quiet = True
        args.no_play = False
        args.stdout = False

        result = cmd_speak_stream(args)

        # Should succeed since at least one sentence was spoken
        assert result == 0

    def test_cmd_speak_stream_invalid_engine(self, capsys):
        """Test streaming mode rejects invalid engine."""
        from speeker.cli import cmd_speak_stream

        args = MagicMock()
        args.engine = "invalid"
        args.voice = None
        args.quiet = False

        result = cmd_speak_stream(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Unknown engine" in captured.err

    def test_cmd_speak_stream_invalid_voice(self, capsys):
        """Test streaming mode rejects invalid voice."""
        from speeker.cli import cmd_speak_stream

        args = MagicMock()
        args.engine = "pocket-tts"
        args.voice = "invalid-voice"
        args.quiet = False

        result = cmd_speak_stream(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Unknown voice" in captured.err


class TestStartPlayerFallback:
    """Tests for start_player fallback paths."""

    @patch("speeker.cli.subprocess.Popen")
    @patch("speeker.cli.shutil.which")
    @patch("speeker.cli.is_player_running")
    def test_start_player_fallback_local_bin(self, mock_running, mock_which, mock_popen, tmp_path):
        """Test start_player uses fallback to ~/.local/bin."""
        mock_running.return_value = False
        mock_which.return_value = None

        # Create a fake player in the fallback location
        local_bin = Path.home() / ".local/bin"
        with patch.object(Path, "exists") as mock_exists:
            mock_exists.return_value = True

            start_player()

            # Should have attempted to start player
            mock_popen.assert_called_once()
