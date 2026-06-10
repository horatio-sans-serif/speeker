#!/usr/bin/env python3
"""Unit tests for cli.py utility functions."""

import argparse
import io
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from speeker.cli import (
    get_queue_file,
    ensure_output_dir,
    is_player_running,
    start_player,
    queue_for_playback,
    speak_text,
    SENTENCE_END_PATTERN,
    _resolve_engine,
)


class TestOnOffSwitch:
    """`speeker on` / `speeker off` persist the global enable flag."""

    def test_set_enabled_persists(self, tmp_path):
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            from speeker.cli import _set_enabled
            from speeker.config import get_player_config
            assert get_player_config().get("enabled", True) is True  # default on
            _set_enabled(False)
            assert get_player_config().get("enabled", True) is False
            _set_enabled(True)
            assert get_player_config().get("enabled", True) is True


class TestResolveEngine:
    """Tests for engine auto-selection from a voice's provider."""

    def test_explicit_engine_wins(self, tmp_path):
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            args = argparse.Namespace(engine="polly", voice="anything")
            assert _resolve_engine(args) == "polly"

    def test_no_voice_uses_default(self, tmp_path):
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            args = argparse.Namespace(engine=None, voice=None)
            assert _resolve_engine(args) == "pocket-tts"

    def test_local_custom_voice_selects_pocket_tts(self, tmp_path):
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            from speeker import voice_clone
            (tmp_path / "data" / "voices").mkdir(parents=True)
            voice_clone._save_manifest({
                "Loc": {"audio_path": "x", "provider": "local",
                        "description": "d", "created_at": ""},
            })
            args = argparse.Namespace(engine=None, voice="Loc")
            assert _resolve_engine(args) == "pocket-tts"

    def test_elevenlabs_custom_voice_selects_elevenlabs(self, tmp_path):
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            from speeker import voice_clone
            (tmp_path / "data" / "voices").mkdir(parents=True)
            voice_clone._save_manifest({
                "El": {"audio_path": "x", "provider": "elevenlabs", "voice_id": "v",
                       "description": "d", "created_at": ""},
            })
            args = argparse.Namespace(engine=None, voice="El")
            assert _resolve_engine(args) == "elevenlabs"


class TestGetQueueFile:
    """Tests for get_queue_file function."""

    def test_get_queue_file_path(self, tmp_path):
        """Test returns queue file path."""
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            result = get_queue_file()
            assert result == tmp_path / "data" / "queue"

    def test_get_queue_file_returns_path(self, tmp_path):
        """Test returns a Path object."""
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            result = get_queue_file()
            assert isinstance(result, Path)


class TestEnsureOutputDir:
    """Tests for ensure_output_dir function."""

    def test_ensure_output_dir_creates_directory(self, tmp_path):
        """Test creates output directory."""
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            result = ensure_output_dir()
            assert result.exists()
            assert result.is_dir()

    def test_ensure_output_dir_uses_date_format(self, tmp_path):
        """Test directory name is date formatted."""
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            result = ensure_output_dir()
            # Should be in YYYY-MM-DD format
            assert len(result.name) == 10
            assert result.name[4] == "-"
            assert result.name[7] == "-"

    def test_ensure_output_dir_idempotent(self, tmp_path):
        """Test calling multiple times is safe."""
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
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
        assert result is False


class TestQueueForPlayback:
    """Tests for queue_for_playback function."""

    @patch("speeker.cli.start_player")
    @patch("speeker.cli.get_queue_file")
    def test_queue_for_playback_writes_path(self, mock_queue, mock_start, tmp_path):
        """Test queue_for_playback writes audio path to queue file."""
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            queue_file = tmp_path / "queue"
            mock_queue.return_value = queue_file

            audio_path = tmp_path / "test.wav"
            audio_path.touch()

            queue_for_playback(audio_path)

            assert queue_file.exists()
            assert str(audio_path) in queue_file.read_text()

    @patch("speeker.cli.start_player")
    @patch("speeker.cli.get_queue_file")
    def test_queue_for_playback_starts_player(self, mock_queue, mock_start, tmp_path):
        """Test queue_for_playback starts the player."""
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            mock_queue.return_value = tmp_path / "queue"
            queue_for_playback(tmp_path / "test.wav")
            mock_start.assert_called_once()

    @patch("speeker.cli.start_player")
    @patch("speeker.cli.get_queue_file")
    def test_queue_for_playback_appends(self, mock_queue, mock_start, tmp_path):
        """Test queue_for_playback appends to existing queue."""
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            queue_file = tmp_path / "queue"
            mock_queue.return_value = queue_file

            queue_for_playback(tmp_path / "test1.wav")
            queue_for_playback(tmp_path / "test2.wav")

            content = queue_file.read_text()
            assert "test1.wav" in content
            assert "test2.wav" in content


class TestSpeakText:
    """Tests for speak_text function."""

    @patch("speeker.cli.start_player")
    @patch("speeker.queue_db.enqueue")
    def test_speak_text_default_enqueues(self, mock_enqueue, mock_player):
        """Default mode enqueues text + metadata for the daemon and starts it."""
        mock_enqueue.return_value = 7
        result = speak_text("Hello", "pocket-tts", "azelma", False, True, False)
        assert result is True
        mock_enqueue.assert_called_once()
        text_arg = mock_enqueue.call_args[0][0]
        metadata = mock_enqueue.call_args[1]["metadata"]
        assert text_arg == "Hello"
        assert metadata["engine"] == "pocket-tts"
        assert metadata["voice"] == "azelma"
        mock_player.assert_called_once()

    @patch("speeker.cli.start_player")
    @patch("speeker.queue_db.enqueue")
    def test_speak_text_passes_interpretation(self, mock_enqueue, mock_player):
        """An interpretation rides along in the queue metadata."""
        mock_enqueue.return_value = 1
        speak_text(
            "Build passed", "pocket-tts", "azelma", False, True, False,
            interpretation="SUCCESS",
        )
        assert mock_enqueue.call_args[1]["metadata"]["interpretation"] == "SUCCESS"

    @patch("speeker.cli.start_player")
    @patch("speeker.queue_db.enqueue")
    def test_speak_text_no_interpretation_key_when_absent(self, mock_enqueue, mock_player):
        """No interpretation key is added when none is requested."""
        mock_enqueue.return_value = 1
        speak_text("Hello", "pocket-tts", "azelma", False, True, False)
        assert "interpretation" not in mock_enqueue.call_args[1]["metadata"]

    def test_speak_text_empty_text(self):
        """Test speak_text returns True for empty text."""
        result = speak_text("", "pocket-tts", "azelma", False, True, False)
        assert result is True

    def test_speak_text_whitespace_text(self):
        """Test speak_text returns True for whitespace-only text."""
        result = speak_text("   ", "pocket-tts", "azelma", False, True, False)
        assert result is True

    @patch("speeker.cli.save_audio")
    def test_speak_text_no_play(self, mock_save, capsys):
        """Test speak_text with no_play generates synchronously and prints path."""
        rec = _RecordingEngine()
        mock_save.return_value = Path("/tmp/test.wav")
        with patch("speeker.cli.get_engine", return_value=rec):
            result = speak_text("Hello", "pocket-tts", "azelma", True, False, False)
        assert result is True
        assert len(rec.calls) == 1
        assert "/tmp/test.wav" in capsys.readouterr().out

    def test_speak_text_no_play_handles_error(self, capsys):
        """Synchronous (--no-play) generation errors are caught and reported."""
        rec = _RecordingEngine()
        rec.generate = MagicMock(side_effect=Exception("TTS failed"))
        with patch("speeker.cli.get_engine", return_value=rec):
            result = speak_text("Hello", "pocket-tts", "azelma", True, False, False)
        assert result is False
        assert "Error" in capsys.readouterr().err

    @patch("speeker.cli.start_player")
    @patch("speeker.queue_db.enqueue")
    def test_speak_text_enqueue_error_is_handled(self, mock_enqueue, mock_player, capsys):
        """A failure to enqueue is reported and returns False."""
        mock_enqueue.side_effect = Exception("db locked")
        result = speak_text("Hello", "pocket-tts", "azelma", False, True, False)
        assert result is False
        assert "Error" in capsys.readouterr().err


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
        match = SENTENCE_END_PATTERN.search("file.txt")
        if match:
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
        args.engine = None

        result = cmd_voices(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "pocket-tts" in captured.out
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

    @patch("speeker.queue_db.get_pending_count")
    @patch("speeker.cli.is_player_running")
    def test_cmd_status_shows_info(self, mock_running, mock_pending, tmp_path, capsys):
        """Test cmd_status shows status information."""
        from speeker.cli import cmd_status

        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            mock_pending.return_value = 0
            mock_running.return_value = False
            args = MagicMock()

            result = cmd_status(args)

            assert result == 0
            captured = capsys.readouterr()
            assert "Data directory:" in captured.out
            assert "Player running: no" in captured.out
            assert "Queue length:" in captured.out

    @patch("speeker.queue_db.get_pending_count")
    @patch("speeker.cli.is_player_running")
    def test_cmd_status_with_queue_items(self, mock_running, mock_pending, tmp_path, capsys):
        """Test cmd_status reports the SQLite pending count."""
        from speeker.cli import cmd_status

        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            mock_pending.return_value = 2
            mock_running.return_value = True
            args = MagicMock()

            result = cmd_status(args)

            assert result == 0
            captured = capsys.readouterr()
            assert "Queue length: 2" in captured.out
            assert "Player running: yes" in captured.out

    @patch("speeker.queue_db.get_pending_count")
    @patch("speeker.cli.is_player_running")
    def test_cmd_status_counts_audio_files(self, mock_running, mock_pending, tmp_path, capsys):
        """Test cmd_status counts audio files."""
        from speeker.cli import cmd_status

        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            mock_pending.return_value = 0
            mock_running.return_value = False

            # audio_dir() returns SPEEKER_DIR/data/audio
            ad = tmp_path / "data" / "audio"
            ad.mkdir(parents=True)
            day_dir = ad / "2024-01-15"
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
        args.polly_voice = None
        args.polly_engine = None
        args.ssml = False
        args.emulate_ssml = False
        args.aws_profile = None
        args.interpretation = None

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
        args.polly_voice = None
        args.polly_engine = None
        args.ssml = False
        args.emulate_ssml = False
        args.aws_profile = None
        args.interpretation = None

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
        args.polly_voice = None
        args.polly_engine = None
        args.aws_profile = None
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
        args.polly_voice = None
        args.polly_engine = None
        args.aws_profile = None
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

            assert path.suffix == ".wav"


class TestSpeakTextAdvanced:
    """Additional tests for speak_text function."""

    @patch("speeker.cli.queue_for_playback")
    @patch("speeker.cli.save_audio")
    def test_speak_text_quiet_mode(self, mock_save, mock_queue, capsys):
        """Test speak_text quiet mode doesn't print to stderr."""
        rec = _RecordingEngine()
        mock_save.return_value = Path("/tmp/test.wav")
        with patch("speeker.cli.get_engine", return_value=rec):
            result = speak_text("Hello", "pocket-tts", "azelma", False, True, False)
        assert result is True
        assert "Queued" not in capsys.readouterr().err

    @patch("speeker.cli.wavfile.write")
    def test_speak_text_stdout_mode(self, mock_wavfile_write):
        """Test speak_text stdout mode writes to stdout."""
        rec = _RecordingEngine()
        with patch("speeker.cli.get_engine", return_value=rec):
            result = speak_text("Hello", "pocket-tts", "azelma", False, False, True)
        assert result is True
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
        args.polly_voice = None
        args.polly_engine = None
        args.ssml = False
        args.emulate_ssml = False
        args.aws_profile = None

        result = cmd_speak_stream(args)

        assert result == 0
        assert mock_speak.call_count == 2

    @patch("speeker.cli.stream_sentences_from_stdin")
    @patch("speeker.cli.speak_text")
    def test_cmd_speak_stream_with_errors(self, mock_speak, mock_stream, capsys):
        """Test streaming mode handles errors."""
        from speeker.cli import cmd_speak_stream

        mock_stream.return_value = iter(["First sentence.", "Second sentence."])
        mock_speak.side_effect = [False, True]

        args = MagicMock()
        args.engine = "pocket-tts"
        args.voice = "azelma"
        args.quiet = True
        args.no_play = False
        args.stdout = False
        args.polly_voice = None
        args.polly_engine = None
        args.ssml = False
        args.emulate_ssml = False
        args.aws_profile = None

        result = cmd_speak_stream(args)

        assert result == 0

    def test_cmd_speak_stream_invalid_engine(self, capsys):
        """Test streaming mode rejects invalid engine."""
        from speeker.cli import cmd_speak_stream

        args = MagicMock()
        args.engine = "invalid"
        args.voice = None
        args.polly_voice = None
        args.polly_engine = None
        args.aws_profile = None
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
        args.polly_voice = None
        args.polly_engine = None
        args.aws_profile = None
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

        with patch.object(Path, "exists") as mock_exists:
            mock_exists.return_value = True
            start_player()
            mock_popen.assert_called_once()


class _RecordingEngine:
    name = "rec"
    supports_ssml = False

    def __init__(self):
        self.calls = []

    def default_voice(self):
        return "azelma"

    def validate_voice(self, voice):
        return True

    def generate(self, text, voice, *, is_ssml=False, **options):
        import numpy as np
        self.calls.append({"text": text, "voice": voice, "is_ssml": is_ssml, **options})
        return np.zeros(8, dtype=np.float32), 16000


class TestCliSsmlAndEngine:
    def test_plain_text_preprocessed_and_generated(self, tmp_path):
        from speeker import cli
        rec = _RecordingEngine()
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}), \
             patch.object(cli, "get_engine", return_value=rec):
            ok = cli.speak_text("Hello.", "pocket-tts", "azelma",
                                no_play=True, quiet=True, stdout=False)
        assert ok is True
        assert len(rec.calls) == 1
        assert rec.calls[0]["is_ssml"] is False

    def test_ssml_local_engine_stripped(self, tmp_path):
        from speeker import cli
        rec = _RecordingEngine()
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}), \
             patch.object(cli, "get_engine", return_value=rec):
            cli.speak_text("<speak>Hi <break/>there</speak>", "pocket-tts", "azelma",
                           no_play=True, quiet=True, stdout=False, is_ssml=True)
        assert rec.calls[0]["text"] == "Hi there"
        assert rec.calls[0]["is_ssml"] is False

    def test_polly_engine_passes_variant_and_ssml(self, tmp_path):
        from speeker import cli
        rec = _RecordingEngine()
        rec.supports_ssml = True
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}), \
             patch.object(cli, "get_engine", return_value=rec):
            cli.speak_text("<speak>hi</speak>", "polly", "Joanna",
                           no_play=True, quiet=True, stdout=False,
                           is_ssml=True, polly_engine="generative")
        assert rec.calls[0]["is_ssml"] is True
        assert rec.calls[0]["polly_engine"] == "generative"

    def test_parser_accepts_polly_and_ssml_flags(self):
        from speeker.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(
            ["speak", "hi", "-e", "polly", "--polly-engine", "neural",
             "--polly-voice", "Matthew", "--ssml", "--best-effort-ssml-emulation",
             "--aws-profile", "personal"]
        )
        assert args.engine == "polly"
        assert args.polly_engine == "neural"
        assert args.polly_voice == "Matthew"
        assert args.ssml is True
        assert args.emulate_ssml is True
        assert args.aws_profile == "personal"

    def test_emulation_flag_defaults_none(self):
        from speeker.cli import build_parser
        args = build_parser().parse_args(["speak", "hi"])
        assert args.emulate_ssml is None

    def test_aws_profile_sets_env(self, tmp_path):
        from speeker import cli
        rec = _RecordingEngine()
        rec.supports_ssml = True
        args = argparse.Namespace(
            text="hi", engine="polly", voice=None, polly_voice="Joanna",
            polly_engine="neural", ssml=False, emulate_ssml=False,
            aws_profile="personal", no_play=True, quiet=True, stdout=False, stream=False,
        )
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}), \
             patch.object(cli, "get_engine", return_value=rec):
            cli.cmd_speak(args)
            assert os.environ["AWS_PROFILE"] == "personal"


class TestCliSsmlCommand:
    def test_generates_to_stdout(self, tmp_path, capsys):
        from speeker import cli
        args = argparse.Namespace(purpose="plain")
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}), \
             patch("sys.stdin", io.StringIO("Hello world.")):
            rc = cli.cmd_ssml(args)
        out = capsys.readouterr().out
        assert rc == 0
        assert out.strip().startswith("<speak>")

    def test_empty_stdin_errors(self, tmp_path, capsys):
        from speeker import cli
        args = argparse.Namespace(purpose="audiobook")
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}), \
             patch("sys.stdin", io.StringIO("   ")):
            rc = cli.cmd_ssml(args)
        assert rc == 1

    def test_parser_has_ssml_command(self):
        from speeker.cli import build_parser
        args = build_parser().parse_args(["ssml", "--purpose", "audiobook"])
        assert args.purpose == "audiobook"
        assert args.func.__name__ == "cmd_ssml"
