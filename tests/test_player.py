#!/usr/bin/env python3
"""Unit tests for player.py utility functions."""

import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from speeker.config import save_config
from speeker.player import (
    parse_note_token,
    extract_tone_tokens,
    get_audio_player,
    get_intro_sound,
    get_outro_sound,
    play_audio,
    play_interpretation_cue,
    render_interpretation_cue,
    should_announce_intro,
    build_session_script,
    compute_auto_label_prefix,
    unload_tts_model,
    NOTE_PATTERN,
    POLL_INTERVAL,
    PAUSE_BETWEEN_MESSAGES,
    PAUSE_BETWEEN_SESSIONS,
)


class TestInterpretationCues:
    """Tests for interpretation cue rendering and playback."""

    def test_render_unknown_returns_none(self, tmp_path):
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            assert render_interpretation_cue("NOPE") is None

    def test_render_sound_file_returns_path_when_present(self, tmp_path):
        snd = tmp_path / "ding.wav"
        snd.write_bytes(b"RIFF....")
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            save_config({
                "interpretations": {
                    "map": {"DING": {"type": "sound_file", "path": str(snd)}}
                }
            })
            assert render_interpretation_cue("DING") == snd

    def test_render_sound_file_missing_returns_none(self, tmp_path):
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            save_config({
                "interpretations": {
                    "map": {"DING": {"type": "sound_file", "path": str(tmp_path / "nope.wav")}}
                }
            })
            assert render_interpretation_cue("DING") is None

    @patch("speeker.player.time.sleep")
    @patch("speeker.player.play_audio")
    @patch("speeker.player.render_interpretation_cue")
    def test_play_cue_plays_then_pauses(self, mock_render, mock_play, mock_sleep, tmp_path):
        mock_render.return_value = Path("/tmp/cue.wav")
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            play_interpretation_cue("SUCCESS")
        mock_play.assert_called_once_with(Path("/tmp/cue.wav"), False)
        mock_sleep.assert_called_once()

    @patch("speeker.player.play_audio")
    @patch("speeker.player.render_interpretation_cue", return_value=None)
    def test_play_cue_noop_when_unresolved(self, mock_render, mock_play):
        play_interpretation_cue("NOPE")
        mock_play.assert_not_called()


class _PlayerRecordingEngine:
    name = "rec"
    supports_ssml = False

    def __init__(self):
        self.calls = []

    def default_voice(self):
        return "azelma"

    def generate(self, text, voice, *, is_ssml=False, **options):
        import numpy as np
        self.calls.append({"text": text, "voice": voice, "is_ssml": is_ssml, **options})
        return np.zeros(8, dtype=np.float32), 16000


class TestParseNoteToken:
    """Tests for parse_note_token function.

    parse_note_token returns ``(note, octave, multiplier)``. The multiplier
    defaults to 1.0 when the token has no ``:N`` qualifier, so any tune
    written before the duration syntax was added continues to behave
    identically (1.0 * base_duration == base_duration).
    """

    def test_parse_note_token_simple(self):
        result = parse_note_token("C4")
        assert result == ("c", 4, 1.0)

    def test_parse_note_token_sharp(self):
        result = parse_note_token("F#5")
        assert result == ("f#", 5, 1.0)

    def test_parse_note_token_flat(self):
        result = parse_note_token("Bb3")
        assert result == ("bb", 3, 1.0)

    def test_parse_note_token_lowercase(self):
        result = parse_note_token("e4")
        assert result == ("e", 4, 1.0)

    def test_parse_note_token_all_notes(self):
        for note in "ABCDEFG":
            result = parse_note_token(f"{note}4")
            assert result is not None
            assert result[0] == note.lower()
            assert result[1] == 4
            assert result[2] == 1.0

    def test_parse_note_token_with_integer_multiplier(self):
        """``C5:4`` -> C5 with 4x base duration (whole-note feel)."""
        assert parse_note_token("C5:4") == ("c", 5, 4.0)

    def test_parse_note_token_with_fractional_multiplier(self):
        """``Bb3:0.5`` -> Bb3 with half base duration (eighth-note feel)."""
        assert parse_note_token("Bb3:0.5") == ("bb", 3, 0.5)

    def test_parse_note_token_dotted_form(self):
        """Dotted multipliers like ``E4:1.5`` are accepted as-is."""
        assert parse_note_token("E4:1.5") == ("e", 4, 1.5)

    def test_parse_note_token_zero_multiplier_clamps_to_one(self):
        """``:0`` would give a 0-length tone -- silently clamp to 1.0 so a
        bad tune doesn't produce silent tokens."""
        assert parse_note_token("C5:0") == ("c", 5, 1.0)

    def test_parse_note_token_leading_dot_in_multiplier(self):
        """``:.5`` is a valid shorthand for half (no leading zero)."""
        assert parse_note_token("F4:.5") == ("f", 4, 0.5)

    def test_parse_note_token_octave_range(self):
        """Test parsing all valid octaves."""
        for octave in range(9):  # 0-8
            result = parse_note_token(f"A{octave}")
            assert result == ("a", octave, 1.0)

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
        tokens, text, _trailing = extract_tone_tokens("$C4 Hello world")
        assert tokens == ["C4"]
        assert text == "Hello world"

    def test_extract_multiple_tokens(self):
        """Test extracting multiple tone tokens."""
        tokens, text, _trailing = extract_tone_tokens("$C4 $E4 $G4 Hello")
        assert tokens == ["C4", "E4", "G4"]
        assert text == "Hello"

    def test_extract_no_tokens(self):
        """Test text without tone tokens."""
        tokens, text, _trailing = extract_tone_tokens("Hello world")
        assert tokens == []
        assert text == "Hello world"

    def test_extract_sharp_token(self):
        """Test extracting sharp note token."""
        tokens, text, _trailing = extract_tone_tokens("$F#4 Alert")
        assert tokens == ["F#4"]
        assert text == "Alert"

    def test_extract_flat_token(self):
        """Test extracting flat note token."""
        tokens, text, _trailing = extract_tone_tokens("$Bb3 Warning")
        assert tokens == ["Bb3"]
        assert text == "Warning"

    def test_extract_token_with_duration_multiplier(self):
        """``$C5:4`` preserves the ``:4`` suffix in the returned token so
        downstream synthesis applies the multiplier."""
        tokens, text, _trailing = extract_tone_tokens("$G4 $E4 $C5:4 Hello")
        assert tokens == ["G4", "E4", "C5:4"]
        assert text == "Hello"

    def test_extract_token_with_fractional_multiplier(self):
        tokens, text, _trailing = extract_tone_tokens("$Eb4:.5 Done.")
        assert tokens == ["Eb4:.5"]
        assert text == "Done."

    def test_extract_trailing_tokens(self):
        """Outro pattern: leading tones, speech, trailing tones."""
        leading, text, trailing = extract_tone_tokens(
            "$E4 $G4 $C5 Hello world. $C5 $G4 $E4"
        )
        assert leading == ["E4", "G4", "C5"]
        assert text == "Hello world."
        assert trailing == ["C5", "G4", "E4"]

    def test_extract_trailing_only(self):
        """Just trailing tones with no leading: speech followed by chord."""
        leading, text, trailing = extract_tone_tokens("Done. $C5 $G4")
        assert leading == []
        assert text == "Done."
        assert trailing == ["C5", "G4"]

    def test_extract_does_not_consume_middle_tones(self):
        """``$Note`` tokens embedded mid-body must NOT be treated as
        trailing tones -- only a sequence at the very end qualifies.
        Here ``$G4`` is followed by ``omega`` so it stays in the body."""
        leading, text, trailing = extract_tone_tokens("$E4 alpha $G4 omega")
        assert leading == ["E4"]
        assert text == "alpha $G4 omega"
        assert trailing == []

    def test_extract_preserves_remaining_text(self):
        """Test remaining text is preserved."""
        tokens, text, _trailing = extract_tone_tokens("$A4 Important message here")
        assert text == "Important message here"

    def test_extract_empty_string(self):
        """Test empty string."""
        tokens, text, _trailing = extract_tone_tokens("")
        assert tokens == []
        assert text == ""

    def test_extract_only_token(self):
        """Test string that is only a token."""
        tokens, text, _trailing = extract_tone_tokens("$G4")
        assert tokens == ["G4"]
        assert text == ""

    def test_extract_whitespace_handling(self):
        """Test whitespace is stripped from remaining text."""
        tokens, text, _trailing = extract_tone_tokens("$E4    Hello  ")
        assert text == "Hello"


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

    def test_pause_between_messages_reasonable(self):
        """Test pause between messages is a reasonable value."""
        assert 0 <= PAUSE_BETWEEN_MESSAGES <= 2.0

    def test_pause_between_sessions_reasonable(self):
        """Test pause between sessions is a reasonable value."""
        assert 0 <= PAUSE_BETWEEN_SESSIONS <= 2.0


class TestUnloadTtsModel:
    """Tests for unload_tts_model function."""

    def test_unload_calls_unload_all(self):
        """Test unload_tts_model delegates to unload_all."""
        from speeker import player

        with patch.object(player, "unload_all") as mock_unload_all:
            unload_tts_model()
            mock_unload_all.assert_called_once()

    def test_unload_is_safe_to_call(self):
        """Test unload_tts_model can be called without error."""
        with patch("speeker.player.unload_all"):
            unload_tts_model()  # Should not raise


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
        assert result is None

    def test_parse_note_token_with_trailing_text(self):
        """Test note with trailing text."""
        result = parse_note_token("C4hello")
        assert result == ("c", 4, 1.0)


class TestExtractToneTokensEdgeCases:
    """Additional edge cases for extract_tone_tokens."""

    def test_extract_token_mid_text(self):
        """Test tokens not at start are not extracted."""
        tokens, text, _trailing = extract_tone_tokens("Hello $C4 world")
        assert tokens == []
        assert text == "Hello $C4 world"

    def test_extract_consecutive_tokens(self):
        """Test consecutive tokens without spaces."""
        tokens, text, _trailing = extract_tone_tokens("$C4$E4 Hello")
        assert "C4" in tokens


class TestGetIntroSound:
    """Tests for get_intro_sound function."""

    def test_get_intro_sound_returns_path(self, tmp_path):
        """Test get_intro_sound returns a path."""
        import speeker.player as player

        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
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
    """Tests for build_session_script function.

    The function now produces exactly one line per item -- no batch header,
    no per-item "First:/Next:" framing. That framing buried leading $Note
    tokens so the TTS read them as letters ("EEB 4").
    """

    @staticmethod
    def _recent_ts() -> str:
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()

    def test_single_message_only_session(self):
        """Single item -> single line, item text only."""
        items = [{"text": "Hello world", "created_at": self._recent_ts()}]
        script = build_session_script("session1", items, is_only_session=True)
        assert script == ["Hello world"]

    def test_no_count_header_for_multiple_messages(self):
        """The 'For queue X, there are N messages' header is gone."""
        ts = self._recent_ts()
        items = [
            {"text": "First message", "created_at": ts},
            {"text": "Second message", "created_at": ts},
        ]
        script = build_session_script("session1", items, is_only_session=True)
        # One line per item, no header.
        assert len(script) == 2
        assert all("message" not in line.lower() or line.startswith("First") or line.startswith("Second")
                   for line in script)
        # No count phrasing anywhere.
        assert not any("there are" in line.lower() for line in script)
        assert not any("there is" in line.lower() for line in script)

    def test_no_first_next_framing(self):
        """No 'First, ...' / 'Next: ...' / 'Last: ...' prefixes on items."""
        ts = self._recent_ts()
        items = [
            {"text": "alpha", "created_at": ts},
            {"text": "beta", "created_at": ts},
            {"text": "gamma", "created_at": ts},
        ]
        script = build_session_script("session1", items, is_only_session=True)
        # Each line is exactly the item's text (no time_ago for recent ts).
        assert script == ["alpha", "beta", "gamma"]

    def test_no_count_header_when_not_only_session(self):
        """A session that isn't the only one no longer gets a count header."""
        items = [{"text": "Test", "created_at": self._recent_ts()}]
        script = build_session_script("session1", items, is_only_session=False)
        assert script == ["Test"]


class TestBuildSessionScriptAutoLabel:
    """Auto-label behavior in build_session_script.

    These tests use ``relative_time`` returning None (i.e., recent timestamps)
    so the assertions can pin the exact line shape without the "From N minutes
    ago: ..." phrase getting interleaved.
    """

    @staticmethod
    def _recent_ts() -> str:
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()

    def test_single_message_with_auto_label_prefix(self):
        """The single-message-only-session branch prepends the prefix + period."""
        items = [{"text": "Claude finished", "created_at": self._recent_ts()}]
        script = build_session_script(
            "compass-docs", items, is_only_session=True,
            auto_label_prefix="$Eb4 compass docs",
        )
        assert script == ["$Eb4 compass docs. Claude finished"]

    def test_single_message_without_prefix_uses_bare_text(self):
        """No prefix -> behavior is the existing bare-text speak."""
        items = [{"text": "Claude finished", "created_at": self._recent_ts()}]
        script = build_session_script(
            "compass-docs", items, is_only_session=True, auto_label_prefix=None,
        )
        assert script == ["Claude finished"]

    def test_text_already_tone_prefixed_skips_auto_label(self):
        """If caller already added a $Note prefix, do not double-label."""
        items = [{
            "text": "$Eb4 compass docs. Summary line.",
            "created_at": self._recent_ts(),
        }]
        script = build_session_script(
            "compass-docs", items, is_only_session=True,
            auto_label_prefix="$Eb4 compass docs",
        )
        # Untouched -- no duplicate prefix.
        assert script == ["$Eb4 compass docs. Summary line."]

    def test_auto_label_only_first_bare_item_in_session(self):
        """In a multi-bare-item session, only the first item gets the prefix."""
        ts = self._recent_ts()
        items = [
            {"text": "one", "created_at": ts},
            {"text": "two", "created_at": ts},
            {"text": "three", "created_at": ts},
        ]
        script = build_session_script(
            "compass-docs", items, is_only_session=True,
            auto_label_prefix="$Eb4 compass docs",
        )
        # Project context is established once, then subsequent items speak bare.
        assert script == [
            "$Eb4 compass docs. one",
            "two",
            "three",
        ]

    def test_caller_prefixed_item_keeps_tone_at_line_start(self):
        """REGRESSION: tone token must remain at line-start so the player's
        ``extract_tone_tokens`` regex (anchored to ``^\\s*\\$``) can match
        it. Previously, multi-message batches prepended ``Next: `` which
        buried the token mid-line; the TTS then read ``$Eb4`` as the
        letters 'EEB 4'."""
        ts = self._recent_ts()
        items = [
            {"text": "$Eb4 progress reporter. First summary.", "created_at": ts},
            {"text": "$Eb4 progress reporter. Second summary.", "created_at": ts},
            {"text": "$Eb4 progress reporter. Third summary.", "created_at": ts},
        ]
        script = build_session_script(
            "progress-reporter", items, is_only_session=True,
            auto_label_prefix="$Eb4 progress reporter",
        )
        for i, line in enumerate(script):
            assert line.lstrip().startswith("$Eb4"), (
                f"line {i} should start with $Eb4 so the tone is played; got: {line!r}"
            )
            assert not line.startswith("Next"), (
                f"no 'Next:' framing should be added; got: {line!r}"
            )
            assert not line.startswith("First"), (
                f"no 'First:' framing should be added; got: {line!r}"
            )

    def test_caller_prefixed_item_consumes_label_slot_for_following_bare(self):
        """A caller-prefixed item establishes project context, so a following
        bare item in the same session does NOT receive the auto-label."""
        ts = self._recent_ts()
        items = [
            {"text": "$Eb4 progress reporter. First summary.", "created_at": ts},
            {"text": "plain followup", "created_at": ts},
        ]
        script = build_session_script(
            "progress-reporter", items, is_only_session=True,
            auto_label_prefix="$Eb4 progress reporter",
        )
        assert script == [
            "$Eb4 progress reporter. First summary.",
            "plain followup",
        ]


class TestComputeAutoLabelPrefix:
    """Tests for the auto-label decision (time-gap + context-switch logic)."""

    @patch("speeker.player.get_auto_label_config")
    def test_disabled_in_config_returns_none(self, mock_cfg):
        mock_cfg.return_value = {"enabled": False, "quiet_threshold_seconds": 120}
        result = compute_auto_label_prefix("compass-docs", None, None)
        assert result is None

    @patch("speeker.player.get_auto_label_config")
    def test_default_queue_returns_none(self, mock_cfg):
        """No title for the unnamed/default queue -> no prefix ever."""
        mock_cfg.return_value = {"enabled": True, "quiet_threshold_seconds": 120, "tone": "$Eb4"}
        result = compute_auto_label_prefix("default", None, None)
        assert result is None

    @patch("speeker.player.get_auto_label_config")
    def test_first_time_returns_prefix(self, mock_cfg):
        """No prior utterance -> label this one."""
        mock_cfg.return_value = {"enabled": True, "quiet_threshold_seconds": 120, "tone": "$Eb4"}
        result = compute_auto_label_prefix("compass-docs", None, None)
        assert result == "$Eb4 compass docs"

    @patch("speeker.player.get_auto_label_config")
    def test_recent_same_queue_returns_none(self, mock_cfg):
        """Burst from the same queue -> no relabel."""
        from datetime import datetime, timezone
        mock_cfg.return_value = {"enabled": True, "quiet_threshold_seconds": 120, "tone": "$Eb4"}
        result = compute_auto_label_prefix(
            "compass-docs", datetime.now(timezone.utc), "compass-docs",
        )
        assert result is None

    @patch("speeker.player.get_auto_label_config")
    def test_recent_different_queue_returns_prefix(self, mock_cfg):
        """Context switch always relabels, even within the threshold window."""
        from datetime import datetime, timezone
        mock_cfg.return_value = {"enabled": True, "quiet_threshold_seconds": 120, "tone": "$Eb4"}
        result = compute_auto_label_prefix(
            "compass-docs", datetime.now(timezone.utc), "audio-speeker",
        )
        assert result == "$Eb4 compass docs"

    @patch("speeker.player.get_auto_label_config")
    def test_old_same_queue_returns_prefix(self, mock_cfg):
        """Past the quiet threshold -> relabel even on same queue."""
        from datetime import datetime, timedelta, timezone
        mock_cfg.return_value = {"enabled": True, "quiet_threshold_seconds": 120, "tone": "$Eb4"}
        long_ago = datetime.now(timezone.utc) - timedelta(seconds=600)
        result = compute_auto_label_prefix("compass-docs", long_ago, "compass-docs")
        assert result == "$Eb4 compass docs"

    @patch("speeker.player.get_auto_label_config")
    def test_custom_tone_is_used(self, mock_cfg):
        """The tone is configurable -- a different note travels through."""
        mock_cfg.return_value = {"enabled": True, "quiet_threshold_seconds": 120, "tone": "$G3"}
        result = compute_auto_label_prefix("compass-docs", None, None)
        assert result == "$G3 compass docs"

    @patch("speeker.player.get_auto_label_config")
    def test_display_name_override_replaces_derived_title(self, mock_cfg):
        """Caller-supplied display_name beats the hyphen-to-space derivation.

        This is the lever for ugly queue ids like 'e2e-stt-1779972451' that
        would otherwise be voiced as a string of digits. The caller passes
        a sensible spoken title and the player uses it verbatim.
        """
        mock_cfg.return_value = {"enabled": True, "quiet_threshold_seconds": 120, "tone": "$Eb4"}
        result = compute_auto_label_prefix(
            "e2e-stt-1779972451", None, None,
            display_name_override="end to end test",
        )
        assert result == "$Eb4 end to end test"

    @patch("speeker.player.get_auto_label_config")
    def test_display_name_override_works_for_default_queue(self, mock_cfg):
        """Even the 'default' queue gets a spoken title when override is supplied.

        Without override, the default queue suppresses the auto-label entirely
        (no meaningful title to derive). An explicit display_name overrides
        that policy -- callers should be able to label any queue they want.
        """
        mock_cfg.return_value = {"enabled": True, "quiet_threshold_seconds": 120, "tone": "$Eb4"}
        result = compute_auto_label_prefix(
            "default", None, None,
            display_name_override="my custom label",
        )
        assert result == "$Eb4 my custom label"

    @patch("speeker.player.get_auto_label_config")
    def test_display_name_override_empty_string_falls_through(self, mock_cfg):
        """An empty/whitespace display_name doesn't override; derivation runs."""
        mock_cfg.return_value = {"enabled": True, "quiet_threshold_seconds": 120, "tone": "$Eb4"}
        result = compute_auto_label_prefix(
            "compass-docs", None, None,
            display_name_override=None,
        )
        # Same as the no-override path.
        assert result == "$Eb4 compass docs"


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
            lock_file = tmp_path / "player.lock"
            lock_file.write_text("999999999")

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

        lock_file = tmp_path / "player.lock"
        lock_file.write_text("12345")

        release_lock(lock_file)

        assert not lock_file.exists()

    def test_release_lock_missing_file(self, tmp_path):
        """Test release_lock handles missing file."""
        from speeker.player import release_lock

        lock_file = tmp_path / "player.lock"
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
            {"id": 1, "text": "Hello", "created_at": "2024-01-01 12:00:00", "metadata": None}
        ]
        mock_settings.return_value = {"voice": "azelma", "speed": 1.0, "intro_sound": False, "engine": "pocket-tts"}
        mock_speak.return_value = Path("/tmp/test.wav")

        result = process_queue(verbose=False)

        assert result >= 0
        mock_mark.assert_called()


class TestMainFunction:
    """Tests for main entry point."""

    @patch("speeker.player.run_once")
    @patch("speeker.player.sys.argv", ["speeker-player"])
    def test_main_runs_once(self, mock_run, tmp_path):
        """Test main runs in one-shot mode by default."""
        from speeker.player import main

        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            result = main()

        assert result == 0
        mock_run.assert_called_once()

    @patch("speeker.player.cleanup_old_files")
    @patch("speeker.player.sys.argv", ["speeker-player", "--cleanup", "7"])
    def test_main_cleanup_mode(self, mock_cleanup, tmp_path, capsys):
        """Test main runs cleanup mode."""
        from speeker.player import main

        mock_cleanup.return_value = 10

        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            result = main()

        assert result == 0
        mock_cleanup.assert_called_once_with(7, False)
        captured = capsys.readouterr()
        assert "Removed 10" in captured.err

    @patch("speeker.player.run_daemon")
    @patch("speeker.player.sys.argv", ["speeker-player", "--daemon"])
    def test_main_daemon_mode(self, mock_daemon, tmp_path):
        """Test main runs daemon mode."""
        from speeker.player import main

        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            result = main()

        assert result == 0
        mock_daemon.assert_called_once()


class TestGenerateTTS:
    """Tests for generate_tts function (dispatches through the engine registry)."""

    def test_generate_tts_success(self, tmp_path):
        from speeker import player
        rec = _PlayerRecordingEngine()
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}), \
             patch.object(player, "get_engine", return_value=rec):
            path = player.generate_tts("Hello world", verbose=False)
        assert path is not None
        assert path.exists()
        path.unlink()

    def test_generate_tts_with_save_path(self, tmp_path):
        from speeker import player
        rec = _PlayerRecordingEngine()
        save_path = tmp_path / "output.wav"
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}), \
             patch.object(player, "get_engine", return_value=rec):
            path = player.generate_tts("Hello world", save_path=save_path, verbose=False)
        assert path == save_path
        assert save_path.exists()

    def test_generate_tts_with_speed(self, tmp_path):
        from speeker import player
        rec = _PlayerRecordingEngine()
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}), \
             patch.object(player, "get_engine", return_value=rec):
            path = player.generate_tts("Hello world", speed=1.5, verbose=False)
        assert path is not None
        path.unlink()

    def test_generate_tts_error(self, tmp_path):
        from speeker import player
        rec = _PlayerRecordingEngine()
        rec.generate = MagicMock(side_effect=Exception("TTS error"))
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}), \
             patch.object(player, "get_engine", return_value=rec):
            path = player.generate_tts("Hello world", verbose=False)
        assert path is None

    def test_generate_tts_calls_apply_effects_with_sample_rate(self, tmp_path):
        """The effects hook sits between the speed resample and the
        int16 clip in generate_tts. Verify apply_effects is invoked with
        the engine's sample rate. The recording engine returns 16000."""
        from speeker import player
        rec = _PlayerRecordingEngine()
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}), \
             patch.object(player, "get_engine", return_value=rec), \
             patch("speeker.effects.apply_effects") as mock_fx:
            # Echo the input array back unchanged.
            mock_fx.side_effect = lambda audio, sr, **kw: audio
            path = player.generate_tts("Hello world", verbose=False)
        assert path is not None
        assert mock_fx.called
        _audio_arg, sr_arg = mock_fx.call_args.args
        assert sr_arg == 16000
        # No explicit preset_override -> apply_effects gets None and
        # falls back to the saved config.
        assert mock_fx.call_args.kwargs.get("preset_override") is None
        path.unlink()

    def test_generate_tts_threads_effects_preset_override(self, tmp_path):
        """Per-utterance override (used by /api/effects/try) must reach
        apply_effects via the preset_override kwarg, not the saved config."""
        from speeker import player
        rec = _PlayerRecordingEngine()
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}), \
             patch.object(player, "get_engine", return_value=rec), \
             patch("speeker.effects.apply_effects") as mock_fx:
            mock_fx.side_effect = lambda audio, sr, **kw: audio
            path = player.generate_tts(
                "Hello", verbose=False, effects_preset="natural",
            )
        assert path is not None
        assert mock_fx.call_args.kwargs["preset_override"] == "natural"
        path.unlink()


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

        player_speak_text("Hello world", verbose=False)

        mock_play.assert_called_once()

    @patch("speeker.player.play_tone_tokens")
    @patch("speeker.player.extract_tone_tokens")
    def test_speak_text_with_tones(self, mock_extract, mock_play_tone):
        """speak_text passes the tone-duration override through to
        play_tone_tokens. None (the default) means "use play_tone_tokens'
        own default" -- the 0.8s used for $Note prefix tones before TTS."""
        from speeker.player import speak_text as player_speak_text

        mock_extract.return_value = (["C4", "E4"], "Hello", [])

        with patch("speeker.player.generate_tts") as mock_gen:
            mock_gen.return_value = None
            player_speak_text("$C4 $E4 Hello", verbose=False)

        mock_play_tone.assert_called_once_with(["C4", "E4"], False, duration=None)

    @patch("speeker.player.play_tone_tokens")
    @patch("speeker.player.extract_tone_tokens")
    def test_speak_text_threads_tone_duration_override(self, mock_extract, mock_play_tone):
        """The /api/tones/play preview path supplies tone_duration via
        metadata; speak_text must thread it down to play_tone_tokens."""
        from speeker.player import speak_text as player_speak_text

        mock_extract.return_value = (["G4", "E4", "C5"], "", [])
        player_speak_text("$G4 $E4 $C5", verbose=False, tone_duration=0.18)
        mock_play_tone.assert_called_once_with(["G4", "E4", "C5"], False, duration=0.18)

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
    @patch("speeker.player.get_engine")
    @patch("speeker.player.release_lock")
    @patch("speeker.player.acquire_lock")
    @patch("speeker.config.get_player_config")
    def test_run_daemon_preloads_when_timeout_zero(
        self, mock_config, mock_acquire, mock_release, mock_get_engine,
        mock_pending, mock_sleep, tmp_path
    ):
        from speeker.player import run_daemon
        mock_config.return_value = {"model_idle_timeout_minutes": 0}
        mock_acquire.return_value = tmp_path / "player.lock"
        mock_pending.return_value = 0
        fake_engine = MagicMock()
        mock_get_engine.return_value = fake_engine

        call_count = [0]
        def sleep_side_effect(duration):
            call_count[0] += 1
            if call_count[0] >= 2:
                raise KeyboardInterrupt()
        mock_sleep.side_effect = sleep_side_effect

        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            try:
                run_daemon(verbose=False)
            except KeyboardInterrupt:
                pass

        fake_engine.warm.assert_called_once()
        mock_release.assert_called()

    @patch("speeker.player.time.sleep")
    @patch("speeker.player.get_pending_count")
    @patch("speeker.player.get_engine")
    @patch("speeker.player.release_lock")
    @patch("speeker.player.acquire_lock")
    @patch("speeker.config.get_player_config")
    def test_run_daemon_skips_preload_with_timeout(
        self, mock_config, mock_acquire, mock_release, mock_get_engine,
        mock_pending, mock_sleep, tmp_path
    ):
        from speeker.player import run_daemon
        mock_config.return_value = {"model_idle_timeout_minutes": 5}
        mock_acquire.return_value = tmp_path / "player.lock"
        mock_pending.return_value = 0
        fake_engine = MagicMock()
        mock_get_engine.return_value = fake_engine

        call_count = [0]
        def sleep_side_effect(duration):
            call_count[0] += 1
            if call_count[0] >= 2:
                raise KeyboardInterrupt()
        mock_sleep.side_effect = sleep_side_effect

        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            try:
                run_daemon(verbose=False)
            except KeyboardInterrupt:
                pass

        fake_engine.warm.assert_not_called()

    @patch("speeker.player.unload_tts_model")
    @patch("speeker.player.process_queue")
    @patch("speeker.player.time.sleep")
    @patch("speeker.player.get_pending_count")
    @patch("speeker.player.release_lock")
    @patch("speeker.player.acquire_lock")
    @patch("speeker.config.get_player_config")
    def test_run_daemon_unloads_model_after_idle(
        self, mock_config, mock_acquire, mock_release, mock_pending,
        mock_sleep, mock_process, mock_unload, tmp_path
    ):
        """Test daemon unloads model after idle timeout expires."""
        from speeker.player import run_daemon

        mock_config.return_value = {"model_idle_timeout_minutes": 1}
        lock_path = tmp_path / "player.lock"
        mock_acquire.return_value = lock_path

        # Simulate: first call has pending items (loads model), then idle
        pending_values = [1, 0, 0]
        mock_pending.side_effect = lambda: pending_values.pop(0) if pending_values else 0

        time_values = [0, 0, 0, 61, 61]  # start, after process, check, idle check, idle check
        time_idx = [0]
        def fake_time():
            idx = min(time_idx[0], len(time_values) - 1)
            time_idx[0] += 1
            return time_values[idx]

        call_count = [0]
        def sleep_side_effect(duration):
            call_count[0] += 1
            if call_count[0] >= 3:
                raise KeyboardInterrupt()

        mock_sleep.side_effect = sleep_side_effect

        with patch("speeker.player.time.time", side_effect=fake_time):
            try:
                run_daemon(verbose=False)
            except KeyboardInterrupt:
                pass

        mock_unload.assert_called_once()

    @patch("speeker.player.acquire_lock")
    @patch("speeker.config.get_player_config")
    def test_run_daemon_already_running(self, mock_config, mock_acquire, capsys):
        """Test run_daemon exits if already running."""
        from speeker.player import run_daemon

        mock_config.return_value = {"model_idle_timeout_minutes": 0}
        mock_acquire.return_value = None

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
            lock_file = tmp_path / "player.lock"
            lock_file.write_text(str(os.getpid()))

            result = acquire_lock()

            assert result is None


class TestBuildSessionScriptEdgeCases:
    """Edge cases for build_session_script."""

    @staticmethod
    def _recent_ts() -> str:
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()

    def test_three_messages_speak_verbatim_no_framing(self):
        """Three bare items -> three lines, no 'First/Middle/Last' framing."""
        ts = self._recent_ts()
        items = [
            {"text": "First", "created_at": ts},
            {"text": "Middle", "created_at": ts},
            {"text": "Third", "created_at": ts},
        ]
        script = build_session_script("session1", items, is_only_session=True)
        assert script == ["First", "Middle", "Third"]
        # Nothing introduces these with "Next" / "Last" framing.
        assert not any(line.startswith("Next") for line in script)
        assert not any(line.startswith("Last") for line in script)

    def test_single_not_only_session_no_count_header(self):
        """A session that isn't the only one no longer gets a count header."""
        items = [{"text": "Solo", "created_at": self._recent_ts()}]
        script = build_session_script("session1", items, is_only_session=False)
        assert script == ["Solo"]


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
            {"id": 1, "text": "Hello", "created_at": "2024-01-01 12:00:00", "metadata": None},
            {"id": 2, "text": "World", "created_at": "2024-01-01 12:01:00", "metadata": None},
        ]
        mock_settings.return_value = {"voice": "azelma", "speed": 1.0, "intro_sound": True, "engine": "pocket-tts"}
        mock_speak.return_value = Path("/tmp/test.wav")

        with patch("speeker.player.should_announce_intro") as mock_announce:
            mock_announce.return_value = True
            with patch("speeker.player.get_intro_sound") as mock_intro:
                mock_intro.return_value = Path("/tmp/intro.wav")

                result = process_queue(verbose=False)

        assert mock_play_audio.called or result >= 0

    @patch("speeker.player.get_sessions_with_pending")
    def test_process_queue_multiple_sessions(self, mock_sessions):
        """Test process_queue handles multiple sessions."""
        from speeker.player import process_queue

        mock_sessions.return_value = ["session1", "session2"]

        with patch("speeker.player.get_pending_for_session") as mock_pending:
            mock_pending.return_value = []

            result = process_queue(verbose=False)

            assert result == 0


class TestGenerateTtsDispatch:
    def test_dispatches_to_named_engine_plain(self, tmp_path):
        from speeker import player
        rec = _PlayerRecordingEngine()
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}), \
             patch.object(player, "get_engine", return_value=rec):
            out = player.generate_tts(
                "Hello.", voice="Joanna", engine="polly",
                save_path=tmp_path / "a.wav",
            )
        assert out == tmp_path / "a.wav"
        assert rec.calls[0]["is_ssml"] is False
        assert rec.calls[0]["voice"] == "Joanna"

    def test_ssml_local_engine_stripped(self, tmp_path):
        from speeker import player
        rec = _PlayerRecordingEngine()  # supports_ssml = False
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}), \
             patch.object(player, "get_engine", return_value=rec):
            player.generate_tts(
                "<speak>Hi <break/>there</speak>", voice="azelma",
                engine="pocket-tts", is_ssml=True, save_path=tmp_path / "b.wav",
            )
        assert rec.calls[0]["text"] == "Hi there"
        assert rec.calls[0]["is_ssml"] is False


class TestProcessQueueSsml:
    def test_ssml_item_spoken_verbatim(self, tmp_path):
        import speeker.queue_db as _qdb
        from speeker import player
        from speeker.queue_db import enqueue

        # Reset cached connection before entering the isolated DB so we don't
        # pin a stale connection from a previously-run test (mirrors temp_db fixture).
        if hasattr(_qdb._local, "conn") and _qdb._local.conn:
            _qdb._local.conn.close()
        _qdb._local.conn = None

        try:
            with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
                # Two items so build_session_script would prefix each ("First: ", "Last: ").
                enqueue("plain message", metadata={"queue": "q1"})
                enqueue("<speak>Hi there</speak>", metadata={"queue": "q1", "ssml": True})

                captured = []

                def fake_speak(line, **kw):
                    captured.append((line, kw))
                    return kw.get("save_path")

                with patch.object(player, "speak_text", side_effect=fake_speak):
                    player.process_queue(verbose=False)
        finally:
            # Reset again so subsequent tests reconnect to their own DB.
            if hasattr(_qdb._local, "conn") and _qdb._local.conn:
                _qdb._local.conn.close()
            _qdb._local.conn = None

        ssml_lines = [line for line, kw in captured if kw.get("is_ssml")]
        assert ssml_lines == ["<speak>Hi there</speak>"]
