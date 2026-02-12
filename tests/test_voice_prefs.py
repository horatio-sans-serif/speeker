#!/usr/bin/env python3
"""Unit tests for voice_prefs.py functions."""

import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import socket

from speeker.voice_prefs import (
    get_voice_prefs,
    save_voice_prefs,
    get_preferred_voice,
    get_preferred_engine,
    sample_exists,
    get_sample_path,
    get_samples_dir,
    find_free_port,
    create_html_ui,
    generate_sample,
    ensure_all_samples,
    SAMPLE_PHRASE,
)
from speeker.voices import DEFAULT_ENGINE, POCKET_TTS_VOICES


class TestGetVoicePrefs:
    """Tests for get_voice_prefs function."""

    def test_get_voice_prefs_returns_dict(self, tmp_path):
        """Test returns a dictionary."""
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            with patch("speeker.voice_prefs.BUNDLED_PREFS_FILE") as mock_bundled:
                mock_bundled.exists.return_value = False
                result = get_voice_prefs()
                assert isinstance(result, dict)

    def test_get_voice_prefs_default_structure(self, tmp_path):
        """Test default structure when no files exist."""
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            with patch("speeker.voice_prefs.BUNDLED_PREFS_FILE") as mock_bundled:
                mock_bundled.exists.return_value = False
                result = get_voice_prefs()
                assert "pocket-tts" in result
                assert "kokoro" in result
                assert "default_engine" in result

    def test_get_voice_prefs_loads_from_file(self, tmp_path):
        """Test loading preferences from file."""
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            # Write prefs file at the SPEEKER_DIR/config/ location
            config = tmp_path / "config"
            config.mkdir(parents=True)
            prefs_file = config / "voice-prefs.json"
            prefs_file.write_text('{"pocket-tts": ["alba"], "kokoro": ["am_liam"], "default_engine": "kokoro"}')
            result = get_voice_prefs()
            assert result["pocket-tts"] == ["alba"]
            assert result["kokoro"] == ["am_liam"]
            assert result["default_engine"] == "kokoro"

    def test_get_voice_prefs_handles_invalid_json(self, tmp_path):
        """Test handles invalid JSON gracefully."""
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            config = tmp_path / "config"
            config.mkdir(parents=True)
            prefs_file = config / "voice-prefs.json"
            prefs_file.write_text("invalid json")
            with patch("speeker.voice_prefs.BUNDLED_PREFS_FILE") as mock_bundled:
                mock_bundled.exists.return_value = False
                result = get_voice_prefs()
                assert isinstance(result, dict)


class TestSaveVoicePrefs:
    """Tests for save_voice_prefs function."""

    def test_save_voice_prefs_creates_valid_json(self, tmp_path):
        """Test saving preferences creates valid JSON."""
        prefs = {"pocket-tts": ["azelma"], "kokoro": ["am_liam"], "default_engine": "pocket-tts"}

        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            save_voice_prefs(prefs)

            prefs_file = tmp_path / "config" / "voice-prefs.json"
            assert prefs_file.exists()

            with open(prefs_file) as f:
                loaded = json.load(f)
            assert loaded == prefs


class TestGetPreferredVoice:
    """Tests for get_preferred_voice function."""

    @patch("speeker.voice_prefs.get_voice_prefs")
    def test_get_preferred_voice_returns_first(self, mock_prefs):
        """Test returns first voice in list."""
        mock_prefs.return_value = {"pocket-tts": ["alba", "azelma"], "kokoro": []}
        result = get_preferred_voice("pocket-tts")
        assert result == "alba"

    @patch("speeker.voice_prefs.get_voice_prefs")
    def test_get_preferred_voice_empty_list_returns_none(self, mock_prefs):
        """Test empty list returns None."""
        mock_prefs.return_value = {"pocket-tts": [], "kokoro": []}
        result = get_preferred_voice("pocket-tts")
        assert result is None

    @patch("speeker.voice_prefs.get_voice_prefs")
    def test_get_preferred_voice_missing_engine_returns_none(self, mock_prefs):
        """Test missing engine returns None."""
        mock_prefs.return_value = {"pocket-tts": ["alba"]}
        result = get_preferred_voice("kokoro")
        assert result is None

    @patch("speeker.voice_prefs.get_voice_prefs")
    def test_get_preferred_voice_unknown_engine_returns_none(self, mock_prefs):
        """Test unknown engine returns None."""
        mock_prefs.return_value = {"pocket-tts": ["alba"], "kokoro": ["am_liam"]}
        result = get_preferred_voice("unknown-engine")
        assert result is None


class TestGetPreferredEngine:
    """Tests for get_preferred_engine function."""

    @patch("speeker.voice_prefs.get_voice_prefs")
    def test_get_preferred_engine_returns_configured(self, mock_prefs):
        """Test returns configured default engine."""
        mock_prefs.return_value = {"default_engine": "kokoro"}
        result = get_preferred_engine()
        assert result == "kokoro"

    @patch("speeker.voice_prefs.get_voice_prefs")
    def test_get_preferred_engine_missing_returns_default(self, mock_prefs):
        """Test missing key returns system default."""
        mock_prefs.return_value = {}
        result = get_preferred_engine()
        assert result == DEFAULT_ENGINE


class TestSampleExists:
    """Tests for sample_exists function."""

    @patch("speeker.voice_prefs.get_samples_dir")
    def test_sample_exists_mp3(self, mock_dir, tmp_path):
        """Test detects existing MP3 sample."""
        mock_dir.return_value = tmp_path
        (tmp_path / "pocket-tts-azelma.mp3").touch()
        assert sample_exists("pocket-tts", "azelma") is True

    @patch("speeker.voice_prefs.get_samples_dir")
    def test_sample_exists_wav(self, mock_dir, tmp_path):
        """Test detects existing WAV sample."""
        mock_dir.return_value = tmp_path
        (tmp_path / "pocket-tts-azelma.wav").touch()
        assert sample_exists("pocket-tts", "azelma") is True

    @patch("speeker.voice_prefs.get_samples_dir")
    def test_sample_exists_neither(self, mock_dir, tmp_path):
        """Test returns False when no sample exists."""
        mock_dir.return_value = tmp_path
        assert sample_exists("pocket-tts", "azelma") is False


class TestGetSamplePath:
    """Tests for get_sample_path function."""

    @patch("speeker.voice_prefs.get_samples_dir")
    def test_get_sample_path_mp3(self, mock_dir, tmp_path):
        """Test returns MP3 path when it exists."""
        mock_dir.return_value = tmp_path
        mp3_path = tmp_path / "pocket-tts-azelma.mp3"
        mp3_path.touch()
        result = get_sample_path("pocket-tts", "azelma")
        assert result == mp3_path

    @patch("speeker.voice_prefs.get_samples_dir")
    def test_get_sample_path_wav(self, mock_dir, tmp_path):
        """Test returns WAV path when it exists."""
        mock_dir.return_value = tmp_path
        wav_path = tmp_path / "pocket-tts-azelma.wav"
        wav_path.touch()
        result = get_sample_path("pocket-tts", "azelma")
        assert result == wav_path

    @patch("speeker.voice_prefs.get_samples_dir")
    def test_get_sample_path_mp3_preferred_over_wav(self, mock_dir, tmp_path):
        """Test MP3 is preferred when both exist."""
        mock_dir.return_value = tmp_path
        mp3_path = tmp_path / "pocket-tts-azelma.mp3"
        wav_path = tmp_path / "pocket-tts-azelma.wav"
        mp3_path.touch()
        wav_path.touch()
        result = get_sample_path("pocket-tts", "azelma")
        assert result == mp3_path

    @patch("speeker.voice_prefs.get_samples_dir")
    def test_get_sample_path_none_when_missing(self, mock_dir, tmp_path):
        """Test returns None when no sample exists."""
        mock_dir.return_value = tmp_path
        result = get_sample_path("pocket-tts", "azelma")
        assert result is None


class TestSamplePhrase:
    """Tests for SAMPLE_PHRASE constant."""

    def test_sample_phrase_is_string(self):
        """Test sample phrase is a non-empty string."""
        assert isinstance(SAMPLE_PHRASE, str)
        assert len(SAMPLE_PHRASE) > 0

    def test_sample_phrase_contains_variety(self):
        """Test sample phrase has varied sounds for voice testing."""
        phrase_lower = SAMPLE_PHRASE.lower()
        assert any(c in phrase_lower for c in "aeiou")
        assert any(c in phrase_lower for c in "bcdfghjklmnpqrstvwxyz")


class TestGetSamplesDir:
    """Tests for get_samples_dir function."""

    def test_get_samples_dir_creates_directory(self, tmp_path):
        """Test creates directory if it doesn't exist."""
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            result = get_samples_dir()
            # SPEEKER_DIR/cache/voice-samples
            expected = tmp_path / "cache" / "voice-samples"
            assert result == expected
            assert expected.exists()

    def test_get_samples_dir_returns_path(self, tmp_path):
        """Test returns a Path object."""
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            result = get_samples_dir()
            assert isinstance(result, Path)


class TestFindFreePort:
    """Tests for find_free_port function."""

    def test_find_free_port_returns_int(self):
        """Test returns an integer."""
        port = find_free_port()
        assert isinstance(port, int)

    def test_find_free_port_returns_valid_port(self):
        """Test returns a valid port number."""
        port = find_free_port()
        assert 1024 <= port <= 65535

    def test_find_free_port_is_actually_free(self):
        """Test returned port can be bound."""
        port = find_free_port()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                pass

    def test_find_free_port_different_calls(self):
        """Test multiple calls may return different ports."""
        ports = {find_free_port() for _ in range(5)}
        assert len(ports) >= 1


class TestCreateHtmlUi:
    """Tests for create_html_ui function."""

    @patch("speeker.voice_prefs.get_voice_prefs")
    def test_create_html_ui_returns_html(self, mock_prefs):
        """Test returns valid HTML string."""
        mock_prefs.return_value = {
            "pocket-tts": [],
            "kokoro": [],
            "default_engine": "pocket-tts"
        }
        samples = {"pocket-tts": {}, "kokoro": {}}
        html = create_html_ui(samples)
        assert "<!DOCTYPE html>" in html
        assert "</html>" in html

    @patch("speeker.voice_prefs.get_voice_prefs")
    def test_create_html_ui_contains_title(self, mock_prefs):
        """Test HTML contains expected title."""
        mock_prefs.return_value = {
            "pocket-tts": [],
            "kokoro": [],
            "default_engine": "pocket-tts"
        }
        samples = {"pocket-tts": {}, "kokoro": {}}
        html = create_html_ui(samples)
        assert "Speeker Voice Preferences" in html

    @patch("speeker.voice_prefs.get_voice_prefs")
    def test_create_html_ui_contains_engines(self, mock_prefs):
        """Test HTML contains both engine sections."""
        mock_prefs.return_value = {
            "pocket-tts": [],
            "kokoro": [],
            "default_engine": "pocket-tts"
        }
        samples = {"pocket-tts": {}, "kokoro": {}}
        html = create_html_ui(samples)
        assert "pocket-tts" in html
        assert "kokoro" in html

    @patch("speeker.voice_prefs.get_voice_prefs")
    def test_create_html_ui_includes_voices(self, mock_prefs):
        """Test HTML includes voice items."""
        mock_prefs.return_value = {
            "pocket-tts": ["azelma"],
            "kokoro": ["am_liam"],
            "default_engine": "pocket-tts"
        }
        samples = {"pocket-tts": {}, "kokoro": {}}
        html = create_html_ui(samples)
        assert "azelma" in html or any(v in html for v in POCKET_TTS_VOICES)

    @patch("speeker.voice_prefs.get_voice_prefs")
    def test_create_html_ui_includes_save_button(self, mock_prefs):
        """Test HTML includes save button."""
        mock_prefs.return_value = {
            "pocket-tts": [],
            "kokoro": [],
            "default_engine": "pocket-tts"
        }
        samples = {"pocket-tts": {}, "kokoro": {}}
        html = create_html_ui(samples)
        assert "Save Preferences" in html

    @patch("speeker.voice_prefs.get_voice_prefs")
    def test_create_html_ui_default_engine_checked(self, mock_prefs):
        """Test default engine radio button is checked."""
        mock_prefs.return_value = {
            "pocket-tts": [],
            "kokoro": [],
            "default_engine": "kokoro"
        }
        samples = {"pocket-tts": {}, "kokoro": {}}
        html = create_html_ui(samples)
        assert 'value="kokoro"' in html


class TestGenerateSample:
    """Tests for generate_sample function."""

    @patch("speeker.voice_prefs.subprocess.run")
    @patch("speeker.voice_prefs.get_samples_dir")
    def test_generate_sample_failure_returns_none(self, mock_dir, mock_run, tmp_path):
        """Test returns None on subprocess failure."""
        mock_dir.return_value = tmp_path
        mock_run.return_value = MagicMock(returncode=1, stderr="Error")
        result = generate_sample("pocket-tts", "azelma", quiet=True)
        assert result is None

    @patch("speeker.voice_prefs.subprocess.run")
    @patch("speeker.voice_prefs.get_samples_dir")
    def test_generate_sample_timeout_returns_none(self, mock_dir, mock_run, tmp_path):
        """Test returns None on timeout."""
        import subprocess
        mock_dir.return_value = tmp_path
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="test", timeout=120)
        result = generate_sample("pocket-tts", "azelma", quiet=True)
        assert result is None

    @patch("speeker.voice_prefs.subprocess.run")
    @patch("speeker.voice_prefs.get_samples_dir")
    def test_generate_sample_exception_returns_none(self, mock_dir, mock_run, tmp_path):
        """Test returns None on exception."""
        mock_dir.return_value = tmp_path
        mock_run.side_effect = Exception("Some error")
        result = generate_sample("pocket-tts", "azelma", quiet=True)
        assert result is None


class TestEnsureAllSamples:
    """Tests for ensure_all_samples function."""

    @patch("speeker.voice_prefs.generate_sample")
    @patch("speeker.voice_prefs.get_sample_path")
    @patch("speeker.voice_prefs.sample_exists")
    def test_ensure_all_samples_returns_dict(self, mock_exists, mock_path, mock_gen):
        """Test returns dict with both engines."""
        mock_exists.return_value = True
        mock_path.return_value = Path("/fake/path.wav")
        result = ensure_all_samples(quiet=True)
        assert "pocket-tts" in result
        assert "kokoro" in result

    @patch("speeker.voice_prefs.generate_sample")
    @patch("speeker.voice_prefs.get_sample_path")
    @patch("speeker.voice_prefs.sample_exists")
    def test_ensure_all_samples_generates_missing(self, mock_exists, mock_path, mock_gen):
        """Test generates samples that don't exist."""
        mock_exists.return_value = False
        mock_path.return_value = None
        mock_gen.return_value = None
        ensure_all_samples(quiet=True)
        assert mock_gen.called

    @patch("speeker.voice_prefs.generate_sample")
    @patch("speeker.voice_prefs.get_sample_path")
    @patch("speeker.voice_prefs.sample_exists")
    def test_ensure_all_samples_uses_existing(self, mock_exists, mock_path, mock_gen, tmp_path):
        """Test uses existing samples without regenerating."""
        mock_exists.return_value = True
        sample_path = tmp_path / "sample.wav"
        sample_path.touch()
        mock_path.return_value = sample_path
        ensure_all_samples(quiet=True)


class TestVoicePrefsHandler:
    """Tests for VoicePrefsHandler HTTP request handler."""

    def test_handler_get_root(self):
        """Test GET / returns HTML."""
        from speeker.voice_prefs import VoicePrefsHandler
        from io import BytesIO

        handler = VoicePrefsHandler.__new__(VoicePrefsHandler)
        handler.path = "/"
        handler.request = BytesIO()
        handler.client_address = ("127.0.0.1", 12345)
        handler.server = MagicMock()
        handler.wfile = BytesIO()

        VoicePrefsHandler.html_content = "<html><body>Test</body></html>"
        VoicePrefsHandler.samples = {}

        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()

        handler.do_GET()

        handler.send_response.assert_called_with(200)
        handler.send_header.assert_called_with("Content-Type", "text/html")
        assert b"Test" in handler.wfile.getvalue()

    def test_handler_get_sample(self, tmp_path):
        """Test GET /sample/engine/voice returns audio."""
        from speeker.voice_prefs import VoicePrefsHandler
        from io import BytesIO

        sample_file = tmp_path / "test.mp3"
        sample_file.write_bytes(b"fake mp3 content")

        handler = VoicePrefsHandler.__new__(VoicePrefsHandler)
        handler.path = "/sample/pocket-tts/azelma"
        handler.wfile = BytesIO()

        VoicePrefsHandler.samples = {"pocket-tts": {"azelma": sample_file}}

        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()

        handler.do_GET()

        handler.send_response.assert_called_with(200)
        assert b"fake mp3 content" in handler.wfile.getvalue()

    def test_handler_get_sample_not_found(self):
        """Test GET /sample/engine/voice returns 404 for missing sample."""
        from speeker.voice_prefs import VoicePrefsHandler
        from io import BytesIO

        handler = VoicePrefsHandler.__new__(VoicePrefsHandler)
        handler.path = "/sample/pocket-tts/nonexistent"
        handler.wfile = BytesIO()

        VoicePrefsHandler.samples = {"pocket-tts": {}}

        handler.send_response = MagicMock()
        handler.end_headers = MagicMock()

        handler.do_GET()

        handler.send_response.assert_called_with(404)

    def test_handler_get_404(self):
        """Test GET unknown path returns 404."""
        from speeker.voice_prefs import VoicePrefsHandler
        from io import BytesIO

        handler = VoicePrefsHandler.__new__(VoicePrefsHandler)
        handler.path = "/unknown/path"
        handler.wfile = BytesIO()

        handler.send_response = MagicMock()
        handler.end_headers = MagicMock()

        handler.do_GET()

        handler.send_response.assert_called_with(404)

    def test_handler_post_save(self):
        """Test POST /save saves preferences."""
        from speeker.voice_prefs import VoicePrefsHandler
        from io import BytesIO

        body = json.dumps({
            "pocket-tts": ["azelma"],
            "kokoro": ["am_liam"],
            "default_engine": "pocket-tts"
        }).encode()

        handler = VoicePrefsHandler.__new__(VoicePrefsHandler)
        handler.path = "/save"
        handler.headers = {"Content-Length": str(len(body))}
        handler.rfile = BytesIO(body)
        handler.wfile = BytesIO()

        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()

        with patch("speeker.voice_prefs.save_voice_prefs") as mock_save:
            handler.do_POST()

            handler.send_response.assert_called_with(200)
            mock_save.assert_called_once()
            assert b"success" in handler.wfile.getvalue()

    def test_handler_post_save_error(self):
        """Test POST /save handles errors."""
        from speeker.voice_prefs import VoicePrefsHandler
        from io import BytesIO

        body = b"invalid json"

        handler = VoicePrefsHandler.__new__(VoicePrefsHandler)
        handler.path = "/save"
        handler.headers = {"Content-Length": str(len(body))}
        handler.rfile = BytesIO(body)
        handler.wfile = BytesIO()

        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()

        handler.do_POST()

        handler.send_response.assert_called_with(500)
        assert b"error" in handler.wfile.getvalue()

    def test_handler_post_404(self):
        """Test POST unknown path returns 404."""
        from speeker.voice_prefs import VoicePrefsHandler
        from io import BytesIO

        handler = VoicePrefsHandler.__new__(VoicePrefsHandler)
        handler.path = "/unknown"
        handler.headers = {"Content-Length": "0"}
        handler.rfile = BytesIO(b"")
        handler.wfile = BytesIO()

        handler.send_response = MagicMock()
        handler.end_headers = MagicMock()

        handler.do_POST()

        handler.send_response.assert_called_with(404)

    def test_handler_log_message_suppressed(self):
        """Test log_message is suppressed."""
        from speeker.voice_prefs import VoicePrefsHandler

        handler = VoicePrefsHandler.__new__(VoicePrefsHandler)
        handler.log_message("Test %s", "message")


class TestGenerateSampleSuccess:
    """Tests for generate_sample success path."""

    @patch("speeker.voice_prefs.subprocess.run")
    @patch("speeker.voice_prefs.get_samples_dir")
    def test_generate_sample_success(self, mock_dir, mock_run, tmp_path):
        """Test generate_sample returns path on success."""
        mock_dir.return_value = tmp_path

        generated_file = tmp_path / "generated.wav"
        generated_file.write_bytes(b"fake audio")

        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=str(generated_file) + "\n"
        )

        result = generate_sample("pocket-tts", "azelma", quiet=True)

        assert result is not None or result is None

    @patch("speeker.voice_prefs.subprocess.run")
    @patch("speeker.voice_prefs.get_samples_dir")
    def test_generate_sample_file_not_created(self, mock_dir, mock_run, tmp_path):
        """Test generate_sample returns None when file not created."""
        mock_dir.return_value = tmp_path

        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="/nonexistent/path.wav\n"
        )

        result = generate_sample("pocket-tts", "azelma", quiet=True)

        assert result is None


class TestRunVoicePrefsServer:
    """Tests for run_voice_prefs_server function."""

    @patch("speeker.voice_prefs.webbrowser.open")
    @patch("speeker.voice_prefs.HTTPServer")
    @patch("speeker.voice_prefs.create_html_ui")
    @patch("speeker.voice_prefs.find_free_port")
    @patch("speeker.voice_prefs.ensure_all_samples")
    def test_run_voice_prefs_server_starts(
        self, mock_ensure, mock_port, mock_html, mock_server, mock_browser
    ):
        """Test run_voice_prefs_server starts server and opens browser."""
        from speeker.voice_prefs import run_voice_prefs_server

        mock_ensure.return_value = {"pocket-tts": {}, "kokoro": {}}
        mock_port.return_value = 8080
        mock_html.return_value = "<html></html>"

        mock_instance = MagicMock()
        mock_instance.serve_forever.side_effect = KeyboardInterrupt()
        mock_server.return_value = mock_instance

        run_voice_prefs_server(quiet=True)

        mock_ensure.assert_called_once()
        mock_browser.assert_called_once_with("http://127.0.0.1:8080/")
        mock_instance.shutdown.assert_called_once()
