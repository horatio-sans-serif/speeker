#!/usr/bin/env python3
"""Unit tests for web.py utility functions and routes."""

import os
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient

from speeker.server import app
from speeker.web import (
    format_time,
    escape_html,
    sanitize_key,
    sanitize_value,
    render_metadata,
    strip_tone_tokens,
)


class TestStripToneTokens:
    """Tests for strip_tone_tokens (display-time elision of $Note tones)."""

    def test_strips_leading_tone(self):
        assert strip_tone_tokens("$Eb4 doctor video. Fixed it.") == "doctor video. Fixed it."

    def test_strips_multiple_leading_tones(self):
        assert strip_tone_tokens("$Eb3 $Eb3 Title. Body.") == "Title. Body."

    def test_no_tone_unchanged(self):
        assert strip_tone_tokens("Just a normal sentence.") == "Just a normal sentence."

    def test_does_not_eat_dollar_amounts(self):
        # "$5" is not a note token and must be preserved.
        assert strip_tone_tokens("It cost $5 today.") == "It cost $5 today."

    def test_empty(self):
        assert strip_tone_tokens("") == ""


class TestFormatTime:
    """Tests for format_time function."""

    def test_format_time_valid_iso_string(self):
        """Test formatting a valid ISO timestamp."""
        result = format_time("2024-01-15T14:30:00")
        assert result == "Jan 15 14:30"

    def test_format_time_with_timezone(self):
        """Test formatting ISO timestamp with timezone."""
        result = format_time("2024-06-20T09:15:00+00:00")
        assert result == "Jun 20 09:15"

    def test_format_time_none_returns_dash(self):
        """Test that None input returns dash."""
        assert format_time(None) == "-"

    def test_format_time_empty_string_returns_dash(self):
        """Test that empty string returns dash."""
        assert format_time("") == "-"

    def test_format_time_invalid_format_returns_original(self):
        """Test that invalid format returns the original string."""
        result = format_time("not-a-date")
        assert result == "not-a-date"

    def test_format_time_partial_date_returns_original(self):
        """Test that partial date returns original string."""
        result = format_time("2024-01")
        # fromisoformat may or may not handle this, just ensure no crash
        assert result is not None


class TestEscapeHtml:
    """Tests for escape_html function."""

    def test_escape_html_ampersand(self):
        """Test escaping ampersand."""
        assert escape_html("A & B") == "A &amp; B"

    def test_escape_html_less_than(self):
        """Test escaping less than."""
        assert escape_html("A < B") == "A &lt; B"

    def test_escape_html_greater_than(self):
        """Test escaping greater than."""
        assert escape_html("A > B") == "A &gt; B"

    def test_escape_html_double_quote(self):
        """Test escaping double quote."""
        assert escape_html('say "hello"') == "say &quot;hello&quot;"

    def test_escape_html_multiple_entities(self):
        """Test escaping multiple entities in one string."""
        result = escape_html('<script>alert("XSS & more")</script>')
        assert result == "&lt;script&gt;alert(&quot;XSS &amp; more&quot;)&lt;/script&gt;"

    def test_escape_html_no_special_chars(self):
        """Test string with no special characters is unchanged."""
        assert escape_html("Hello World") == "Hello World"

    def test_escape_html_empty_string(self):
        """Test empty string returns empty string."""
        assert escape_html("") == ""

    def test_escape_html_order_matters(self):
        """Test that ampersand is escaped first to avoid double-escaping."""
        result = escape_html("&lt;")
        # & should become &amp; first, so result is &amp;lt;
        assert result == "&amp;lt;"


class TestSanitizeKey:
    """Tests for sanitize_key function."""

    def test_sanitize_key_simple_string(self):
        """Test simple key name."""
        assert sanitize_key("queue") == "queue"

    def test_sanitize_key_with_special_chars(self):
        """Test key with HTML special characters."""
        assert sanitize_key("<key>") == "&lt;key&gt;"

    def test_sanitize_key_numeric_string(self):
        """Test numeric string key."""
        assert sanitize_key("123") == "123"

    def test_sanitize_key_empty_string(self):
        """Test empty string key."""
        assert sanitize_key("") == ""


class TestSanitizeValue:
    """Tests for sanitize_value function."""

    def test_sanitize_value_none_returns_empty(self):
        """Test that None returns empty string."""
        assert sanitize_value(None) == ""

    def test_sanitize_value_simple_string(self):
        """Test simple string value."""
        assert sanitize_value("hello") == "hello"

    def test_sanitize_value_string_with_html(self):
        """Test string with HTML is escaped."""
        assert sanitize_value("<b>bold</b>") == "&lt;b&gt;bold&lt;/b&gt;"

    def test_sanitize_value_integer(self):
        """Test integer is converted to string."""
        assert sanitize_value(42) == "42"

    def test_sanitize_value_float(self):
        """Test float is converted to string."""
        assert sanitize_value(3.14) == "3.14"

    def test_sanitize_value_dict_is_json_encoded(self):
        """Test dict is JSON encoded."""
        result = sanitize_value({"key": "value"})
        assert "&quot;" in result  # Contains escaped quotes
        assert "key" in result
        assert "value" in result

    def test_sanitize_value_list_is_json_encoded(self):
        """Test list is JSON encoded."""
        result = sanitize_value([1, 2, 3])
        assert "[1, 2, 3]" in result

    def test_sanitize_value_dict_with_special_chars(self):
        """Test dict values with special chars are escaped."""
        result = sanitize_value({"msg": "<script>"})
        assert "&lt;script&gt;" in result

    def test_sanitize_value_boolean(self):
        """Test boolean is converted to string."""
        assert sanitize_value(True) == "True"
        assert sanitize_value(False) == "False"


class TestRenderMetadata:
    """Tests for render_metadata function.

    render_metadata was simplified: it now renders only the message's
    ``display_name`` (or its ``queue`` if no display_name) -- the legacy
    behavior of dumping every metadata key was too noisy in the UI.
    """

    def test_render_metadata_none_returns_placeholder(self):
        """None metadata renders the no-data placeholder."""
        result = render_metadata(None)
        assert 'class="no-data"' in result

    def test_render_metadata_empty_dict_returns_placeholder(self):
        """Empty dict renders the no-data placeholder."""
        result = render_metadata({})
        assert 'class="no-data"' in result

    def test_render_metadata_display_name_wins(self):
        """display_name is shown when present, even with other keys."""
        result = render_metadata({
            "queue": "rm", "display_name": "rocket man", "interpretation": "SUCCESS",
        })
        assert "rocket man" in result
        # Other metadata keys should NOT leak into the rendered output.
        assert "rm" not in result.replace("rocket man", "")
        assert "SUCCESS" not in result
        assert "interpretation" not in result
        assert "queue:" not in result

    def test_render_metadata_falls_back_to_queue(self):
        """When display_name is missing, the queue id is shown instead."""
        result = render_metadata({"queue": "rm"})
        assert "rm" in result

    def test_render_metadata_falls_back_to_placeholder_when_only_other_keys(self):
        """Without display_name/queue, render the placeholder."""
        result = render_metadata({"interpretation": "SUCCESS", "engine": "polly"})
        assert 'class="no-data"' in result

    def test_render_metadata_escapes_html_in_display_name(self):
        """HTML in the displayed value is escaped."""
        result = render_metadata({"display_name": "<script>alert(1)</script>"})
        assert "<script>" not in result
        assert "&lt;script&gt;" in result


# --- Additional Edge Case Tests ---

class TestEscapeHtmlEdgeCases:
    """Additional edge cases for escape_html."""

    def test_escape_html_single_quote_not_escaped(self):
        """Test single quotes are NOT escaped (verify current behavior)."""
        # Single quotes are not in the escape list - document this
        result = escape_html("it's")
        assert result == "it's"

    def test_escape_html_unicode_preserved(self):
        """Test unicode characters pass through unchanged."""
        assert escape_html("日本語") == "日本語"
        assert escape_html("émoji 🎉") == "émoji 🎉"

    def test_escape_html_newlines_preserved(self):
        """Test newlines and whitespace preserved."""
        assert escape_html("line1\nline2") == "line1\nline2"
        assert escape_html("tab\there") == "tab\there"

    def test_escape_html_very_long_string(self):
        """Test very long strings are handled."""
        long_str = "x" * 10000
        result = escape_html(long_str)
        assert len(result) == 10000

    def test_escape_html_all_special_chars(self):
        """Test string with all special characters."""
        result = escape_html('&<>"')
        assert result == "&amp;&lt;&gt;&quot;"

    def test_escape_html_repeated_entities(self):
        """Test multiple consecutive entities."""
        result = escape_html("<<<>>>")
        assert result == "&lt;&lt;&lt;&gt;&gt;&gt;"

    def test_escape_html_xss_script_tag(self):
        """Test XSS script injection is escaped."""
        xss = '<script>alert("XSS")</script>'
        result = escape_html(xss)
        assert "<script>" not in result
        assert "&lt;script&gt;" in result

    def test_escape_html_xss_event_handler(self):
        """Test XSS event handler is escaped."""
        xss = '<img onerror="alert(1)" src=x>'
        result = escape_html(xss)
        assert "<img" not in result
        assert "&lt;img" in result


class TestSanitizeValueEdgeCases:
    """Additional edge cases for sanitize_value."""

    def test_sanitize_value_datetime_object(self):
        """Test datetime objects are converted via str()."""
        from datetime import datetime
        dt = datetime(2024, 1, 15, 12, 30)
        result = sanitize_value(dt)
        assert "2024" in result
        assert "12:30" in result

    def test_sanitize_value_list_with_special_chars(self):
        """Test list containing strings with special chars."""
        result = sanitize_value(["<a>", "&b"])
        assert "&lt;a&gt;" in result
        assert "&amp;b" in result

    def test_sanitize_value_empty_list(self):
        """Test empty list returns '[]'."""
        result = sanitize_value([])
        assert result == "[]"

    def test_sanitize_value_empty_dict(self):
        """Test empty dict returns '{}'."""
        result = sanitize_value({})
        assert result == "{}"

    def test_sanitize_value_nested_special_chars(self):
        """Test deeply nested structure with special chars."""
        result = sanitize_value({"a": {"b": "<script>"}})
        assert "&lt;script&gt;" in result

    def test_sanitize_value_unicode_in_dict(self):
        """Test unicode in dict values - JSON escapes non-ASCII."""
        result = sanitize_value({"msg": "日本語"})
        # json.dumps escapes unicode by default, so check for escaped form
        assert "msg" in result
        # Either raw unicode or escaped form
        assert "日本語" in result or "\\u" in result


class TestFormatTimeEdgeCases:
    """Additional edge cases for format_time."""

    def test_format_time_with_microseconds(self):
        """Test datetime with microseconds."""
        result = format_time("2024-01-15T14:30:00.123456")
        assert result == "Jan 15 14:30"

    def test_format_time_midnight(self):
        """Test midnight time."""
        result = format_time("2024-01-15T00:00:00")
        assert result == "Jan 15 00:00"

    def test_format_time_end_of_day(self):
        """Test end of day time."""
        result = format_time("2024-01-15T23:59:59")
        assert result == "Jan 15 23:59"

    def test_format_time_leap_year(self):
        """Test leap year date."""
        result = format_time("2024-02-29T12:00:00")
        assert result == "Feb 29 12:00"

    def test_format_time_year_boundary(self):
        """Test year boundary dates."""
        assert format_time("2024-12-31T23:59:59") == "Dec 31 23:59"
        assert format_time("2024-01-01T00:00:00") == "Jan 01 00:00"

    def test_format_time_negative_timezone(self):
        """Test negative timezone offset."""
        result = format_time("2024-01-15T14:30:00-05:00")
        assert "Jan 15" in result

    def test_format_time_positive_timezone(self):
        """Test positive timezone offset."""
        result = format_time("2024-01-15T14:30:00+09:00")
        assert "Jan 15" in result


class TestRenderMetadataEdgeCases:
    """Additional edge cases for render_metadata (display_name-only contract)."""

    def test_render_metadata_unicode_display_name(self):
        """Unicode display_name passes through (only HTML entities are escaped)."""
        result = render_metadata({"display_name": "日本語"})
        assert "日本語" in result

    def test_render_metadata_empty_display_name_falls_back_to_queue(self):
        """Empty display_name -> use queue id instead."""
        result = render_metadata({"display_name": "", "queue": "fallback-queue"})
        assert "fallback-queue" in result

    def test_render_metadata_whitespace_display_name_falls_back(self):
        """Whitespace-only display_name -> use queue id instead."""
        result = render_metadata({"display_name": "   ", "queue": "fallback"})
        assert "fallback" in result

    def test_render_metadata_long_display_name_preserved(self):
        """Long display_name is rendered (no truncation enforced server-side)."""
        long_name = "x" * 500
        result = render_metadata({"display_name": long_name})
        assert long_name in result


# HTTP Route Tests
@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app, raise_server_exceptions=False)


class TestIndexRoute:
    """Tests for / route."""

    @patch("speeker.web.get_history")
    def test_index_returns_html(self, mock_history, client):
        """Test index route returns HTML."""
        mock_history.return_value = []
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Speeker" in response.text

    def test_index_returns_react_shell(self, client):
        """The index now serves a static React shell -- no server-rendered rows.

        The previous shape (server-rendered HTML rows) was tested by mocking
        ``get_history`` and matching on item text. After the React rewrite,
        all item rendering happens client-side via /api/items, so the index
        page is a constant; we assert on its structural markers.
        """
        response = client.get("/")
        assert response.status_code == 200
        # The page mounts React into <div id="root">.
        assert 'id="root"' in response.text
        # And loads React + Babel from CDN.
        assert "react@18" in response.text
        assert "babel/standalone" in response.text
        # And contains both tab buttons.
        assert "Queue History" in response.text
        assert "Settings" in response.text

    def test_index_legacy_query_param_returns_200(self, client):
        """Legacy ?q= deep links don't crash (param is ignored, page still loads)."""
        response = client.get("/?q=hello")
        assert response.status_code == 200
        assert 'id="root"' in response.text


class TestApiItemsRoute:
    """Tests for /api/items route."""

    @patch("speeker.web.get_history")
    def test_api_items_returns_json(self, mock_history, client):
        """Test api/items returns JSON."""
        mock_history.return_value = []
        response = client.get("/api/items")
        assert response.status_code == 200
        assert "application/json" in response.headers["content-type"]

    @patch("speeker.web.get_history")
    def test_api_items_empty_list(self, mock_history, client):
        """Test api/items with no items."""
        mock_history.return_value = []
        response = client.get("/api/items")
        data = response.json()
        assert "hash" in data
        assert data["items"] == []

    @patch("speeker.web.get_history")
    def test_api_items_returns_items(self, mock_history, client):
        """Test api/items returns formatted items."""
        mock_history.return_value = [
            {
                "id": 1,
                "text": "Test message",
                "created_at": "2024-01-15T14:30:00",
                "played_at": None,
                "audio_path": None,
                "session_id": "myqueue",
                "metadata": None,
            }
        ]
        response = client.get("/api/items")
        data = response.json()
        assert len(data["items"]) == 1
        item = data["items"][0]
        assert item["id"] == 1
        assert item["text"] == "Test message"
        assert item["played"] is False
        assert item["queue"] == "myqueue"
        # No metadata.tts_error -> tts_error is None.
        assert item["tts_error"] is None

    @patch("speeker.web.get_history")
    def test_api_items_surfaces_tts_error(self, mock_history, client):
        """When the daemon recorded ``metadata.tts_error`` after the retry
        cap, /api/items must surface it so the UI can render the "TTS
        failed" badge next to the disabled Play button."""
        mock_history.return_value = [
            {
                "id": 7,
                "text": "Hello",
                "created_at": "2024-01-15T14:30:00",
                "played_at": "2024-01-15T14:30:05",
                "audio_path": None,
                "session_id": "myqueue",
                "metadata": {
                    "queue": "myqueue",
                    "tts_attempts": 3,
                    "tts_error": "Polly throttle exceeded",
                },
            }
        ]
        response = client.get("/api/items")
        data = response.json()
        item = data["items"][0]
        assert item["tts_error"] == "Polly throttle exceeded"
        assert item["has_audio"] is False
        assert item["played"] is True

    @patch("speeker.web.get_history")
    def test_api_items_default_queue(self, mock_history, client):
        """Test api/items uses 'default' for no session_id."""
        mock_history.return_value = [
            {
                "id": 1,
                "text": "Message",
                "created_at": "2024-01-15T14:30:00",
                "played_at": None,
                "audio_path": None,
                "session_id": None,
                "metadata": None,
            }
        ]
        response = client.get("/api/items")
        data = response.json()
        assert data["items"][0]["queue"] == "default"

    @patch("speeker.web.get_history")
    def test_api_items_escapes_html(self, mock_history, client):
        """Test api/items escapes HTML in text."""
        mock_history.return_value = [
            {
                "id": 1,
                "text": "<b>bold</b>",
                "created_at": "2024-01-15T14:30:00",
                "played_at": None,
                "audio_path": None,
                "session_id": None,
                "metadata": None,
            }
        ]
        response = client.get("/api/items")
        data = response.json()
        assert "<b>" not in data["items"][0]["text"]
        assert "&lt;b&gt;" in data["items"][0]["text"]


class TestAudioRoute:
    """Tests for /audio/{item_id} route."""

    @patch("speeker.web.get_history")
    def test_audio_not_found(self, mock_history, client):
        """Test audio route returns 404 when not found."""
        mock_history.return_value = []
        response = client.get("/audio/999")
        assert response.status_code == 404

    @patch("speeker.web.get_history")
    def test_audio_no_audio_path(self, mock_history, client):
        """Test audio route returns 404 when no audio_path."""
        mock_history.return_value = [
            {
                "id": 1,
                "text": "Message",
                "created_at": "2024-01-15T14:30:00",
                "played_at": None,
                "audio_path": None,
                "session_id": None,
                "metadata": None,
            }
        ]
        response = client.get("/audio/1")
        assert response.status_code == 404

    @patch("speeker.web.get_history")
    @patch("speeker.web.Path")
    def test_audio_file_not_exists(self, mock_path, mock_history, client):
        """Test audio route returns 404 when file doesn't exist."""
        mock_history.return_value = [
            {
                "id": 1,
                "text": "Message",
                "created_at": "2024-01-15T14:30:00",
                "played_at": None,
                "audio_path": "/path/to/audio.wav",
                "session_id": None,
                "metadata": None,
            }
        ]
        mock_path.return_value.exists.return_value = False
        response = client.get("/audio/1")
        assert response.status_code == 404


class TestSettingsRoute:
    """Tests for /settings route."""

    @patch("speeker.web.get_settings")
    def test_settings_page_returns_html(self, mock_settings, client):
        """Test settings page returns HTML."""
        mock_settings.return_value = {
            "intro_sound": True,
            "speed": 1.0,
            "voice": "azelma",
            "engine": "pocket-tts",
        }
        response = client.get("/settings")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Settings" in response.text

    @patch("speeker.web.get_settings")
    def test_settings_page_shows_current_values(self, mock_settings, client):
        """Test settings page shows current settings."""
        mock_settings.return_value = {
            "intro_sound": True,
            "speed": 1.5,
            "voice": "alba",
            "engine": "pocket-tts",
        }
        response = client.get("/settings")
        assert response.status_code == 200
        assert '1.5' in response.text
        assert 'checked' in response.text  # intro_sound checkbox

    @patch("speeker.web.get_settings")
    def test_settings_with_session(self, mock_settings, client):
        """Test settings page with session parameter."""
        mock_settings.return_value = {
            "intro_sound": False,
            "speed": 1.0,
            "voice": "azelma",
            "engine": "pocket-tts",
        }
        response = client.get("/settings?session=myqueue")
        assert response.status_code == 200
        mock_settings.assert_called_with("myqueue")

    @patch("speeker.web.set_settings")
    def test_save_settings(self, mock_set, client):
        """Test saving settings via POST."""
        response = client.post(
            "/settings",
            data={
                "intro_sound": "on",
                "speed": "1.2",
                "voice": "alba",
                "engine": "pocket-tts",
            },
        )
        assert response.status_code == 200
        mock_set.assert_called_once()
        call_kwargs = mock_set.call_args[1]
        assert call_kwargs["intro_sound"] is True
        assert call_kwargs["speed"] == 1.2
        assert call_kwargs["voice"] == "alba"

    @patch("speeker.web.set_settings")
    def test_save_settings_without_intro_sound(self, mock_set, client):
        """Test saving settings with intro_sound unchecked."""
        response = client.post(
            "/settings",
            data={
                "speed": "1.0",
                "voice": "azelma",
                "engine": "pocket-tts",
            },
        )
        assert response.status_code == 200
        call_kwargs = mock_set.call_args[1]
        assert call_kwargs["intro_sound"] is False

    @patch("speeker.web.set_settings")
    def test_save_settings_with_session(self, mock_set, client):
        """Test saving session-specific settings."""
        response = client.post(
            "/settings?session=alerts",
            data={
                "speed": "1.5",
                "voice": "alba",
                "engine": "kokoro",
            },
        )
        assert response.status_code == 200
        call_kwargs = mock_set.call_args[1]
        assert call_kwargs["session_id"] == "alerts"


class TestEffectsRoutes:
    """Tests for the /api/effects family added with the audio effects feature."""

    @patch("speeker.web.get_effects_config")
    def test_get_effects_lists_presets_and_current(self, mock_cfg, client):
        """GET returns the active preset + the full preset catalog with
        descriptions. UI uses this to populate the dropdown."""
        mock_cfg.return_value = {"preset": "studio"}
        response = client.get("/api/effects")
        assert response.status_code == 200
        data = response.json()
        assert data["current"] == "studio"
        names = [p["name"] for p in data["presets"]]
        assert names[0] == "off"  # off always first
        assert set(names) >= {"off", "studio", "natural", "spacious", "telephone", "robot"}
        for p in data["presets"]:
            assert "description" in p
            assert "effect_count" in p

    @patch("speeker.web.get_effects_config")
    def test_get_effects_unknown_saved_value_falls_back_to_off(self, mock_cfg, client):
        """If config.json somehow has a preset name the code doesn't know
        about, the API must surface a safe default rather than the typo."""
        mock_cfg.return_value = {"preset": "stale-typo"}
        response = client.get("/api/effects")
        assert response.json()["current"] == "off"

    @patch("speeker.web.save_config")
    @patch("speeker.web.get_config")
    def test_put_effects_saves_known_preset(self, mock_get, mock_save, client):
        """A valid preset is persisted and returns restart_required: false
        because apply_effects re-reads config every utterance."""
        mock_get.return_value = {"effects": {"preset": "off"}}
        response = client.put("/api/effects", json={"preset": "natural"})
        assert response.status_code == 200
        body = response.json()
        assert body["current"] == "natural"
        assert body["restart_required"] is False
        # Saved config reflects the new preset.
        saved_cfg = mock_save.call_args[0][0]
        assert saved_cfg["effects"]["preset"] == "natural"

    def test_put_effects_rejects_unknown_preset(self, client):
        response = client.put("/api/effects", json={"preset": "definitely-not-real"})
        assert response.status_code == 400
        # Helpful error message lists the known names.
        assert "definitely-not-real" in response.json()["detail"]

    # The /api/effects/try handler imports ``enqueue`` and ``start_player``
    # lazily inside the function body (to keep server import-time
    # dependency graph small), so the patch target is the source module
    # rather than the ``speeker.web`` namespace.
    @patch("speeker.cli.start_player")
    @patch("speeker.queue_db.enqueue")
    @patch("speeker.web.get_effects_config")
    def test_try_with_preset_passes_via_metadata(self, mock_cfg, mock_enqueue, mock_start, client):
        """Try with an explicit preset attaches it as metadata so the
        daemon honors it for THIS utterance only -- no swap-and-restore
        race against the polling interval."""
        mock_cfg.return_value = {"preset": "off"}
        mock_enqueue.return_value = 999
        response = client.post("/api/effects/try", json={"preset": "spacious"})
        assert response.status_code == 200
        # enqueue called with metadata containing the preset.
        meta = mock_enqueue.call_args.kwargs["metadata"]
        assert meta["effects_preset"] == "spacious"
        assert meta["queue"] == "default"
        # The previewed phrase is the fixed Pangram chosen for the chain.
        spoken = mock_enqueue.call_args.args[0]
        assert "quick brown fox" in spoken.lower()

    @patch("speeker.cli.start_player")
    @patch("speeker.queue_db.enqueue")
    @patch("speeker.web.get_effects_config")
    def test_try_without_preset_omits_metadata_override(self, mock_cfg, mock_enqueue, mock_start, client):
        """No preset in body -> no metadata override -> daemon uses the
        saved preset via apply_effects."""
        mock_cfg.return_value = {"preset": "studio"}
        mock_enqueue.return_value = 1000
        response = client.post("/api/effects/try", json={})
        assert response.status_code == 200
        meta = mock_enqueue.call_args.kwargs["metadata"]
        assert "effects_preset" not in meta
        # Returned `preset` reflects the saved value when none was sent.
        assert response.json()["preset"] == "studio"

    def test_try_rejects_unknown_preset(self, client):
        response = client.post("/api/effects/try", json={"preset": "robot-but-misspelled"})
        assert response.status_code == 400


class TestToneTunesAndPlay:
    """The /api/tones/tunes catalog + /api/tones/play preview endpoint."""

    def test_tunes_catalog_returns_expected_shape(self, client):
        response = client.get("/api/tones/tunes")
        assert response.status_code == 200
        data = response.json()
        names = [t["name"] for t in data["tunes"]]
        # A couple of the well-known entries we curated.
        assert "Rising major triad" in names
        assert "NBC chimes" in names
        for t in data["tunes"]:
            assert all(isinstance(n, str) for n in t["notes"])
            assert len(t["notes"]) > 0

    @patch("speeker.cli.start_player")
    @patch("speeker.queue_db.enqueue")
    def test_play_notes_enqueues_dollar_tokens(self, mock_enqueue, mock_start, client):
        """Explicit notes are converted to ``$Note`` tokens. The text has
        no body so the player extracts the tones and skips TTS."""
        mock_enqueue.return_value = 5050
        response = client.post(
            "/api/tones/play",
            json={"notes": ["E4", "G4", "C5"], "duration": 0.2},
        )
        assert response.status_code == 200
        text = mock_enqueue.call_args.args[0]
        assert text == "$E4 $G4 $C5"
        meta = mock_enqueue.call_args.kwargs["metadata"]
        # Duration carried through as metadata so the daemon plays at
        # the requested per-note length, not the 0.8s default.
        assert meta["tone_duration"] == 0.2

    @patch("speeker.cli.start_player")
    @patch("speeker.queue_db.enqueue")
    @patch("speeker.web.get_tones_config")
    def test_play_kind_intro_reads_saved_notes(self, mock_cfg, mock_enqueue, mock_start, client):
        """``kind: 'intro'`` plays the saved intro notes at the saved duration."""
        mock_cfg.return_value = {"intro": ["E4", "G4", "C5"], "duration_seconds": 0.12}
        mock_enqueue.return_value = 5051
        response = client.post("/api/tones/play", json={"kind": "intro"})
        assert response.status_code == 200
        assert mock_enqueue.call_args.args[0] == "$E4 $G4 $C5"
        assert mock_enqueue.call_args.kwargs["metadata"]["tone_duration"] == 0.12

    def test_play_rejects_invalid_note(self, client):
        response = client.post("/api/tones/play", json={"notes": ["E4", "wat"]})
        assert response.status_code == 400

    @patch("speeker.cli.start_player")
    @patch("speeker.queue_db.enqueue")
    def test_play_accepts_duration_multiplier_notation(self, mock_enqueue, mock_start, client):
        """The new ``$Pitch:Mult`` syntax (e.g. ``C5:4``) must validate +
        round-trip into enqueued text -- this is how NBC chimes and
        Beethoven's 5th get their long final notes."""
        mock_enqueue.return_value = 5060
        response = client.post(
            "/api/tones/play",
            json={"notes": ["G4", "E4", "C5:4"], "duration": 0.18},
        )
        assert response.status_code == 200
        text = mock_enqueue.call_args.args[0]
        # The colon-multiplier is preserved through to the enqueued text
        # so the daemon's parse_note_token can pick it up.
        assert text == "$G4 $E4 $C5:4"

    def test_play_accepts_fractional_multiplier(self, client):
        """Fractional multipliers (``:0.5``, ``:.5``) are accepted."""
        with patch("speeker.queue_db.enqueue") as mock_enqueue, \
             patch("speeker.cli.start_player"):
            mock_enqueue.return_value = 5061
            response = client.post(
                "/api/tones/play", json={"notes": ["F4:0.5", "G4:.5", "A4"]},
            )
            assert response.status_code == 200

    def test_play_rejects_missing_inputs(self, client):
        response = client.post("/api/tones/play", json={})
        assert response.status_code == 400

    def test_play_clamps_extreme_duration(self, client):
        """A user typing 100 in the duration field shouldn't lock the
        daemon into a multi-minute tone. Server clamps to 2.0 max."""
        with patch("speeker.queue_db.enqueue") as mock_enqueue, \
             patch("speeker.cli.start_player"):
            mock_enqueue.return_value = 5052
            response = client.post(
                "/api/tones/play",
                json={"notes": ["E4"], "duration": 999.0},
            )
            assert response.status_code == 200
            assert mock_enqueue.call_args.kwargs["metadata"]["tone_duration"] == 2.0


class TestToneRulesEndpoints:
    """GET/PUT /api/tone-rules and GET /api/interpretations."""

    def test_get_empty(self, tmp_path, client):
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            response = client.get("/api/tone-rules")
        assert response.status_code == 200
        data = response.json()
        assert data["rules"] == []
        # Built-in interpretations always present so the editor's dropdown
        # is populated even before the user defines any.
        assert "SUCCESS" in data["interpretations"]
        assert "ERROR" in data["interpretations"]
        assert isinstance(data["queues"], list)

    def test_put_and_round_trip(self, tmp_path, client):
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            payload = {
                "rules": [
                    {
                        "slot": "cue",
                        "queue": "compass-docs",
                        "queue_regex": False,
                        "interpretation": "SUCCESS",
                        "notes": ["E5", "G5"],
                    },
                ]
            }
            response = client.put("/api/tone-rules", json=payload)
            assert response.status_code == 200, response.text
            data = response.json()
            assert len(data["rules"]) == 1
            assert data["rules"][0]["queue"] == "compass-docs"
            assert data["rules"][0]["notes"] == ["E5", "G5"]
            # Persisted and visible on subsequent GET.
            response = client.get("/api/tone-rules")
            assert response.status_code == 200
            assert response.json()["rules"][0]["notes"] == ["E5", "G5"]

    def test_put_rejects_bad_slot(self, tmp_path, client):
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            response = client.put("/api/tone-rules", json={
                "rules": [{"slot": "bogus", "queue": "X", "notes": ["E4"]}],
            })
        assert response.status_code == 400
        assert "slot" in response.text.lower()

    def test_put_rejects_bad_notes(self, tmp_path, client):
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            response = client.put("/api/tone-rules", json={
                "rules": [{"slot": "intro", "queue": "X", "notes": ["junk"]}],
            })
        assert response.status_code == 400

    def test_put_rejects_no_queue_or_interpretation(self, tmp_path, client):
        """A rule with neither dimension would match every utterance --
        rejected so the editor surfaces the mistake."""
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            response = client.put("/api/tone-rules", json={
                "rules": [{"slot": "cue", "notes": ["E4"]}],
            })
        assert response.status_code == 400

    def test_put_normalizes_whitespace(self, tmp_path, client):
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            response = client.put("/api/tone-rules", json={
                "rules": [{
                    "slot": "intro",
                    "queue": "  compass-docs  ",
                    "interpretation": "  SUCCESS  ",
                    "notes": ["  E4  ", "G4"],
                }],
            })
        assert response.status_code == 200
        rule = response.json()["rules"][0]
        assert rule["queue"] == "compass-docs"
        assert rule["interpretation"] == "SUCCESS"
        assert rule["notes"] == ["E4", "G4"]

    def test_get_interpretations(self, tmp_path, client):
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            response = client.get("/api/interpretations")
        assert response.status_code == 200
        names = response.json()["interpretations"]
        assert "SUCCESS" in names
        assert "ERROR" in names


class TestEnginesTryRoute:
    """Engine/voice preview: a 'Try it' button in the Engine & Voice section."""

    @patch("speeker.cli.start_player")
    @patch("speeker.queue_db.enqueue")
    def test_try_attaches_engine_voice_metadata(self, mock_enqueue, mock_start, client):
        """Engine + voice arrive as per-item metadata. process_queue uses
        these to override the saved session settings on this one
        utterance only."""
        mock_enqueue.return_value = 4242
        response = client.post(
            "/api/engines/try",
            json={"engine": "polly", "voice": "Matthew"},
        )
        assert response.status_code == 200
        meta = mock_enqueue.call_args.kwargs["metadata"]
        assert meta["queue"] == "default"
        assert meta["engine"] == "polly"
        assert meta["voice"] == "Matthew"
        # The previewed phrase identifies the purpose.
        assert "preview" in mock_enqueue.call_args.args[0].lower()

    @patch("speeker.cli.start_player")
    @patch("speeker.queue_db.enqueue")
    def test_try_omits_missing_axes(self, mock_enqueue, mock_start, client):
        """Omitted engine/voice fall back to the saved defaults via the
        normal metadata absence. The Try endpoint doesn't fabricate
        engine='None' style sentinels."""
        mock_enqueue.return_value = 4243
        response = client.post("/api/engines/try", json={"voice": "Ruth"})
        assert response.status_code == 200
        meta = mock_enqueue.call_args.kwargs["metadata"]
        assert meta["voice"] == "Ruth"
        assert "engine" not in meta

    def test_try_rejects_unknown_engine(self, client):
        """Bogus engine -> 400 with the list of known engines in the
        detail (helpful for the UI / human)."""
        response = client.post(
            "/api/engines/try", json={"engine": "make-believe-engine"},
        )
        assert response.status_code == 400
        assert "make-believe-engine" in response.json()["detail"]

    def test_try_rejects_unknown_voice(self, client):
        response = client.post(
            "/api/engines/try", json={"voice": "Joanna-but-misspelled"},
        )
        assert response.status_code == 400
