#!/usr/bin/env python3
"""Unit tests for web.py utility functions and routes."""

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
)


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
    """Tests for render_metadata function."""

    def test_render_metadata_none_returns_placeholder(self):
        """Test None metadata returns placeholder."""
        result = render_metadata(None)
        assert 'class="no-data"' in result
        assert "-" in result

    def test_render_metadata_empty_dict_returns_placeholder(self):
        """Test empty dict returns placeholder."""
        result = render_metadata({})
        assert 'class="no-data"' in result

    def test_render_metadata_single_key(self):
        """Test single key-value pair."""
        result = render_metadata({"queue": "rm"})
        assert 'class="kv"' in result
        assert 'class="key"' in result
        assert 'class="value"' in result
        assert "queue:" in result
        assert "rm" in result

    def test_render_metadata_multiple_keys(self):
        """Test multiple key-value pairs."""
        result = render_metadata({"queue": "rm", "priority": "high"})
        assert "queue:" in result
        assert "rm" in result
        assert "priority:" in result
        assert "high" in result

    def test_render_metadata_escapes_html_in_keys(self):
        """Test HTML in keys is escaped."""
        result = render_metadata({"<script>": "value"})
        assert "&lt;script&gt;" in result
        assert "<script>" not in result

    def test_render_metadata_escapes_html_in_values(self):
        """Test HTML in values is escaped."""
        result = render_metadata({"key": "<b>bold</b>"})
        assert "&lt;b&gt;" in result
        assert "<b>" not in result

    def test_render_metadata_handles_nested_dict(self):
        """Test nested dict value is JSON encoded."""
        result = render_metadata({"config": {"nested": "value"}})
        assert "nested" in result
        assert "value" in result


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
    """Additional edge cases for render_metadata."""

    def test_render_metadata_numeric_values(self):
        """Test numeric values."""
        result = render_metadata({"count": 42, "rate": 3.14})
        assert "42" in result
        assert "3.14" in result

    def test_render_metadata_boolean_values(self):
        """Test boolean values."""
        result = render_metadata({"active": True, "deleted": False})
        assert "True" in result
        assert "False" in result

    def test_render_metadata_none_value(self):
        """Test None as a value (not the whole metadata)."""
        result = render_metadata({"key": None})
        # None value should become empty string
        assert "key:" in result

    def test_render_metadata_xss_in_key_and_value(self):
        """Test XSS attempt in both key and value."""
        result = render_metadata({'<script>alert(1)</script>': '<img onerror=alert(1)>'})
        assert "<script>" not in result
        assert "<img" not in result
        assert "&lt;" in result

    def test_render_metadata_very_long_value(self):
        """Test very long string value."""
        result = render_metadata({"key": "x" * 1000})
        assert "x" * 100 in result  # At least partial

    def test_render_metadata_unicode_key_and_value(self):
        """Test unicode in both key and value."""
        result = render_metadata({"键": "值"})
        assert "键" in result
        assert "值" in result

    def test_render_metadata_spaces_in_key(self):
        """Test key with spaces."""
        result = render_metadata({"my key": "value"})
        assert "my key:" in result

    def test_render_metadata_empty_string_value(self):
        """Test empty string as value."""
        result = render_metadata({"key": ""})
        assert "key:" in result


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

    @patch("speeker.web.get_history")
    def test_index_shows_empty_state(self, mock_history, client):
        """Test index shows no messages when empty."""
        mock_history.return_value = []
        response = client.get("/")
        assert response.status_code == 200
        assert "No messages yet" in response.text

    @patch("speeker.web.get_history")
    def test_index_shows_items(self, mock_history, client):
        """Test index renders queue items."""
        mock_history.return_value = [
            {
                "id": 1,
                "text": "Test message",
                "created_at": "2024-01-15T14:30:00",
                "played_at": None,
                "audio_path": None,
                "session_id": None,
                "metadata": None,
            }
        ]
        response = client.get("/")
        assert response.status_code == 200
        assert "Test message" in response.text
        assert "Pending" in response.text

    @patch("speeker.web.get_history")
    def test_index_escapes_html_in_text(self, mock_history, client):
        """Test index escapes HTML in message text."""
        mock_history.return_value = [
            {
                "id": 1,
                "text": "<script>alert('xss')</script>",
                "created_at": "2024-01-15T14:30:00",
                "played_at": None,
                "audio_path": None,
                "session_id": None,
                "metadata": None,
            }
        ]
        response = client.get("/")
        assert response.status_code == 200
        # The script tags should be escaped so they don't execute
        assert "&lt;script&gt;" in response.text
        assert "&lt;/script&gt;" in response.text

    @patch("speeker.web.search")
    def test_index_with_search_query(self, mock_search, client):
        """Test index with search query uses search function."""
        mock_search.return_value = [
            {
                "id": 1,
                "text": "Found item",
                "created_at": "2024-01-15T14:30:00",
                "played_at": "2024-01-15T14:31:00",
                "audio_path": None,
                "session_id": None,
                "metadata": None,
                "score": 0.95,
            }
        ]
        response = client.get("/?q=test")
        assert response.status_code == 200
        assert "Found item" in response.text
        mock_search.assert_called_once_with("test", limit=200)

    @patch("speeker.web.get_history")
    def test_index_shows_played_status(self, mock_history, client):
        """Test index shows Played status for played items."""
        mock_history.return_value = [
            {
                "id": 1,
                "text": "Played message",
                "created_at": "2024-01-15T14:30:00",
                "played_at": "2024-01-15T14:31:00",
                "audio_path": None,
                "session_id": None,
                "metadata": None,
            }
        ]
        response = client.get("/")
        assert response.status_code == 200
        assert "Played" in response.text

    @patch("speeker.web.get_history")
    def test_index_renders_metadata(self, mock_history, client):
        """Test index renders metadata."""
        mock_history.return_value = [
            {
                "id": 1,
                "text": "Message",
                "created_at": "2024-01-15T14:30:00",
                "played_at": None,
                "audio_path": None,
                "session_id": None,
                "metadata": {"queue": "alerts", "source": "monitor"},
            }
        ]
        response = client.get("/")
        assert response.status_code == 200
        assert "queue:" in response.text
        assert "alerts" in response.text


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
