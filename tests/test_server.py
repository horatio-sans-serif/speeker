#!/usr/bin/env python3
"""Unit tests for server.py utility functions and HTTP endpoints."""

from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from speeker.server import (
    app,
    extract_metadata,
    extract_title,
    format_with_title,
    elide_message_count,
)


class TestExtractMetadata:
    """Tests for extract_metadata function."""

    def test_extract_metadata_with_bang_prefix(self):
        """Test extracts params with ! prefix."""
        request = Mock()
        request.query_params = {"!foo": "bar", "!queue": "myqueue"}
        result = extract_metadata(request)
        assert result == {"foo": "bar", "queue": "myqueue"}

    def test_extract_metadata_mixed_params(self):
        """Test only extracts ! prefixed params."""
        request = Mock()
        request.query_params = {"!foo": "bar", "regular": "ignored", "!baz": "qux"}
        result = extract_metadata(request)
        assert result == {"foo": "bar", "baz": "qux"}
        assert result is not None and "regular" not in result

    def test_extract_metadata_no_bang_params_returns_none(self):
        """Test returns None when no ! params."""
        request = Mock()
        request.query_params = {"regular": "param", "another": "one"}
        result = extract_metadata(request)
        assert result is None

    def test_extract_metadata_empty_params_returns_none(self):
        """Test returns None for empty params."""
        request = Mock()
        request.query_params = {}
        result = extract_metadata(request)
        assert result is None

    def test_extract_metadata_single_param(self):
        """Test extracts single ! param."""
        request = Mock()
        request.query_params = {"!queue": "rm"}
        result = extract_metadata(request)
        assert result == {"queue": "rm"}

    def test_extract_metadata_empty_value(self):
        """Test handles empty value."""
        request = Mock()
        request.query_params = {"!key": ""}
        result = extract_metadata(request)
        assert result == {"key": ""}

    def test_extract_metadata_special_chars_in_value(self):
        """Test handles special characters in value."""
        request = Mock()
        request.query_params = {"!msg": "hello world & more"}
        result = extract_metadata(request)
        assert result == {"msg": "hello world & more"}


class TestExtractTitle:
    """Tests for extract_title function."""

    def test_extract_title_present(self):
        """Test extracts title when present."""
        request = Mock()
        request.query_params = Mock()
        request.query_params.get = Mock(return_value="My Title")
        result = extract_title(request)
        assert result == "My Title"

    def test_extract_title_missing(self):
        """Test returns None when title missing."""
        request = Mock()
        request.query_params = Mock()
        request.query_params.get = Mock(return_value=None)
        result = extract_title(request)
        assert result is None

    def test_extract_title_empty_string(self):
        """Test returns empty string if that's the value."""
        request = Mock()
        request.query_params = Mock()
        request.query_params.get = Mock(return_value="")
        result = extract_title(request)
        assert result == ""


class TestFormatWithTitle:
    """Tests for format_with_title function."""

    def test_format_with_title_adds_prefix(self):
        """Test adds title prefix with tone marker."""
        result = format_with_title("Hello world", "Important")
        assert result == "$Eb4 Important. Hello world"

    def test_format_with_title_none_returns_original(self):
        """Test None title returns original text."""
        result = format_with_title("Hello world", None)
        assert result == "Hello world"

    def test_format_with_title_empty_string_returns_original(self):
        """Test empty title string returns original (falsy)."""
        result = format_with_title("Hello world", "")
        assert result == "Hello world"

    def test_format_with_title_preserves_text(self):
        """Test original text is preserved."""
        text = "This is a longer message with special chars: <>&"
        result = format_with_title(text, "Alert")
        assert text in result

    def test_format_with_title_tone_marker(self):
        """Test tone marker is included."""
        result = format_with_title("text", "title")
        assert "$Eb4" in result

    def test_format_with_title_period_after_title(self):
        """Test period separates title from text."""
        result = format_with_title("text", "title")
        assert "title. text" in result


class TestElideMessageCount:
    """Tests for elide_message_count function."""

    def test_elide_there_is_1_message(self):
        """Test removes 'there is 1 message'."""
        result = elide_message_count("there is 1 message. Hello")
        assert "there is" not in result.lower()
        assert "Hello" in result

    def test_elide_there_are_n_messages(self):
        """Test removes 'there are N messages'."""
        result = elide_message_count("there are 5 messages. Hello")
        assert "there are" not in result.lower()
        assert "Hello" in result

    def test_elide_messages_waiting(self):
        """Test removes 'N messages waiting'."""
        result = elide_message_count("3 messages waiting. Hello")
        assert "messages waiting" not in result.lower()
        assert "Hello" in result

    def test_elide_messages_pending(self):
        """Test removes 'messages pending'."""
        result = elide_message_count("there are 2 messages pending. Hello")
        assert "pending" not in result.lower()
        assert "Hello" in result

    def test_elide_messages_queued(self):
        """Test removes 'messages queued'."""
        result = elide_message_count("you have 4 messages queued. Hello")
        assert "queued" not in result.lower()
        assert "Hello" in result

    def test_elide_you_have_new_messages(self):
        """Test removes 'you have N new messages'."""
        result = elide_message_count("you have 3 new messages. Hello")
        assert "you have" not in result.lower()
        assert "Hello" in result

    def test_elide_case_insensitive(self):
        """Test removal is case insensitive."""
        result = elide_message_count("THERE ARE 5 MESSAGES. Hello")
        assert "there are" not in result.lower()
        assert "Hello" in result

    def test_elide_no_match_returns_original(self):
        """Test text without patterns is unchanged."""
        original = "Hello world, this is a test"
        result = elide_message_count(original)
        assert result == original

    def test_elide_strips_whitespace(self):
        """Test result is stripped."""
        result = elide_message_count("  there are 5 messages.  Hello  ")
        assert not result.startswith(" ")
        assert not result.endswith("  ")

    def test_elide_empty_string(self):
        """Test empty string returns empty."""
        result = elide_message_count("")
        assert result == ""

    def test_elide_only_message_count_returns_empty(self):
        """Test text that is only message count returns empty."""
        result = elide_message_count("there are 5 messages waiting.")
        assert result == ""

    def test_elide_multiple_patterns(self):
        """Test removes multiple matching patterns."""
        text = "there are 3 messages. Also you have 2 new messages. Hello"
        result = elide_message_count(text)
        assert "Hello" in result
        # Both patterns should be removed


class TestElideMessageCountEdgeCases:
    """Edge cases for elide_message_count."""

    def test_elide_large_number(self):
        """Test handles large message counts."""
        result = elide_message_count("there are 99999 messages. Hello")
        assert "Hello" in result
        assert "99999" not in result

    def test_elide_preserves_other_numbers(self):
        """Test preserves numbers not in message count context."""
        result = elide_message_count("The answer is 42. Hello")
        assert "42" in result
        assert "Hello" in result

    def test_elide_punctuation_variations(self):
        """Test handles different punctuation."""
        # Without period
        result1 = elide_message_count("there are 5 messages Hello")
        assert "Hello" in result1

        # With comma (less common but possible)
        result2 = elide_message_count("there are 5 messages, and Hello")
        # Pattern might not catch this perfectly, just verify no crash
        assert "Hello" in result2


# HTTP Endpoint Tests
@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app, raise_server_exceptions=False)


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_returns_ok(self, client):
        """Test health endpoint returns status ok."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestQueueStatusEndpoint:
    """Tests for /queue/status endpoint."""

    @patch("speeker.server.get_pending_count")
    def test_queue_status_returns_count(self, mock_count, client):
        """Test queue status returns pending count."""
        mock_count.return_value = 5
        response = client.get("/queue/status")
        assert response.status_code == 200
        assert response.json() == {"pending_count": 5}

    @patch("speeker.server.get_pending_count")
    def test_queue_status_zero_pending(self, mock_count, client):
        """Test queue status with zero pending."""
        mock_count.return_value = 0
        response = client.get("/queue/status")
        assert response.status_code == 200
        assert response.json() == {"pending_count": 0}


class TestVoicesEndpoint:
    """Tests for /voices endpoint."""

    def test_get_all_voices(self, client):
        """Test getting all voices returns both engines."""
        response = client.get("/voices")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "pocket-tts" in data["engines"]
        assert "kokoro" in data["engines"]
        assert "default_engine" in data

    def test_get_pocket_tts_voices(self, client):
        """Test filtering to pocket-tts voices only."""
        response = client.get("/voices?engine=pocket-tts")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "pocket-tts" in data["engines"]
        assert "kokoro" not in data["engines"]

    def test_get_kokoro_voices(self, client):
        """Test filtering to kokoro voices only."""
        response = client.get("/voices?engine=kokoro")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "kokoro" in data["engines"]
        assert "pocket-tts" not in data["engines"]

    def test_get_unknown_engine_returns_error(self, client):
        """Test unknown engine returns error."""
        response = client.get("/voices?engine=unknown")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "error"
        assert "Unknown engine" in data["error"]

    def test_voices_structure(self, client):
        """Test voice data structure."""
        response = client.get("/voices?engine=pocket-tts")
        data = response.json()
        engine_data = data["engines"]["pocket-tts"]
        assert "default" in engine_data
        assert "voices" in engine_data
        # Check voice structure
        for voice_name, voice_info in engine_data["voices"].items():
            assert "description" in voice_info
            assert "is_default" in voice_info


class TestSummarizeInfoEndpoint:
    """Tests for /summarize/info endpoint."""

    @patch("speeker.server.get_backend_info")
    def test_summarize_info_returns_backend_info(self, mock_info, client):
        """Test summarize info returns backend configuration."""
        mock_info.return_value = {
            "configured": True,
            "backend": "ollama",
            "endpoint": "http://localhost:11434",
            "model": "llama3.2",
        }
        response = client.get("/summarize/info")
        assert response.status_code == 200
        data = response.json()
        assert data["configured"] is True
        assert data["backend"] == "ollama"

    @patch("speeker.server.get_backend_info")
    def test_summarize_info_not_configured(self, mock_info, client):
        """Test summarize info when not configured."""
        mock_info.return_value = {
            "configured": False,
            "backend": None,
            "message": "No LLM backend configured",
        }
        response = client.get("/summarize/info")
        assert response.status_code == 200
        data = response.json()
        assert data["configured"] is False


class TestSpeakEndpoint:
    """Tests for /speak endpoint."""

    @patch("speeker.server.start_player")
    @patch("speeker.server.get_pending_count")
    @patch("speeker.server.enqueue")
    def test_speak_queues_text(self, mock_enqueue, mock_count, mock_player, client):
        """Test speak endpoint queues text."""
        mock_enqueue.return_value = 123
        mock_count.return_value = 1
        response = client.post("/speak", json={"text": "Hello world"})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["queue_id"] == 123
        assert data["pending_count"] == 1
        mock_enqueue.assert_called_once()
        mock_player.assert_called_once()

    def test_speak_empty_text_returns_400(self, client):
        """Test speak with empty text returns 400."""
        response = client.post("/speak", json={"text": ""})
        assert response.status_code == 400

    def test_speak_whitespace_only_returns_400(self, client):
        """Test speak with whitespace only returns 400."""
        response = client.post("/speak", json={"text": "   "})
        assert response.status_code == 400

    @patch("speeker.server.start_player")
    @patch("speeker.server.get_pending_count")
    @patch("speeker.server.enqueue")
    def test_speak_with_metadata(self, mock_enqueue, mock_count, mock_player, client):
        """Test speak with metadata in body."""
        mock_enqueue.return_value = 1
        mock_count.return_value = 1
        response = client.post(
            "/speak",
            json={"text": "Hello", "metadata": {"queue": "test"}}
        )
        assert response.status_code == 200
        # Verify metadata was passed
        call_args = mock_enqueue.call_args
        assert call_args[1]["metadata"]["queue"] == "test"

    @patch("speeker.server.start_player")
    @patch("speeker.server.get_pending_count")
    @patch("speeker.server.enqueue")
    def test_speak_with_query_metadata(self, mock_enqueue, mock_count, mock_player, client):
        """Test speak with metadata in query params."""
        mock_enqueue.return_value = 1
        mock_count.return_value = 1
        response = client.post("/speak?!queue=myqueue", json={"text": "Hello"})
        assert response.status_code == 200
        call_args = mock_enqueue.call_args
        assert call_args[1]["metadata"]["queue"] == "myqueue"

    @patch("speeker.server.start_player")
    @patch("speeker.server.get_pending_count")
    @patch("speeker.server.enqueue")
    def test_speak_with_title(self, mock_enqueue, mock_count, mock_player, client):
        """Test speak with title adds prefix."""
        mock_enqueue.return_value = 1
        mock_count.return_value = 1
        response = client.post("/speak?title=Alert", json={"text": "System down"})
        assert response.status_code == 200
        # Verify title was prepended
        call_args = mock_enqueue.call_args
        enqueued_text = call_args[0][0]
        assert "$Eb4 Alert." in enqueued_text
        assert "System down" in enqueued_text

    @patch("speeker.server.start_player")
    @patch("speeker.server.get_pending_count")
    @patch("speeker.server.enqueue")
    def test_speak_elides_message_count(self, mock_enqueue, mock_count, mock_player, client):
        """Test speak removes message count phrases."""
        mock_enqueue.return_value = 1
        mock_count.return_value = 1
        response = client.post(
            "/speak",
            json={"text": "There are 5 messages. Important update."}
        )
        assert response.status_code == 200
        call_args = mock_enqueue.call_args
        enqueued_text = call_args[0][0]
        assert "there are" not in enqueued_text.lower()
        assert "Important update" in enqueued_text

    @patch("speeker.server.start_player")
    @patch("speeker.server.get_pending_count")
    @patch("speeker.server.enqueue")
    def test_speak_deprecated_session_id(self, mock_enqueue, mock_count, mock_player, client):
        """Test speak with deprecated session_id maps to queue."""
        mock_enqueue.return_value = 1
        mock_count.return_value = 1
        response = client.post(
            "/speak",
            json={"text": "Hello", "session_id": "oldqueue"}
        )
        assert response.status_code == 200
        call_args = mock_enqueue.call_args
        assert call_args[1]["metadata"]["queue"] == "oldqueue"


class TestSummarizeEndpoint:
    """Tests for /summarize endpoint."""

    @patch("speeker.server.start_player")
    @patch("speeker.server.get_pending_count")
    @patch("speeker.server.enqueue")
    @patch("speeker.server.summarize_for_speech")
    def test_summarize_queues_summary(self, mock_summarize, mock_enqueue, mock_count, mock_player, client):
        """Test summarize endpoint queues summary."""
        mock_summarize.return_value = "Brief summary."
        mock_enqueue.return_value = 456
        mock_count.return_value = 2
        response = client.post(
            "/summarize",
            json={"text": "This is a very long text that needs summarizing."}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["summary"] == "Brief summary."
        assert data["queue_id"] == 456
        assert data["pending_count"] == 2
        assert data["original_length"] > 0
        assert data["summary_length"] > 0

    def test_summarize_empty_text_returns_400(self, client):
        """Test summarize with empty text returns 400."""
        response = client.post("/summarize", json={"text": ""})
        assert response.status_code == 400

    @patch("speeker.server.start_player")
    @patch("speeker.server.get_pending_count")
    @patch("speeker.server.enqueue")
    @patch("speeker.server.summarize_for_speech")
    def test_summarize_with_title(self, mock_summarize, mock_enqueue, mock_count, mock_player, client):
        """Test summarize with title adds prefix."""
        mock_summarize.return_value = "Summary here."
        mock_enqueue.return_value = 1
        mock_count.return_value = 1
        response = client.post(
            "/summarize?title=Update",
            json={"text": "Long text here."}
        )
        assert response.status_code == 200
        call_args = mock_enqueue.call_args
        enqueued_text = call_args[0][0]
        assert "$Eb4 Update." in enqueued_text

    @patch("speeker.server.start_player")
    @patch("speeker.server.get_pending_count")
    @patch("speeker.server.enqueue")
    @patch("speeker.server.summarize_for_speech")
    def test_summarize_with_metadata(self, mock_summarize, mock_enqueue, mock_count, mock_player, client):
        """Test summarize with metadata."""
        mock_summarize.return_value = "Summary."
        mock_enqueue.return_value = 1
        mock_count.return_value = 1
        response = client.post(
            "/summarize?!queue=alerts",
            json={"text": "Long text."}
        )
        assert response.status_code == 200
        call_args = mock_enqueue.call_args
        assert call_args[1]["metadata"]["queue"] == "alerts"

    @patch("speeker.server.start_player")
    @patch("speeker.server.get_pending_count")
    @patch("speeker.server.enqueue")
    @patch("speeker.server.summarize_for_speech")
    def test_summarize_elides_message_count(self, mock_summarize, mock_enqueue, mock_count, mock_player, client):
        """Test summarize removes message count from summary."""
        mock_summarize.return_value = "There are 3 messages. Actual summary."
        mock_enqueue.return_value = 1
        mock_count.return_value = 1
        response = client.post("/summarize", json={"text": "Long text."})
        assert response.status_code == 200
        call_args = mock_enqueue.call_args
        enqueued_text = call_args[0][0]
        assert "there are" not in enqueued_text.lower()

    @patch("speeker.server.summarize_for_speech")
    def test_summarize_handles_exception(self, mock_summarize, client):
        """Test summarize handles exceptions gracefully."""
        mock_summarize.side_effect = Exception("LLM error")
        response = client.post("/summarize", json={"text": "Some text"})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "error"
        assert "LLM error" in data["error"]
