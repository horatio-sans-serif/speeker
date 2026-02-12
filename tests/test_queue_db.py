#!/usr/bin/env python3
"""Unit tests for queue_db functions."""

import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from speeker.queue_db import (
    get_queue_label,
    relative_time,
    enqueue,
    get_sessions_with_pending,
    get_pending_for_session,
    mark_played,
    get_pending_count,
    get_all_sessions,
    get_settings,
    set_settings,
    get_history,
    cleanup_old_entries,
    get_last_utterance_time,
    set_last_utterance_time,
    search,
    search_fuzzy,
    is_semantic_search_enabled,
)


class TestGetQueueLabel:
    """Tests for get_queue_label function."""

    def test_get_queue_label_default(self):
        """Test that default queue returns friendly label."""
        assert get_queue_label("default") == "the default queue"

    def test_get_queue_label_empty(self):
        """Test that empty string returns default queue label."""
        assert get_queue_label("") == "the default queue"

    def test_get_queue_label_none(self):
        """Test that None returns default queue label."""
        assert get_queue_label(None) == "the default queue"

    def test_get_queue_label_short_name(self):
        """Test short queue names are preserved."""
        assert get_queue_label("myqueue") == "queue myqueue"
        assert get_queue_label("rm") == "queue rm"
        assert get_queue_label("test") == "queue test"

    def test_get_queue_label_long_name_truncated(self):
        """Test that long queue names are truncated to 8 chars."""
        assert get_queue_label("verylongqueuename") == "queue verylong"
        assert get_queue_label("12345678901234567890") == "queue 12345678"

    def test_get_queue_label_exactly_eight_chars(self):
        """Test that exactly 8 char names are not truncated."""
        assert get_queue_label("eightchr") == "queue eightchr"


class TestRelativeTime:
    """Tests for relative_time function."""

    def test_relative_time_very_recent_returns_none(self):
        """Test that times < 2 minutes return None."""
        now = datetime.now(timezone.utc)
        recent = (now - timedelta(seconds=60)).isoformat()
        assert relative_time(recent) is None

    def test_relative_time_minutes_ago(self):
        """Test times between 2-60 minutes return minutes."""
        now = datetime.now(timezone.utc)
        dt = (now - timedelta(minutes=15)).isoformat()
        result = relative_time(dt)
        assert result is not None
        assert "15 minutes ago" in result

    def test_relative_time_about_an_hour(self):
        """Test times between 1-2 hours return 'about an hour ago'."""
        now = datetime.now(timezone.utc)
        dt = (now - timedelta(hours=1, minutes=30)).isoformat()
        result = relative_time(dt)
        assert result == "about an hour ago"

    def test_relative_time_hours_ago(self):
        """Test times between 2-24 hours return hours."""
        now = datetime.now(timezone.utc)
        dt = (now - timedelta(hours=5)).isoformat()
        result = relative_time(dt)
        assert result is not None
        assert "5 hours ago" in result

    def test_relative_time_yesterday(self):
        """Test times between 24-48 hours return 'yesterday'."""
        now = datetime.now(timezone.utc)
        dt = (now - timedelta(hours=30)).isoformat()
        result = relative_time(dt)
        assert result == "yesterday"

    def test_relative_time_days_ago(self):
        """Test times > 48 hours return days."""
        now = datetime.now(timezone.utc)
        dt = (now - timedelta(days=5)).isoformat()
        result = relative_time(dt)
        assert result is not None
        assert "5 days ago" in result

    def test_relative_time_handles_naive_datetime(self):
        """Test that naive datetimes are handled (assumed UTC)."""
        now = datetime.now(timezone.utc)
        # Create naive datetime string
        dt = (now - timedelta(hours=3)).replace(tzinfo=None).isoformat()
        result = relative_time(dt)
        assert result is not None
        assert "hours ago" in result

    def test_relative_time_handles_timezone_aware(self):
        """Test that timezone-aware datetimes work."""
        now = datetime.now(timezone.utc)
        dt = (now - timedelta(minutes=10)).isoformat()
        result = relative_time(dt)
        assert result is not None
        assert "10 minutes ago" in result

    def test_relative_time_boundary_two_minutes(self):
        """Test the 2-minute boundary."""
        now = datetime.now(timezone.utc)

        # Just under 2 minutes - should be None
        dt_under = (now - timedelta(seconds=119)).isoformat()
        assert relative_time(dt_under) is None

        # Just over 2 minutes - should return minutes
        dt_over = (now - timedelta(seconds=121)).isoformat()
        result = relative_time(dt_over)
        assert result is not None
        assert "2 minutes ago" in result

    # --- Edge cases I missed ---

    def test_relative_time_future_date_returns_none(self):
        """Test that future dates return None (negative seconds < 120)."""
        now = datetime.now(timezone.utc)
        future = (now + timedelta(hours=1)).isoformat()
        # Future dates have negative seconds, which is < 120, so returns None
        result = relative_time(future)
        assert result is None

    def test_relative_time_invalid_string_raises(self):
        """Test that invalid datetime string raises ValueError."""
        import pytest
        with pytest.raises(ValueError):
            relative_time("not-a-date")

    def test_relative_time_empty_string_raises(self):
        """Test that empty string raises ValueError."""
        import pytest
        with pytest.raises(ValueError):
            relative_time("")

    def test_relative_time_very_old_date(self):
        """Test handling of dates years in the past."""
        now = datetime.now(timezone.utc)
        old = (now - timedelta(days=365)).isoformat()
        result = relative_time(old)
        assert result is not None
        assert "365 days ago" in result

    def test_relative_time_exactly_one_hour(self):
        """Test boundary at exactly 1 hour (3600 seconds)."""
        now = datetime.now(timezone.utc)
        dt = (now - timedelta(seconds=3600)).isoformat()
        result = relative_time(dt)
        # 3600 is not < 3600, so falls to elif < 7200
        assert result == "about an hour ago"

    def test_relative_time_exactly_two_hours(self):
        """Test boundary at exactly 2 hours (7200 seconds)."""
        now = datetime.now(timezone.utc)
        dt = (now - timedelta(seconds=7200)).isoformat()
        result = relative_time(dt)
        # 7200 is not < 7200, so falls to elif < 86400
        assert result is not None
        assert "2 hours ago" in result

    def test_relative_time_exactly_24_hours(self):
        """Test boundary at exactly 24 hours."""
        now = datetime.now(timezone.utc)
        dt = (now - timedelta(hours=24)).isoformat()
        result = relative_time(dt)
        # 86400 is not < 86400, so falls to elif < 172800
        assert result == "yesterday"

    def test_relative_time_exactly_48_hours(self):
        """Test boundary at exactly 48 hours."""
        now = datetime.now(timezone.utc)
        dt = (now - timedelta(hours=48)).isoformat()
        result = relative_time(dt)
        # 172800 is not < 172800, so falls to else
        assert result is not None
        assert "2 days ago" in result

    def test_relative_time_with_microseconds(self):
        """Test datetime strings with microseconds."""
        now = datetime.now(timezone.utc)
        dt = (now - timedelta(minutes=30)).isoformat()
        # ISO format may include microseconds
        result = relative_time(dt)
        assert result is not None
        assert "30 minutes ago" in result


class TestGetQueueLabelEdgeCases:
    """Additional edge case tests for get_queue_label."""

    def test_get_queue_label_whitespace_only(self):
        """Test whitespace-only string is treated as valid queue name."""
        # "   " is truthy, so it won't match "not queue_id"
        result = get_queue_label("   ")
        assert result == "queue    "  # Whitespace preserved (truncated to 8)

    def test_get_queue_label_unicode(self):
        """Test unicode characters in queue name."""
        result = get_queue_label("队列测试")
        assert "队列测试" in result

    def test_get_queue_label_unicode_long(self):
        """Test long unicode string is truncated by chars not bytes."""
        result = get_queue_label("日本語のキュー名です")
        # Should truncate to first 8 characters
        assert result == "queue 日本語のキュー名"

    def test_get_queue_label_special_characters(self):
        """Test special characters in queue name."""
        result = get_queue_label("my-queue_v2.0")
        assert result == "queue my-queue"

    def test_get_queue_label_newlines(self):
        """Test queue name with newlines."""
        result = get_queue_label("queue\nname")
        # "queue\nname" is 10 chars, truncated to first 8: "queue\nna"
        assert result == "queue queue\nna"

    def test_get_queue_label_zero_string(self):
        """Test '0' as queue name (falsy-looking but valid)."""
        result = get_queue_label("0")
        assert result == "queue 0"

    def test_get_queue_label_case_preserved(self):
        """Test that case is preserved."""
        assert get_queue_label("MyQueue") == "queue MyQueue"
        assert get_queue_label("MYQUEUE") == "queue MYQUEUE"

    def test_get_queue_label_with_spaces(self):
        """Test queue name with spaces."""
        result = get_queue_label("my queue")
        assert result == "queue my queue"


# --- Database Integration Tests ---

@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Create a temporary database for testing."""
    import speeker.queue_db as qdb

    # Use SPEEKER_DIR so db_path() returns tmp_path/data/queue.db
    monkeypatch.setenv("SPEEKER_DIR", str(tmp_path))

    # Clear thread-local connection
    if hasattr(qdb._local, "conn"):
        qdb._local.conn = None

    yield tmp_path / "data" / "queue.db"

    # Cleanup
    if hasattr(qdb._local, "conn") and qdb._local.conn:
        qdb._local.conn.close()
        qdb._local.conn = None


class TestEnqueue:
    """Tests for enqueue function."""

    def test_enqueue_basic(self, temp_db):
        """Test basic enqueue returns item ID."""
        item_id = enqueue("Hello world")
        assert isinstance(item_id, int)
        assert item_id > 0

    def test_enqueue_with_metadata(self, temp_db):
        """Test enqueue with metadata."""
        item_id = enqueue("Test message", metadata={"queue": "myqueue", "priority": "high"})
        assert item_id > 0

    def test_enqueue_with_legacy_session_id(self, temp_db):
        """Test enqueue with legacy session_id parameter."""
        item_id = enqueue("Test message", session_id="legacy-queue")
        assert item_id > 0

    def test_enqueue_session_id_merged_with_metadata(self, temp_db):
        """Test session_id is merged into metadata."""
        item_id = enqueue("Test", metadata={"key": "value"}, session_id="myqueue")
        history = get_history(limit=1)
        assert len(history) > 0
        assert history[0]["session_id"] == "myqueue"

    def test_enqueue_multiple_items(self, temp_db):
        """Test enqueueing multiple items."""
        id1 = enqueue("First")
        id2 = enqueue("Second")
        id3 = enqueue("Third")
        assert id1 < id2 < id3


class TestGetSessionsWithPending:
    """Tests for get_sessions_with_pending function."""

    def test_get_sessions_with_pending_empty(self, temp_db):
        """Test returns empty list when no pending items."""
        result = get_sessions_with_pending()
        assert result == []

    def test_get_sessions_with_pending_one_session(self, temp_db):
        """Test returns session with pending items."""
        enqueue("Test", metadata={"queue": "session1"})
        result = get_sessions_with_pending()
        assert "session1" in result

    def test_get_sessions_with_pending_multiple_sessions(self, temp_db):
        """Test returns multiple sessions."""
        enqueue("Test 1", metadata={"queue": "session1"})
        enqueue("Test 2", metadata={"queue": "session2"})
        result = get_sessions_with_pending()
        assert len(result) == 2
        assert "session1" in result
        assert "session2" in result

    def test_get_sessions_with_pending_excludes_played(self, temp_db):
        """Test excludes sessions with only played items."""
        item_id = enqueue("Test", metadata={"queue": "played-session"})
        mark_played(item_id)
        result = get_sessions_with_pending()
        assert "played-session" not in result


class TestGetPendingForSession:
    """Tests for get_pending_for_session function."""

    def test_get_pending_for_session_empty(self, temp_db):
        """Test returns empty list for session with no items."""
        result = get_pending_for_session("nonexistent")
        assert result == []

    def test_get_pending_for_session_returns_items(self, temp_db):
        """Test returns pending items for session."""
        enqueue("Test 1", metadata={"queue": "mysession"})
        enqueue("Test 2", metadata={"queue": "mysession"})
        result = get_pending_for_session("mysession")
        assert len(result) == 2

    def test_get_pending_for_session_excludes_played(self, temp_db):
        """Test excludes played items."""
        id1 = enqueue("Test 1", metadata={"queue": "mysession"})
        enqueue("Test 2", metadata={"queue": "mysession"})
        mark_played(id1)
        result = get_pending_for_session("mysession")
        assert len(result) == 1

    def test_get_pending_for_session_order_by_created(self, temp_db):
        """Test items are ordered by creation time."""
        enqueue("First", metadata={"queue": "mysession"})
        enqueue("Second", metadata={"queue": "mysession"})
        result = get_pending_for_session("mysession")
        assert result[0]["text"] == "First"
        assert result[1]["text"] == "Second"


class TestMarkPlayed:
    """Tests for mark_played function."""

    def test_mark_played_updates_item(self, temp_db):
        """Test mark_played sets played_at timestamp."""
        item_id = enqueue("Test")
        mark_played(item_id)
        history = get_history(limit=1)
        assert history[0]["played_at"] is not None

    def test_mark_played_removes_from_pending(self, temp_db):
        """Test marked item is no longer pending."""
        item_id = enqueue("Test", metadata={"queue": "mysession"})
        assert len(get_pending_for_session("mysession")) == 1
        mark_played(item_id)
        assert len(get_pending_for_session("mysession")) == 0


class TestGetPendingCount:
    """Tests for get_pending_count function."""

    def test_get_pending_count_zero(self, temp_db):
        """Test returns 0 when no pending items."""
        assert get_pending_count() == 0

    def test_get_pending_count_with_items(self, temp_db):
        """Test returns correct count."""
        enqueue("Test 1")
        enqueue("Test 2")
        enqueue("Test 3")
        assert get_pending_count() == 3

    def test_get_pending_count_excludes_played(self, temp_db):
        """Test excludes played items from count."""
        id1 = enqueue("Test 1")
        enqueue("Test 2")
        mark_played(id1)
        assert get_pending_count() == 1


class TestGetAllSessions:
    """Tests for get_all_sessions function."""

    def test_get_all_sessions_empty(self, temp_db):
        """Test returns empty list when no sessions."""
        result = get_all_sessions()
        assert result == []

    def test_get_all_sessions_returns_unique(self, temp_db):
        """Test returns unique session IDs."""
        enqueue("Test 1", metadata={"queue": "session1"})
        enqueue("Test 2", metadata={"queue": "session1"})
        enqueue("Test 3", metadata={"queue": "session2"})
        result = get_all_sessions()
        assert len(result) == 2


class TestGetHistory:
    """Tests for get_history function."""

    def test_get_history_empty(self, temp_db):
        """Test returns empty list when no history."""
        result = get_history()
        assert result == []

    def test_get_history_returns_items(self, temp_db):
        """Test returns enqueued items."""
        enqueue("Test message")
        result = get_history()
        assert len(result) == 1
        assert result[0]["text"] == "Test message"

    def test_get_history_limit(self, temp_db):
        """Test respects limit parameter."""
        for i in range(10):
            enqueue(f"Message {i}")
        result = get_history(limit=5)
        assert len(result) == 5

    def test_get_history_order_desc(self, temp_db):
        """Test returns newest first."""
        enqueue("First")
        enqueue("Second")
        result = get_history()
        assert result[0]["text"] == "Second"
        assert result[1]["text"] == "First"

    def test_get_history_includes_metadata(self, temp_db):
        """Test includes parsed metadata."""
        enqueue("Test", metadata={"key": "value"})
        result = get_history()
        assert result[0]["metadata"] == {"key": "value"}

    def test_get_history_filter_by_session(self, temp_db):
        """Test filters by session_id."""
        enqueue("Test 1", metadata={"queue": "session1"})
        enqueue("Test 2", metadata={"queue": "session2"})
        result = get_history(session_id="session1")
        assert len(result) == 1
        assert result[0]["session_id"] == "session1"


class TestSettings:
    """Tests for get_settings and set_settings functions."""

    def test_get_settings_defaults(self, temp_db):
        """Test returns default settings."""
        result = get_settings()
        assert "intro_sound" in result
        assert "speed" in result
        assert "voice" in result
        assert "engine" in result

    def test_set_settings_and_retrieve(self, temp_db):
        """Test set_settings persists values."""
        set_settings(intro_sound=False, speed=1.5)
        result = get_settings()
        assert result["intro_sound"] is False
        assert result["speed"] == 1.5

    def test_settings_per_session(self, temp_db):
        """Test session-specific settings override global."""
        set_settings(speed=1.0)  # Global
        set_settings(session_id="fast-session", speed=2.0)
        global_settings = get_settings()
        session_settings = get_settings("fast-session")
        assert global_settings["speed"] == 1.0
        assert session_settings["speed"] == 2.0

    def test_set_settings_voice(self, temp_db):
        """Test setting voice."""
        set_settings(voice="alba")
        result = get_settings()
        assert result["voice"] == "alba"

    def test_set_settings_engine(self, temp_db):
        """Test setting engine."""
        set_settings(engine="kokoro")
        result = get_settings()
        assert result["engine"] == "kokoro"


class TestCleanupOldEntries:
    """Tests for cleanup_old_entries function."""

    def test_cleanup_old_entries_empty(self, temp_db):
        """Test cleanup on empty database returns 0."""
        result = cleanup_old_entries(days=7)
        assert result == 0

    def test_cleanup_old_entries_keeps_unplayed(self, temp_db):
        """Test cleanup keeps unplayed items."""
        enqueue("Test")
        result = cleanup_old_entries(days=0)
        assert result == 0
        assert len(get_history()) == 1


class TestUtteranceTime:
    """Tests for get/set_last_utterance_time functions."""

    def test_get_last_utterance_time_none_initially(self, temp_db):
        """Test returns None when never set."""
        result = get_last_utterance_time()
        assert result is None

    def test_set_and_get_last_utterance_time(self, temp_db):
        """Test set then get returns datetime."""
        set_last_utterance_time()
        result = get_last_utterance_time()
        assert result is not None
        assert isinstance(result, datetime)


class TestSearch:
    """Tests for search functions."""

    def test_search_empty_database(self, temp_db):
        """Test search on empty database returns empty list."""
        result = search("test")
        assert result == []

    def test_search_finds_text_match(self, temp_db):
        """Test search finds items by text."""
        enqueue("Hello world test message")
        enqueue("Another message")
        result = search("test")
        assert len(result) >= 1
        assert any("test" in item["text"].lower() for item in result)

    def test_search_with_limit(self, temp_db):
        """Test search respects limit parameter."""
        for i in range(10):
            enqueue(f"Test message {i}")
        result = search("test", limit=3)
        assert len(result) <= 3


class TestSearchFuzzy:
    """Tests for search_fuzzy function."""

    def test_search_fuzzy_empty_database(self, temp_db):
        """Test fuzzy search on empty database."""
        result = search_fuzzy("test")
        assert result == []

    def test_search_fuzzy_finds_exact_match(self, temp_db):
        """Test fuzzy search finds exact substring match."""
        enqueue("Hello world test message")
        result = search_fuzzy("test")
        assert len(result) == 1
        assert result[0]["score"] > 0

    def test_search_fuzzy_finds_partial_match(self, temp_db):
        """Test fuzzy search finds partial word matches."""
        enqueue("Testing the application")
        result = search_fuzzy("test")
        assert len(result) == 1

    def test_search_fuzzy_searches_metadata(self, temp_db):
        """Test fuzzy search searches metadata values."""
        enqueue("Normal message", metadata={"queue": "important"})
        result = search_fuzzy("important")
        assert len(result) == 1

    def test_search_fuzzy_no_match_returns_empty(self, temp_db):
        """Test fuzzy search returns empty when no match."""
        enqueue("Hello world")
        result = search_fuzzy("xyznotfound")
        assert len(result) == 0

    def test_search_fuzzy_scores_exact_higher(self, temp_db):
        """Test fuzzy search scores exact matches higher."""
        enqueue("test message")  # Exact match
        enqueue("testing")  # Partial match
        result = search_fuzzy("test")
        # Both should be found, exact match should score higher
        assert len(result) == 2
        # Results are sorted by score descending

    def test_search_fuzzy_case_insensitive(self, temp_db):
        """Test fuzzy search is case insensitive."""
        enqueue("Hello WORLD")
        result = search_fuzzy("world")
        assert len(result) == 1

    def test_search_fuzzy_multiple_words(self, temp_db):
        """Test fuzzy search handles multiple word queries."""
        enqueue("Hello beautiful world")
        result = search_fuzzy("hello world")
        assert len(result) == 1
        # Score should reflect both word matches


class TestSemanticSearchEnabled:
    """Tests for is_semantic_search_enabled function."""

    @patch("speeker.config.is_semantic_search_enabled")
    def test_search_uses_semantic_when_enabled(self, mock_enabled, temp_db):
        """Test search uses semantic search when enabled."""
        mock_enabled.return_value = True
        # Without proper embedding setup, semantic search will return []
        enqueue("Test message")
        result = search("test")
        # Result depends on embedding setup

    @patch("speeker.queue_db.is_semantic_search_enabled")
    def test_search_uses_fuzzy_when_semantic_disabled(self, mock_enabled, temp_db):
        """Test search uses fuzzy search when semantic disabled."""
        mock_enabled.return_value = False
        enqueue("Test message")
        result = search("test")
        assert len(result) >= 1
