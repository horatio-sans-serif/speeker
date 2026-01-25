"""SQLite-based queue for TTS playback with per-session support."""

import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

# Default database location
DEFAULT_DB_PATH = Path.home() / ".speeker" / "queue.db"

# Thread-local storage for connections
_local = threading.local()


def get_db_path() -> Path:
    """Get the database path, creating parent directory if needed."""
    db_path = DEFAULT_DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    """Get a thread-local database connection."""
    if not hasattr(_local, "conn") or _local.conn is None:
        db_path = get_db_path()
        _local.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _init_db(_local.conn)

    yield _local.conn


def _init_db(conn: sqlite3.Connection) -> None:
    """Initialize the database schema."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            text TEXT NOT NULL,
            audio_path TEXT,
            created_at TEXT NOT NULL,
            played_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS playback_state (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            last_utterance_at TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_queue_session ON queue(session_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_queue_played ON queue(played_at)")
    conn.commit()


def enqueue(session_id: str, text: str, audio_path: str | Path | None = None) -> int:
    """Add an item to the queue. Returns the item ID."""
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO queue (session_id, text, audio_path, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                session_id,
                text,
                str(audio_path) if audio_path else None,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
        return cursor.lastrowid or 0


def get_sessions_with_pending() -> list[str]:
    """Get list of session IDs that have unplayed items, ordered by oldest pending item first."""
    with get_connection() as conn:
        cursor = conn.execute(
            """
            SELECT session_id, MIN(created_at) as oldest
            FROM queue
            WHERE played_at IS NULL
            GROUP BY session_id
            ORDER BY oldest ASC
            """
        )
        return [row["session_id"] for row in cursor.fetchall()]


def get_pending_for_session(session_id: str) -> list[dict]:
    """Get all unplayed items for a session, ordered by creation time."""
    with get_connection() as conn:
        cursor = conn.execute(
            """
            SELECT id, session_id, text, audio_path, created_at
            FROM queue
            WHERE session_id = ? AND played_at IS NULL
            ORDER BY created_at ASC
            """,
            (session_id,),
        )
        return [dict(row) for row in cursor.fetchall()]


def mark_played(item_id: int) -> None:
    """Mark an item as played."""
    with get_connection() as conn:
        conn.execute(
            "UPDATE queue SET played_at = ? WHERE id = ?",
            (datetime.now(timezone.utc).isoformat(), item_id),
        )
        conn.commit()


def get_pending_count() -> int:
    """Get count of unplayed items across all sessions."""
    with get_connection() as conn:
        cursor = conn.execute(
            "SELECT COUNT(*) as count FROM queue WHERE played_at IS NULL"
        )
        return cursor.fetchone()["count"]


def get_last_utterance_time() -> datetime | None:
    """Get the time of the last TTS utterance."""
    with get_connection() as conn:
        cursor = conn.execute(
            "SELECT last_utterance_at FROM playback_state WHERE id = 1"
        )
        row = cursor.fetchone()
        if row and row["last_utterance_at"]:
            return datetime.fromisoformat(row["last_utterance_at"])
        return None


def set_last_utterance_time() -> None:
    """Update the last utterance time to now."""
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO playback_state (id, last_utterance_at) VALUES (1, ?)
            ON CONFLICT(id) DO UPDATE SET last_utterance_at = excluded.last_utterance_at
            """,
            (datetime.now(timezone.utc).isoformat(),),
        )
        conn.commit()


def cleanup_old_entries(days: int = 7) -> int:
    """Remove played entries older than specified days. Returns count removed."""
    with get_connection() as conn:
        cursor = conn.execute(
            """
            DELETE FROM queue
            WHERE played_at IS NOT NULL
            AND datetime(played_at) < datetime('now', ?)
            """,
            (f"-{days} days",),
        )
        conn.commit()
        return cursor.rowcount


def relative_time(dt_str: str) -> str | None:
    """Convert ISO datetime string to relative time phrase.

    Returns None for very recent times (< 2 minutes) to skip the phrase.
    """
    dt = datetime.fromisoformat(dt_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    now = datetime.now(timezone.utc)
    diff = now - dt

    seconds = int(diff.total_seconds())

    if seconds < 120:
        return None  # Skip time phrase for recent messages
    elif seconds < 3600:
        minutes = seconds // 60
        return f"about {minutes} minutes ago"
    elif seconds < 7200:
        return "about an hour ago"
    elif seconds < 86400:
        hours = seconds // 3600
        return f"about {hours} hours ago"
    elif seconds < 172800:
        return "yesterday"
    else:
        days = seconds // 86400
        return f"about {days} days ago"


def get_session_label(session_id: str) -> str:
    """Get a human-friendly label for a session ID."""
    if not session_id or session_id == "default":
        return "the default session"
    # Use first 8 chars of session ID
    short_id = session_id[:8] if len(session_id) > 8 else session_id
    return f"session {short_id}"
