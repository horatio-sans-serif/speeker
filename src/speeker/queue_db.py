"""SQLite-based queue for TTS playback with metadata support."""

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import numpy as np

from .config import is_semantic_search_enabled, get_embedding_model, get_embedding_cache_dir

# Default database location
DEFAULT_DB_PATH = Path.home() / ".speeker" / "queue.db"

# Lazy-loaded embedding model
_embedding_model = None
_embedding_lock = threading.Lock()

# Thread-local storage for connections
_local = threading.local()


def get_db_path() -> Path:
    """Get the database path, creating parent directory if needed."""
    db_path = DEFAULT_DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    """Get a thread-local database connection with proper locking."""
    if not hasattr(_local, "conn") or _local.conn is None:
        db_path = get_db_path()
        _local.conn = sqlite3.connect(
            str(db_path),
            check_same_thread=False,
            timeout=30.0,  # Wait up to 30s for locks
        )
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")  # Better concurrency
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
            played_at TEXT,
            metadata TEXT
        )
    """)
    # Migration: add metadata column if missing (for existing databases)
    try:
        conn.execute("ALTER TABLE queue ADD COLUMN metadata TEXT")
    except sqlite3.OperationalError:
        pass  # Column already exists
    conn.execute("""
        CREATE TABLE IF NOT EXISTS playback_state (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            last_utterance_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            session_id TEXT PRIMARY KEY,
            intro_sound INTEGER DEFAULT 1,
            speed REAL DEFAULT 1.0,
            voice TEXT DEFAULT NULL,
            engine TEXT DEFAULT NULL
        )
    """)
    # Migration: add engine column if missing
    try:
        conn.execute("ALTER TABLE settings ADD COLUMN engine TEXT DEFAULT NULL")
    except sqlite3.OperationalError:
        pass  # Column already exists
    conn.execute("CREATE INDEX IF NOT EXISTS idx_queue_session ON queue(session_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_queue_played ON queue(played_at)")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            queue_id INTEGER PRIMARY KEY,
            embedding BLOB NOT NULL,
            FOREIGN KEY (queue_id) REFERENCES queue(id) ON DELETE CASCADE
        )
    """)
    conn.commit()

    # Ensure global defaults exist (use a separate try/except to avoid lock issues)
    try:
        conn.execute("""
            INSERT OR IGNORE INTO settings (session_id, intro_sound, speed, voice, engine)
            VALUES ('__global__', 1, 1.0, NULL, NULL)
        """)
        conn.commit()
    except sqlite3.OperationalError:
        pass  # Already exists or locked, that's fine


def _get_embedding_model():
    """Lazy-load the embedding model."""
    global _embedding_model
    with _embedding_lock:
        if _embedding_model is None:
            from sentence_transformers import SentenceTransformer
            model_name = get_embedding_model()
            cache_dir = get_embedding_cache_dir()
            _embedding_model = SentenceTransformer(model_name, cache_folder=cache_dir)
        return _embedding_model


def _generate_embedding(text: str) -> bytes | None:
    """Generate embedding for text if semantic search is enabled."""
    if not is_semantic_search_enabled():
        return None
    try:
        model = _get_embedding_model()
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.astype(np.float32).tobytes()
    except Exception:
        return None


def _store_embedding(conn: sqlite3.Connection, queue_id: int, embedding: bytes) -> None:
    """Store embedding for a queue item."""
    conn.execute(
        "INSERT OR REPLACE INTO embeddings (queue_id, embedding) VALUES (?, ?)",
        (queue_id, embedding)
    )


def enqueue(
    text: str,
    metadata: dict | None = None,
    audio_path: str | Path | None = None,
    session_id: str | None = None,  # Deprecated, use metadata instead
) -> int:
    """Add an item to the queue. Returns the item ID.

    Args:
        text: The text to queue for TTS
        metadata: Optional dict of key-value pairs to store with the item
        audio_path: Optional path to pre-generated audio file
        session_id: Deprecated - use metadata={'session': ...} instead
    """
    # Handle legacy session_id parameter
    if session_id and not metadata:
        metadata = {"session": session_id}
    elif session_id and metadata and "session" not in metadata:
        metadata["session"] = session_id

    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO queue (session_id, text, audio_path, created_at, metadata)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                metadata.get("session", "default") if metadata else "default",
                text,
                str(audio_path) if audio_path else None,
                datetime.now(timezone.utc).isoformat(),
                json.dumps(metadata) if metadata else None,
            ),
        )
        queue_id = cursor.lastrowid or 0

        # Generate and store embedding if enabled
        embedding = _generate_embedding(text)
        if embedding:
            _store_embedding(conn, queue_id, embedding)

        conn.commit()
        return queue_id


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


# --- Settings ---

def get_settings(session_id: str | None = None) -> dict:
    """Get settings for a session, with global defaults as fallback.

    Returns dict with: intro_sound (bool), speed (float), voice (str), engine (str)

    Settings cascade (highest to lowest priority):
    1. Session-specific settings (from database)
    2. Global settings (from database)
    3. User voice preferences (from ~/.config/speeker/voice-prefs.json)
    4. System defaults (pocket-tts, azelma)
    """
    # Import here to avoid circular import
    from .voice_prefs import get_preferred_voice, get_preferred_engine
    from .voices import DEFAULT_POCKET_TTS_VOICE, DEFAULT_ENGINE

    # Start with system defaults, then overlay user preferences
    preferred_engine = get_preferred_engine() or DEFAULT_ENGINE
    preferred_voice = get_preferred_voice(preferred_engine) or DEFAULT_POCKET_TTS_VOICE

    settings = {
        "intro_sound": True,
        "speed": 1.0,
        "voice": preferred_voice,
        "engine": preferred_engine,
    }

    with get_connection() as conn:
        # Get global settings (overlay on preferences)
        cursor = conn.execute(
            "SELECT intro_sound, speed, voice, engine FROM settings WHERE session_id = '__global__'"
        )
        row = cursor.fetchone()
        if row:
            if row["intro_sound"] is not None:
                settings["intro_sound"] = bool(row["intro_sound"])
            if row["speed"] is not None:
                settings["speed"] = float(row["speed"])
            if row["voice"] is not None:
                settings["voice"] = row["voice"]
            if row["engine"] is not None:
                settings["engine"] = row["engine"]

        # Override with session-specific settings if they exist
        if session_id and session_id != "__global__":
            cursor = conn.execute(
                "SELECT intro_sound, speed, voice, engine FROM settings WHERE session_id = ?",
                (session_id,)
            )
            row = cursor.fetchone()
            if row:
                if row["intro_sound"] is not None:
                    settings["intro_sound"] = bool(row["intro_sound"])
                if row["speed"] is not None:
                    settings["speed"] = float(row["speed"])
                if row["voice"] is not None:
                    settings["voice"] = row["voice"]
                if row["engine"] is not None:
                    settings["engine"] = row["engine"]

        return settings


def set_settings(
    session_id: str | None = None,
    intro_sound: bool | None = None,
    speed: float | None = None,
    voice: str | None = None,
    engine: str | None = None,
) -> None:
    """Set settings for a session (or global if session_id is None)."""
    target = session_id or "__global__"

    with get_connection() as conn:
        # Check if row exists
        cursor = conn.execute(
            "SELECT 1 FROM settings WHERE session_id = ?", (target,)
        )
        exists = cursor.fetchone() is not None

        if exists:
            # Update existing
            updates = []
            values = []
            if intro_sound is not None:
                updates.append("intro_sound = ?")
                values.append(int(intro_sound))
            if speed is not None:
                updates.append("speed = ?")
                values.append(speed)
            if voice is not None:
                updates.append("voice = ?")
                values.append(voice)
            if engine is not None:
                updates.append("engine = ?")
                values.append(engine)

            if updates:
                values.append(target)
                conn.execute(
                    f"UPDATE settings SET {', '.join(updates)} WHERE session_id = ?",
                    values
                )
        else:
            # Insert new
            conn.execute(
                """
                INSERT INTO settings (session_id, intro_sound, speed, voice, engine)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    target,
                    int(intro_sound) if intro_sound is not None else 1,
                    speed if speed is not None else 1.0,
                    voice,
                    engine,
                )
            )

        conn.commit()


def get_all_sessions() -> list[dict]:
    """Get all sessions with their message counts and last activity."""
    with get_connection() as conn:
        cursor = conn.execute("""
            SELECT
                session_id,
                COUNT(*) as total_messages,
                SUM(CASE WHEN played_at IS NULL THEN 1 ELSE 0 END) as pending,
                MAX(created_at) as last_activity
            FROM queue
            GROUP BY session_id
            ORDER BY last_activity DESC
        """)
        return [dict(row) for row in cursor.fetchall()]


def get_history(session_id: str | None = None, limit: int = 100) -> list[dict]:
    """Get message history, optionally filtered by session."""
    with get_connection() as conn:
        if session_id:
            cursor = conn.execute(
                """
                SELECT id, session_id, text, audio_path, created_at, played_at, metadata
                FROM queue
                WHERE session_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (session_id, limit)
            )
        else:
            cursor = conn.execute(
                """
                SELECT id, session_id, text, audio_path, created_at, played_at, metadata
                FROM queue
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,)
            )
        results = []
        for row in cursor.fetchall():
            item = dict(row)
            # Parse metadata JSON
            if item.get("metadata"):
                try:
                    item["metadata"] = json.loads(item["metadata"])
                except (json.JSONDecodeError, TypeError):
                    item["metadata"] = None
            results.append(item)
        return results


# --- Search ---

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def search_semantic(query: str, limit: int = 50) -> list[dict]:
    """Search using semantic similarity. Returns items sorted by relevance."""
    if not is_semantic_search_enabled():
        return []

    query_embedding = _generate_embedding(query)
    if not query_embedding:
        return []

    query_vec = np.frombuffer(query_embedding, dtype=np.float32)

    with get_connection() as conn:
        # Get all items with embeddings
        cursor = conn.execute("""
            SELECT q.id, q.session_id, q.text, q.audio_path, q.created_at, q.played_at, q.metadata, e.embedding
            FROM queue q
            JOIN embeddings e ON q.id = e.queue_id
        """)

        results = []
        for row in cursor.fetchall():
            item_vec = np.frombuffer(row["embedding"], dtype=np.float32)
            similarity = _cosine_similarity(query_vec, item_vec)
            metadata = None
            if row["metadata"]:
                try:
                    metadata = json.loads(row["metadata"])
                except (json.JSONDecodeError, TypeError):
                    pass
            results.append({
                "id": row["id"],
                "session_id": row["session_id"],
                "text": row["text"],
                "audio_path": row["audio_path"],
                "created_at": row["created_at"],
                "played_at": row["played_at"],
                "metadata": metadata,
                "score": similarity,
            })

        # Sort by similarity descending
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]


def search_fuzzy(query: str, limit: int = 50) -> list[dict]:
    """Search using fuzzy text matching on text and metadata values."""
    query_lower = query.lower()
    query_parts = query_lower.split()

    with get_connection() as conn:
        cursor = conn.execute("""
            SELECT id, session_id, text, audio_path, created_at, played_at, metadata
            FROM queue
            ORDER BY created_at DESC
        """)

        results = []
        for row in cursor.fetchall():
            text_lower = row["text"].lower()

            # Parse metadata
            metadata = None
            metadata_text = ""
            if row["metadata"]:
                try:
                    metadata = json.loads(row["metadata"])
                    # Concatenate all metadata values for searching
                    metadata_text = " ".join(str(v).lower() for v in metadata.values())
                except (json.JSONDecodeError, TypeError):
                    pass

            # Score based on matches
            score = 0.0

            # Exact substring match in text
            if query_lower in text_lower:
                score += 1.0

            # Exact substring match in metadata values
            if query_lower in metadata_text:
                score += 0.5

            # Partial word matches
            for part in query_parts:
                if part in text_lower:
                    score += 0.3
                if part in metadata_text:
                    score += 0.2

            if score > 0:
                results.append({
                    "id": row["id"],
                    "session_id": row["session_id"],
                    "text": row["text"],
                    "audio_path": row["audio_path"],
                    "created_at": row["created_at"],
                    "played_at": row["played_at"],
                    "metadata": metadata,
                    "score": score,
                })

        # Sort by score descending, then by created_at descending
        results.sort(key=lambda x: (-x["score"], x["created_at"]), reverse=False)
        return results[:limit]


def search(query: str, limit: int = 50) -> list[dict]:
    """Search queue history. Uses semantic search if enabled, else fuzzy search."""
    if is_semantic_search_enabled():
        return search_semantic(query, limit)
    return search_fuzzy(query, limit)
