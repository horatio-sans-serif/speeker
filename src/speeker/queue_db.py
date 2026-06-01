"""SQLite-based queue for TTS playback with metadata support."""

import hashlib
import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import numpy as np

from .config import is_semantic_search_enabled, get_embedding_model, get_embedding_cache_dir
from .paths import db_path as _db_path, ensure_dir

# Lazy-loaded embedding model
_embedding_model = None
_embedding_lock = threading.Lock()

# Thread-local storage for connections
_local = threading.local()


def get_db_path() -> Path:
    """Get the database path, creating parent directory if needed."""
    p = _db_path()
    ensure_dir(p.parent)
    return p


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
            last_utterance_at TEXT,
            last_queue_id TEXT,
            currently_playing_id INTEGER,
            currently_playing_started_at TEXT
        )
    """)
    # Migration: add columns on pre-existing databases.
    # - last_queue_id: detects queue context switches between bursts.
    # - currently_playing_id / currently_playing_started_at: the side channel
    #   the player uses to tell the web UI "this item is being spoken NOW".
    #   started_at lets the API age-out a stale value after a daemon crash.
    for ddl in (
        "ALTER TABLE playback_state ADD COLUMN last_queue_id TEXT",
        "ALTER TABLE playback_state ADD COLUMN currently_playing_id INTEGER",
        "ALTER TABLE playback_state ADD COLUMN currently_playing_started_at TEXT",
    ):
        try:
            conn.execute(ddl)
        except sqlite3.OperationalError:
            pass  # Column already exists
    conn.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            session_id TEXT PRIMARY KEY,
            intro_sound INTEGER DEFAULT 1,
            speed REAL DEFAULT 1.0,
            voice TEXT DEFAULT NULL,
            engine TEXT DEFAULT NULL,
            color TEXT DEFAULT NULL,
            effects_preset TEXT DEFAULT NULL
        )
    """)
    # Migration: add columns to pre-existing databases. Each ALTER is
    # in its own try/except because sqlite3 raises OperationalError
    # immediately on the first column that already exists, aborting the
    # rest of the migration if they're grouped.
    for ddl in (
        "ALTER TABLE settings ADD COLUMN engine TEXT DEFAULT NULL",
        # Per-queue accent color, used by the UI for card stripes / row
        # accent / queue-picker highlight. NULL falls back to a stable
        # auto-derived color (see ``default_color_for_queue``).
        "ALTER TABLE settings ADD COLUMN color TEXT DEFAULT NULL",
        # Per-queue effects preset override. NULL falls back to the
        # global ``effects.preset`` config value at speech time.
        "ALTER TABLE settings ADD COLUMN effects_preset TEXT DEFAULT NULL",
    ):
        try:
            conn.execute(ddl)
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
        session_id: Deprecated - use metadata={'queue': ...} instead
    """
    # Handle legacy session_id parameter (maps to queue)
    if session_id and not metadata:
        metadata = {"queue": session_id}
    elif session_id and metadata and "queue" not in metadata:
        metadata["queue"] = session_id

    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO queue (session_id, text, audio_path, created_at, metadata)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                metadata.get("queue", "default") if metadata else "default",
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
            SELECT id, session_id, text, audio_path, created_at, metadata
            FROM queue
            WHERE session_id = ? AND played_at IS NULL
            ORDER BY created_at ASC
            """,
            (session_id,),
        )
        items = []
        for row in cursor.fetchall():
            item = dict(row)
            if item.get("metadata"):
                try:
                    item["metadata"] = json.loads(item["metadata"])
                except (json.JSONDecodeError, TypeError):
                    item["metadata"] = None
            items.append(item)
        return items


def mark_played(item_id: int) -> None:
    """Mark an item as played."""
    with get_connection() as conn:
        conn.execute(
            "UPDATE queue SET played_at = ? WHERE id = ?",
            (datetime.now(timezone.utc).isoformat(), item_id),
        )
        conn.commit()


def update_metadata(item_id: int, patches: dict) -> dict:
    """Merge ``patches`` into the item's metadata JSON column.

    Returns the merged metadata dict. Existing keys are overwritten by
    ``patches`` (shallow merge). Used by the daemon to record
    ``tts_attempts`` and ``tts_error`` on failed utterances without
    disturbing the rest of metadata (queue, engine, voice, ...).
    """
    with get_connection() as conn:
        cursor = conn.execute(
            "SELECT metadata FROM queue WHERE id = ?",
            (item_id,),
        )
        row = cursor.fetchone()
        current: dict = {}
        if row and row["metadata"]:
            try:
                current = json.loads(row["metadata"]) or {}
            except (json.JSONDecodeError, TypeError):
                current = {}
        merged = {**current, **patches}
        conn.execute(
            "UPDATE queue SET metadata = ? WHERE id = ?",
            (json.dumps(merged), item_id),
        )
        conn.commit()
    return merged


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


def set_last_utterance_time(queue_id: str | None = None) -> None:
    """Update the last utterance time (and optionally the last queue id) to now.

    Passing ``queue_id`` also records *which* queue was last spoken; the player
    uses both values to decide whether to auto-prepend a queue title before
    the next single message (see ``get_auto_label_config`` and
    ``process_queue``). When ``queue_id`` is ``None`` the existing value is
    preserved.
    """
    now = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        if queue_id is None:
            conn.execute(
                """
                INSERT INTO playback_state (id, last_utterance_at) VALUES (1, ?)
                ON CONFLICT(id) DO UPDATE SET last_utterance_at = excluded.last_utterance_at
                """,
                (now,),
            )
        else:
            conn.execute(
                """
                INSERT INTO playback_state (id, last_utterance_at, last_queue_id)
                VALUES (1, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    last_utterance_at = excluded.last_utterance_at,
                    last_queue_id = excluded.last_queue_id
                """,
                (now, queue_id),
            )
        conn.commit()


def get_last_played_queue() -> str | None:
    """Return the queue id of the most recent utterance, or None if unknown."""
    with get_connection() as conn:
        cursor = conn.execute(
            "SELECT last_queue_id FROM playback_state WHERE id = 1"
        )
        row = cursor.fetchone()
        if row and row["last_queue_id"]:
            return row["last_queue_id"]
        return None


# Stale "currently_playing" entries time out after this many seconds. Real
# utterances rarely exceed ~30s; anything older is presumed orphaned by a
# daemon crash so the UI doesn't permanently highlight a phantom card.
_CURRENTLY_PLAYING_STALE_AFTER_SECONDS = 90


def set_currently_playing(item_id: int) -> None:
    """Mark *item_id* as the currently-playing queue row.

    Called by the player daemon immediately before speaking. The web UI
    polls and decorates the matching card with a "speaking" background --
    so the listener can see which card is being read aloud right now.
    """
    now = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO playback_state (id, currently_playing_id, currently_playing_started_at)
            VALUES (1, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                currently_playing_id = excluded.currently_playing_id,
                currently_playing_started_at = excluded.currently_playing_started_at
            """,
            (int(item_id), now),
        )
        conn.commit()


def clear_currently_playing() -> None:
    """Clear the currently-playing marker (called when an utterance ends)."""
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE playback_state
            SET currently_playing_id = NULL,
                currently_playing_started_at = NULL
            WHERE id = 1
            """,
        )
        conn.commit()


def get_currently_playing() -> int | None:
    """Return the item id being spoken NOW, or None.

    Ages out a stale value: if the started_at timestamp is older than
    ``_CURRENTLY_PLAYING_STALE_AFTER_SECONDS``, return None (and don't
    re-clear -- the next ``set_currently_playing`` will overwrite it).
    """
    with get_connection() as conn:
        cursor = conn.execute(
            """
            SELECT currently_playing_id, currently_playing_started_at
            FROM playback_state WHERE id = 1
            """
        )
        row = cursor.fetchone()
        if not row or row["currently_playing_id"] is None:
            return None
        started_at = row["currently_playing_started_at"]
        if started_at:
            try:
                started = datetime.fromisoformat(started_at)
                if started.tzinfo is None:
                    started = started.replace(tzinfo=timezone.utc)
                age = (datetime.now(timezone.utc) - started).total_seconds()
                if age > _CURRENTLY_PLAYING_STALE_AFTER_SECONDS:
                    return None
            except ValueError:
                pass  # Bad timestamp -> trust the id anyway, the cap above will catch real staleness later
        return int(row["currently_playing_id"])


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


def get_queue_label(queue_id: str | None) -> str:
    """Get a human-friendly label for a queue ID.

    Returns the legacy "queue <8-char-prefix>" form used by the multi-message
    "For queue X, there are N messages" header. For the spoken title used by
    auto-labeling (e.g., 'compass docs'), use ``get_spoken_queue_title``.
    """
    if not queue_id or queue_id == "default":
        return "the default queue"
    # Use first 8 chars of queue ID
    short_id = queue_id[:8] if len(queue_id) > 8 else queue_id
    return f"queue {short_id}"


def get_spoken_queue_title(queue_id: str | None) -> str | None:
    """Friendly spoken title for a queue id, or None when no title applies.

    Returns ``None`` for the unnamed/default queue (so the auto-label path
    skips adding a meaningless prefix). For named queues, hyphens and
    underscores become spaces — matching the convention the Stop hook's
    ``spoken_title`` already uses — so 'compass-docs' speaks as 'compass docs'.
    """
    if not queue_id or queue_id == "default":
        return None
    title = queue_id.replace("-", " ").replace("_", " ").strip()
    return title or None


# --- Settings ---

def default_color_for_queue(session_id: str | None) -> str:
    """Stable accent color derived from a queue id, for queues that haven't
    set an explicit color. Uses an HSL palette tuned to be readable on both
    light and dark surfaces (saturation 65%, lightness 55% -- bright enough
    on dark, dim enough on light). Returns a CSS hex color.
    """
    if not session_id:
        # __global__ / unnamed queue -- neutral accent.
        return "#7a8694"
    # 12-step golden-ratio hue ladder so adjacent queues alphabetically
    # don't always look similar. md5 keeps the mapping stable across runs.
    h = int(hashlib.md5(session_id.encode("utf-8")).hexdigest()[:8], 16)
    hue = (h * 137) % 360  # golden-angle spacing for good separation
    return _hsl_to_hex(hue, 0.55, 0.55)


def _hsl_to_hex(h: float, s: float, l: float) -> str:
    """HSL (h in degrees, s/l in 0..1) -> #rrggbb."""
    import colorsys
    r, g, b = colorsys.hls_to_rgb(h / 360.0, l, s)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def get_settings(session_id: str | None = None) -> dict:
    """Get settings for a session, with global defaults as fallback.

    Returns dict with: intro_sound (bool), speed (float), voice (str),
    engine (str), color (str), effects_preset (str | None).

    ``color`` is always populated -- explicit setting first, otherwise a
    stable auto-derived color from the queue id (see
    ``default_color_for_queue``). ``effects_preset`` is None when not set
    so callers can distinguish "use global default" from "explicitly set
    to off".

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
        "color": default_color_for_queue(session_id),
        "effects_preset": None,
    }

    explicit_color: str | None = None

    with get_connection() as conn:
        # Get global settings (overlay on preferences)
        cursor = conn.execute(
            "SELECT intro_sound, speed, voice, engine, color, effects_preset"
            " FROM settings WHERE session_id = '__global__'"
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
            # color / effects_preset on __global__ are odd but harmless:
            # they'd only apply if no per-queue value were set AND the
            # caller asked for __global__.

        # Override with session-specific settings if they exist
        if session_id and session_id != "__global__":
            cursor = conn.execute(
                "SELECT intro_sound, speed, voice, engine, color, effects_preset"
                " FROM settings WHERE session_id = ?",
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
                if row["color"] is not None and str(row["color"]).strip():
                    explicit_color = str(row["color"]).strip()
                if row["effects_preset"] is not None and str(row["effects_preset"]).strip():
                    settings["effects_preset"] = str(row["effects_preset"]).strip()

        if explicit_color:
            settings["color"] = explicit_color
        return settings


def set_settings(
    session_id: str | None = None,
    intro_sound: bool | None = None,
    speed: float | None = None,
    voice: str | None = None,
    engine: str | None = None,
    color: str | None = None,
    effects_preset: str | None = None,
) -> None:
    """Set settings for a session (or global if session_id is None).

    Only fields with a non-None value are written; existing values are
    left untouched. To clear a per-queue ``color`` or ``effects_preset``
    (so it falls back to the auto-derived color / global preset), pass
    the empty string ``""`` -- this is normalized to NULL.
    """
    target = session_id or "__global__"
    # Normalize empty strings to None so the "clear override" UX has a
    # single representation in the DB (NULL = "fall back").
    if isinstance(color, str) and not color.strip():
        color = ""  # sentinel for explicit clear -- handled below as NULL.
    if isinstance(effects_preset, str) and not effects_preset.strip():
        effects_preset = ""

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
            if color is not None:
                # Empty string -> NULL (clear the override).
                updates.append("color = ?")
                values.append(color if color else None)
            if effects_preset is not None:
                updates.append("effects_preset = ?")
                values.append(effects_preset if effects_preset else None)

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
                INSERT INTO settings (session_id, intro_sound, speed, voice, engine, color, effects_preset)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    target,
                    int(intro_sound) if intro_sound is not None else 1,
                    speed if speed is not None else 1.0,
                    voice,
                    engine,
                    color if color else None,
                    effects_preset if effects_preset else None,
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
