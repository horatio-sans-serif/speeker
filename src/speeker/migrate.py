"""Auto-migrate legacy ~/.speeker/ and ~/.config/speeker/ to XDG paths.

Runs once (marker file .migrated_v2 in data_dir).  Skipped when SPEEKER_DIR
is set.  Moves files; does not overwrite existing destinations.
"""

import logging
import os
import shutil
from pathlib import Path

from . import paths

log = logging.getLogger(__name__)

_LEGACY_BASE = Path.home() / ".speeker"
_LEGACY_CONFIG = Path.home() / ".config" / "speeker"


def _marker_path() -> Path:
    return paths.data_dir() / ".migrated_v2"


def _needs_migration() -> bool:
    if os.environ.get("SPEEKER_DIR"):
        return False
    if _marker_path().exists():
        return False
    return _LEGACY_BASE.exists() or _LEGACY_CONFIG.exists()


def _move(src: Path, dst: Path) -> None:
    """Move *src* to *dst* unless *dst* already exists."""
    if not src.exists():
        return
    if dst.exists():
        log.debug("skip (exists): %s -> %s", src, dst)
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    log.debug("moved: %s -> %s", src, dst)


def _migrate_data() -> None:
    """Move queue.db, audio date-dirs, and voices/."""
    data = paths.data_dir()
    data.mkdir(parents=True, exist_ok=True)

    # queue.db
    _move(_LEGACY_BASE / "queue.db", data / "queue.db")

    # audio date directories (YYYY-MM-DD)
    if _LEGACY_BASE.exists():
        audio_dst = paths.audio_dir()
        for child in _LEGACY_BASE.iterdir():
            if child.is_dir() and len(child.name) == 10 and child.name[4] == "-":
                _move(child, audio_dst / child.name)

        # audio/ subdir (newer layout already used audio/)
        legacy_audio = _LEGACY_BASE / "audio"
        if legacy_audio.is_dir():
            for child in legacy_audio.iterdir():
                if child.is_dir():
                    _move(child, audio_dst / child.name)

    # voices/  (was under var/voices in voice_clone.py)
    _move(_LEGACY_BASE / "var" / "voices", paths.voices_dir())


def _migrate_cache() -> None:
    """Move tones/, voice-samples/, tone wav files."""
    _move(_LEGACY_BASE / "tones", paths.tones_dir())

    # tone intro/outro were .tone_intro.wav / .tone_outro.wav
    _move(_LEGACY_BASE / ".tone_intro.wav", paths.tone_intro_path())
    _move(_LEGACY_BASE / ".tone_outro.wav", paths.tone_outro_path())

    # voice-samples lived under ~/.config/speeker/voice-samples
    _move(_LEGACY_CONFIG / "voice-samples", paths.voice_samples_dir())


def _migrate_config() -> None:
    """Move config.json and voice-prefs.json."""
    cfg = paths.config_dir()
    cfg.mkdir(parents=True, exist_ok=True)

    _move(_LEGACY_CONFIG / "config.json", cfg / "config.json")
    _move(_LEGACY_CONFIG / "voice-prefs.json", cfg / "voice-prefs.json")


def _cleanup_stale() -> None:
    """Remove stale lock file from legacy location."""
    lock = _LEGACY_BASE / ".player.lock"
    if lock.exists():
        lock.unlink(missing_ok=True)


def migrate() -> None:
    """Run one-time migration if needed.  Safe to call on every startup."""
    if not _needs_migration():
        return

    log.info("Migrating legacy paths to XDG layout...")

    try:
        _migrate_data()
        _migrate_cache()
        _migrate_config()
        _cleanup_stale()
    except Exception:
        log.warning("Migration encountered errors; will retry next launch", exc_info=True)
        return

    # Write marker so we don't run again
    marker = _marker_path()
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("migrated")
    log.info("Migration complete.")
