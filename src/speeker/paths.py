"""Centralized path management for Speeker using XDG conventions.

All path functions (not constants) so SPEEKER_DIR is checked at call time.
When SPEEKER_DIR is set, all subdirectories live under it.
Otherwise, platformdirs determines OS-appropriate locations.
"""

import os
from pathlib import Path

from platformdirs import user_cache_dir, user_data_dir, user_runtime_dir

_APP_NAME = "speeker"


def _override_root() -> Path | None:
    """Return the SPEEKER_DIR override path, or None."""
    val = os.environ.get("SPEEKER_DIR")
    return Path(val) if val else None


# -- Config ------------------------------------------------------------------

def config_dir() -> Path:
    """Config directory (config.json, voice-prefs.json)."""
    root = _override_root()
    if root:
        return root / "config"
    # On macOS platformdirs returns ~/Library/Application Support/speeker for
    # both user_data_dir and user_config_dir (which is correct).  We share
    # the data dir for config since macOS has no separate XDG config dir.
    return Path(user_data_dir(_APP_NAME))


def config_file() -> Path:
    return config_dir() / "config.json"


def voice_prefs_file() -> Path:
    return config_dir() / "voice-prefs.json"


# -- Data --------------------------------------------------------------------

def data_dir() -> Path:
    """Persistent data (queue.db, audio/, voices/)."""
    root = _override_root()
    if root:
        return root / "data"
    return Path(user_data_dir(_APP_NAME))


def db_path() -> Path:
    return data_dir() / "queue.db"


def audio_dir() -> Path:
    return data_dir() / "audio"


def voices_dir() -> Path:
    return data_dir() / "voices"


def voices_manifest() -> Path:
    return voices_dir() / "manifest.json"


# -- Cache -------------------------------------------------------------------

def cache_dir() -> Path:
    """Expendable cached files (tones/, voice-samples/)."""
    root = _override_root()
    if root:
        return root / "cache"
    return Path(user_cache_dir(_APP_NAME))


def tones_dir() -> Path:
    return cache_dir() / "tones"


def voice_samples_dir() -> Path:
    return cache_dir() / "voice-samples"


def tone_intro_path() -> Path:
    return cache_dir() / "tone_intro.wav"


def tone_outro_path() -> Path:
    return cache_dir() / "tone_outro.wav"


# -- Runtime -----------------------------------------------------------------

def runtime_dir() -> Path:
    """Ephemeral runtime files (player.lock)."""
    root = _override_root()
    if root:
        return root
    return Path(user_runtime_dir(_APP_NAME))


def player_lock_path() -> Path:
    return runtime_dir() / "player.lock"


# -- Helpers -----------------------------------------------------------------

def ensure_dir(path: Path) -> Path:
    """Create *path* (and parents) if missing, then return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path
