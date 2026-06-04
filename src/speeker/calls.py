"""Call detection: read the monitor-active-calls state file.

Lightweight (json + config only) so both the player daemon and the web server
can import it without pulling in TTS engine dependencies.
"""

import json
from pathlib import Path

from .config import get_calls_config


def call_status() -> str:
    """Return 'active', 'idle', or 'unavailable'.

    'unavailable' means the monitor-active-calls state file is absent or
    unreadable (e.g. the daemon isn't installed), in which case callers should
    treat the feature as off.
    """
    path_str = get_calls_config().get("state_file", "")
    path = Path(path_str) if path_str else None
    if path is None or not path.exists():
        return "unavailable"
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return "unavailable"
    return "active" if data.get("active") else "idle"


def should_pause_for_call() -> bool:
    """True when pause-on-call is enabled in config and a call is active.

    A missing/unreadable state file never pauses, so installing the monitor is
    optional.
    """
    if not get_calls_config().get("pause_when_active"):
        return False
    return call_status() == "active"
