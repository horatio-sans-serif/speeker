"""macOS Focus detection: read the Do Not Disturb / Focus assertions file.

A macOS Focus (Do Not Disturb, Reduce Interruptions, Sleep, custom, ...) writes
an active assertion to ``~/Library/DoNotDisturb/DB/Assertions.json`` while it's
on. We read that file to optionally pause speeker, mirroring the call-pause path.

Lightweight (json only) so the player can import it without heavy deps. macOS
only; on other platforms there is no Focus and detection returns nothing.
"""

import json
from pathlib import Path

from .config import get_focus_config

_ASSERTIONS = Path("~/Library/DoNotDisturb/DB/Assertions.json").expanduser()


def active_focus_modes() -> list[str]:
    """Identifiers of currently-active Focus modes (empty if none / unsupported).

    Active assertions live under ``data[].storeAssertionRecords`` (ended ones
    are in the ``...Invalidation...`` lists, which we ignore). Each carries an
    ``assertionDetailsModeIdentifier`` like ``com.apple.focus.reduce-interruptions``
    or ``com.apple.donotdisturb.mode.default``.
    """
    try:
        data = json.loads(_ASSERTIONS.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    modes: list[str] = []
    for block in data.get("data", []) or []:
        for rec in block.get("storeAssertionRecords") or []:
            mode = (rec.get("assertionDetails") or {}).get("assertionDetailsModeIdentifier")
            if mode:
                modes.append(mode)
    return modes


def focus_status() -> str:
    """'active' (a Focus is on), 'idle' (none), or 'unavailable' (no file)."""
    if not _ASSERTIONS.exists():
        return "unavailable"
    return "active" if active_focus_modes() else "idle"


def should_pause_for_focus() -> bool:
    """True when pause-on-Focus is enabled and a matching Focus is active.

    ``focus.modes`` restricts to specific mode identifiers (case-insensitive
    substring); empty means any active Focus pauses. A missing/unreadable
    assertions file never pauses.
    """
    cfg = get_focus_config()
    if not cfg.get("pause_when_active"):
        return False
    modes = active_focus_modes()
    if not modes:
        return False
    wanted = [str(m).lower() for m in (cfg.get("modes") or []) if str(m).strip()]
    if not wanted:
        return True
    return any(any(w in mode.lower() for w in wanted) for mode in modes)
