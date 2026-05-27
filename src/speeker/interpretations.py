"""Interpretation cues for utterances.

An *interpretation* is a named label (SUCCESS, ERROR, or any custom name) that
an utterance can carry. Each interpretation maps to an *indication* — either a
sequence of musical notes or a sound file — that plays before the speech to
signal the outcome.

This module is pure: it resolves names to indications and converts a notes
indication into a synthesis spec. It does no audio I/O (that lives in
``player.py``), so it can be imported and tested without a TTS engine.
"""

import re

from .config import get_interpretations_config

# Built-in interpretations are always available, even when a user's config
# file defines its own ``map`` (the shallow config merge would otherwise
# replace the defaults entirely). Config entries override built-ins of the
# same name; see ``effective_map``.
BUILTIN_INTERPRETATIONS: dict[str, dict] = {
    "SUCCESS": {
        "type": "notes",
        "notes": [
            {"pitch": "Eb3", "seconds": 0.15},
            {"pitch": "G#3", "seconds": 0.9},
        ],
    },
    "ERROR": {
        "type": "notes",
        "notes": [
            {"pitch": "Eb4", "seconds": 0.3},
            {"pitch": "D4", "seconds": 0.2},
            {"pitch": "Bb2", "seconds": 0.2},
            {"pitch": "Bb2", "seconds": 0.2},
        ],
    },
}

# Pitch like "Eb3", "G#3", "Bb2", "D4": letter, optional accidental, octave.
_PITCH_RE = re.compile(r"^([A-Ga-g])([b#]?)([0-8])$")

DEFAULT_PAUSE_AFTER_SECONDS = 0.3
DEFAULT_NOTE_SECONDS = 0.2


def effective_map() -> dict[str, dict]:
    """Built-in interpretations overlaid with the config ``map``."""
    config_map = get_interpretations_config().get("map") or {}
    return {**BUILTIN_INTERPRETATIONS, **config_map}


def interpretation_names() -> list[str]:
    """Sorted list of all valid interpretation names."""
    return sorted(effective_map().keys())


def resolve_interpretation(name: str | None) -> dict | None:
    """Resolve an interpretation name to its indication, or None if unknown.

    Matching is exact first, then case-insensitive (so ``success`` resolves to
    the built-in ``SUCCESS``)."""
    if not name:
        return None
    mapping = effective_map()
    if name in mapping:
        return mapping[name]
    lowered = {key.lower(): value for key, value in mapping.items()}
    return lowered.get(name.lower())


def is_valid_interpretation(name: str | None) -> bool:
    """True if ``name`` resolves to a known interpretation."""
    return resolve_interpretation(name) is not None


def pause_after_seconds() -> float:
    """Seconds to pause after a cue before the utterance speaks."""
    value = get_interpretations_config().get(
        "pause_after_seconds", DEFAULT_PAUSE_AFTER_SECONDS
    )
    try:
        return float(value)
    except (TypeError, ValueError):
        return DEFAULT_PAUSE_AFTER_SECONDS


def parse_pitch(pitch: str) -> tuple[str, int] | None:
    """Parse a pitch like ``"Eb3"`` into ``("eb", 3)`` for the tones library.

    Returns None for malformed pitches. The note name is lowercased with its
    accidental preserved (``"G#3"`` -> ``("g#", 3)``)."""
    match = _PITCH_RE.match(pitch.strip())
    if not match:
        return None
    note = match.group(1).lower() + match.group(2)
    octave = int(match.group(3))
    return note, octave


def notes_to_spec(indication: dict) -> list[tuple[str, int, float]]:
    """Convert a ``notes`` indication into ``[(note, octave, seconds), ...]``.

    Notes with malformed pitches are skipped so one bad entry never aborts the
    whole cue."""
    spec: list[tuple[str, int, float]] = []
    for note in indication.get("notes", []) or []:
        parsed = parse_pitch(str(note.get("pitch", "")))
        if parsed is None:
            continue
        try:
            seconds = float(note.get("seconds", DEFAULT_NOTE_SECONDS))
        except (TypeError, ValueError):
            seconds = DEFAULT_NOTE_SECONDS
        spec.append((parsed[0], parsed[1], seconds))
    return spec
