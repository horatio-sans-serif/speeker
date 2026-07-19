"""Resolve a background-music track for an utterance by (queue, interpretation).

Mirrors ``tone_rules`` resolution: a rule scores +2 for a matching queue and +1
for a matching interpretation; the most specific match wins; no match means no
music. Reuses the tone-rules match helpers so queue/interpretation semantics
(regex, case-insensitivity) stay identical across the two features.
"""

import os
from pathlib import Path

from .config import get_music_rules
from .tone_rules import _queue_match, _interp_match


def resolve_music_track(queue: str | None, interpretation: str | None) -> Path | None:
    """Return the best-matching rule's track path, or None for no music.

    A rule whose ``track`` is empty or whose file doesn't exist is treated as
    non-matching, so a half-typed UI row never silences or breaks the bed.
    """
    best_score = -1
    best_track: Path | None = None

    for rule in get_music_rules():
        if not isinstance(rule, dict):
            continue

        track = rule.get("track")
        if not isinstance(track, str) or not track.strip():
            continue
        path = Path(os.path.expanduser(track.strip()))
        if not path.exists():
            continue

        rule_queue = rule.get("queue")
        rule_queue = rule_queue.strip() or None if isinstance(rule_queue, str) else None
        rule_is_regex = bool(rule.get("queue_regex"))

        rule_interp = rule.get("interpretation")
        rule_interp = rule_interp.strip() or None if isinstance(rule_interp, str) else None

        if not _queue_match(rule_queue, rule_is_regex, queue):
            continue
        if not _interp_match(rule_interp, interpretation):
            continue

        score = (2 if rule_queue is not None else 0) + (1 if rule_interp is not None else 0)
        if score > best_score:
            best_score = score
            best_track = path

    return best_track
