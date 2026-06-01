"""Per-queue / per-interpretation tune resolution.

Speeker plays three kinds of tones:

* ``intro`` -- a chord at the start of a batch.
* ``outro`` -- a chord at the end of a batch.
* ``cue``   -- the indication for an utterance's interpretation
  (SUCCESS / ERROR / custom).

Each of these has a global default (``tones.intro``, ``tones.outro``,
``interpretations.map.<name>``). The ``tone_rules`` config list lets users
override the notes for specific queues, interpretations, or combinations
without editing the global default.

A rule looks like::

    {
      "slot": "intro" | "outro" | "cue",
      "queue": "<queue id or regex>" | null,
      "queue_regex": false,
      "interpretation": "SUCCESS" | null,
      "notes": ["E4", "G4", "C5"],
    }

Resolution scores each candidate rule against the call context
(``current_queue``, ``current_interpretation``):

* +2 when the rule's ``queue`` matches the call's queue
* +1 when the rule's ``interpretation`` matches the call's interpretation

The highest-scoring matching rule wins; ties are broken by definition
order (first match within the highest tier). When no rule matches the
caller falls back to the global default.

This yields the user-stated hierarchy:

    queue + interpretation  (score 3)
        > queue only        (score 2)
        > interpretation    (score 1)
        > global default    (no rule)

The resolver is pure: it reads config and returns a notes list, doing no
I/O. ``player.py`` decides what to do with the result.
"""

from __future__ import annotations

import re

from .config import get_tone_rules

_VALID_SLOTS = frozenset({"intro", "outro", "cue"})

# Note notation accepted in rule.notes: ``[A-G][b#]?[0-8](:multiplier)?``.
# Mirrors ``web._NOTE_RE`` and ``player.parse_note_token`` -- if you
# change one, update them all.
_NOTE_RE = re.compile(r"^[A-Ga-g][b#]?[0-8](?::[0-9]*\.?[0-9]+)?$")


def _clean_notes(notes_raw) -> list[str]:
    if not isinstance(notes_raw, list):
        return []
    cleaned: list[str] = []
    for entry in notes_raw:
        if not isinstance(entry, str):
            continue
        token = entry.strip()
        if not token:
            continue
        if not _NOTE_RE.match(token):
            continue
        cleaned.append(token)
    return cleaned


def _queue_match(rule_queue: str | None, rule_is_regex: bool, current_queue: str | None) -> bool:
    """True when ``current_queue`` satisfies the rule's queue criterion.

    A rule with no ``queue`` criterion (``None``) trivially matches every
    call -- the caller is responsible for deciding whether such a rule
    deserves a queue-tier score bump (it doesn't).
    """
    if rule_queue is None:
        return True
    if current_queue is None:
        return False
    if rule_is_regex:
        try:
            return re.search(rule_queue, current_queue) is not None
        except re.error:
            return False
    return rule_queue == current_queue


def _interp_match(rule_interp: str | None, current_interp: str | None) -> bool:
    """True when ``current_interp`` satisfies the rule's interpretation criterion.

    Case-insensitive so ``success`` and ``SUCCESS`` are equivalent, matching
    ``interpretations.resolve_interpretation``.
    """
    if rule_interp is None:
        return True
    if current_interp is None:
        return False
    return rule_interp.strip().lower() == current_interp.strip().lower()


def resolve_tone_notes(
    slot: str,
    queue: str | None,
    interpretation: str | None,
) -> list[str] | None:
    """Find the most-specific matching rule's notes, or None for fallback.

    Returns ``None`` when no rule matches -- the caller (intro/outro
    synthesis or cue rendering) should then use its existing global
    default. A rule with no notes (or only malformed notes) is treated as
    if it didn't match, so a half-typed UI row never silences a tone.
    """
    if slot not in _VALID_SLOTS:
        return None

    best_score = -1
    best_notes: list[str] | None = None

    for rule in get_tone_rules():
        if not isinstance(rule, dict):
            continue
        if rule.get("slot") != slot:
            continue

        rule_queue = rule.get("queue")
        rule_is_regex = bool(rule.get("queue_regex"))
        if isinstance(rule_queue, str):
            rule_queue = rule_queue.strip() or None
        else:
            rule_queue = None

        rule_interp = rule.get("interpretation")
        if isinstance(rule_interp, str):
            rule_interp = rule_interp.strip() or None
        else:
            rule_interp = None

        if not _queue_match(rule_queue, rule_is_regex, queue):
            continue
        if not _interp_match(rule_interp, interpretation):
            continue

        score = 0
        if rule_queue is not None:
            score += 2
        if rule_interp is not None:
            score += 1

        notes = _clean_notes(rule.get("notes"))
        if not notes:
            continue

        if score > best_score:
            best_score = score
            best_notes = notes

    return best_notes


def validate_rule(rule: dict) -> tuple[bool, str]:
    """Return (ok, error_message). Empty error when the rule is well-formed."""
    if not isinstance(rule, dict):
        return False, "rule must be an object"
    slot = rule.get("slot")
    if slot not in _VALID_SLOTS:
        return False, f"slot must be one of {sorted(_VALID_SLOTS)}, got {slot!r}"

    rule_queue = rule.get("queue")
    if rule_queue is not None and not isinstance(rule_queue, str):
        return False, "queue must be a string or null"
    if isinstance(rule_queue, str) and bool(rule.get("queue_regex")):
        try:
            re.compile(rule_queue)
        except re.error as e:
            return False, f"queue_regex invalid: {e}"

    rule_interp = rule.get("interpretation")
    if rule_interp is not None and not isinstance(rule_interp, str):
        return False, "interpretation must be a string or null"

    notes_raw = rule.get("notes")
    if not isinstance(notes_raw, list) or not notes_raw:
        return False, "notes must be a non-empty list"
    cleaned = _clean_notes(notes_raw)
    if len(cleaned) != len(notes_raw):
        return False, "one or more notes invalid (expected [A-G][b#]?[0-8][:mult]?)"

    if rule_queue in (None, "") and rule_interp in (None, ""):
        return False, "rule must specify queue, interpretation, or both"

    return True, ""


def normalize_rule(rule: dict) -> dict:
    """Return a canonical rule dict with whitespace stripped and notes cleaned."""
    slot = rule.get("slot")
    rule_queue = rule.get("queue")
    if isinstance(rule_queue, str):
        rule_queue = rule_queue.strip() or None
    else:
        rule_queue = None

    rule_interp = rule.get("interpretation")
    if isinstance(rule_interp, str):
        rule_interp = rule_interp.strip() or None
    else:
        rule_interp = None

    return {
        "slot": slot,
        "queue": rule_queue,
        "queue_regex": bool(rule.get("queue_regex")),
        "interpretation": rule_interp,
        "notes": _clean_notes(rule.get("notes")),
    }
