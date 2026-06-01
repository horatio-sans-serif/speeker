#!/usr/bin/env python3
"""Unit tests for tone_rules.py (per-queue / per-interpretation resolution).

The resolver scores each candidate rule against the call context:
  queue match -> +2, interpretation match -> +1.
The highest-scoring rule wins; falls back to None so the caller (player.py)
applies the global default.
"""

import os
from unittest.mock import patch

from speeker.config import save_config
from speeker.tone_rules import (
    normalize_rule,
    resolve_tone_notes,
    validate_rule,
)


def _save_rules(tmp_path, rules):
    """Persist a tone_rules list to the per-test config dir."""
    cfg = {"tone_rules": rules}
    with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
        save_config(cfg)


class TestNoRules:
    def test_resolve_returns_none_when_empty(self, tmp_path):
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            assert resolve_tone_notes("intro", "any-queue", None) is None
            assert resolve_tone_notes("cue", "any-queue", "SUCCESS") is None


class TestQueueOnly:
    def test_queue_match_wins(self, tmp_path):
        _save_rules(tmp_path, [
            {"slot": "intro", "queue": "compass-docs",
             "queue_regex": False, "interpretation": None,
             "notes": ["A4", "B4", "C5"]},
        ])
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            assert resolve_tone_notes("intro", "compass-docs", None) == ["A4", "B4", "C5"]

    def test_queue_mismatch_returns_none(self, tmp_path):
        _save_rules(tmp_path, [
            {"slot": "intro", "queue": "compass-docs",
             "queue_regex": False, "interpretation": None,
             "notes": ["A4", "B4", "C5"]},
        ])
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            assert resolve_tone_notes("intro", "other-queue", None) is None

    def test_queue_regex_match(self, tmp_path):
        _save_rules(tmp_path, [
            {"slot": "outro", "queue": "^rm-", "queue_regex": True,
             "interpretation": None, "notes": ["C5", "G4", "E4"]},
        ])
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            assert resolve_tone_notes("outro", "rm-anything", None) == ["C5", "G4", "E4"]
            assert resolve_tone_notes("outro", "other", None) is None


class TestInterpretationOnly:
    def test_interpretation_match(self, tmp_path):
        _save_rules(tmp_path, [
            {"slot": "cue", "queue": None, "queue_regex": False,
             "interpretation": "SUCCESS", "notes": ["E5", "G5"]},
        ])
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            assert resolve_tone_notes("cue", "any-queue", "SUCCESS") == ["E5", "G5"]
            assert resolve_tone_notes("cue", "any-queue", "ERROR") is None

    def test_interpretation_case_insensitive(self, tmp_path):
        _save_rules(tmp_path, [
            {"slot": "cue", "queue": None, "queue_regex": False,
             "interpretation": "SUCCESS", "notes": ["E5"]},
        ])
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            assert resolve_tone_notes("cue", None, "success") == ["E5"]


class TestSpecificity:
    """Higher-scoring rules win over lower ones."""

    def test_queue_plus_interp_beats_queue_only(self, tmp_path):
        _save_rules(tmp_path, [
            {"slot": "cue", "queue": "compass-docs", "queue_regex": False,
             "interpretation": None, "notes": ["A4"]},  # score 2
            {"slot": "cue", "queue": "compass-docs", "queue_regex": False,
             "interpretation": "SUCCESS", "notes": ["E5", "G5"]},  # score 3
        ])
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            assert resolve_tone_notes("cue", "compass-docs", "SUCCESS") == ["E5", "G5"]

    def test_queue_only_beats_interp_only(self, tmp_path):
        _save_rules(tmp_path, [
            {"slot": "cue", "queue": None, "queue_regex": False,
             "interpretation": "SUCCESS", "notes": ["C5"]},  # score 1
            {"slot": "cue", "queue": "compass-docs", "queue_regex": False,
             "interpretation": None, "notes": ["A4"]},  # score 2
        ])
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            assert resolve_tone_notes("cue", "compass-docs", "SUCCESS") == ["A4"]

    def test_interp_only_when_no_queue_match(self, tmp_path):
        _save_rules(tmp_path, [
            {"slot": "cue", "queue": None, "queue_regex": False,
             "interpretation": "SUCCESS", "notes": ["C5"]},
            {"slot": "cue", "queue": "compass-docs", "queue_regex": False,
             "interpretation": None, "notes": ["A4"]},
        ])
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            assert resolve_tone_notes("cue", "other-queue", "SUCCESS") == ["C5"]


class TestSlot:
    def test_slot_isolates_rules(self, tmp_path):
        _save_rules(tmp_path, [
            {"slot": "intro", "queue": "X", "queue_regex": False,
             "interpretation": None, "notes": ["A4"]},
        ])
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            assert resolve_tone_notes("intro", "X", None) == ["A4"]
            assert resolve_tone_notes("outro", "X", None) is None
            assert resolve_tone_notes("cue", "X", None) is None

    def test_invalid_slot_returns_none(self, tmp_path):
        _save_rules(tmp_path, [
            {"slot": "intro", "queue": "X", "queue_regex": False,
             "interpretation": None, "notes": ["A4"]},
        ])
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            assert resolve_tone_notes("bogus", "X", None) is None


class TestMalformed:
    def test_bad_notes_drop_rule(self, tmp_path):
        _save_rules(tmp_path, [
            {"slot": "intro", "queue": "X", "queue_regex": False,
             "interpretation": None, "notes": ["not-a-note"]},
        ])
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            assert resolve_tone_notes("intro", "X", None) is None

    def test_partial_bad_notes_dropped(self, tmp_path):
        # The whole rule is dropped only if NO notes survive cleaning.
        # Here one good + one bad: the good survives.
        _save_rules(tmp_path, [
            {"slot": "intro", "queue": "X", "queue_regex": False,
             "interpretation": None, "notes": ["E4", "junk"]},
        ])
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            assert resolve_tone_notes("intro", "X", None) == ["E4"]

    def test_bad_regex_skips_rule(self, tmp_path):
        _save_rules(tmp_path, [
            {"slot": "intro", "queue": "(", "queue_regex": True,
             "interpretation": None, "notes": ["E4"]},
        ])
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            assert resolve_tone_notes("intro", "anything", None) is None

    def test_non_dict_rule_skipped(self, tmp_path):
        _save_rules(tmp_path, [
            "not-a-dict",
            {"slot": "intro", "queue": "X", "queue_regex": False,
             "interpretation": None, "notes": ["E4"]},
        ])
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            assert resolve_tone_notes("intro", "X", None) == ["E4"]


class TestMultiplierNotation:
    """Rules accept the same :multiplier syntax used inline."""

    def test_multiplier_preserved(self, tmp_path):
        _save_rules(tmp_path, [
            {"slot": "intro", "queue": "X", "queue_regex": False,
             "interpretation": None, "notes": ["G4", "E4", "C5:6"]},
        ])
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            assert resolve_tone_notes("intro", "X", None) == ["G4", "E4", "C5:6"]


class TestValidateRule:
    """API-side validation. Used by PUT /api/tone-rules."""

    def test_ok(self):
        ok, err = validate_rule({
            "slot": "intro", "queue": "X", "queue_regex": False,
            "interpretation": None, "notes": ["E4"],
        })
        assert ok, err

    def test_bad_slot(self):
        ok, err = validate_rule({"slot": "bogus", "queue": "X", "notes": ["E4"]})
        assert not ok and "slot" in err

    def test_empty_notes(self):
        ok, err = validate_rule({
            "slot": "intro", "queue": "X", "notes": [],
        })
        assert not ok and "notes" in err

    def test_bad_note_token(self):
        ok, err = validate_rule({
            "slot": "intro", "queue": "X", "notes": ["E4", "junk"],
        })
        assert not ok and "invalid" in err.lower()

    def test_no_queue_or_interp_rejected(self):
        # A rule with no queue and no interpretation has score 0 -- it would
        # match every utterance and act as a hidden global override. Reject
        # at the API boundary so users can't shoot themselves in the foot.
        ok, err = validate_rule({
            "slot": "cue", "queue": None, "interpretation": None,
            "notes": ["E4"],
        })
        assert not ok and "queue" in err.lower() and "interpretation" in err.lower()

    def test_bad_regex_rejected(self):
        ok, err = validate_rule({
            "slot": "cue", "queue": "(", "queue_regex": True,
            "interpretation": "SUCCESS", "notes": ["E4"],
        })
        assert not ok and "regex" in err.lower()


class TestNormalize:
    def test_strips_whitespace(self):
        n = normalize_rule({
            "slot": "intro", "queue": "  X  ", "queue_regex": False,
            "interpretation": "  SUCCESS  ", "notes": ["E4 ", " G4"],
        })
        assert n == {
            "slot": "intro", "queue": "X", "queue_regex": False,
            "interpretation": "SUCCESS", "notes": ["E4", "G4"],
        }

    def test_empty_string_becomes_none(self):
        n = normalize_rule({
            "slot": "cue", "queue": "", "queue_regex": False,
            "interpretation": "SUCCESS", "notes": ["E4"],
        })
        assert n["queue"] is None
