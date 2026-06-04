#!/usr/bin/env python3
"""Unit tests for interpretations.py (interpretation cue resolution)."""

import os
from unittest.mock import patch

from speeker.config import save_config
from speeker.interpretations import (
    BUILTIN_INTERPRETATIONS,
    interpretation_names,
    is_valid_interpretation,
    notes_to_spec,
    parse_pitch,
    pause_after_seconds,
    resolve_interpretation,
)


class TestBuiltins:
    def test_success_and_error_always_present(self, tmp_path):
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            names = interpretation_names()
        assert "SUCCESS" in names
        assert "ERROR" in names

    def test_success_is_notes(self, tmp_path):
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            indication = resolve_interpretation("SUCCESS")
        assert indication["type"] == "notes"
        assert indication["notes"][0]["pitch"] == "Eb3"
        assert indication["notes"][-1]["pitch"] == "G#3"

    def test_error_ends_on_doubled_low_bb(self, tmp_path):
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            spec = notes_to_spec(resolve_interpretation("ERROR"))
        assert spec == [("eb", 4, 0.3), ("d", 4, 0.2), ("bb", 2, 0.2), ("bb", 2, 0.2)]

    def test_info_is_single_eb4(self, tmp_path):
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            ind = resolve_interpretation("INFO")
            names = interpretation_names()
        assert "INFO" in names
        assert ind["type"] == "notes"
        assert [n["pitch"] for n in ind["notes"]] == ["Eb4"]

    def test_warning_is_double_eb4(self, tmp_path):
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            ind = resolve_interpretation("WARNING")
            names = interpretation_names()
        assert "WARNING" in names
        assert [n["pitch"] for n in ind["notes"]] == ["Eb4", "Eb4"]


class TestResolve:
    def test_unknown_returns_none(self, tmp_path):
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            assert resolve_interpretation("NOPE") is None

    def test_case_insensitive(self, tmp_path):
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            assert resolve_interpretation("success") == BUILTIN_INTERPRETATIONS["SUCCESS"]

    def test_empty_and_none_are_invalid(self, tmp_path):
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            assert not is_valid_interpretation("")
            assert not is_valid_interpretation(None)
            assert is_valid_interpretation("ERROR")


class TestConfigOverlay:
    def test_custom_entry_does_not_erase_builtins(self, tmp_path):
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            save_config({
                "interpretations": {
                    "map": {"DEPLOY": {"type": "sound_file", "path": "/x.wav"}}
                }
            })
            names = interpretation_names()
            assert "DEPLOY" in names
            # Built-ins survive a config that defines only a custom map.
            assert "SUCCESS" in names
            assert "ERROR" in names

    def test_config_overrides_builtin(self, tmp_path):
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            save_config({
                "interpretations": {
                    "map": {"SUCCESS": {"type": "sound_file", "path": "/custom.wav"}}
                }
            })
            indication = resolve_interpretation("SUCCESS")
        assert indication == {"type": "sound_file", "path": "/custom.wav"}

    def test_pause_after_seconds_from_config(self, tmp_path):
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            save_config({"interpretations": {"pause_after_seconds": 0.5}})
            assert pause_after_seconds() == 0.5

    def test_pause_after_seconds_default(self, tmp_path):
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            assert pause_after_seconds() == 0.3


class TestParsePitch:
    def test_natural(self):
        assert parse_pitch("D4") == ("d", 4)

    def test_flat(self):
        assert parse_pitch("Eb3") == ("eb", 3)
        assert parse_pitch("Bb2") == ("bb", 2)

    def test_sharp(self):
        assert parse_pitch("G#3") == ("g#", 3)

    def test_whitespace_tolerated(self):
        assert parse_pitch("  C5 ") == ("c", 5)

    def test_invalid_returns_none(self):
        assert parse_pitch("H4") is None
        assert parse_pitch("Eb") is None
        assert parse_pitch("4") is None
        assert parse_pitch("") is None


class TestNotesToSpec:
    def test_skips_invalid_pitch(self):
        indication = {"notes": [{"pitch": "Eb3", "seconds": 0.1}, {"pitch": "H9"}]}
        assert notes_to_spec(indication) == [("eb", 3, 0.1)]

    def test_defaults_seconds(self):
        indication = {"notes": [{"pitch": "C4"}]}
        assert notes_to_spec(indication) == [("c", 4, 0.2)]

    def test_empty(self):
        assert notes_to_spec({"notes": []}) == []
        assert notes_to_spec({}) == []
