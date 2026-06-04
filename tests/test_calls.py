#!/usr/bin/env python3
"""Tests for call detection (calls.py) and the player pause gate."""

import json
import os
from unittest.mock import patch

import pytest


@pytest.fixture
def speeker_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEEKER_DIR", str(tmp_path))
    return tmp_path


def _write_state(path, active):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"active": active, "source": "coreaudio-input", "version": 1}))


def _set_calls_config(monkeypatch, *, pause_when_active, state_file):
    from speeker import calls
    monkeypatch.setattr(
        calls, "get_calls_config",
        lambda: {"pause_when_active": pause_when_active, "state_file": str(state_file)},
    )


class TestCallStatus:
    def test_unavailable_when_missing(self, speeker_dir, monkeypatch):
        from speeker import calls
        _set_calls_config(monkeypatch, pause_when_active=True,
                          state_file=speeker_dir / "nope.json")
        assert calls.call_status() == "unavailable"

    def test_unavailable_when_malformed(self, speeker_dir, monkeypatch):
        from speeker import calls
        p = speeker_dir / "state.json"
        p.write_text("{not json")
        _set_calls_config(monkeypatch, pause_when_active=True, state_file=p)
        assert calls.call_status() == "unavailable"

    def test_active_and_idle(self, speeker_dir, monkeypatch):
        from speeker import calls
        p = speeker_dir / "state.json"
        _set_calls_config(monkeypatch, pause_when_active=True, state_file=p)
        _write_state(p, True)
        assert calls.call_status() == "active"
        _write_state(p, False)
        assert calls.call_status() == "idle"


class TestShouldPause:
    def test_disabled_never_pauses(self, speeker_dir, monkeypatch):
        from speeker import calls
        p = speeker_dir / "state.json"
        _write_state(p, True)
        _set_calls_config(monkeypatch, pause_when_active=False, state_file=p)
        assert calls.should_pause_for_call() is False

    def test_enabled_pauses_only_when_active(self, speeker_dir, monkeypatch):
        from speeker import calls
        p = speeker_dir / "state.json"
        _set_calls_config(monkeypatch, pause_when_active=True, state_file=p)
        _write_state(p, True)
        assert calls.should_pause_for_call() is True
        _write_state(p, False)
        assert calls.should_pause_for_call() is False

    def test_enabled_but_monitor_absent_is_noop(self, speeker_dir, monkeypatch):
        from speeker import calls
        _set_calls_config(monkeypatch, pause_when_active=True,
                          state_file=speeker_dir / "absent.json")
        assert calls.should_pause_for_call() is False


class TestConfigAccessor:
    def test_expands_tilde(self, speeker_dir):
        from speeker.config import get_calls_config
        cfg = get_calls_config()
        assert cfg["state_file"].startswith(os.path.expanduser("~"))
        assert cfg["pause_when_active"] is False  # default off
