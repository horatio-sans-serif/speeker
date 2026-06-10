#!/usr/bin/env python3
"""Tests for macOS Focus detection (focus.py) and the pause-on-Focus API."""

import json
import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from speeker import focus
from speeker.server import app


def _assertions(modes):
    """Build an Assertions.json-shaped dict with the given active mode ids."""
    return {"data": [{
        "storeInvalidationRecords": [],
        "storeAssertionRecords": [
            {"assertionDetails": {"assertionDetailsModeIdentifier": m}} for m in modes
        ],
    }]}


@pytest.fixture
def assertions_file(tmp_path, monkeypatch):
    p = tmp_path / "Assertions.json"
    monkeypatch.setattr(focus, "_ASSERTIONS", p)
    return p


def _set_focus_config(monkeypatch, *, pause_when_active, modes=None):
    monkeypatch.setattr(
        focus, "get_focus_config",
        lambda: {"pause_when_active": pause_when_active, "modes": modes or []},
    )


class TestActiveModes:
    def test_missing_file(self, assertions_file):
        assert focus.active_focus_modes() == []
        assert focus.focus_status() == "unavailable"

    def test_no_active(self, assertions_file):
        assertions_file.write_text(json.dumps(_assertions([])))
        assert focus.active_focus_modes() == []
        assert focus.focus_status() == "idle"

    def test_active_reduce_interruptions(self, assertions_file):
        assertions_file.write_text(json.dumps(_assertions(["com.apple.focus.reduce-interruptions"])))
        assert focus.active_focus_modes() == ["com.apple.focus.reduce-interruptions"]
        assert focus.focus_status() == "active"

    def test_ignores_ended_assertions(self, assertions_file):
        # Only storeInvalidationRecords present (an ended focus) -> not active.
        assertions_file.write_text(json.dumps({"data": [{
            "storeInvalidationRecords": [
                {"invalidationAssertion": {"assertionDetails": {
                    "assertionDetailsModeIdentifier": "com.apple.sleep.sleep-mode"}}}
            ]
        }]}))
        assert focus.active_focus_modes() == []


class TestShouldPause:
    def test_disabled(self, assertions_file, monkeypatch):
        assertions_file.write_text(json.dumps(_assertions(["com.apple.focus.reduce-interruptions"])))
        _set_focus_config(monkeypatch, pause_when_active=False)
        assert focus.should_pause_for_focus() is False

    def test_any_focus_when_no_filter(self, assertions_file, monkeypatch):
        assertions_file.write_text(json.dumps(_assertions(["com.apple.donotdisturb.mode.default"])))
        _set_focus_config(monkeypatch, pause_when_active=True)
        assert focus.should_pause_for_focus() is True

    def test_idle_does_not_pause(self, assertions_file, monkeypatch):
        assertions_file.write_text(json.dumps(_assertions([])))
        _set_focus_config(monkeypatch, pause_when_active=True)
        assert focus.should_pause_for_focus() is False

    def test_mode_filter_matches_substring(self, assertions_file, monkeypatch):
        assertions_file.write_text(json.dumps(_assertions(["com.apple.focus.reduce-interruptions"])))
        _set_focus_config(monkeypatch, pause_when_active=True, modes=["reduce-interruptions"])
        assert focus.should_pause_for_focus() is True

    def test_mode_filter_excludes_others(self, assertions_file, monkeypatch):
        assertions_file.write_text(json.dumps(_assertions(["com.apple.donotdisturb.mode.default"])))
        _set_focus_config(monkeypatch, pause_when_active=True, modes=["reduce-interruptions"])
        assert focus.should_pause_for_focus() is False


class TestFocusApi:
    def test_default_off_and_toggle(self, tmp_path):
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            client = TestClient(app)
            assert client.get("/api/focus").json()["pause_when_active"] is False
            r = client.put("/api/focus", json={"pause_when_active": True}).json()
            assert r["pause_when_active"] is True
            assert client.get("/api/focus").json()["pause_when_active"] is True


class TestConfigDefault:
    def test_focus_config_default(self, tmp_path):
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            from speeker.config import get_focus_config
            cfg = get_focus_config()
            assert cfg["pause_when_active"] is False
            assert cfg["modes"] == []
