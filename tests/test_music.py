#!/usr/bin/env python3
"""Tests for music track resolution, the mpv engine logic, and the music API."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from speeker.server import app


# --- resolve_music_track -----------------------------------------------------

@pytest.fixture
def tracks(tmp_path):
    """Create real track files so resolution's existence check passes."""
    a = tmp_path / "a.mp3"; a.write_bytes(b"x")
    b = tmp_path / "b.mp3"; b.write_bytes(b"x")
    c = tmp_path / "c.mp3"; c.write_bytes(b"x")
    return a, b, c


def _resolve(rules, queue, interp):
    from speeker import music
    with patch.object(music, "get_music_rules", return_value=rules):
        return music.resolve_music_track(queue, interp)


class TestResolve:
    def test_no_rules(self, tracks):
        assert _resolve([], "alpha", "SUCCESS") is None

    def test_queue_only(self, tracks):
        a, _, _ = tracks
        assert _resolve([{"queue": "alpha", "track": str(a)}], "alpha", None) == a
        assert _resolve([{"queue": "alpha", "track": str(a)}], "beta", None) is None

    def test_interp_only(self, tracks):
        a, _, _ = tracks
        r = [{"interpretation": "ERROR", "track": str(a)}]
        assert _resolve(r, "any", "ERROR") == a
        assert _resolve(r, "any", "SUCCESS") is None

    def test_queue_plus_interp_wins(self, tracks):
        a, b, _ = tracks
        rules = [
            {"queue": "alpha", "track": str(a)},                      # score 2
            {"queue": "alpha", "interpretation": "ERROR", "track": str(b)},  # score 3
        ]
        assert _resolve(rules, "alpha", "ERROR") == b
        assert _resolve(rules, "alpha", "SUCCESS") == a  # falls to queue-only

    def test_regex_queue(self, tracks):
        a, _, _ = tracks
        r = [{"queue": r"^proj-", "queue_regex": True, "track": str(a)}]
        assert _resolve(r, "proj-x", None) == a
        assert _resolve(r, "other", None) is None

    def test_missing_track_ignored(self, tmp_path):
        r = [{"queue": "alpha", "track": str(tmp_path / "gone.mp3")}]
        assert _resolve(r, "alpha", None) is None

    def test_case_insensitive_interp(self, tracks):
        a, _, _ = tracks
        r = [{"interpretation": "success", "track": str(a)}]
        assert _resolve(r, "any", "SUCCESS") == a


# --- MusicEngine (fake transport) --------------------------------------------

class FakeTransport:
    """Records commands instead of launching mpv."""
    def __init__(self):
        self.sent = []      # (sock, command)
        self.started = []
        self.stopped = []
    def available(self):
        return True
    def start(self, sock):
        self.started.append(sock)
    def send(self, sock, command):
        self.sent.append((sock, command))
    def stop(self, sock):
        self.stopped.append(sock)

    # convenience: volumes set on a given socket, in order
    def volumes(self, sock):
        return [c[2] for (s, c) in self.sent if s == sock and c[:2] == ["set_property", "volume"]]


def _engine(monkeypatch, tmp_path, enabled=True, **cfg):
    from speeker import music_engine
    base = {"enabled": enabled, "volume": 1.0, "duck_level": 0.2, "fade_ms": 0, "crossfade_ms": 0}
    base.update(cfg)
    monkeypatch.setattr(music_engine, "get_music_config", lambda: base, raising=False)
    # Patch the config accessor used inside the engine (imported lazily).
    monkeypatch.setattr("speeker.config.get_music_config", lambda: base)
    ft = FakeTransport()
    eng = music_engine.MusicEngine(
        transport=ft, run_async=lambda fn: fn(), sleep=lambda s: None,
        sock_dir=str(tmp_path),
    )
    return eng, ft


class TestEngine:
    def test_unavailable_no_commands(self, monkeypatch, tmp_path):
        eng, ft = _engine(monkeypatch, tmp_path, enabled=False)
        eng.set_track(tmp_path / "a.mp3")
        eng.duck(True)
        assert ft.sent == [] and ft.started == []

    def test_set_track_loads_and_ramps_up(self, monkeypatch, tmp_path):
        eng, ft = _engine(monkeypatch, tmp_path)
        track = tmp_path / "a.mp3"
        eng.set_track(track)
        sock0 = eng._socks[1 - 0]  # idle slot used first (active starts 0 -> idle 1)
        loads = [c for (_, c) in ft.sent if c and c[0] == "loadfile"]
        assert loads and loads[0][1] == str(track)
        # ramped up to base volume (100) on the now-active slot
        assert eng._vol[eng._active] == pytest.approx(100.0)

    def test_duck_lowers_then_restores(self, monkeypatch, tmp_path):
        eng, ft = _engine(monkeypatch, tmp_path)
        eng.set_track(tmp_path / "a.mp3")
        active_sock = eng._socks[eng._active]
        eng.duck(True)
        assert eng._vol[eng._active] == pytest.approx(20.0)  # 100 * 0.2
        eng.duck(False)
        assert eng._vol[eng._active] == pytest.approx(100.0)

    def test_crossfade_switches_active_slot(self, monkeypatch, tmp_path):
        eng, ft = _engine(monkeypatch, tmp_path)
        eng.set_track(tmp_path / "a.mp3")
        first_active = eng._active
        eng.set_track(tmp_path / "b.mp3")
        assert eng._active != first_active            # switched slots
        assert eng._vol[eng._active] == pytest.approx(100.0)   # new up
        assert eng._vol[first_active] == pytest.approx(0.0)    # old down

    def test_same_track_is_noop(self, monkeypatch, tmp_path):
        eng, ft = _engine(monkeypatch, tmp_path)
        t = tmp_path / "a.mp3"
        eng.set_track(t)
        n = len(ft.sent)
        eng.set_track(t)
        assert len(ft.sent) == n  # no new commands

    def test_transport_failure_does_not_raise(self, monkeypatch, tmp_path):
        eng, ft = _engine(monkeypatch, tmp_path)
        def boom(*a, **k):
            raise RuntimeError("ipc down")
        ft.send = boom
        eng.set_track(tmp_path / "a.mp3")  # must not raise
        eng.duck(True)


# --- web API + config default ------------------------------------------------

class TestMusicApi:
    def test_get_and_put_music(self, tmp_path):
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            client = TestClient(app)
            g = client.get("/api/music").json()
            assert g["enabled"] is False and "mpv_installed" in g
            client.put("/api/music", json={"enabled": True, "duck_level": 0.3})
            g2 = client.get("/api/music").json()
            assert g2["enabled"] is True and g2["duck_level"] == 0.3

    def test_music_rules_roundtrip_drops_trackless(self, tmp_path):
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            client = TestClient(app)
            r = client.put("/api/music-rules", json={"rules": [
                {"queue": "alpha", "interpretation": "ERROR", "track": "/m/e.mp3"},
                {"queue": "beta", "track": ""},  # dropped (no track)
            ]}).json()
            assert len(r["rules"]) == 1
            assert r["rules"][0]["queue"] == "alpha"
            assert client.get("/api/music-rules").json()["rules"] == r["rules"]


class TestConfigDefault:
    def test_music_defaults(self, tmp_path):
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            from speeker.config import get_music_config, get_music_rules
            cfg = get_music_config()
            assert cfg["enabled"] is False and cfg["volume"] == 0.6
            assert get_music_rules() == []
