"""Background-music engine: drive mpv instances and duck under speech.

Runs inside the player daemon. Music plays through ``mpv`` (looping, live volume
via its JSON IPC socket) so it can be ducked while ``afplay`` speaks; the OS
mixes the two. Two mpv instances are kept so tracks can crossfade.

The transport (process launch + IPC send + stop) is injectable so the engine's
ducking / crossfade logic can be tested without a real mpv. All public calls are
best-effort: a missing mpv, disabled config, or IPC error is a silent no-op and
never blocks or breaks TTS.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path


# mpv is typically Homebrew-installed, but the launchd-managed daemon runs with
# a minimal PATH that omits /opt/homebrew/bin -- so resolve an absolute path
# (PATH first, then the usual install locations) instead of relying on "mpv".
_MPV_CANDIDATES = ("/opt/homebrew/bin/mpv", "/usr/local/bin/mpv", "/usr/bin/mpv")


def _find_mpv() -> str | None:
    found = shutil.which("mpv")
    if found:
        return found
    for candidate in _MPV_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    return None


class RealTransport:
    """Launch mpv per slot and talk to it over a Unix IPC socket."""

    def __init__(self) -> None:
        self._procs: dict[str, subprocess.Popen] = {}

    def available(self) -> bool:
        return _find_mpv() is not None

    def start(self, sock: str) -> None:
        if sock in self._procs and self._procs[sock].poll() is None:
            return
        mpv = _find_mpv() or "mpv"
        Path(sock).unlink(missing_ok=True)
        self._procs[sock] = subprocess.Popen(
            [
                mpv, "--no-video", "--idle=yes", "--loop-file=inf",
                "--really-quiet", "--volume=0", f"--input-ipc-server={sock}",
            ],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    def send(self, sock: str, command: list) -> None:
        payload = json.dumps({"command": command}).encode() + b"\n"
        # mpv creates the socket shortly after launch; retry briefly.
        for _ in range(20):
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                    s.settimeout(0.5)
                    s.connect(sock)
                    s.sendall(payload)
                return
            except (FileNotFoundError, ConnectionRefusedError, OSError):
                time.sleep(0.05)

    def stop(self, sock: str) -> None:
        proc = self._procs.pop(sock, None)
        if proc is not None:
            try:
                proc.terminate()
            except Exception:
                pass
        Path(sock).unlink(missing_ok=True)


class MusicEngine:
    """Two-slot mpv controller with gate ducking and track crossfade."""

    def __init__(
        self,
        transport: RealTransport | None = None,
        *,
        run_async=None,
        sleep=time.sleep,
        sock_dir: str | None = None,
    ) -> None:
        self._t = transport if transport is not None else RealTransport()
        self._run_async = run_async if run_async is not None else _start_thread
        self._sleep = sleep
        base = sock_dir or tempfile.gettempdir()
        self._socks = [str(Path(base) / "speeker-music-0.sock"),
                       str(Path(base) / "speeker-music-1.sock")]
        self._started = [False, False]
        self._active = 0                # slot index currently playing
        self._track: Path | None = None
        self._ducking = False
        self._vol = [0.0, 0.0]          # last sent volume per slot (0..100)
        self._ramp_token = [0, 0]
        self._lock = threading.Lock()

    # -- config / availability -------------------------------------------------

    def available(self) -> bool:
        from .config import get_music_config
        return bool(get_music_config().get("enabled")) and self._t.available()

    def _cfg(self) -> dict:
        from .config import get_music_config
        return get_music_config()

    def _base_volume(self) -> float:
        try:
            return max(0.0, min(1.0, float(self._cfg().get("volume", 0.6)))) * 100.0
        except (TypeError, ValueError):
            return 60.0

    def _target_volume(self) -> float:
        """Effective volume for the active slot given the ducking state."""
        base = self._base_volume()
        if not self._ducking:
            return base
        try:
            duck = max(0.0, min(1.0, float(self._cfg().get("duck_level", 0.2))))
        except (TypeError, ValueError):
            duck = 0.2
        return base * duck

    def _ms(self, key: str, default: int) -> int:
        try:
            return max(0, int(self._cfg().get(key, default)))
        except (TypeError, ValueError):
            return default

    # -- public API (best-effort) ---------------------------------------------

    def set_track(self, track: Path | None) -> None:
        if not self.available():
            return
        try:
            self._set_track(track)
        except Exception as e:  # noqa: BLE001 - never break playback
            _log(f"set_track failed: {e}")

    def duck(self, on: bool) -> None:
        if not self.available():
            return
        try:
            self._ducking = bool(on)
            if self._track is not None:
                self._ramp(self._active, self._target_volume(), self._ms("fade_ms", 400))
        except Exception as e:  # noqa: BLE001
            _log(f"duck failed: {e}")

    def stop(self, fade: bool = True) -> None:
        try:
            if self._track is not None:
                self._ramp(self._active, 0.0, self._ms("fade_ms", 400) if fade else 0)
            self._track = None
        except Exception as e:  # noqa: BLE001
            _log(f"stop failed: {e}")

    def shutdown(self) -> None:
        for i, sock in enumerate(self._socks):
            if self._started[i]:
                try:
                    self._t.stop(sock)
                except Exception:
                    pass
            self._started[i] = False
        self._track = None

    # -- internals -------------------------------------------------------------

    def _ensure_started(self, slot: int) -> None:
        if not self._started[slot]:
            self._t.start(self._socks[slot])
            self._started[slot] = True

    def _send(self, slot: int, command: list) -> None:
        self._t.send(self._socks[slot], command)

    def _set_volume(self, slot: int, vol: float) -> None:
        vol = max(0.0, min(100.0, vol))
        self._vol[slot] = vol
        self._send(slot, ["set_property", "volume", round(vol, 2)])

    def _ramp(self, slot: int, target: float, ms: int) -> None:
        """Ramp *slot* volume to *target* over *ms*, superseding any prior ramp."""
        with self._lock:
            self._ramp_token[slot] += 1
            token = self._ramp_token[slot]
        start = self._vol[slot]
        if ms <= 0 or abs(target - start) < 0.01:
            self._set_volume(slot, target)
            return
        steps = max(1, ms // 40)
        step_sleep = (ms / 1000.0) / steps

        def run():
            for i in range(1, steps + 1):
                if self._ramp_token[slot] != token:
                    return  # superseded
                self._set_volume(slot, start + (target - start) * i / steps)
                self._sleep(step_sleep)

        self._run_async(run)

    def _set_track(self, track: Path | None) -> None:
        if track == self._track:
            return
        if track is None:
            self.stop(fade=True)
            return

        idle = 1 - self._active
        self._ensure_started(idle)
        self._set_volume(idle, 0.0)
        self._send(idle, ["loadfile", str(track), "replace"])
        self._send(idle, ["set_property", "loop-file", "inf"])

        old = self._active
        had_track = self._track is not None
        cross = self._ms("crossfade_ms", 600)
        # New slot up to the (possibly ducked) target; old slot down to 0.
        self._ramp(idle, self._target_volume(), cross if had_track else self._ms("fade_ms", 400))
        if had_track:
            self._ramp(old, 0.0, cross)

        self._active = idle
        self._track = track


def _start_thread(fn) -> None:
    threading.Thread(target=fn, daemon=True, name="speeker-music-ramp").start()


def _log(msg: str) -> None:
    print(f"[music] {msg}", file=sys.stderr, flush=True)
