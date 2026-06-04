# monitor-active-calls + Speeker pause-on-call

Date: 2026-06-02

## Context

Speeker speaks queued text aloud. When the user is on a call (Zoom, FaceTime,
Google Meet, Teams, phone via Continuity, etc.), spoken notifications talk over
the conversation. The user wants the system to know when a call is active and
optionally pause Speeker's queues until the call ends.

Rather than couple call-detection into Speeker, this is split into a generic,
reusable macOS daemon (`monitor-active-calls`) that any client can consume, plus
an optional Speeker integration.

## Component A — monitor-active-calls (standalone)

Home: `~/projects/sys/mac/monitor-active-calls/`. Language: Swift (SwiftPM
executable). Single compiled binary run as a launchd agent.

### Detection: mic-in-use via CoreAudio (event-driven)

A call always uses the microphone. Instead of per-app logic (brittle; can't see
a Google Meet browser tab), watch whether the default input device is in use by
any process:

- Resolve the default input device via `kAudioHardwarePropertyDefaultInputDevice`.
- Add a property listener on `kAudioDevicePropertyDeviceIsRunningSomewhere` on
  that device. It fires whenever any process starts/stops capturing.
- Add a listener on `kAudioHardwarePropertyDefaultInputDevice` to re-bind when
  the default mic changes (e.g. switching to AirPods).
- `active = device-is-running-somewhere`.

Reading run-state does not capture audio, so it requires no microphone TCC
permission (verified at build time). Known behavior: this also reports active
for dictation / Voice Memos / any mic use — acceptable and arguably desirable
("don't talk while my mic is live").

Camera detection is intentionally excluded (no clean public macOS API; mic
already covers every call).

### Outputs (both)

1. **State file** `~/.local/monitor-active-calls/state.json`, written atomically
   (write temp in same dir, then `rename`) on every transition. fsnotify-friendly.

   ```json
   {
     "active": true,
     "since": "2026-06-02T17:00:00Z",
     "updated_at": "2026-06-02T17:00:00Z",
     "source": "coreaudio-input",
     "version": 1
   }
   ```

   `since` is when the current state began (null when idle since start);
   `updated_at` is the last write. Plain JSON (not json5) so any client parses it
   with no extra dependency.

2. **HTTP** via `Network.framework` `NWListener` on `127.0.0.1:7850` (Speeker
   uses 7849). No third-party deps. Routes:
   - `GET /` → `1` or `0` (text/plain)
   - `GET /state` → the JSON document
   - `GET /health` → `ok` (200)

### Structure

```
monitor-active-calls/
  Package.swift                       # executable, platforms: .macOS(.v13)
  Sources/monitor-active-calls/
    main.swift                        # arg parsing, wiring, run loop
    CallMonitor.swift                 # CoreAudio listeners -> onChange(Bool)
    CallState.swift                   # Codable state model + JSON encoding
    StateWriter.swift                 # atomic write to state file
    HTTPServer.swift                  # NWListener minimal HTTP/1.1 responder
  Tests/monitor-active-callsTests/
    CallStateTests.swift              # encoding + transition (since/updated_at)
  launchd/com.fictorial.monitor-active-calls.plist
  install.sh                          # swift build -c release; copy binary; load agent
  README.md
```

Flags: `--port <n>` (default 7850), `--state-file <path>` (default
`~/.local/monitor-active-calls/state.json`), `--no-http`, `--verbose`.

### Components / responsibilities

- `CallMonitor`: owns CoreAudio listeners; exposes current `active` and an
  `onChange: (Bool) -> Void` callback. Knows nothing about files or HTTP.
- `CallState`: the value type + JSON; pure, trivially testable.
- `StateWriter`: turns state changes into atomic file writes.
- `HTTPServer`: serves the latest state; reads a thread-safe snapshot.
- `main`: wires monitor → (writer + server snapshot), parses flags, runs the
  CFRunLoop.

## Component B — Speeker integration (optional, off by default)

- **Config** (`src/speeker/config.py`): add section
  ```python
  "calls": {
      "pause_when_active": False,
      "state_file": "~/.local/monitor-active-calls/state.json",
  }
  ```
  and `get_calls_config()` (expands `~`).
- **Helper** `player._call_active()`: read the state file; return `False` if the
  file is missing, unreadable, or malformed (feature degrades to off). Parse
  `active`. Cheap; called once per poll (~2/s). Cache on mtime to avoid re-parsing.
- **Gate** (`player.run_daemon` loop): before `process_queue`, if
  `pause_when_active` and `_call_active()`, skip this cycle — pending items are
  left untouched and flush automatically when the call ends. Log once on each
  pause→resume / resume→pause transition.
- **Graceful absence**: missing state file ⇒ never pauses (the "optional if
  detected" behavior).
- **Visibility**:
  - `speeker status` prints `Calls: active (pausing)` / `idle` /
    `monitor not installed`.
  - Web UI settings: a "Pause while on a call" toggle bound to
    `calls.pause_when_active` (follows the existing settings save pattern;
    no daemon restart needed since the gate reads config live).

### Known limit (v1)

The gate is per poll cycle, so a call starting mid-batch finishes the current
batch before pausing. Per-utterance interruption is a later enhancement.

## Testing

- **Swift**: `CallStateTests` — JSON shape, and that a transition updates
  `since` only on change while `updated_at` always advances.
- **Speeker**: `_call_active()` against present-active / present-idle / absent /
  malformed files; the player gate holds items pending when active and processes
  when idle (mocked `process_queue` + state file).

## Verification

1. `cd ~/projects/sys/mac/monitor-active-calls && swift build -c release && swift test`.
2. Run the binary; open Photo Booth / a Zoom test call and confirm
   `curl 127.0.0.1:7850/` flips `0`→`1` and `state.json` updates.
3. Speeker: set `calls.pause_when_active`, point at a temp state file, enqueue
   items, flip the file `active:true` → daemon holds (pending count stays), flip
   back → items flush. `uv run pytest`.
