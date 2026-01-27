#!/usr/bin/env python3
"""Voice preferences ranking with drag-and-drop UI."""

import json
import socket
import subprocess
import sys
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

from .config import CONFIG_DIR
from .voices import POCKET_TTS_VOICES, KOKORO_VOICES, DEFAULT_ENGINE

SAMPLE_PHRASE = "My name is Joe and I am trapped in a bubblegum factory! Help!"
PREFS_FILE = CONFIG_DIR / "voice-prefs.json"
SAMPLES_DIR = CONFIG_DIR / "voice-samples"

# Bundled default preferences (will be populated by developer)
BUNDLED_PREFS_FILE = Path(__file__).parent / "default-voice-prefs.json"


def get_samples_dir() -> Path:
    """Get the voice samples directory."""
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    return SAMPLES_DIR


def get_voice_prefs() -> dict:
    """Load voice preferences from disk."""
    if PREFS_FILE.exists():
        try:
            with open(PREFS_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass

    # Fall back to bundled defaults
    if BUNDLED_PREFS_FILE.exists():
        try:
            with open(BUNDLED_PREFS_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass

    return {"pocket-tts": [], "kokoro": [], "default_engine": DEFAULT_ENGINE}


def save_voice_prefs(prefs: dict) -> None:
    """Save voice preferences to disk."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(PREFS_FILE, "w") as f:
        json.dump(prefs, f, indent=2)


def get_preferred_voice(engine: str) -> str | None:
    """Get the top-ranked voice for an engine."""
    prefs = get_voice_prefs()
    engine_prefs = prefs.get(engine, [])
    if engine_prefs:
        return engine_prefs[0]
    return None


def get_preferred_engine() -> str:
    """Get the preferred engine."""
    prefs = get_voice_prefs()
    return prefs.get("default_engine", DEFAULT_ENGINE)


def sample_exists(engine: str, voice: str) -> bool:
    """Check if a voice sample already exists."""
    samples_dir = get_samples_dir()
    mp3_path = samples_dir / f"{engine}-{voice}.mp3"
    wav_path = samples_dir / f"{engine}-{voice}.wav"
    return mp3_path.exists() or wav_path.exists()


def get_sample_path(engine: str, voice: str) -> Path | None:
    """Get the path to a voice sample if it exists."""
    samples_dir = get_samples_dir()
    mp3_path = samples_dir / f"{engine}-{voice}.mp3"
    wav_path = samples_dir / f"{engine}-{voice}.wav"
    if mp3_path.exists():
        return mp3_path
    if wav_path.exists():
        return wav_path
    return None


def generate_sample(engine: str, voice: str, quiet: bool = False) -> Path | None:
    """Generate a voice sample using speeker CLI."""
    samples_dir = get_samples_dir()
    output_name = f"{engine}-{voice}"

    if not quiet:
        print(f"  Generating {engine}/{voice}...", file=sys.stderr)

    try:
        # Use speeker CLI to generate audio
        result = subprocess.run(
            [
                sys.executable, "-m", "speeker.cli",
                "speak", SAMPLE_PHRASE,
                "-e", engine,
                "-v", voice,
                "--no-play",
                "-q",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            if not quiet:
                print(f"    Failed: {result.stderr}", file=sys.stderr)
            return None

        # The output is the path to the generated file
        generated_path = Path(result.stdout.strip())
        if not generated_path.exists():
            return None

        # Move to samples directory with standardized name
        dest_path = samples_dir / f"{output_name}{generated_path.suffix}"
        generated_path.rename(dest_path)

        # Also move the txt file if it exists
        txt_path = generated_path.with_suffix(".txt")
        if txt_path.exists():
            txt_path.unlink()

        return dest_path

    except subprocess.TimeoutExpired:
        if not quiet:
            print(f"    Timeout generating {engine}/{voice}", file=sys.stderr)
        return None
    except Exception as e:
        if not quiet:
            print(f"    Error: {e}", file=sys.stderr)
        return None


def ensure_all_samples(quiet: bool = False) -> dict[str, dict[str, Path]]:
    """Ensure all voice samples exist, generating missing ones."""
    samples = {"pocket-tts": {}, "kokoro": {}}

    all_voices = [
        ("pocket-tts", voice) for voice in POCKET_TTS_VOICES
    ] + [
        ("kokoro", voice) for voice in KOKORO_VOICES
    ]

    missing = [(e, v) for e, v in all_voices if not sample_exists(e, v)]

    if missing and not quiet:
        print(f"Generating {len(missing)} voice samples...", file=sys.stderr)

    for engine, voice in all_voices:
        path = get_sample_path(engine, voice)
        if path is None:
            path = generate_sample(engine, voice, quiet)
        if path:
            samples[engine][voice] = path

    return samples


def find_free_port() -> int:
    """Find a free port to use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def create_html_ui(samples: dict[str, dict[str, Path]]) -> str:
    """Generate the HTML for the voice preferences UI."""
    prefs = get_voice_prefs()

    # Build voice items for each engine
    def voice_items(engine: str, voices: dict[str, str]) -> str:
        ordered = prefs.get(engine, [])
        # Add any voices not in prefs at the end
        for v in voices:
            if v not in ordered:
                ordered.append(v)

        items = []
        for voice in ordered:
            if voice not in voices:
                continue
            desc = voices[voice]
            sample_path = samples.get(engine, {}).get(voice)
            sample_url = f"/sample/{engine}/{voice}" if sample_path else ""
            play_btn = ""
            if sample_url:
                play_btn = f'<button class="play-btn" onclick="playAudio(\'{sample_url}\')">&#9654;</button>'
            items.append(f'''
                <div class="voice-item" data-voice="{voice}" data-engine="{engine}">
                    <span class="handle">&#9776;</span>
                    <span class="voice-name">{voice}</span>
                    <span class="voice-desc">{desc}</span>
                    {play_btn}
                </div>
            ''')
        return "\n".join(items)

    pocket_items = voice_items("pocket-tts", POCKET_TTS_VOICES)
    kokoro_items = voice_items("kokoro", KOKORO_VOICES)
    default_engine = prefs.get("default_engine", DEFAULT_ENGINE)

    return f'''<!DOCTYPE html>
<html>
<head>
    <title>Speeker Voice Preferences</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background: #1a1a2e;
            color: #eee;
        }}
        h1 {{ color: #4cc9f0; text-align: center; }}
        h2 {{ color: #7b2cbf; margin-top: 30px; border-bottom: 2px solid #7b2cbf; padding-bottom: 10px; }}
        .engine-section {{
            background: #16213e;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
        }}
        .voice-list {{
            min-height: 100px;
        }}
        .voice-item {{
            display: flex;
            align-items: center;
            background: #0f3460;
            border-radius: 8px;
            padding: 12px 15px;
            margin: 8px 0;
            cursor: grab;
            transition: all 0.2s;
        }}
        .voice-item:hover {{ background: #1a4a7a; }}
        .voice-item.dragging {{
            opacity: 0.5;
            transform: scale(1.02);
        }}
        .voice-item.drag-over {{
            border: 2px dashed #4cc9f0;
        }}
        .handle {{
            color: #4cc9f0;
            margin-right: 15px;
            font-size: 18px;
        }}
        .voice-name {{
            font-weight: bold;
            min-width: 120px;
            color: #f72585;
        }}
        .voice-desc {{
            flex: 1;
            color: #aaa;
            font-size: 14px;
        }}
        .play-btn {{
            background: #4cc9f0;
            border: none;
            border-radius: 50%;
            width: 36px;
            height: 36px;
            color: #1a1a2e;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.2s;
        }}
        .play-btn:hover {{ background: #7b2cbf; color: white; }}
        .actions {{
            text-align: center;
            margin-top: 30px;
        }}
        .save-btn {{
            background: #4cc9f0;
            border: none;
            padding: 15px 40px;
            font-size: 18px;
            border-radius: 8px;
            cursor: pointer;
            color: #1a1a2e;
            font-weight: bold;
            transition: all 0.2s;
        }}
        .save-btn:hover {{ background: #7b2cbf; color: white; }}
        .status {{
            text-align: center;
            margin-top: 15px;
            color: #4cc9f0;
            height: 24px;
        }}
        .rank-badge {{
            background: #f72585;
            color: white;
            border-radius: 50%;
            width: 24px;
            height: 24px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 12px;
            margin-right: 10px;
        }}
        .engine-default {{
            display: flex;
            align-items: center;
            margin-bottom: 15px;
        }}
        .engine-default label {{
            margin-right: 10px;
        }}
        .engine-default input {{
            margin-right: 5px;
        }}
        .instructions {{
            text-align: center;
            color: #888;
            margin-bottom: 20px;
        }}
        audio {{ display: none; }}
    </style>
</head>
<body>
    <h1>Speeker Voice Preferences</h1>
    <p class="instructions">Drag voices to rank them. Top voice is used as default. Click play to preview.</p>

    <div class="engine-default">
        <label>Default Engine:</label>
        <input type="radio" name="engine" value="pocket-tts" {'checked' if default_engine == 'pocket-tts' else ''} onchange="updateEngine()"> pocket-tts (faster)
        <input type="radio" name="engine" value="kokoro" {'checked' if default_engine == 'kokoro' else ''} onchange="updateEngine()"> kokoro (higher quality)
    </div>

    <div class="engine-section">
        <h2>pocket-tts</h2>
        <div class="voice-list" id="pocket-tts-list">
            {pocket_items}
        </div>
    </div>

    <div class="engine-section">
        <h2>kokoro</h2>
        <div class="voice-list" id="kokoro-list">
            {kokoro_items}
        </div>
    </div>

    <div class="actions">
        <button class="save-btn" onclick="savePrefs()">Save Preferences</button>
    </div>
    <div class="status" id="status"></div>

    <audio id="audio-player"></audio>

    <script>
        let draggedItem = null;

        function initDragDrop() {{
            document.querySelectorAll('.voice-item').forEach(item => {{
                item.draggable = true;

                item.addEventListener('dragstart', e => {{
                    draggedItem = item;
                    item.classList.add('dragging');
                }});

                item.addEventListener('dragend', e => {{
                    item.classList.remove('dragging');
                    document.querySelectorAll('.voice-item').forEach(i => i.classList.remove('drag-over'));
                    updateRankBadges();
                }});

                item.addEventListener('dragover', e => {{
                    e.preventDefault();
                    if (item !== draggedItem && item.dataset.engine === draggedItem.dataset.engine) {{
                        item.classList.add('drag-over');
                    }}
                }});

                item.addEventListener('dragleave', e => {{
                    item.classList.remove('drag-over');
                }});

                item.addEventListener('drop', e => {{
                    e.preventDefault();
                    item.classList.remove('drag-over');
                    if (item !== draggedItem && item.dataset.engine === draggedItem.dataset.engine) {{
                        const list = item.parentNode;
                        const items = Array.from(list.children);
                        const draggedIdx = items.indexOf(draggedItem);
                        const targetIdx = items.indexOf(item);
                        if (draggedIdx < targetIdx) {{
                            item.after(draggedItem);
                        }} else {{
                            item.before(draggedItem);
                        }}
                    }}
                }});
            }});

            updateRankBadges();
        }}

        function updateRankBadges() {{
            ['pocket-tts', 'kokoro'].forEach(engine => {{
                const list = document.getElementById(engine + '-list');
                list.querySelectorAll('.voice-item').forEach((item, idx) => {{
                    let badge = item.querySelector('.rank-badge');
                    if (!badge) {{
                        badge = document.createElement('span');
                        badge.className = 'rank-badge';
                        item.insertBefore(badge, item.querySelector('.handle'));
                    }}
                    badge.textContent = idx + 1;
                }});
            }});
        }}

        function playAudio(url) {{
            if (!url) return;
            const player = document.getElementById('audio-player');
            player.src = url;
            player.play();
        }}

        function updateEngine() {{
            // Just tracks the selection, saved with preferences
        }}

        function getPrefs() {{
            const prefs = {{}};
            ['pocket-tts', 'kokoro'].forEach(engine => {{
                const list = document.getElementById(engine + '-list');
                prefs[engine] = Array.from(list.querySelectorAll('.voice-item')).map(i => i.dataset.voice);
            }});
            prefs.default_engine = document.querySelector('input[name="engine"]:checked').value;
            return prefs;
        }}

        async function savePrefs() {{
            const prefs = getPrefs();
            const status = document.getElementById('status');
            status.textContent = 'Saving...';

            try {{
                const resp = await fetch('/save', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify(prefs)
                }});
                const data = await resp.json();
                if (data.status === 'success') {{
                    status.textContent = 'Saved! You can close this window.';
                    status.style.color = '#4cc9f0';
                }} else {{
                    status.textContent = 'Error: ' + data.error;
                    status.style.color = '#f72585';
                }}
            }} catch (e) {{
                status.textContent = 'Error: ' + e.message;
                status.style.color = '#f72585';
            }}
        }}

        initDragDrop();
    </script>
</body>
</html>'''


class VoicePrefsHandler(BaseHTTPRequestHandler):
    """HTTP request handler for voice preferences UI."""

    samples: dict[str, dict[str, Path]] = {}
    html_content: str = ""

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(self.html_content.encode())

        elif parsed.path.startswith("/sample/"):
            parts = parsed.path.split("/")
            if len(parts) >= 4:
                engine, voice = parts[2], parts[3]
                sample_path = self.samples.get(engine, {}).get(voice)
                if sample_path and sample_path.exists():
                    content_type = "audio/mpeg" if sample_path.suffix == ".mp3" else "audio/wav"
                    self.send_response(200)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", str(sample_path.stat().st_size))
                    self.end_headers()
                    with open(sample_path, "rb") as f:
                        self.wfile.write(f.read())
                    return

            self.send_response(404)
            self.end_headers()

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/save":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode()

            try:
                prefs = json.loads(body)
                save_voice_prefs(prefs)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "success"}).encode())
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "error": str(e)}).encode())
        else:
            self.send_response(404)
            self.end_headers()


def run_voice_prefs_server(quiet: bool = False) -> None:
    """Run the voice preferences server and open browser."""
    if not quiet:
        print("Preparing voice samples...", file=sys.stderr)

    samples = ensure_all_samples(quiet)
    port = find_free_port()

    VoicePrefsHandler.samples = samples
    VoicePrefsHandler.html_content = create_html_ui(samples)

    server = HTTPServer(("127.0.0.1", port), VoicePrefsHandler)
    url = f"http://127.0.0.1:{port}/"

    if not quiet:
        print(f"\nVoice preferences UI running at: {url}", file=sys.stderr)
        print("Press Ctrl+C to stop.", file=sys.stderr)

    # Open browser
    webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        if not quiet:
            print("\nShutting down...", file=sys.stderr)
        server.shutdown()
