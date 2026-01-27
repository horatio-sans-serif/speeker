"""Web UI for viewing TTS queue history."""

from datetime import datetime
from pathlib import Path

import hashlib
import json

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from .queue_db import get_history, get_settings, set_settings, search
from .voices import POCKET_TTS_VOICES, KOKORO_VOICES

router = APIRouter()

# Simple HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Speeker Queue History</title>
    <style>
        * { box-sizing: border-box; }
        *::-webkit-scrollbar { width: 6px; height: 6px; }
        *::-webkit-scrollbar-track { background: transparent; }
        *::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.2); border-radius: 3px; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #0a0a0f;
            color: rgba(255, 255, 255, 0.75);
        }
        .header {
            display: flex;
            align-items: baseline;
            justify-content: space-between;
            margin-bottom: 20px;
        }
        .header-left {
            display: flex;
            align-items: baseline;
            gap: 15px;
        }
        h1 {
            color: #00d9ff;
            margin: 0;
        }
        .subtitle {
            color: rgba(255, 255, 255, 0.5);
            font-size: 1em;
        }
        .search-box input[type="text"] {
            padding: 8px 12px;
            border: 1px solid #222;
            background: #1e1e2a;
            color: rgba(255, 255, 255, 0.75);
            border-radius: 4px;
            width: 250px;
            font-size: 14px;
        }
        .search-box input[type="text"]:focus {
            outline: none;
            border-color: #00d9ff;
        }
        .search-box input[type="text"]::placeholder {
            color: rgba(255, 255, 255, 0.3);
        }
        .cards-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
            gap: 16px;
        }
        .cards-grid.playing .card:not(.playing) {
            opacity: 0.4;
            pointer-events: none;
        }
        .card {
            background: #1e1e2a;
            border: 2px solid transparent;
            border-radius: 8px;
            padding: 16px;
            display: flex;
            flex-direction: column;
            gap: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
            transition: opacity 0.2s, border-color 0.2s;
        }
        .card:hover {
            box-shadow: 0 6px 16px rgba(0, 0, 0, 0.5);
        }
        .card.playing {
            animation: border-pulse 1.5s ease-in-out infinite;
        }
        @keyframes border-pulse {
            0%, 100% { border-color: #00d9ff; }
            50% { border-color: #00ffaa; }
        }
        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .card-text {
            font-size: 0.9em;
            max-height: 8em;
            overflow-y: auto;
            white-space: pre-wrap;
            word-break: break-word;
            line-height: 1.5;
            flex: 1;
        }
        .card-footer {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-top: 8px;
            border-top: 1px solid #2a2a3a;
        }
        .card-meta {
            display: flex;
            align-items: flex-start;
            gap: 12px;
            flex: 1;
            min-width: 0;
        }
        .metadata {
            font-size: 0.7em;
            line-height: 1.2;
            color: rgba(255, 255, 255, 0.4);
            font-family: monospace;
            max-height: 50px;
            overflow-y: auto;
            flex: 1;
            min-width: 0;
        }
        .metadata .kv { white-space: nowrap; }
        .metadata .key { color: rgba(255, 255, 255, 0.35); }
        .metadata .value { color: rgba(255, 255, 255, 0.5); margin-right: 8px; }
        .play-btn {
            background: #00d9ff;
            color: #000;
            border: none;
            width: 32px;
            height: 32px;
            border-radius: 50%;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
        }
        .play-btn:hover { background: #00b8d4; }
        .play-btn:disabled {
            background: #333;
            color: rgba(255, 255, 255, 0.2);
            cursor: not-allowed;
        }
        .play-btn svg { width: 14px; height: 14px; fill: currentColor; }
        .time {
            color: rgba(255, 255, 255, 0.5);
            font-size: 0.85em;
            white-space: nowrap;
        }
        .status {
            font-size: 0.85em;
        }
        .status.played {
            color: rgba(255, 255, 255, 0.35);
        }
        .status.pending {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 10px;
            background: #2d1a0a;
            color: #fb923c;
        }
        .score {
            color: #00d9ff;
            font-size: 0.8em;
            margin-left: 6px;
        }
        .no-results {
            text-align: center;
            color: rgba(255, 255, 255, 0.4);
            padding: 40px;
            grid-column: 1 / -1;
        }
        .no-data {
            color: rgba(255, 255, 255, 0.3);
            font-style: italic;
        }
        audio { display: none; }
    </style>
</head>
<body>
    <div class="header">
        <div class="header-left">
            <h1>Speeker</h1>
            <span class="subtitle">TTS Queue History</span>
        </div>
        <div class="search-box">
            <input type="text" id="search" placeholder="Search..." value="{query}">
        </div>
    </div>

    <div class="cards-grid" id="cards-grid">
        {rows}
    </div>

    <audio id="player"></audio>

    <script>
        const player = document.getElementById('player');
        const grid = document.getElementById('cards-grid');
        let currentCard = null;

        function playAudio(id, btn) {
            const card = btn.closest('.card');
            if (currentCard) currentCard.classList.remove('playing');
            currentCard = card;
            card.classList.add('playing');
            grid.classList.add('playing');
            player.src = '/audio/' + id;
            player.play();
        }

        player.addEventListener('ended', () => {
            if (currentCard) currentCard.classList.remove('playing');
            currentCard = null;
            grid.classList.remove('playing');
        });

        player.addEventListener('pause', () => {
            if (currentCard) currentCard.classList.remove('playing');
            currentCard = null;
            grid.classList.remove('playing');
        });

        // Debounced search
        let searchTimeout;
        const searchInput = document.getElementById('search');
        searchInput.addEventListener('keyup', function() {
            clearTimeout(searchTimeout);
            searchTimeout = setTimeout(() => {
                const q = searchInput.value.trim();
                const url = q ? '/?q=' + encodeURIComponent(q) : '/';
                window.location.href = url;
            }, 300);
        });

        // Real-time updates
        let lastHash = '{items_hash}';
        async function checkUpdates() {
            if (searchInput.value.trim()) return; // Skip polling during search
            try {
                const resp = await fetch('/api/items');
                const data = await resp.json();
                if (data.hash !== lastHash) {
                    lastHash = data.hash;
                    updateCards(data.items);
                }
            } catch (e) {}
        }

        function updateCards(items) {
            const playingId = currentCard ? currentCard.dataset.id : null;
            let html = '';
            for (const item of items) {
                const statusClass = item.played ? 'played' : 'pending';
                const statusText = item.played ? 'Played' : 'Pending';
                const playIcon = '<svg viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>';
                const playBtn = item.has_audio
                    ? `<button class="play-btn" onclick="playAudio(${item.id}, this)">${playIcon}</button>`
                    : `<button class="play-btn" disabled>${playIcon}</button>`;
                const isPlaying = playingId === String(item.id);
                html += `
                    <div class="card${isPlaying ? ' playing' : ''}" data-id="${item.id}">
                        <div class="card-header">
                            <span class="status ${statusClass}">${statusText}</span>
                            <span class="time">${item.time}</span>
                        </div>
                        <div class="card-text">${item.text}</div>
                        <div class="card-footer">
                            <div class="card-meta">
                                <div class="metadata">${item.metadata}</div>
                            </div>
                            ${playBtn}
                        </div>
                    </div>`;
            }
            grid.innerHTML = html || '<div class="no-results">No messages yet</div>';
            if (playingId) {
                currentCard = grid.querySelector(`[data-id="${playingId}"]`);
            }
        }

        setInterval(checkUpdates, 2000);
    </script>
</body>
</html>
"""


def format_time(iso_str: str | None) -> str:
    """Format ISO timestamp for display."""
    if not iso_str:
        return "-"
    try:
        dt = datetime.fromisoformat(iso_str)
        return dt.strftime("%b %d %H:%M")
    except (ValueError, TypeError):
        return iso_str


def escape_html(text: str) -> str:
    """Escape text for HTML display."""
    return (
        text.replace('&', '&amp;')
        .replace('<', '&lt;')
        .replace('>', '&gt;')
        .replace('"', '&quot;')
    )


def sanitize_key(key: str) -> str:
    """Sanitize metadata key for display."""
    return escape_html(str(key))


def sanitize_value(value) -> str:
    """Sanitize metadata value for display."""
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        import json
        return escape_html(json.dumps(value, default=str))
    return escape_html(str(value))


def render_metadata(metadata: dict | None) -> str:
    """Render metadata as inline key-value pairs."""
    if not metadata:
        return '<span class="no-data">-</span>'

    pairs = []
    for key, value in metadata.items():
        pairs.append(
            f'<span class="kv"><span class="key">{sanitize_key(key)}:</span> '
            f'<span class="value">{sanitize_value(value)}</span></span>'
        )

    return " ".join(pairs)


@router.get("/", response_class=HTMLResponse)
async def index(q: str | None = None):
    """Main page showing queue history with search."""
    # Get items
    if q and q.strip():
        items = search(q.strip(), limit=200)
        show_score = True
    else:
        items = get_history(limit=200)
        show_score = False

    # Build table rows
    rows = []
    for item in items:
        status_class = "played" if item["played_at"] else "pending"
        status_text = "Played" if item["played_at"] else "Pending"
        has_audio = bool(item["audio_path"]) and Path(item["audio_path"]).exists() if item["audio_path"] else False

        play_icon = '<svg viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>'
        play_btn = (
            f'<button class="play-btn" onclick="playAudio({item["id"]}, this)">{play_icon}</button>'
            if has_audio
            else f'<button class="play-btn" disabled>{play_icon}</button>'
        )

        text_escaped = escape_html(item['text'])

        # Show score if searching
        score_html = ""
        if show_score and "score" in item:
            score_html = f' <span class="score">({item["score"]:.2f})</span>'

        # Render metadata
        metadata_html = render_metadata(item.get("metadata"))

        rows.append(f"""
            <div class="card" data-id="{item['id']}">
                <div class="card-header">
                    <span class="status {status_class}">{status_text}</span>{score_html}
                    <span class="time">{format_time(item['created_at'])}</span>
                </div>
                <div class="card-text">{text_escaped}</div>
                <div class="card-footer">
                    <div class="card-meta">
                        <div class="metadata">{metadata_html}</div>
                    </div>
                    {play_btn}
                </div>
            </div>
        """)

    # Compute hash of items for change detection
    items_hash = hashlib.md5(json.dumps([(i["id"], i["played_at"]) for i in items]).encode()).hexdigest()[:8]

    html = HTML_TEMPLATE.replace("{query}", escape_html(q) if q else "")
    html = html.replace("{rows}", "\n".join(rows) if rows else '<div class="no-results">No messages yet</div>')
    html = html.replace("{items_hash}", items_hash)

    return HTMLResponse(content=html)


@router.get("/audio/{item_id}")
async def get_audio(item_id: int):
    """Serve audio file for a queue item."""
    history = get_history(limit=1000)
    for item in history:
        if item["id"] == item_id and item["audio_path"]:
            audio_path = Path(item["audio_path"])
            if audio_path.exists():
                return FileResponse(audio_path, media_type="audio/wav")

    return HTMLResponse(content="Audio not found", status_code=404)


@router.get("/api/items")
async def api_items():
    """JSON endpoint for real-time updates."""
    items = get_history(limit=200)
    items_hash = hashlib.md5(json.dumps([(i["id"], i["played_at"]) for i in items]).encode()).hexdigest()[:8]

    result = []
    for item in items:
        has_audio = bool(item["audio_path"]) and Path(item["audio_path"]).exists() if item["audio_path"] else False
        result.append({
            "id": item["id"],
            "text": escape_html(item["text"]),
            "time": format_time(item["created_at"]),
            "played": bool(item["played_at"]),
            "has_audio": has_audio,
            "metadata": render_metadata(item.get("metadata")),
        })

    return JSONResponse({"hash": items_hash, "items": result})


@router.get("/settings")
async def settings_page(session: str | None = None):
    """Settings page."""
    settings = get_settings(session)
    target = session or "Global"

    # Build voice options for each engine
    pocket_options = []
    for voice, desc in POCKET_TTS_VOICES.items():
        selected = 'selected' if settings['voice'] == voice else ''
        pocket_options.append(f'<option value="{voice}" {selected}>{voice} - {desc}</option>')

    kokoro_options = []
    for voice, desc in KOKORO_VOICES.items():
        selected = 'selected' if settings['voice'] == voice else ''
        kokoro_options.append(f'<option value="{voice}" {selected}>{voice} - {desc}</option>')

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Speeker Settings</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, sans-serif;
                max-width: 600px;
                margin: 50px auto;
                padding: 20px;
                background: #0a0a0f;
                color: rgba(255, 255, 255, 0.75);
            }}
            h1 {{ color: #00d9ff; }}
            form {{ background: #151520; padding: 20px; border-radius: 8px; }}
            .field {{ margin-bottom: 20px; }}
            .field label {{ display: block; margin-bottom: 8px; font-weight: bold; }}
            input, select {{
                padding: 8px;
                background: #0a0a0f;
                border: 1px solid #222;
                color: rgba(255, 255, 255, 0.75);
                border-radius: 4px;
                width: 100%;
            }}
            select {{ width: 100%; }}
            input[type="checkbox"] {{ width: auto; }}
            button {{
                background: #00d9ff;
                color: #000;
                border: none;
                padding: 10px 20px;
                border-radius: 4px;
                cursor: pointer;
            }}
            a {{ color: #00d9ff; }}
            optgroup {{ font-weight: bold; }}
        </style>
    </head>
    <body>
        <h1>Settings: {target}</h1>
        <form method="POST">
            <div class="field">
                <label>Intro/Outro Sound:</label>
                <input type="checkbox" name="intro_sound" {'checked' if settings['intro_sound'] else ''}>
            </div>
            <div class="field">
                <label>Speed:</label>
                <input type="number" name="speed" value="{settings['speed']}" min="0.5" max="2.0" step="0.1">
            </div>
            <div class="field">
                <label>Engine:</label>
                <select name="engine">
                    <option value="pocket-tts" {'selected' if settings.get('engine') == 'pocket-tts' else ''}>pocket-tts (faster)</option>
                    <option value="kokoro" {'selected' if settings.get('engine') == 'kokoro' else ''}>kokoro (higher quality)</option>
                </select>
            </div>
            <div class="field">
                <label>Voice:</label>
                <select name="voice">
                    <optgroup label="pocket-tts">
                        {''.join(pocket_options)}
                    </optgroup>
                    <optgroup label="kokoro">
                        {''.join(kokoro_options)}
                    </optgroup>
                </select>
            </div>
            <button type="submit">Save</button>
        </form>
        <p><a href="/">Back to history</a> | <a href="/settings">Global Settings</a></p>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


@router.post("/settings")
async def save_settings(request: Request, session: str | None = None):
    """Save settings."""
    form = await request.form()
    set_settings(
        session_id=session,
        intro_sound="intro_sound" in form,
        speed=float(form.get("speed", 1.0)),
        voice=form.get("voice"),
        engine=form.get("engine"),
    )
    return HTMLResponse(
        content='<script>alert("Settings saved!"); window.location="/";</script>'
    )
