"""Web UI for viewing TTS queue history."""

from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, HTMLResponse

from .config import is_semantic_search_enabled
from .queue_db import get_history, get_settings, set_settings, search

router = APIRouter()

# Simple HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Speeker Queue History</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
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
        .search-box {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .search-box input[type="text"] {
            padding: 8px 12px;
            border: 1px solid #222;
            background: #151520;
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
        .search-box button {
            padding: 8px 16px;
            background: #00d9ff;
            color: #000;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }
        .search-box button:hover {
            background: #00b8d4;
        }
        .search-mode {
            color: rgba(255, 255, 255, 0.4);
            font-size: 0.8em;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #1a1a25;
        }
        th {
            color: rgba(255, 255, 255, 0.5);
            font-weight: normal;
        }
        tr:hover {
            background: #101018;
        }
        .text-cell {
            font-size: 0.85em;
            max-height: 10em;
            overflow-y: auto;
            white-space: pre-wrap;
            word-break: break-word;
            line-height: 1.4;
        }
        .data-cell details {
            font-size: 0.8em;
        }
        .data-cell summary {
            cursor: pointer;
            color: rgba(255, 255, 255, 0.5);
            font-family: monospace;
        }
        .data-cell summary:hover {
            color: rgba(255, 255, 255, 0.75);
        }
        .data-cell table {
            margin-top: 8px;
            font-size: 0.9em;
        }
        .data-cell table td {
            padding: 4px 8px;
            border-bottom: 1px solid #1a1a25;
        }
        .data-cell .key {
            color: rgba(255, 255, 255, 0.5);
            font-family: monospace;
        }
        .data-cell .value {
            color: rgba(255, 255, 255, 0.75);
            font-family: monospace;
            word-break: break-all;
        }
        .play-btn {
            background: #00d9ff;
            color: #000;
            border: none;
            padding: 6px 12px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }
        .play-btn:hover { background: #00b8d4; }
        .play-btn:disabled {
            background: #222;
            color: rgba(255, 255, 255, 0.3);
            cursor: not-allowed;
        }
        .time {
            color: rgba(255, 255, 255, 0.5);
            font-size: 0.9em;
            white-space: nowrap;
        }
        .status {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 0.85em;
        }
        .status.played {
            background: #0d2818;
            color: #4ade80;
        }
        .status.pending {
            background: #2d1a0a;
            color: #fb923c;
        }
        .score {
            color: #00d9ff;
            font-size: 0.8em;
        }
        .no-results {
            text-align: center;
            color: rgba(255, 255, 255, 0.4);
            padding: 40px;
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
        <form class="search-box" method="GET" action="/">
            <input type="text" name="q" placeholder="Search..." value="{query}">
            <button type="submit">Search</button>
            <span class="search-mode">{search_mode}</span>
        </form>
    </div>

    <table>
        <thead>
            <tr>
                <th>Data</th>
                <th>Text</th>
                <th>Created</th>
                <th>Status</th>
                <th>Play</th>
            </tr>
        </thead>
        <tbody>
            {rows}
        </tbody>
    </table>

    <audio id="player"></audio>

    <script>
        function playAudio(id) {
            const player = document.getElementById('player');
            player.src = '/audio/' + id;
            player.play();
        }
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
    """Render metadata as a details/summary element with key-value table."""
    if not metadata:
        return '<span class="no-data">-</span>'

    # Create summary text (first key or count)
    if len(metadata) == 1:
        key = list(metadata.keys())[0]
        summary_text = f"{sanitize_key(key)[:12]}"
    else:
        summary_text = f"{len(metadata)} items"

    # Build key-value table
    rows = []
    for key, value in metadata.items():
        rows.append(
            f'<tr><td class="key">{sanitize_key(key)}</td>'
            f'<td class="value">{sanitize_value(value)}</td></tr>'
        )

    return f'''<details>
        <summary>{summary_text}</summary>
        <table>{"".join(rows)}</table>
    </details>'''


@router.get("/", response_class=HTMLResponse)
async def index(q: str | None = None):
    """Main page showing queue history with search."""
    # Determine search mode
    semantic_enabled = is_semantic_search_enabled()
    search_mode = "semantic" if semantic_enabled else "fuzzy"

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

        play_btn = (
            f'<button class="play-btn" onclick="playAudio({item["id"]})">Play</button>'
            if has_audio
            else '<button class="play-btn" disabled>No audio</button>'
        )

        text_escaped = escape_html(item['text'])

        # Show score if searching
        score_html = ""
        if show_score and "score" in item:
            score_html = f' <span class="score">({item["score"]:.2f})</span>'

        # Render metadata
        metadata_html = render_metadata(item.get("metadata"))

        rows.append(f"""
            <tr>
                <td class="data-cell">{metadata_html}</td>
                <td class="text-cell">{text_escaped}</td>
                <td class="time">{format_time(item['created_at'])}</td>
                <td><span class="status {status_class}">{status_text}</span>{score_html}</td>
                <td>{play_btn}</td>
            </tr>
        """)

    html = HTML_TEMPLATE.replace("{query}", escape_html(q) if q else "")
    html = html.replace("{search_mode}", search_mode)
    html = html.replace("{rows}", "\n".join(rows) if rows else '<tr><td colspan="5" class="no-results">No messages yet</td></tr>')

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


@router.get("/settings")
async def settings_page(session: str | None = None):
    """Settings page."""
    settings = get_settings(session)
    target = session or "Global"

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
            label {{ display: block; margin-bottom: 15px; }}
            input, select {{
                margin-left: 10px;
                padding: 5px;
                background: #0a0a0f;
                border: 1px solid #222;
                color: rgba(255, 255, 255, 0.75);
                border-radius: 4px;
            }}
            button {{
                background: #00d9ff;
                color: #000;
                border: none;
                padding: 10px 20px;
                border-radius: 4px;
                cursor: pointer;
            }}
            a {{ color: #00d9ff; }}
        </style>
    </head>
    <body>
        <h1>Settings: {target}</h1>
        <form method="POST">
            <label>
                Intro/Outro Sound:
                <input type="checkbox" name="intro_sound" {'checked' if settings['intro_sound'] else ''}>
            </label>
            <label>
                Speed:
                <input type="number" name="speed" value="{settings['speed']}" min="0.5" max="2.0" step="0.1">
            </label>
            <label>
                Voice:
                <select name="voice">
                    <option value="azelma" {'selected' if settings['voice'] == 'azelma' else ''}>Azelma</option>
                    <option value="javert" {'selected' if settings['voice'] == 'javert' else ''}>Javert</option>
                    <option value="marius" {'selected' if settings['voice'] == 'marius' else ''}>Marius</option>
                </select>
            </label>
            <button type="submit">Save</button>
        </form>
        <p><a href="/">Back to history</a></p>
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
        voice=form.get("voice", "azelma"),
    )
    return HTMLResponse(
        content='<script>alert("Settings saved!"); window.location="/";</script>'
    )
