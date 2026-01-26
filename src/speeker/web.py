"""Web UI for viewing TTS queue history."""

from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, HTMLResponse

from .queue_db import get_all_sessions, get_history, get_settings, set_settings

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
            background: #1a1a2e;
            color: #eee;
        }
        h1 { color: #00d9ff; margin-bottom: 10px; }
        h2 { color: #888; font-weight: normal; margin-top: 0; }
        .sessions {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-bottom: 20px;
        }
        .session-btn {
            padding: 8px 16px;
            border: 1px solid #333;
            background: #252540;
            color: #eee;
            border-radius: 20px;
            cursor: pointer;
            text-decoration: none;
            transition: all 0.2s;
        }
        .session-btn:hover { background: #353560; border-color: #00d9ff; }
        .session-btn.active { background: #00d9ff; color: #000; }
        .session-btn .count {
            background: #333;
            padding: 2px 8px;
            border-radius: 10px;
            margin-left: 8px;
            font-size: 0.85em;
        }
        .session-btn.active .count { background: rgba(0,0,0,0.2); }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #333;
        }
        th { color: #888; font-weight: normal; }
        tr:hover { background: #252540; }
        .text-cell {
            max-width: 500px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .text-cell:hover {
            white-space: normal;
            word-break: break-word;
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
        .play-btn:disabled { background: #555; color: #888; cursor: not-allowed; }
        .time { color: #888; font-size: 0.9em; }
        .status {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 0.85em;
        }
        .status.played { background: #1b5e20; color: #a5d6a7; }
        .status.pending { background: #e65100; color: #ffcc80; }
        .settings-form {
            background: #252540;
            padding: 20px;
            border-radius: 8px;
            margin-top: 20px;
        }
        .settings-form label {
            display: block;
            margin-bottom: 15px;
        }
        .settings-form input, .settings-form select {
            margin-left: 10px;
            padding: 5px;
            background: #1a1a2e;
            border: 1px solid #333;
            color: #eee;
            border-radius: 4px;
        }
        .settings-form button {
            background: #00d9ff;
            color: #000;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
        }
        audio { display: none; }
    </style>
</head>
<body>
    <h1>Speeker</h1>
    <h2>TTS Queue History</h2>

    <div class="sessions">
        <a href="/" class="session-btn {all_active}">All <span class="count">{total_count}</span></a>
        {session_buttons}
    </div>

    <table>
        <thead>
            <tr>
                <th>Session</th>
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


@router.get("/", response_class=HTMLResponse)
async def index(session: str | None = None):
    """Main page showing queue history."""
    sessions = get_all_sessions()
    history = get_history(session_id=session, limit=200)

    # Build session buttons
    total_count = sum(s["total_messages"] for s in sessions)
    session_buttons = []
    for s in sessions:
        active = "active" if session == s["session_id"] else ""
        short_id = s["session_id"][:8] if len(s["session_id"]) > 8 else s["session_id"]
        session_buttons.append(
            f'<a href="/?session={s["session_id"]}" class="session-btn {active}">'
            f'{short_id} <span class="count">{s["total_messages"]}</span></a>'
        )

    # Build table rows
    rows = []
    for item in history:
        short_session = item["session_id"][:8] if len(item["session_id"]) > 8 else item["session_id"]
        status_class = "played" if item["played_at"] else "pending"
        status_text = "Played" if item["played_at"] else "Pending"
        has_audio = bool(item["audio_path"]) and Path(item["audio_path"]).exists() if item["audio_path"] else False

        play_btn = (
            f'<button class="play-btn" onclick="playAudio({item["id"]})">Play</button>'
            if has_audio
            else '<button class="play-btn" disabled>No audio</button>'
        )

        # Escape text for HTML
        text_escaped = item['text'].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')

        rows.append(f"""
            <tr>
                <td>{short_session}</td>
                <td class="text-cell" title="{text_escaped}">{text_escaped}</td>
                <td class="time">{format_time(item['created_at'])}</td>
                <td><span class="status {status_class}">{status_text}</span></td>
                <td>{play_btn}</td>
            </tr>
        """)

    # Use % formatting to avoid issues with CSS braces
    html = HTML_TEMPLATE.replace("{all_active}", "active" if session is None else "")
    html = html.replace("{total_count}", str(total_count))
    html = html.replace("{session_buttons}", " ".join(session_buttons))
    html = html.replace("{rows}", "\n".join(rows) if rows else "<tr><td colspan='5'>No messages yet</td></tr>")

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
                background: #1a1a2e;
                color: #eee;
            }}
            h1 {{ color: #00d9ff; }}
            form {{ background: #252540; padding: 20px; border-radius: 8px; }}
            label {{ display: block; margin-bottom: 15px; }}
            input, select {{ margin-left: 10px; padding: 5px; background: #1a1a2e; border: 1px solid #333; color: #eee; border-radius: 4px; }}
            button {{ background: #00d9ff; color: #000; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; }}
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
