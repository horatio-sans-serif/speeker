#!/usr/bin/env python3
"""Speeker HTTP server - queues text for TTS playback."""

import argparse
import shutil
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from .queue_db import enqueue, get_pending_count, get_settings, set_settings, get_all_sessions, get_history
from .cli import start_player
from .summarize import summarize_for_speech, get_backend_info
from .web import router as web_router
from .voices import (
    POCKET_TTS_VOICES,
    KOKORO_VOICES,
    DEFAULT_ENGINE,
    DEFAULT_POCKET_TTS_VOICE,
    DEFAULT_KOKORO_VOICE,
)


def extract_metadata(request: Request) -> dict | None:
    """Extract metadata from query params with ! prefix.

    e.g., ?!foo=bar&!queue=myqueue -> {"foo": "bar", "queue": "myqueue"}
    """
    metadata = {}
    for key, value in request.query_params.items():
        if key.startswith("!"):
            metadata[key[1:]] = value
    return metadata if metadata else None


def extract_title(request: Request) -> str | None:
    """Extract title from query params (non-metadata param)."""
    return request.query_params.get("title")


def format_with_title(text: str, title: str | None) -> str:
    """Format text with optional title prefix and attention tone."""
    if title:
        return f"$Eb4 {title}. {text}"
    return text


def elide_message_count(text: str) -> str:
    """Remove 'there is/are N message(s)' type phrases."""
    import re
    patterns = [
        r"there (?:is|are) \d+ messages?(?: waiting| pending| queued)?\.?\s*",
        r"you have \d+ (?:new )?messages?(?: waiting| pending| queued)?\.?\s*",
        r"\d+ messages? (?:waiting|pending|queued)\.?\s*",
        r"(?:waiting|pending|queued)\.\s*",  # Standalone remnants
    ]
    result = text
    for pattern in patterns:
        result = re.sub(pattern, "", result, flags=re.IGNORECASE)
    return result.strip()


class SpeakRequest(BaseModel):
    text: str
    metadata: dict | None = None
    session_id: str | None = None  # Deprecated


class SpeakResponse(BaseModel):
    status: str
    message: str | None = None
    queue_id: int | None = None
    pending_count: int | None = None
    error: str | None = None


class SummarizeRequest(BaseModel):
    text: str
    metadata: dict | None = None
    session_id: str | None = None  # Deprecated


class SummarizeResponse(BaseModel):
    status: str
    summary: str | None = None
    queue_id: int | None = None
    original_length: int | None = None
    summary_length: int | None = None
    pending_count: int | None = None
    error: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup checks."""
    print("Speeker server starting...", file=sys.stderr)

    # Check for player binary
    player_cmd = shutil.which("speeker-player")
    if not player_cmd:
        for path in [
            Path.home() / ".local/bin/speeker-player",
            Path("/usr/local/bin/speeker-player"),
        ]:
            if path.exists():
                player_cmd = str(path)
                break

    if player_cmd:
        print(f"  Player found: {player_cmd}", file=sys.stderr)
    else:
        print("  WARNING: speeker-player not found in PATH", file=sys.stderr)

    print("Speeker server ready!", file=sys.stderr)
    yield


app = FastAPI(title="Speeker TTS Server", lifespan=lifespan)

# Mount web UI
app.include_router(web_router)


@app.post("/speak", response_model=SpeakResponse)
async def speak(body: SpeakRequest, request: Request):
    """Queue text for TTS playback.

    Accepts metadata via:
    - JSON body: {"text": "...", "metadata": {"key": "value"}}
    - Query params: ?!key=value&!another=thing
    Query params are merged with body metadata (query params take precedence).

    Non-metadata query params:
    - title: Spoken before the text as "$Eb3 $Eb3 {title}. {text}"
    """
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        # Merge metadata from body and query params
        metadata = body.metadata.copy() if body.metadata else {}
        query_metadata = extract_metadata(request)
        if query_metadata:
            metadata.update(query_metadata)

        # Handle deprecated session_id (maps to queue)
        if body.session_id and "queue" not in metadata:
            metadata["queue"] = body.session_id

        # Apply title prefix if provided
        title = extract_title(request)
        text = elide_message_count(text)
        text = format_with_title(text, title)

        # Queue the text for playback
        queue_id = enqueue(text, metadata=metadata if metadata else None)

        # Start player if not running
        start_player()

        return SpeakResponse(
            status="success",
            message="Queued for playback",
            queue_id=queue_id,
            pending_count=get_pending_count(),
        )

    except Exception as e:
        return SpeakResponse(
            status="error",
            error=str(e),
        )


@app.post("/summarize", response_model=SummarizeResponse)
async def summarize_and_speak(body: SummarizeRequest, request: Request):
    """Summarize text and queue for TTS playback.

    Uses configured LLM backend to generate a short, speakable summary.
    Falls back to heuristic extraction if no LLM is configured.

    Accepts metadata via query params: ?!key=value

    Non-metadata query params:
    - title: Spoken before the summary as "$Eb3 $Eb3 {title}. {summary}"
    """
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        # Merge metadata from body and query params
        metadata = body.metadata.copy() if body.metadata else {}
        query_metadata = extract_metadata(request)
        if query_metadata:
            metadata.update(query_metadata)

        # Handle deprecated session_id (maps to queue)
        if body.session_id and "queue" not in metadata:
            metadata["queue"] = body.session_id

        # Generate summary
        summary = summarize_for_speech(text)
        summary = elide_message_count(summary)

        # Apply title prefix if provided
        title = extract_title(request)
        spoken_text = format_with_title(summary, title)

        # Queue the summary for playback
        queue_id = enqueue(spoken_text, metadata=metadata if metadata else None)

        # Start player if not running
        start_player()

        return SummarizeResponse(
            status="success",
            summary=summary,
            queue_id=queue_id,
            original_length=len(text),
            summary_length=len(summary),
            pending_count=get_pending_count(),
        )

    except Exception as e:
        return SummarizeResponse(
            status="error",
            summary=None,
            error=str(e),
        )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/summarize/info")
async def summarize_info():
    """Get information about the configured summarization backend."""
    return get_backend_info()


@app.get("/queue/status")
async def queue_status():
    """Get queue status."""
    return {
        "pending_count": get_pending_count(),
    }


@app.get("/voices")
async def get_voices(engine: str | None = None):
    """Get available TTS voices."""
    def format_voices(voice_dict: dict[str, str], default_voice: str) -> dict:
        return {
            "default": default_voice,
            "voices": {
                name: {"description": desc, "is_default": name == default_voice}
                for name, desc in voice_dict.items()
            },
        }

    engines = {}
    if engine is None or engine == "pocket-tts":
        engines["pocket-tts"] = format_voices(POCKET_TTS_VOICES, DEFAULT_POCKET_TTS_VOICE)
    if engine is None or engine == "kokoro":
        engines["kokoro"] = format_voices(KOKORO_VOICES, DEFAULT_KOKORO_VOICE)

    if engine and engine not in ("pocket-tts", "kokoro"):
        return {"status": "error", "error": f"Unknown engine: {engine}"}

    return {
        "status": "success",
        "default_engine": DEFAULT_ENGINE,
        "engines": engines,
    }


def main():
    """Run the server."""
    parser = argparse.ArgumentParser(
        prog="speeker-server",
        description="Speeker TTS HTTP server - queues text for playback",
    )
    parser.add_argument("-p", "--port", type=int, default=7849, help="Port to listen on")
    parser.add_argument("-H", "--host", default="127.0.0.1", help="Host to bind to")
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
