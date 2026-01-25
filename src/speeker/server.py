#!/usr/bin/env python3
"""Speeker HTTP server - queues text for TTS playback."""

import argparse
import shutil
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .queue_db import enqueue, get_pending_count
from .cli import start_player
from .summarize import summarize_for_speech, get_backend_info


class SpeakRequest(BaseModel):
    text: str
    session_id: str = "default"


class SpeakResponse(BaseModel):
    status: str
    message: str | None = None
    queue_id: int | None = None
    pending_count: int | None = None
    error: str | None = None


class SummarizeRequest(BaseModel):
    text: str
    session_id: str = "default"


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


@app.post("/speak", response_model=SpeakResponse)
async def speak(request: SpeakRequest):
    """Queue text for TTS playback."""
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        # Queue the text for playback
        queue_id = enqueue(request.session_id, text)

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
async def summarize_and_speak(request: SummarizeRequest):
    """Summarize text and queue for TTS playback.

    Uses configured LLM backend to generate a short, speakable summary.
    Falls back to heuristic extraction if no LLM is configured.
    """
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        # Generate summary
        summary = summarize_for_speech(text)

        # Queue the summary for playback
        queue_id = enqueue(request.session_id, summary)

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
