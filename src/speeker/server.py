#!/usr/bin/env python3
"""Speeker HTTP server - keeps TTS models warm for fast responses."""

import argparse
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from pathlib import Path

from .cli import (
    generate_kokoro,
    generate_pocket_tts,
    get_base_dir,
    get_kokoro_pipeline,
    get_pocket_tts_model,
    queue_for_playback,
    save_audio,
    start_player,
)
from .preprocessing import preprocess_for_tts
from .summarize import summarize_for_speech, get_backend_info
from .voices import DEFAULT_ENGINE, get_default_voice, get_voices, validate_voice


class SpeakRequest(BaseModel):
    text: str
    engine: str | None = None
    voice: str | None = None
    session_id: str | None = None  # For per-session queues


class SpeakResponse(BaseModel):
    status: str
    engine: str
    voice: str
    audio_file: str | None = None
    error: str | None = None


# Track which engines are available
_available_engines: set[str] = set()


def queue_for_session(audio_path: Path, session_id: str | None) -> None:
    """Queue audio to a session-specific queue file."""
    import fcntl

    base_dir = get_base_dir()

    if session_id:
        # Per-session queue
        session_dir = base_dir / "sessions" / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        queue_file = session_dir / "queue"
    else:
        # Default global queue
        queue_file = base_dir / "queue"

    with open(queue_file, "a") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(f"{audio_path}\n")
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    start_player()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up TTS models on startup."""
    print("Warming up TTS models...", file=sys.stderr)

    # Warm up pocket-tts
    try:
        print("  Loading pocket-tts...", file=sys.stderr)
        get_pocket_tts_model()
        print("  pocket-tts ready", file=sys.stderr)
        _available_engines.add("pocket-tts")
    except Exception as e:
        print(f"  pocket-tts failed: {e}", file=sys.stderr)

    # Kokoro warmup disabled - requires spacy models which are tricky in uv tool env
    # To enable: install spacy model, then uncomment below
    # try:
    #     print("  Loading kokoro...", file=sys.stderr)
    #     get_kokoro_pipeline()
    #     print("  kokoro ready", file=sys.stderr)
    #     _available_engines.add("kokoro")
    # except Exception as e:
    #     print(f"  kokoro failed: {e}", file=sys.stderr)
    print("  kokoro: skipped (requires spacy model setup)", file=sys.stderr)

    if not _available_engines:
        print("ERROR: No TTS engines available!", file=sys.stderr)
        sys.exit(1)

    print(f"TTS ready! Engines: {', '.join(sorted(_available_engines))}", file=sys.stderr)
    yield


app = FastAPI(title="Speeker TTS Server", lifespan=lifespan)


@app.post("/speak", response_model=SpeakResponse)
async def speak(request: SpeakRequest):
    """Generate speech and queue for playback."""
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    engine = request.engine or DEFAULT_ENGINE
    voice = request.voice or get_default_voice(engine)

    if engine not in ("pocket-tts", "kokoro"):
        raise HTTPException(status_code=400, detail=f"Unknown engine: {engine}")

    if engine not in _available_engines:
        # Fall back to available engine
        engine = next(iter(_available_engines))
        voice = get_default_voice(engine)

    if not validate_voice(engine, voice):
        available = list(get_voices(engine).get(engine, {}).keys())
        raise HTTPException(
            status_code=400,
            detail=f"Unknown voice '{voice}' for engine '{engine}'. Available: {available}",
        )

    try:
        # Preprocess text for better TTS output
        processed_text = preprocess_for_tts(text)

        if engine == "pocket-tts":
            audio, sample_rate = generate_pocket_tts(processed_text, voice)
        else:
            audio, sample_rate = generate_kokoro(processed_text, voice)

        audio_path = save_audio(audio, sample_rate, text)
        queue_for_session(audio_path, request.session_id)

        return SpeakResponse(
            status="success",
            engine=engine,
            voice=voice,
            audio_file=str(audio_path),
        )

    except Exception as e:
        return SpeakResponse(
            status="error",
            engine=engine,
            voice=voice,
            error=str(e),
        )


class SummarizeRequest(BaseModel):
    text: str
    engine: str | None = None
    voice: str | None = None
    session_id: str | None = None


class SummarizeResponse(BaseModel):
    status: str
    engine: str
    voice: str
    summary: str | None = None
    audio_file: str | None = None
    original_length: int | None = None
    summary_length: int | None = None
    error: str | None = None


@app.post("/summarize", response_model=SummarizeResponse)
async def summarize_and_speak(request: SummarizeRequest):
    """Summarize text and queue for playback.

    Uses configured LLM backend to generate a short, speakable summary.
    Falls back to heuristic extraction if no LLM is configured.
    """
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    # Generate summary
    summary = summarize_for_speech(text)

    engine = request.engine or DEFAULT_ENGINE
    voice = request.voice or get_default_voice(engine)

    if engine not in ("pocket-tts", "kokoro"):
        raise HTTPException(status_code=400, detail=f"Unknown engine: {engine}")

    if engine not in _available_engines:
        engine = next(iter(_available_engines))
        voice = get_default_voice(engine)

    if not validate_voice(engine, voice):
        available = list(get_voices(engine).get(engine, {}).keys())
        raise HTTPException(
            status_code=400,
            detail=f"Unknown voice '{voice}' for engine '{engine}'. Available: {available}",
        )

    try:
        # Preprocess summary for TTS
        processed_summary = preprocess_for_tts(summary)

        if engine == "pocket-tts":
            audio, sample_rate = generate_pocket_tts(processed_summary, voice)
        else:
            audio, sample_rate = generate_kokoro(processed_summary, voice)

        audio_path = save_audio(audio, sample_rate, summary)
        queue_for_session(audio_path, request.session_id)

        return SummarizeResponse(
            status="success",
            engine=engine,
            voice=voice,
            summary=summary,
            audio_file=str(audio_path),
            original_length=len(text),
            summary_length=len(summary),
        )

    except Exception as e:
        return SummarizeResponse(
            status="error",
            engine=engine,
            voice=voice,
            summary=summary,
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


@app.get("/voices")
async def voices(engine: str | None = None):
    """List available voices."""
    return get_voices(engine)


def main():
    """Run the server."""
    parser = argparse.ArgumentParser(
        prog="speeker-server",
        description="Speeker TTS HTTP server with warm models",
    )
    parser.add_argument("-p", "--port", type=int, default=7849, help="Port to listen on")
    parser.add_argument("-H", "--host", default="127.0.0.1", help="Host to bind to")
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
