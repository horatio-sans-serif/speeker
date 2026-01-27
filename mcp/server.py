#!/usr/bin/env python3
"""MCP server for Speeker text-to-speech capabilities.

Provides tools for Claude Code to generate and queue speech via Speeker.
"""

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

SPEEKER_URL = os.environ.get("SPEEKER_URL", "http://127.0.0.1:7849")


def get_default_queue() -> str:
    """Get the default queue name from the current project directory."""
    # Try Claude Code specific env vars first
    for var in ["CLAUDE_PROJECT_DIR", "PROJECT_DIR", "PWD"]:
        path = os.environ.get(var, "")
        if path:
            name = Path(path).name
            if name and name != "mcp":  # Avoid MCP server's own directory
                return name
    return Path.cwd().name or "default"

mcp = FastMCP("speeker-tts")


def call_speeker(endpoint: str, data: dict[str, Any]) -> dict[str, Any]:
    """Make a POST request to the Speeker service."""
    url = f"{SPEEKER_URL}{endpoint}"
    body = json.dumps(data).encode("utf-8")
    headers = {"Content-Type": "application/json"}

    req = urllib.request.Request(url, data=body, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8") if e.fp else str(e)
        return {"status": "error", "error": f"HTTP {e.code}: {error_body}"}
    except urllib.error.URLError as e:
        return {"status": "error", "error": f"Connection failed: {e.reason}. Is speeker-server running?"}
    except TimeoutError:
        return {"status": "error", "error": "Request timed out"}


def get_speeker(endpoint: str) -> dict[str, Any]:
    """Make a GET request to the Speeker service."""
    url = f"{SPEEKER_URL}{endpoint}"
    req = urllib.request.Request(url, method="GET")

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8") if e.fp else str(e)
        return {"status": "error", "error": f"HTTP {e.code}: {error_body}"}
    except urllib.error.URLError as e:
        return {"status": "error", "error": f"Connection failed: {e.reason}. Is speeker-server running?"}
    except TimeoutError:
        return {"status": "error", "error": "Request timed out"}


@mcp.tool()
def speak(
    text: str,
    engine: str | None = None,
    voice: str | None = None,
    queue: str | None = None,
) -> dict[str, Any]:
    """
    Generate speech from text and queue for playback.

    Args:
        text: The text to speak
        engine: TTS engine to use: "pocket-tts" or "kokoro" (default: pocket-tts)
        voice: Voice to use (depends on engine, uses default if not specified)
        queue: Queue name for grouping utterances (default: current project name)

    Returns:
        Dictionary with status, queue_id, and pending_count
    """
    if not text or not text.strip():
        return {"status": "error", "error": "Text cannot be empty"}

    data: dict[str, Any] = {"text": text}
    metadata: dict[str, Any] = {"queue": queue or get_default_queue()}
    if engine:
        metadata["engine"] = engine
    if voice:
        metadata["voice"] = voice
    data["metadata"] = metadata

    result = call_speeker("/speak", data)

    if result.get("status") == "success":
        return {
            "status": "success",
            "message": result.get("message", "Queued for playback"),
            "queue_id": result.get("queue_id"),
            "pending_count": result.get("pending_count"),
        }

    return result


@mcp.tool()
def summarize_and_speak(
    text: str,
    engine: str | None = None,
    voice: str | None = None,
    queue: str | None = None,
) -> dict[str, Any]:
    """
    Summarize text to one concise sentence and queue for playback.

    Uses LLM to generate a short, speakable summary (max 15 words).
    Ideal for announcing completion of tasks or summarizing responses.

    Args:
        text: The text to summarize and speak (e.g., Claude's full response)
        engine: TTS engine to use: "pocket-tts" or "kokoro" (default: pocket-tts)
        voice: Voice to use (depends on engine, uses default if not specified)
        queue: Queue name for grouping utterances (default: current project name)

    Returns:
        Dictionary with status, summary, queue_id, and pending_count
    """
    if not text or not text.strip():
        return {"status": "error", "error": "Text cannot be empty"}

    data: dict[str, Any] = {"text": text}
    metadata: dict[str, Any] = {"queue": queue or get_default_queue()}
    if engine:
        metadata["engine"] = engine
    if voice:
        metadata["voice"] = voice
    data["metadata"] = metadata

    result = call_speeker("/summarize", data)

    if result.get("status") == "success":
        return {
            "status": "success",
            "summary": result.get("summary"),
            "original_length": result.get("original_length"),
            "summary_length": result.get("summary_length"),
            "queue_id": result.get("queue_id"),
            "pending_count": result.get("pending_count"),
            "message": "Summary generated and queued for playback",
        }

    return result


@mcp.tool()
def list_voices(engine: str | None = None) -> dict[str, Any]:
    """
    List available TTS voices.

    Args:
        engine: Filter by engine ("pocket-tts" or "kokoro"), or None for all

    Returns:
        Dictionary with available voices grouped by engine
    """
    # Query the actual voices from Speeker's API
    result = get_speeker("/voices")
    if result.get("status") == "error":
        # Fallback to static list if API not available
        return _fallback_voices(engine)

    if engine:
        engines = result.get("engines", {})
        if engine not in engines:
            return {"status": "error", "error": f"Unknown engine: {engine}"}
        return {
            "status": "success",
            "default_engine": result.get("default_engine", "pocket-tts"),
            "engines": {engine: engines[engine]},
        }

    return result


def _fallback_voices(engine: str | None = None) -> dict[str, Any]:
    """Fallback voice list when API is unavailable."""
    voices = {
        "pocket-tts": {
            "default": "azelma",
            "voices": {
                "azelma": {"description": "Female, natural and conversational", "is_default": True},
                "alba": {"description": "Female, soft and warm", "is_default": False},
                "marius": {"description": "Male, clear and articulate", "is_default": False},
                "javert": {"description": "Male, deep and authoritative", "is_default": False},
                "jean": {"description": "Male, gentle and expressive", "is_default": False},
                "fantine": {"description": "Female, emotional and melodic", "is_default": False},
                "cosette": {"description": "Female, young and bright", "is_default": False},
                "eponine": {"description": "Female, spirited and dynamic", "is_default": False},
            },
        },
        "kokoro": {
            "default": "am_liam",
            "voices": {
                "am_liam": {"description": "American male, clear and professional", "is_default": True},
                "af_bella": {"description": "American female, warm and friendly", "is_default": False},
                "af_nicole": {"description": "American female, bright and energetic", "is_default": False},
                "af_sarah": {"description": "American female, calm and soothing", "is_default": False},
                "am_adam": {"description": "American male, deep and resonant", "is_default": False},
                "am_michael": {"description": "American male, natural and casual", "is_default": False},
                "bf_emma": {"description": "British female, refined and elegant", "is_default": False},
                "bf_isabella": {"description": "British female, warm and expressive", "is_default": False},
                "bm_george": {"description": "British male, classic and articulate", "is_default": False},
                "bm_lewis": {"description": "British male, modern and conversational", "is_default": False},
            },
        },
    }

    if engine:
        if engine not in voices:
            return {"status": "error", "error": f"Unknown engine: {engine}"}
        return {
            "status": "success",
            "default_engine": "pocket-tts",
            "engines": {engine: voices[engine]},
        }

    return {
        "status": "success",
        "default_engine": "pocket-tts",
        "engines": voices,
    }


if __name__ == "__main__":
    mcp.run()
