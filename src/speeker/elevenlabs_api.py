"""ElevenLabs API access: client, instant voice cloning, and synthesis.

Single home for every ElevenLabs SDK call so the engine (``engines.py``) and the
cloning pipeline (``voice_clone.py``) share one credential/client path and there's
no import cycle between them. The ``elevenlabs`` package is imported lazily inside
functions so importing this module never requires it to be installed.
"""

from __future__ import annotations

from pathlib import Path


def get_client():
    """Build an ElevenLabs client from config + ELEVENLABS_API_KEY.

    Raises a clear error if no API key is configured or the SDK is missing.
    """
    from .config import get_elevenlabs_config

    cfg = get_elevenlabs_config()
    api_key = cfg.get("api_key")
    if not api_key:
        raise RuntimeError(
            "ElevenLabs API key not set. Export ELEVENLABS_API_KEY (or set "
            "elevenlabs.api_key in config)."
        )
    try:
        from elevenlabs.client import ElevenLabs
    except ImportError as e:
        raise RuntimeError(
            "The 'elevenlabs' package is required for the elevenlabs engine. "
            "Install it with: uv pip install 'speeker[elevenlabs]'"
        ) from e
    return ElevenLabs(api_key=api_key)


def create_ivc_voice(name: str, description: str | None, files: list[Path]) -> str:
    """Create an Instant Voice Clone from *files* and return its voice_id."""
    client = get_client()
    with _open_files(files) as handles:
        voice = client.voices.ivc.create(
            name=name,
            description=description,
            files=handles,
        )
    return voice.voice_id


def delete_voice(voice_id: str) -> None:
    """Delete a server-side voice. Best-effort: errors are swallowed."""
    try:
        get_client().voices.delete(voice_id)
    except Exception:  # noqa: BLE001 - cleanup must not block local deletion
        pass


def synthesize(voice_id: str, text: str, model_id: str, output_format: str) -> bytes:
    """Synthesize *text* with *voice_id*, joining the chunk iterator into bytes."""
    client = get_client()
    audio = client.text_to_speech.convert(
        text=text,
        voice_id=voice_id,
        model_id=model_id,
        output_format=output_format,
    )
    if isinstance(audio, (bytes, bytearray)):
        return bytes(audio)
    return b"".join(audio)


class _open_files:
    """Context manager opening *files* for reading, closing all on exit."""

    def __init__(self, files: list[Path]) -> None:
        self._paths = files
        self._handles: list = []

    def __enter__(self) -> list:
        for p in self._paths:
            self._handles.append(open(p, "rb"))
        return self._handles

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            try:
                h.close()
            except Exception:  # noqa: BLE001
                pass
