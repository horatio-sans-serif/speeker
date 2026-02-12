"""Voice configuration for TTS engines."""

from pathlib import Path

POCKET_TTS_VOICES = {
    "alba": "Female, soft and warm",
    "marius": "Male, clear and articulate",
    "javert": "Male, deep and authoritative",
    "jean": "Male, gentle and expressive",
    "fantine": "Female, emotional and melodic",
    "cosette": "Female, young and bright",
    "eponine": "Female, spirited and dynamic",
    "azelma": "Female, natural and conversational",
}

KOKORO_VOICES = {
    "am_liam": "American male, clear and professional",
    "af_bella": "American female, warm and friendly",
    "af_nicole": "American female, bright and energetic",
    "af_sarah": "American female, calm and soothing",
    "am_adam": "American male, deep and resonant",
    "am_michael": "American male, natural and casual",
    "bf_emma": "British female, refined and elegant",
    "bf_isabella": "British female, warm and expressive",
    "bm_george": "British male, classic and articulate",
    "bm_lewis": "British male, modern and conversational",
}

DEFAULT_ENGINE = "pocket-tts"
DEFAULT_POCKET_TTS_VOICE = "azelma"
DEFAULT_KOKORO_VOICE = "am_liam"


def is_custom_voice(name: str) -> bool:
    """Check if a voice name refers to a custom cloned voice."""
    from .voice_clone import get_custom_voice_path

    return get_custom_voice_path(name) is not None


def get_voices(engine: str | None = None) -> dict[str, dict[str, str]]:
    """Get available voices, optionally filtered by engine."""
    result = {}
    if engine is None or engine == "pocket-tts":
        result["pocket-tts"] = POCKET_TTS_VOICES
    if engine is None or engine == "kokoro":
        result["kokoro"] = KOKORO_VOICES

    # Include custom voices when not filtering to a specific engine,
    # or when explicitly requesting custom voices.
    if engine is None or engine == "custom":
        from .voice_clone import get_custom_voices

        custom = get_custom_voices()
        if custom:
            result["custom"] = {
                name: entry.get("description", "Custom cloned voice")
                for name, entry in custom.items()
            }

    return result


def get_default_voice(engine: str) -> str:
    """Get the default voice for an engine."""
    if engine == "kokoro":
        return DEFAULT_KOKORO_VOICE
    return DEFAULT_POCKET_TTS_VOICE


def validate_voice(engine: str, voice: str) -> bool:
    """Check if a voice is valid for the given engine."""
    if engine == "pocket-tts":
        return voice in POCKET_TTS_VOICES or is_custom_voice(voice)
    if engine == "kokoro":
        return voice in KOKORO_VOICES
    return False


def get_pocket_tts_voice_path(voice: str) -> str | Path:
    """Get the voice identifier for pocket-tts.

    Returns a string for built-in voices (e.g. "azelma") or a Path
    for custom cloned voices. pocket-tts's get_state_for_audio_prompt()
    handles both types.
    """
    if voice in POCKET_TTS_VOICES:
        return voice

    # Check custom voices
    from .voice_clone import get_custom_voice_path

    custom_path = get_custom_voice_path(voice)
    if custom_path is not None:
        return custom_path

    return DEFAULT_POCKET_TTS_VOICE
