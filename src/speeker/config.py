"""Configuration management for Speeker."""

import json

from .paths import config_dir, config_file, ensure_dir

DEFAULT_CONFIG = {
    "semantic_search": {
        "enabled": False,
        "model": "all-MiniLM-L6-v2",
        "cache_dir": None,  # None = default (~/.cache), or set to "/tmp/speeker-models"
    },
    "llm": {
        "backend": None,  # "ollama", "anthropic", or "openai"
        "endpoint": None,  # API endpoint (default per backend if None)
        "api_key": None,  # Required for anthropic/openai
        "model": None,  # Model name (default per backend if None)
    },
    "player": {
        "model_idle_timeout_minutes": 0,  # 0 = never unload
    },
    "polly": {
        "region": None,    # None = boto3 default (profile/env region)
        "profile": None,   # AWS profile name; None = default credential chain
        "engine": "neural",  # default Polly engine variant
        "voice": "Joanna",   # default Polly VoiceId
    },
    "ssml": {
        "emulate_for_local": False,  # if True, approximate SSML for local TTS engines
        "acronyms_file": None,       # path to a file of extra spell-out acronyms
    },
    "effects": {
        # Active audio-effects preset applied to TTS speech (not tones).
        # See effects.PRESETS for valid names. "off" disables the chain
        # entirely (no pedalboard import / instantiation cost). Read on
        # every utterance, so changes take effect on the next message
        # without a daemon restart.
        "preset": "off",
    },
    "tones": {
        # Intro/outro chord notation: a sequence of note names, each
        # ``[A-G][b#]?[0-8]``. They are played in order via the same path
        # the $Note inline tone tokens use, so the notation is consistent
        # across the codebase.
        "intro": ["E4", "G4", "C5"],     # rising major triad
        "outro": ["C5", "G4", "E4"],     # falling major triad
        # Per-note duration in seconds when synthesizing intro/outro.
        "duration_seconds": 0.12,
    },
    "pronunciation": {
        # User-supplied {word: respelling} map applied during text
        # preprocessing. Whole-word (regex \b), case-insensitive. Respelling
        # is the *literal text* the active TTS voice will see -- tune it to
        # your voice ("kom pass" for some neural voices, "kom-pass" for
        # others). The override runs LAST in preprocess_for_tts so it wins
        # against the built-in TERM_PRONUNCIATIONS (uv, jq, todo, ...).
        # Polly users: SSML <phoneme> tags via the /speak ssml=true path
        # give deterministic IPA control; this dict is for the plain-text
        # path that every engine shares.
        "overrides": {},
    },
    "auto_label": {
        # Announce the queue's spoken title before a single bare message after
        # a quiet period or when the queue context changes. Multi-message
        # batches already announce a "For queue X, there are N messages"
        # header, so this only fires for the single-message-only-session case.
        "enabled": True,
        # Silence (seconds) before the *next* single message is auto-labeled.
        "quiet_threshold_seconds": 120,
        # Tone token rendered before the spoken title; matches the server's
        # format_with_title() prefix so an auto-label sounds like a
        # caller-supplied title.
        "tone": "$Eb4",
    },
    "interpretations": {
        # Pause (seconds) after a cue finishes, before the utterance speaks.
        "pause_after_seconds": 0.3,
        # Map an interpretation name to an indication. An indication is either
        #   {"type": "notes", "notes": [{"pitch": "Eb3", "seconds": 0.15}, ...]}
        # or
        #   {"type": "sound_file", "path": "/abs/path/to/cue.wav"}
        # SUCCESS/ERROR are also built in (see interpretations.py); entries here
        # override the built-ins of the same name.
        "map": {
            "SUCCESS": {
                "type": "notes",
                "notes": [
                    {"pitch": "Eb3", "seconds": 0.15},
                    {"pitch": "G#3", "seconds": 0.9},
                ],
            },
            "ERROR": {
                "type": "notes",
                "notes": [
                    {"pitch": "Eb4", "seconds": 0.3},
                    {"pitch": "D4", "seconds": 0.2},
                    {"pitch": "Bb2", "seconds": 0.2},
                    {"pitch": "Bb2", "seconds": 0.2},
                ],
            },
        },
    },
}


def get_config() -> dict:
    """Load configuration, creating default if needed."""
    ensure_dir(config_dir())
    cfg_file = config_file()

    if cfg_file.exists():
        try:
            with open(cfg_file) as f:
                config = json.load(f)
            # Merge with defaults for any missing keys
            merged = DEFAULT_CONFIG.copy()
            for key, value in config.items():
                if isinstance(value, dict) and key in merged:
                    merged[key] = {**merged[key], **value}
                else:
                    merged[key] = value
            return merged
        except (json.JSONDecodeError, IOError):
            return DEFAULT_CONFIG.copy()
    else:
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG.copy()


def save_config(config: dict) -> None:
    """Save configuration to file."""
    ensure_dir(config_dir())
    with open(config_file(), "w") as f:
        json.dump(config, f, indent=2)


def is_semantic_search_enabled() -> bool:
    """Check if semantic search is enabled."""
    config = get_config()
    return config.get("semantic_search", {}).get("enabled", False)


def get_embedding_model() -> str:
    """Get the configured embedding model name."""
    config = get_config()
    return config.get("semantic_search", {}).get("model", "all-MiniLM-L6-v2")


def get_embedding_cache_dir() -> str | None:
    """Get the configured cache directory for embedding models."""
    config = get_config()
    return config.get("semantic_search", {}).get("cache_dir")


def get_player_config() -> dict:
    """Get player configuration."""
    config = get_config()
    return config.get("player", {})


def get_llm_config() -> dict:
    """Get LLM configuration (config file overrides env vars)."""
    import os

    config = get_config()
    llm_config = config.get("llm", {})

    # Environment variables override config file
    backend = os.environ.get("SPEEKER_LLM_BACKEND") or llm_config.get("backend")
    endpoint = os.environ.get("SPEEKER_LLM_ENDPOINT") or llm_config.get("endpoint")
    api_key = os.environ.get("SPEEKER_LLM_API_KEY") or llm_config.get("api_key")
    model = os.environ.get("SPEEKER_LLM_MODEL") or llm_config.get("model")

    return {
        "backend": backend.lower() if backend else None,
        "endpoint": endpoint,
        "api_key": api_key,
        "model": model,
    }


def get_polly_config() -> dict:
    """Get Amazon Polly configuration."""
    config = get_config()
    return config.get("polly", {})


def get_ssml_config() -> dict:
    """Get SSML configuration."""
    config = get_config()
    return config.get("ssml", {})


def get_interpretations_config() -> dict:
    """Get interpretation cue configuration."""
    config = get_config()
    return config.get("interpretations", {})


def get_auto_label_config() -> dict:
    """Get auto-label configuration (queue title announcement after silence)."""
    config = get_config()
    return config.get("auto_label", {})


def get_tones_config() -> dict:
    """Get the intro/outro tone configuration."""
    config = get_config()
    return config.get("tones", {})


def get_effects_config() -> dict:
    """Get the audio-effects configuration (preset name + future params)."""
    config = get_config()
    return config.get("effects", {})


def get_pronunciation_overrides() -> dict[str, str | dict[str, str]]:
    """Get user-supplied {word: respelling} pronunciation overrides.

    Whole-word, case-insensitive substitutions applied during text
    preprocessing. Two value shapes are accepted:

    - ``"kom-piss"`` -- single string, applied for every engine.
    - ``{"polly": "...", "pocket-tts": "...", "default": "..."}`` --
      per-engine variants. The preprocessor picks the entry that matches
      the active engine, falling back to ``"default"`` if present.

    Returns ``{}`` when not configured. Malformed entries are dropped
    rather than raising so a half-typed UI row doesn't break TTS.
    """
    config = get_config()
    overrides = config.get("pronunciation", {}).get("overrides", {})
    cleaned: dict[str, str | dict[str, str]] = {}
    for k, v in overrides.items():
        if not isinstance(k, str) or not k.strip():
            continue
        if isinstance(v, str):
            cleaned[k] = v
        elif isinstance(v, dict):
            inner: dict[str, str] = {
                str(ek): str(ev)
                for ek, ev in v.items()
                if isinstance(ek, str) and isinstance(ev, str) and ev.strip()
            }
            if inner:
                cleaned[k] = inner
        # else: drop silently
    return cleaned
