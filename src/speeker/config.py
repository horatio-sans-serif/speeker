"""Configuration management for Speeker."""

import json
from pathlib import Path

CONFIG_DIR = Path.home() / ".config" / "speeker"
CONFIG_FILE = CONFIG_DIR / "config.json"

DEFAULT_CONFIG = {
    "semantic_search": {
        "enabled": False,
        "model": "all-MiniLM-L6-v2",
        "cache_dir": None,  # None = default (~/.cache), or set to "/tmp/speeker-models"
    },
}


def get_config() -> dict:
    """Load configuration, creating default if needed."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
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
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
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
