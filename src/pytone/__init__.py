"""pytone - Pure Python music tone synthesis."""

from .synthesis import (
    note_to_frequency,
    generate_tone,
    generate_tones,
    samples_to_wav,
    NOTE_SEMITONES,
)
from .parser import extract_tone_tokens

__version__ = "0.1.0"
__all__ = [
    "note_to_frequency",
    "generate_tone",
    "generate_tones",
    "samples_to_wav",
    "extract_tone_tokens",
    "NOTE_SEMITONES",
]
