"""Audio effects chain applied to TTS speech before the WAV write.

Hook point: ``player.generate_tts`` calls ``apply_effects(audio_np, sr)``
after speed-resampling and before clipping/quantization. Tones (intro,
outro, ``$Note`` prefixes, interpretation cues) bypass this module
entirely -- they're synthesized through ``tones.Mixer``, not
``generate_tts`` -- so the audio language of cues stays clean.

Library: pedalboard (Spotify), C++/JUCE-backed, numpy-first. Imported
lazily so speeker still works when the optional ``effects`` extra is
not installed; in that case ``apply_effects`` logs once and returns the
input unchanged.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from .config import get_effects_config

log = logging.getLogger(__name__)

_PEDALBOARD_WARNED = False


@dataclass(frozen=True)
class EffectSpec:
    """One effect in a chain. ``name`` is a pedalboard class name; ``params``
    is the kwargs passed to its constructor.

    Frozen so a tuple of specs is hashable -- used as an ``lru_cache`` key
    in ``build_board`` so identical chains share one compiled Pedalboard.
    """
    name: str
    params: tuple[tuple[str, Any], ...] = ()

    @classmethod
    def of(cls, name: str, **kw: Any) -> "EffectSpec":
        return cls(name=name, params=tuple(sorted(kw.items())))


# Preset chains. Tuned for the user's choices (see plan file).
#   off       -- passthrough, no Pedalboard cost.
#   studio    -- broadcast tilt: HPF + comp + slight HF cut + brick-wall limiter.
#   natural   -- studio chain + small room reverb (~150 ms tail).
#   spacious  -- studio chain + larger room reverb (~450 ms tail).
#   telephone -- band-pass 300-3400 + drive + limit.
#   robot     -- bitcrush + chorus + phaser + limit.
PRESETS: dict[str, tuple[EffectSpec, ...]] = {
    "off": (),
    "studio": (
        EffectSpec.of("HighpassFilter", cutoff_frequency_hz=80),
        EffectSpec.of("Compressor",
                      threshold_db=-18.0, ratio=2.5,
                      attack_ms=5.0, release_ms=80.0),
        EffectSpec.of("HighShelfFilter",
                      cutoff_frequency_hz=8000, gain_db=-1.5, q=0.707),
        EffectSpec.of("Limiter", threshold_db=-1.0, release_ms=100.0),
    ),
    "natural": (
        EffectSpec.of("HighpassFilter", cutoff_frequency_hz=80),
        EffectSpec.of("Compressor",
                      threshold_db=-18.0, ratio=2.5,
                      attack_ms=5.0, release_ms=80.0),
        EffectSpec.of("HighShelfFilter",
                      cutoff_frequency_hz=8000, gain_db=-1.5, q=0.707),
        EffectSpec.of("Reverb",
                      room_size=0.15, damping=0.7,
                      wet_level=0.12, dry_level=0.88, width=1.0),
        EffectSpec.of("Limiter", threshold_db=-1.0, release_ms=100.0),
    ),
    "spacious": (
        EffectSpec.of("HighpassFilter", cutoff_frequency_hz=80),
        EffectSpec.of("Compressor",
                      threshold_db=-18.0, ratio=2.5,
                      attack_ms=5.0, release_ms=80.0),
        EffectSpec.of("HighShelfFilter",
                      cutoff_frequency_hz=8000, gain_db=-1.5, q=0.707),
        EffectSpec.of("Reverb",
                      room_size=0.45, damping=0.5,
                      wet_level=0.28, dry_level=0.72, width=1.0),
        EffectSpec.of("Limiter", threshold_db=-1.0, release_ms=100.0),
    ),
    "telephone": (
        EffectSpec.of("HighpassFilter", cutoff_frequency_hz=300),
        EffectSpec.of("LowpassFilter", cutoff_frequency_hz=3400),
        EffectSpec.of("Gain", gain_db=2.0),
        EffectSpec.of("Distortion", drive_db=8.0),
        EffectSpec.of("Limiter", threshold_db=-1.0, release_ms=100.0),
    ),
    "robot": (
        EffectSpec.of("Bitcrush", bit_depth=8),
        EffectSpec.of("Chorus",
                      rate_hz=0.5, depth=0.4, mix=0.5,
                      centre_delay_ms=7.0, feedback=0.0),
        EffectSpec.of("Phaser",
                      rate_hz=0.5, depth=0.4, mix=0.5,
                      centre_frequency_hz=1300.0, feedback=0.0),
        EffectSpec.of("Limiter", threshold_db=-1.0, release_ms=100.0),
    ),
}

# Short human-readable description per preset, surfaced through the API
# so the UI can show it under the dropdown without duplicating copy.
PRESET_DESCRIPTIONS: dict[str, str] = {
    "off":       "Passthrough -- no effects, original engine output.",
    "studio":    "Broadcast-style leveling: compressor + soft HF cut + limiter.",
    "natural":   "Studio chain plus a small room reverb (~150 ms tail).",
    "spacious":  "Studio chain plus a larger room reverb (~450 ms tail).",
    "telephone": "Narrow-band 300-3400 Hz with light drive -- old-phone feel.",
    "robot":     "Bitcrush + chorus + phaser. Lo-fi novelty preset.",
}


def preset_names() -> list[str]:
    """Names of all built-in presets, ``off`` first."""
    return ["off"] + [k for k in PRESETS if k != "off"]


def _resolve_preset_name(name: str | None) -> str:
    """Map an arbitrary input to a known preset name. Unknown -> ``off``."""
    if isinstance(name, str) and name in PRESETS:
        return name
    return "off"


@lru_cache(maxsize=8)
def _build_board(specs: tuple[EffectSpec, ...]):
    """Instantiate a pedalboard chain from a hashable tuple of specs.

    Cached so flipping presets back-and-forth (or reusing the same chain
    across many utterances) doesn't pay the JUCE-object construction cost
    per call. Returns ``None`` for an empty chain or when pedalboard isn't
    importable.
    """
    if not specs:
        return None
    pb = _import_pedalboard()
    if pb is None:
        return None
    plugins = []
    for spec in specs:
        cls = getattr(pb, spec.name, None)
        if cls is None:
            log.warning("effects: unknown pedalboard plugin %r -- skipping", spec.name)
            continue
        try:
            plugins.append(cls(**dict(spec.params)))
        except TypeError as e:
            log.warning("effects: %r init failed (%s) -- skipping", spec.name, e)
    if not plugins:
        return None
    return pb.Pedalboard(plugins)


def build_board(name: str):
    """Public entry: name -> pedalboard.Pedalboard | None."""
    return _build_board(PRESETS[_resolve_preset_name(name)])


def _import_pedalboard():
    """Lazy import. Returns the module, or ``None`` (with one-time warning)
    when the optional ``effects`` extra hasn't been installed."""
    global _PEDALBOARD_WARNED
    try:
        import pedalboard
        return pedalboard
    except ImportError:
        if not _PEDALBOARD_WARNED:
            print(
                "[effects] pedalboard not installed; effects preset will be "
                "ignored. Install with: uv tool install --reinstall -e "
                "--with pedalboard --with boto3 <speeker-src>",
                file=sys.stderr,
            )
            _PEDALBOARD_WARNED = True
        return None


def apply_effects(
    audio_np: np.ndarray,
    sample_rate: int,
    *,
    preset_override: str | None = None,
) -> np.ndarray:
    """Apply the configured effects chain to *audio_np*.

    ``preset_override`` -- if given, this preset is used instead of the
    saved configuration. Lets ``/api/effects/try`` preview an unsaved
    selection on a single utterance without mutating ``config.json``.

    No-op (returns the input unchanged) when:

    - the resolved preset is ``"off"`` or unknown,
    - the chain is empty,
    - the ``pedalboard`` dependency isn't importable.

    Pedalboard mixes effect output (including reverb / delay tails)
    into the *same-length* array as the input -- the tail overlaps the
    dry signal rather than appending a new segment. Callers that need a
    longer tail should pad the input first.
    """
    if preset_override is not None:
        preset_name = _resolve_preset_name(preset_override)
    else:
        preset_name = _resolve_preset_name(get_effects_config().get("preset"))
    if preset_name == "off":
        return audio_np
    board = _build_board(PRESETS[preset_name])
    if board is None:
        return audio_np
    # pedalboard wants float32; our pipeline is already float32 but cast
    # defensively in case the engine produced float64.
    arr = np.asarray(audio_np, dtype=np.float32)
    try:
        return board(arr, int(sample_rate))
    except Exception as e:  # pragma: no cover - defensive; per-utterance
        log.warning("effects: chain %r raised %s -- passing audio through", preset_name, e)
        return audio_np
