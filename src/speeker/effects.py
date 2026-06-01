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
    """Names of all known presets (built-in + custom), ``off`` first.

    The built-in names ``off`` and the canonical six (studio, natural,
    spacious, telephone, robot) cannot be deleted via the API but can be
    *shadowed* by a custom preset of the same name (the custom wins).
    """
    builtins = ["off"] + [k for k in PRESETS if k != "off"]
    customs = list(_load_custom_presets().keys())
    # Keep insertion order: builtins first, then customs not already in
    # builtins (a shadowing custom appears in the builtin slot).
    seen = set(builtins)
    return builtins + [c for c in customs if c not in seen]


def _load_custom_presets() -> dict[str, tuple[EffectSpec, ...]]:
    """Read ``effects.custom_presets`` and convert to EffectSpec tuples.

    Malformed entries are dropped silently (the UI's validation gates
    save; a corrupted config file shouldn't crash playback). Returns an
    empty dict when none configured or pedalboard unavailable.
    """
    raw = get_effects_config().get("custom_presets") or {}
    if not isinstance(raw, dict):
        return {}
    result: dict[str, tuple[EffectSpec, ...]] = {}
    for name, effects in raw.items():
        if not isinstance(name, str) or not name.strip():
            continue
        if not isinstance(effects, list):
            continue
        specs: list[EffectSpec] = []
        for entry in effects:
            if not isinstance(entry, dict):
                continue
            cls_name = entry.get("name")
            params = entry.get("params") or {}
            if not isinstance(cls_name, str) or not isinstance(params, dict):
                continue
            specs.append(EffectSpec.of(cls_name, **params))
        result[name.strip()] = tuple(specs)
    return result


def all_presets() -> dict[str, tuple[EffectSpec, ...]]:
    """Built-in PRESETS overlaid with custom presets from config."""
    merged: dict[str, tuple[EffectSpec, ...]] = dict(PRESETS)
    merged.update(_load_custom_presets())
    return merged


def _resolve_preset_name(name: str | None) -> str:
    """Map an arbitrary input to a known preset name. Unknown -> ``off``."""
    if isinstance(name, str) and name in all_presets():
        return name
    return "off"


def is_builtin_preset(name: str) -> bool:
    """True when ``name`` is one of the read-only built-in presets."""
    return name in PRESETS


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
    """Public entry: name -> pedalboard.Pedalboard | None.

    Resolves through both built-in and custom presets. Unknown names
    fall back to ``off`` (passthrough).
    """
    return _build_board(all_presets()[_resolve_preset_name(name)])


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


# Editable pedalboard plugins surfaced to the preset editor. Keys are
# the pedalboard class names; values describe each constructor parameter
# the UI should render. The ``range`` is advisory -- pedalboard validates
# on construction; we just suggest sane bounds for sliders.
#
# Curated rather than introspected because pedalboard's classes have many
# parameters and not all of them are user-meaningful (some are read-only
# state). A curated list keeps the editor's forms small and the UX
# obvious. Extend as needed.
PLUGIN_CATALOG: dict[str, list[dict]] = {
    "HighpassFilter": [
        {"name": "cutoff_frequency_hz", "type": "float", "default": 80.0, "min": 20, "max": 20000, "step": 1},
    ],
    "LowpassFilter": [
        {"name": "cutoff_frequency_hz", "type": "float", "default": 3400.0, "min": 20, "max": 20000, "step": 1},
    ],
    "HighShelfFilter": [
        {"name": "cutoff_frequency_hz", "type": "float", "default": 8000.0, "min": 20, "max": 20000, "step": 1},
        {"name": "gain_db", "type": "float", "default": -1.5, "min": -24, "max": 24, "step": 0.1},
        {"name": "q", "type": "float", "default": 0.707, "min": 0.1, "max": 10.0, "step": 0.01},
    ],
    "LowShelfFilter": [
        {"name": "cutoff_frequency_hz", "type": "float", "default": 120.0, "min": 20, "max": 20000, "step": 1},
        {"name": "gain_db", "type": "float", "default": 0.0, "min": -24, "max": 24, "step": 0.1},
        {"name": "q", "type": "float", "default": 0.707, "min": 0.1, "max": 10.0, "step": 0.01},
    ],
    "PeakFilter": [
        {"name": "cutoff_frequency_hz", "type": "float", "default": 1000.0, "min": 20, "max": 20000, "step": 1},
        {"name": "gain_db", "type": "float", "default": 0.0, "min": -24, "max": 24, "step": 0.1},
        {"name": "q", "type": "float", "default": 1.0, "min": 0.1, "max": 10.0, "step": 0.01},
    ],
    "Compressor": [
        {"name": "threshold_db", "type": "float", "default": -18.0, "min": -60, "max": 0, "step": 0.5},
        {"name": "ratio", "type": "float", "default": 2.5, "min": 1.0, "max": 20.0, "step": 0.1},
        {"name": "attack_ms", "type": "float", "default": 5.0, "min": 0.1, "max": 200.0, "step": 0.1},
        {"name": "release_ms", "type": "float", "default": 80.0, "min": 5.0, "max": 2000.0, "step": 1},
    ],
    "Limiter": [
        {"name": "threshold_db", "type": "float", "default": -1.0, "min": -24, "max": 0, "step": 0.1},
        {"name": "release_ms", "type": "float", "default": 100.0, "min": 5.0, "max": 2000.0, "step": 1},
    ],
    "Gain": [
        {"name": "gain_db", "type": "float", "default": 0.0, "min": -24, "max": 24, "step": 0.1},
    ],
    "Distortion": [
        {"name": "drive_db", "type": "float", "default": 8.0, "min": 0.0, "max": 40.0, "step": 0.5},
    ],
    "Reverb": [
        {"name": "room_size", "type": "float", "default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01},
        {"name": "damping", "type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
        {"name": "wet_level", "type": "float", "default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01},
        {"name": "dry_level", "type": "float", "default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01},
        {"name": "width", "type": "float", "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
    ],
    "Chorus": [
        {"name": "rate_hz", "type": "float", "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01},
        {"name": "depth", "type": "float", "default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01},
        {"name": "centre_delay_ms", "type": "float", "default": 7.0, "min": 0.0, "max": 50.0, "step": 0.1},
        {"name": "feedback", "type": "float", "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01},
        {"name": "mix", "type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
    ],
    "Phaser": [
        {"name": "rate_hz", "type": "float", "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01},
        {"name": "depth", "type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
        {"name": "centre_frequency_hz", "type": "float", "default": 1300.0, "min": 20.0, "max": 20000.0, "step": 1},
        {"name": "feedback", "type": "float", "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01},
        {"name": "mix", "type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
    ],
    "Delay": [
        {"name": "delay_seconds", "type": "float", "default": 0.5, "min": 0.0, "max": 4.0, "step": 0.01},
        {"name": "feedback", "type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
        {"name": "mix", "type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
    ],
    "Bitcrush": [
        {"name": "bit_depth", "type": "float", "default": 8.0, "min": 1.0, "max": 16.0, "step": 0.5},
    ],
    "PitchShift": [
        {"name": "semitones", "type": "float", "default": 0.0, "min": -24.0, "max": 24.0, "step": 0.5},
    ],
}


def plugin_catalog() -> dict[str, list[dict]]:
    """Public copy of PLUGIN_CATALOG -- the editor's source of truth for
    which effects can be added and what parameters each accepts."""
    # Return deep copy so callers can't mutate the module's data.
    return {k: [dict(p) for p in v] for k, v in PLUGIN_CATALOG.items()}


def apply_effects_to_wav(
    wav_path,
    *,
    preset_override: str | None = None,
) -> bool:
    """Read a WAV, run it through the effects chain, write it back.

    Used by the tones synthesis path so a non-``off`` preset shapes the
    intro / outro / $Note / cue chords too. Cache integrity is the
    caller's job -- include the preset name in the cache key so a flip
    from ``natural`` to ``robot`` regenerates files.

    Returns True when the file was processed, False on any failure
    (file missing, pedalboard not installed, preset off). The original
    WAV is left untouched on failure.
    """
    from pathlib import Path
    p = Path(wav_path)
    if not p.exists():
        return False
    preset_name = _resolve_preset_name(
        preset_override
        if preset_override is not None
        else get_effects_config().get("preset"),
    )
    if preset_name == "off":
        return False
    board = _build_board(all_presets()[preset_name])
    if board is None:
        return False
    try:
        from scipy.io import wavfile
        sample_rate, data = wavfile.read(str(p))
        # Convert int16 -> float32 in [-1, 1] for pedalboard, then back.
        if data.dtype == np.int16:
            float_data = data.astype(np.float32) / 32767.0
        elif data.dtype == np.float32:
            float_data = data
        else:
            float_data = np.asarray(data, dtype=np.float32)
        processed = board(float_data, int(sample_rate))
        # Clip + quantize back to int16 to match the rest of the WAV cache.
        processed = np.clip(processed, -1.0, 1.0)
        wavfile.write(str(p), int(sample_rate), (processed * 32767).astype(np.int16))
        return True
    except Exception as e:
        log.warning("effects: post-processing %s failed (%s) -- left untouched", p, e)
        return False


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
    board = _build_board(all_presets()[preset_name])
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
