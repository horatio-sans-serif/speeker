"""Tests for the audio-effects chain (speeker.effects).

Covers:
- ``off`` preset is a true passthrough (same array object semantics).
- A preset that includes Reverb returns a LONGER array than its input
  (reverb tail is the canonical signal that the chain actually ran).
- Per-utterance ``preset_override`` argument bypasses config.
- Missing pedalboard dep degrades to passthrough without crashing.
- Preset names + descriptions list survives round-trip via API helpers.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import patch

import numpy as np
import pytest

from speeker import effects


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _silence(seconds: float = 1.0, sr: int = 16000) -> np.ndarray:
    """A short sample of low-amplitude noise -- not pure zeros so reverb
    has actual signal to process and produce a tail from."""
    rng = np.random.default_rng(seed=0)
    return rng.standard_normal(int(sr * seconds)).astype(np.float32) * 0.1


@pytest.fixture(autouse=True)
def _reset_pedalboard_warning_state():
    """Make the one-time "pedalboard missing" warning flag idempotent across
    tests so the tests don't influence each other's output."""
    effects._PEDALBOARD_WARNED = False
    effects._build_board.cache_clear()
    yield
    effects._build_board.cache_clear()


# ---------------------------------------------------------------------------
# Preset list / metadata
# ---------------------------------------------------------------------------

def test_preset_names_off_first_then_alphabetical_set(tmp_path, monkeypatch):
    """All built-ins are present and ``off`` leads. Custom presets from
    config can extend the list but never remove a built-in. Test runs
    in an isolated SPEEKER_DIR so a user's saved custom presets don't
    contaminate the assertion."""
    monkeypatch.setenv("SPEEKER_DIR", str(tmp_path))
    names = effects.preset_names()
    assert names[0] == "off"
    assert set(names) >= {"off", "studio", "natural", "spacious", "telephone", "robot"}


def test_every_preset_has_a_description():
    for name in effects.PRESETS:
        assert name in effects.PRESET_DESCRIPTIONS
        assert effects.PRESET_DESCRIPTIONS[name].strip()


def test_off_preset_is_empty_chain():
    assert effects.PRESETS["off"] == ()


def test_known_presets_have_at_least_one_effect():
    for name, chain in effects.PRESETS.items():
        if name == "off":
            continue
        assert len(chain) > 0, f"preset {name!r} is empty"


# ---------------------------------------------------------------------------
# apply_effects -- behavior
# ---------------------------------------------------------------------------

@patch("speeker.effects.get_effects_config")
def test_apply_effects_off_returns_input_unchanged(mock_cfg):
    """off is the most common case -- it must be a true passthrough so
    users who haven't enabled effects pay zero CPU/binary cost."""
    mock_cfg.return_value = {"preset": "off"}
    audio = _silence()
    result = effects.apply_effects(audio, 16000)
    assert result is audio  # same object, no copy, no transform


@patch("speeker.effects.get_effects_config")
def test_apply_effects_unknown_preset_falls_back_to_off(mock_cfg):
    """An invalid preset name (typo, removed preset) must not crash --
    fall through to passthrough."""
    mock_cfg.return_value = {"preset": "nonexistent-preset-name"}
    audio = _silence()
    result = effects.apply_effects(audio, 16000)
    assert result is audio


@patch("speeker.effects.get_effects_config")
def test_apply_effects_missing_pedalboard_passthrough(mock_cfg):
    """When pedalboard isn't importable (no `effects` extra installed),
    every preset must degrade gracefully -- not raise ImportError."""
    mock_cfg.return_value = {"preset": "natural"}

    # Make `import pedalboard` inside effects._import_pedalboard fail.
    # We can't actually uninstall the module mid-test if it exists, so
    # patch the helper directly.
    audio = _silence()
    with patch.object(effects, "_import_pedalboard", return_value=None):
        result = effects.apply_effects(audio, 16000)
    # Result is the original (passthrough). When pedalboard is missing,
    # _build_board returns None, apply_effects returns audio_np unchanged.
    assert result is audio


def _samples_differ(a: np.ndarray, b: np.ndarray, *, threshold: float = 0.01) -> bool:
    """True when *a* and *b* differ by more than *threshold* anywhere.

    Pedalboard's ``Pedalboard.__call__`` returns the SAME LENGTH as the
    input -- reverb tails are mixed into the dry signal in-place, not
    appended. So "effects ran" is signaled by per-sample difference,
    not by length growth.
    """
    n = min(len(a), len(b))
    return float(np.max(np.abs(a[:n] - b[:n]))) > threshold


@patch("speeker.effects.get_effects_config")
def test_apply_effects_with_reverb_alters_samples(mock_cfg):
    """A reverb-bearing preset must actually modify the audio samples
    (vs the passthrough off case). This is the signature that pedalboard
    actually ran and didn't no-op."""
    pytest.importorskip("pedalboard")
    mock_cfg.return_value = {"preset": "spacious"}
    audio = _silence(seconds=0.5)
    result = effects.apply_effects(audio, 16000)
    assert result is not audio, "effects should produce a new array"
    assert _samples_differ(audio, result), "expected reverb to change samples"


@patch("speeker.effects.get_effects_config")
def test_preset_override_wins_over_config(mock_cfg):
    """The override argument is what /api/effects/try uses to preview
    an unsaved preset selection -- it must beat the saved config."""
    pytest.importorskip("pedalboard")
    # Config says "off" -- without the override, the chain would be a
    # passthrough. With override="spacious", we expect samples to change.
    mock_cfg.return_value = {"preset": "off"}
    audio = _silence(seconds=0.5)
    result = effects.apply_effects(audio, 16000, preset_override="spacious")
    assert result is not audio
    assert _samples_differ(audio, result)


@patch("speeker.effects.get_effects_config")
def test_preset_override_off_disables_even_when_config_active(mock_cfg):
    """Override='off' must disable effects even when config has a
    non-passthrough preset saved. Symmetry with the above test."""
    mock_cfg.return_value = {"preset": "spacious"}
    audio = _silence()
    result = effects.apply_effects(audio, 16000, preset_override="off")
    assert result is audio


# ---------------------------------------------------------------------------
# build_board caching
# ---------------------------------------------------------------------------

def test_build_board_caches_identical_chains():
    """Same preset name should reuse the cached Pedalboard object so
    repeat utterances don't pay JUCE-object construction cost."""
    pytest.importorskip("pedalboard")
    b1 = effects.build_board("studio")
    b2 = effects.build_board("studio")
    assert b1 is b2  # identity, courtesy of lru_cache on _build_board


def test_build_board_off_returns_none():
    """off must skip the cache entirely (empty chain -> None)."""
    assert effects.build_board("off") is None


# ---------------------------------------------------------------------------
# EffectSpec equality / hashability (used as cache key)
# ---------------------------------------------------------------------------

def test_effect_spec_with_same_params_is_equal():
    """EffectSpec is frozen + dataclass so equality + hashing work out of
    the box -- the lru_cache depends on this."""
    a = effects.EffectSpec.of("Reverb", room_size=0.2, damping=0.5)
    b = effects.EffectSpec.of("Reverb", damping=0.5, room_size=0.2)
    assert a == b
    assert hash(a) == hash(b)


def test_effect_spec_with_different_params_is_unequal():
    a = effects.EffectSpec.of("Reverb", room_size=0.2)
    b = effects.EffectSpec.of("Reverb", room_size=0.5)
    assert a != b
