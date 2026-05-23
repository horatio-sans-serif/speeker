#!/usr/bin/env python3
"""Unit tests for engines.py (registry + payload prep). No real models run."""

import numpy as np
import pytest

from speeker.engines import (
    BaseEngine,
    PocketTTSEngine,
    KokoroEngine,
    get_engine,
    unload_all,
    prepare_payload,
)


class TestRegistry:
    def setup_method(self):
        unload_all()

    def test_pocket_tts_singleton(self):
        a = get_engine("pocket-tts")
        b = get_engine("pocket-tts")
        assert a is b
        assert isinstance(a, PocketTTSEngine)

    def test_kokoro_engine(self):
        assert isinstance(get_engine("kokoro"), KokoroEngine)

    def test_default_when_none(self):
        assert get_engine(None).name == "pocket-tts"

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            get_engine("nope")

    def test_metadata(self):
        eng = get_engine("pocket-tts")
        assert eng.name == "pocket-tts"
        assert eng.supports_ssml is False
        assert eng.default_voice() == "azelma"
        assert "azelma" in eng.list_voices()

    def test_unload_all_resets_singletons(self):
        a = get_engine("pocket-tts")
        unload_all()
        assert get_engine("pocket-tts") is not a


class _FakeSsmlEngine(BaseEngine):
    name = "fake"
    supports_ssml = True


class _FakeLocalEngine(BaseEngine):
    name = "fakelocal"
    supports_ssml = False


class TestPreparePayload:
    def test_plain_text_passthrough(self):
        payload, is_ssml = prepare_payload(
            _FakeLocalEngine(), "hello", is_ssml=False, emulate=False
        )
        assert payload == "hello" and is_ssml is False

    def test_ssml_engine_passthrough(self):
        payload, is_ssml = prepare_payload(
            _FakeSsmlEngine(), "<speak>hi</speak>", is_ssml=True, emulate=False
        )
        assert payload == "<speak>hi</speak>" and is_ssml is True

    def test_local_engine_strips_when_no_emulation(self):
        payload, is_ssml = prepare_payload(
            _FakeLocalEngine(), "<speak>Hello <break/>world</speak>",
            is_ssml=True, emulate=False,
        )
        assert payload == "Hello world" and is_ssml is False

    def test_local_engine_emulates_when_enabled(self):
        payload, is_ssml = prepare_payload(
            _FakeLocalEngine(),
            '<say-as interpret-as="characters">PHI</say-as>',
            is_ssml=True, emulate=True,
        )
        assert payload == "P-H-I" and is_ssml is False
