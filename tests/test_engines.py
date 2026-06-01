#!/usr/bin/env python3
"""Unit tests for engines.py (registry + payload prep). No real models run."""

import io
import sys
from unittest.mock import MagicMock, patch

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



def _mock_boto3_returning(pcm_bytes: bytes):
    """Build a fake boto3 module whose Polly client returns pcm_bytes."""
    client = MagicMock()
    client.synthesize_speech.return_value = {"AudioStream": io.BytesIO(pcm_bytes)}
    session = MagicMock()
    session.client.return_value = client
    boto3 = MagicMock()
    boto3.Session.return_value = session
    return boto3, client


class TestPollyEngine:
    def setup_method(self):
        unload_all()

    def test_generate_text_mode(self, tmp_path):
        from speeker.engines import PollyEngine
        pcm = (np.array([0, 16384, -16384, 32767], dtype=np.int16)).tobytes()
        boto3, client = _mock_boto3_returning(pcm)
        with patch.dict(sys.modules, {"boto3": boto3}), \
             patch.dict("os.environ", {"SPEEKER_DIR": str(tmp_path)}):
            eng = PollyEngine()
            audio, sr = eng.generate("hello", "Joanna", is_ssml=False)
        assert sr == 16000
        assert audio.dtype == np.float32
        assert audio.max() <= 1.0 and audio.min() >= -1.0
        np.testing.assert_allclose(
            audio, [0.0, 16384 / 32768.0, -16384 / 32768.0, 32767 / 32768.0], rtol=1e-6
        )
        kwargs = client.synthesize_speech.call_args.kwargs
        assert kwargs["TextType"] == "text"
        assert kwargs["OutputFormat"] == "pcm"
        assert kwargs["VoiceId"] == "Joanna"
        assert kwargs["Engine"] == "neural"  # config default

    def test_generate_ssml_mode_wraps_and_sets_texttype(self, tmp_path):
        from speeker.engines import PollyEngine
        boto3, client = _mock_boto3_returning(np.array([0], dtype=np.int16).tobytes())
        with patch.dict(sys.modules, {"boto3": boto3}), \
             patch.dict("os.environ", {"SPEEKER_DIR": str(tmp_path)}):
            eng = PollyEngine()
            eng.generate("hi", "Joanna", is_ssml=True, polly_engine="long-form")
        kwargs = client.synthesize_speech.call_args.kwargs
        assert kwargs["TextType"] == "ssml"
        assert kwargs["Text"] == "<speak>hi</speak>"
        assert kwargs["Engine"] == "long-form"

    def test_supports_ssml_and_noops(self):
        from speeker.engines import PollyEngine
        eng = PollyEngine()
        assert eng.supports_ssml is True
        eng.warm()    # no-op, must not raise
        eng.unload()  # clears client + variant cache, must not raise

    def test_falls_back_when_voice_rejects_engine(self, tmp_path):
        """A voice that doesn't support the configured variant (e.g.
        'Gregory' is long-form only but config says 'generative') must
        transparently fall back to a supported variant instead of
        failing. Polly's own ValidationException drives the fallback."""
        from speeker.engines import PollyEngine, _polly_variant_cache
        from speeker.config import save_config
        _polly_variant_cache.clear()

        pcm = np.array([0, 16384], dtype=np.int16).tobytes()
        boto3, client = _mock_boto3_returning(pcm)

        # generative -> rejected; anything else -> success.
        def _synth(**kwargs):
            if kwargs["Engine"] == "generative":
                raise Exception(
                    "An error occurred (ValidationException) when calling the "
                    "SynthesizeSpeech operation: This voice does not support "
                    "the selected engine: generative"
                )
            return {"AudioStream": io.BytesIO(pcm)}
        client.synthesize_speech.side_effect = _synth

        with patch.dict(sys.modules, {"boto3": boto3}), \
             patch.dict("os.environ", {"SPEEKER_DIR": str(tmp_path)}):
            save_config({"polly": {"engine": "generative"}})
            eng = PollyEngine()
            audio, sr = eng.generate("hello", "Gregory", is_ssml=False)

        assert sr == 16000
        assert audio.dtype == np.float32
        # The final successful call used the first fallback variant, neural.
        used_engines = [c.kwargs["Engine"] for c in client.synthesize_speech.call_args_list]
        assert used_engines[0] == "generative"   # tried requested first
        assert used_engines[-1] == "neural"       # resolved to fallback
        # Cache now maps (voice, requested) -> resolved so the next
        # utterance skips the failed generative round-trip.
        assert _polly_variant_cache[("Gregory", "generative")] == "neural"

    def test_cached_variant_skips_failed_roundtrip(self, tmp_path):
        """Once a (voice, requested) pair resolves, the next call starts
        from the cached working variant -- no repeated failures."""
        from speeker.engines import PollyEngine, _polly_variant_cache
        from speeker.config import save_config
        _polly_variant_cache.clear()
        _polly_variant_cache[("Gregory", "generative")] = "neural"

        pcm = np.array([0], dtype=np.int16).tobytes()
        boto3, client = _mock_boto3_returning(pcm)
        with patch.dict(sys.modules, {"boto3": boto3}), \
             patch.dict("os.environ", {"SPEEKER_DIR": str(tmp_path)}):
            save_config({"polly": {"engine": "generative"}})
            eng = PollyEngine()
            eng.generate("hi", "Gregory")
        # Only one call, and it went straight to neural -- no generative attempt.
        used = [c.kwargs["Engine"] for c in client.synthesize_speech.call_args_list]
        assert used == ["neural"]

    def test_non_engine_validation_error_propagates(self, tmp_path):
        """Errors that aren't engine/voice incompatibility (bad SSML,
        throttling) must NOT be swallowed by the fallback loop -- they
        propagate so the daemon's retry/surface path handles them."""
        from speeker.engines import PollyEngine, _polly_variant_cache
        _polly_variant_cache.clear()
        boto3, client = _mock_boto3_returning(b"")
        client.synthesize_speech.side_effect = Exception(
            "An error occurred (ThrottlingException): Rate exceeded"
        )
        with patch.dict(sys.modules, {"boto3": boto3}), \
             patch.dict("os.environ", {"SPEEKER_DIR": str(tmp_path)}):
            eng = PollyEngine()
            with pytest.raises(Exception, match="ThrottlingException"):
                eng.generate("hi", "Joanna")
        # Only one attempt -- we don't retry across variants for non-engine errors.
        assert client.synthesize_speech.call_count == 1

    def test_uses_configured_profile_and_region(self, tmp_path):
        from speeker.engines import PollyEngine
        from speeker.config import save_config
        boto3, client = _mock_boto3_returning(np.array([0], dtype=np.int16).tobytes())
        with patch.dict(sys.modules, {"boto3": boto3}), \
             patch.dict("os.environ", {"SPEEKER_DIR": str(tmp_path)}):
            save_config({"polly": {"profile": "personal", "region": "us-west-2"}})
            eng = PollyEngine()
            eng.generate("hi", "Joanna")
        boto3.Session.assert_called_once_with(profile_name="personal")
        boto3.Session.return_value.client.assert_called_once_with(
            "polly", region_name="us-west-2"
        )

    def test_registry_creates_polly(self):
        assert get_engine("polly").name == "polly"
