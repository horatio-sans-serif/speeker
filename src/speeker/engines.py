"""TTS engine abstraction and registry.

Each engine exposes a uniform interface so the CLI and the player daemon can
dispatch by name instead of hardcoding one engine. Engine instances are cached
singletons (they hold warm model state where applicable). Heavy / optional
imports (pocket_tts, kokoro, boto3) are done lazily inside methods so importing
this module never requires them.
"""

from __future__ import annotations

import sys

import numpy as np

# pocket-tts voice cloning is a continuation model: cloned voices start at full
# amplitude from sample 0 (no pre-roll), so the audio device's startup latency
# clips the first word on playback. Ensure a small lead-in. Built-in voices
# already include ~0.9s of lead, so this is a no-op for them.
_MIN_LEADING_SILENCE_SECONDS = 0.25


def _ensure_min_leading_silence(
    audio: np.ndarray, sample_rate: int, min_seconds: float = _MIN_LEADING_SILENCE_SECONDS
) -> np.ndarray:
    """Prepend silence so *audio* has at least *min_seconds* of lead-in.

    No-op when the audio already starts with enough silence.
    """
    if audio.size == 0:
        return audio
    above = np.abs(audio) > 0.02
    onset = int(np.argmax(above)) if bool(above.any()) else audio.size
    if onset / sample_rate >= min_seconds:
        return audio
    pad = int(round((min_seconds - onset / sample_rate) * sample_rate))
    if pad <= 0:
        return audio
    return np.concatenate([np.zeros(pad, dtype=audio.dtype), audio])


class BaseEngine:
    """Interface implemented by every TTS engine."""

    name: str = ""
    supports_ssml: bool = False

    def default_voice(self) -> str:
        raise NotImplementedError

    def list_voices(self) -> dict[str, str]:
        raise NotImplementedError

    def validate_voice(self, voice: str) -> bool:
        raise NotImplementedError

    def generate(
        self, text: str, voice: str, *, is_ssml: bool = False, **options
    ) -> tuple[np.ndarray, int]:
        """Return (audio samples as a numpy array, sample_rate).

        Samples are nominally floating-point near [-1, 1]; callers must clip to
        [-1, 1] and cast to int16 before WAV output.
        """
        raise NotImplementedError

    def warm(self) -> None:
        """Pre-load any heavy state. No-op by default."""

    def unload(self) -> None:
        """Free heavy state. No-op by default."""


class PocketTTSEngine(BaseEngine):
    name = "pocket-tts"
    supports_ssml = False

    def __init__(self) -> None:
        self._model = None
        self._voice_states: dict[str, object] = {}

    def _get_model(self):
        if self._model is None:
            from pocket_tts import TTSModel
            self._model = TTSModel.load_model()
        return self._model

    def _voice_state(self, voice: str):
        # Key on the RESOLVED voice (built-in name or reference WAV path), not
        # the requested name. A custom voice whose path is briefly unresolvable
        # falls back to the default; keying on the name would then cache that
        # fallback under the custom name forever (even after the path is fixed).
        from .voices import get_pocket_tts_voice_path
        resolved = get_pocket_tts_voice_path(voice)
        key = str(resolved)
        if key not in self._voice_states:
            model = self._get_model()
            # truncate=True caps the audio prompt at 30s. pocket-tts 2.x emits
            # immediate EOS (near-silent output) on long un-truncated prompts;
            # its own server passes truncate=True. Ignored for predefined
            # voices (passed as a name string).
            self._voice_states[key] = model.get_state_for_audio_prompt(
                resolved, truncate=True
            )
        return self._voice_states[key]

    def default_voice(self) -> str:
        from .voices import DEFAULT_POCKET_TTS_VOICE
        return DEFAULT_POCKET_TTS_VOICE

    def list_voices(self) -> dict[str, str]:
        from .voices import POCKET_TTS_VOICES
        return dict(POCKET_TTS_VOICES)

    def validate_voice(self, voice: str) -> bool:
        from .voices import validate_voice
        return validate_voice("pocket-tts", voice)

    def generate(self, text, voice, *, is_ssml=False, **options):
        model = self._get_model()
        audio = model.generate_audio(self._voice_state(voice), text).numpy()
        audio = _ensure_min_leading_silence(audio, model.sample_rate)
        return audio, model.sample_rate

    def warm(self) -> None:
        self._voice_state(self.default_voice())

    def unload(self) -> None:
        self._model = None
        self._voice_states = {}


class KokoroEngine(BaseEngine):
    name = "kokoro"
    supports_ssml = False

    def __init__(self) -> None:
        self._pipeline = None

    def _get_pipeline(self):
        if self._pipeline is None:
            from kokoro import KPipeline
            self._pipeline = KPipeline(lang_code="a")
        return self._pipeline

    def default_voice(self) -> str:
        from .voices import DEFAULT_KOKORO_VOICE
        return DEFAULT_KOKORO_VOICE

    def list_voices(self) -> dict[str, str]:
        from .voices import KOKORO_VOICES
        return dict(KOKORO_VOICES)

    def validate_voice(self, voice: str) -> bool:
        from .voices import validate_voice
        return validate_voice("kokoro", voice)

    def generate(self, text, voice, *, is_ssml=False, **options):
        pipeline = self._get_pipeline()
        chunks = [audio for _, _, audio in pipeline(text, voice=voice)]
        if not chunks:
            raise ValueError("Kokoro generated no audio")
        return np.concatenate(chunks), 24000

    def warm(self) -> None:
        self._get_pipeline()

    def unload(self) -> None:
        self._pipeline = None


# Quality-preference order for auto-fallback when a voice doesn't support
# the configured Polly engine variant. neural first (broad modern support
# + good quality), then long-form and generative, with standard as the
# universal floor that essentially every Polly voice supports.
_POLLY_VARIANT_FALLBACK_ORDER: tuple[str, ...] = (
    "neural", "long-form", "generative", "standard",
)

# Per-voice resolved-variant cache. Once we discover that a given
# (voice, requested_variant) pair actually works with some variant,
# reuse it so a misconfigured queue doesn't pay a failed Polly
# round-trip on every utterance. Keyed by (voice, requested) ->
# working_variant. Process-local; cleared on engine unload.
_polly_variant_cache: dict[tuple[str, str], str] = {}


def _is_unsupported_engine_error(exc: Exception) -> bool:
    """True when *exc* is Polly's 'voice does not support engine' rejection.

    Polly raises ``ValidationException`` for several distinct reasons;
    we only auto-fallback on the engine/voice incompatibility, not on
    (say) a malformed SSML or unknown voice -- those are real errors
    the user must fix. Matched on the message text because boto3 surfaces
    all of these under the same exception class and error code.
    """
    msg = str(exc).lower()
    return "does not support the selected engine" in msg


class PollyEngine(BaseEngine):
    name = "polly"
    supports_ssml = True

    def __init__(self) -> None:
        self._client = None

    def _get_client(self):
        if self._client is None:
            import boto3
            from .config import get_polly_config
            cfg = get_polly_config()
            session_kwargs = {}
            if cfg.get("profile"):
                session_kwargs["profile_name"] = cfg["profile"]
            session = boto3.Session(**session_kwargs)
            client_kwargs = {}
            if cfg.get("region"):
                client_kwargs["region_name"] = cfg["region"]
            self._client = session.client("polly", **client_kwargs)
        return self._client

    def default_voice(self) -> str:
        from .voices import DEFAULT_POLLY_VOICE
        return DEFAULT_POLLY_VOICE

    def list_voices(self) -> dict[str, str]:
        from .voices import POLLY_VOICES
        return dict(POLLY_VOICES)

    def validate_voice(self, voice: str) -> bool:
        from .voices import validate_voice
        return validate_voice("polly", voice)

    def _build_payload(self, text: str, is_ssml: bool, variant: str) -> tuple[str, str]:
        """Return (payload, text_type) for a given variant.

        SSML sanitization is variant-aware (e.g. <emphasis> is stripped
        for neural/standard), so the payload must be rebuilt per variant
        when we fall back across engines.
        """
        if is_ssml:
            from .ssml import ensure_speak_wrapped, sanitize_ssml
            return sanitize_ssml(ensure_speak_wrapped(text), polly_engine=variant), "ssml"
        return text, "text"

    def _synthesize(self, payload: str, voice: str, variant: str, text_type: str):
        return self._get_client().synthesize_speech(
            Text=payload,
            VoiceId=voice,
            Engine=variant,
            OutputFormat="pcm",
            SampleRate="16000",
            TextType=text_type,
        )

    def generate(self, text, voice, *, is_ssml=False, **options):
        from .config import get_polly_config
        from .polly_chunk import split_payload
        cfg = get_polly_config()
        requested = options.get("polly_engine") or cfg.get("engine") or "neural"

        # Polly's real-time API caps a request at 3000 billed / 6000 total
        # characters, so long input is split here and the audio concatenated.
        chunks = split_payload(text, is_ssml)
        if not chunks:
            return np.zeros(0, dtype=np.float32), 16000
        parts = [
            self._synthesize_chunk(chunk, voice, is_ssml, requested)
            for chunk in chunks
        ]
        audio = parts[0] if len(parts) == 1 else np.concatenate(parts)
        return audio, 16000

    def _synthesize_chunk(self, text, voice, is_ssml, requested) -> np.ndarray:
        """Synthesize one within-limits chunk, resolving the engine variant.

        Build the ordered candidate list: start from the variant we last
        resolved for this (voice, requested) pair -- or the requested variant
        itself -- then append the quality-ordered fallbacks, de-duplicated. A
        voice/variant mismatch (e.g. a 'Gregory' voice with a 'generative'
        engine -- Gregory is long-form only) transparently resolves to a
        supported variant instead of failing. Polly is the authority on what's
        supported, so we keep no hardcoded per-voice matrix. The resolved
        variant is cached so later chunks skip re-probing.
        """
        cache_key = (voice, requested)
        start = _polly_variant_cache.get(cache_key, requested)
        candidates: list[str] = [start]
        for v in _POLLY_VARIANT_FALLBACK_ORDER:
            if v not in candidates:
                candidates.append(v)

        last_err: Exception | None = None
        for variant in candidates:
            payload, text_type = self._build_payload(text, is_ssml, variant)
            try:
                resp = self._synthesize(payload, voice, variant, text_type)
            except Exception as e:  # noqa: BLE001 - inspect, then re-raise or fall back
                if _is_unsupported_engine_error(e):
                    # This variant isn't valid for this voice; try the next.
                    last_err = e
                    continue
                # Any other error (bad SSML, throttling, auth) is real --
                # let it propagate so the daemon's retry/surface path
                # handles it honestly rather than masking it as a
                # variant problem.
                raise
            # Success. Record the working variant and note any substitution
            # so a misconfigured queue is visible in the daemon log.
            if variant != requested:
                print(
                    f"[polly] voice {voice!r} does not support engine "
                    f"{requested!r}; used {variant!r} instead. Set this "
                    f"queue's Polly engine (or voice) to a compatible pair "
                    f"to silence this.",
                    file=sys.stderr,
                    flush=True,
                )
            _polly_variant_cache[cache_key] = variant
            pcm = resp["AudioStream"].read()
            return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0

        # Every candidate was rejected as an unsupported engine -- this
        # shouldn't happen (standard is near-universal) but surface the
        # last error rather than returning silence.
        raise last_err if last_err is not None else RuntimeError(
            f"Polly synthesis failed for voice {voice!r}: no supported engine variant."
        )

    def unload(self) -> None:
        # Drop the boto3 client and the resolved-variant cache so a
        # config change (e.g. switching profiles or the default engine)
        # is picked up fresh on the next synthesize after an idle unload.
        self._client = None
        _polly_variant_cache.clear()


class ElevenLabsEngine(BaseEngine):
    name = "elevenlabs"
    supports_ssml = False

    def __init__(self) -> None:
        self._client = None

    def _get_client(self):
        if self._client is None:
            from . import elevenlabs_api
            self._client = elevenlabs_api.get_client()
        return self._client

    def default_voice(self) -> str:
        from .voices import get_default_voice
        voice = get_default_voice("elevenlabs")
        if voice:
            return voice
        # Fall back to the first cloned el voice if one exists.
        from .voices import _elevenlabs_custom_voices
        el = _elevenlabs_custom_voices()
        if el:
            return next(iter(el))
        raise ValueError(
            "No ElevenLabs voice configured. Clone one with "
            "'speeker voice-clone NAME SOURCE --provider elevenlabs' or set "
            "elevenlabs.voice in config."
        )

    def list_voices(self) -> dict[str, str]:
        from .voices import _elevenlabs_custom_voices
        return _elevenlabs_custom_voices()

    def validate_voice(self, voice: str) -> bool:
        from .voices import validate_voice
        return validate_voice("elevenlabs", voice)

    def generate(self, text, voice, *, is_ssml=False, **options):
        from . import elevenlabs_api
        from .config import get_elevenlabs_config
        from .voice_clone import get_elevenlabs_voice_id

        cfg = get_elevenlabs_config()
        output_format = cfg.get("output_format", "pcm_24000")
        # We decode raw PCM directly (like Polly); other containers (mp3_*)
        # would need a decode step we don't do here.
        if not output_format.startswith("pcm_"):
            raise ValueError(
                f"elevenlabs.output_format must be a pcm_* format, got {output_format!r}"
            )
        sample_rate = int(output_format.split("_", 1)[1])
        # A cloned-voice name resolves to its server voice_id; anything else is
        # treated as a raw ElevenLabs voice_id and passed through.
        voice_id = get_elevenlabs_voice_id(voice) or voice
        pcm = elevenlabs_api.synthesize(
            voice_id,
            text,
            model_id=cfg.get("model", "eleven_multilingual_v2"),
            output_format=output_format,
        )
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        return audio, sample_rate

    def unload(self) -> None:
        self._client = None


_ENGINES: dict[str, BaseEngine] = {}


def _create_engine(name: str) -> BaseEngine:
    if name == "pocket-tts":
        return PocketTTSEngine()
    if name == "kokoro":
        return KokoroEngine()
    if name == "polly":
        return PollyEngine()
    if name == "elevenlabs":
        return ElevenLabsEngine()
    raise ValueError(f"Unknown engine: {name}")


def get_engine(name: str | None) -> BaseEngine:
    """Return the cached engine singleton for *name* (default engine if None)."""
    from .voices import DEFAULT_ENGINE
    name = name or DEFAULT_ENGINE
    if name not in _ENGINES:
        _ENGINES[name] = _create_engine(name)
    return _ENGINES[name]


def unload_all() -> None:
    """Unload and drop every cached engine (frees model memory)."""
    for engine in _ENGINES.values():
        engine.unload()
    _ENGINES.clear()


def prepare_payload(
    engine: BaseEngine,
    text: str,
    *,
    is_ssml: bool,
    emulate: bool,
    acronyms_file: str | None = None,
) -> tuple[str, bool]:
    """Resolve the text to send to *engine* and whether it should be SSML.

    - Non-SSML text passes through unchanged (caller does plain preprocessing).
    - SSML for an SSML-capable engine passes through (engine wraps it).
    - SSML for a local engine is emulated (if enabled) or stripped to plain text.
    """
    if not is_ssml:
        return text, False
    if engine.supports_ssml:
        return text, True
    from .ssml import emulate_ssml, strip_ssml, load_acronyms
    if emulate:
        return emulate_ssml(text, load_acronyms(acronyms_file)), False
    return strip_ssml(text), False
