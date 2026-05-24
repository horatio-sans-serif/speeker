"""TTS engine abstraction and registry.

Each engine exposes a uniform interface so the CLI and the player daemon can
dispatch by name instead of hardcoding one engine. Engine instances are cached
singletons (they hold warm model state where applicable). Heavy / optional
imports (pocket_tts, kokoro, boto3) are done lazily inside methods so importing
this module never requires them.
"""

from __future__ import annotations

import numpy as np


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
        if voice not in self._voice_states:
            from .voices import get_pocket_tts_voice_path
            model = self._get_model()
            self._voice_states[voice] = model.get_state_for_audio_prompt(
                get_pocket_tts_voice_path(voice)
            )
        return self._voice_states[voice]

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
        audio = model.generate_audio(self._voice_state(voice), text)
        return audio.numpy(), model.sample_rate

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

    def generate(self, text, voice, *, is_ssml=False, **options):
        from .config import get_polly_config
        from .ssml import ensure_speak_wrapped
        cfg = get_polly_config()
        variant = options.get("polly_engine") or cfg.get("engine") or "neural"
        if is_ssml:
            payload, text_type = ensure_speak_wrapped(text), "ssml"
        else:
            payload, text_type = text, "text"
        resp = self._get_client().synthesize_speech(
            Text=payload,
            VoiceId=voice,
            Engine=variant,
            OutputFormat="pcm",
            SampleRate="16000",
            TextType=text_type,
        )
        pcm = resp["AudioStream"].read()
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        return audio, 16000


_ENGINES: dict[str, BaseEngine] = {}


def _create_engine(name: str) -> BaseEngine:
    if name == "pocket-tts":
        return PocketTTSEngine()
    if name == "kokoro":
        return KokoroEngine()
    if name == "polly":
        return PollyEngine()
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
