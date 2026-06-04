#!/usr/bin/env python3
"""Speeker playback daemon - watches SQLite queue and plays TTS immediately.

Keeps TTS model warm for low-latency speech generation.
"""

import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .config import get_auto_label_config, get_player_config
from .engines import get_engine, prepare_payload, unload_all
from .ssml import looks_like_ssml
from .paths import (
    audio_dir as _audio_dir,
    ensure_dir,
    player_lock_path,
    tone_intro_path as _tone_intro_path,
    tone_outro_path as _tone_outro_path,
    tones_dir as _tones_dir,
)
from .queue_db import (
    clear_currently_playing,
    get_connection,
    get_last_played_queue,
    get_last_utterance_time,
    get_pending_count,
    get_pending_for_session,
    get_sessions_with_pending,
    get_settings,
    get_spoken_queue_title,
    mark_played,
    relative_time,
    get_queue_label,
    set_currently_playing,
    set_last_utterance_time,
    cleanup_old_entries,
    update_metadata,
)

# Timing
PAUSE_BETWEEN_MESSAGES = 0.3
PAUSE_BETWEEN_SESSIONS = 0.5
POLL_INTERVAL = 0.5  # Check queue every 500ms

# How long before we re-announce "This is Claude Code"
ANNOUNCE_THRESHOLD_MINUTES = 30

# Cached sound files
_intro_sound_path: Path | None = None
_outro_sound_path: Path | None = None
_tone_cache: dict[str, Path] = {}
_interpretation_cue_cache: dict[str, Path] = {}

# Musical note parsing for tone tokens.
#
# Notation: $<letter><accidental?><octave>[:<multiplier>]
#   - $C4        -- C, octave 4, default duration (1x base)
#   - $Eb4       -- E-flat, octave 4
#   - $C5:2      -- C, octave 5, 2x base duration
#   - $C5:.5     -- C, octave 5, half of base
#   - $C5:4      -- C, octave 5, 4x base (whole note if base = quarter)
#
# The multiplier is a positive float multiplied by the synthesis
# function's per-note base duration. There's no de facto standard for
# inline music in plain text -- ABC and Lilypond both reuse the digit
# after the letter for length, which collides with our scientific-pitch
# octave digit. The explicit `:` separator avoids that ambiguity.
NOTE_PATTERN = re.compile(r"^\s*\$([A-Ga-g])([b#]?)([0-8])(?::([0-9]*\.?[0-9]+))?")
_NOTE_BODY_RE = re.compile(r"([A-Ga-g])([b#]?)([0-8])(?::([0-9]*\.?[0-9]+))?")


def parse_note_token(token: str) -> tuple[str, int, float] | None:
    """Parse a note token like 'Eb4' or 'Eb4:2' into ``(note, octave, multiplier)``.

    The multiplier is a positive float; defaults to ``1.0`` when no ``:``
    qualifier is present. Returns ``None`` for unparseable input.
    Multipliers of zero or negative parse-but-clamp to 1.0 so a bad
    user-supplied tune doesn't produce a 0-length tone.
    """
    match = _NOTE_BODY_RE.match(token)
    if not match:
        return None
    note = match.group(1).lower()
    accidental = match.group(2)
    octave = int(match.group(3))
    # tones uses '#' for sharp, 'b' for flat in note name
    if accidental:
        note = note + accidental
    mult_str = match.group(4)
    multiplier = 1.0
    if mult_str:
        try:
            m = float(mult_str)
            if m > 0:
                multiplier = m
        except ValueError:
            pass
    return note, octave, multiplier


# Trailing $Note tokens (e.g. an outro chord wrapping a TTS preview).
# Anchored at the end so we don't accidentally consume tokens that
# happen to appear mid-text.
_TRAILING_NOTE_PATTERN = re.compile(
    r"\s*\$([A-Ga-g])([b#]?)([0-8])(?::([0-9]*\.?[0-9]+))?\s*$"
)


def extract_tone_tokens(text: str) -> tuple[list[str], str, list[str]]:
    """Extract $Note tokens from the start AND end of ``text``.

    Returns ``(leading, body, trailing)``. The leading tokens are played
    before TTS (with TTS generation overlapped in ``speak_text``), and
    the trailing tokens are played after. Both lists preserve any
    ``:multiplier`` suffix so synthesis can apply per-note durations.

    The trailing pattern is anchored to end-of-string so $Note sequences
    appearing inside the body don't get stripped -- only a run of one or
    more $Note tokens at the very end is treated as an outro.
    """
    leading: list[str] = []
    remaining = text
    while True:
        match = NOTE_PATTERN.match(remaining)
        if not match:
            break
        full = match.group(1) + match.group(2) + match.group(3)
        if match.group(4):
            full += ":" + match.group(4)
        leading.append(full)
        remaining = remaining[match.end():]

    trailing: list[str] = []
    while True:
        m = _TRAILING_NOTE_PATTERN.search(remaining)
        if not m:
            break
        full = m.group(1) + m.group(2) + m.group(3)
        if m.group(4):
            full += ":" + m.group(4)
        trailing.insert(0, full)  # preserve textual order
        remaining = remaining[:m.start()]

    return leading, remaining.strip(), trailing


def get_audio_player() -> list[str] | None:
    """Get the appropriate audio player command for this platform."""
    system = platform.system()

    if system == "Darwin":
        if shutil.which("afplay"):
            return ["afplay"]
    elif system == "Linux":
        if shutil.which("paplay"):
            return ["paplay"]
        if shutil.which("aplay"):
            return ["aplay", "-q"]
        if shutil.which("ffplay"):
            return ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"]
    elif system == "Windows":
        if shutil.which("powershell"):
            return None

    if shutil.which("ffplay"):
        return ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"]

    return None


AUDIO_PLAYER = get_audio_player()


def unload_tts_model() -> None:
    """Unload all cached TTS engines to free memory."""
    unload_all()


def generate_tone(frequencies: list[int], rising: bool = True) -> Path:
    """Generate a multi-note tone.

    Args:
        frequencies: List of frequencies to play in sequence
        rising: If True, play in order (intro). If False, reverse (outro).
    """
    import math
    import struct
    import wave

    tone_path = _tone_intro_path() if rising else _tone_outro_path()

    if tone_path.exists():
        return tone_path

    if not rising:
        frequencies = list(reversed(frequencies))

    sample_rate = 44100
    amplitude = 0.3
    note_duration = 0.12
    gap_duration = 0.03

    samples = []
    for note_idx, frequency in enumerate(frequencies):
        n_samples = int(sample_rate * note_duration)
        fade_samples = int(sample_rate * 0.015)

        for i in range(n_samples):
            t = i / sample_rate
            if i < fade_samples:
                envelope = i / fade_samples
            elif i > n_samples - fade_samples:
                envelope = (n_samples - i) / fade_samples
            else:
                envelope = 1.0
            value = amplitude * envelope * math.sin(2 * math.pi * frequency * t)
            samples.append(int(value * 32767))

        if note_idx < len(frequencies) - 1:
            gap_samples = int(sample_rate * gap_duration)
            samples.extend([0] * gap_samples)

    ensure_dir(tone_path.parent)
    with wave.open(str(tone_path), "w") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(struct.pack(f"{len(samples)}h", *samples))

    return tone_path


def _tones_from_config(key: str, fallback: list[str]) -> list[str]:
    """Read intro/outro note list from config, validating each entry.

    Falls back to *fallback* if config is missing or any entry doesn't
    parse as ``[A-G][b#]?[0-8]`` -- a malformed config should not silence
    the daemon.
    """
    from .config import get_tones_config
    cfg = get_tones_config()
    raw = cfg.get(key, fallback)
    if not isinstance(raw, list) or not raw:
        return fallback
    cleaned = []
    for entry in raw:
        if not isinstance(entry, str):
            continue
        if parse_note_token(entry) is None:
            continue
        cleaned.append(entry)
    return cleaned or fallback


def _tone_duration_seconds() -> float:
    """Per-note synthesis duration for intro/outro tones, from config."""
    from .config import get_tones_config
    try:
        return float(get_tones_config().get("duration_seconds", 0.12))
    except (TypeError, ValueError):
        return 0.12


def _resolve_slot_notes(slot: str, queue: str | None, fallback: list[str]) -> list[str]:
    """Find the tune for ``slot`` honoring per-queue / per-interpretation rules.

    Intro/outro have no interpretation context, so only queue-tier rules
    can match. Falls back to ``config.tones.<slot>`` when no rule applies.
    """
    from .tone_rules import resolve_tone_notes
    override = resolve_tone_notes(slot, queue, None)
    if override:
        # The resolver already validated note notation; trust its output.
        return override
    return _tones_from_config(slot, fallback)


def get_intro_sound(queue: str | None = None) -> Path:
    """Get path to the intro tone, synthesized from configured notes.

    With no queue context the global ``config.tones.intro`` is used. With
    a queue, ``tone_rules`` is consulted first so a per-queue intro can
    override the global tune. The synthesized WAV is cached by content
    hash (see ``generate_combined_tones_from_tokens``) so per-queue tunes
    don't trample each other.
    """
    notes = _resolve_slot_notes("intro", queue, ["E4", "G4", "C5"])
    return generate_combined_tones_from_tokens(notes, duration=_tone_duration_seconds())


def get_outro_sound(queue: str | None = None) -> Path:
    """Get path to the outro tone, synthesized from configured notes.

    With a queue, ``tone_rules`` is consulted first so a per-queue outro
    overrides the global tune. Falls back to ``config.tones.outro``.
    """
    notes = _resolve_slot_notes("outro", queue, ["C5", "G4", "E4"])
    return generate_combined_tones_from_tokens(notes, duration=_tone_duration_seconds())


def play_audio(audio_path: Path, verbose: bool = False) -> bool:
    """Play an audio file. Returns True on success."""
    if not audio_path.exists():
        if verbose:
            print(f"[WARN] Audio file not found: {audio_path}", file=sys.stderr)
        return False

    if AUDIO_PLAYER is None:
        if platform.system() == "Windows":
            try:
                ps_cmd = f'(New-Object Media.SoundPlayer "{audio_path}").PlaySync()'
                subprocess.run(["powershell", "-c", ps_cmd], check=True)
                return True
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                if verbose:
                    print(f"[ERROR] PowerShell audio failed: {e}", file=sys.stderr)
                return False
        else:
            print("[ERROR] No audio player found.", file=sys.stderr)
            return False

    if verbose:
        print(f"[PLAY] {audio_path}", file=sys.stderr)

    try:
        subprocess.run([*AUDIO_PLAYER, str(audio_path)], check=True)
        return True
    except subprocess.CalledProcessError as e:
        if verbose:
            print(f"[ERROR] Audio playback failed: {e}", file=sys.stderr)
        return False
    except FileNotFoundError:
        print(f"[ERROR] Audio player not found: {AUDIO_PLAYER[0]}", file=sys.stderr)
        return False


class TTSError(Exception):
    """Raised by ``generate_tts`` when audio synthesis fails.

    Carries the engine name, voice, and a short text snippet so a single
    line in /tmp/speeker-player.err identifies the failed utterance. The
    daemon catches this to drive its retry policy (see ``process_queue``).
    Callers that don't care about retry can catch ``Exception`` -- TTSError
    inherits from it.
    """

    def __init__(self, message: str, *, engine: str | None = None,
                 voice: str | None = None, text: str | None = None) -> None:
        super().__init__(message)
        self.engine = engine
        self.voice = voice
        self.text = text


def generate_tts(
    text: str,
    voice: str | None = None,
    speed: float = 1.0,
    save_path: Path | None = None,
    verbose: bool = False,
    *,
    engine: str | None = None,
    is_ssml: bool = False,
    polly_engine: str | None = None,
    effects_preset: str | None = None,
) -> Path:
    """Generate TTS audio for text using the named engine.

    ``effects_preset`` -- when supplied, overrides the saved effects
    preset for this one utterance (used by ``/api/effects/try`` to
    preview an unsaved selection without mutating ``config.json``).

    Raises ``TTSError`` on any synthesis failure -- the underlying
    engine exception is chained via ``__cause__``. Errors are logged to
    stderr unconditionally (not gated on ``verbose``) so failures stay
    visible in /tmp/speeker-player.err even when the daemon was launched
    without ``--verbose``.
    """
    try:
        import numpy as np
        from scipy.io import wavfile
        from .preprocessing import preprocess_for_tts
        from .config import get_ssml_config

        if verbose:
            print(f"[TTS] {text[:60]}{'...' if len(text) > 60 else ''}", file=sys.stderr)

        eng = get_engine(engine)
        voice = voice or eng.default_voice()
        is_ssml = is_ssml or looks_like_ssml(text)

        ssml_cfg = get_ssml_config()
        if is_ssml:
            payload, ssml_for_engine = prepare_payload(
                eng, text, is_ssml=True,
                emulate=ssml_cfg.get("emulate_for_local", False),
                acronyms_file=ssml_cfg.get("acronyms_file"),
            )
        else:
            from .ssml import load_acronyms
            # Pass the engine name so per-engine pronunciation override
            # variants (e.g. polly-specific vs pocket-tts-specific
            # respelling for the same word) resolve to the right value.
            payload = preprocess_for_tts(
                text,
                acronyms=load_acronyms(ssml_cfg.get("acronyms_file")),
                engine=engine,
            )
            ssml_for_engine = False

        audio_np, sample_rate = eng.generate(
            payload, voice, is_ssml=ssml_for_engine, polly_engine=polly_engine
        )

        if speed != 1.0 and speed > 0:
            from scipy import signal
            new_length = int(len(audio_np) / speed)
            audio_np = signal.resample(audio_np, new_length)

        # Audio effects chain (reverb / EQ / compressor / etc). No-op when
        # the configured preset is "off" or the optional pedalboard dep is
        # missing. Runs only on TTS speech -- tones synthesized via the
        # `tones` library never pass through this function.
        from .effects import apply_effects
        audio_np = apply_effects(
            audio_np, sample_rate, preset_override=effects_preset
        )

        audio_normalized = np.clip(audio_np, -1.0, 1.0)
        audio_int16 = (audio_normalized * 32767).astype(np.int16)

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            wavfile.write(str(save_path), sample_rate, audio_int16)
            return save_path
        else:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                wavfile.write(f.name, sample_rate, audio_int16)
                return Path(f.name)

    except Exception as e:
        # Always log -- the daemon usually runs without --verbose, so a
        # silent return makes failures invisible (cards lose their Play
        # button with no clue why). One concise line per failure makes
        # /tmp/speeker-player.err greppable.
        snippet = (text or "")[:60].replace("\n", " ")
        suffix = "..." if text and len(text) > 60 else ""
        print(
            f"[TTS-ERROR] engine={engine or 'default'} voice={voice or 'default'}"
            f" err={type(e).__name__}: {e} text={snippet!r}{suffix}",
            file=sys.stderr,
            flush=True,
        )
        raise TTSError(
            str(e), engine=engine, voice=voice, text=text,
        ) from e


def _tones_effects_suffix() -> str:
    """Return a cache-key suffix capturing the active tones-effects state.

    When ``tones.apply_effects`` is True AND the active effects preset is
    not ``off``, the rendered WAV bakes in the chain -- so a flip from
    ``natural`` to ``robot`` (or toggling apply_effects off) must yield a
    different cache key. Empty string means "no effects bake".
    """
    from .config import get_tones_config, get_effects_config
    if not bool(get_tones_config().get("apply_effects", True)):
        return ""
    preset = (get_effects_config().get("preset") or "off").strip()
    if preset == "off":
        return ""
    return f"_fx-{preset}"


def generate_combined_tones_from_tokens(tokens: list[str], duration: float = 0.8) -> Path:
    """Generate a single WAV with tones using the tones library.

    Each token may carry a ``:multiplier`` suffix (e.g. ``"C5:4"``) that
    scales the per-note duration. ``duration`` is the base; the actual
    per-note length is ``duration * multiplier``. A tune like
    ``["G4", "E4", "C5:4"]`` therefore plays two short notes and one
    long-ringing note -- the canonical NBC chime pattern.

    When ``tones.apply_effects`` is True (default) and an effects preset
    is active, the synthesized WAV is post-processed through the chain
    so tones share the same audio character as TTS speech. The preset
    name is folded into the cache key so toggling presets regenerates.
    """
    from tones import SINE_WAVE
    from tones.mixer import Mixer

    # Cache key: bumped to v7 because per-note multipliers change the
    # rendered audio for tokens that carry a `:N` suffix. Tokens without
    # the suffix still render identically to v6, but the version bump
    # also guarantees a clean break for any cached files that happen to
    # share a name pattern. The fx- suffix isolates effects-processed
    # variants from the dry baseline.
    fx_suffix = _tones_effects_suffix()
    cache_key = "_".join(tokens) + f"_{duration}_v7{fx_suffix}"
    if cache_key in _tone_cache and _tone_cache[cache_key].exists():
        return _tone_cache[cache_key]

    tone_dir = ensure_dir(_tones_dir())
    # Filesystem-safe: `:` is fine on macOS HFS+/APFS but a few sync tools
    # and Windows mounts choke on it. Strip from the on-disk name only.
    safe_key = cache_key.replace(":", "x")
    tone_path = tone_dir / f"combined_{safe_key}.wav"

    if tone_path.exists():
        _tone_cache[cache_key] = tone_path
        return tone_path

    # Use tones library for synthesis
    mixer = Mixer(44100, 0.5)
    mixer.create_track(0, SINE_WAVE, vibrato_frequency=5.5, vibrato_variance=0.02, attack=0.01, decay=0.3)

    for token in tokens:
        parsed = parse_note_token(token)
        if parsed:
            note, octave, multiplier = parsed
            mixer.add_note(0, note=note, octave=octave, duration=duration * multiplier)

    mixer.write_wav(str(tone_path))
    # Bake the active effects chain into the file when enabled. apply_effects_to_wav
    # is a no-op for preset=off or missing pedalboard, so this is safe to
    # always call -- the fx_suffix above ensures cache integrity either way.
    if fx_suffix:
        from .effects import apply_effects_to_wav
        apply_effects_to_wav(tone_path)
    _tone_cache[cache_key] = tone_path
    return tone_path


def play_tone_tokens(
    tokens: list[str],
    verbose: bool = False,
    *,
    duration: float | None = None,
) -> None:
    """Play a sequence of tone tokens (e.g., ["Eb4"]) as a single audio file.

    *duration* -- per-note seconds. When ``None``, falls back to
    ``generate_combined_tones_from_tokens``'s default (0.8s). Callers
    that want to honor the configured intro/outro duration (0.12s
    default) supply it explicitly via ``metadata.tone_duration``.
    """
    if not tokens:
        return

    if verbose:
        print(f"[TONE] Playing tokens: {tokens} dur={duration}", file=sys.stderr)

    if duration is None:
        tone_path = generate_combined_tones_from_tokens(tokens)
    else:
        tone_path = generate_combined_tones_from_tokens(tokens, duration=duration)
    play_audio(tone_path, verbose)


def synthesize_note_cue(name: str, spec: list[tuple[str, int, float]]) -> Path | None:
    """Synthesize a note cue with per-note durations into a cached WAV.

    Unlike generate_combined_tones_from_tokens, each note carries its own
    duration in seconds, so a cue can mix short notes with a ringing one. The
    cache key includes a hash of the spec, so editing the config regenerates
    the file."""
    if not spec:
        return None

    import hashlib

    from tones import SINE_WAVE
    from tones.mixer import Mixer

    digest = hashlib.md5(repr(spec).encode()).hexdigest()[:10]
    fx_suffix = _tones_effects_suffix()
    cache_key = f"{name}_{digest}{fx_suffix}"
    cached = _interpretation_cue_cache.get(cache_key)
    if cached and cached.exists():
        return cached

    tone_dir = ensure_dir(_tones_dir())
    safe_key = cache_key.replace(":", "x")
    cue_path = tone_dir / f"cue_{safe_key}.wav"
    if cue_path.exists():
        _interpretation_cue_cache[cache_key] = cue_path
        return cue_path

    mixer = Mixer(44100, 0.5)
    mixer.create_track(0, SINE_WAVE, vibrato_frequency=5.5, vibrato_variance=0.02,
                       attack=0.01, decay=0.3)
    prev: tuple[str, int] | None = None
    for note, octave, seconds in spec:
        # A short rest between two consecutive identical notes makes a
        # repeated-note cue (e.g. WARNING's double Eb4) read clearly as two
        # tones rather than one slightly-wobbly tone.
        if prev == (note, octave):
            mixer.add_silence(0, duration=0.08)
        mixer.add_note(0, note=note, octave=octave, duration=seconds)
        prev = (note, octave)

    mixer.write_wav(str(cue_path))
    if fx_suffix:
        from .effects import apply_effects_to_wav
        apply_effects_to_wav(cue_path)
    _interpretation_cue_cache[cache_key] = cue_path
    return cue_path


def _cue_spec_from_notes(notes: list[str]) -> list[tuple[str, int, float]]:
    """Convert a tone_rules notes list into a synthesize_note_cue spec.

    Notes carry an optional ``:multiplier`` suffix. The base cue note
    duration is ``DEFAULT_NOTE_SECONDS`` (from interpretations); the
    multiplier scales each note's seconds. Malformed tokens are dropped
    so one bad entry doesn't abort the cue.
    """
    from .interpretations import DEFAULT_NOTE_SECONDS
    spec: list[tuple[str, int, float]] = []
    for token in notes:
        parsed = parse_note_token(token)
        if parsed is None:
            continue
        note, octave, mult = parsed
        spec.append((note, octave, DEFAULT_NOTE_SECONDS * mult))
    return spec


def render_interpretation_cue(
    name: str,
    verbose: bool = False,
    *,
    queue: str | None = None,
) -> Path | None:
    """Resolve an interpretation name to a playable cue path, or None.

    Consults ``tone_rules`` first: a per-queue+interpretation, per-queue,
    or per-interpretation rule overrides the global ``interpretations.map``
    entry. Falls back to ``resolve_interpretation`` when no rule matches.

    Unknown names, unsupported indication types, and missing sound files all
    return None (with a warning) so a misconfigured cue never aborts playback.
    """
    from .interpretations import notes_to_spec, resolve_interpretation
    from .tone_rules import resolve_tone_notes

    rule_notes = resolve_tone_notes("cue", queue, name)
    if rule_notes:
        spec = _cue_spec_from_notes(rule_notes)
        if spec:
            # Cache key folds in the queue so per-queue cues for the same
            # interpretation don't collide. The hash inside
            # synthesize_note_cue already differentiates by spec content,
            # but adding the queue makes the on-disk filename readable.
            cache_name = f"{name}@{queue}" if queue else name
            return synthesize_note_cue(cache_name, spec)

    indication = resolve_interpretation(name)
    if indication is None:
        if verbose:
            print(f"[WARN] Unknown interpretation '{name}', skipping cue", file=sys.stderr)
        return None

    kind = indication.get("type")
    if kind == "notes":
        return synthesize_note_cue(name, notes_to_spec(indication))
    if kind == "sound_file":
        path = Path(os.path.expanduser(str(indication.get("path", "")))).expanduser()
        if path.exists():
            return path
        if verbose:
            print(f"[WARN] Sound file for '{name}' not found: {path}", file=sys.stderr)
        return None

    if verbose:
        print(f"[WARN] Unknown indication type '{kind}' for '{name}'", file=sys.stderr)
    return None


def play_interpretation_cue(
    name: str,
    verbose: bool = False,
    *,
    queue: str | None = None,
) -> None:
    """Render and play an interpretation cue, then pause. No-op if unresolved.

    The cue blocks until it finishes (play_audio waits on the player), then we
    pause before the utterance — matching the 'play and wait, then continue'
    behavior for both note cues and sound files."""
    from .interpretations import pause_after_seconds

    cue = render_interpretation_cue(name, verbose, queue=queue)
    if cue is None:
        return
    if verbose:
        print(f"[CUE] {name}: {cue}", file=sys.stderr)
    play_audio(cue, verbose)
    time.sleep(pause_after_seconds())


def speak_text(
    text: str,
    voice: str | None = None,
    speed: float = 1.0,
    save_path: Path | None = None,
    verbose: bool = False,
    *,
    engine: str | None = None,
    is_ssml: bool = False,
    polly_engine: str | None = None,
    effects_preset: str | None = None,
    tone_duration: float | None = None,
) -> Path | None:
    """Generate and play TTS for text. Handles leading $Note tone tokens.

    The TTS audio is generated *first*, then any leading ``$Note`` tone plays
    immediately before the speech, then the speech, then any trailing tone:

        generate -> leading tone -> speech -> trailing tone

    Generating before the tone means a slow engine (e.g. chunked Polly on a
    long passage) leaves no dead air between the tone and the speech -- the
    tone always sits right against the utterance. On generation failure we
    raise before any tone plays, so a failed item produces no orphan tone.
    """
    leading_tones, clean_text, trailing_tones = extract_tone_tokens(text)

    def _play_trailing():
        if trailing_tones:
            play_tone_tokens(trailing_tones, verbose, duration=tone_duration)

    # Only-tone case: no speech to generate. Play leading then trailing
    # tones (rare; usually one or the other).
    if not clean_text:
        if leading_tones:
            play_tone_tokens(leading_tones, verbose, duration=tone_duration)
        _play_trailing()
        return save_path

    # Generate first. ``generate_tts`` raises ``TTSError`` on failure -- we
    # let it propagate *before any tone plays* so ``process_queue`` can drive
    # the retry policy and no orphan tone is heard for a failed item.
    audio_path = generate_tts(
        clean_text, voice=voice, speed=speed, save_path=save_path, verbose=verbose,
        engine=engine, is_ssml=is_ssml, polly_engine=polly_engine,
        effects_preset=effects_preset,
    )

    # Leading tone immediately before the speech, then speech, then trailing.
    if leading_tones:
        play_tone_tokens(leading_tones, verbose, duration=tone_duration)
    try:
        play_audio(audio_path, verbose)
    finally:
        if save_path is None and audio_path:
            try:
                audio_path.unlink()
            except OSError:
                pass
    _play_trailing()
    return save_path


def should_announce_intro() -> bool:
    """Check if we should say 'This is Claude Code'."""
    last_time = get_last_utterance_time()
    if last_time is None:
        return True
    threshold = datetime.now(timezone.utc) - timedelta(minutes=ANNOUNCE_THRESHOLD_MINUTES)
    return last_time < threshold


def compute_auto_label_prefix(
    session_id: str,
    last_utterance_at: datetime | None,
    last_queue_id: str | None,
    display_name_override: str | None = None,
) -> str | None:
    """Decide whether to prepend a queue-title prefix to the next utterance.

    Returns the spoken prefix (e.g. ``"$Eb4 project compass docs"``) when
    *either* the quiet threshold has elapsed since the last utterance *or*
    the queue context just changed; ``None`` means speak without a prefix.
    The literal word "project" precedes the title so the listener hears
    the category boundary before the project name itself.

    ``display_name_override`` -- if a caller supplied ``metadata.display_name``
    on any item in this session, use that verbatim as the spoken title
    instead of deriving one from the queue id. Lets callers escape the
    "hyphen-to-space" heuristic when the queue id has digits or other text
    that doesn't read aloud well (e.g. ``e2e-stt-1779972451``).

    The two signals combine intentionally:
      - **Time gap** catches "I forgot which project this was."
      - **Queue change** catches "back-to-back messages, but a different
        project just started." -- relabeling on context switch is the only
        way the listener knows the project flipped.
    """
    cfg = get_auto_label_config()
    if not cfg.get("enabled", True):
        return None

    title = display_name_override or get_spoken_queue_title(session_id)
    if title is None:
        return None  # default/unnamed queue: no meaningful title to speak

    threshold = float(cfg.get("quiet_threshold_seconds", 120))
    tone = cfg.get("tone", "$Eb4")

    # First-ever utterance: there is no prior context, so label it.
    if last_utterance_at is None:
        return f"{tone} project {title}"

    # Context switch: a different queue just spoke. Always relabel so the
    # listener notices the project changed even within a fast burst.
    if last_queue_id != session_id:
        return f"{tone} project {title}"

    # Same queue: only relabel after the quiet threshold has elapsed.
    elapsed = (datetime.now(timezone.utc) - last_utterance_at).total_seconds()
    if elapsed > threshold:
        return f"{tone} project {title}"

    return None


def build_session_script(
    session_id: str,
    items: list[dict],
    is_only_session: bool,
    auto_label_prefix: str | None = None,
) -> list[str]:
    """Build the speech script for a session's messages.

    One item -> one spoken line. No batch header ("For queue X, there are N
    messages") and no per-item framing ("First:", "Next:") -- both were
    designed when items had no inherent project context, but they buried any
    caller-supplied $Note tone token inside the line, where the line-anchored
    ``NOTE_PATTERN`` regex won't see it. The TTS engine then voiced "$Eb4" as
    the letters "EEB 4". Now items keep their leading $Note position so
    ``extract_tone_tokens`` plays it as audio.

    Auto-labeling: ``auto_label_prefix`` (e.g. ``"$Eb4 project compass docs"``) is
    prepended to the FIRST item that doesn't already start with a $Note
    token. Items that begin with a $Note token already carry project context
    (from the server's ``format_with_title`` or the hook's
    ``speak_label_only``), so they speak verbatim AND consume the
    "project context established" slot for the rest of the session -- a
    subsequent bare item won't get a duplicate label.

    The ``session_id`` and ``is_only_session`` parameters are retained for
    API compatibility (and so callers can compute their own auto-label
    decision externally), but they no longer affect the output.
    """
    _ = session_id, is_only_session  # retained for API compatibility

    lines: list[str] = []
    label_established = False
    for item in items:
        text = item["text"]
        time_ago = relative_time(item["created_at"])
        already_prefixed = text.lstrip().startswith("$")

        if already_prefixed:
            # Speak verbatim so the leading $Note reaches extract_tone_tokens.
            # The caller crafted the message; we don't insert "From N minutes
            # ago" into someone else's prefix structure.
            lines.append(text)
            label_established = True
            continue

        if auto_label_prefix and not label_established:
            if time_ago:
                lines.append(f"{auto_label_prefix}. From {time_ago}: {text}")
            else:
                lines.append(f"{auto_label_prefix}. {text}")
            label_established = True
            continue

        if time_ago:
            lines.append(f"From {time_ago}: {text}")
        else:
            lines.append(text)

    return lines


def _save_tone_only_fallback(item: dict, tone_duration: float | None) -> None:
    """Record the cached tone WAV as the item's audio_path, if applicable.

    Used after a giving-up TTS failure on an item whose text starts with
    ``$Note`` tone tokens. The leading tone played before TTS failed --
    saving its cached WAV here lets the history Play button replay the
    audible portion the listener actually heard. No-op for body-only
    items (no tone to fall back to).
    """
    try:
        tokens, _body, _trailing = extract_tone_tokens(item.get("text") or "")
        if not tokens:
            return
        cache_path = generate_combined_tones_from_tokens(
            tokens, duration=(tone_duration or 0.8),
        )
        if cache_path and Path(cache_path).exists():
            update_audio_path(item["id"], Path(cache_path))
    except Exception:
        # Best-effort -- a failure here just leaves audio_path NULL.
        pass


def update_audio_path(item_id: int, audio_path: Path) -> None:
    """Update the audio_path for a queue item."""
    with get_connection() as conn:
        conn.execute(
            "UPDATE queue SET audio_path = ? WHERE id = ?",
            (str(audio_path), item_id)
        )
        conn.commit()


def get_audio_save_path(item_id: int) -> Path:
    """Get the path where audio for this item should be saved."""
    from datetime import datetime
    today = datetime.now().strftime("%Y-%m-%d")
    day_dir = ensure_dir(_audio_dir() / today)
    return day_dir / f"{item_id}.wav"


def process_queue(verbose: bool = False) -> int:
    """Process all pending messages. Returns count of messages played."""
    sessions = get_sessions_with_pending()
    if not sessions:
        return 0

    # Count total messages
    total_messages = sum(len(get_pending_for_session(s)) for s in sessions)
    is_single_message = total_messages == 1

    total_played = 0

    # Get global settings for intro
    global_settings = get_settings()

    # Intro sound if enabled and not single message. No explicit sleep
    # between the chord and "This is Claude Code." -- the natural envelope
    # decay of the tone provides enough separation, and the previous 200ms
    # padding was perceptible dead air.
    #
    # When a per-queue intro rule exists for the FIRST session, use that
    # tune instead of the global default. Multi-queue batches still use
    # the first session's intro -- there's only one intro signal per
    # batch.
    if should_announce_intro() and not is_single_message and global_settings["intro_sound"]:
        if verbose:
            print("[INFO] Playing intro sound", file=sys.stderr)
        intro_queue = sessions[0] if sessions else None
        play_audio(get_intro_sound(queue=intro_queue), verbose)
        # The "This is Claude Code." greeting is a nicety, not a guarantee.
        # Swallow TTSError so a Polly blip on the greeting doesn't abort
        # the whole batch -- the intro tone already announced the batch.
        try:
            speak_text("This is Claude Code.", verbose=verbose, engine=global_settings["engine"])
        except TTSError:
            pass
        time.sleep(PAUSE_BETWEEN_SESSIONS)

    # Auto-label state: seeded from the DB, then updated in memory between
    # sessions in this batch. Reading from the DB only once means a batch with
    # multiple back-to-back queues still re-labels on each context switch
    # without each session re-reading playback_state.
    last_utterance_at = get_last_utterance_time()
    last_played_queue = get_last_played_queue()

    # Process each session
    is_only_session = len(sessions) == 1
    for session_idx, session_id in enumerate(sessions):
        items = get_pending_for_session(session_id)
        if not items:
            continue

        # Items whose TTS raised under the retry cap. Populated by the
        # speak_text TTSError handler below; consumed by the mark_played
        # loop at session end so retried rows stay played_at=NULL and
        # get_pending_for_session picks them up on the next poll.
        pending_retry_ids: set[int] = set()

        # Get settings for this session
        settings = get_settings(session_id)
        speed = settings["speed"]

        if session_idx > 0:
            time.sleep(PAUSE_BETWEEN_SESSIONS)

        # If any item in this session supplied a metadata.display_name, use
        # the first one as the spoken title -- callers know better than our
        # hyphen-to-space heuristic for queue ids like "e2e-stt-1779972451"
        # that would otherwise speak as a string of digits.
        display_name_override: str | None = None
        for it in items:
            meta = it.get("metadata") or {}
            candidate = meta.get("display_name")
            if isinstance(candidate, str) and candidate.strip():
                display_name_override = candidate.strip()
                break

        auto_label = compute_auto_label_prefix(
            session_id, last_utterance_at, last_played_queue,
            display_name_override=display_name_override,
        )
        script_lines = build_session_script(
            session_id, items, is_only_session, auto_label_prefix=auto_label
        )

        # Each script line now corresponds 1:1 to an item -- the old header
        # ("For queue X, there are N messages.") was removed, so there is no
        # offset between line_idx and item_idx anymore.
        for line_idx, line in enumerate(script_lines):
            if line_idx > 0:
                time.sleep(PAUSE_BETWEEN_MESSAGES)

            item_idx = line_idx
            save_path = None
            line_engine = settings["engine"]
            line_voice = settings["voice"]
            line_polly_engine = None
            line_is_ssml = False
            line_interpretation = None
            # Per-item effects preset override. Lets the UI's "Try sample"
            # button preview a preset without mutating the saved config --
            # see /api/effects/try in web.py. None falls back to the
            # saved value via apply_effects.
            line_effects_preset: str | None = None
            # Per-item tone duration override (seconds). Used by the
            # "Play intro/outro" Try buttons to preview tones at the
            # *configured* duration rather than the default 0.8s used for
            # the $Note prefix tones that precede TTS messages.
            line_tone_duration: float | None = None
            if 0 <= item_idx < len(items):
                item = items[item_idx]
                save_path = get_audio_save_path(item["id"])
                meta = item.get("metadata") or {}
                line_engine = meta.get("engine") or settings["engine"]
                line_voice = meta.get("voice") or settings["voice"]
                line_polly_engine = meta.get("polly_engine")
                line_interpretation = meta.get("interpretation")
                # Effects preset cascade (highest priority first):
                #   per-item metadata override (e.g. /api/effects/try)
                #   per-queue setting (settings.effects_preset)
                #   None -> generate_tts falls back to global config
                line_effects_preset = (
                    meta.get("effects_preset")
                    or settings.get("effects_preset")
                )
                td_raw = meta.get("tone_duration")
                # ``bool`` is a subclass of ``int`` in Python, so a stray
                # ``"tone_duration": true`` in metadata would otherwise
                # parse as 1.0 seconds per note. Exclude bools explicitly.
                if (
                    isinstance(td_raw, (int, float))
                    and not isinstance(td_raw, bool)
                    and td_raw > 0
                ):
                    line_tone_duration = float(td_raw)
                line_is_ssml = bool(meta.get("ssml")) or looks_like_ssml(item["text"])
                if line_is_ssml:
                    # SSML must be spoken verbatim: a spoken prefix like "First: "
                    # would sit outside the <speak> root and corrupt the markup.
                    line = item["text"]

            # An interpretation cue plays before the item's speech, then a
            # short pause, then the utterance.
            #
            # When an interpretation is present, strip any leading $Note tone
            # from the line so the interpretation cue is the *only* tone
            # cluster heard. Without this strip, a SUCCESS item that ALSO
            # received an auto-label prefix ("$Eb4 styling test. ...") would
            # play the SUCCESS chord, then the $Eb4 lead tone, then speak --
            # two tone clusters back-to-back for one outcome. The spoken
            # project name still happens (it's text, not a tone token), so
            # project context is preserved.
            if line_interpretation:
                if not line_is_ssml:
                    _stripped_tones, line, _trailing_drop = extract_tone_tokens(line)
                play_interpretation_cue(line_interpretation, verbose, queue=session_id)

            # Tell the web UI which item is being spoken right now so its
            # card can be highlighted. Always cleared in a try/finally so a
            # speak_text exception doesn't leave a stale highlight.
            current_item_id = (
                items[item_idx]["id"] if 0 <= item_idx < len(items) else None
            )
            if current_item_id is not None:
                set_currently_playing(current_item_id)
            tts_error: TTSError | None = None
            try:
                result = speak_text(
                    line, voice=line_voice, speed=speed, save_path=save_path, verbose=verbose,
                    engine=line_engine, is_ssml=line_is_ssml, polly_engine=line_polly_engine,
                    effects_preset=line_effects_preset,
                    tone_duration=line_tone_duration,
                )
            except TTSError as e:
                # Caught here so other items in the batch still process.
                # Already logged in generate_tts; we just track that this
                # one failed for the retry-or-surface decision below.
                tts_error = e
                result = None
            finally:
                if current_item_id is not None:
                    clear_currently_playing()

            if tts_error is not None and 0 <= item_idx < len(items):
                # Failure path: increment attempt counter in metadata. If
                # under the cap, leave the item PENDING (don't append to
                # played_ids) so the next daemon poll retries it. If at
                # the cap, save the tone WAV (if any played before the
                # failure) so the audible portion replays from history,
                # record the error message, and let the outer mark_played
                # loop finalize it.
                fail_item = items[item_idx]
                max_attempts = max(
                    1, int(get_player_config().get("tts_max_attempts", 3) or 3)
                )
                prior_meta = fail_item.get("metadata") or {}
                attempts = int(prior_meta.get("tts_attempts", 0) or 0) + 1
                if attempts < max_attempts:
                    update_metadata(fail_item["id"], {"tts_attempts": attempts})
                    pending_retry_ids.add(fail_item["id"])
                else:
                    # Give up. Record the error + save tone WAV for replay.
                    update_metadata(fail_item["id"], {
                        "tts_attempts": attempts,
                        "tts_error": str(tts_error),
                    })
                    _save_tone_only_fallback(
                        fail_item, line_tone_duration,
                    )
                continue  # Skip the success bookkeeping below.

            if result is not None or save_path is None:
                total_played += 1
                if save_path and 0 <= item_idx < len(items):
                    # Only record the audio_path when something was actually
                    # written there. Tone-only items (text is just $Note
                    # tokens) reach this branch with save_path set but the
                    # WAV missing -- the audible output came from the
                    # shared tone cache, not the per-item path. For those,
                    # record the cache WAV path so /audio/<id> can serve
                    # something, and the UI's Play button is honestly
                    # enabled.
                    actual = save_path if save_path.exists() else None
                    if actual is None:
                        # Tone-only fallback: resolve the cache WAV for the
                        # leading $Note tokens at the per-item duration.
                        try:
                            text = items[item_idx]["text"]
                            tone_tokens, clean, _trail = extract_tone_tokens(text)
                            if tone_tokens and not clean:
                                cache_path = generate_combined_tones_from_tokens(
                                    tone_tokens,
                                    duration=(line_tone_duration or 0.8),
                                )
                                if cache_path and Path(cache_path).exists():
                                    actual = Path(cache_path)
                        except Exception:
                            actual = None
                    if actual is not None:
                        update_audio_path(items[item_idx]["id"], actual)

        # Mark items as played -- except those left pending for retry.
        # ``pending_retry_ids`` was populated above when speak_text raised
        # TTSError under the attempt cap; those rows stay played_at=NULL
        # so get_pending_for_session picks them up on the next poll.
        for item in items:
            if item["id"] in pending_retry_ids:
                continue
            mark_played(item["id"])

        # Advance the in-memory auto-label state so the next session in this
        # batch sees a fresh "last spoken" pair (used for context-switch
        # detection on the next iteration).
        last_played_queue = session_id
        last_utterance_at = datetime.now(timezone.utc)

    # Outro (skip if single message): tone only -- the descending chord is the
    # end-of-batch signal. No spoken filler ("That is all.") -- it added
    # nothing the tone didn't already convey and grated when batches were
    # frequent.
    if total_played > 0 and not is_single_message and global_settings["intro_sound"]:
        time.sleep(PAUSE_BETWEEN_SESSIONS)
        # Use the LAST played session for outro context so a multi-queue
        # batch ends with that queue's outro tune (if defined).
        play_audio(get_outro_sound(queue=last_played_queue), verbose)

    if total_played > 0:
        # Persist both the time and the last queue so the *next* batch can
        # decide whether to auto-label based on its own gap+context-switch
        # signals.
        set_last_utterance_time(queue_id=last_played_queue)

    return total_played


def acquire_lock() -> Path | None:
    """Try to acquire a lock file. Returns lock path if acquired, None if already running."""
    lock = player_lock_path()
    ensure_dir(lock.parent)
    lock_path = lock

    # Check if lock exists and if process is still running
    if lock_path.exists():
        try:
            pid = int(lock_path.read_text().strip())
            # Check if process is still alive
            os.kill(pid, 0)
            return None  # Process is still running
        except (ValueError, OSError, ProcessLookupError):
            # Lock file is stale, remove it
            lock_path.unlink(missing_ok=True)

    # Create lock file with our PID
    lock_path.write_text(str(os.getpid()))
    return lock_path


def release_lock(lock_path: Path) -> None:
    """Release the lock file."""
    try:
        lock_path.unlink(missing_ok=True)
    except OSError:
        pass


def run_daemon(verbose: bool = False) -> None:
    """Run as a daemon - watch queue and process items immediately."""
    # Local import (despite the module-level one) so tests can patch
    # ``speeker.config.get_player_config`` and have run_daemon see the
    # mock. Without this, the module-level binding shadows the patch.
    from .config import get_player_config  # noqa: F811
    from .paths import restart_sentinel_path

    lock_path = acquire_lock()
    if lock_path is None:
        print("[ERROR] Another speeker-player daemon is already running", file=sys.stderr)
        sys.exit(1)

    # Clear the restart-needed sentinel: a fresh daemon has by definition
    # picked up the latest config, so any prior "restart needed" flag is
    # satisfied. The web UI's pill disappears the next time it polls.
    try:
        restart_sentinel_path().unlink(missing_ok=True)
    except OSError:
        pass

    if verbose:
        print("[INFO] Speeker player daemon starting...", file=sys.stderr)

    idle_timeout = get_player_config().get("model_idle_timeout_minutes", 0)

    # Pre-warm the active engine unless configured to lazy-load
    if idle_timeout == 0:
        default_engine = get_settings()["engine"]
        if verbose:
            print(f"[INFO] Warming up {default_engine} engine...", file=sys.stderr)
        get_engine(default_engine).warm()
        if verbose:
            print("[INFO] TTS engine ready!", file=sys.stderr)
    elif verbose:
        print(f"[INFO] Model idle timeout: {idle_timeout} min (lazy-load)", file=sys.stderr)

    last_activity = time.time()
    model_loaded = idle_timeout == 0
    paused_for_call = False

    try:
        while True:
            # Hold the queue while a call is active (mic in use). Pending items
            # are left untouched and flush automatically when the call ends.
            from .calls import should_pause_for_call
            if should_pause_for_call():
                if not paused_for_call:
                    paused_for_call = True
                    print("[INFO] Call active -- pausing queue", file=sys.stderr, flush=True)
                time.sleep(POLL_INTERVAL)
                continue
            if paused_for_call:
                paused_for_call = False
                print("[INFO] Call ended -- resuming queue", file=sys.stderr, flush=True)

            pending = get_pending_count()

            if pending > 0:
                if verbose:
                    print(f"[INFO] Processing {pending} pending item(s)", file=sys.stderr)
                process_queue(verbose)
                last_activity = time.time()
                model_loaded = True
            elif model_loaded and idle_timeout > 0:
                if time.time() - last_activity > idle_timeout * 60:
                    unload_tts_model()
                    model_loaded = False
                    if verbose:
                        print("[INFO] TTS model unloaded (idle)", file=sys.stderr)

            time.sleep(POLL_INTERVAL)
    finally:
        release_lock(lock_path)


def run_once(verbose: bool = False) -> None:
    """Run once - process queue and exit."""
    if verbose:
        print("[INFO] Speeker player (one-shot mode)", file=sys.stderr)

    played = process_queue(verbose)

    if verbose:
        print(f"[INFO] Done. Played {played} utterance(s)", file=sys.stderr)


def cleanup_old_files(days: int, verbose: bool = False) -> int:
    """Remove old database entries."""
    removed = cleanup_old_entries(days)
    if verbose:
        print(f"[CLEANUP] Removed {removed} database entries", file=sys.stderr)
    return removed


def main() -> int:
    """Main entry point."""
    import argparse
    from .migrate import migrate
    migrate()

    parser = argparse.ArgumentParser(
        prog="speeker-player",
        description="Speeker TTS playback daemon",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--daemon", "-d", action="store_true",
                        help="Run as daemon (watch queue, keep model warm)")
    parser.add_argument("--cleanup", type=int, metavar="DAYS",
                        help="Remove entries older than DAYS days and exit")

    args = parser.parse_args()

    if args.cleanup is not None:
        removed = cleanup_old_files(args.cleanup, args.verbose)
        print(f"Removed {removed} item(s)", file=sys.stderr)
        return 0

    if args.daemon:
        run_daemon(args.verbose)
    else:
        run_once(args.verbose)

    return 0


if __name__ == "__main__":
    sys.exit(main())
