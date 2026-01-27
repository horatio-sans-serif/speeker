"""Tone synthesis using additive synthesis."""

import math
import struct
import wave
from pathlib import Path

# Note frequencies (A4 = 440 Hz standard tuning)
# Formula: freq = 440 * 2^((n-49)/12) where n is the key number
NOTE_SEMITONES = {
    "C": -9, "D": -7, "E": -5, "F": -4, "G": -2, "A": 0, "B": 2
}


def note_to_frequency(note: str) -> float | None:
    """Convert a note like 'Eb4' to frequency in Hz.

    Format: [A-G][b/#]?[0-8]
    Examples: C4=261.63, A4=440, Eb3=155.56, Eb4=311.13
    """
    import re
    match = re.match(r"^([A-Ga-g])([b#])?([0-8])$", note)
    if not match:
        return None

    note_name = match.group(1).upper()
    accidental = match.group(2)
    octave = int(match.group(3))

    semitones = NOTE_SEMITONES.get(note_name)
    if semitones is None:
        return None

    if accidental == "b":
        semitones -= 1
    elif accidental == "#":
        semitones += 1

    semitones += (octave - 4) * 12
    return 440.0 * (2.0 ** (semitones / 12.0))


def generate_tone(
    frequency: float,
    duration: float = 0.8,
    sample_rate: int = 44100,
    style: str = "vibraphone",
) -> list[float]:
    """Generate a single tone at the given frequency.

    Args:
        frequency: Tone frequency in Hz
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        style: Synthesis style ("vibraphone", "bell", "sine")

    Returns:
        List of float samples normalized to [-1, 1]
    """
    if style == "vibraphone":
        return _generate_vibraphone(frequency, duration, sample_rate)
    elif style == "bell":
        return _generate_bell(frequency, duration, sample_rate)
    else:
        return _generate_sine(frequency, duration, sample_rate)


def _generate_sine(freq: float, duration: float, sample_rate: int) -> list[float]:
    """Simple sine wave with fade in/out."""
    n_samples = int(sample_rate * duration)
    fade_samples = int(sample_rate * 0.01)
    samples = []

    for i in range(n_samples):
        t = i / sample_rate
        if i < fade_samples:
            envelope = i / fade_samples
        elif i > n_samples - fade_samples:
            envelope = (n_samples - i) / fade_samples
        else:
            envelope = 1.0
        samples.append(envelope * math.sin(2 * math.pi * freq * t))

    return samples


def _generate_bell(freq: float, duration: float, sample_rate: int) -> list[float]:
    """Bell-like tone with harmonics."""
    harmonics = [
        (1.0, 1.0, 1.0),
        (2.0, 0.5, 1.5),
        (3.0, 0.25, 2.0),
        (4.0, 0.15, 2.5),
        (5.0, 0.1, 3.0),
    ]

    n_samples = int(sample_rate * duration)
    samples = []

    for i in range(n_samples):
        t = i / sample_rate
        progress = i / n_samples
        envelope = math.exp(-progress * 4) * (1 - math.exp(-progress * 50))

        value = 0.0
        for mult, amp, decay in harmonics:
            harmonic_env = math.exp(-progress * decay * 3)
            value += amp * harmonic_env * math.sin(2 * math.pi * freq * mult * t)

        samples.append(0.4 * envelope * value)

    return samples


def _generate_vibraphone(freq: float, duration: float, sample_rate: int) -> list[float]:
    """Vibraphone-style tone with tremolo and harmonics."""
    # Vibraphone harmonic structure
    harmonics = [
        (1.0, 1.0, 0.4),     # Strong fundamental
        (4.0, 0.25, 0.6),    # Strong 4th harmonic (vibraphone characteristic)
        (10.0, 0.08, 1.0),   # Shimmer
        (3.0, 0.12, 0.8),    # Some 3rd
    ]

    vibrato_rate = 5.5  # Hz
    vibrato_depth = 0.12

    n_samples = int(sample_rate * duration)
    samples = []

    for i in range(n_samples):
        t = i / sample_rate
        progress = i / n_samples

        # Envelope: soft attack, sustain, release
        attack_time = 0.008
        release_start = 0.7

        if t < attack_time:
            envelope = t / attack_time
        elif progress < release_start:
            envelope = math.exp(-progress * 0.5)
        else:
            release_progress = (progress - release_start) / (1 - release_start)
            envelope = math.exp(-progress * 0.5) * (1 - release_progress ** 2)

        # Tremolo
        vibrato_buildup = min(1.0, t / 0.1)
        tremolo = 1.0 + vibrato_depth * vibrato_buildup * math.sin(2 * math.pi * vibrato_rate * t)

        # Additive synthesis
        value = 0.0
        for mult, amp, decay in harmonics:
            harmonic_env = math.exp(-progress * decay)
            value += amp * harmonic_env * math.sin(2 * math.pi * freq * mult * t)

        samples.append(0.45 * envelope * tremolo * value)

    return samples


def apply_reverb(
    samples: list[float],
    sample_rate: int = 44100,
    decay: float = 0.3,
) -> list[float]:
    """Apply multi-tap delay reverb."""
    # Delay taps in samples
    delays = [
        int(sample_rate * 0.030),
        int(sample_rate * 0.060),
        int(sample_rate * 0.100),
        int(sample_rate * 0.150),
        int(sample_rate * 0.200),
    ]

    max_delay = max(delays)
    output = samples + [0.0] * max_delay

    for i, s in enumerate(samples):
        for d_idx, delay in enumerate(delays):
            tap_amp = decay * (0.55 ** d_idx)
            if i + delay < len(output):
                output[i + delay] += s * tap_amp

    return output


def generate_tones(
    frequencies: list[float],
    duration: float = 0.8,
    gap: float = 0.06,
    sample_rate: int = 44100,
    style: str = "vibraphone",
    reverb: bool = True,
) -> list[float]:
    """Generate multiple tones in sequence.

    Args:
        frequencies: List of frequencies in Hz
        duration: Duration per tone in seconds
        gap: Gap between tones in seconds
        sample_rate: Sample rate in Hz
        style: Synthesis style
        reverb: Apply reverb

    Returns:
        List of float samples normalized to [-1, 1]
    """
    gap_samples = int(sample_rate * gap)
    all_samples = []

    for i, freq in enumerate(frequencies):
        tone = generate_tone(freq, duration, sample_rate, style)
        if reverb:
            tone = apply_reverb(tone, sample_rate)

        # Soft clip
        for s in tone:
            all_samples.append(math.tanh(s))

        if i < len(frequencies) - 1:
            all_samples.extend([0.0] * gap_samples)

    return all_samples


def samples_to_wav(samples: list[float], path: Path, sample_rate: int = 44100) -> None:
    """Write samples to a WAV file."""
    int_samples = [int(s * 32767) for s in samples]

    with wave.open(str(path), "w") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(struct.pack(f"{len(int_samples)}h", *int_samples))


def samples_to_bytes(samples: list[float]) -> bytes:
    """Convert samples to raw PCM bytes."""
    int_samples = [int(s * 32767) for s in samples]
    return struct.pack(f"{len(int_samples)}h", *int_samples)
