"""Voice cloning -- download audio, extract reference clips, save for TTS."""

import json
import logging
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .paths import voices_dir as _voices_dir, voices_manifest as _voices_manifest

# pocket-tts needs 24kHz mono WAV for voice cloning input.
TARGET_SAMPLE_RATE = 24000


def _get_voice_dir(name: str) -> Path:
    """Get the directory for a specific voice."""
    return _voices_dir() / _safe_filename(name)


def _get_voice_log(name: str) -> logging.Logger:
    """Get a logger that writes to the voice's .log file."""
    voice_dir = _get_voice_dir(name)
    voice_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"voice_clone.{_safe_filename(name)}")
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers on repeated calls
    log_path = voice_dir / "clone.log"
    if not any(
        isinstance(h, logging.FileHandler) and h.baseFilename == str(log_path)
        for h in logger.handlers
    ):
        handler = logging.FileHandler(log_path)
        handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(handler)

    return logger


def _load_manifest() -> dict[str, dict]:
    """Load the voices manifest, returning empty dict if missing."""
    if _voices_manifest().exists():
        try:
            return json.loads(_voices_manifest().read_text())
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def _save_manifest(manifest: dict[str, dict]) -> None:
    """Save the voices manifest to disk."""
    _voices_dir().mkdir(parents=True, exist_ok=True)
    _voices_manifest().write_text(json.dumps(manifest, indent=2))


def download_audio(url: str, output_dir: Path, log: logging.Logger | None = None) -> Path:
    """Download audio from a URL. Uses yt-dlp for YouTube, curl for direct URLs.

    Returns path to the downloaded file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if _is_youtube_url(url):
        return _download_youtube(url, output_dir, log)
    return _download_direct(url, output_dir, log)


def _is_youtube_url(url: str) -> bool:
    """Check if URL is a YouTube link."""
    return any(domain in url for domain in [
        "youtube.com", "youtu.be", "youtube-nocookie.com",
    ])


def _download_youtube(url: str, output_dir: Path, log: logging.Logger | None = None) -> Path:
    """Download audio from YouTube using yt-dlp."""
    if not shutil.which("yt-dlp"):
        raise RuntimeError("yt-dlp is required for YouTube downloads. Install with: uv tool install yt-dlp")

    output_template = str(output_dir / "%(title).50s_%(id)s.%(ext)s")
    cmd = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "--output", output_template,
        "--no-playlist",
        url,
    ]
    if log:
        log.info("yt-dlp download: %s", url)
        log.info("yt-dlp cmd: %s", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if log:
        if result.stdout.strip():
            log.info("yt-dlp stdout: %s", result.stdout.strip())
        if result.stderr.strip():
            log.info("yt-dlp stderr: %s", result.stderr.strip())

    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {result.stderr.strip()}")

    # Find the downloaded file (yt-dlp may have converted it)
    wav_files = sorted(output_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not wav_files:
        raise RuntimeError("yt-dlp produced no WAV output")

    downloaded = wav_files[0]
    if log:
        log.info("Downloaded: %s (%d bytes)", downloaded.name, downloaded.stat().st_size)
    return downloaded


def _download_direct(url: str, output_dir: Path, log: logging.Logger | None = None) -> Path:
    """Download a file directly via curl."""
    if not shutil.which("curl"):
        raise RuntimeError("curl is required for direct URL downloads")

    from urllib.parse import urlparse
    parsed = urlparse(url)
    filename = Path(parsed.path).name or "download"
    output_path = output_dir / filename

    if log:
        log.info("curl download: %s -> %s", url, output_path.name)

    result = subprocess.run(
        ["curl", "-fsSL", "-o", str(output_path), url],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(f"curl failed: {result.stderr.strip()}")

    if not output_path.exists():
        raise RuntimeError(f"Download failed -- file not created: {output_path}")

    if log:
        log.info("Downloaded: %s (%d bytes)", output_path.name, output_path.stat().st_size)
    return output_path


def extract_audio(
    input_path: Path,
    output_path: Path,
    log: logging.Logger | None = None,
) -> Path:
    """Extract audio from a video/audio file, converting to 24kHz mono WAV.

    Works for both video (strips video track) and audio (resamples/remixes).
    """
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required for audio extraction")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-vn",  # strip video
        "-ac", "1",  # mono
        "-ar", str(TARGET_SAMPLE_RATE),
        "-sample_fmt", "s16",
        "-f", "wav",
        str(output_path),
    ]
    if log:
        log.info("Extract audio: %s -> %s", input_path.name, output_path.name)

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    if log and result.stderr.strip():
        # ffmpeg writes progress to stderr even on success; just log it
        for line in result.stderr.strip().splitlines()[-3:]:
            log.info("  ffmpeg: %s", line.strip())

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg extraction failed: {result.stderr.strip()}")

    if log:
        log.info("Extracted: %s (%d bytes)", output_path.name, output_path.stat().st_size)
    return output_path


def trim_audio(
    audio_path: Path,
    output_path: Path,
    start_secs: float = 0,
    duration_secs: float = 30,
    log: logging.Logger | None = None,
) -> Path:
    """Trim audio to a specific segment using ffmpeg."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required for audio trimming")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_secs),
        "-i", str(audio_path),
        "-t", str(duration_secs),
        "-c", "copy",
        str(output_path),
    ]
    if log:
        log.info("Trim: %s [%gs + %gs] -> %s", audio_path.name, start_secs, duration_secs, output_path.name)

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg trim failed: {result.stderr.strip()}")

    if log:
        log.info("Trimmed: %s (%d bytes)", output_path.name, output_path.stat().st_size)
    return output_path


def clone_voice(
    name: str,
    sources: list[str],
    start_secs: float = 0,
    duration_secs: float = 30,
    description: str | None = None,
) -> Path:
    """Clone a voice from one or more audio/video sources.

    Each source can be a local file path or a URL (YouTube or direct).
    Downloads if needed, extracts audio, trims, and saves to the voice's
    directory under _voices_dir()/<name>/.

    All intermediate files (downloads, extractions, trims) are kept
    for later re-use and inspection.

    Returns the path to the saved reference WAV.
    """
    voice_dir = _get_voice_dir(name)
    voice_dir.mkdir(parents=True, exist_ok=True)

    log = _get_voice_log(name)
    log.info("=" * 60)
    log.info("Clone voice: %s", name)
    log.info("Sources: %s", sources)
    log.info("Start: %gs, Duration: %gs", start_secs, duration_secs)

    downloads_dir = voice_dir / "downloads"
    downloads_dir.mkdir(exist_ok=True)

    extracted_clips: list[Path] = []

    for i, source in enumerate(sources):
        source_path = Path(source).expanduser()

        # Download if URL
        if source.startswith(("http://", "https://")):
            log.info("Source %d: downloading %s", i, source)
            source_path = download_audio(source, downloads_dir, log)
        else:
            log.info("Source %d: local file %s", i, source)
            if not source_path.exists():
                raise FileNotFoundError(f"Source not found: {source}")
            # Copy local files into downloads dir for the audit trail
            dest = downloads_dir / source_path.name
            if dest != source_path:
                shutil.copy2(source_path, dest)
                log.info("Copied local file to: %s", dest.name)
            source_path = dest

        # Extract to 24kHz mono WAV
        extracted = voice_dir / f"extracted_{i}.wav"
        extract_audio(source_path, extracted, log)

        # Trim
        trimmed = voice_dir / f"trimmed_{i}.wav"
        trim_audio(extracted, trimmed, start_secs=start_secs, duration_secs=duration_secs, log=log)
        extracted_clips.append(trimmed)

    # Concatenate if multiple clips, otherwise just copy
    final_path = voice_dir / "reference.wav"
    if len(extracted_clips) == 1:
        shutil.copy2(extracted_clips[0], final_path)
        log.info("Single clip -> reference.wav (%d bytes)", final_path.stat().st_size)
    else:
        _concatenate_audio(extracted_clips, final_path)
        log.info("Concatenated %d clips -> reference.wav (%d bytes)", len(extracted_clips), final_path.stat().st_size)

    # Update manifest
    manifest = _load_manifest()
    manifest[name] = {
        "voice_dir": str(voice_dir),
        "audio_path": str(final_path),
        "description": description or f"Cloned from {len(sources)} source(s)",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sources": sources,
        "start_secs": start_secs,
        "duration_secs": duration_secs,
    }
    _save_manifest(manifest)
    log.info("Manifest updated. Voice ready: %s", name)

    return final_path


def _safe_filename(name: str) -> str:
    """Convert a voice name to a safe filename."""
    import re
    safe = re.sub(r"[^\w\s-]", "", name).strip().lower()
    safe = re.sub(r"[\s]+", "_", safe)
    return safe or "voice"


def _concatenate_audio(clips: list[Path], output_path: Path) -> Path:
    """Concatenate multiple audio files using ffmpeg."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required for audio concatenation")

    concat_list = output_path.parent / ".concat_list.txt"
    with open(concat_list, "w") as f:
        for clip in clips:
            f.write(f"file '{clip}'\n")

    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_list),
                "-c", "copy",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg concat failed: {result.stderr.strip()}")
    finally:
        concat_list.unlink(missing_ok=True)

    return output_path


def get_custom_voices() -> dict[str, dict]:
    """Get all custom voices from the manifest."""
    return _load_manifest()


def get_custom_voice_path(name: str) -> Path | None:
    """Look up a custom voice's audio path by name."""
    manifest = _load_manifest()
    entry = manifest.get(name)
    if entry:
        path = Path(entry["audio_path"])
        if path.exists():
            return path
    return None


def delete_custom_voice(name: str) -> bool:
    """Delete a custom voice from manifest and disk (removes entire voice directory)."""
    manifest = _load_manifest()
    if name not in manifest:
        return False

    # Remove the entire voice directory
    voice_dir = _get_voice_dir(name)
    if voice_dir.exists():
        shutil.rmtree(voice_dir)

    del manifest[name]
    _save_manifest(manifest)
    return True
