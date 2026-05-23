# Amazon Polly + SSML Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Amazon Polly as a third TTS engine, add SSML support (native for Polly, best-effort emulation for local engines), and add an SSML generator that turns plain text into purpose-tuned SSML.

**Architecture:** Introduce a shared `Engine` abstraction (`engines.py`) so both the CLI and the player daemon dispatch by the stored `engine` setting instead of hardcoding pocket-tts. SSML primitives live in `ssml.py`; SSML generation in `ssml_generate.py` (hybrid LLM + rule-based, reusing `summarize.call_llm`). Polly uses boto3 with PCM output converted to float32.

**Tech Stack:** Python 3.11+, numpy, scipy, FastAPI, boto3 (optional), pytest + pytest-mock. Run tests with `uv run pytest`.

**Conventions (match existing tests):** Test files in `tests/`, class-grouped, `unittest.mock`. Isolate filesystem with `patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)})`. Never run real pocket-tts/kokoro/AWS in tests — mock them.

---

## Task 1: Config sections for Polly and SSML

**Files:**

- Modify: `src/speeker/config.py` (DEFAULT_CONFIG dict; add accessors at end)
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_config.py`:

```python
from speeker.config import get_polly_config, get_ssml_config


class TestPollyConfig:
    def test_defaults(self, tmp_path):
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            cfg = get_polly_config()
            assert cfg["region"] is None
            assert cfg["profile"] is None
            assert cfg["engine"] == "neural"
            assert cfg["voice"] == "Joanna"

    def test_merge_partial(self, tmp_path):
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            save_config({"polly": {"voice": "Matthew"}})
            cfg = get_polly_config()
            assert cfg["voice"] == "Matthew"
            assert cfg["engine"] == "neural"  # default preserved by merge


class TestSsmlConfig:
    def test_defaults(self, tmp_path):
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            cfg = get_ssml_config()
            assert cfg["emulate_for_local"] is False
            assert cfg["acronyms_file"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_config.py -k "Polly or Ssml" -v`
Expected: FAIL with `ImportError: cannot import name 'get_polly_config'`

- [ ] **Step 3: Implement config additions**

In `src/speeker/config.py`, add these two sections inside `DEFAULT_CONFIG` (after the `"player"` section):

```python
    "polly": {
        "region": None,    # None = boto3 default (profile/env region)
        "profile": None,   # AWS profile name; None = default credential chain
        "engine": "neural",  # default Polly engine variant
        "voice": "Joanna",   # default Polly VoiceId
    },
    "ssml": {
        "emulate_for_local": False,  # CLI --best-effort-ssml-emulation overrides
        "acronyms_file": None,       # path to a file of extra spell-out acronyms
    },
```

Add these accessors at the end of the file:

```python
def get_polly_config() -> dict:
    """Get Amazon Polly configuration."""
    config = get_config()
    return config.get("polly", {})


def get_ssml_config() -> dict:
    """Get SSML configuration."""
    config = get_config()
    return config.get("ssml", {})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_config.py -k "Polly or Ssml" -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/speeker/config.py tests/test_config.py
git commit -m "Add polly and ssml config sections with accessors"
```

---

## Task 2: SSML detection, stripping, and sanitization

**Files:**

- Create: `src/speeker/ssml.py`
- Test: `tests/test_ssml.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ssml.py`:

```python
#!/usr/bin/env python3
"""Unit tests for ssml.py."""

from speeker.ssml import (
    looks_like_ssml,
    ensure_speak_wrapped,
    strip_ssml,
    sanitize_ssml,
    escape_text,
    POLLY_SAFE_TAGS,
)


class TestLooksLikeSsml:
    def test_speak_wrapper(self):
        assert looks_like_ssml("<speak>hi</speak>") is True

    def test_leading_whitespace(self):
        assert looks_like_ssml("   <speak>hi</speak>") is True

    def test_case_insensitive(self):
        assert looks_like_ssml("<SPEAK>hi</SPEAK>") is True

    def test_plain_text(self):
        assert looks_like_ssml("hello world") is False


class TestEnsureSpeakWrapped:
    def test_wraps_plain(self):
        assert ensure_speak_wrapped("hi") == "<speak>hi</speak>"

    def test_leaves_wrapped(self):
        assert ensure_speak_wrapped("<speak>hi</speak>") == "<speak>hi</speak>"


class TestStripSsml:
    def test_removes_tags_keeps_text(self):
        assert strip_ssml("<speak>Hello <break/>world</speak>") == "Hello world"

    def test_unescapes_entities(self):
        assert strip_ssml("<speak>a &amp; b</speak>") == "a & b"


class TestEscapeText:
    def test_escapes_bare_ampersand(self):
        assert escape_text("a & b") == "a &amp; b"

    def test_preserves_entities(self):
        assert escape_text("a &amp; b") == "a &amp; b"

    def test_escapes_angle_brackets(self):
        assert escape_text("1 < 2 > 0") == "1 &lt; 2 &gt; 0"


class TestSanitizeSsml:
    def test_keeps_allowed_tags(self):
        out = sanitize_ssml('<speak>Hi <break time="500ms"/>there</speak>')
        assert '<break time="500ms"/>' in out
        assert out.startswith("<speak>") and out.endswith("</speak>")

    def test_drops_disallowed_tags_keeps_text(self):
        out = sanitize_ssml("<speak>Hello <script>x</script>world</speak>")
        assert "<script>" not in out
        assert "Hello" in out and "world" in out and "x" in out

    def test_wraps_unwrapped_input(self):
        out = sanitize_ssml("just text")
        assert out == "<speak>just text</speak>"

    def test_single_speak_root_when_nested(self):
        out = sanitize_ssml("<speak><speak>hi</speak></speak>")
        assert out.count("<speak>") == 1
        assert out.count("</speak>") == 1

    def test_escapes_stray_ampersand_in_text(self):
        out = sanitize_ssml("<speak>Tom & Jerry</speak>")
        assert "Tom &amp; Jerry" in out

    def test_survives_malformed_input(self):
        out = sanitize_ssml("<speak>a <b roken tag")
        assert out.startswith("<speak>") and out.endswith("</speak>")

    def test_polly_safe_tags_present(self):
        assert "prosody" in POLLY_SAFE_TAGS
        assert "say-as" in POLLY_SAFE_TAGS
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_ssml.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'speeker.ssml'`

- [ ] **Step 3: Implement `ssml.py` (detection, tokenizer, strip, sanitize)**

Create `src/speeker/ssml.py`:

```python
"""SSML primitives: detection, stripping, emulation, and sanitization.

These helpers are deliberately built on a tolerant regex tokenizer rather than a
strict XML parser, so they degrade gracefully on malformed input (e.g. LLM
output). Amazon Polly remains the authority on fine attribute validity.
"""

import html
import re

# Tags Amazon Polly accepts. Anything else is dropped (its text content kept).
POLLY_SAFE_TAGS = {
    "speak", "break", "emphasis", "lang", "mark", "p", "s",
    "phoneme", "prosody", "say-as", "sub", "w",
    "amazon:domain", "amazon:effect", "amazon:breath", "amazon:auto-breaths",
}

# Matches a single tag: opening, closing, or self-closing, with attributes.
# Groups: 1=leading slash (closing), 2=name, 3=attributes, 4=trailing slash.
_TAG_RE = re.compile(
    r"<\s*(/?)\s*([a-zA-Z][\w:-]*)"
    r"((?:\s+[\w:-]+(?:\s*=\s*(?:\"[^\"]*\"|'[^']*'|[^\s>]+))?)*)"
    r"\s*(/?)\s*>"
)

# An & that begins a valid XML/HTML entity (so we don't double-escape it).
_ENTITY_RE = re.compile(r"&(?:#\d+|#x[0-9a-fA-F]+|[a-zA-Z][a-zA-Z0-9]*);")


def looks_like_ssml(text: str) -> bool:
    """True if *text* appears to be SSML (starts with a <speak> tag)."""
    return text.lstrip()[:6].lower().startswith("<speak")


def ensure_speak_wrapped(text: str) -> str:
    """Wrap *text* in a single <speak> element if it is not already."""
    stripped = text.strip()
    if stripped[:6].lower().startswith("<speak"):
        return stripped
    return f"<speak>{stripped}</speak>"


def escape_text(s: str) -> str:
    """XML-escape a text node: bare & -> &amp;, and < > always escaped.

    An & that already starts a valid entity (&amp; &#10; etc.) is preserved.
    """
    s = _ENTITY_RE.sub(lambda m: "\x00" + m.group(0)[1:], s)  # protect entities
    s = s.replace("&", "&amp;")
    s = s.replace("\x00", "&")  # restore protected entities
    s = s.replace("<", "&lt;").replace(">", "&gt;")
    return s


def strip_ssml(text: str) -> str:
    """Remove all tags, keep text content, unescape entities, collapse spaces."""
    no_tags = _TAG_RE.sub(" ", text)
    unescaped = html.unescape(no_tags)
    return re.sub(r"\s+", " ", unescaped).strip()


def _unwrap_speak(s: str) -> str:
    """Remove every <speak>/</speak> tag so we can re-wrap with a single root."""
    return re.sub(r"<\s*/?\s*speak\b[^>]*>", "", s, flags=re.IGNORECASE).strip()


def sanitize_ssml(text: str, allowed_tags: set[str] = POLLY_SAFE_TAGS) -> str:
    """Return valid SSML containing only whitelisted tags, with one <speak> root.

    Disallowed tags are removed but their text content is kept. Text nodes are
    XML-escaped. Robust to malformed input (unmatched/garbled tags become text).
    """
    out: list[str] = []
    pos = 0
    for m in _TAG_RE.finditer(text):
        out.append(escape_text(text[pos:m.start()]))
        name = m.group(2).lower()
        if name in allowed_tags:
            out.append(m.group(0))
        # else: drop the tag marker, surrounding text already handled
        pos = m.end()
    out.append(escape_text(text[pos:]))
    inner = _unwrap_speak("".join(out))
    return f"<speak>{inner}</speak>"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_ssml.py -v`
Expected: PASS (all tests in the file so far)

- [ ] **Step 5: Commit**

```bash
git add src/speeker/ssml.py tests/test_ssml.py
git commit -m "Add ssml detection, stripping, and sanitization"
```

---

## Task 3: SSML emulation and acronym loading

**Files:**

- Modify: `src/speeker/ssml.py` (append)
- Test: `tests/test_ssml.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ssml.py`:

```python
from speeker.ssml import emulate_ssml, load_acronyms, COMMON_ACRONYMS


class TestLoadAcronyms:
    def test_builtin_present(self):
        acr = load_acronyms()
        assert "PHI" in acr

    def test_file_all_separators(self, tmp_path):
        f = tmp_path / "acr.txt"
        f.write_text("EHR,EMR|HL7;FHIR ICD")
        acr = load_acronyms(str(f))
        for token in ("EHR", "EMR", "HL7", "FHIR", "ICD"):
            assert token in acr

    def test_missing_file_returns_builtin(self, tmp_path):
        acr = load_acronyms(str(tmp_path / "nope.txt"))
        assert "PHI" in acr


class TestEmulateSsml:
    def test_say_as_characters_spells_out(self):
        out = emulate_ssml('<say-as interpret-as="characters">PHI</say-as>')
        assert out == "P-H-I"

    def test_sub_uses_alias(self):
        out = emulate_ssml('<sub alias="World Wide Web">WWW</sub>')
        assert out == "World Wide Web"

    def test_break_becomes_punctuation(self):
        out = emulate_ssml('Hello<break time="500ms"/>world')
        assert "Hello." in out and "world" in out

    def test_known_acronym_spelled(self):
        out = emulate_ssml("Patient PHI today")
        assert "P-H-I" in out

    def test_unknown_caps_normalized(self):
        out = emulate_ssml("Please STOP now")
        assert "STOP" not in out
        assert "Stop" in out

    def test_other_tags_dropped_text_kept(self):
        out = emulate_ssml("<emphasis>really</emphasis> good")
        assert out == "really good"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_ssml.py -k "Acronym or Emulate" -v`
Expected: FAIL with `ImportError: cannot import name 'emulate_ssml'`

- [ ] **Step 3: Implement emulation and acronym loading**

Append to `src/speeker/ssml.py`:

```python
from pathlib import Path

# Small built-in set of acronyms that should be spelled out letter-by-letter.
# Users extend this via the ssml.acronyms_file config (see load_acronyms).
COMMON_ACRONYMS = {"PHI", "PII", "SSN", "DOB"}

_ACRONYM_SPLIT_RE = re.compile(r"[,\s|;]+")
_CAPS_WORD_RE = re.compile(r"\b[A-Z]{2,}\b")


def load_acronyms(config_path: str | None = None) -> set[str]:
    """Built-in acronyms plus any from *config_path* (split on , whitespace | ;)."""
    acronyms = set(COMMON_ACRONYMS)
    if config_path:
        try:
            content = Path(config_path).read_text(encoding="utf-8")
        except OSError:
            return acronyms
        for token in _ACRONYM_SPLIT_RE.split(content):
            token = token.strip()
            if token:
                acronyms.add(token.upper())
    return acronyms


def _spell_out(word: str) -> str:
    """'PHI' -> 'P-H-I' (alphanumerics only, upper-cased)."""
    letters = [c for c in word if c.isalnum()]
    return "-".join(letters).upper()


def _spell_segment(segment: str) -> str:
    """Spell out every whitespace-separated token in a segment."""
    return " ".join(_spell_out(w) for w in segment.split() if w)


def _break_to_punct(attrs: str) -> str:
    """Map a <break> tag's attributes to spoken punctuation."""
    m = re.search(r'time\s*=\s*"(\d+(?:\.\d+)?)(ms|s)"', attrs)
    if m:
        ms = float(m.group(1)) * (1000 if m.group(2) == "s" else 1)
        if ms >= 1000:
            return "... "
        if ms >= 400:
            return ". "
        return ", "
    m2 = re.search(r'strength\s*=\s*"([\w-]+)"', attrs)
    if m2:
        return {
            "none": " ", "x-weak": " ", "weak": ", ", "medium": ", ",
            "strong": ". ", "x-strong": "... ",
        }.get(m2.group(1), ", ")
    return ", "


def _normalize_caps(segment: str, acronyms: set[str]) -> str:
    """Spell known acronyms; title-case other ALL-CAPS runs so engines don't shout."""
    def repl(mm: "re.Match[str]") -> str:
        word = mm.group(0)
        if word.upper() in acronyms:
            return _spell_out(word)
        return word.capitalize()

    return _CAPS_WORD_RE.sub(repl, segment)


def emulate_ssml(text: str, acronyms: set[str] | None = None) -> str:
    """Best-effort conversion of SSML to plain text for engines lacking SSML.

    - <say-as interpret-as="characters"|"spell-out"> content is spelled out.
    - <sub alias="X"> emits X (its inner text is dropped).
    - <break> / prosody pauses become punctuation by duration/strength.
    - ALL-CAPS runs are spelled (if a known acronym) or title-cased.
    - Every other tag is dropped, keeping its text content.
    """
    if acronyms is None:
        acronyms = load_acronyms()

    out: list[str] = []
    pos = 0
    spell_depth = 0       # inside <say-as characters/spell-out>
    suppress_depth = 0    # inside <sub> (drop inner text)
    sub_alias_stack: list[str] = []

    def emit(segment: str) -> None:
        if suppress_depth > 0:
            return
        if spell_depth > 0:
            out.append(_spell_segment(segment))
        else:
            out.append(_normalize_caps(segment, acronyms))

    for m in _TAG_RE.finditer(text):
        emit(text[pos:m.start()])
        pos = m.end()
        closing = m.group(1) == "/"
        name = m.group(2).lower()
        attrs = m.group(3) or ""
        self_closing = m.group(4) == "/"

        if name == "break" and not closing:
            if suppress_depth == 0:
                out.append(_break_to_punct(attrs))
        elif name == "say-as":
            spell = bool(
                re.search(r'interpret-as\s*=\s*"(characters|spell-out)"', attrs)
            )
            if not closing and not self_closing and spell:
                spell_depth += 1
            elif closing and spell_depth > 0:
                spell_depth -= 1
        elif name == "sub":
            if not closing and not self_closing:
                alias = re.search(r'alias\s*=\s*"([^"]*)"', attrs)
                sub_alias_stack.append(alias.group(1) if alias else "")
                suppress_depth += 1
            elif closing:
                if suppress_depth > 0:
                    suppress_depth -= 1
                if sub_alias_stack:
                    emit_alias = sub_alias_stack.pop()
                    if suppress_depth == 0:
                        out.append(emit_alias)
        # other tags: ignore the marker; text content handled by emit()

    emit(text[pos:])
    result = html.unescape("".join(out))
    return re.sub(r"\s+", " ", result).strip()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_ssml.py -v`
Expected: PASS (all tests in the file)

- [ ] **Step 5: Commit**

```bash
git add src/speeker/ssml.py tests/test_ssml.py
git commit -m "Add best-effort SSML emulation and acronym loading"
```

---

## Task 4: Polly voices in voices.py

**Files:**

- Modify: `src/speeker/voices.py`
- Test: `tests/test_voices.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_voices.py`:

```python
from speeker.voices import (
    POLLY_VOICES,
    DEFAULT_POLLY_VOICE,
    DEFAULT_POLLY_ENGINE,
    POLLY_VARIANT_DEFAULT_VOICE,
    get_voices,
    get_default_voice,
    validate_voice,
)


class TestPollyVoices:
    def test_default_voice_constant(self):
        assert DEFAULT_POLLY_VOICE == "Joanna"
        assert DEFAULT_POLLY_VOICE in POLLY_VOICES

    def test_default_engine_variant(self):
        assert DEFAULT_POLLY_ENGINE == "neural"

    def test_variant_defaults(self):
        assert POLLY_VARIANT_DEFAULT_VOICE["long-form"] == "Danielle"
        assert POLLY_VARIANT_DEFAULT_VOICE["generative"] == "Ruth"

    def test_get_voices_includes_polly(self):
        voices = get_voices("polly")
        assert "polly" in voices
        assert "Joanna" in voices["polly"]

    def test_get_default_voice_polly(self):
        assert get_default_voice("polly") == "Joanna"

    def test_validate_voice_polly_lenient(self):
        assert validate_voice("polly", "AnyNewPollyVoice") is True
        assert validate_voice("polly", "") is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_voices.py -k Polly -v`
Expected: FAIL with `ImportError: cannot import name 'POLLY_VOICES'`

- [ ] **Step 3: Implement Polly voice support**

In `src/speeker/voices.py`, add after `KOKORO_VOICES`:

```python
POLLY_VOICES = {
    "Joanna": "US English, female (standard/neural)",
    "Matthew": "US English, male (standard/neural)",
    "Danielle": "US English, female (long-form)",
    "Gregory": "US English, male (long-form)",
    "Ruth": "US English, female (generative)",
    "Amy": "British English, female (neural)",
    "Brian": "British English, male (neural)",
}

# Default Polly voice per engine variant.
POLLY_VARIANT_DEFAULT_VOICE = {
    "standard": "Joanna",
    "neural": "Joanna",
    "long-form": "Danielle",
    "generative": "Ruth",
}
```

Add these constants near the other `DEFAULT_*` definitions:

```python
DEFAULT_POLLY_VOICE = "Joanna"
DEFAULT_POLLY_ENGINE = "neural"  # Polly engine variant
```

In `get_voices`, add this block before the custom-voices block:

```python
    if engine is None or engine == "polly":
        result["polly"] = POLLY_VOICES
```

In `get_default_voice`, add before the final `return`:

```python
    if engine == "polly":
        return DEFAULT_POLLY_VOICE
```

In `validate_voice`, add before the final `return False`:

```python
    if engine == "polly":
        # Polly's catalog is large and region-dependent; Polly is the authority.
        return isinstance(voice, str) and bool(voice.strip())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_voices.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/speeker/voices.py tests/test_voices.py
git commit -m "Add Polly voices, defaults, and lenient validation"
```

---

## Task 5: Engine abstraction with local engines

**Files:**

- Create: `src/speeker/engines.py`
- Test: `tests/test_engines.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_engines.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_engines.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'speeker.engines'`

- [ ] **Step 3: Implement `engines.py` (base, local engines, registry, prepare_payload)**

Create `src/speeker/engines.py`:

```python
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
        """Return (float32 audio in [-1, 1], sample_rate)."""
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


_ENGINES: dict[str, BaseEngine] = {}


def _create_engine(name: str) -> BaseEngine:
    if name == "pocket-tts":
        return PocketTTSEngine()
    if name == "kokoro":
        return KokoroEngine()
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_engines.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/speeker/engines.py tests/test_engines.py
git commit -m "Add engine abstraction with pocket-tts and kokoro engines"
```

---

## Task 6: Polly engine

**Files:**

- Modify: `src/speeker/engines.py`
- Test: `tests/test_engines.py` (append)

- [ ] **Step 1: Write the failing tests (boto3 mocked via sys.modules)**

Append to `tests/test_engines.py`:

```python
import io
import sys
from unittest.mock import MagicMock, patch


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
        eng.unload()  # no-op, must not raise

    def test_registry_creates_polly(self):
        assert get_engine("polly").name == "polly"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_engines.py -k Polly -v`
Expected: FAIL with `ImportError: cannot import name 'PollyEngine'`

- [ ] **Step 3: Implement `PollyEngine` and register it**

In `src/speeker/engines.py`, add the class after `KokoroEngine`:

```python
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
```

Update `_create_engine` to register Polly:

```python
def _create_engine(name: str) -> BaseEngine:
    if name == "pocket-tts":
        return PocketTTSEngine()
    if name == "kokoro":
        return KokoroEngine()
    if name == "polly":
        return PollyEngine()
    raise ValueError(f"Unknown engine: {name}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_engines.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/speeker/engines.py tests/test_engines.py
git commit -m "Add Polly engine with PCM-to-float32 conversion"
```

---

## Task 7: Declare boto3 as an optional dependency

**Files:**

- Modify: `pyproject.toml`

- [ ] **Step 1: Add the optional dependency**

In `pyproject.toml`, under `[project.optional-dependencies]`, add:

```toml
polly = [
    "boto3>=1.34",
]
```

- [ ] **Step 2: Verify the project still resolves**

Run: `uv sync --extra polly`
Expected: completes without error; boto3 installed.

- [ ] **Step 3: Verify existing tests still pass**

Run: `uv run pytest tests/test_engines.py tests/test_config.py -q`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "Add boto3 as optional 'polly' dependency"
```

---

## Task 8: Route CLI through the engine registry; add Polly + SSML flags

**Files:**

- Modify: `src/speeker/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cli.py` (add imports `import argparse`, `from unittest.mock import patch, MagicMock` if absent):

```python
class _RecordingEngine:
    name = "rec"
    supports_ssml = False

    def __init__(self):
        self.calls = []

    def default_voice(self):
        return "azelma"

    def validate_voice(self, voice):
        return True

    def generate(self, text, voice, *, is_ssml=False, **options):
        import numpy as np
        self.calls.append({"text": text, "voice": voice, "is_ssml": is_ssml, **options})
        return np.zeros(8, dtype=np.float32), 16000


class TestCliSsmlAndEngine:
    def test_plain_text_preprocessed_and_generated(self, tmp_path):
        from speeker import cli
        rec = _RecordingEngine()
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}), \
             patch.object(cli, "get_engine", return_value=rec):
            ok = cli.speak_text("Hello.", "pocket-tts", "azelma",
                                no_play=True, quiet=True, stdout=False)
        assert ok is True
        assert len(rec.calls) == 1
        assert rec.calls[0]["is_ssml"] is False

    def test_ssml_local_engine_stripped(self, tmp_path):
        from speeker import cli
        rec = _RecordingEngine()
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}), \
             patch.object(cli, "get_engine", return_value=rec):
            cli.speak_text("<speak>Hi <break/>there</speak>", "pocket-tts", "azelma",
                           no_play=True, quiet=True, stdout=False, is_ssml=True)
        assert rec.calls[0]["text"] == "Hi there"
        assert rec.calls[0]["is_ssml"] is False

    def test_polly_engine_passes_variant_and_ssml(self, tmp_path):
        from speeker import cli
        rec = _RecordingEngine()
        rec.supports_ssml = True
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}), \
             patch.object(cli, "get_engine", return_value=rec):
            cli.speak_text("<speak>hi</speak>", "polly", "Joanna",
                           no_play=True, quiet=True, stdout=False,
                           is_ssml=True, polly_engine="generative")
        assert rec.calls[0]["is_ssml"] is True
        assert rec.calls[0]["polly_engine"] == "generative"

    def test_parser_accepts_polly_and_ssml_flags(self):
        from speeker.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(
            ["speak", "hi", "-e", "polly", "--polly-engine", "neural",
             "--polly-voice", "Matthew", "--ssml", "--best-effort-ssml-emulation",
             "--aws-profile", "personal"]
        )
        assert args.engine == "polly"
        assert args.polly_engine == "neural"
        assert args.polly_voice == "Matthew"
        assert args.ssml is True
        assert args.emulate_ssml is True
        assert args.aws_profile == "personal"

    def test_aws_profile_sets_env(self, tmp_path):
        from speeker import cli
        rec = _RecordingEngine()
        rec.supports_ssml = True
        args = argparse.Namespace(
            text="hi", engine="polly", voice=None, polly_voice="Joanna",
            polly_engine="neural", ssml=False, emulate_ssml=False,
            aws_profile="personal", no_play=True, quiet=True, stdout=False, stream=False,
        )
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}), \
             patch.object(cli, "get_engine", return_value=rec):
            cli.cmd_speak(args)
        assert os.environ["AWS_PROFILE"] == "personal"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -k "SsmlAndEngine" -v`
Expected: FAIL (`build_parser` undefined; `speak_text` signature mismatch / uses removed helpers)

- [ ] **Step 3: Refactor `cli.py`**

In `src/speeker/cli.py`:

(a) Remove the model globals and the `TYPE_CHECKING` block importing `KPipeline`/`TTSModel`. Delete these definitions: `_pocket_tts_model`, `_pocket_tts_voice_states`, `_kokoro_pipeline`.

Add these imports near the top (add `import os` to the stdlib imports if absent; `cli.py` does not currently import it):

```python
import os
```

```python
from .engines import get_engine, prepare_payload
from .ssml import looks_like_ssml
from .config import get_ssml_config
from .voices import POLLY_VARIANT_DEFAULT_VOICE
```

Replace `get_pocket_tts_model`, `get_pocket_tts_voice_state`, `get_kokoro_pipeline`, `generate_pocket_tts`, and `generate_kokoro` with thin wrappers that delegate to the engine registry. These remain part of the public surface (`tests/test_integration.py` calls them directly to exercise real models), but now hold no duplicated model logic:

```python
def get_pocket_tts_model():
    """Backwards-compatible accessor: the warm pocket-tts model."""
    return get_engine("pocket-tts")._get_model()


def get_pocket_tts_voice_state(voice: str):
    """Backwards-compatible accessor: a pocket-tts voice state."""
    return get_engine("pocket-tts")._voice_state(voice)


def get_kokoro_pipeline():
    """Backwards-compatible accessor: the kokoro pipeline."""
    return get_engine("kokoro")._get_pipeline()


def generate_pocket_tts(text: str, voice: str):
    """Backwards-compatible helper used by integration tests."""
    return get_engine("pocket-tts").generate(text, voice)


def generate_kokoro(text: str, voice: str):
    """Backwards-compatible helper used by integration tests."""
    return get_engine("kokoro").generate(text, voice)
```

(b) Replace the entire `speak_text` function with:

```python
def speak_text(
    text: str,
    engine: str,
    voice: str,
    no_play: bool,
    quiet: bool,
    stdout: bool,
    *,
    is_ssml: bool = False,
    polly_engine: str | None = None,
    emulate: bool | None = None,
) -> bool:
    """Generate and optionally queue speech for a piece of text. Returns True on success."""
    if not text or not text.strip():
        return True  # Empty text is not an error

    if emulate is None:
        emulate = get_ssml_config().get("emulate_for_local", False)
    acronyms_file = get_ssml_config().get("acronyms_file")

    is_ssml = is_ssml or looks_like_ssml(text)

    try:
        eng = get_engine(engine)
        if is_ssml:
            payload, ssml_for_engine = prepare_payload(
                eng, text, is_ssml=True, emulate=emulate, acronyms_file=acronyms_file
            )
        else:
            payload, ssml_for_engine = preprocess_for_tts(text), False

        audio, sample_rate = eng.generate(
            payload, voice, is_ssml=ssml_for_engine, polly_engine=polly_engine
        )

        if stdout:
            audio_normalized = np.clip(audio, -1.0, 1.0)
            audio_int16 = (audio_normalized * 32767).astype(np.int16)
            wavfile.write(sys.stdout.buffer, sample_rate, audio_int16)
        elif no_play:
            audio_path = save_audio(audio, sample_rate, text)
            print(audio_path)
        else:
            audio_path = save_audio(audio, sample_rate, text)
            queue_for_playback(audio_path)
            if not quiet:
                print(f"Queued: {audio_path}", file=sys.stderr)
        return True

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return False
```

(c) Add a helper used by both command handlers and tests, before `cmd_speak`:

```python
def _resolve_voice(args: argparse.Namespace, engine: str) -> str:
    """Resolve the voice to use, honoring --polly-voice and per-variant defaults."""
    if getattr(args, "polly_voice", None):
        return args.polly_voice
    if args.voice:
        return args.voice
    if engine == "polly":
        variant = getattr(args, "polly_engine", None)
        if variant:
            return POLLY_VARIANT_DEFAULT_VOICE.get(variant, get_default_voice(engine))
    return get_default_voice(engine)
```

Also add a small helper near `_resolve_voice` that applies the AWS profile override (boto3 reads `AWS_PROFILE` natively, so setting it before generation is the idiomatic way to select a profile without touching `PollyEngine`):

```python
def _apply_aws_profile(args: argparse.Namespace) -> None:
    """Set AWS_PROFILE from --aws-profile so boto3 (Polly) picks it up."""
    if getattr(args, "aws_profile", None):
        os.environ["AWS_PROFILE"] = args.aws_profile
```

(d) In `cmd_speak`, replace the engine/voice resolution and validation block and the final `speak_text` call:

```python
    engine = args.engine or DEFAULT_ENGINE
    voice = _resolve_voice(args, engine)
    _apply_aws_profile(args)

    if engine not in ("pocket-tts", "kokoro", "polly"):
        print(f"Error: Unknown engine '{engine}'.", file=sys.stderr)
        return 1

    if not validate_voice(engine, voice):
        available = list(get_voices(engine).get(engine, {}).keys())
        print(f"Error: Unknown voice '{voice}' for engine '{engine}'.", file=sys.stderr)
        print(f"Available voices: {', '.join(available)}", file=sys.stderr)
        return 1

    if not args.quiet:
        print(f"Generating speech with {engine}/{voice}...", file=sys.stderr)

    if not speak_text(
        text, engine, voice, args.no_play, args.quiet, args.stdout,
        is_ssml=args.ssml, polly_engine=getattr(args, "polly_engine", None),
        emulate=args.emulate_ssml,
    ):
        return 1
    return 0
```

(e) In `cmd_speak_stream`, mirror the same resolution and pass-through. Replace its engine/voice block and the per-sentence `speak_text` call:

```python
    engine = args.engine or DEFAULT_ENGINE
    voice = _resolve_voice(args, engine)
    _apply_aws_profile(args)

    if engine not in ("pocket-tts", "kokoro", "polly"):
        print(f"Error: Unknown engine '{engine}'.", file=sys.stderr)
        return 1

    if not validate_voice(engine, voice):
        available = list(get_voices(engine).get(engine, {}).keys())
        print(f"Error: Unknown voice '{voice}' for engine '{engine}'.", file=sys.stderr)
        print(f"Available voices: {', '.join(available)}", file=sys.stderr)
        return 1

    if not args.quiet:
        print(f"Streaming with {engine}/{voice}...", file=sys.stderr)

    sentence_count = 0
    error_count = 0
    for sentence in stream_sentences_from_stdin():
        if not speak_text(
            sentence, engine, voice, args.no_play, args.quiet, args.stdout,
            is_ssml=args.ssml, polly_engine=getattr(args, "polly_engine", None),
            emulate=args.emulate_ssml,
        ):
            error_count += 1
        else:
            sentence_count += 1

    if not args.quiet:
        print(f"Streamed {sentence_count} sentence(s)", file=sys.stderr)
    return 1 if error_count > 0 and sentence_count == 0 else 0
```

(f) Extract the parser into a `build_parser()` function so tests can construct it. In `main()`, everything between `parser = argparse.ArgumentParser(...)` and the line `args = parser.parse_args()` is parser construction. Define `def build_parser() -> argparse.ArgumentParser:` and move that entire block verbatim into it (the `argparse.ArgumentParser(...)` call, `subparsers = parser.add_subparsers(...)`, and every `subparsers.add_parser(...)`/`add_argument(...)`/`set_defaults(...)` statement for the speak, voices, play, status, voice-prefs, generate-samples, bundle-prefs, and voice-clone subcommands), then `return parser`. Apply the speak/voices argument changes from (g) below within this function. Then change `main()` to:

```python
def main() -> int:
    from .migrate import migrate
    migrate()
    parser = build_parser()
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return 0
    return args.func(args)
```

(g) In the `speak` subparser setup, change the `-e/--engine` choices and add the new flags:

```python
    speak_parser.add_argument(
        "-e", "--engine", choices=["pocket-tts", "kokoro", "polly"], help="TTS engine"
    )
    speak_parser.add_argument(
        "--polly-engine",
        choices=["standard", "neural", "long-form", "generative"],
        help="Polly engine variant (only used with -e polly)",
    )
    speak_parser.add_argument(
        "--polly-voice", help="Polly VoiceId (overrides --voice when -e polly)"
    )
    speak_parser.add_argument(
        "--ssml", action="store_true", help="Treat input as SSML"
    )
    speak_parser.add_argument(
        "--best-effort-ssml-emulation", dest="emulate_ssml", action="store_true",
        help="Approximate SSML on local engines (spell acronyms, pauses as "
             "punctuation, normalize ALL-CAPS). No effect on Polly.",
    )
    speak_parser.add_argument(
        "--aws-profile",
        help="AWS profile for Polly (sets AWS_PROFILE; overrides the default "
             "credential chain). Equivalent to exporting AWS_PROFILE.",
    )
```

Also update the `voices` subparser `-e/--engine` choices to include `polly`:

```python
    voices_parser.add_argument(
        "-e", "--engine", choices=["pocket-tts", "kokoro", "polly"], help="Filter by engine"
    )
```

- [ ] **Step 4: Migrate the existing speak_text unit tests**

`speak_text` no longer calls `generate_pocket_tts`; it calls `get_engine(...).generate(...)`. The existing tests in `tests/test_cli.py` that patch `speeker.cli.generate_pocket_tts` and assert it was called must be migrated to patch `speeker.cli.get_engine` with a fake engine. Replace the bodies of `TestSpeakText.test_speak_text_success`, `test_speak_text_no_play`, `test_speak_text_handles_error`, and `TestSpeakTextAdvanced.test_speak_text_quiet_mode`, `test_speak_text_stdout_mode` with these versions (they reuse the `_RecordingEngine` added in Step 1):

```python
class TestSpeakText:
    """Tests for speak_text function."""

    @patch("speeker.cli.queue_for_playback")
    @patch("speeker.cli.save_audio")
    def test_speak_text_success(self, mock_save, mock_queue):
        """Test speak_text generates and queues audio."""
        rec = _RecordingEngine()
        mock_save.return_value = Path("/tmp/test.wav")
        with patch("speeker.cli.get_engine", return_value=rec):
            result = speak_text("Hello", "pocket-tts", "azelma", False, True, False)
        assert result is True
        assert len(rec.calls) == 1
        mock_save.assert_called_once()
        mock_queue.assert_called_once()

    def test_speak_text_empty_text(self):
        """Test speak_text returns True for empty text."""
        result = speak_text("", "pocket-tts", "azelma", False, True, False)
        assert result is True

    def test_speak_text_whitespace_text(self):
        """Test speak_text returns True for whitespace-only text."""
        result = speak_text("   ", "pocket-tts", "azelma", False, True, False)
        assert result is True

    @patch("speeker.cli.save_audio")
    def test_speak_text_no_play(self, mock_save, capsys):
        """Test speak_text with no_play prints path."""
        rec = _RecordingEngine()
        mock_save.return_value = Path("/tmp/test.wav")
        with patch("speeker.cli.get_engine", return_value=rec):
            result = speak_text("Hello", "pocket-tts", "azelma", True, False, False)
        assert result is True
        assert "/tmp/test.wav" in capsys.readouterr().out

    def test_speak_text_handles_error(self, capsys):
        """Test speak_text handles generation error."""
        rec = _RecordingEngine()
        rec.generate = MagicMock(side_effect=Exception("TTS failed"))
        with patch("speeker.cli.get_engine", return_value=rec):
            result = speak_text("Hello", "pocket-tts", "azelma", False, False, False)
        assert result is False
        assert "Error" in capsys.readouterr().err
```

```python
class TestSpeakTextAdvanced:
    """Additional tests for speak_text function."""

    @patch("speeker.cli.queue_for_playback")
    @patch("speeker.cli.save_audio")
    def test_speak_text_quiet_mode(self, mock_save, mock_queue, capsys):
        """Test speak_text quiet mode doesn't print to stderr."""
        rec = _RecordingEngine()
        mock_save.return_value = Path("/tmp/test.wav")
        with patch("speeker.cli.get_engine", return_value=rec):
            result = speak_text("Hello", "pocket-tts", "azelma", False, True, False)
        assert result is True
        assert "Queued" not in capsys.readouterr().err

    @patch("speeker.cli.wavfile.write")
    def test_speak_text_stdout_mode(self, mock_wavfile_write):
        """Test speak_text stdout mode writes to stdout."""
        rec = _RecordingEngine()
        with patch("speeker.cli.get_engine", return_value=rec):
            result = speak_text("Hello", "pocket-tts", "azelma", False, False, True)
        assert result is True
        mock_wavfile_write.assert_called_once()
```

(`tests/test_integration.py` is unchanged — it uses the `generate_pocket_tts` / `get_pocket_tts_model` wrappers kept in Step 3a.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -v`
Expected: PASS (new tests pass; migrated pre-existing CLI tests pass)

- [ ] **Step 6: Commit**

```bash
git add src/speeker/cli.py tests/test_cli.py
git commit -m "Route CLI generation through engine registry; add Polly and SSML flags"
```

---

## Task 9: Make the player daemon dispatch by engine and honor SSML

**Files:**

- Modify: `src/speeker/player.py`
- Modify: `src/speeker/queue_db.py` (`get_pending_for_session` returns metadata)
- Test: `tests/test_player.py`, `tests/test_queue_db.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_queue_db.py`:

```python
class TestPendingMetadata:
    def test_get_pending_returns_parsed_metadata(self, tmp_path):
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            from speeker.queue_db import enqueue, get_pending_for_session
            enqueue("hi", metadata={"queue": "q1", "ssml": True, "engine": "polly"})
            items = get_pending_for_session("q1")
            assert items[0]["metadata"]["ssml"] is True
            assert items[0]["metadata"]["engine"] == "polly"
```

Append to `tests/test_player.py`:

```python
class TestGenerateTtsDispatch:
    def test_dispatches_to_named_engine_plain(self, tmp_path):
        from speeker import player
        rec = _PlayerRecordingEngine()
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}), \
             patch.object(player, "get_engine", return_value=rec):
            out = player.generate_tts(
                "Hello.", voice="Joanna", engine="polly",
                save_path=tmp_path / "a.wav",
            )
        assert out == tmp_path / "a.wav"
        assert rec.calls[0]["is_ssml"] is False
        assert rec.calls[0]["voice"] == "Joanna"

    def test_ssml_local_engine_stripped(self, tmp_path):
        from speeker import player
        rec = _PlayerRecordingEngine()  # supports_ssml = False
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}), \
             patch.object(player, "get_engine", return_value=rec):
            player.generate_tts(
                "<speak>Hi <break/>there</speak>", voice="azelma",
                engine="pocket-tts", is_ssml=True, save_path=tmp_path / "b.wav",
            )
        assert rec.calls[0]["text"] == "Hi there"
        assert rec.calls[0]["is_ssml"] is False
```

Add this fake near the top of `tests/test_player.py` (after imports):

```python
class _PlayerRecordingEngine:
    name = "rec"
    supports_ssml = False

    def __init__(self):
        self.calls = []

    def default_voice(self):
        return "azelma"

    def generate(self, text, voice, *, is_ssml=False, **options):
        import numpy as np
        self.calls.append({"text": text, "voice": voice, "is_ssml": is_ssml, **options})
        return np.zeros(8, dtype=np.float32), 16000
```

Also add a test proving SSML items are spoken verbatim (not decorated with the
"First: " / "from ... ago:" prefixes that `build_session_script` adds):

```python
class TestProcessQueueSsml:
    def test_ssml_item_spoken_verbatim(self, tmp_path):
        from speeker import player
        from speeker.queue_db import enqueue
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            # Two items so build_session_script would prefix each ("First: ", "Last: ").
            enqueue("plain message", metadata={"queue": "q1"})
            enqueue("<speak>Hi there</speak>", metadata={"queue": "q1", "ssml": True})

            captured = []

            def fake_speak(line, **kw):
                captured.append((line, kw))
                return kw.get("save_path")

            with patch.object(player, "speak_text", side_effect=fake_speak):
                player.process_queue(verbose=False)

        ssml_lines = [line for line, kw in captured if kw.get("is_ssml")]
        assert ssml_lines == ["<speak>Hi there</speak>"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_player.py -k Dispatch tests/test_queue_db.py -k PendingMetadata -v`
Expected: FAIL (`generate_tts` has no `engine`/`is_ssml` kwargs; metadata not returned)

- [ ] **Step 3a: Return metadata from `get_pending_for_session`**

In `src/speeker/queue_db.py`, replace the body of `get_pending_for_session`:

```python
def get_pending_for_session(session_id: str) -> list[dict]:
    """Get all unplayed items for a session, ordered by creation time."""
    with get_connection() as conn:
        cursor = conn.execute(
            """
            SELECT id, session_id, text, audio_path, created_at, metadata
            FROM queue
            WHERE session_id = ? AND played_at IS NULL
            ORDER BY created_at ASC
            """,
            (session_id,),
        )
        items = []
        for row in cursor.fetchall():
            item = dict(row)
            if item.get("metadata"):
                try:
                    item["metadata"] = json.loads(item["metadata"])
                except (json.JSONDecodeError, TypeError):
                    item["metadata"] = None
            items.append(item)
        return items
```

- [ ] **Step 3b: Rewrite `player.py` generation to use engines**

In `src/speeker/player.py`:

Remove the `TYPE_CHECKING` import of `TTSModel`, the globals `_tts_model` and `_voice_states`, and the functions `get_tts_model` and `get_voice_state`. Add at top with other imports:

```python
from .engines import get_engine, prepare_payload, unload_all
from .ssml import looks_like_ssml
```

Replace `unload_tts_model` with:

```python
def unload_tts_model() -> None:
    """Unload all cached TTS engines to free memory."""
    unload_all()
```

Replace the entire `generate_tts` function with:

```python
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
) -> Path | None:
    """Generate TTS audio for text using the named engine."""
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

        if is_ssml:
            ssml_cfg = get_ssml_config()
            payload, ssml_for_engine = prepare_payload(
                eng, text, is_ssml=True,
                emulate=ssml_cfg.get("emulate_for_local", False),
                acronyms_file=ssml_cfg.get("acronyms_file"),
            )
        else:
            payload, ssml_for_engine = preprocess_for_tts(text), False

        audio_np, sample_rate = eng.generate(
            payload, voice, is_ssml=ssml_for_engine, polly_engine=polly_engine
        )

        if speed != 1.0 and speed > 0:
            from scipy import signal
            new_length = int(len(audio_np) / speed)
            audio_np = signal.resample(audio_np, new_length)

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
        if verbose:
            print(f"[ERROR] TTS failed: {e}", file=sys.stderr)
        return None
```

Update `speak_text` (player.py) to thread the engine/ssml options through to `generate_tts`:

```python
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
) -> Path | None:
    """Generate and play TTS for text. Handles leading $Note tone tokens."""
    tone_tokens, clean_text = extract_tone_tokens(text)
    if tone_tokens:
        play_tone_tokens(tone_tokens, verbose)

    if not clean_text:
        return save_path

    audio_path = generate_tts(
        clean_text, voice=voice, speed=speed, save_path=save_path, verbose=verbose,
        engine=engine, is_ssml=is_ssml, polly_engine=polly_engine,
    )
    if audio_path is None:
        return None

    try:
        play_audio(audio_path, verbose)
        return save_path
    finally:
        if save_path is None and audio_path:
            try:
                audio_path.unlink()
            except OSError:
                pass
```

- [ ] **Step 3c: Resolve per-item engine/voice/ssml in `process_queue`**

In `process_queue`, within the per-session loop, after `settings = get_settings(session_id)` keep `voice`/`speed` for the header line, but resolve per item inside the line loop. Replace the line loop body's `speak_text` call and the index logic with:

```python
        for line_idx, line in enumerate(script_lines):
            if line_idx > 0:
                time.sleep(PAUSE_BETWEEN_MESSAGES)

            item_idx = line_idx - 1 if not (len(items) == 1 and is_only_session) else line_idx
            save_path = None
            line_engine = settings["engine"]
            line_voice = settings["voice"]
            line_polly_engine = None
            line_is_ssml = False
            if 0 <= item_idx < len(items):
                item = items[item_idx]
                save_path = get_audio_save_path(item["id"])
                meta = item.get("metadata") or {}
                line_engine = meta.get("engine") or settings["engine"]
                line_voice = meta.get("voice") or settings["voice"]
                line_polly_engine = meta.get("polly_engine")
                line_is_ssml = bool(meta.get("ssml")) or looks_like_ssml(item["text"])
                if line_is_ssml:
                    # SSML must be spoken verbatim: a spoken prefix like "First: "
                    # would sit outside the <speak> root and corrupt the markup.
                    line = item["text"]

            result = speak_text(
                line, voice=line_voice, speed=speed, save_path=save_path, verbose=verbose,
                engine=line_engine, is_ssml=line_is_ssml, polly_engine=line_polly_engine,
            )

            if result is not None or save_path is None:
                total_played += 1
                if save_path and 0 <= item_idx < len(items):
                    update_audio_path(items[item_idx]["id"], save_path)
```

In `process_queue`, the intro/outro `speak_text("This is Claude Code.", ...)` and `speak_text("That is all.", ...)` calls should use the global engine. Change them to pass `engine=global_settings["engine"]`:

```python
        speak_text("This is Claude Code.", verbose=verbose, engine=global_settings["engine"])
```

```python
        speak_text("That is all.", verbose=verbose, engine=global_settings["engine"])
```

- [ ] **Step 3d: Warm the active engine in `run_daemon`**

In `run_daemon`, replace the warm-up block (the `if idle_timeout == 0:` branch that calls `get_tts_model()` / `get_voice_state("azelma")`) with:

`get_settings` is already imported at the top of `player.py`, so use it directly:

```python
    if idle_timeout == 0:
        default_engine = get_settings()["engine"]
        if verbose:
            print(f"[INFO] Warming up {default_engine} engine...", file=sys.stderr)
        get_engine(default_engine).warm()
        if verbose:
            print("[INFO] TTS engine ready!", file=sys.stderr)
    elif verbose:
        print(f"[INFO] Model idle timeout: {idle_timeout} min (lazy-load)", file=sys.stderr)
```

- [ ] **Step 3e: Migrate existing player tests that patched removed functions**

Removing `get_tts_model`/`get_voice_state` and rewriting `generate_tts`/`run_daemon` breaks tests that patched those names. In `tests/test_player.py`, replace the `TestGenerateTTS` class and the two affected `TestRunDaemon` tests. Add this fake near the top of the file if Step 1 has not already (it is the same `_PlayerRecordingEngine` from Step 1):

Replace the entire `TestGenerateTTS` class with:

```python
class TestGenerateTTS:
    """Tests for generate_tts function (dispatches through the engine registry)."""

    def test_generate_tts_success(self, tmp_path):
        from speeker import player
        rec = _PlayerRecordingEngine()
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}), \
             patch.object(player, "get_engine", return_value=rec):
            path = player.generate_tts("Hello world", verbose=False)
        assert path is not None
        assert path.exists()
        path.unlink()

    def test_generate_tts_with_save_path(self, tmp_path):
        from speeker import player
        rec = _PlayerRecordingEngine()
        save_path = tmp_path / "output.wav"
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}), \
             patch.object(player, "get_engine", return_value=rec):
            path = player.generate_tts("Hello world", save_path=save_path, verbose=False)
        assert path == save_path
        assert save_path.exists()

    def test_generate_tts_with_speed(self, tmp_path):
        from speeker import player
        rec = _PlayerRecordingEngine()
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}), \
             patch.object(player, "get_engine", return_value=rec):
            path = player.generate_tts("Hello world", speed=1.5, verbose=False)
        assert path is not None
        path.unlink()

    def test_generate_tts_error(self, tmp_path):
        from speeker import player
        rec = _PlayerRecordingEngine()
        rec.generate = MagicMock(side_effect=Exception("TTS error"))
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}), \
             patch.object(player, "get_engine", return_value=rec):
            path = player.generate_tts("Hello world", verbose=False)
        assert path is None
```

Replace `TestRunDaemon.test_run_daemon_preloads_when_timeout_zero` with a version that asserts the active engine is warmed (no more `get_tts_model`/`get_voice_state`):

```python
    @patch("speeker.player.time.sleep")
    @patch("speeker.player.get_pending_count")
    @patch("speeker.player.get_engine")
    @patch("speeker.player.release_lock")
    @patch("speeker.player.acquire_lock")
    @patch("speeker.config.get_player_config")
    def test_run_daemon_preloads_when_timeout_zero(
        self, mock_config, mock_acquire, mock_release, mock_get_engine,
        mock_pending, mock_sleep, tmp_path
    ):
        from speeker.player import run_daemon
        mock_config.return_value = {"model_idle_timeout_minutes": 0}
        mock_acquire.return_value = tmp_path / "player.lock"
        mock_pending.return_value = 0
        fake_engine = MagicMock()
        mock_get_engine.return_value = fake_engine

        call_count = [0]
        def sleep_side_effect(duration):
            call_count[0] += 1
            if call_count[0] >= 2:
                raise KeyboardInterrupt()
        mock_sleep.side_effect = sleep_side_effect

        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            try:
                run_daemon(verbose=False)
            except KeyboardInterrupt:
                pass

        fake_engine.warm.assert_called_once()
        mock_release.assert_called()
```

Replace `TestRunDaemon.test_run_daemon_skips_preload_with_timeout` with a version that patches `get_engine` and asserts `warm` is not called:

```python
    @patch("speeker.player.time.sleep")
    @patch("speeker.player.get_pending_count")
    @patch("speeker.player.get_engine")
    @patch("speeker.player.release_lock")
    @patch("speeker.player.acquire_lock")
    @patch("speeker.config.get_player_config")
    def test_run_daemon_skips_preload_with_timeout(
        self, mock_config, mock_acquire, mock_release, mock_get_engine,
        mock_pending, mock_sleep, tmp_path
    ):
        from speeker.player import run_daemon
        mock_config.return_value = {"model_idle_timeout_minutes": 5}
        mock_acquire.return_value = tmp_path / "player.lock"
        mock_pending.return_value = 0
        fake_engine = MagicMock()
        mock_get_engine.return_value = fake_engine

        call_count = [0]
        def sleep_side_effect(duration):
            call_count[0] += 1
            if call_count[0] >= 2:
                raise KeyboardInterrupt()
        mock_sleep.side_effect = sleep_side_effect

        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            try:
                run_daemon(verbose=False)
            except KeyboardInterrupt:
                pass

        fake_engine.warm.assert_not_called()
```

In `TestRunDaemon.test_run_daemon_unloads_model_after_idle`, remove the now-invalid `@patch("speeker.player.get_tts_model")` decorator and its corresponding `mock_model` parameter (the test still patches `process_queue` and `unload_tts_model` and asserts the unload happens after idle). The remaining decorators/params and body are unchanged.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_player.py tests/test_queue_db.py -v`
Expected: PASS (new dispatch tests pass; migrated player/queue tests pass)

- [ ] **Step 5: Commit**

```bash
git add src/speeker/player.py src/speeker/queue_db.py tests/test_player.py tests/test_queue_db.py
git commit -m "Player daemon dispatches by engine and honors SSML per message"
```

---

## Task 10: Server `/speak` SSML flag and Polly in `/voices`

**Files:**

- Modify: `src/speeker/server.py`
- Test: `tests/test_server.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_server.py` (it already builds a `TestClient(app)` — reuse the existing client fixture/pattern; if it uses a module-level `client = TestClient(app)`, reuse that name):

```python
class TestSsmlAndPolly:
    def test_speak_ssml_body_flag_stored_in_metadata(self, tmp_path):
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            from fastapi.testclient import TestClient
            from speeker.server import app
            from speeker.queue_db import get_history
            c = TestClient(app)
            r = c.post("/speak", json={"text": "<speak>hi</speak>", "ssml": True})
            assert r.json()["status"] == "success"
            hist = get_history(limit=1)
            assert hist[0]["metadata"]["ssml"] is True

    def test_speak_ssml_query_flag(self, tmp_path):
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            from fastapi.testclient import TestClient
            from speeker.server import app
            from speeker.queue_db import get_history
            c = TestClient(app)
            r = c.post("/speak?ssml=true", json={"text": "<speak>hi</speak>"})
            assert r.json()["status"] == "success"
            assert get_history(limit=1)[0]["metadata"]["ssml"] is True

    def test_voices_includes_polly(self, tmp_path):
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            from fastapi.testclient import TestClient
            from speeker.server import app
            c = TestClient(app)
            data = c.get("/voices").json()
            assert "polly" in data["engines"]
            assert "Joanna" in data["engines"]["polly"]["voices"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_server.py -k "SsmlAndPolly" -v`
Expected: FAIL (`ssml` not accepted / not stored; polly absent from /voices)

- [ ] **Step 3: Implement server changes**

In `src/speeker/server.py`:

(a) Add `ssml` to `SpeakRequest`:

```python
class SpeakRequest(BaseModel):
    text: str
    metadata: dict | None = None
    ssml: bool = False
    session_id: str | None = None  # Deprecated
```

(b) In the `speak` handler, after building `metadata` (merging body + query metadata) and before `enqueue`, add the SSML flag from the body or query param:

```python
        ssml = body.ssml or request.query_params.get("ssml", "").lower() == "true"
        if ssml:
            metadata["ssml"] = True
```

(c) Import Polly constants and add Polly to `/voices`. Update the import:

```python
from .voices import (
    POCKET_TTS_VOICES,
    KOKORO_VOICES,
    POLLY_VOICES,
    DEFAULT_ENGINE,
    DEFAULT_POCKET_TTS_VOICE,
    DEFAULT_KOKORO_VOICE,
    DEFAULT_POLLY_VOICE,
)
```

In `get_voices`, add after the kokoro block and extend `known_engines`:

```python
    if engine is None or engine == "polly":
        engines["polly"] = format_voices(POLLY_VOICES, DEFAULT_POLLY_VOICE)
```

```python
    known_engines = {"pocket-tts", "kokoro", "polly", "custom"}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_server.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/speeker/server.py tests/test_server.py
git commit -m "Server: accept ssml flag on /speak and list Polly in /voices"
```

---

## Task 11: Rule-based SSML generator with purpose presets

**Files:**

- Create: `src/speeker/ssml_generate.py`
- Test: `tests/test_ssml_generate.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ssml_generate.py`:

```python
#!/usr/bin/env python3
"""Unit tests for ssml_generate.py."""

from speeker.ssml_generate import (
    rule_based_ssml,
    PURPOSE_PRESETS,
    PURPOSE_ALIASES,
)


def _wrapped(s: str) -> bool:
    return s.startswith("<speak>") and s.endswith("</speak>")


class TestPresets:
    def test_expected_purposes_present(self):
        for p in ("audiobook", "article", "announcement", "conversational",
                  "technical", "plain"):
            assert p in PURPOSE_PRESETS

    def test_news_alias(self):
        assert PURPOSE_ALIASES["news"] == "article"


class TestRuleBasedSsml:
    def test_audiobook_structure(self):
        out = rule_based_ssml("Para one.\n\nPara two.", "audiobook")
        assert _wrapped(out)
        assert '<prosody rate="95%">' in out
        assert out.count("<p>") == 2
        assert '<break time="800ms"/>' in out

    def test_plain_has_no_prosody(self):
        out = rule_based_ssml("Just text.", "plain")
        assert _wrapped(out)
        assert "<prosody" not in out

    def test_announcement_emphasizes_first(self):
        out = rule_based_ssml("Big news. Details.", "announcement")
        assert "<emphasis" in out
        assert "<break" in out

    def test_technical_spells_acronyms(self):
        out = rule_based_ssml("The PHI record.", "technical")
        assert 'interpret-as="characters"' in out
        assert "PHI" in out

    def test_news_alias_resolves(self):
        out = rule_based_ssml("Hello.", "news")
        assert _wrapped(out)

    def test_escapes_specials(self):
        out = rule_based_ssml("Tom & Jerry", "audiobook")
        assert "Tom &amp; Jerry" in out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_ssml_generate.py -k "Presets or RuleBased" -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'speeker.ssml_generate'`

- [ ] **Step 3: Implement the rule-based generator**

Create `src/speeker/ssml_generate.py`:

```python
"""Generate purpose-tuned SSML from plain text.

Hybrid: uses the configured LLM backend when available (see generate_ssml in the
next task) and falls back to this deterministic rule-based generator otherwise.
All output is sanitized to the Polly-safe whitelist.
"""

import re

from .ssml import sanitize_ssml, escape_text, load_acronyms, _CAPS_WORD_RE

# Each preset documents itself (used by --help) and drives both the LLM prompt
# and the rule-based generator.
PURPOSE_PRESETS = {
    "audiobook": {
        "description": "Narration: slower pace, paragraph pauses, sentence pacing.",
        "rate": "95%",
        "para_break_ms": 800,
        "technical": False,
    },
    "article": {
        "description": "Measured and clear, for articles/blog posts (alias: news).",
        "rate": "100%",
        "para_break_ms": 500,
        "technical": False,
    },
    "announcement": {
        "description": "Emphatic: leading pause and a strong opening sentence.",
        "rate": "97%",
        "para_break_ms": 500,
        "technical": False,
    },
    "conversational": {
        "description": "Lighter and quicker with natural short pauses.",
        "rate": "105%",
        "para_break_ms": 350,
        "technical": False,
    },
    "technical": {
        "description": "Spells identifiers/acronyms; slower for clarity.",
        "rate": "95%",
        "para_break_ms": 500,
        "technical": True,
    },
    "plain": {
        "description": "No added markup; just wrap and escape.",
        "rate": "100%",
        "para_break_ms": 0,
        "technical": False,
    },
}

PURPOSE_ALIASES = {"news": "article"}


def resolve_purpose(purpose: str) -> str:
    """Resolve an alias to its canonical purpose name."""
    return PURPOSE_ALIASES.get(purpose, purpose)


def _markup_acronyms(paragraph: str, acronyms: set[str]) -> str:
    """Escape text and wrap known ALL-CAPS acronyms in say-as spell-out tags."""
    out: list[str] = []
    pos = 0
    for m in _CAPS_WORD_RE.finditer(paragraph):
        out.append(escape_text(paragraph[pos:m.start()]))
        word = m.group(0)
        if word.upper() in acronyms:
            out.append(f'<say-as interpret-as="characters">{escape_text(word)}</say-as>')
        else:
            out.append(escape_text(word))
        pos = m.end()
    out.append(escape_text(paragraph[pos:]))
    return "".join(out)


def rule_based_ssml(text: str, purpose: str = "audiobook") -> str:
    """Deterministically build SSML for *text* tuned to *purpose*."""
    purpose = resolve_purpose(purpose)
    preset = PURPOSE_PRESETS.get(purpose, PURPOSE_PRESETS["plain"])

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return "<speak></speak>"

    if purpose == "plain":
        return sanitize_ssml(" ".join(paragraphs))

    acronyms = load_acronyms() if preset["technical"] else set()
    parts: list[str] = []
    for i, para in enumerate(paragraphs):
        if preset["technical"]:
            content = _markup_acronyms(para, acronyms)
        else:
            content = escape_text(para)
        if purpose == "announcement" and i == 0:
            content = f'<emphasis level="strong">{content}</emphasis>'
        parts.append(f"<p>{content}</p>")
        if i < len(paragraphs) - 1 and preset["para_break_ms"]:
            parts.append(f'<break time="{preset["para_break_ms"]}ms"/>')

    body = "".join(parts)
    if purpose == "announcement":
        body = '<break time="500ms"/>' + body
    ssml = f'<speak><prosody rate="{preset["rate"]}">{body}</prosody></speak>'
    return sanitize_ssml(ssml)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_ssml_generate.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/speeker/ssml_generate.py tests/test_ssml_generate.py
git commit -m "Add rule-based SSML generator with purpose presets"
```

---

## Task 12: Hybrid SSML generation (LLM + fallback)

**Files:**

- Modify: `src/speeker/ssml_generate.py`
- Test: `tests/test_ssml_generate.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ssml_generate.py`:

````python
from unittest.mock import patch
from speeker import ssml_generate
from speeker.ssml_generate import generate_ssml


class TestGenerateSsml:
    def test_unknown_purpose_raises(self):
        import pytest
        with pytest.raises(ValueError):
            generate_ssml("hi", purpose="bogus")

    def test_empty_text(self):
        assert generate_ssml("   ", purpose="audiobook") == "<speak></speak>"

    def test_no_backend_falls_back_to_rule_based(self):
        with patch.object(ssml_generate, "_get_llm_settings",
                          return_value=("", "", "", "")):
            out = generate_ssml("Para one.\n\nPara two.", purpose="audiobook")
        assert '<prosody rate="95%">' in out

    def test_llm_output_sanitized_and_used(self):
        llm = '```xml\n<speak>Hi <script>x</script><break time="500ms"/>there</speak>\n```'
        with patch.object(ssml_generate, "_get_llm_settings",
                          return_value=("ollama", "", "", "")), \
             patch.object(ssml_generate, "call_llm", return_value=llm):
            out = generate_ssml("whatever", purpose="conversational")
        assert out.startswith("<speak>") and out.endswith("</speak>")
        assert "<script>" not in out
        assert '<break time="500ms"/>' in out

    def test_invalid_llm_output_falls_back(self):
        with patch.object(ssml_generate, "_get_llm_settings",
                          return_value=("ollama", "", "", "")), \
             patch.object(ssml_generate, "call_llm", return_value="<<<>>>"):
            out = generate_ssml("Para one.\n\nPara two.", purpose="audiobook")
        assert '<prosody rate="95%">' in out  # came from rule-based fallback
````

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_ssml_generate.py -k GenerateSsml -v`
Expected: FAIL with `ImportError: cannot import name 'generate_ssml'`

- [ ] **Step 3: Implement hybrid generation**

Append to `src/speeker/ssml_generate.py`:

````python
from .summarize import call_llm, _get_llm_settings
from .ssml import strip_ssml

_CODE_FENCE_RE = re.compile(r"```(?:xml|ssml)?\s*(.*?)\s*```", re.DOTALL)
_SPEAK_BLOCK_RE = re.compile(r"<speak\b.*?</speak>", re.DOTALL | re.IGNORECASE)

SSML_PROMPT_TEMPLATE = """Convert the text below into Amazon Polly SSML for {purpose} delivery.

Style: {style}

Rules:
- Output ONLY SSML wrapped in a single <speak> element. No explanation, no code fences.
- Use only these tags: speak, p, s, break, emphasis, prosody, say-as, sub.
- Do not change the wording; only add markup and pacing.

Text:
{text}

SSML:"""


def _extract_ssml(response: str) -> str:
    """Pull SSML out of an LLM response (strip code fences, prefer <speak> block)."""
    fence = _CODE_FENCE_RE.search(response)
    if fence:
        response = fence.group(1)
    block = _SPEAK_BLOCK_RE.search(response)
    if block:
        return block.group(0)
    return response.strip()


def _has_content(ssml: str) -> bool:
    """True if the sanitized SSML carries any spoken text."""
    return bool(strip_ssml(ssml).strip())


def build_prompt(text: str, purpose: str) -> str:
    preset = PURPOSE_PRESETS[purpose]
    return SSML_PROMPT_TEMPLATE.format(
        purpose=purpose, style=preset["description"], text=text
    )


def generate_ssml(text: str, purpose: str = "audiobook") -> str:
    """Generate purpose-tuned SSML from plain text (hybrid LLM + rule-based)."""
    purpose = resolve_purpose(purpose)
    if purpose not in PURPOSE_PRESETS:
        valid = ", ".join(sorted(set(PURPOSE_PRESETS) | set(PURPOSE_ALIASES)))
        raise ValueError(f"Unknown purpose '{purpose}'. Valid: {valid}")

    if not text or not text.strip():
        return "<speak></speak>"

    backend = _get_llm_settings()[0]
    if backend:
        try:
            response = call_llm(build_prompt(text, purpose))
            if response and response.strip():
                sanitized = sanitize_ssml(_extract_ssml(response))
                if _has_content(sanitized):
                    return sanitized
        except Exception:
            pass  # fall through to rule-based

    return rule_based_ssml(text, purpose)
````

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_ssml_generate.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/speeker/ssml_generate.py tests/test_ssml_generate.py
git commit -m "Add hybrid SSML generation with LLM and rule-based fallback"
```

---

## Task 13: CLI `ssml` subcommand

**Files:**

- Modify: `src/speeker/cli.py`
- Test: `tests/test_cli.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cli.py`:

```python
import io


class TestCliSsmlCommand:
    def test_generates_to_stdout(self, tmp_path, capsys):
        from speeker import cli
        args = argparse.Namespace(purpose="plain")
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}), \
             patch("sys.stdin", io.StringIO("Hello world.")):
            rc = cli.cmd_ssml(args)
        out = capsys.readouterr().out
        assert rc == 0
        assert out.strip().startswith("<speak>")

    def test_empty_stdin_errors(self, tmp_path, capsys):
        from speeker import cli
        args = argparse.Namespace(purpose="audiobook")
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}), \
             patch("sys.stdin", io.StringIO("   ")):
            rc = cli.cmd_ssml(args)
        assert rc == 1

    def test_parser_has_ssml_command(self):
        from speeker.cli import build_parser
        args = build_parser().parse_args(["ssml", "--purpose", "audiobook"])
        assert args.purpose == "audiobook"
        assert args.func.__name__ == "cmd_ssml"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -k SsmlCommand -v`
Expected: FAIL (`cmd_ssml` undefined; no `ssml` subcommand)

- [ ] **Step 3: Implement the command and subparser**

In `src/speeker/cli.py`, add the handler near the other `cmd_*` functions:

```python
def cmd_ssml(args: argparse.Namespace) -> int:
    """Read text on stdin, write purpose-tuned SSML to stdout."""
    from .ssml_generate import generate_ssml

    text = sys.stdin.read()
    if not text or not text.strip():
        print("Error: No input text on stdin", file=sys.stderr)
        return 1
    try:
        print(generate_ssml(text, purpose=args.purpose))
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0
```

In `build_parser()`, register the subcommand. First build the help epilog from the presets so every purpose is documented:

```python
    from .ssml_generate import PURPOSE_PRESETS, PURPOSE_ALIASES
    purpose_lines = "\n".join(
        f"  {name:<14} {preset['description']}"
        for name, preset in PURPOSE_PRESETS.items()
    )
    alias_lines = "\n".join(f"  {a:<14} alias for '{t}'" for a, t in PURPOSE_ALIASES.items())
    ssml_parser = subparsers.add_parser(
        "ssml",
        help="Generate SSML from stdin text",
        description="Read plain text on stdin and write purpose-tuned SSML to stdout.",
        epilog="Purposes:\n" + purpose_lines + "\n" + alias_lines,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ssml_parser.add_argument(
        "--purpose",
        choices=list(PURPOSE_PRESETS) + list(PURPOSE_ALIASES),
        default="audiobook",
        help="Delivery style (default: audiobook)",
    )
    ssml_parser.set_defaults(func=cmd_ssml)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -k SsmlCommand -v`
Expected: PASS

Then verify help renders:
Run: `uv run speeker ssml --help`
Expected: usage text listing every purpose with its description.

- [ ] **Step 5: Commit**

```bash
git add src/speeker/cli.py tests/test_cli.py
git commit -m "Add 'speeker ssml' command to generate SSML from stdin"
```

---

## Task 14: Server `/ssml` generation endpoint

**Files:**

- Modify: `src/speeker/server.py`
- Test: `tests/test_server.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_server.py`:

```python
class TestSsmlEndpoint:
    def test_generate_ssml(self, tmp_path):
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            from fastapi.testclient import TestClient
            from speeker.server import app
            c = TestClient(app)
            r = c.post("/ssml", json={"text": "Hello world.", "purpose": "plain"})
            data = r.json()
            assert data["status"] == "success"
            assert data["ssml"].startswith("<speak>")
            assert data["purpose"] == "plain"

    def test_unknown_purpose_errors(self, tmp_path):
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            from fastapi.testclient import TestClient
            from speeker.server import app
            c = TestClient(app)
            r = c.post("/ssml", json={"text": "hi", "purpose": "bogus"})
            assert r.json()["status"] == "error"

    def test_empty_text_400(self, tmp_path):
        with patch.dict(os.environ, {"SPEEKER_DIR": str(tmp_path)}):
            from fastapi.testclient import TestClient
            from speeker.server import app
            c = TestClient(app)
            r = c.post("/ssml", json={"text": "  "})
            assert r.status_code == 400
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_server.py -k SsmlEndpoint -v`
Expected: FAIL with 404 (endpoint missing)

- [ ] **Step 3: Implement the endpoint**

In `src/speeker/server.py`, add the models near the other Pydantic models:

```python
class SsmlRequest(BaseModel):
    text: str
    purpose: str = "audiobook"


class SsmlResponse(BaseModel):
    status: str
    ssml: str | None = None
    purpose: str | None = None
    error: str | None = None
```

Add the route (near `/summarize`):

```python
@app.post("/ssml", response_model=SsmlResponse)
async def make_ssml(body: SsmlRequest):
    """Generate purpose-tuned SSML from plain text. Pure transform; no enqueue."""
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    try:
        from .ssml_generate import generate_ssml
        ssml = generate_ssml(text, purpose=body.purpose)
        return SsmlResponse(status="success", ssml=ssml, purpose=body.purpose)
    except ValueError as e:
        return SsmlResponse(status="error", error=str(e))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_server.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/speeker/server.py tests/test_server.py
git commit -m "Add POST /ssml generation endpoint"
```

---

## Task 15: MCP tools — SSML/Polly on speak, and generate_ssml

**Files:**

- Modify: `mcp/server.py`
- Test: `tests/test_mcp.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_mcp.py`:

```python
#!/usr/bin/env python3
"""Tests for the MCP server tool wrappers (call_speeker mocked)."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# The MCP server lives outside the package; add its dir to the path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mcp"))
pytest.importorskip("mcp")

import server as mcp_server  # noqa: E402


class TestSpeakTool:
    def test_ssml_and_polly_payload(self):
        captured = {}

        def fake_call(endpoint, data):
            captured["endpoint"] = endpoint
            captured["data"] = data
            return {"status": "success", "queue_id": 1, "pending_count": 1}

        with patch.object(mcp_server, "call_speeker", side_effect=fake_call):
            mcp_server.speak(
                "<speak>hi</speak>", engine="polly", polly_engine="long-form",
                polly_voice="Danielle", ssml=True, queue="q1",
            )
        assert captured["endpoint"] == "/speak"
        data = captured["data"]
        assert data["ssml"] is True
        assert data["metadata"]["engine"] == "polly"
        assert data["metadata"]["voice"] == "Danielle"
        assert data["metadata"]["polly_engine"] == "long-form"
        assert data["metadata"]["queue"] == "q1"


class TestGenerateSsmlTool:
    def test_posts_to_ssml(self):
        captured = {}

        def fake_call(endpoint, data):
            captured["endpoint"] = endpoint
            captured["data"] = data
            return {"status": "success", "ssml": "<speak>hi</speak>", "purpose": "audiobook"}

        with patch.object(mcp_server, "call_speeker", side_effect=fake_call):
            out = mcp_server.generate_ssml("hello", purpose="audiobook")
        assert captured["endpoint"] == "/ssml"
        assert captured["data"] == {"text": "hello", "purpose": "audiobook"}
        assert out["ssml"] == "<speak>hi</speak>"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_mcp.py -v`
Expected: FAIL (`speak` has no `ssml`/`polly_engine` kwargs; `generate_ssml` undefined). If `mcp` is not installed, the test is skipped — install it for this task with `uv pip install mcp` or rely on CI where it is present.

- [ ] **Step 3: Implement MCP changes**

In `mcp/server.py`, replace the `speak` tool with:

```python
@mcp.tool()
def speak(
    text: str,
    engine: str | None = None,
    voice: str | None = None,
    queue: str | None = None,
    ssml: bool = False,
    polly_engine: str | None = None,
    polly_voice: str | None = None,
) -> dict[str, Any]:
    """
    Generate speech from text and queue for playback.

    Args:
        text: The text to speak (plain text, or SSML if ssml=True)
        engine: TTS engine: "pocket-tts", "kokoro", or "polly" (default: pocket-tts)
        voice: Voice to use (engine-specific; custom cloned voices allowed)
        queue: Queue name for grouping utterances (default: current project name)
        ssml: Treat text as SSML (native on Polly; emulated/stripped on local engines)
        polly_engine: Polly variant: "standard", "neural", "long-form", "generative"
        polly_voice: Polly VoiceId (overrides voice when engine="polly")

    Returns:
        Dictionary with status, queue_id, and pending_count
    """
    if not text or not text.strip():
        return {"status": "error", "error": "Text cannot be empty"}

    data: dict[str, Any] = {"text": text}
    metadata: dict[str, Any] = {"queue": queue or get_default_queue()}
    if engine:
        metadata["engine"] = engine
    chosen_voice = polly_voice or voice
    if chosen_voice:
        metadata["voice"] = chosen_voice
    if polly_engine:
        metadata["polly_engine"] = polly_engine
    data["metadata"] = metadata
    if ssml:
        data["ssml"] = True

    result = call_speeker("/speak", data)
    if result.get("status") == "success":
        return {
            "status": "success",
            "message": result.get("message", "Queued for playback"),
            "queue_id": result.get("queue_id"),
            "pending_count": result.get("pending_count"),
        }
    return result
```

Add a new tool after `list_voices`:

```python
@mcp.tool()
def generate_ssml(text: str, purpose: str = "audiobook") -> dict[str, Any]:
    """
    Convert plain text into purpose-tuned SSML (does not speak it).

    Hand the returned SSML to speak(text=ssml, ssml=True) to play it.

    Args:
        text: The plain text to convert
        purpose: Delivery style — "audiobook" (default), "article"/"news",
                 "announcement", "conversational", "technical", or "plain"

    Returns:
        Dictionary with status, ssml, and purpose
    """
    if not text or not text.strip():
        return {"status": "error", "error": "Text cannot be empty"}
    return call_speeker("/ssml", {"text": text, "purpose": purpose})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_mcp.py -v`
Expected: PASS (or SKIPPED if `mcp` unavailable in this environment)

- [ ] **Step 5: Commit**

```bash
git add mcp/server.py tests/test_mcp.py
git commit -m "MCP: SSML/Polly options on speak and a generate_ssml tool"
```

---

## Task 16: Documentation

**Files:**

- Modify: `README.md`

- [ ] **Step 1: Update the features and engines sections**

In `README.md`, change the engines feature bullet to mention Polly and SSML:

```markdown
- **Multiple TTS engines**: pocket-tts (fast), kokoro (higher quality), and Amazon Polly (cloud, SSML-native)
- **SSML support**: native on Polly; best-effort emulation on local engines
```

After the kokoro voices section, add a Polly section:

```markdown
### Amazon Polly (cloud, SSML-native)

Requires the optional dependency and AWS credentials:

\`\`\`bash
uv sync --extra polly # installs boto3
\`\`\`

Credentials come from the standard AWS chain (`~/.aws/credentials`, `AWS_PROFILE`,
environment, or instance role). Configure region/profile/voice in `config.json`:

\`\`\`json
{
"polly": {
"region": "us-east-1",
"profile": null,
"engine": "neural",
"voice": "Joanna"
}
}
\`\`\`

Selecting a profile (in order of precedence):

- `speeker speak --aws-profile NAME ...` (CLI; sets `AWS_PROFILE` for that run)
- `polly.profile` in `config.json`
- the `AWS_PROFILE` environment variable (use this for the server/daemon, e.g.
  `AWS_PROFILE=personal speeker-server`); `AWS_DEFAULT_REGION` selects the region

Engine variants (`--polly-engine`): `standard` (cheapest), `neural` (natural,
widest voice selection), `long-form` (narration), `generative` (most human).

\`\`\`bash
speeker speak -e polly --polly-engine neural --polly-voice Matthew --aws-profile personal "Hello there."
speeker speak -e polly --polly-engine long-form --polly-voice Danielle < chapter.txt
\`\`\`
```

- [ ] **Step 2: Add an SSML section**

Add a new top-level section before `## Architecture`:

```markdown
## SSML

Mark input as SSML with `--ssml`, or it is auto-detected when the text starts
with `<speak>`:

\`\`\`bash
speeker speak -e polly --ssml '<speak>Hello <break time="500ms"/> world.</speak>'
\`\`\`

- **Polly** renders SSML natively.
- **Local engines** (pocket-tts, kokoro) do not support SSML. By default the tags
  are stripped to plain text. With `--best-effort-ssml-emulation` (or
  `ssml.emulate_for_local: true` in config), Speeker approximates SSML: spelling
  acronyms (`<say-as interpret-as="characters">PHI</say-as>` → "P-H-I"), turning
  `<break>` into punctuation, and normalizing ALL-CAPS so the engine does not shout.

Extra acronyms to spell out can be listed in a file referenced by
`ssml.acronyms_file` (tokens separated by whitespace, commas, pipes, or semicolons);
they are merged with the built-in set.

### Generating SSML

`speeker ssml` reads plain text on stdin and writes purpose-tuned SSML to stdout.
It uses the configured LLM backend when available and falls back to a rule-based
generator; output is always sanitized to Polly-safe tags.

\`\`\`bash
speeker ssml --purpose audiobook < chapter.txt | speeker speak --ssml -e polly --polly-engine long-form
\`\`\`

Purposes: `audiobook` (default), `article` (alias `news`), `announcement`,
`conversational`, `technical`, `plain`. Run `speeker ssml --help` for descriptions.

Also available over HTTP (`POST /ssml` with `{"text": "...", "purpose": "..."}`)
and as the MCP `generate_ssml` tool.
```

- [ ] **Step 3: Update the config/settings reference**

In the config/settings area of `README.md`, note the new sections:

```markdown
- `polly`: `region`, `profile`, `engine` (variant), `voice`
- `ssml`: `emulate_for_local`, `acronyms_file`
```

And add `polly` to the engine value list in the settings table row:

```markdown
| `engine` | "pocket-tts" | TTS engine (pocket-tts/kokoro/polly) |
```

- [ ] **Step 4: Verify the docs match the CLI**

Run: `uv run speeker speak --help && uv run speeker ssml --help`
Expected: the flags and purposes shown match what the README documents.

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "Document Polly engine, SSML input, and the ssml generator"
```

---

## Final verification

- [ ] **Run the full test suite**

Run: `uv run pytest -q --ignore=tests/test_preprocessing_stt.py`
Expected: all tests pass (note: `test_preprocessing_stt.py` and `test_voice_clone_roundtrip.py` are known pre-existing failures per project memory — exclude or ignore them).

- [ ] **Lint**

Run: `uv run ruff check src/ tests/`
Expected: no errors.
