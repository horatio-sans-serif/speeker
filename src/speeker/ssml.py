"""SSML primitives: detection, stripping, emulation, and sanitization.

These helpers are deliberately built on a tolerant regex tokenizer rather than a
strict XML parser, so they degrade gracefully on malformed input (e.g. LLM
output). Amazon Polly remains the authority on fine attribute validity.
"""

import html
import re
from pathlib import Path

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
