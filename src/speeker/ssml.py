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
