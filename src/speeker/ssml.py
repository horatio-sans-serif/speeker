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

# Per-tag attribute whitelist. LLMs occasionally invent attributes that aren't
# in Polly's SSML vocabulary (e.g. `<say-as type="prosody">`) — these produce
# well-formed XML that Polly's SSML parser still rejects. Anything not listed
# here is dropped from the output tag. Tags absent from this map have no
# allowed attributes and are emitted bare.
POLLY_ALLOWED_ATTRS: dict[str, set[str]] = {
    "speak": {"version", "xml:lang", "xmlns"},
    "break": {"time", "strength"},
    "emphasis": {"level"},
    "lang": {"xml:lang"},
    "mark": {"name"},
    "p": set(),
    "s": set(),
    "phoneme": {"alphabet", "ph"},
    "prosody": {"rate", "pitch", "volume"},
    "say-as": {"interpret-as", "format", "detail"},
    "sub": {"alias"},
    "w": {"role"},
    "amazon:domain": {"name"},
    "amazon:effect": {"name", "phonation", "vocal-tract-length"},
    "amazon:breath": {"duration", "volume"},
    "amazon:auto-breaths": {"volume", "frequency", "duration"},
}

# Per-engine restrictions. Polly's Neural and Standard engines support a
# strict subset of the full SSML grammar — features like <emphasis>,
# <prosody volume>, and the amazon: namespace tags raise an
# "Unsupported Neural feature" error if sent to those engines, even though
# Long-form accepts them. We filter SSML down to the engine's subset before
# sending to keep the audiobook pipeline working regardless of LLM output.
#
# Reference: https://docs.aws.amazon.com/polly/latest/dg/ssml.html (feature
# tables per engine variant).
_NEURAL_UNSUPPORTED_TAGS: set[str] = {
    "emphasis",
    "amazon:auto-breaths",
    "amazon:breath",
    "amazon:domain",
    "w",
}
_NEURAL_UNSUPPORTED_ATTRS: dict[str, set[str]] = {
    "prosody": {"volume"},
}
# Engines that share Neural's restrictions. Long-form and Generative accept
# the full Polly grammar; Standard is the same as Neural for our purposes.
_RESTRICTED_ENGINES: set[str] = {"neural", "standard"}

# Tags that Polly forbids nesting. When an opener for one of these appears
# while one is already open, close the previous one first.
_NON_NESTABLE: set[str] = {"p", "s"}

# Tags that lose their meaning without a specific attribute. If filtering
# leaves them attribute-less, drop the tag entirely (preserve text content).
_TAGS_REQUIRING_ATTRS: dict[str, str] = {
    "say-as": "interpret-as",
    "sub": "alias",
    "phoneme": "ph",
    "mark": "name",
    "amazon:domain": "name",
    "amazon:effect": "name",
}

# Matches a single tag: opening, closing, or self-closing, with attributes.
# Groups: 1=leading slash (closing), 2=name, 3=attributes, 4=trailing slash.
_TAG_RE = re.compile(
    r"<\s*(/?)\s*([a-zA-Z][\w:-]*)"
    r"((?:\s+[\w:-]+(?:\s*=\s*(?:\"[^\"]*\"|'[^']*'|[^\s>]+))?)*)"
    r"\s*(/?)\s*>"
)

# Splits an attribute string into (name, raw-value-including-quotes) pairs.
_ATTR_RE = re.compile(
    r"([\w:-]+)(?:\s*=\s*(\"[^\"]*\"|'[^']*'|[^\s>]+))?"
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


def _filter_attrs(
    name: str, raw_attrs: str, banned_attrs: set[str] | None = None
) -> str | None:
    """Reconstruct an attribute string with only Polly-allowed attributes.

    Returns the cleaned attribute string (with a leading space if non-empty),
    or ``None`` if the tag requires a specific attribute that's missing —
    signalling that the caller should drop the tag entirely. ``banned_attrs``
    is an extra blacklist (engine-specific) that's removed even if the
    attribute is in the global allowed set."""
    allowed = POLLY_ALLOWED_ATTRS.get(name, set())
    banned = banned_attrs or set()
    kept: list[str] = []
    has_required = name not in _TAGS_REQUIRING_ATTRS
    required = _TAGS_REQUIRING_ATTRS.get(name)
    for am in _ATTR_RE.finditer(raw_attrs or ""):
        attr_name = am.group(1).lower()
        if attr_name in allowed and attr_name not in banned:
            value = am.group(2)
            if value is None:
                kept.append(attr_name)
            else:
                # Normalize to double-quoted form so reconstruction is stable.
                if value.startswith("'") and value.endswith("'"):
                    inner = value[1:-1].replace('"', "&quot;")
                    value = f'"{inner}"'
                elif not (value.startswith('"') and value.endswith('"')):
                    value = f'"{value}"'
                kept.append(f"{attr_name}={value}")
            if attr_name == required:
                has_required = True
    if not has_required:
        return None
    return (" " + " ".join(kept)) if kept else ""


def sanitize_ssml(
    text: str,
    allowed_tags: set[str] = POLLY_SAFE_TAGS,
    polly_engine: str | None = None,
) -> str:
    """Return valid SSML containing only whitelisted tags, with one <speak> root.

    Disallowed tags are removed but their text content is kept. Per-tag
    attribute whitelisting strips Polly-invalid attributes (e.g. an
    LLM-hallucinated `<say-as type="prosody">`) — without this, the document
    is well-formed XML but Polly's stricter SSML parser still rejects it. Tags
    that lose their meaning without a required attribute (say-as without
    interpret-as, sub without alias, etc.) are dropped entirely and their
    inner text is preserved.

    Robust to malformed input: unmatched/garbled tags become text, and any
    container tags (p, s, prosody, emphasis, ...) the LLM left open are
    auto-closed before the trailing </speak>. This matters specifically when
    an LLM truncates mid-paragraph — without this pass, the result has an
    unclosed <p> that Polly's strict XML parser rejects.

    ``polly_engine`` narrows the allowed feature set further. The Neural and
    Standard engines reject features Long-form accepts (<emphasis>, prosody
    volume, amazon:domain, etc.); when ``polly_engine`` is one of those, the
    extra restrictions kick in so the output goes through cleanly.
    """
    engine_restricted = polly_engine and polly_engine.lower() in _RESTRICTED_ENGINES
    effective_allowed = (
        allowed_tags - _NEURAL_UNSUPPORTED_TAGS if engine_restricted else allowed_tags
    )
    out: list[str] = []
    open_stack: list[str] = []  # tag names that have been opened, in order
    # Tracks tag names whose opening tag was dropped (because a required
    # attribute was missing) so the matching close tag is also dropped.
    # A list rather than a counter to handle interleaved drops correctly.
    dropped_opens: list[str] = []
    pos = 0
    for m in _TAG_RE.finditer(text):
        out.append(escape_text(text[pos:m.start()]))
        name = m.group(2).lower()
        raw_attrs = m.group(3) or ""
        closing = m.group(1) == "/"
        self_closing = m.group(4) == "/"
        if name in effective_allowed:
            if closing:
                if name in dropped_opens:
                    # Matching opener was dropped — drop this close too.
                    dropped_opens.remove(name)
                else:
                    out.append(f"</{name}>")
                    if open_stack and open_stack[-1] == name:
                        open_stack.pop()
            else:
                banned = (
                    _NEURAL_UNSUPPORTED_ATTRS.get(name, set())
                    if engine_restricted
                    else None
                )
                cleaned_attrs = _filter_attrs(name, raw_attrs, banned)
                if cleaned_attrs is None:
                    # Tag is meaningless without its required attribute — drop
                    # opener and remember to drop the matching close too.
                    if not self_closing:
                        dropped_opens.append(name)
                else:
                    # Polly forbids nested <p> and <s>. If the LLM opens a new
                    # paragraph or sentence while one is already open, close
                    # the open one first so we emit valid SSML.
                    if name in _NON_NESTABLE and not self_closing:
                        while name in open_stack:
                            top = open_stack.pop()
                            out.append(f"</{top}>")
                            if top == name:
                                break
                    suffix = "/" if self_closing else ""
                    out.append(f"<{name}{cleaned_attrs}{suffix}>")
                    if name != "speak" and not self_closing:
                        open_stack.append(name)
        # else: drop the tag marker, surrounding text already handled
        pos = m.end()
    out.append(escape_text(text[pos:]))
    # Close any container tags the input left open. Done before _unwrap_speak
    # so the closes are inside the soon-to-be-rewrapped <speak> element.
    while open_stack:
        out.append(f"</{open_stack.pop()}>")
    inner = _unwrap_speak("".join(out))
    return f"<speak>{inner}</speak>"


def is_well_formed_ssml(text: str) -> bool:
    """True if *text* parses as XML. Used by generate_ssml to decide whether
    to trust LLM output or fall back to the rule-based generator."""
    import xml.etree.ElementTree as ET

    try:
        ET.fromstring(text)
        return True
    except ET.ParseError:
        return False


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
