"""Text preprocessing for TTS - converts symbols to spoken words."""

import re
from functools import lru_cache

from .config import get_pronunciation_overrides

# Symbol to spoken word mappings
SYMBOL_REPLACEMENTS = [
    # Arrows
    ("→", " to "),
    ("←", " from "),
    ("↔", " between "),
    ("⇒", " implies "),
    ("⇐", " implied by "),
    ("↑", " up "),
    ("↓", " down "),
    ("⬆", " up "),
    ("⬇", " down "),

    # Math/Logic
    ("≠", " not equal to "),
    ("≈", " approximately "),
    ("≤", " less than or equal to "),
    ("≥", " greater than or equal to "),
    ("±", " plus or minus "),
    ("×", " times "),
    ("÷", " divided by "),
    ("∞", " infinity "),
    ("√", " square root of "),
    ("∑", " sum of "),
    ("∏", " product of "),
    ("∈", " in "),
    ("∉", " not in "),
    ("⊂", " subset of "),
    ("⊃", " superset of "),
    ("∩", " intersection "),
    ("∪", " union "),
    ("∧", " and "),
    ("∨", " or "),
    ("¬", " not "),

    # Common symbols
    ("•", ", "),
    ("·", " "),
    ("…", "..."),
    ("—", ", "),  # em dash
    ("–", " to "),  # en dash (often used for ranges)
    ("©", " copyright "),
    ("®", " registered "),
    ("™", " trademark "),
    ("°", " degrees "),
    ("′", " prime "),
    ("″", " double prime "),
    ("§", " section "),
    ("¶", " paragraph "),
    ("†", " dagger "),
    ("‡", " double dagger "),
    ("※", " note "),

    # Currency
    ("€", " euros "),
    ("£", " pounds "),
    ("¥", " yen "),
    ("₹", " rupees "),
    ("₿", " bitcoin "),

    # Checkmarks and X marks
    ("✓", " check "),
    ("✔", " check "),
    ("✕", " x "),
    ("✖", " x "),
    ("✗", " x "),
    ("✘", " x "),
    ("☑", " checked "),
    ("☐", " unchecked "),

    # Stars and ratings
    ("★", " star "),
    ("☆", " star "),
    ("⭐", " star "),

    # Common emoji-like symbols
    ("❤", " heart "),
    ("♥", " heart "),
    ("👍", " thumbs up "),
    ("👎", " thumbs down "),
]

# Regex patterns applied FIRST (order matters - more specific first)
EARLY_PATTERNS = [
    # File path patterns (must come before other dot handling)
    (r"\.\./", " dot dot slash "),  # ../
    (r"\./", " dot slash "),        # ./
    (r"~/", " home slash "),        # ~/

    # File extensions - common ones
    (r"\.([a-zA-Z]{1,4})(?=\s|$|[,;:\)])", r" dot \1 "),  # .py, .sh, .json, etc.

    # Path slashes (after ~/ ./ ../ handling)
    (r"(?<=[a-zA-Z0-9])/(?=[a-zA-Z0-9])", " slash "),  # path/to/file

    # Version numbers with dots (before general decimal handling)
    (r"(\d+)\.(\d+)\.(\d+)", r"\1 point \2 point \3"),  # 3.11.1
    (r"(\d+)\.(\d+)(?!\d)", r"\1 point \2"),  # 3.11

    # Percentage - must come before other number handling
    # Add comma for pause to help TTS emphasize
    (r"(\d+(?:\.\d+)?)\s*%", r"\1, percent"),

    # Programming operators (before general symbol handling)
    (r"===", " triple equals "),
    (r"!==", " not triple equals "),
    (r"==", " equals "),
    (r"!=", " not equals "),
    (r">=", " greater or equal "),
    (r"<=", " less or equal "),
    (r"&&", " and "),
    (r"\|\|", " or "),
    (r"=>", " arrow "),
    (r"->", " arrow "),
    (r"::", " double colon "),
]

# Abbreviation expansions
ABBREVIATION_PATTERNS = [
    # Common abbreviations with periods
    (r"\be\.g\.(?:,|\s|$)", " for example "),
    (r"\bi\.e\.(?:,|\s|$)", " that is "),
    (r"\betc\.(?:,|\s|$)", " etcetera "),
    (r"\bvs\.(?:\s|$)", " versus "),
    (r"\ba\.k\.a\.(?:\s|$)", " also known as "),

    # Without periods
    (r"\bvs\b", " versus "),
    (r"\bw/o\b", " without "),
    (r"\bw/\b", " with "),
    (r"\bb/c\b", " because "),
    (r"\baka\b", " also known as "),

    # Acronyms
    (r"\bFYI\b", " F Y I "),
    (r"\bASAP\b", " A S A P "),
    (r"\bIMO\b", " in my opinion "),
    (r"\bIMHO\b", " in my humble opinion "),
    (r"\bTBD\b", " T B D "),
    (r"\bTBA\b", " T B A "),
    (r"\bN/A\b", " N A "),
    (r"\bn/a\b", " N A "),

    # Email prefixes
    (r"\bRE:\s*", " regarding "),
    (r"\bFW:\s*", " forwarded "),
]

# Technical terms / tool names that TTS engines mispronounce as words.
# Matched case-insensitively as whole words (so "Louvre" or "value" are untouched).
# Extend this list as new offenders turn up.
TERM_PRONUNCIATIONS = [
    (re.compile(r"\buvx\b", re.IGNORECASE), "you vee x"),
    (re.compile(r"\buv\b", re.IGNORECASE), "you vee"),  # python's uv, not "oove"
    (re.compile(r"\bnpx\b", re.IGNORECASE), "n p x"),
    (re.compile(r"\bjq\b", re.IGNORECASE), "jay queue"),
    (re.compile(r"\byaml\b", re.IGNORECASE), "yammel"),
    (re.compile(r"\btodos\b", re.IGNORECASE), "to dos"),
    (re.compile(r"\btodo\b", re.IGNORECASE), "to do"),  # not "TOTO"
]

# Single letter handling - add slight pause/emphasis
SINGLE_LETTER_PATTERNS = [
    # Single letters that might get lost - add "letter" for clarity when standalone
    # Only when surrounded by spaces/punctuation (not in words)
    (r"(?<![a-zA-Z])([A-Z])(?![a-zA-Z])", r" \1 "),  # Preserve but space out capitals
]

# Late patterns - cleanup and edge cases
LATE_PATTERNS = [
    # Clean up any remaining forward slashes in isolation
    (r"\s/\s", " slash "),
    (r"^/", "slash "),

    # Handle remaining dots that might be problematic
    # (but not sentence-ending periods)
    (r"\.(?=[a-zA-Z])", " dot "),
]


def _compile_overrides(items: tuple[tuple[str, str], ...]) -> list[tuple[re.Pattern, str]]:
    """Compile a hashable items tuple into (pattern, replacement) pairs.

    Kept module-private and hashable-argument-only so ``lru_cache`` can store
    the compiled patterns across calls without re-running ``re.compile``.
    """
    compiled = []
    for word, replacement in items:
        if not word:
            continue
        compiled.append((re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE), replacement))
    return compiled


@lru_cache(maxsize=16)
def _cached_overrides(items: tuple[tuple[str, str], ...]) -> tuple[tuple[re.Pattern, str], ...]:
    return tuple(_compile_overrides(items))


def _resolve_override_value(value, engine: str | None) -> str | None:
    """Pick the right replacement for *engine* from a config entry.

    A value can be either a plain string (applies to all engines) or a
    dict ``{engine_name: replacement, ...}`` with an optional ``"default"``
    fallback. Returns the chosen replacement or ``None`` to skip.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        # Engine-specific entry wins.
        if engine and engine in value and isinstance(value[engine], str):
            return value[engine]
        if "default" in value and isinstance(value["default"], str):
            return value["default"]
    return None


def _load_pronunciation_overrides(engine: str | None = None) -> tuple[tuple[re.Pattern, str], ...]:
    """Load and compile user-supplied pronunciation overrides for *engine*.

    Patterns are cached per ``(engine, items)`` tuple so the regex compile
    is paid once per distinct configuration -- changing the active engine
    or updating the config recomputes naturally.
    """
    try:
        from .config import get_pronunciation_disabled
        raw = get_pronunciation_overrides()
        disabled = get_pronunciation_disabled()
    except Exception:
        return ()
    resolved: list[tuple[str, str]] = []
    for word, value in raw.items():
        if word in disabled:
            continue
        chosen = _resolve_override_value(value, engine)
        if chosen:
            resolved.append((word, chosen))
    return _cached_overrides(tuple(sorted(resolved)))


def preprocess_for_tts(
    text: str,
    acronyms: set[str] | None = None,
    *,
    engine: str | None = None,
) -> str:
    """Preprocess text for better TTS output.

    Converts symbols, abbreviations, and technical notation to spoken
    equivalents, fixes commonly mispronounced tool names (e.g. "uv" ->
    "you vee"), and spells out known acronyms letter-by-letter.

    User-supplied ``pronunciation.overrides`` from ``config.json`` run
    LAST so they win against any built-in transformation. When *engine*
    is supplied, per-engine override variants are honored (e.g. a
    ``{"polly": "...", "pocket-tts": "..."}`` entry picks the right one);
    otherwise the ``"default"`` value of any per-engine entry is used.

    *acronyms* defaults to the built-in set (``COMMON_ACRONYMS``); the player
    passes that set extended by the configured ``ssml.acronyms_file`` so every
    playback path -- CLI, HTTP, and MCP -- gets the same acronym handling that
    previously only applied to SSML text.
    """
    if not text:
        return text

    if acronyms is None:
        from .ssml import load_acronyms
        acronyms = load_acronyms()

    result = text

    # Apply early patterns (order-sensitive)
    for pattern, replacement in EARLY_PATTERNS:
        result = re.sub(pattern, replacement, result)

    # Apply symbol replacements (simple string replace)
    for symbol, spoken in SYMBOL_REPLACEMENTS:
        result = result.replace(symbol, spoken)

    # Apply abbreviation patterns
    for pattern, replacement in ABBREVIATION_PATTERNS:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

    # Fix mispronounced technical terms / tool names (uv, jq, ...)
    for pattern, spoken in TERM_PRONUNCIATIONS:
        result = pattern.sub(spoken, result)

    # Spell out known acronyms (ALL-CAPS words in the acronym set) as spaced
    # letters, e.g. "PHI" -> "P H I", which engines read letter-by-letter.
    # Unknown ALL-CAPS words are left as-is so common words like "DEPLOYED"
    # aren't mangled.
    if acronyms:
        from .ssml import _CAPS_WORD_RE
        result = _CAPS_WORD_RE.sub(
            lambda m: " ".join(m.group(0).upper())
            if m.group(0).upper() in acronyms
            else m.group(0),
            result,
        )

    # Apply single letter patterns
    for pattern, replacement in SINGLE_LETTER_PATTERNS:
        result = re.sub(pattern, replacement, result)

    # Apply late patterns
    for pattern, replacement in LATE_PATTERNS:
        result = re.sub(pattern, replacement, result)

    # User pronunciation overrides -- applied LAST so a user override wins
    # over any built-in TERM_PRONUNCIATIONS / acronym substitution above.
    # Engine-specific variants (when *engine* is set) take precedence over
    # the entry's "default" string.
    for pattern, replacement in _load_pronunciation_overrides(engine):
        result = pattern.sub(replacement, result)

    # Clean up multiple spaces
    result = re.sub(r'\s+', ' ', result)

    # Clean up space before punctuation
    result = re.sub(r'\s+([.,!?;:])', r'\1', result)

    return result.strip()
