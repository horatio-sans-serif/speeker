"""Text preprocessing for TTS - converts symbols to spoken words."""

import re

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


def preprocess_for_tts(text: str) -> str:
    """Preprocess text for better TTS output.

    Converts symbols, abbreviations, and technical notation
    to spoken equivalents.
    """
    if not text:
        return text

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

    # Apply single letter patterns
    for pattern, replacement in SINGLE_LETTER_PATTERNS:
        result = re.sub(pattern, replacement, result)

    # Apply late patterns
    for pattern, replacement in LATE_PATTERNS:
        result = re.sub(pattern, replacement, result)

    # Clean up multiple spaces
    result = re.sub(r'\s+', ' ', result)

    # Clean up space before punctuation
    result = re.sub(r'\s+([.,!?;:])', r'\1', result)

    return result.strip()
