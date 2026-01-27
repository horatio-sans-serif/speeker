"""Parse musical notation from text."""

import re


def extract_tone_tokens(text: str) -> tuple[list[str], str]:
    """Extract $Note tokens from text and return (tokens, clean_text).

    Example: "$Eb4 $Eb4 Hello world" -> (["Eb4", "Eb4"], "Hello world")

    Args:
        text: Input text possibly containing tone tokens

    Returns:
        Tuple of (list of note tokens, remaining text)
    """
    tokens = []
    pattern = r"^\s*\$([A-Ga-g][b#]?[0-8])"
    remaining = text

    while True:
        match = re.match(pattern, remaining)
        if not match:
            break
        tokens.append(match.group(1))
        remaining = remaining[match.end():]

    return tokens, remaining.strip()
