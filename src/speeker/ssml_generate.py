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


from .summarize import call_llm, _get_llm_settings  # noqa: E402
from .ssml import strip_ssml  # noqa: E402

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
            # Only trust output that actually produced SSML markup; otherwise the
            # sanitizer would happily escape garbage like "<<<>>>" into text.
            if response and "<speak" in response.lower():
                sanitized = sanitize_ssml(_extract_ssml(response))
                if _has_content(sanitized):
                    return sanitized
        except Exception:
            pass  # fall through to rule-based

    return rule_based_ssml(text, purpose)
