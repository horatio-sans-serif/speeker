"""Split text/SSML into chunks within Amazon Polly's SynthesizeSpeech limits.

Polly's real-time ``SynthesizeSpeech`` caps a request at 3,000 *billed*
(spoken) characters and 6,000 *total* characters (SSML tags count toward total
but not billed). Longer input is rejected with TextLengthExceededException, so
the engine splits long input here and concatenates the resulting audio.

Splitting prefers natural boundaries: sentences for plain text; top-level
elements (e.g. ``<p>``) for SSML, never breaking inside a tag.
"""

from __future__ import annotations

import re

# Margins under Polly's hard limits (3000 billed / 6000 total).
MAX_BILLED = 2900
MAX_TOTAL = 5800

_SPEAK_RE = re.compile(r"^\s*<speak(\s[^>]*)?>(.*)</speak>\s*$", re.DOTALL)
_ATOM_RE = re.compile(r"<[^>]+>|[^<]+")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_ELEMENT_RE = re.compile(r"^(<([A-Za-z][\w:-]*)(?:\s[^>]*)?>)(.*)(</\2>)$", re.DOTALL)


def billed_len(text: str, is_ssml: bool) -> int:
    """Number of billed (spoken) characters: all text outside SSML tags."""
    if not is_ssml:
        return len(text)
    return sum(len(a) for a in _ATOM_RE.findall(text) if not a.startswith("<"))


def split_payload(text: str, is_ssml: bool) -> list[str]:
    """Split *text* into payloads each within Polly's limits.

    Returns a list of one or more payloads (empty input -> empty list). When the
    input already fits, returns it unchanged as a single-element list.
    """
    text = text.strip()
    if not text:
        return []
    if is_ssml:
        if billed_len(text, True) <= MAX_BILLED and len(text) <= MAX_TOTAL:
            return [text]
        return _split_ssml(text)
    if len(text) <= MAX_BILLED:
        return [text]
    return _split_plain(text, MAX_BILLED)


# --- plain text ---------------------------------------------------------------

def _split_plain(text: str, max_chars: int) -> list[str]:
    chunks: list[str] = []
    cur = ""
    for sent in _SENT_SPLIT.split(text):
        sent = sent.strip()
        if not sent:
            continue
        if len(sent) > max_chars:
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.extend(_split_words(sent, max_chars))
            continue
        if not cur:
            cur = sent
        elif len(cur) + 1 + len(sent) <= max_chars:
            cur += " " + sent
        else:
            chunks.append(cur)
            cur = sent
    if cur:
        chunks.append(cur)
    return chunks


def _split_words(s: str, max_chars: int) -> list[str]:
    out: list[str] = []
    cur = ""
    for word in s.split():
        if len(word) > max_chars:
            if cur:
                out.append(cur)
                cur = ""
            for i in range(0, len(word), max_chars):
                out.append(word[i:i + max_chars])
            continue
        if not cur:
            cur = word
        elif len(cur) + 1 + len(word) <= max_chars:
            cur += " " + word
        else:
            out.append(cur)
            cur = word
    if cur:
        out.append(cur)
    return out


# --- SSML ---------------------------------------------------------------------

def _split_ssml(ssml: str) -> list[str]:
    m = _SPEAK_RE.match(ssml)
    if m:
        attrs = m.group(1) or ""
        inner = m.group(2)
    else:
        attrs = ""
        inner = ssml
    open_speak = f"<speak{attrs}>"
    close_speak = "</speak>"
    wrap_overhead = len(open_speak) + len(close_speak)

    inner_chunks = _split_inner(inner, MAX_BILLED, MAX_TOTAL - wrap_overhead)
    return [f"{open_speak}{c}{close_speak}" for c in inner_chunks]


def _top_level_segments(inner: str) -> list[str]:
    """Break *inner* into top-level (depth-0) segments without splitting tags.

    A balanced element at depth 0 becomes one segment; runs of top-level text
    become their own segments (so they can be sentence-split if oversized).
    """
    segments: list[str] = []
    depth = 0
    buf = ""
    for atom in _ATOM_RE.findall(inner):
        if atom.startswith("<"):
            buf += atom
            if atom.startswith("</"):
                depth -= 1
            elif not atom.endswith("/>"):
                depth += 1
            if depth == 0:
                segments.append(buf)
                buf = ""
        else:
            if depth == 0:
                if buf:
                    segments.append(buf)
                    buf = ""
                segments.append(atom)
            else:
                buf += atom
    if buf:
        segments.append(buf)
    return segments


def _split_inner(inner: str, max_billed: int, max_total: int) -> list[str]:
    """Pack top-level segments into chunks within billed/total budgets."""
    chunks: list[str] = []
    cur = ""

    def billed(s: str) -> int:
        return billed_len(s, True)

    for seg in _top_level_segments(inner):
        units = [seg]
        # Oversized single segment: break it down so it can fit on its own.
        if billed(seg) > max_billed or len(seg) > max_total:
            units = _split_oversized_segment(seg, max_billed, max_total)

        for unit in units:
            if not cur:
                cur = unit
            elif billed(cur + unit) <= max_billed and len(cur + unit) <= max_total:
                cur += unit
            else:
                if cur.strip():
                    chunks.append(cur)
                cur = unit
    if cur.strip():
        chunks.append(cur)
    return chunks


def _split_oversized_segment(seg: str, max_billed: int, max_total: int) -> list[str]:
    elem = _ELEMENT_RE.match(seg)
    if elem:
        open_tag, _name, body, close_tag = (
            elem.group(1), elem.group(2), elem.group(3), elem.group(4)
        )
        overhead = len(open_tag) + len(close_tag)
        sub = _split_inner(body, max_billed, max_total - overhead)
        return [f"{open_tag}{c}{close_tag}" for c in sub]
    if not seg.startswith("<"):
        # Top-level text run: sentence/word split.
        return _split_plain(seg.strip(), max_billed)
    # Unparseable tag soup: return as-is (rare; Polly will surface any error).
    return [seg]
