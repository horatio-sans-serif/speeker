"""Summarization for TTS with flexible LLM backends.

Supports:
- Local models via Ollama
- Anthropic API
- OpenAI-compatible endpoints

Configuration via environment variables:
- SPEEKER_LLM_BACKEND: "ollama", "anthropic", or "openai" (default: none/fallback)
- SPEEKER_LLM_ENDPOINT: API endpoint URL
- SPEEKER_LLM_API_KEY: API key (for anthropic/openai)
- SPEEKER_LLM_MODEL: Model name (default depends on backend)
"""

import json
import re
import urllib.request
import urllib.error
from typing import Any

from .config import get_llm_config


def _get_llm_settings():
    """Get current LLM settings (re-read on each call to pick up config changes)."""
    config = get_llm_config()
    return (
        config.get("backend") or "",
        config.get("endpoint") or "",
        config.get("api_key") or "",
        config.get("model") or "",
    )

# Default models per backend
DEFAULT_MODELS = {
    "ollama": "llama3.2:1b",  # Small, fast
    "anthropic": "claude-3-haiku-20240307",
    "openai": "gpt-4o-mini",
}

# Default endpoints
DEFAULT_ENDPOINTS = {
    "ollama": "http://localhost:11434",
    "anthropic": "https://api.anthropic.com",
    "openai": "https://api.openai.com",
}

SUMMARIZE_PROMPT = """Summarize what was accomplished in one or two complete sentences for someone listening hands-free.

Rules:
- One or two complete sentences (about 30 words total) — never stop mid-sentence
- Start with a past-tense action verb (Fixed, Updated, Added, Completed, Resolved, etc.)
- No file paths, URLs, code, or technical jargon
- Describe the outcome, not the process
- Natural spoken English

Text:
{text}

Summary:"""


def summarize_for_speech(text: str, max_words: int = 30) -> str:
    """Summarize text for TTS using configured LLM backend.

    Args:
        text: The text to summarize
        max_words: Maximum words in summary

    Returns:
        A short, speakable summary
    """
    if not text or not text.strip():
        return "Task completed"

    # Truncate very long inputs
    if len(text) > 4000:
        text = text[:4000] + "..."

    # Try LLM summarization if configured
    llm_backend, _, _, _ = _get_llm_settings()
    if llm_backend:
        try:
            response = call_llm(SUMMARIZE_PROMPT.format(text=text))
            if response:
                summary = clean_summary(response, max_words)
                if summary:
                    return summary
        except Exception as e:
            print(f"LLM summarization error: {e}", flush=True)

    # Fallback to heuristic
    return fallback_summarize(text, max_words)


# Categories the assessor may assign. NEUTRAL means "no outcome cue" and is
# surfaced to callers as ``None`` -- a normal turn that is neither a clear
# success/failure nor a question gets no chime (matches the hook's "else, do
# not set an interpretation" rule).
ASSESS_CATEGORIES = ("SUCCESS", "FAILURE", "USER_PROMPT", "NEUTRAL")

ASSESS_PROMPT = """You are labeling an AI assistant's FINAL message to its user, then summarizing it for hands-free listening.

Choose exactly one category:
- USER_PROMPT: the message ends by asking the user a question, or asks for a decision, confirmation, or clarification it needs before it can continue.
- SUCCESS: the message reports that the requested work was completed successfully.
- FAILURE: the message reports that the work failed, errored, was blocked, or could not be completed.
- NEUTRAL: anything else (status, partial progress, an explanation, an answer to a question) with no clear success, failure, or question to the user.

Then write a summary:
- One or two complete sentences, about 30 words, natural spoken English, no file paths, URLs, or code.
- For SUCCESS, FAILURE, or NEUTRAL: start with a past-tense verb describing the outcome (Fixed, Added, Failed, Explained, ...).
- For USER_PROMPT: phrase it as a question to the listener, e.g. "I have a question about whether to delete the old records."

Respond with ONLY a JSON object and nothing else:
{{"category": "<one of SUCCESS, FAILURE, USER_PROMPT, NEUTRAL>", "summary": "<the summary>"}}

Message:
{text}

JSON:"""

# Last-line / decision-request signals for the no-LLM heuristic. A question
# aimed at the user almost always lands at the very end of the turn, so the
# heuristic inspects the tail rather than the whole message.
_USER_PROMPT_PHRASES = re.compile(
    r"\b(would you like|do you want|should i|shall i|which (?:one|option|approach|of)|"
    r"could you (?:confirm|clarify|let me know)|can you (?:confirm|clarify)|"
    r"please (?:confirm|clarify|advise|let me know)|let me know|"
    r"do you have a preference|what would you|how would you|"
    r"want me to|like me to)\b",
    re.IGNORECASE,
)
# Deliberately strict: only constructions that narrate the *turn* failing, not
# any mention of the word "error"/"fail" (which appears just as often in
# success reports like "fixed the failing test"). The LLM path is authoritative;
# this only has to catch explicit failure narration when no LLM is configured.
_FAILURE_PHRASES = re.compile(
    r"\b(could not|could ?n'?t|cannot|can ?not|can'?t|unable to|"
    r"failed(?: to| with)?|failing (?:with|to|on)|"
    r"did ?n'?t work|does ?n'?t work|wo ?n'?t work|"
    r"blocked|aborted|gave up|ran into (?:an? )?(?:error|problem|issue)|"
    r"hit (?:an? )?(?:error|problem|issue))\b",
    re.IGNORECASE,
)
_SUCCESS_PHRASES = re.compile(
    r"\b(done|completed?|finished|fixed|resolved|implemented|deployed|"
    r"all tests pass(?:ed|ing)?|tests? (?:now )?pass|succeed(?:ed)?|success|"
    r"is now working|works now|ready to go|merged|shipped)\b",
    re.IGNORECASE,
)


def _message_tail(text: str, chars: int = 400) -> str:
    """Last paragraph (or trailing ``chars``) -- where a question to the user lands."""
    stripped = text.strip()
    # Prefer the final non-empty paragraph; fall back to a raw character tail.
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", stripped) if p.strip()]
    tail = paragraphs[-1] if paragraphs else stripped
    return tail[-chars:]


def _last_question_sentence(text: str) -> str | None:
    """The final sentence ending in '?', cleaned for speech, or None."""
    questions = re.findall(r"[^.!?\n]*\?", text)
    for q in reversed(questions):
        q = q.strip().lstrip("-•* ").strip()
        if len(q) >= 8:
            return q
    return None


def _last_prompt_sentence(text: str) -> str | None:
    """Final sentence that *reads* as a request to the user, even without a '?'.

    Catches statement-form asks ("Would you like me to use the cached path.")
    so the spoken summary can be the actual request rather than a generic
    "I have a question for you." Inspects the tail where such asks land.
    """
    tail = _message_tail(text)
    sentences = re.split(r"(?<=[.!?])\s+", tail)
    for sentence in reversed(sentences):
        s = sentence.strip().lstrip("-•* ").strip()
        if len(s) >= 8 and _USER_PROMPT_PHRASES.search(s):
            return s
    return None


def _heuristic_category(text: str) -> str | None:
    """Classify without an LLM. Returns a category name or None for neutral.

    Order matters: a turn that mentions an error but ends by asking the user a
    question is a USER_PROMPT, not a FAILURE -- the question is the actionable
    part, so it wins.
    """
    tail = _message_tail(text)
    if tail.rstrip().endswith("?") or _USER_PROMPT_PHRASES.search(tail):
        return "USER_PROMPT"
    if _FAILURE_PHRASES.search(tail) or _FAILURE_PHRASES.search(text[-800:]):
        return "FAILURE"
    if _SUCCESS_PHRASES.search(tail) or _SUCCESS_PHRASES.search(text[:200]):
        return "SUCCESS"
    return None


def _heuristic_assess(text: str, max_words: int) -> tuple[str | None, str]:
    """No-LLM assessment: category + a style-appropriate spoken summary."""
    category = _heuristic_category(text)
    if category == "USER_PROMPT":
        question = _last_question_sentence(text) or _last_prompt_sentence(text)
        summary = question or "I have a question for you."
        # Keep the spoken question within the word budget.
        words = summary.split()
        if len(words) > max_words:
            summary = " ".join(words[:max_words]).rstrip(" ,;:") + "?"
        return "USER_PROMPT", summary
    return category, fallback_summarize(text, max_words)


def _parse_assessment(raw: str) -> tuple[str | None, str] | None:
    """Parse the LLM's JSON object into (category-or-None, summary).

    Returns None if no valid object with a known category is found, so the
    caller can fall back to the heuristic. NEUTRAL maps to a None category.
    """
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return None
    category = str(obj.get("category", "")).strip().upper()
    summary = str(obj.get("summary", "")).strip()
    if category not in ASSESS_CATEGORIES or not summary:
        return None
    return (None if category == "NEUTRAL" else category), summary


def assess_and_summarize(text: str, max_words: int = 30) -> tuple[str | None, str]:
    """Classify a turn's outcome and produce a style-matched spoken summary.

    Returns ``(interpretation, summary)`` where ``interpretation`` is one of
    ``"SUCCESS"``, ``"FAILURE"``, ``"USER_PROMPT"``, or ``None`` (a neutral
    turn that warrants no outcome cue). USER_PROMPT summaries are phrased as a
    question to the listener.

    A single LLM call does both jobs when a backend is configured; otherwise a
    conservative keyword heuristic classifies and the existing extractive
    summarizer supplies the text. The heuristic is also the fallback whenever
    the LLM output can't be parsed into a known category.
    """
    if not text or not text.strip():
        return None, "Task completed"

    if len(text) > 4000:
        text = text[:4000] + "..."

    llm_backend, _, _, _ = _get_llm_settings()
    if llm_backend:
        try:
            response = call_llm(ASSESS_PROMPT.format(text=text))
            if response:
                parsed = _parse_assessment(response)
                if parsed is not None:
                    category, summary = parsed
                    return category, clean_summary(summary, max_words) or summary
        except Exception as e:
            print(f"LLM assessment error: {e}", flush=True)

    return _heuristic_assess(text, max_words)


def call_llm(prompt: str, max_tokens: int = 100) -> str | None:
    """Call the configured LLM backend. *max_tokens* defaults to 100 for
    short-summary callers; longer-form callers (e.g. SSML generation
    where the output must include both markup and the full input text)
    should pass a larger value, typically 4000+."""
    llm_backend, _, _, _ = _get_llm_settings()
    if llm_backend == "ollama":
        return call_ollama(prompt, max_tokens=max_tokens)
    elif llm_backend == "anthropic":
        return call_anthropic(prompt, max_tokens=max_tokens)
    elif llm_backend == "openai":
        return call_openai(prompt, max_tokens=max_tokens)
    return None


def call_ollama(prompt: str, max_tokens: int = 100) -> str | None:
    """Call Ollama API."""
    _, llm_endpoint, _, llm_model = _get_llm_settings()
    endpoint = llm_endpoint or DEFAULT_ENDPOINTS["ollama"]
    model = llm_model or DEFAULT_MODELS["ollama"]

    data = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": max_tokens,
        }
    }).encode('utf-8')

    req = urllib.request.Request(
        f"{endpoint}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode('utf-8'))
            return result.get("response", "")
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None


def call_anthropic(prompt: str, max_tokens: int = 100) -> str | None:
    """Call Anthropic API."""
    _, llm_endpoint, llm_api_key, llm_model = _get_llm_settings()
    if not llm_api_key:
        return None

    endpoint = llm_endpoint or DEFAULT_ENDPOINTS["anthropic"]
    model = llm_model or DEFAULT_MODELS["anthropic"]

    data = json.dumps({
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }).encode('utf-8')

    req = urllib.request.Request(
        f"{endpoint}/v1/messages",
        data=data,
        headers={
            "Content-Type": "application/json",
            "x-api-key": llm_api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read().decode('utf-8'))
            content = result.get("content", [])
            if isinstance(content, list) and content:
                return content[0].get("text", "")
            return None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None


def call_openai(prompt: str, max_tokens: int = 100) -> str | None:
    """Call OpenAI-compatible API."""
    _, llm_endpoint, llm_api_key, llm_model = _get_llm_settings()
    if not llm_api_key:
        return None

    endpoint = llm_endpoint or DEFAULT_ENDPOINTS["openai"]
    model = llm_model or DEFAULT_MODELS["openai"]

    data = json.dumps({
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0.3,
        "messages": [{"role": "user", "content": prompt}],
    }).encode('utf-8')

    req = urllib.request.Request(
        f"{endpoint}/v1/chat/completions",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {llm_api_key}",
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read().decode('utf-8'))
            choices = result.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")
            return None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None


def clean_summary(text: str, max_words: int) -> str:
    """Clean up an LLM response into one or two concise, *complete* sentences.

    The word budget is enforced by dropping whole trailing sentences rather than
    slicing mid-phrase, so a spoken summary never cuts off abruptly. A single
    over-long sentence is trimmed only as a last resort, ending on a clause
    boundary.
    """
    text = text.strip()

    # If there are multiple lines, take the last non-empty one
    # (LLMs often add headers before the actual summary)
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if len(lines) > 1:
        # Take the last line that looks like a sentence
        for line in reversed(lines):
            if len(line) > 10 and not line.endswith(':'):
                text = line
                break

    # Remove quotes if wrapped
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    if text.startswith("'") and text.endswith("'"):
        text = text[1:-1]

    # Remove common prefixes (case-insensitive)
    prefixes = [
        "summary:", "here's a summary:", "speakable summary:",
        "here is a summary:", "tts summary:", "short summary:",
        "here's the summary:", "the summary is:",
        "one-sentence summary:", "one sentence summary:",
    ]
    text_lower = text.lower()
    for prefix in prefixes:
        if text_lower.startswith(prefix):
            text = text[len(prefix):].strip()
            text_lower = text.lower()

    # Remove leading dashes or bullets
    text = re.sub(r'^[-•*]\s*', '', text)

    # Keep at most the first two sentences (a complete thought, not a fragment).
    sentences = re.findall(r'[^.!?]+[.!?]', text)
    if sentences:
        text = ' '.join(s.strip() for s in sentences[:2])

    # Enforce the word budget without chopping mid-sentence: keep whole
    # sentences while they fit; only hard-trim a lone over-long sentence.
    if len(text.split()) > max_words:
        parts = re.findall(r'[^.!?]+[.!?]', text) or [text]
        kept: list[str] = []
        count = 0
        for sentence in parts:
            sentence = sentence.strip()
            n = len(sentence.split())
            if kept and count + n > max_words:
                break
            kept.append(sentence)
            count += n
        text = ' '.join(kept) if kept else text

        words = text.split()
        if len(words) > max_words:
            clipped = ' '.join(words[:max_words])
            if ',' in clipped:  # prefer to stop at a clause boundary
                clipped = clipped.rsplit(',', 1)[0]
            text = clipped.rstrip(' ,;:')
            if text and text[-1] not in '.!?':
                text += '.'

    return text.strip()


def fallback_summarize(text: str, max_words: int = 30) -> str:
    """Fallback summarization without LLM - extracts key outcome sentence."""
    # Remove code blocks
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'`[^`]+`', '', text)

    # Remove file paths
    text = re.sub(r'[/~][a-zA-Z0-9_./-]+', '', text)

    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)

    # Remove markdown formatting
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[-*]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\|[^|]*\|', ' ', text)
    text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)

    # Remove common verbose patterns that introduce summaries/lists
    verbose_patterns = [
        r"Here'?s a summary[:\s].*",
        r"Summary[:\s].*",
        r"Here'?s what (?:was|I) (?:done|did|changed|added|fixed|updated)[:\s].*",
        r"The following (?:changes|updates|fixes) were (?:made|applied)[:\s].*",
        r"Changes[:\s].*",
        r"What was done[:\s].*",
        r"Key (?:changes|findings|points)[:\s].*",
    ]
    for pattern in verbose_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    # Clean whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    if not text or len(text) < 5:
        return "Task completed."

    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)

    # Find the first good sentence (short, starts with action verb, no colons)
    action_verbs = [
        'fixed', 'added', 'updated', 'completed', 'resolved', 'created',
        'removed', 'implemented', 'deployed', 'configured', 'enabled',
        'disabled', 'changed', 'modified', 'refactored', 'moved', 'done',
        'finished', 'built', 'installed', 'set', 'applied', 'merged',
    ]

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Skip sentences that are too short or introduce lists
        if len(sentence) < 10:
            continue
        if sentence.endswith(':'):
            continue
        if re.search(r'\d+\.\s', sentence):
            continue

        words = sentence.split()
        word_count = len(words)

        # Prefer sentences that start with action verbs
        first_word = words[0].lower().rstrip('.,!?:')
        if first_word in action_verbs and word_count <= max_words:
            if not sentence.endswith('.'):
                sentence += '.'
            return sentence

        # Accept any reasonably short sentence
        if word_count <= max_words and word_count >= 3:
            if not sentence.endswith('.'):
                sentence += '.'
            return sentence

    # Last resort: take first max_words from first sentence
    if sentences:
        first = sentences[0].strip()
        words = first.split()[:max_words]
        summary = ' '.join(words)
        if summary and summary[-1] not in '.!?':
            summary = summary.rstrip(',;:') + '.'
        return summary

    return "Task completed."


def get_backend_info() -> dict[str, Any]:
    """Get information about the configured backend."""
    llm_backend, llm_endpoint, llm_api_key, llm_model = _get_llm_settings()

    if not llm_backend:
        return {
            "configured": False,
            "backend": None,
            "message": "No LLM backend configured. Set llm.backend in ~/.config/speeker/config.json or SPEEKER_LLM_BACKEND env var.",
        }

    return {
        "configured": True,
        "backend": llm_backend,
        "endpoint": llm_endpoint or DEFAULT_ENDPOINTS.get(llm_backend, ""),
        "model": llm_model or DEFAULT_MODELS.get(llm_backend, ""),
        "has_api_key": bool(llm_api_key),
    }
