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

SUMMARIZE_PROMPT = """Write ONE short sentence (max 15 words) summarizing what was accomplished.

Rules:
- ONE sentence only, no more
- Max 15 words
- Start with a past-tense action verb (Fixed, Updated, Added, Completed, Resolved, etc.)
- No file paths, URLs, code, or technical jargon
- Describe the outcome, not the process
- Natural spoken English

Text:
{text}

One-sentence summary:"""


def summarize_for_speech(text: str, max_words: int = 15) -> str:
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


def call_llm(prompt: str) -> str | None:
    """Call the configured LLM backend."""
    llm_backend, _, _, _ = _get_llm_settings()
    if llm_backend == "ollama":
        return call_ollama(prompt)
    elif llm_backend == "anthropic":
        return call_anthropic(prompt)
    elif llm_backend == "openai":
        return call_openai(prompt)
    return None


def call_ollama(prompt: str) -> str | None:
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
            "num_predict": 100,
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


def call_anthropic(prompt: str) -> str | None:
    """Call Anthropic API."""
    _, llm_endpoint, llm_api_key, llm_model = _get_llm_settings()
    if not llm_api_key:
        return None

    endpoint = llm_endpoint or DEFAULT_ENDPOINTS["anthropic"]
    model = llm_model or DEFAULT_MODELS["anthropic"]

    data = json.dumps({
        "model": model,
        "max_tokens": 100,
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


def call_openai(prompt: str) -> str | None:
    """Call OpenAI-compatible API."""
    _, llm_endpoint, llm_api_key, llm_model = _get_llm_settings()
    if not llm_api_key:
        return None

    endpoint = llm_endpoint or DEFAULT_ENDPOINTS["openai"]
    model = llm_model or DEFAULT_MODELS["openai"]

    data = json.dumps({
        "model": model,
        "max_tokens": 100,
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
    """Clean up LLM response to ensure ONE concise sentence."""
    text = text.strip()

    # If there are multiple lines, take the last non-empty one
    # (LLMs often add headers before the actual summary)
    lines = [l.strip() for l in text.split('\n') if l.strip()]
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

    # Keep only the FIRST sentence (truncate at first sentence boundary)
    sentence_match = re.match(r'^[^.!?]+[.!?]', text)
    if sentence_match:
        text = sentence_match.group(0)

    # Enforce word limit
    words = text.split()
    if len(words) > max_words:
        text = ' '.join(words[:max_words])
        if not text.endswith('.'):
            text += '.'

    return text.strip()


def fallback_summarize(text: str, max_words: int = 15) -> str:
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
