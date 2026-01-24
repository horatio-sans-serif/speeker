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
import os
import re
import urllib.request
import urllib.error
from typing import Any

# Configuration
LLM_BACKEND = os.environ.get("SPEEKER_LLM_BACKEND", "").lower()
LLM_ENDPOINT = os.environ.get("SPEEKER_LLM_ENDPOINT", "")
LLM_API_KEY = os.environ.get("SPEEKER_LLM_API_KEY", "")
LLM_MODEL = os.environ.get("SPEEKER_LLM_MODEL", "")

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

SUMMARIZE_PROMPT = """Summarize this text for text-to-speech in 1-2 short sentences (max 30 words).

Rules:
- No file paths, URLs, or code
- No technical jargon - use plain language
- Focus on WHAT was done, not HOW
- Start with an action verb when possible
- Must be natural to speak aloud

Text to summarize:
{text}

Speakable summary:"""


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
    if LLM_BACKEND:
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
    if LLM_BACKEND == "ollama":
        return call_ollama(prompt)
    elif LLM_BACKEND == "anthropic":
        return call_anthropic(prompt)
    elif LLM_BACKEND == "openai":
        return call_openai(prompt)
    return None


def call_ollama(prompt: str) -> str | None:
    """Call Ollama API."""
    endpoint = LLM_ENDPOINT or DEFAULT_ENDPOINTS["ollama"]
    model = LLM_MODEL or DEFAULT_MODELS["ollama"]

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
    if not LLM_API_KEY:
        return None

    endpoint = LLM_ENDPOINT or DEFAULT_ENDPOINTS["anthropic"]
    model = LLM_MODEL or DEFAULT_MODELS["anthropic"]

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
            "x-api-key": LLM_API_KEY,
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
    if not LLM_API_KEY:
        return None

    endpoint = LLM_ENDPOINT or DEFAULT_ENDPOINTS["openai"]
    model = LLM_MODEL or DEFAULT_MODELS["openai"]

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
            "Authorization": f"Bearer {LLM_API_KEY}",
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
    """Clean up LLM response."""
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
    ]
    text_lower = text.lower()
    for prefix in prefixes:
        if text_lower.startswith(prefix):
            text = text[len(prefix):].strip()
            text_lower = text.lower()

    # Remove leading dashes or bullets
    text = re.sub(r'^[-•*]\s*', '', text)

    # Enforce word limit
    words = text.split()
    if len(words) > max_words:
        text = ' '.join(words[:max_words])
        if not text.endswith('.'):
            text += '.'

    return text.strip()


def fallback_summarize(text: str, max_words: int = 30) -> str:
    """Simple fallback summarization without LLM."""
    # Remove code blocks
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'`[^`]+`', '', text)

    # Remove file paths
    text = re.sub(r'[/~][a-zA-Z0-9_./-]+', '', text)

    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)

    # Remove markdown
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[-*]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\|[^|]*\|', ' ', text)

    # Clean whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    if not text or len(text) < 10:
        return "Task completed"

    # Take first sentences up to word limit
    sentences = re.split(r'(?<=[.!?])\s+', text)
    words = []
    for sentence in sentences:
        sentence_words = sentence.split()
        if len(words) + len(sentence_words) <= max_words:
            words.extend(sentence_words)
        else:
            remaining = max_words - len(words)
            if remaining > 3:
                words.extend(sentence_words[:remaining])
            break

    summary = ' '.join(words)
    if summary and summary[-1] not in '.!?':
        summary = summary.rstrip(',;:') + '.'

    return summary if summary else "Task completed"


def get_backend_info() -> dict[str, Any]:
    """Get information about the configured backend."""
    if not LLM_BACKEND:
        return {
            "configured": False,
            "backend": None,
            "message": "No LLM backend configured. Set SPEEKER_LLM_BACKEND to enable.",
        }

    return {
        "configured": True,
        "backend": LLM_BACKEND,
        "endpoint": LLM_ENDPOINT or DEFAULT_ENDPOINTS.get(LLM_BACKEND, ""),
        "model": LLM_MODEL or DEFAULT_MODELS.get(LLM_BACKEND, ""),
        "has_api_key": bool(LLM_API_KEY),
    }
