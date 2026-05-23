#!/usr/bin/env python3
"""Unit tests for ssml.py."""

from speeker.ssml import (
    looks_like_ssml,
    ensure_speak_wrapped,
    strip_ssml,
    sanitize_ssml,
    escape_text,
    POLLY_SAFE_TAGS,
)


class TestLooksLikeSsml:
    def test_speak_wrapper(self):
        assert looks_like_ssml("<speak>hi</speak>") is True

    def test_leading_whitespace(self):
        assert looks_like_ssml("   <speak>hi</speak>") is True

    def test_case_insensitive(self):
        assert looks_like_ssml("<SPEAK>hi</SPEAK>") is True

    def test_plain_text(self):
        assert looks_like_ssml("hello world") is False


class TestEnsureSpeakWrapped:
    def test_wraps_plain(self):
        assert ensure_speak_wrapped("hi") == "<speak>hi</speak>"

    def test_leaves_wrapped(self):
        assert ensure_speak_wrapped("<speak>hi</speak>") == "<speak>hi</speak>"


class TestStripSsml:
    def test_removes_tags_keeps_text(self):
        assert strip_ssml("<speak>Hello <break/>world</speak>") == "Hello world"

    def test_unescapes_entities(self):
        assert strip_ssml("<speak>a &amp; b</speak>") == "a & b"


class TestEscapeText:
    def test_escapes_bare_ampersand(self):
        assert escape_text("a & b") == "a &amp; b"

    def test_preserves_entities(self):
        assert escape_text("a &amp; b") == "a &amp; b"

    def test_escapes_angle_brackets(self):
        assert escape_text("1 < 2 > 0") == "1 &lt; 2 &gt; 0"


class TestSanitizeSsml:
    def test_keeps_allowed_tags(self):
        out = sanitize_ssml('<speak>Hi <break time="500ms"/>there</speak>')
        assert '<break time="500ms"/>' in out
        assert out.startswith("<speak>") and out.endswith("</speak>")

    def test_drops_disallowed_tags_keeps_text(self):
        out = sanitize_ssml("<speak>Hello <script>x</script>world</speak>")
        assert "<script>" not in out
        assert "Hello" in out and "world" in out and "x" in out

    def test_wraps_unwrapped_input(self):
        out = sanitize_ssml("just text")
        assert out == "<speak>just text</speak>"

    def test_single_speak_root_when_nested(self):
        out = sanitize_ssml("<speak><speak>hi</speak></speak>")
        assert out.count("<speak>") == 1
        assert out.count("</speak>") == 1

    def test_escapes_stray_ampersand_in_text(self):
        out = sanitize_ssml("<speak>Tom & Jerry</speak>")
        assert "Tom &amp; Jerry" in out

    def test_survives_malformed_input(self):
        out = sanitize_ssml("<speak>a <b roken tag")
        assert out.startswith("<speak>") and out.endswith("</speak>")

    def test_polly_safe_tags_present(self):
        assert "prosody" in POLLY_SAFE_TAGS
        assert "say-as" in POLLY_SAFE_TAGS
