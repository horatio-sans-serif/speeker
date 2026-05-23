#!/usr/bin/env python3
"""Unit tests for ssml_generate.py."""

from speeker.ssml_generate import (
    rule_based_ssml,
    PURPOSE_PRESETS,
    PURPOSE_ALIASES,
)


def _wrapped(s: str) -> bool:
    return s.startswith("<speak>") and s.endswith("</speak>")


class TestPresets:
    def test_expected_purposes_present(self):
        for p in ("audiobook", "article", "announcement", "conversational",
                  "technical", "plain"):
            assert p in PURPOSE_PRESETS

    def test_news_alias(self):
        assert PURPOSE_ALIASES["news"] == "article"


class TestRuleBasedSsml:
    def test_audiobook_structure(self):
        out = rule_based_ssml("Para one.\n\nPara two.", "audiobook")
        assert _wrapped(out)
        assert '<prosody rate="95%">' in out
        assert out.count("<p>") == 2
        assert '<break time="800ms"/>' in out

    def test_plain_has_no_prosody(self):
        out = rule_based_ssml("Just text.", "plain")
        assert _wrapped(out)
        assert "<prosody" not in out

    def test_announcement_emphasizes_first(self):
        out = rule_based_ssml("Big news. Details.", "announcement")
        assert "<emphasis" in out
        assert "<break" in out

    def test_technical_spells_acronyms(self):
        out = rule_based_ssml("The PHI record.", "technical")
        assert 'interpret-as="characters"' in out
        assert "PHI" in out

    def test_news_alias_resolves(self):
        out = rule_based_ssml("Hello.", "news")
        assert _wrapped(out)

    def test_escapes_specials(self):
        out = rule_based_ssml("Tom & Jerry", "audiobook")
        assert "Tom &amp; Jerry" in out
