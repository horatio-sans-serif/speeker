#!/usr/bin/env python3
"""Unit tests for ssml.py."""

from speeker.ssml import (
    looks_like_ssml,
    ensure_speak_wrapped,
    strip_ssml,
    sanitize_ssml,
    escape_text,
    is_well_formed_ssml,
    POLLY_SAFE_TAGS,
    emulate_ssml,
    load_acronyms,
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

    def test_drops_invalid_say_as_attribute(self):
        """Polly rejects say-as with attributes other than interpret-as.
        The sanitizer must drop the unknown attribute; with no interpret-as
        left, the whole tag is dropped and the text content is preserved."""
        out = sanitize_ssml(
            '<speak>The <say-as type="prosody">StepRun</say-as> ran.</speak>'
        )
        assert "type=" not in out
        assert "say-as" not in out  # whole tag dropped (no interpret-as)
        assert "StepRun" in out

    def test_keeps_valid_say_as_drops_extra_attrs(self):
        """Valid interpret-as is kept; unknown sibling attributes are stripped."""
        out = sanitize_ssml(
            '<speak><say-as interpret-as="characters" type="prosody">PHI</say-as></speak>'
        )
        assert 'interpret-as="characters"' in out
        assert "type=" not in out
        assert "PHI" in out

    def test_drops_invalid_prosody_attrs(self):
        """Prosody only accepts rate/pitch/volume — others are stripped."""
        out = sanitize_ssml('<speak><prosody style="bold">x</prosody></speak>')
        assert "style=" not in out
        assert "<prosody>x</prosody>" in out

    def test_drops_sub_without_alias(self):
        """A <sub> without alias is meaningless — drop the tag, keep text."""
        out = sanitize_ssml("<speak><sub>foo</sub></speak>")
        assert "<sub>" not in out
        assert "foo" in out

    def test_auto_closes_nested_p(self):
        """Polly forbids nested <p>. An opener for one while another is open
        must auto-close the previous one — the result should parse and contain
        no nesting."""
        out = sanitize_ssml("<speak><p>first <p>second</p></speak>")
        assert is_well_formed_ssml(out)
        # No nested p in the output — every <p> must be closed before another.
        import re
        depth = 0
        for m in re.finditer(r"</?p\b[^>]*>", out):
            if m.group(0).startswith("</"):
                depth -= 1
            else:
                depth += 1
            assert depth <= 1, f"nested <p> at {m.start()}: {out!r}"

    def test_auto_closes_nested_s(self):
        out = sanitize_ssml("<speak><s>one <s>two</s></speak>")
        assert is_well_formed_ssml(out)

    def test_neural_strips_emphasis(self):
        """Neural engine rejects <emphasis>. With polly_engine='neural', the
        sanitizer drops it but keeps the inner text."""
        out = sanitize_ssml(
            '<speak>Hi <emphasis level="strong">there</emphasis>.</speak>',
            polly_engine="neural",
        )
        assert "<emphasis" not in out
        assert "</emphasis>" not in out
        assert "there" in out

    def test_long_form_keeps_emphasis(self):
        """Long-form supports <emphasis> — don't strip it."""
        out = sanitize_ssml(
            '<speak>Hi <emphasis level="strong">there</emphasis>.</speak>',
            polly_engine="long-form",
        )
        assert "<emphasis" in out

    def test_neural_strips_prosody_volume(self):
        """Neural rejects prosody volume — strip just that attribute, keep tag."""
        out = sanitize_ssml(
            '<speak><prosody rate="95%" volume="loud">x</prosody></speak>',
            polly_engine="neural",
        )
        assert 'rate="95%"' in out
        assert "volume=" not in out

    def test_neural_strips_amazon_namespace_tags(self):
        out = sanitize_ssml(
            '<speak><amazon:domain name="news">x</amazon:domain></speak>',
            polly_engine="neural",
        )
        assert "amazon:domain" not in out
        assert "x" in out



class TestLoadAcronyms:
    def test_builtin_present(self):
        acr = load_acronyms()
        assert "PHI" in acr

    def test_file_all_separators(self, tmp_path):
        f = tmp_path / "acr.txt"
        f.write_text("EHR,EMR|HL7;FHIR ICD")
        acr = load_acronyms(str(f))
        for token in ("EHR", "EMR", "HL7", "FHIR", "ICD"):
            assert token in acr

    def test_missing_file_returns_builtin(self, tmp_path):
        acr = load_acronyms(str(tmp_path / "nope.txt"))
        assert "PHI" in acr


class TestEmulateSsml:
    def test_say_as_characters_spells_out(self):
        out = emulate_ssml('<say-as interpret-as="characters">PHI</say-as>')
        assert out == "P-H-I"

    def test_sub_uses_alias(self):
        out = emulate_ssml('<sub alias="World Wide Web">WWW</sub>')
        assert out == "World Wide Web"

    def test_break_becomes_punctuation(self):
        out = emulate_ssml('Hello<break time="500ms"/>world')
        assert "Hello." in out and "world" in out

    def test_known_acronym_spelled(self):
        out = emulate_ssml("Patient PHI today")
        assert "P-H-I" in out

    def test_unknown_caps_normalized(self):
        out = emulate_ssml("Please STOP now")
        assert "STOP" not in out
        assert "Stop" in out

    def test_other_tags_dropped_text_kept(self):
        out = emulate_ssml("<emphasis>really</emphasis> good")
        assert out == "really good"


class TestIsWellFormedSsml:
    def test_valid_ssml(self):
        assert is_well_formed_ssml("<speak>hi</speak>") is True

    def test_unclosed_tag(self):
        assert is_well_formed_ssml("<speak><p>hi</speak>") is False

    def test_unescaped_ampersand(self):
        assert is_well_formed_ssml("<speak>Tom & Jerry</speak>") is False

    def test_balanced_nested(self):
        ssml = '<speak><prosody rate="95%"><p>Hi.</p></prosody></speak>'
        assert is_well_formed_ssml(ssml) is True


class TestSanitizeBalancesTags:
    """sanitize_ssml must close any container tags the LLM left open.
    Without this, Polly rejects truncated LLM output with InvalidSsmlException."""

    def test_unclosed_p_gets_closed(self):
        out = sanitize_ssml("<speak><p>truncated mid-paragraph</speak>")
        assert is_well_formed_ssml(out)
        assert out.endswith("</p></speak>")

    def test_unclosed_nested_tags(self):
        out = sanitize_ssml('<speak><prosody rate="95%"><p>hi</speak>')
        assert is_well_formed_ssml(out)
        # close in reverse-open order: </p> then </prosody>
        assert "</p></prosody></speak>" in out

    def test_no_close_for_self_closing(self):
        out = sanitize_ssml('<speak>hi<break time="500ms"/></speak>')
        assert is_well_formed_ssml(out)
        # break should not be added to the open stack
        assert "</break>" not in out

    def test_already_balanced_unchanged(self):
        ssml = "<speak><p>hi</p></speak>"
        out = sanitize_ssml(ssml)
        assert is_well_formed_ssml(out)
        # idempotent
        assert sanitize_ssml(out) == out
