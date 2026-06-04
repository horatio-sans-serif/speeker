#!/usr/bin/env python3
"""Tests for Polly text/SSML chunking and PollyEngine concatenation."""

import numpy as np
import pytest

from speeker import polly_chunk as pc
from speeker.polly_chunk import split_payload, billed_len, MAX_BILLED, MAX_TOTAL


class TestPlainText:
    def test_short_text_unchanged(self):
        assert split_payload("Hello world.", False) == ["Hello world."]

    def test_empty(self):
        assert split_payload("   ", False) == []

    def test_long_text_splits_under_budget(self):
        text = ("This is a sentence. " * 400).strip()  # ~8000 chars
        chunks = split_payload(text, False)
        assert len(chunks) > 1
        assert all(len(c) <= MAX_BILLED for c in chunks)
        # No content lost (sentence tokens preserved).
        assert sum(c.count("This is a sentence.") for c in chunks) == 400

    def test_splits_on_sentence_boundaries(self):
        text = ("A" * 2000 + ". ") + ("B" * 2000 + ".")
        chunks = split_payload(text, False)
        assert len(chunks) == 2
        assert chunks[0].startswith("A") and chunks[1].startswith("B")

    def test_giant_single_token_hard_split(self):
        text = "x" * (MAX_BILLED * 2 + 50)
        chunks = split_payload(text, False)
        assert all(len(c) <= MAX_BILLED for c in chunks)
        assert "".join(chunks) == text


class TestSSML:
    def test_short_ssml_unchanged(self):
        s = "<speak><p>Hi.</p></speak>"
        assert split_payload(s, True) == [s]

    def test_billed_len_excludes_tags(self):
        s = "<speak><p>abc</p></speak>"
        assert billed_len(s, True) == 3

    def test_long_ssml_splits_between_paragraphs_each_wrapped(self):
        body = "".join(f"<p>{'word ' * 120}</p>" for _ in range(20))  # big
        s = f"<speak>{body}</speak>"
        chunks = split_payload(s, True)
        assert len(chunks) > 1
        for c in chunks:
            assert c.startswith("<speak>") and c.endswith("</speak>")
            assert billed_len(c, True) <= MAX_BILLED
            assert len(c) <= MAX_TOTAL
            # Tags are never split.
            assert c.count("<speak>") == 1 and c.count("</speak>") == 1

    def test_oversized_single_element_recurses(self):
        # One <p> whose text alone exceeds the billed budget.
        big = "Sentence here. " * 400
        s = f"<speak><p>{big}</p></speak>"
        chunks = split_payload(s, True)
        assert len(chunks) > 1
        for c in chunks:
            assert c.startswith("<speak><p>") and c.endswith("</p></speak>")
            assert billed_len(c, True) <= MAX_BILLED


class TestPollyEngineConcatenation:
    def test_generate_concatenates_chunks(self, monkeypatch):
        from speeker.engines import PollyEngine

        eng = PollyEngine()
        # Force two chunks.
        monkeypatch.setattr(
            "speeker.polly_chunk.split_payload",
            lambda text, is_ssml: ["chunk one", "chunk two"],
        )
        # Each chunk -> 10 samples of a constant so we can verify concat length.
        calls = []

        def fake_chunk(self, text, voice, is_ssml, requested):
            calls.append(text)
            return np.full(10, 0.1, dtype=np.float32)

        monkeypatch.setattr(PollyEngine, "_synthesize_chunk", fake_chunk)
        monkeypatch.setattr("speeker.config.get_polly_config", lambda: {"engine": "neural"})

        audio, sr = eng.generate("long text", "Joanna")
        assert sr == 16000
        assert audio.shape[0] == 20  # two chunks * 10 samples
        assert calls == ["chunk one", "chunk two"]

    def test_generate_empty_text(self, monkeypatch):
        from speeker.engines import PollyEngine
        monkeypatch.setattr("speeker.config.get_polly_config", lambda: {"engine": "neural"})
        audio, sr = PollyEngine().generate("   ", "Joanna")
        assert sr == 16000 and audio.shape[0] == 0
