#!/usr/bin/env python3
"""Tests for the MCP server tool wrappers (call_speeker mocked)."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# The MCP server lives outside the package; add its dir to the path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mcp"))
pytest.importorskip("mcp")

import server as mcp_server  # noqa: E402


class TestSpeakTool:
    def test_ssml_and_polly_payload(self):
        captured = {}

        def fake_call(endpoint, data):
            captured["endpoint"] = endpoint
            captured["data"] = data
            return {"status": "success", "queue_id": 1, "pending_count": 1}

        with patch.object(mcp_server, "call_speeker", side_effect=fake_call):
            mcp_server.speak(
                "<speak>hi</speak>", engine="polly", polly_engine="long-form",
                polly_voice="Danielle", ssml=True, queue="q1",
            )
        assert captured["endpoint"] == "/speak"
        data = captured["data"]
        assert data["ssml"] is True
        assert data["metadata"]["engine"] == "polly"
        assert data["metadata"]["voice"] == "Danielle"
        assert data["metadata"]["polly_engine"] == "long-form"
        assert data["metadata"]["queue"] == "q1"


class TestGenerateSsmlTool:
    def test_posts_to_ssml(self):
        captured = {}

        def fake_call(endpoint, data):
            captured["endpoint"] = endpoint
            captured["data"] = data
            return {"status": "success", "ssml": "<speak>hi</speak>", "purpose": "audiobook"}

        with patch.object(mcp_server, "call_speeker", side_effect=fake_call):
            out = mcp_server.generate_ssml("hello", purpose="audiobook")
        assert captured["endpoint"] == "/ssml"
        assert captured["data"] == {"text": "hello", "purpose": "audiobook"}
        assert out["ssml"] == "<speak>hi</speak>"
