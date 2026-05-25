#!/bin/bash
# Run the Speeker TTS MCP server.
# Use `uv run` so the environment is resolved from pyproject.toml/uv.lock at the
# script's own location — this survives moving the project (a hardcoded-path
# .venv/bin/activate does not).
cd "$(dirname "$0")"
exec uv run --quiet python server.py
