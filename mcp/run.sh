#!/bin/bash
# Run the Speeker TTS MCP server
cd "$(dirname "$0")"
source .venv/bin/activate
exec python server.py
