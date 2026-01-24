#!/bin/bash
# Uninstall speeker-server launchd service

PLIST_DST="$HOME/Library/LaunchAgents/com.speeker.server.plist"

if [ -f "$PLIST_DST" ]; then
    echo "Stopping speeker-server..."
    launchctl unload "$PLIST_DST" 2>/dev/null || true
    rm "$PLIST_DST"
    echo "Done! speeker-server launchd service removed."
else
    echo "speeker-server launchd service not installed."
fi
