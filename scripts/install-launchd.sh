#!/bin/bash
# Install speeker-server as a launchd service (auto-start on login)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PLIST_SRC="$SCRIPT_DIR/../etc/com.speeker.server.plist"
PLIST_DST="$HOME/Library/LaunchAgents/com.speeker.server.plist"

# Check if speeker-server is installed
if ! command -v speeker-server &> /dev/null; then
    echo "Error: speeker-server not found. Install with: uv tool install ~/projects/speeker"
    exit 1
fi

# Get the actual path to speeker-server
SPEEKER_SERVER_PATH=$(which speeker-server)

# Create LaunchAgents directory if needed
mkdir -p "$HOME/Library/LaunchAgents"

# Stop existing service if running
if launchctl list | grep -q com.speeker.server; then
    echo "Stopping existing speeker-server..."
    launchctl unload "$PLIST_DST" 2>/dev/null || true
fi

# Copy and customize plist
echo "Installing launchd plist..."
sed "s|__SPEEKER_SERVER_PATH__|$SPEEKER_SERVER_PATH|g" "$PLIST_SRC" > "$PLIST_DST"

# Load the service
echo "Starting speeker-server..."
launchctl load "$PLIST_DST"

echo "Done! speeker-server will now start automatically on login."
echo "Check status: launchctl list | grep speeker"
echo "View logs: tail -f /tmp/speeker-server.log"
