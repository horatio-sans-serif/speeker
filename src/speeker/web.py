"""Web UI for viewing TTS queue history."""

import os
import shlex
import subprocess
from datetime import datetime
from pathlib import Path

import hashlib
import json
import re
from functools import lru_cache

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from .config import (
    get_config,
    get_effects_config,
    get_polly_config,
    get_pronunciation_overrides,
    get_tone_rules,
    get_tones_config,
    save_config,
)
from .effects import PRESET_DESCRIPTIONS, PRESETS, preset_names
from .interpretations import interpretation_names
from .tone_rules import normalize_rule, validate_rule
from .queue_db import (
    get_all_sessions,
    get_currently_playing,
    get_history,
    get_settings,
    set_settings,
    search,
)
from .voices import POCKET_TTS_VOICES, KOKORO_VOICES, POLLY_VOICES

router = APIRouter()

# $Note tone markers (e.g. "$Eb4") trigger attention tones at playback and are
# consumed by the player; they must not appear in the displayed transcript.
# Mirrors player.NOTE_PATTERN; kept local so the server needn't import the
# player (which pulls in TTS engine deps).
_TONE_TOKEN_RE = re.compile(r"\$[A-Ga-g][b#]?[0-8](?::[0-9]*\.?[0-9]+)?\s*")


def strip_tone_tokens(text: str) -> str:
    """Remove $Note tone markers so they aren't shown in the queue history."""
    return _TONE_TOKEN_RE.sub("", text or "").strip()

# React-based single-page UI. Loaded once; the app fetches data via the
# /api/* endpoints. We use Babel-standalone to compile JSX in-browser so the
# whole thing ships in one Python module with no Node build step
# (--CHOICE: "inline React via CDN" per the design discussion).
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Speeker Queue History</title>
    <style>
        /* --------------------------------------------------------------
           Theme tokens. Switches automatically via
           prefers-color-scheme. The accent is a muted steel-blue rather
           than neon cyan -- it reads as professional UI rather than
           dashboard chrome. Every color used below is one of these.
           -------------------------------------------------------------- */
        :root {
            color-scheme: dark light;

            --bg:           #0c0c12;
            --surface-1:    #16161e;
            --surface-2:    #1c1c26;
            --surface-3:    #0e0e15;   /* form input fill */
            --border:       #232330;
            --border-strong:#33334a;

            --text-1:       rgba(255, 255, 255, 0.95);
            --text-2:       rgba(255, 255, 255, 0.70);
            --text-3:       rgba(255, 255, 255, 0.50);
            --text-mute:    rgba(255, 255, 255, 0.32);

            --accent:        #5eb4d6;
            --accent-strong: #7ec7e3;
            --accent-fg:     #051319;
            --accent-soft:   rgba(94, 180, 214, 0.13);
            --accent-line:   rgba(94, 180, 214, 0.45);

            --success:      #4fae7f;
            --error:        #db5757;
            --warn:         #d4a35c;

            --code-fg:      #e2c07b;
            --code-bg:      #15151f;

            --scrollbar:    rgba(255, 255, 255, 0.20);

            --shadow-1:     0 4px 14px rgba(0, 0, 0, 0.4);
            --shadow-2:     0 6px 18px rgba(0, 0, 0, 0.5);
        }
        @media (prefers-color-scheme: light) {
            :root {
                --bg:           #f4f4f8;
                --surface-1:    #ffffff;
                --surface-2:    #fafbfc;
                --surface-3:    #fafbfc;
                --border:       #e4e5ec;
                --border-strong:#cdd0db;

                --text-1:       rgba(15, 18, 26, 0.92);
                --text-2:       rgba(15, 18, 26, 0.66);
                --text-3:       rgba(15, 18, 26, 0.46);
                --text-mute:    rgba(15, 18, 26, 0.30);

                --accent:        #1c6b8e;
                --accent-strong: #155575;
                --accent-fg:     #ffffff;
                --accent-soft:   rgba(28, 107, 142, 0.10);
                --accent-line:   rgba(28, 107, 142, 0.55);

                --success:      #257554;
                --error:        #b53028;
                --warn:         #9a6f1a;

                --code-fg:      #855900;
                --code-bg:      #f0ede0;

                --scrollbar:    rgba(0, 0, 0, 0.22);

                --shadow-1:     0 2px 6px rgba(20, 25, 40, 0.08);
                --shadow-2:     0 6px 16px rgba(20, 25, 40, 0.12);
            }
        }
        * { box-sizing: border-box; }
        *::-webkit-scrollbar { width: 8px; height: 8px; }
        *::-webkit-scrollbar-track { background: transparent; }
        *::-webkit-scrollbar-thumb { background: var(--scrollbar); border-radius: 4px; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', Roboto, sans-serif;
            font-size: 16px;
            line-height: 1.5;
            margin: 0;
            padding: 24px 28px;
            background: var(--bg);
            color: var(--text-2);
            /* Antialiased on macOS so the slightly larger text renders cleanly. */
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }
        .header {
            display: flex;
            align-items: baseline;
            justify-content: space-between;
            margin-bottom: 20px;
        }
        .header-left {
            display: flex;
            align-items: baseline;
            gap: 15px;
        }
        h1 {
            color: var(--text-1);
            margin: 0;
            font-weight: 600;
        }
        .subtitle {
            color: var(--text-3);
            font-size: 1em;
        }
        .search-box input[type="text"] {
            padding: 9px 12px;
            border: 1px solid var(--border);
            background: var(--surface-3);
            color: var(--text-1);
            border-radius: 6px;
            width: 260px;
            font-size: 14px;
            transition: border-color 0.12s;
        }
        .search-box input[type="text"]:focus {
            outline: none;
            border-color: var(--accent);
        }
        .search-box input[type="text"]::placeholder {
            color: var(--text-mute);
        }
        .cards-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
            gap: 16px;
        }
        .cards-grid.playing .card:not(.playing) {
            opacity: 0.4;
            pointer-events: none;
        }
        .card {
            background: var(--surface-1);
            border: 1px solid var(--border);
            border-left: 4px solid transparent;
            border-radius: 10px;
            padding: 18px;
            display: flex;
            flex-direction: column;
            gap: 12px;
            box-shadow: var(--shadow-1);
            transition: opacity 0.2s, border-color 0.2s, box-shadow 0.2s;
        }
        .card.interp-success { border-left-color: var(--success); }
        .card.interp-error   { border-left-color: var(--error); }
        .card.interp-other   { border-left-color: var(--warn); }
        .card:hover {
            box-shadow: var(--shadow-2);
        }
        .card.playing {
            animation: border-pulse 1.5s ease-in-out infinite;
        }
        @keyframes border-pulse {
            0%, 100% { border-color: var(--accent); }
            50% { border-color: var(--accent-strong); }
        }
        .card.speaking {
            background: var(--surface-1);
            border-color: var(--accent-line);
            animation: speaking-glow 2.2s ease-in-out infinite;
        }
        @keyframes speaking-glow {
            0%, 100% { box-shadow: 0 0 14px var(--accent-soft); }
            50%      { box-shadow: 0 0 24px var(--accent-line); }
        }
        /* --queue-color is set inline per card (and per table row) from the
           per-queue accent in /api/items. The 4px left stripe is the
           card's primary cue for project identity; falling back to the
           border color when no per-queue color is set keeps existing
           visuals on undecorated rows. */
        .card {
            position: relative;
            border-left: 4px solid var(--queue-color, var(--border));
        }
        .card-text {
            color: var(--text-1);
            font-size: 0.95em;
            max-height: 8em;
            overflow-y: auto;
            white-space: pre-wrap;
            word-break: break-word;
            line-height: 1.55;
        }
        /* Single explicit footer: status pill, queue chip, time, then a
           right-justified action cluster (copy / download / play). */
        .card-footer {
            display: flex;
            align-items: center;
            gap: 10px;
            padding-top: 10px;
            margin-top: 8px;
            border-top: 1px solid var(--border);
            flex-wrap: wrap;
        }
        .card-footer-actions {
            display: flex;
            gap: 6px;
            align-items: center;
            margin-left: auto;
        }
        /* Queue chip: monospace code-like pill with the queue's accent
           color as a left border. The full queue name lives in the title=
           attribute so hover reveals it. */
        .queue-chip {
            display: inline-block;
            font-family: 'SF Mono', Menlo, Monaco, Consolas, monospace;
            font-size: 0.78em;
            background: var(--surface-2);
            color: var(--text-2);
            border: 1px solid var(--border);
            border-left: 4px solid var(--queue-color, var(--accent-line));
            padding: 2px 8px;
            border-radius: 4px;
            cursor: help;
            white-space: nowrap;
        }
        /* Full-name variant for the Per-queue overrides view, where the
           whole queue id matters for identification. Allow wrap so long
           UUID-style ids don't blow the column width. */
        .queue-chip-full {
            white-space: normal;
            word-break: break-all;
            cursor: default;
        }
        /* Copy button: ghost circle that pulses to "copied" on success.
           Uses the same 34px footprint as the play / download buttons so
           the action row aligns visually. */
        .copy-btn {
            background: transparent;
            color: var(--text-3);
            border: 1px solid var(--border-strong);
            width: 34px;
            height: 34px;
            border-radius: 50%;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
            transition: background 0.12s, color 0.12s, border-color 0.12s;
        }
        .copy-btn:hover {
            background: var(--surface-2);
            color: var(--text-1);
            border-color: var(--accent-line);
        }
        .copy-btn.copied {
            background: var(--accent-soft);
            color: var(--success);
            border-color: var(--success);
        }
        .copy-btn.small {
            width: 28px;
            height: 28px;
        }
        .copy-btn svg {
            width: 15px;
            height: 15px;
            fill: none;
            stroke: currentColor;
            stroke-width: 2;
        }
        .copy-btn.small svg { width: 13px; height: 13px; }
        .play-btn {
            background: var(--accent);
            color: var(--accent-fg);
            border: none;
            width: 34px;
            height: 34px;
            border-radius: 50%;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
            transition: background 0.12s;
        }
        .play-btn:hover { background: var(--accent-strong); }
        .play-btn:disabled {
            background: var(--surface-2);
            color: var(--text-mute);
            cursor: not-allowed;
        }
        .play-btn svg { width: 15px; height: 15px; fill: currentColor; }

        /* Download glyph -- a circular ghost button matching .play-btn's
           footprint so the two affordances align cleanly side by side. */
        .download-btn {
            background: transparent;
            color: var(--text-3);
            border: 1px solid var(--border-strong);
            width: 34px;
            height: 34px;
            border-radius: 50%;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
            text-decoration: none;
            transition: background 0.12s, color 0.12s, border-color 0.12s;
        }
        .download-btn:hover {
            color: var(--accent);
            border-color: var(--accent-line);
            background: var(--accent-soft);
        }
        .download-btn.disabled,
        .download-btn[aria-disabled="true"] {
            color: var(--text-mute);
            border-color: var(--border);
            cursor: not-allowed;
            pointer-events: none;
        }
        .download-btn svg { width: 15px; height: 15px; stroke: currentColor; fill: none; stroke-width: 2; }
        .history-table .download-btn { width: 28px; height: 28px; }
        .history-table .download-btn svg { width: 13px; height: 13px; }
        .time {
            color: var(--text-3);
            font-size: 0.88em;
            white-space: nowrap;
        }
        .status {
            font-size: 0.85em;
        }
        .status.played {
            color: var(--text-mute);
        }
        .status.pending {
            display: inline-block;
            padding: 2px 9px;
            border-radius: 10px;
            background: var(--accent-soft);
            color: var(--warn);
            font-weight: 500;
        }
        /* "TTS failed" badge that appears next to the (disabled) Play
           button when the daemon gave up retrying. Borrows the .status
           pill geometry and lifts it onto an error tint so it reads as
           a failure rather than just a state label. */
        .tts-error-badge {
            display: inline-block;
            padding: 2px 9px;
            border-radius: 10px;
            background: rgba(220, 70, 70, 0.15);
            color: var(--warn);
            font-size: 0.78em;
            font-weight: 600;
            letter-spacing: 0.02em;
            cursor: help;
            user-select: none;
        }
        /* Built-in marker on the Effects Preset Editor heading; reuses
           the muted accent so it reads as informational, not blocking. */
        .badge-builtin {
            display: inline-block;
            margin-left: 8px;
            padding: 1px 7px;
            border-radius: 8px;
            background: var(--surface-2);
            color: var(--text-3);
            font-size: 0.7em;
            font-weight: 600;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }
        .preset-editor {
            margin-top: 20px;
            padding: 14px;
            background: var(--surface-1);
            border: 1px solid var(--border);
            border-radius: 8px;
        }
        .preset-editor-header { margin-bottom: 10px; }
        .preset-editor-chain { display: flex; flex-direction: column; gap: 8px; }
        .effect-card {
            background: var(--surface-2);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 10px 12px;
        }
        .effect-card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 6px;
        }
        .effect-card-header .effect-name {
            font-weight: 600;
            color: var(--text-1);
            font-family: 'SF Mono', Menlo, Monaco, Consolas, monospace;
            font-size: 0.92em;
        }
        .effect-card-actions { display: flex; gap: 4px; }
        .effect-params {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 6px 14px;
        }
        .effect-param {
            display: grid;
            grid-template-columns: 90px 1fr 70px;
            gap: 6px;
            align-items: center;
            font-size: 0.85em;
        }
        .effect-param label {
            color: var(--text-3);
            font-family: 'SF Mono', Menlo, Monaco, Consolas, monospace;
            font-size: 0.85em;
            text-align: right;
            cursor: help;
        }
        .effect-param input[type="range"] { width: 100%; }
        .effect-param .param-num {
            width: 100%;
            font-family: 'SF Mono', Menlo, Monaco, Consolas, monospace;
            font-size: 0.82em;
            padding: 2px 6px;
            border-radius: 4px;
            background: var(--surface-3);
            border: 1px solid var(--border);
            color: var(--text-1);
        }
        .score {
            color: var(--accent);
            font-size: 0.82em;
            margin-left: 6px;
        }
        .no-results {
            text-align: center;
            color: var(--text-3);
            padding: 48px 20px;
            grid-column: 1 / -1;
        }
        .no-data {
            color: var(--text-mute);
            font-style: italic;
        }
        audio { display: none; }

        /* Topbar: title + main tabs. Title is the calmer typographic
           weight; main tabs own the bright accent in the topbar. */
        .topbar {
            display: flex;
            align-items: center;
            gap: 32px;
            margin-bottom: 28px;
            border-bottom: 1px solid var(--border);
            padding-bottom: 0;
        }
        .topbar h1 {
            color: var(--text-1);
            font-size: 1.4em;
            font-weight: 600;
            letter-spacing: 0.005em;
            margin: 0;
            padding-bottom: 16px;
            padding-top: 4px;
        }
        .tabs {
            display: flex;
            gap: 2px;
            flex: 1;
        }
        .tab {
            background: transparent;
            color: var(--text-3);
            border: none;
            padding: 10px 20px 16px;
            font-size: 15px;
            font-weight: 500;
            cursor: pointer;
            border-bottom: 2px solid transparent;
            transition: color 0.15s, border-color 0.15s;
        }
        .tab:hover { color: var(--text-1); }
        .tab.active {
            color: var(--accent);
            border-bottom-color: var(--accent);
        }
        .tab:focus-visible {
            outline: 2px solid var(--accent-line);
            outline-offset: -1px;
            border-radius: 4px;
        }
        .settings-section {
            background: var(--surface-1);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 24px;
            margin-bottom: 22px;
        }
        .settings-section h2 {
            color: var(--text-1);
            font-size: 1.1em;
            font-weight: 600;
            margin-top: 0;
            margin-bottom: 8px;
        }
        .settings-section .help,
        .section .help {
            color: var(--text-3);
            font-size: 0.9em;
            line-height: 1.55;
            margin: 0 0 18px 0;
        }
        .section .help code,
        .section code {
            background: var(--code-bg);
            color: var(--code-fg);
            padding: 1px 6px;
            border-radius: 3px;
            font-size: 0.92em;
            font-family: 'SF Mono', Menlo, Monaco, Consolas, monospace;
        }
        .field-row {
            display: flex;
            gap: 10px;
            align-items: center;
            margin-bottom: 14px;
        }
        .field-row label {
            min-width: 130px;
            color: var(--text-3);
            font-size: 0.92em;
            font-weight: 500;
        }
        .field-row select, .field-row input[type="text"], .field-row input[type="number"] {
            flex: 1;
            padding: 9px 12px;
            background: var(--surface-3);
            border: 1px solid var(--border);
            color: var(--text-1);
            border-radius: 6px;
            font-size: 15px;
            transition: border-color 0.12s, background 0.12s;
        }
        .field-row input::placeholder { color: var(--text-mute); }
        .field-row select:hover, .field-row input:hover {
            border-color: var(--border-strong);
        }
        .field-row select:focus, .field-row input:focus {
            outline: none;
            border-color: var(--accent);
        }
        .field-row select {
            -webkit-appearance: none;
            appearance: none;
            background-image: linear-gradient(45deg, transparent 50%, var(--text-3) 50%),
                              linear-gradient(135deg, var(--text-3) 50%, transparent 50%);
            background-position: calc(100% - 16px) 50%, calc(100% - 11px) 50%;
            background-size: 5px 5px;
            background-repeat: no-repeat;
            padding-right: 30px;
        }

        .btn {
            background: var(--accent);
            color: var(--accent-fg);
            border: none;
            padding: 9px 18px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14.5px;
            font-weight: 600;
            transition: background 0.12s, transform 0.06s;
        }
        .btn:hover { background: var(--accent-strong); }
        .btn:active { transform: translateY(1px); }
        .btn:disabled {
            background: var(--surface-2);
            color: var(--text-mute);
            cursor: not-allowed;
        }
        .btn.danger { background: var(--error); color: #ffffff; }
        .btn.danger:hover { filter: brightness(1.12); }
        .btn.subtle {
            background: transparent;
            color: var(--text-2);
            border: 1px solid var(--border-strong);
            font-weight: 500;
        }
        .btn.subtle:hover {
            background: var(--surface-2);
            color: var(--text-1);
            border-color: var(--accent-line);
        }
        .btn:focus-visible {
            outline: 2px solid var(--accent-line);
            outline-offset: 2px;
        }
        .pronunciation-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 14.5px;
        }
        .pronunciation-table th {
            text-align: left;
            color: var(--text-3);
            font-weight: 500;
            padding: 8px 10px;
            border-bottom: 1px solid var(--border);
        }
        .pronunciation-table td { padding: 6px 6px; }
        .pronunciation-table input {
            width: 100%;
            background: var(--surface-3);
            border: 1px solid var(--border);
            color: var(--text-1);
            padding: 8px 10px;
            border-radius: 5px;
            font-family: 'SF Mono', Menlo, Monaco, Consolas, monospace;
            font-size: 13.5px;
        }
        .pronunciation-table input:focus {
            outline: none;
            border-color: var(--accent);
        }
        .restart-banner {
            background: var(--accent-soft);
            color: var(--warn);
            padding: 12px 16px;
            border-radius: 8px;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .restart-banner .restart-msg { flex: 1; }
        .ssml-guide {
            font-size: 0.95em;
            line-height: 1.6;
        }
        .ssml-guide code {
            background: var(--code-bg);
            color: var(--code-fg);
            padding: 1px 6px;
            border-radius: 3px;
            font-size: 0.9em;
        }
        .ssml-guide table {
            width: 100%;
            margin-top: 10px;
            border-collapse: collapse;
        }
        .ssml-guide th, .ssml-guide td {
            padding: 8px 12px;
            border-bottom: 1px solid var(--border);
            text-align: left;
        }
        .ssml-guide th { color: var(--text-3); font-weight: 500; }
        .saving-spinner {
            color: var(--text-3);
            font-size: 0.85em;
        }
        .save-success { color: var(--success); font-size: 0.9em; font-weight: 500; }
        .save-error { color: var(--error); font-size: 0.9em; font-weight: 500; }

        /* Both History, Settings, and the per-queue editor use the same
           flex pattern: a fixed-width sidebar in the document flow
           pushes the main column to the right. */
        .layout-history,
        .layout-settings,
        .layout-split {
            display: flex;
            gap: 20px;
            align-items: flex-start;
        }
        /* Sidebar variant for a panel nested INSIDE a section card
           (per-queue's project picker). Same width as the page-level
           sidebar but without the sticky positioning -- a Section can
           be scrolled away cleanly and a sticky sidebar would clip. */
        .sidebar.sidebar-inset {
            position: static;
            max-height: none;
            top: auto;
            padding: 14px 14px;
        }
        .sidebar {
            flex: 0 0 280px;
            background: var(--surface-1);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 18px 16px;
            overflow-y: auto;
            position: sticky;
            top: 16px;
            max-height: calc(100vh - 32px);
            box-sizing: border-box;
        }
        .main {
            flex: 1;
            min-width: 0;
        }
        .sidebar h3 {
            color: var(--text-2);
            font-size: 0.78em;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin: 0 0 10px 0;
        }
        .sidebar-section { margin-bottom: 22px; }
        .view-toggle {
            display: inline-flex;
            border: 1px solid var(--border-strong);
            border-radius: 6px;
            overflow: hidden;
            background: var(--surface-1);
        }
        .view-toggle button {
            background: transparent;
            color: var(--text-2);
            border: none;
            padding: 7px 14px;
            cursor: pointer;
            font-size: 13.5px;
            font-weight: 500;
        }
        .view-toggle button:hover { background: var(--surface-2); color: var(--text-1); }
        .view-toggle button.active { background: var(--accent); color: var(--accent-fg); }

        /* SSML editor: layered <pre> rendering colorized markup behind
           the editable <textarea>. The textarea is transparent so the
           highlighted pre shows through; both share padding and font so
           the cursor stays aligned with the rendered tokens. */
        .ssml-editor {
            position: relative;
            border: 1px solid var(--border);
            border-radius: 8px;
            background: var(--surface-3);
            font-family: 'SF Mono', Menlo, Monaco, Consolas, monospace;
            font-size: 14px;
            line-height: 1.55;
            overflow: hidden;
        }
        .ssml-editor pre,
        .ssml-editor textarea {
            margin: 0;
            padding: 14px 16px;
            font: inherit;
            white-space: pre-wrap;
            word-break: break-word;
            min-height: 180px;
            border: none;
            background: transparent;
            color: transparent;
            caret-color: var(--text-1);
            box-sizing: border-box;
        }
        .ssml-editor pre {
            position: absolute;
            inset: 0;
            pointer-events: none;
            color: var(--text-1);
        }
        .ssml-editor textarea {
            position: relative;
            width: 100%;
            resize: vertical;
            outline: none;
            -webkit-text-fill-color: transparent;
        }
        .ssml-editor:focus-within {
            border-color: var(--accent);
        }
        .ssml-tok-tag      { color: #6cb6e0; }    /* <speak>, </speak> */
        .ssml-tok-attr     { color: #d09b6e; }    /* alphabet="ipa" */
        .ssml-tok-string   { color: #c4d96a; }    /* "..." values */
        .ssml-tok-comment  { color: var(--text-mute); font-style: italic; }
        .ssml-tok-text     { color: var(--text-1); }

        .ssml-feedback {
            margin-top: 10px;
            font-size: 0.92em;
        }
        .ssml-error {
            color: var(--error);
            background: rgba(219, 87, 87, 0.10);
            border-left: 3px solid var(--error);
            padding: 8px 12px;
            border-radius: 4px;
            margin-bottom: 6px;
        }
        .ssml-warning {
            color: var(--warn);
            background: rgba(212, 163, 92, 0.10);
            border-left: 3px solid var(--warn);
            padding: 8px 12px;
            border-radius: 4px;
            margin-bottom: 6px;
        }
        .ssml-ok {
            color: var(--success);
            font-weight: 500;
        }

        /* Restart-needed pill in the top tab bar. Only shows when the
           server reports a startup-cached config change since the
           daemon last booted. */
        .restart-pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 14px;
            border-radius: 999px;
            background: var(--warn);
            color: #1a1207;
            font-size: 13px;
            font-weight: 600;
            border: none;
            cursor: pointer;
            margin-left: 8px;
            transition: filter 0.12s;
        }
        .restart-pill:hover { filter: brightness(1.08); }
        .restart-pill::before {
            content: "●";
            font-size: 10px;
            animation: pill-pulse 1.6s ease-in-out infinite;
        }
        @keyframes pill-pulse {
            0%, 100% { opacity: 0.6; }
            50%      { opacity: 1; }
        }

        /* Project list inside the sidebar -- vertical, filterable. */
        .project-search {
            width: 100%;
            padding: 8px 10px;
            background: var(--surface-3);
            border: 1px solid var(--border);
            color: var(--text-1);
            border-radius: 6px;
            margin-bottom: 10px;
            font-size: 13.5px;
            box-sizing: border-box;
        }
        .project-search:focus { outline: none; border-color: var(--accent); }
        .project-list {
            display: flex;
            flex-direction: column;
            gap: 2px;
            max-height: 320px;
            overflow-y: auto;
        }
        /* Queue picker row: per-queue color is used as ACCENT, not full
           background. A 3px left stripe colors the row at rest; selected
           rows get an 8% tint of the same color plus a brighter stripe.
           --queue-color is set inline per row from the queue's settings.color
           (auto-derived from queue id when no explicit color is set). */
        .project-row {
            display: flex;
            align-items: center;
            padding: 7px 10px;
            border-radius: 5px;
            cursor: pointer;
            color: var(--text-2);
            font-size: 13.5px;
            user-select: none;
            border-left: 3px solid var(--queue-color, transparent);
        }
        .project-row:hover { background: var(--surface-2); color: var(--text-1); }
        .project-row.selected {
            /* Tinted fill from the queue color using color-mix where
               supported (modern browsers); falls back to accent-soft so
               older browsers still show a clear selected state. */
            background: var(--accent-soft);
            background: color-mix(in srgb, var(--queue-color, var(--accent-line)) 14%, transparent);
            color: var(--text-1);
            border-left-width: 4px;
        }
        .project-row .checkbox {
            width: 14px;
            height: 14px;
            border: 1px solid var(--border-strong);
            border-radius: 3px;
            margin-right: 9px;
            flex-shrink: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            background: var(--surface-3);
        }
        .project-row.selected .checkbox {
            /* Use the queue color for the check fill so the affordance
               echoes the row accent. Older browsers without color-mix
               fall back to the accent variable. */
            background: var(--accent);
            background: var(--queue-color, var(--accent));
            border-color: var(--queue-color, var(--accent));
            color: var(--accent-fg);
        }
        .project-row .checkbox::after {
            content: "";
            display: block;
            width: 8px;
            height: 4px;
            border-left: 2px solid currentColor;
            border-bottom: 2px solid currentColor;
            transform: rotate(-45deg) translate(1px, -1px);
            opacity: 0;
        }
        .project-row.selected .checkbox::after { opacity: 1; }
        .project-row .project-name {
            flex: 1;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .project-row .project-count {
            color: var(--text-mute);
            font-size: 0.85em;
            margin-left: 6px;
        }
        .project-row.selected .project-count { color: var(--text-2); }

        /* Inline range calendar -- single month, click-to-set start/end. */
        .calendar {
            background: var(--surface-3);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 12px;
            user-select: none;
        }
        .calendar-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 8px;
        }
        .calendar-header button {
            background: transparent;
            border: 1px solid var(--border-strong);
            color: var(--text-2);
            width: 24px;
            height: 24px;
            border-radius: 4px;
            cursor: pointer;
            padding: 0;
            line-height: 1;
        }
        .calendar-header button:hover { background: var(--surface-2); color: var(--text-1); }
        .calendar-header button:disabled { opacity: 0.3; cursor: not-allowed; }
        .calendar-title { font-size: 0.95em; color: var(--text-1); font-weight: 500; }
        .calendar-grid {
            display: grid;
            grid-template-columns: repeat(7, 1fr);
            gap: 2px;
        }
        .calendar-dow {
            text-align: center;
            font-size: 0.72em;
            color: var(--text-mute);
            padding: 2px 0;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .calendar-day {
            text-align: center;
            padding: 7px 0;
            font-size: 0.9em;
            border-radius: 4px;
            cursor: pointer;
            color: var(--text-2);
            background: transparent;
            border: none;
        }
        .calendar-day:hover:not(:disabled) {
            background: var(--surface-2);
            color: var(--text-1);
        }
        .calendar-day:disabled {
            color: var(--text-mute);
            cursor: not-allowed;
        }
        .calendar-day.outside { color: var(--text-mute); }
        .calendar-day.in-range {
            background: var(--accent-soft);
            color: var(--text-1);
            border-radius: 0;
        }
        .calendar-day.range-start {
            background: var(--accent);
            color: var(--accent-fg);
            border-radius: 4px 0 0 4px;
        }
        .calendar-day.range-end {
            background: var(--accent);
            color: var(--accent-fg);
            border-radius: 0 4px 4px 0;
        }
        .calendar-day.range-start.range-end { border-radius: 4px; }
        .calendar-day.today {
            border: 1px solid var(--accent-line);
        }
        .calendar-footer {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 10px;
            font-size: 0.85em;
            color: var(--text-3);
        }
        .calendar-footer button {
            background: transparent;
            border: none;
            color: var(--accent);
            cursor: pointer;
            font-size: 0.95em;
            padding: 2px 6px;
            font-weight: 500;
        }

        .history-table {
            width: 100%;
            border-collapse: collapse;
            background: var(--surface-1);
            border: 1px solid var(--border);
            border-radius: 10px;
            overflow: hidden;
        }
        .history-table th {
            background: var(--surface-2);
            color: var(--text-3);
            font-weight: 600;
            text-align: left;
            padding: 11px 14px;
            font-size: 0.78em;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            border-bottom: 1px solid var(--border);
        }
        .history-table td {
            padding: 11px 14px;
            border-bottom: 1px solid var(--border);
            font-size: 0.92em;
            vertical-align: top;
            color: var(--text-1);
        }
        /* Per-row queue-color accent on the leading cell. Interpretation
           classes (.row-success / .row-error / .row-other) re-color this
           border to the outcome cue's semantic color, which takes
           visual priority -- the per-queue tint is a passive identity
           cue, the outcome cue is an active signal. */
        .history-table tr td:first-child {
            border-left: 4px solid var(--queue-color, transparent);
        }
        .history-table tr:last-child td { border-bottom: none; }
        .history-table tr.row-speaking td {
            background: var(--accent-soft);
            animation: row-playing-pulse 2.2s ease-in-out infinite;
        }
        .history-table tr.row-speaking td:first-child {
            border-left: 4px solid var(--accent);
        }
        .history-table tr.row-playing td {
            background: var(--accent-soft);
            animation: row-playing-pulse 1.2s ease-in-out infinite;
        }
        .history-table tr.row-playing td:first-child {
            border-left: 4px solid var(--accent-strong);
        }
        @keyframes row-playing-pulse {
            0%, 100% { background-color: var(--accent-soft); }
            50%      { background-color: var(--accent-line); }
        }
        .history-table tr.row-success td:first-child {
            border-left: 4px solid var(--success);
        }
        .history-table tr.row-error td:first-child {
            border-left: 4px solid var(--error);
        }
        .history-table tr.row-other td:first-child {
            border-left: 4px solid var(--warn);
        }
        .history-table .col-date { white-space: nowrap; }
        .history-table .col-text {
            max-width: 480px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .history-table .col-text-expanded { white-space: normal; }
        .history-table .play-btn {
            width: 28px;
            height: 28px;
        }
        .history-table .play-btn svg { width: 13px; height: 13px; }

        /* Collapsible settings sections (legacy -- Settings now uses
           always-expanded sections with a sidebar TOC; the Collapsible
           component is preserved for ad-hoc reuse). */
        .collapsible-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            cursor: pointer;
            user-select: none;
        }
        .collapsible-header h2 { margin: 0; }
        .collapsible-toggle {
            color: rgba(255,255,255,0.5);
            font-size: 1.2em;
            transition: transform 0.15s;
        }
        .collapsible-toggle.open { transform: rotate(90deg); }
        .collapsible-body { margin-top: 12px; }

        .section {
            background: var(--surface-1);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 28px 32px;
        }
        .section-title {
            color: var(--text-1);
            font-size: 1.12em;
            font-weight: 600;
            letter-spacing: 0.005em;
            margin: 0 0 6px 0;
        }
        .section-subtitle {
            color: var(--text-3);
            font-size: 0.9em;
            font-weight: 400;
            margin: 0 0 22px 0;
            padding-bottom: 18px;
            border-bottom: 1px solid var(--border);
        }
        .section-body { margin-top: 0; }
        .section-body > .help:first-child { margin-top: 0; }

        .subtabs {
            display: flex;
            gap: 2px;
            margin-bottom: 24px;
            padding: 4px;
            background: var(--surface-2);
            border: 1px solid var(--border);
            border-radius: 9px;
            overflow-x: auto;
            scrollbar-width: thin;
        }
        .subtab {
            background: transparent;
            color: var(--text-3);
            border: none;
            padding: 9px 16px;
            border-radius: 6px;
            font-size: 14px;
            font-weight: 500;
            white-space: nowrap;
            cursor: pointer;
            transition: background 0.12s, color 0.12s;
        }
        .subtab:hover {
            background: var(--surface-1);
            color: var(--text-1);
        }
        .subtab.active {
            background: var(--accent-soft);
            color: var(--accent);
        }
        .subtab:focus-visible {
            outline: 2px solid var(--accent-line);
            outline-offset: -1px;
        }

        .btn-try {
            background: transparent;
            color: var(--accent);
            border: 1px solid var(--accent-line);
            padding: 7px 14px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.9em;
            font-weight: 500;
            transition: background 0.12s, border-color 0.12s, color 0.12s;
        }
        .btn-try:hover {
            background: var(--accent-soft);
            border-color: var(--accent);
            color: var(--accent-strong);
        }
        .btn-try:disabled { opacity: 0.35; cursor: not-allowed; }
        .btn-try:focus-visible {
            outline: 2px solid var(--accent-line);
            outline-offset: 2px;
        }

        /* Queue history filter bar -- text search, date range, and per-project
           chip toggles. All three filter axes compose (AND), so e.g. selecting
           two projects + a date range + text shows items in either project,
           in that range, matching the text. */
        .filter-bar { margin-bottom: 16px; }
        .filter-row {
            display: flex;
            gap: 10px;
            align-items: center;
            flex-wrap: wrap;
            margin-bottom: 10px;
        }
        .filter-text {
            flex: 1;
            min-width: 220px;
            padding: 9px 12px;
            background: var(--surface-1);
            border: 1px solid var(--border);
            color: var(--text-1);
            border-radius: 6px;
            font-size: 14px;
        }
        .filter-text:focus { outline: none; border-color: var(--accent); }
        .filter-date {
            padding: 8px 10px;
            background: var(--surface-1);
            border: 1px solid var(--border);
            color: var(--text-1);
            border-radius: 6px;
            font-size: 13.5px;
        }
        .filter-date:focus { outline: none; border-color: var(--accent); }
        .filter-label {
            color: var(--text-3);
            font-size: 0.88em;
        }
        .filter-chips {
            display: flex;
            gap: 6px;
            align-items: center;
            flex-wrap: wrap;
            margin-bottom: 10px;
        }
        .chip {
            background: var(--surface-1);
            color: var(--text-2);
            border: 1px solid var(--border-strong);
            padding: 5px 12px;
            border-radius: 999px;
            cursor: pointer;
            font-size: 0.88em;
            transition: all 0.15s;
        }
        .chip:hover { color: var(--text-1); border-color: var(--accent-line); }
        .chip.chip-on {
            background: var(--accent);
            color: var(--accent-fg);
            border-color: var(--accent);
        }
        .chip-count {
            opacity: 0.65;
            font-size: 0.85em;
            margin-left: 4px;
        }
        .chip.chip-on .chip-count { opacity: 0.85; }
        .filter-summary {
            color: var(--text-3);
            font-size: 0.85em;
        }
    </style>

    <!-- React + Babel via CDN. Babel compiles JSX in-browser at first paint;
         no Node build step is required. Fine for a small admin UI; revisit
         if the app grows beyond a few KB of components. -->
    <script crossorigin src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
    <script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
    <script src="https://unpkg.com/@babel/standalone@7/babel.min.js"></script>
</head>
<body>
    <div id="root">Loading...</div>
    <audio id="player"></audio>

    <script type="text/babel" data-presets="env,react">
const { useState, useEffect, useCallback, useMemo, useRef } = React;

// ----- Top-level App: routes between the History and Settings tabs.
function App() {
    const [tab, setTab] = useState(() => {
        const valid = new Set(['history', 'generate', 'settings']);
        const fromHash = window.location.hash.replace(/^#/, '');
        return valid.has(fromHash) ? fromHash : 'history';
    });
    useEffect(() => {
        window.location.hash = tab;
    }, [tab]);

    // Restart-needed pill: polls /api/restart-needed every 10s. The
    // pill only renders when the server reports a startup-cached config
    // change since the daemon's last boot. Clicking POSTs the restart;
    // the next poll clears the flag because the daemon deletes the
    // sentinel on startup.
    const [restartNeeded, setRestartNeeded] = useState(false);
    const [restarting, setRestarting] = useState(false);
    useEffect(() => {
        let cancelled = false;
        const check = async () => {
            try {
                const r = await fetch('/api/restart-needed');
                const d = await r.json();
                if (!cancelled) setRestartNeeded(!!d.required);
            } catch (e) {}
        };
        check();
        const id = setInterval(check, 10000);
        return () => { cancelled = true; clearInterval(id); };
    }, []);
    const onRestart = async () => {
        if (restarting) return;
        setRestarting(true);
        try {
            await fetch('/api/restart-player', { method: 'POST' });
        } catch (e) {}
        // Give the daemon a moment to delete the sentinel, then re-check.
        setTimeout(async () => {
            try {
                const r = await fetch('/api/restart-needed');
                const d = await r.json();
                setRestartNeeded(!!d.required);
            } catch (e) {}
            setRestarting(false);
        }, 2500);
    };

    return (
        <div>
            <div className="topbar">
                <h1>Speeker</h1>
                <div className="tabs">
                    <button className={'tab' + (tab === 'history' ? ' active' : '')} onClick={() => setTab('history')}>Queue History</button>
                    <button className={'tab' + (tab === 'generate' ? ' active' : '')} onClick={() => setTab('generate')}>Generate</button>
                    <button className={'tab' + (tab === 'settings' ? ' active' : '')} onClick={() => setTab('settings')}>Settings</button>
                    {restartNeeded && (
                        <button
                            className="restart-pill"
                            onClick={onRestart}
                            disabled={restarting}
                            title="A configuration change requires the daemon to restart"
                        >
                            {restarting ? 'Restarting...' : 'Restart daemon'}
                        </button>
                    )}
                </div>
            </div>
            {tab === 'history' ? <HistoryView />
                : tab === 'generate' ? <GenerateView />
                : <SettingsView />}
        </div>
    );
}

// ----- Queue History: lists recent items, polls /api/items every 1s.
function HistoryView() {
    const [items, setItems] = useState([]);
    // Filter state. fromDate/toDate are YYYY-MM-DD or ''.
    const [search, setSearch] = useState('');
    const [selectedQueues, setSelectedQueues] = useState(() => new Set());
    const [fromDate, setFromDate] = useState('');
    const [toDate, setToDate] = useState('');
    // View mode: cards or table. Persisted to localStorage.
    const [viewMode, setViewMode] = useState(() => localStorage.getItem('speeker.viewMode') || 'cards');
    useEffect(() => { localStorage.setItem('speeker.viewMode', viewMode); }, [viewMode]);
    const [playingId, setPlayingId] = useState(null);
    // The daemon's "currently speaking" id (vs `playingId` which is the
    // browser <audio> element's manual playback). Updated every poll.
    const [speakingId, setSpeakingId] = useState(null);
    const playerRef = useRef(null);
    // Hash lives in a ref, NOT in React state. If it were state, every poll
    // that detected new items would recreate fetchItems (it'd capture the
    // new hash via useCallback's deps), which would re-fire the useEffect,
    // which would clearInterval/setInterval each cycle. Using a ref keeps
    // fetchItems stable (deps = []) so the interval is created exactly once.
    const hashRef = useRef('');

    // Reuse the legacy <audio> element so playback state survives re-renders.
    useEffect(() => { playerRef.current = document.getElementById('player'); }, []);

    const fetchItems = useCallback(async () => {
        try {
            const resp = await fetch('/api/items');
            const data = await resp.json();
            // speaking_id changes more often than the item list, so we set
            // it on every response regardless of hash.
            setSpeakingId(data.speaking_id ?? null);
            if (data.hash !== hashRef.current) {
                hashRef.current = data.hash;
                setItems(data.items);
            }
        } catch (e) { /* network blip -- keep prior state */ }
    }, []);

    useEffect(() => {
        fetchItems();
        // Poll at 1Hz: the daemon's currently-speaking marker changes per
        // utterance, so 2s felt sluggish.
        const id = setInterval(fetchItems, 1000);
        return () => clearInterval(id);
    }, [fetchItems]);

    // Distinct queue names ordered by frequency (most-active project first
    // so the chips reflect what's actively making noise). Each option
    // carries the queue's per-queue color so ProjectPicker can use it
    // as the row accent (left stripe + selected pill border).
    const queueOptions = useMemo(() => {
        const counts = new Map();
        const colors = new Map();
        for (const it of items) {
            const q = it.queue || 'default';
            counts.set(q, (counts.get(q) || 0) + 1);
            if (it.queue_color && !colors.has(q)) colors.set(q, it.queue_color);
        }
        return [...counts.entries()]
            .sort((a, b) => b[1] - a[1])
            .map(([name, count]) => ({ name, count, color: colors.get(name) || null }));
    }, [items]);

    const toggleQueue = (name) => {
        setSelectedQueues(prev => {
            const next = new Set(prev);
            if (next.has(name)) next.delete(name); else next.add(name);
            return next;
        });
    };

    // Apply the three filter axes (queue, date range, text) over the
    // poll-fed item list. Done client-side because the React app already
    // holds the latest 200 items in memory; for older data the server
    // search endpoint exists but isn't wired into this UI yet.
    const filtered = useMemo(() => {
        const q = search.trim().toLowerCase();
        const fromTs = fromDate ? new Date(fromDate + 'T00:00:00').getTime() : null;
        // "to" is inclusive: anchor at end of selected day.
        const toTs = toDate ? new Date(toDate + 'T23:59:59.999').getTime() : null;
        return items.filter(it => {
            if (selectedQueues.size > 0 && !selectedQueues.has(it.queue || 'default')) return false;
            if (fromTs || toTs) {
                // item.time is the display string; the raw created_at isn't
                // currently in the JSON payload, so re-derive from id ordering
                // instead. Better: have the API send created_at. For now we
                // parse item.time best-effort. Items also have stable id
                // ordering by created_at, but date filtering needs the raw ts.
                const t = it.created_at_ms;
                if (typeof t === 'number') {
                    if (fromTs && t < fromTs) return false;
                    if (toTs && t > toTs) return false;
                }
            }
            if (q) {
                const haystack = (it.text + ' ' + (it.metadata || '')).toLowerCase();
                if (!haystack.includes(q)) return false;
            }
            return true;
        });
    }, [items, search, selectedQueues, fromDate, toDate]);

    const clearFilters = () => {
        setSearch('');
        setSelectedQueues(new Set());
        setFromDate('');
        setToDate('');
    };
    const hasActiveFilter = search.trim() || selectedQueues.size > 0 || fromDate || toDate;

    const onPlay = useCallback((id) => {
        const player = playerRef.current;
        if (!player) return;
        // Detach the old handlers BEFORE changing src, otherwise switching
        // tracks fires a pause event on the previous src which would
        // immediately clear our new playingId. Re-bind after the new
        // play() promise resolves.
        player.onended = null;
        player.onpause = null;
        setPlayingId(id);
        player.src = `/audio/${id}`;
        player.play().then(() => {
            // Bind clearers AFTER the new track starts; intermediate pause
            // events during the src swap no longer affect our state.
            const clear = () => setPlayingId(null);
            player.onended = clear;
            player.onpause = clear;
        }).catch(() => setPlayingId(null));
    }, []);

    return (
        <div className="layout-history">
            {/* SIDEBAR: filters live here, always visible alongside content. */}
            <aside className="sidebar">
                <div className="sidebar-section">
                    <h3>Search</h3>
                    <input
                        type="text"
                        className="filter-text"
                        placeholder="Fuzzy search text..."
                        value={search}
                        onChange={e => setSearch(e.target.value)}
                        style={{ width: '100%', boxSizing: 'border-box' }}
                    />
                </div>
                <div className="sidebar-section">
                    <h3>Date range</h3>
                    <InlineCalendar
                        fromDate={fromDate}
                        toDate={toDate}
                        onChange={(from, to) => { setFromDate(from); setToDate(to); }}
                    />
                </div>
                <div className="sidebar-section">
                    <h3>Projects</h3>
                    <ProjectPicker
                        projects={queueOptions}
                        selected={selectedQueues}
                        onToggle={toggleQueue}
                    />
                </div>
                {hasActiveFilter && (
                    <div className="sidebar-section">
                        <button className="btn subtle" onClick={clearFilters} style={{ width: '100%' }}>
                            Clear all filters
                        </button>
                    </div>
                )}
            </aside>
            {/* MAIN content: view toggle + cards/table. */}
            <main className="main">
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14, gap: 12 }}>
                    <div className="filter-summary">
                        {hasActiveFilter
                            ? <>Showing {filtered.length} of {items.length} item(s)</>
                            : <>{items.length} item(s)</>}
                    </div>
                    <div className="view-toggle" role="tablist" aria-label="View mode">
                        <button
                            className={viewMode === 'cards' ? 'active' : ''}
                            onClick={() => setViewMode('cards')}
                            title="Card view"
                        >Cards</button>
                        <button
                            className={viewMode === 'table' ? 'active' : ''}
                            onClick={() => setViewMode('table')}
                            title="Table view"
                        >Table</button>
                    </div>
                </div>
                {filtered.length === 0 ? (
                    <div className="no-results">
                        {items.length === 0 ? 'No messages yet' : 'No items match your filters'}
                    </div>
                ) : viewMode === 'cards' ? (
                    <div className={'cards-grid' + (playingId ? ' playing' : '')}>
                        {filtered.map(it => (
                            <ItemCard
                                key={it.id} item={it}
                                isPlaying={String(it.id) === String(playingId)}
                                isSpeaking={String(it.id) === String(speakingId)}
                                onPlay={() => onPlay(it.id)}
                            />
                        ))}
                    </div>
                ) : (
                    <HistoryTable
                        items={filtered}
                        speakingId={speakingId}
                        playingId={playingId}
                        onPlay={onPlay}
                    />
                )}
            </main>
        </div>
    );
}

// Vertical multi-select project picker -- a filterable list with
// per-row checkbox + count. Used by Queue History to filter the visible
// items AND by Per-queue overrides to filter the editor's rows. Both
// surfaces feed it the same shape: an array of {name, count} and a
// Set of currently-selected names.
function ProjectPicker({
    projects, selected, onToggle,
    placeholder = 'Filter projects...',
    emptyLabel = 'No projects match',
}) {
    const [text, setText] = useState('');
    const filtered = useMemo(() => {
        const q = text.trim().toLowerCase();
        if (!q) return projects;
        return projects.filter(p => p.name.toLowerCase().includes(q));
    }, [projects, text]);

    return (
        <div>
            <input
                type="text"
                className="project-search"
                placeholder={placeholder}
                value={text}
                onChange={e => setText(e.target.value)}
            />
            <div className="project-list">
                {filtered.length === 0 ? (
                    <div style={{ color: 'var(--text-mute)', fontSize: '0.85em', padding: '6px 0' }}>
                        {emptyLabel}
                    </div>
                ) : filtered.map(({ name, count, color }) => {
                    const sel = selected.has(name);
                    // Color is an ACCENT, not a fill: a 4px left stripe
                    // colors the row, plus a soft 8% tint on selected rows
                    // so the active state is unmistakable but readable.
                    const style = color ? {
                        '--queue-color': color,
                        borderLeftColor: color,
                    } : {};
                    return (
                        <div
                            key={name}
                            className={'project-row' + (sel ? ' selected' : '')}
                            style={style}
                            onClick={() => onToggle(name)}
                            role="checkbox"
                            aria-checked={sel}
                            tabIndex={0}
                            onKeyDown={e => {
                                if (e.key === ' ' || e.key === 'Enter') {
                                    e.preventDefault();
                                    onToggle(name);
                                }
                            }}
                        >
                            <span className="checkbox"></span>
                            <span className="project-name" title={name}>{name}</span>
                            <span className="project-count">{count}</span>
                        </div>
                    );
                })}
            </div>
        </div>
    );
}

// Single-month inline calendar for picking a date range. Disallows future
// dates. Click sets the start; click again (later) sets the end. Clicking
// before the current start resets to a new start.
function InlineCalendar({ fromDate, toDate, onChange }) {
    const today = useMemo(() => {
        const d = new Date();
        d.setHours(0, 0, 0, 0);
        return d;
    }, []);
    const initialMonth = useMemo(() => {
        const anchor = fromDate ? new Date(fromDate + 'T00:00:00') : today;
        return { year: anchor.getFullYear(), month: anchor.getMonth() };
    }, []);
    const [view, setView] = useState(initialMonth);

    const fromD = fromDate ? new Date(fromDate + 'T00:00:00') : null;
    const toD = toDate ? new Date(toDate + 'T00:00:00') : null;

    const monthName = new Date(view.year, view.month, 1).toLocaleString(undefined, { month: 'long', year: 'numeric' });
    const firstOfMonth = new Date(view.year, view.month, 1);
    const startDow = firstOfMonth.getDay();   // 0=Sun
    const daysInMonth = new Date(view.year, view.month + 1, 0).getDate();
    // Render a 6-row grid that includes some prev/next-month "outside" days.
    const cells = [];
    // Leading days from previous month
    const prevMonthLast = new Date(view.year, view.month, 0).getDate();
    for (let i = startDow - 1; i >= 0; i--) {
        cells.push({ day: prevMonthLast - i, outside: true, date: new Date(view.year, view.month - 1, prevMonthLast - i) });
    }
    for (let d = 1; d <= daysInMonth; d++) {
        cells.push({ day: d, outside: false, date: new Date(view.year, view.month, d) });
    }
    while (cells.length < 42) {
        const i = cells.length - (startDow + daysInMonth) + 1;
        cells.push({ day: i, outside: true, date: new Date(view.year, view.month + 1, i) });
    }

    const toISODate = (d) => {
        const y = d.getFullYear();
        const m = String(d.getMonth() + 1).padStart(2, '0');
        const dd = String(d.getDate()).padStart(2, '0');
        return `${y}-${m}-${dd}`;
    };
    const isFuture = (d) => d > today;
    const inRange = (d) => fromD && toD && d >= fromD && d <= toD;
    const isStart = (d) => fromD && d.getTime() === fromD.getTime();
    const isEnd = (d) => toD && d.getTime() === toD.getTime();
    const isToday = (d) => d.getTime() === today.getTime();

    const pickDay = (d) => {
        if (isFuture(d)) return;
        const iso = toISODate(d);
        // Logic: no range yet -> set start.
        //        only start set, picked day >= start -> set end (range complete).
        //        only start set, picked day < start -> restart from this day.
        //        full range -> restart from this day.
        if (!fromD) { onChange(iso, ''); return; }
        if (fromD && !toD) {
            if (d < fromD) onChange(iso, '');
            else onChange(toISODate(fromD), iso);
            return;
        }
        onChange(iso, '');
    };

    const prevMonth = () => setView(v => v.month === 0 ? { year: v.year - 1, month: 11 } : { year: v.year, month: v.month - 1 });
    const nextMonth = () => setView(v => v.month === 11 ? { year: v.year + 1, month: 0 } : { year: v.year, month: v.month + 1 });
    const atFutureBound = view.year === today.getFullYear() && view.month >= today.getMonth();

    const clear = () => onChange('', '');

    const dows = ['Su', 'Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa'];

    return (
        <div className="calendar">
            <div className="calendar-header">
                <button onClick={prevMonth} aria-label="Previous month">‹</button>
                <span className="calendar-title">{monthName}</span>
                <button onClick={nextMonth} aria-label="Next month" disabled={atFutureBound}>›</button>
            </div>
            <div className="calendar-grid">
                {dows.map(d => <div key={d} className="calendar-dow">{d}</div>)}
                {cells.map(({ day, outside, date }, i) => {
                    const cls = [
                        'calendar-day',
                        outside ? 'outside' : '',
                        inRange(date) ? 'in-range' : '',
                        isStart(date) ? 'range-start' : '',
                        isEnd(date) ? 'range-end' : '',
                        isToday(date) ? 'today' : '',
                    ].filter(Boolean).join(' ');
                    return (
                        <button
                            key={i}
                            className={cls}
                            disabled={isFuture(date)}
                            onClick={() => pickDay(date)}
                        >{day}</button>
                    );
                })}
            </div>
            <div className="calendar-footer">
                <span>
                    {fromDate && toDate ? `${fromDate} → ${toDate}` :
                     fromDate ? `${fromDate} (pick end)` :
                     'No date selected'}
                </span>
                {(fromDate || toDate) && <button onClick={clear}>Clear</button>}
            </div>
        </div>
    );
}

// Table view: same item data as the cards, denser.
//
// Column order matches the user's reading flow: what was said comes first,
// then which project, then when, then status, then the play affordance.
// The "When" column is narrow + no-wrap because the date string is fixed
// width ("May 28 13:24") and wrapping it across two lines wastes vertical
// space for no information.
// Truncate a queue id for compact display. Keeps the leading 8 chars and
// adds an ellipsis. The full id is preserved in the parent's title=
// attribute so hovering reveals it.
const QUEUE_TRUNC_LEN = 8;
function truncateQueue(name) {
    if (!name) return '';
    return name.length > QUEUE_TRUNC_LEN ? name.slice(0, QUEUE_TRUNC_LEN) + '…' : name;
}

// Copy-to-clipboard button. Falls back to a textarea + execCommand when
// the modern clipboard API isn't available (non-https contexts). Shows
// a brief "Copied" affordance so the user knows it worked without a
// noisy toast.
function CopyBtn({ text, small }) {
    const [copied, setCopied] = useState(false);
    const doCopy = async (e) => {
        e.preventDefault();
        e.stopPropagation();
        const payload = (text || '').trim();
        if (!payload) return;
        try {
            if (navigator.clipboard && window.isSecureContext) {
                await navigator.clipboard.writeText(payload);
            } else {
                const ta = document.createElement('textarea');
                ta.value = payload;
                ta.setAttribute('readonly', '');
                ta.style.position = 'absolute';
                ta.style.left = '-9999px';
                document.body.appendChild(ta);
                ta.select();
                document.execCommand('copy');
                document.body.removeChild(ta);
            }
            setCopied(true);
            setTimeout(() => setCopied(false), 1200);
        } catch (_err) {
            // Silently ignore: the worst case is a no-op button, which
            // is better than an exception during a quick paste workflow.
        }
    };
    return (
        <button
            type="button"
            className={'copy-btn' + (small ? ' small' : '') + (copied ? ' copied' : '')}
            onClick={doCopy}
            title={copied ? 'Copied' : 'Copy text'}
            aria-label="Copy spoken text"
        >
            {copied ? (
                <svg viewBox="0 0 24 24" aria-hidden="true">
                    <path d="M5 13l4 4L19 7" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
            ) : (
                <svg viewBox="0 0 24 24" aria-hidden="true">
                    <rect x="9" y="9" width="11" height="11" rx="1.5"/>
                    <path d="M5 15V5a1 1 0 011-1h10" strokeLinecap="round" strokeLinejoin="round" fill="none"/>
                </svg>
            )}
        </button>
    );
}

function HistoryTable({ items, speakingId, playingId, onPlay }) {
    return (
        <table className="history-table">
            <thead>
                <tr>
                    <th>Text</th>
                    <th style={{ width: 130 }}>Project</th>
                    <th className="col-date" style={{ width: 110 }}>When</th>
                    <th style={{ width: 80 }}>Status</th>
                    <th style={{ width: 130 }}></th>
                </tr>
            </thead>
            <tbody>
                {items.map(it => {
                    const isSpeaking = String(it.id) === String(speakingId);
                    const isPlaying = String(it.id) === String(playingId);
                    const rowCls = [
                        // row-speaking wins visually over row-playing because
                        // it's a global signal (daemon is reading aloud right
                        // now), whereas row-playing is local in-browser audio
                        // playback. Both classes can be present; CSS specificity
                        // and the gradient overlay make the distinction clear.
                        isSpeaking ? 'row-speaking' : '',
                        isPlaying ? 'row-playing' : '',
                        it.interp_class === 'interp-success' ? 'row-success' :
                            it.interp_class === 'interp-error' ? 'row-error' :
                                it.interp_class === 'interp-other' ? 'row-other' : '',
                    ].filter(Boolean).join(' ');
                    const status = isSpeaking ? 'Speaking' : (it.played ? 'Played' : 'Pending');
                    // Row-level CSS variable carries the queue's accent color
                    // to a ::before stripe selector defined in the stylesheet.
                    const rowStyle = it.queue_color
                        ? { '--queue-color': it.queue_color }
                        : {};
                    return (
                        <tr key={it.id} className={rowCls} style={rowStyle}>
                            <td className="col-text" title={it.text} dangerouslySetInnerHTML={{ __html: it.text }} />
                            <td>
                                <code
                                    className="queue-chip"
                                    title={it.queue}
                                    style={it.queue_color ? { borderLeftColor: it.queue_color } : {}}
                                >{truncateQueue(it.queue)}</code>
                            </td>
                            <td className="col-date">{it.time}</td>
                            <td>{status}</td>
                            <td>
                                <div style={{ display: 'flex', gap: 4, alignItems: 'center', justifyContent: 'flex-end' }}>
                                    {it.tts_error && (
                                        <span
                                            className="tts-error-badge"
                                            title={'TTS failed: ' + it.tts_error}
                                        >TTS failed</span>
                                    )}
                                    <CopyBtn text={it.raw_text} small />
                                    <DownloadBtn itemId={it.id} hasAudio={it.has_audio} small />
                                    <button
                                        className="play-btn"
                                        onClick={() => onPlay(it.id)}
                                        disabled={!it.has_audio}
                                        title={it.has_audio
                                            ? 'Play'
                                            : (it.tts_error ? 'TTS failed: ' + it.tts_error : 'No audio')}
                                    >
                                        <svg viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>
                                    </button>
                                </div>
                            </td>
                        </tr>
                    );
                })}
            </tbody>
        </table>
    );
}

// History card: text on top, single footer row carrying everything else
// (status pill, queue chip, time, action buttons). Header/footer split
// was removed -- the time and status were visually homeless at the top
// and a single explicit footer makes the action affordances cluster
// where the eye expects them. The queue's accent color drives a 4px
// left stripe via the --queue-color CSS variable.
function ItemCard({ item, isPlaying, isSpeaking, onPlay }) {
    const interpClass = item.interp_class || '';
    const cls = [
        'card',
        isPlaying ? 'playing' : '',
        isSpeaking ? 'speaking' : '',
        interpClass,
    ].filter(Boolean).join(' ');
    const statusClass = item.played ? 'played' : 'pending';
    const statusText = isSpeaking ? 'Speaking' : (item.played ? 'Played' : 'Pending');
    const cardStyle = item.queue_color ? { '--queue-color': item.queue_color } : {};
    return (
        <div className={cls} data-id={item.id} data-interp={interpClass} style={cardStyle}>
            <div className="card-text" dangerouslySetInnerHTML={{ __html: item.text }} />
            <div className="card-footer">
                <span className={'status ' + statusClass}>{statusText}</span>
                <code
                    className="queue-chip"
                    title={item.queue}
                    style={item.queue_color ? { borderLeftColor: item.queue_color } : {}}
                >{truncateQueue(item.queue)}</code>
                <span className="time">{item.time}</span>
                <div className="card-footer-actions">
                    {item.tts_error && (
                        <span
                            className="tts-error-badge"
                            title={'TTS failed: ' + item.tts_error}
                        >TTS failed</span>
                    )}
                    <CopyBtn text={item.raw_text} />
                    <DownloadBtn itemId={item.id} hasAudio={item.has_audio} />
                    <button
                        className="play-btn"
                        onClick={onPlay}
                        disabled={!item.has_audio}
                        title={item.has_audio
                            ? 'Play'
                            : (item.tts_error ? 'TTS failed: ' + item.tts_error : 'No audio')}
                    >
                        <svg viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>
                    </button>
                </div>
            </div>
        </div>
    );
}

// Download glyph button. Right-click for "Save link as..." gives the
// MP3 variant via the same endpoint with ?format=mp3 -- power users
// can also hit /download/<id>?format=mp3 directly. The visible action
// downloads the WAV with a friendly filename.
function DownloadBtn({ itemId, hasAudio, small }) {
    if (!hasAudio) {
        return (
            <a
                className={'download-btn disabled' + (small ? ' history-download' : '')}
                aria-disabled="true"
                title="No audio to download"
            >
                <svg viewBox="0 0 24 24" aria-hidden="true">
                    <path d="M12 4v12m0 0l-5-5m5 5l5-5M5 20h14" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
            </a>
        );
    }
    return (
        <a
            className="download-btn"
            href={`/download/${itemId}?format=wav`}
            download
            title="Download WAV (right-click for more options)"
            onContextMenu={e => {
                // Tiny enhancement: a left-click downloads WAV, a
                // right-click could pop a native menu; offer MP3 via a
                // ctrl/alt+click shortcut so power users have a one-click
                // path without a dropdown component.
            }}
            onAuxClick={e => {
                if (e.button === 1) {
                    // Middle-click downloads MP3 instead.
                    window.location.href = `/download/${itemId}?format=mp3`;
                }
            }}
        >
            <svg viewBox="0 0 24 24" aria-hidden="true">
                <path d="M12 4v12m0 0l-5-5m5 5l5-5M5 20h14" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
        </a>
    );
}

// ----- Generate tab: ad-hoc TTS. Pick engine + voice, type text, hit
// Generate. Server synthesizes synchronously and returns audio URLs;
// no playback through system speakers (so it's safe to generate when
// the user wants a file but not a sermon). Result appears in queue
// history under the "generated" project.
function GenerateView() {
    const [engines, setEngines] = useState([]);
    const [globalSettings, setGlobalSettings] = useState(null);
    const [engine, setEngine] = useState('');
    const [voice, setVoice] = useState('');
    const [speed, setSpeed] = useState(1.0);
    const [text, setText] = useState('');
    const [result, setResult] = useState(null);    // {queue_id, audio_url, download_wav, download_mp3, size_bytes, ...}
    const [busy, setBusy] = useState(false);
    const [error, setError] = useState('');

    // Initial load: engines + saved global defaults so the dropdowns
    // mirror what the user expects from Settings.
    useEffect(() => {
        fetch('/api/engines').then(r => r.json()).then(d => {
            setEngines(d.engines || []);
        });
        fetch('/api/settings/global').then(r => r.json()).then(d => {
            setGlobalSettings(d);
            if (d.engine) setEngine(d.engine);
            if (d.voice) setVoice(d.voice);
            if (typeof d.speed === 'number') setSpeed(d.speed);
        });
    }, []);

    const engineMeta = useMemo(
        () => engines.find(e => e.name === engine),
        [engines, engine],
    );
    const voices = engineMeta ? engineMeta.voices : [];

    // When the user picks a different engine, reset the voice to that
    // engine's default rather than carrying the old engine's selection
    // (which wouldn't even exist in the new engine's voice list).
    useEffect(() => {
        if (!engineMeta) return;
        if (!voices.find(v => v.id === voice)) {
            setVoice(engineMeta.default_voice || (voices[0] && voices[0].id) || '');
        }
    }, [engineMeta]);

    const onGenerate = async () => {
        if (busy) return;
        const t = text.trim();
        if (!t) {
            setError('Type something to synthesize.');
            return;
        }
        setError('');
        setResult(null);
        setBusy(true);
        try {
            const resp = await fetch('/api/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ engine, voice, speed, text: t }),
            });
            if (!resp.ok) {
                const err = await resp.json().catch(() => ({}));
                setError('Error: ' + (err.detail || resp.statusText));
                return;
            }
            const data = await resp.json();
            setResult(data);
        } catch (e) {
            setError('Failed: ' + e.message);
        } finally {
            setBusy(false);
        }
    };

    if (!globalSettings || engines.length === 0) return <div>Loading...</div>;

    const charCount = text.length;
    const charBudget = 5000;
    const overBudget = charCount > charBudget;

    return (
        <div className="section" style={{ maxWidth: 920, margin: '0 auto' }}>
            <h2 className="section-title">Generate</h2>
            <p className="section-subtitle">
                Synthesize text with a specific voice and download the audio.
                Nothing plays through your speakers.
            </p>

            <div className="field-row">
                <label>Engine</label>
                <select value={engine} onChange={e => setEngine(e.target.value)}>
                    {engines.map(en => (
                        <option key={en.name} value={en.name}>
                            {en.label}{en.supports_ssml ? '  [SSML]' : ''}
                        </option>
                    ))}
                </select>
            </div>
            <div className="field-row">
                <label>Voice</label>
                <select value={voice} onChange={e => setVoice(e.target.value)}>
                    {voices.map(v => (
                        <option key={v.id} value={v.id}>{v.id} - {v.label}</option>
                    ))}
                </select>
            </div>
            <div className="field-row">
                <label>Speed</label>
                <input
                    type="number"
                    min="0.5" max="2.0" step="0.05"
                    value={speed}
                    onChange={e => setSpeed(parseFloat(e.target.value) || 1.0)}
                />
            </div>

            <div style={{ marginTop: 16 }}>
                <label
                    style={{
                        display: 'block',
                        color: 'var(--text-3)',
                        fontSize: '0.92em',
                        fontWeight: 500,
                        marginBottom: 6,
                    }}
                >
                    Text
                </label>
                <textarea
                    value={text}
                    onChange={e => setText(e.target.value)}
                    rows={10}
                    placeholder="Type or paste text..."
                    style={{
                        width: '100%',
                        padding: 12,
                        background: 'var(--surface-3)',
                        border: '1px solid ' + (overBudget ? 'var(--error)' : 'var(--border)'),
                        color: 'var(--text-1)',
                        borderRadius: 6,
                        fontSize: 15,
                        fontFamily: 'inherit',
                        resize: 'vertical',
                        lineHeight: 1.55,
                        boxSizing: 'border-box',
                    }}
                />
                <div style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    marginTop: 6,
                    fontSize: '0.85em',
                    color: overBudget ? 'var(--error)' : 'var(--text-mute)',
                }}>
                    <span>{overBudget ? 'Over the per-request limit.' : ''}</span>
                    <span>{charCount.toLocaleString()} / {charBudget.toLocaleString()} chars</span>
                </div>
            </div>

            <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginTop: 16 }}>
                <button
                    className="btn"
                    onClick={onGenerate}
                    disabled={busy || overBudget || !text.trim()}
                >
                    {busy ? 'Generating...' : 'Generate'}
                </button>
                {error && <span className="save-error">{error}</span>}
                {busy && <span className="save-success">Synthesizing... (Polly is ~1-2s)</span>}
            </div>

            {result && (
                <div style={{
                    marginTop: 24,
                    padding: 20,
                    background: 'var(--surface-2)',
                    border: '1px solid var(--border)',
                    borderRadius: 8,
                }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 12 }}>
                        <strong style={{ color: 'var(--text-1)' }}>Ready</strong>
                        <span className="filter-summary">
                            #{result.queue_id} &middot; {result.engine}/{result.voice} &middot;
                            {' '}{(result.size_bytes / 1024).toFixed(1)} KB &middot;
                            {' '}{result.char_count.toLocaleString()} chars
                        </span>
                    </div>
                    {/* HTML5 native controls: play/pause/seek. The src points
                        at the same /audio/<id> the queue history uses, so
                        the file is shared and not re-fetched if the user
                        opens it from another tab too. */}
                    <audio
                        src={result.audio_url}
                        controls
                        style={{ width: '100%', marginBottom: 12 }}
                    />
                    <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                        <a className="btn" href={result.download_wav} download>Download WAV</a>
                        <a className="btn subtle" href={result.download_mp3} download>Download MP3</a>
                        <a
                            className="btn subtle"
                            href="#history"
                            style={{ marginLeft: 'auto' }}
                            title="Open the queue history with this generation visible"
                        >Open in history</a>
                    </div>
                </div>
            )}
        </div>
    );
}

// ----- Settings tab: engine + voice selector, pronunciation editor, SSML guide.
//
// Layout is a sub-tab bar above a single rendered section. State for the
// active sub-tab persists to localStorage; only one section's content is
// in the DOM at a time, which keeps the page lighter than the previous
// always-expanded TOC approach.
function SettingsView() {
    const [engines, setEngines] = useState([]);
    const [globalSettings, setGlobalSettings] = useState(null);
    const [saveMsg, setSaveMsg] = useState('');
    const [section, setSection] = useState(() =>
        localStorage.getItem('speeker.settingsSection') || 'engine'
    );
    useEffect(() => {
        localStorage.setItem('speeker.settingsSection', section);
    }, [section]);

    useEffect(() => {
        fetch('/api/engines').then(r => r.json()).then(d => setEngines(d.engines));
        fetch('/api/settings/global').then(r => r.json()).then(setGlobalSettings);
    }, []);

    const onSaveGlobal = useCallback(async (patch) => {
        setSaveMsg('');
        const resp = await fetch('/api/settings', {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session: null, ...patch }),
        });
        const data = await resp.json();
        setGlobalSettings(data);
        // Engine/voice/speed/intro changes are read live from the settings
        // table on the daemon's next utterance, so no restart needed. The
        // "Restart daemon" button stays available as a manual tool for
        // cases that DO require a fresh process (polly.profile, etc.).
        setSaveMsg('Saved. Takes effect on the next utterance.');
        setTimeout(() => setSaveMsg(''), 2500);
    }, []);

    // Daemon restart now lives as a pill in the top tab bar (App
    // component) -- it appears only when the server reports a
    // startup-cached config change, so SettingsView doesn't carry its
    // own restart UI anymore.
    if (!globalSettings || engines.length === 0) return <div>Loading settings...</div>;

    // Map each sub-tab id to (title, subtitle, body). Subtitles replace
    // most of the in-section help paragraphs -- each section gets one
    // short line of context near the title and the form takes the rest.
    const sections = {
        engine: {
            title: 'Engine & voice',
            subtitle: 'Global defaults. Per-queue overrides in their own tab.',
            body: <EngineSection engines={engines} settings={globalSettings} onSave={onSaveGlobal} />,
        },
        pronunciation: {
            title: 'Pronunciation overrides',
            body: <PronunciationSection />,
        },
        tones: {
            title: 'Intro & outro tones',
            subtitle: 'Notes played before and after multi-message batches.',
            body: <TonesSection />,
        },
        effects: {
            title: 'Audio effects',
            subtitle: 'Reverb, compression, EQ on TTS speech. Tones are unaffected.',
            body: <EffectsSection />,
        },
        ssml: {
            title: 'SSML support',
            subtitle: 'Which engines honor speech-synthesis markup.',
            body: <SSMLGuide engines={engines} />,
        },
        'per-queue': {
            title: 'Per-queue overrides',
            subtitle: 'Engine and voice overrides per project.',
            body: <PerQueueSection engines={engines} />,
        },
    };
    const current = sections[section] || sections.engine;

    return (
        <div>
            <SettingsSubTabs current={section} onChange={setSection} />
            {saveMsg && <div className="save-success" style={{ marginBottom: 14 }}>{saveMsg}</div>}
            <Section title={current.title} subtitle={current.subtitle}>
                {current.body}
            </Section>
            {/* Restart-daemon control moves to the top tab bar in task #11 and
                only renders when a startup-cached value has changed. Nothing
                here in the Settings body. */}
        </div>
    );
}

// Settings sections in tab order.
const SETTINGS_SECTIONS = [
    { id: 'engine',        title: 'Engine & voice' },
    { id: 'pronunciation', title: 'Pronunciation' },
    { id: 'tones',         title: 'Tones' },
    { id: 'effects',       title: 'Effects' },
    { id: 'ssml',          title: 'SSML' },
    { id: 'per-queue',     title: 'Per-queue' },
];

// Second-level tab bar for Settings. Sits between the main tab row and the
// active section's content. Pill-style active state at a muted accent
// keeps the main tabs visually primary.
function SettingsSubTabs({ current, onChange }) {
    return (
        <div className="subtabs" role="tablist" aria-label="Settings sections">
            {SETTINGS_SECTIONS.map(s => (
                <button
                    key={s.id}
                    type="button"
                    role="tab"
                    aria-selected={current === s.id}
                    className={'subtab' + (current === s.id ? ' active' : '')}
                    onClick={() => onChange(s.id)}
                >
                    {s.title}
                </button>
            ))}
        </div>
    );
}

// Section panel with a title row and an optional subtitle. The subtitle
// is meant to replace most in-body help paragraphs -- one short line of
// context per section, the rest is forms.
function Section({ title, subtitle, children }) {
    return (
        <section className="section">
            <h2 className="section-title">{title}</h2>
            {subtitle && <p className="section-subtitle">{subtitle}</p>}
            <div className="section-body">{children}</div>
        </section>
    );
}

// Generic collapsible wrapper. Persists open/closed state to localStorage
// per-title so sections stay the way the user left them between visits.
function Collapsible({ title, defaultOpen = false, children }) {
    const key = `speeker.collapsed.${title}`;
    const [open, setOpen] = useState(() => {
        const v = localStorage.getItem(key);
        if (v === null) return defaultOpen;
        return v === '1';
    });
    useEffect(() => { localStorage.setItem(key, open ? '1' : '0'); }, [key, open]);
    return (
        <div className="settings-section">
            <div className="collapsible-header" onClick={() => setOpen(v => !v)}>
                <h2>{title}</h2>
                <span className={'collapsible-toggle' + (open ? ' open' : '')}>›</span>
            </div>
            {open && <div className="collapsible-body">{children}</div>}
        </div>
    );
}

// ----- Engine/voice selector for the global default.
function EngineSection({ engines, settings, onSave }) {
    const [engine, setEngine] = useState(settings.engine || 'polly');
    const [voice, setVoice] = useState(settings.voice || '');
    const [speed, setSpeed] = useState(settings.speed ?? 1.0);
    const [intro, setIntro] = useState(!!settings.intro_sound);
    useEffect(() => {
        setEngine(settings.engine || 'polly');
        setVoice(settings.voice || '');
        setSpeed(settings.speed ?? 1.0);
        setIntro(!!settings.intro_sound);
    }, [settings]);

    const engineMeta = useMemo(() => engines.find(e => e.name === engine), [engines, engine]);
    const voices = engineMeta ? engineMeta.voices : [];

    return (
        <>
            <div className="field-row">
                <label>Engine</label>
                <select value={engine} onChange={e => setEngine(e.target.value)}>
                    {engines.map(en => (
                        <option key={en.name} value={en.name}>
                            {en.label}{en.supports_ssml ? '  [SSML]' : ''}
                        </option>
                    ))}
                </select>
            </div>
            <div className="field-row">
                <label>Voice</label>
                <select value={voice} onChange={e => setVoice(e.target.value)}>
                    {voices.map(v => (
                        <option key={v.id} value={v.id}>{v.id} - {v.label}</option>
                    ))}
                </select>
            </div>
            <div className="field-row">
                <label>Speed</label>
                <input type="number" min="0.5" max="2.0" step="0.05" value={speed}
                    onChange={e => setSpeed(parseFloat(e.target.value))} />
            </div>
            <div className="field-row">
                <label>Intro/outro tones</label>
                <input type="checkbox" checked={intro} onChange={e => setIntro(e.target.checked)}
                    style={{ width: 'auto', flex: 'none' }} />
            </div>
            <div style={{ display: 'flex', gap: 8, marginTop: 12, alignItems: 'center' }}>
                <button
                    className="btn-try"
                    onClick={async () => {
                        // Pass the live intro/outro tones toggle so the
                        // preview reflects the *full* surface the user is
                        // configuring -- voice plus chord wrap. Server
                        // reads the saved intro/outro notes for wrapping.
                        await fetch('/api/engines/try', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                engine, voice, include_tones: intro,
                            }),
                        });
                    }}
                    title="Speak a sample with the selected engine + voice (and intro/outro tones if toggled)"
                >Try it</button>
                <button className="btn" onClick={() => onSave({ engine, voice, speed, intro_sound: intro })}>
                    Save global defaults
                </button>
            </div>
        </>
    );
}

// ----- Pronunciation overrides editor. Rows are added/removed/edited in
// local state; "Save" persists the full mapping to config.json via PUT
// /api/pronunciation. Changes take effect on the next utterance -- the
// daemon re-reads config.json on every preprocess call, so the optional
// ``onChange`` prop is intentionally a no-op for this section.
function PronunciationSection({ onChange }) {
    const [rows, setRows] = useState([]);   // [{word, replacement}]
    const [status, setStatus] = useState('');
    const [loaded, setLoaded] = useState(false);

    useEffect(() => {
        fetch('/api/pronunciation').then(r => r.json()).then(d => {
            const entries = Object.entries(d.overrides || {});
            setRows(entries.map(([word, replacement]) => ({ word, replacement })));
            setLoaded(true);
        });
    }, []);

    const setRow = (i, key, value) => {
        setRows(prev => prev.map((r, j) => j === i ? { ...r, [key]: value } : r));
    };
    const addRow = () => setRows(prev => [...prev, { word: '', replacement: '' }]);
    const removeRow = (i) => setRows(prev => prev.filter((_, j) => j !== i));

    const save = async () => {
        setStatus('Saving...');
        const overrides = {};
        for (const { word, replacement } of rows) {
            if (word.trim() && replacement.trim()) {
                overrides[word.trim()] = replacement.trim();
            }
        }
        try {
            const resp = await fetch('/api/pronunciation', {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ overrides }),
            });
            const data = await resp.json();
            const persisted = Object.entries(data.overrides || {});
            setRows(persisted.map(([word, replacement]) => ({ word, replacement })));
            setStatus(data.message || 'Saved.');
            if (data.restart_required && onChange) onChange();
            setTimeout(() => setStatus(''), 3000);
        } catch (e) {
            setStatus('Save failed: ' + e.message);
        }
    };

    // Try reflects the row's *current* state -- including unsaved edits.
    // Pass the replacement explicitly so the server enqueues exactly what
    // the user has typed, no round-trip through config.json.
    const tryRow = async (row) => {
        const w = (row.word || '').trim();
        const r = (row.replacement || '').trim();
        if (!w) return;
        await fetch('/api/pronunciation/try', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ word: w, replacement: r || null }),
        });
    };

    return (
        <>
            {!loaded ? <div>Loading...</div> : (
                <>
                    <table className="pronunciation-table">
                        <thead>
                            <tr>
                                <th style={{ width: '38%' }}>Word</th>
                                <th>Spoken as</th>
                                <th style={{ width: 70 }}></th>
                                <th style={{ width: 90 }}></th>
                            </tr>
                        </thead>
                        <tbody>
                            {rows.map((row, i) => (
                                <tr key={i}>
                                    <td><input value={row.word}
                                        placeholder="e.g. compass"
                                        onChange={e => setRow(i, 'word', e.target.value)} /></td>
                                    <td><input value={row.replacement}
                                        placeholder="e.g. kom-piss"
                                        onChange={e => setRow(i, 'replacement', e.target.value)} /></td>
                                    <td>
                                        <button
                                            className="btn-try"
                                            onClick={() => tryRow(row)}
                                            disabled={!row.word.trim()}
                                            title="Speak this replacement now (uses unsaved edits if any)"
                                        >Try</button>
                                    </td>
                                    <td><button className="btn subtle" onClick={() => removeRow(i)}>Remove</button></td>
                                </tr>
                            ))}
                            {rows.length === 0 && (
                                <tr><td colSpan={4} style={{ color: 'rgba(255,255,255,0.4)', padding: 12 }}>
                                    No overrides yet. Click "Add row" to start.
                                </td></tr>
                            )}
                        </tbody>
                    </table>
                    <div style={{ display: 'flex', gap: 8, marginTop: 12, alignItems: 'center' }}>
                        <button className="btn subtle" onClick={addRow}>+ Add row</button>
                        <button className="btn" onClick={save}>Save</button>
                        <span className="save-success">{status}</span>
                    </div>
                </>
            )}
        </>
    );
}

// Intro/outro tone configuration. Notes use the inline ``$Note`` notation
// ([A-G][b#]?[0-8]) -- same syntax as the $Eb4 tokens in queued text.
//
// Per role (intro/outro):
//   - input field for the note sequence
//   - Play button that synthesizes the *current* (possibly unsaved)
//     value at the *current* duration -- the preview always reflects
//     what would happen if the user clicked Save
//   - Tunes dropdown to drop in a public-domain melody
function TonesSection() {
    const [intro, setIntro] = useState('');     // space-separated note string
    const [outro, setOutro] = useState('');
    const [duration, setDuration] = useState(0.12);
    const [applyEffects, setApplyEffects] = useState(true);
    const [tunes, setTunes] = useState([]);     // [{name, notes, note}]
    // Tunes dropdown state: kept controlled so React owns the reset
    // (an uncontrolled select with `e.target.value=''` fights React if
    // anyone later adds a `value` prop). Always empty after a pick so
    // the placeholder "Tunes..." shows.
    const [introTune, setIntroTune] = useState('');
    const [outroTune, setOutroTune] = useState('');
    const [loaded, setLoaded] = useState(false);
    const [status, setStatus] = useState('');

    useEffect(() => {
        fetch('/api/tones').then(r => r.json()).then(d => {
            setIntro((d.intro || []).join(' '));
            setOutro((d.outro || []).join(' '));
            setDuration(d.duration_seconds ?? 0.12);
            setApplyEffects(d.apply_effects !== false);
        });
        fetch('/api/tones/tunes').then(r => r.json()).then(d => {
            setTunes(d.tunes || []);
            setLoaded(true);
        });
    }, []);

    const parseNotes = (s) =>
        s.trim().split(/\s+/).filter(Boolean);

    const playRaw = async (noteString) => {
        const notes = parseNotes(noteString);
        if (notes.length === 0) {
            setStatus('No notes to play.');
            setTimeout(() => setStatus(''), 2000);
            return;
        }
        try {
            const resp = await fetch('/api/tones/play', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ notes, duration }),
            });
            if (!resp.ok) {
                const err = await resp.json().catch(() => ({}));
                setStatus('Error: ' + (err.detail || resp.statusText));
                return;
            }
            setStatus('Playing ' + notes.length + ' note' + (notes.length === 1 ? '' : 's') + '...');
            setTimeout(() => setStatus(''), 2500);
        } catch (e) {
            setStatus('Play failed: ' + e.message);
        }
    };

    const save = async () => {
        setStatus('Saving...');
        try {
            const resp = await fetch('/api/tones', {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    intro: parseNotes(intro),
                    outro: parseNotes(outro),
                    duration_seconds: duration,
                    apply_effects: applyEffects,
                }),
            });
            if (!resp.ok) {
                const err = await resp.json().catch(() => ({}));
                setStatus('Error: ' + (err.detail || resp.statusText));
                return;
            }
            const data = await resp.json();
            setIntro((data.intro || []).join(' '));
            setOutro((data.outro || []).join(' '));
            setStatus(data.message || 'Saved.');
            setTimeout(() => setStatus(''), 3000);
        } catch (e) {
            setStatus('Save failed: ' + e.message);
        }
    };

    // Controlled-component reset: set the notes from the picked tune,
    // then immediately blank the select's own state so React renders the
    // placeholder "Tunes..." again on the next paint.
    const onTunePicked = (setNotes, setSelectedTune) => (e) => {
        const name = e.target.value;
        if (!name) return;
        const tune = tunes.find(t => t.name === name);
        if (tune) setNotes(tune.notes.join(' '));
        setSelectedTune('');
    };

    return (
        <>
            <div className="help">
                Notation: <code>pitch</code> + <code>octave</code> + optional <code>:multiplier</code>.
                E.g. <code>C4</code>, <code>G#5</code>, <code>Bb3:0.5</code> (half),
                <code>C5:4</code> (quadruple). Space-separated.
            </div>
            {!loaded ? <div>Loading...</div> : (
                <>
                    <div className="field-row">
                        <label>Intro notes</label>
                        <input
                            type="text"
                            value={intro}
                            placeholder="e.g. E4 G4 C5"
                            onChange={e => setIntro(e.target.value)}
                        />
                        <button
                            className="btn-try"
                            onClick={() => playRaw(intro)}
                            title="Preview the current intro notes at the current duration"
                        >Play</button>
                        <select
                            value={introTune}
                            onChange={onTunePicked(setIntro, setIntroTune)}
                            title="Drop in a public-domain tune"
                            style={{ flex: '0 0 170px' }}
                        >
                            <option value="">Tunes...</option>
                            {tunes.map(t => (
                                <option key={t.name} value={t.name}>{t.name}</option>
                            ))}
                        </select>
                    </div>
                    <div className="field-row">
                        <label>Outro notes</label>
                        <input
                            type="text"
                            value={outro}
                            placeholder="e.g. C5 G4 E4"
                            onChange={e => setOutro(e.target.value)}
                        />
                        <button
                            className="btn-try"
                            onClick={() => playRaw(outro)}
                            title="Preview the current outro notes at the current duration"
                        >Play</button>
                        <select
                            value={outroTune}
                            onChange={onTunePicked(setOutro, setOutroTune)}
                            title="Drop in a public-domain tune"
                            style={{ flex: '0 0 170px' }}
                        >
                            <option value="">Tunes...</option>
                            {tunes.map(t => (
                                <option key={t.name} value={t.name}>{t.name}</option>
                            ))}
                        </select>
                    </div>
                    <div className="field-row">
                        <label>Note duration (s)</label>
                        <input
                            type="number"
                            min="0.05" max="1.0" step="0.01"
                            value={duration}
                            onChange={e => setDuration(parseFloat(e.target.value))}
                        />
                    </div>
                    <div className="field-row" style={{ alignItems: 'center' }}>
                        <label>Apply effects to tones</label>
                        <label style={{ display: 'inline-flex', alignItems: 'center', gap: 6, fontWeight: 400, color: 'var(--text-2)' }}>
                            <input
                                type="checkbox"
                                checked={applyEffects}
                                onChange={e => setApplyEffects(e.target.checked)}
                            />
                            {applyEffects ? 'On' : 'Off'}
                        </label>
                    </div>
                    <div style={{ display: 'flex', gap: 8, marginTop: 12, alignItems: 'center' }}>
                        <button className="btn" onClick={save}>Save tones</button>
                        <span className="save-success">{status}</span>
                    </div>
                    <ToneRulesEditor tunes={tunes} />
                </>
            )}
        </>
    );
}

// ----- Per-queue / per-interpretation tone rules.
//
// One table row = one rule. Each rule overrides the default intro / outro /
// cue notes for a specific queue, a specific interpretation, or both. The
// hierarchy from broad to narrow is: global default -> interpretation-only
// rule -> queue-only rule -> queue + interpretation rule (most specific).
// The daemon picks the highest-specificity match at speech time.
function ToneRulesEditor({ tunes }) {
    const [rules, setRules] = useState([]);
    const [interps, setInterps] = useState([]);
    const [knownQueues, setKnownQueues] = useState([]);
    const [loaded, setLoaded] = useState(false);
    const [status, setStatus] = useState('');

    const load = useCallback(() => {
        fetch('/api/tone-rules').then(r => r.json()).then(d => {
            setRules((d.rules || []).map(r => ({
                slot: r.slot || 'cue',
                queue: r.queue || '',
                queue_regex: !!r.queue_regex,
                interpretation: r.interpretation || '',
                notes: Array.isArray(r.notes) ? r.notes.join(' ') : '',
            })));
            setInterps(d.interpretations || []);
            setKnownQueues(d.queues || []);
            setLoaded(true);
        });
    }, []);

    useEffect(() => { load(); }, [load]);

    const updateRow = (idx, key, value) => {
        setRules(rs => rs.map((r, i) => i === idx ? { ...r, [key]: value } : r));
    };

    const addRow = () => {
        setRules(rs => [...rs, {
            slot: 'cue', queue: '', queue_regex: false,
            interpretation: '', notes: '',
        }]);
    };

    const removeRow = (idx) => {
        setRules(rs => rs.filter((_, i) => i !== idx));
    };

    const parseNotes = (s) =>
        (s || '').trim().split(/\s+/).filter(Boolean);

    const tryRow = async (idx) => {
        const r = rules[idx];
        const notes = parseNotes(r.notes);
        if (notes.length === 0) {
            setStatus('Row ' + (idx + 1) + ': no notes to play.');
            setTimeout(() => setStatus(''), 2500);
            return;
        }
        try {
            const resp = await fetch('/api/tones/play', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ notes }),
            });
            if (!resp.ok) {
                const err = await resp.json().catch(() => ({}));
                setStatus('Error: ' + (err.detail || resp.statusText));
                return;
            }
            setStatus('Playing row ' + (idx + 1) + '...');
            setTimeout(() => setStatus(''), 2500);
        } catch (e) {
            setStatus('Play failed: ' + e.message);
        }
    };

    const save = async () => {
        const payload = rules.map(r => ({
            slot: r.slot,
            queue: r.queue.trim() || null,
            queue_regex: !!r.queue_regex,
            interpretation: r.interpretation.trim() || null,
            notes: parseNotes(r.notes),
        }));
        setStatus('Saving...');
        try {
            const resp = await fetch('/api/tone-rules', {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ rules: payload }),
            });
            if (!resp.ok) {
                const err = await resp.json().catch(() => ({}));
                setStatus('Error: ' + (err.detail || resp.statusText));
                return;
            }
            const data = await resp.json();
            setStatus(data.message || 'Saved.');
            setTimeout(() => setStatus(''), 3000);
        } catch (e) {
            setStatus('Save failed: ' + e.message);
        }
    };

    // Tune dropdown sets row.notes from a preset.
    const onTunePicked = (idx) => (e) => {
        const name = e.target.value;
        if (!name) return;
        const tune = tunes.find(t => t.name === name);
        if (tune) updateRow(idx, 'notes', tune.notes.join(' '));
        // Reset the select so the placeholder shows again. The select is
        // uncontrolled-by-rule because we don't want to track its pick state
        // per row; the *effect* is on row.notes, not the select itself.
        e.target.value = '';
    };

    return (
        <div style={{ marginTop: 28 }}>
            <h3 style={{ margin: '0 0 6px', fontSize: '1.05em' }}>Custom rules</h3>
            <p className="section-subtitle" style={{ marginTop: 0 }}>
                Override the default tune for specific queues, interpretations, or both.
                Most-specific match wins. Empty fields mean &ldquo;any&rdquo;.
            </p>
            {!loaded ? <div>Loading...</div> : (
                <>
                    <table className="pronunciation-table" style={{ marginTop: 8 }}>
                        <thead>
                            <tr>
                                <th style={{ width: 90 }}>Slot</th>
                                <th>Queue</th>
                                <th style={{ width: 70 }}>Regex</th>
                                <th style={{ width: 160 }}>Interpretation</th>
                                <th>Notes</th>
                                <th style={{ width: 200 }}></th>
                            </tr>
                        </thead>
                        <tbody>
                            {rules.length === 0 && (
                                <tr><td colSpan="6" style={{ color: 'var(--text-3)', padding: '12px 6px' }}>
                                    No custom rules. Add one below to override the defaults.
                                </td></tr>
                            )}
                            {rules.map((r, idx) => (
                                <tr key={idx}>
                                    <td>
                                        <select
                                            value={r.slot}
                                            onChange={e => updateRow(idx, 'slot', e.target.value)}
                                            style={{ width: '100%' }}
                                        >
                                            <option value="intro">intro</option>
                                            <option value="outro">outro</option>
                                            <option value="cue">cue</option>
                                        </select>
                                    </td>
                                    <td>
                                        <input
                                            type="text"
                                            placeholder="any"
                                            list={`tone-rule-queues-${idx}`}
                                            value={r.queue}
                                            onChange={e => updateRow(idx, 'queue', e.target.value)}
                                        />
                                        <datalist id={`tone-rule-queues-${idx}`}>
                                            {knownQueues.map(q => <option key={q} value={q} />)}
                                        </datalist>
                                    </td>
                                    <td style={{ textAlign: 'center' }}>
                                        <input
                                            type="checkbox"
                                            checked={r.queue_regex}
                                            onChange={e => updateRow(idx, 'queue_regex', e.target.checked)}
                                            title="Treat queue as a regex pattern"
                                        />
                                    </td>
                                    <td>
                                        <input
                                            type="text"
                                            placeholder="any"
                                            list={`tone-rule-interps-${idx}`}
                                            value={r.interpretation}
                                            onChange={e => updateRow(idx, 'interpretation', e.target.value)}
                                        />
                                        <datalist id={`tone-rule-interps-${idx}`}>
                                            {interps.map(i => <option key={i} value={i} />)}
                                        </datalist>
                                    </td>
                                    <td>
                                        <input
                                            type="text"
                                            placeholder="e.g. E4 G4 C5:2"
                                            value={r.notes}
                                            onChange={e => updateRow(idx, 'notes', e.target.value)}
                                        />
                                    </td>
                                    <td style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                                        <button className="btn-try" onClick={() => tryRow(idx)}>Play</button>
                                        <select
                                            onChange={onTunePicked(idx)}
                                            defaultValue=""
                                            title="Drop in a public-domain tune"
                                            style={{ flex: 1 }}
                                        >
                                            <option value="">Tune...</option>
                                            {tunes.map(t => (
                                                <option key={t.name} value={t.name}>{t.name}</option>
                                            ))}
                                        </select>
                                        <button
                                            className="btn-try"
                                            onClick={() => removeRow(idx)}
                                            title="Remove this rule"
                                        >×</button>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                    <div style={{ display: 'flex', gap: 8, marginTop: 12, alignItems: 'center' }}>
                        <button className="btn" onClick={addRow}>Add rule</button>
                        <button className="btn" onClick={save}>Save rules</button>
                        <span className="save-success">{status}</span>
                    </div>
                </>
            )}
        </div>
    );
}

// ----- Audio effects preset picker + "Try sample" preview.
//
// Live preset choice lives in local state. Clicking "Try sample" sends
// the *current* (possibly unsaved) selection to /api/effects/try, which
// attaches it as metadata on a one-shot queue item -- so the preview
// reflects the picker without mutating config.json. "Save" persists the
// choice for all future utterances.
function EffectsSection() {
    const [presets, setPresets] = useState([]);
    const [current, setCurrent] = useState('off');
    const [selected, setSelected] = useState('off');
    const [status, setStatus] = useState('');
    const [plugins, setPlugins] = useState({});  // {pluginName: [{name, type, default, min, max, step}, ...]}
    const [editorOpen, setEditorOpen] = useState(false);

    const reloadPresets = useCallback(() => {
        fetch('/api/effects').then(r => r.json()).then(d => {
            setPresets(d.presets || []);
            setCurrent(d.current || 'off');
        });
    }, []);

    useEffect(() => {
        fetch('/api/effects').then(r => r.json()).then(d => {
            setPresets(d.presets || []);
            setCurrent(d.current || 'off');
            setSelected(d.current || 'off');
        });
        fetch('/api/effects/plugins').then(r => r.json()).then(d => setPlugins(d.plugins || {}));
    }, []);

    const save = async () => {
        setStatus('Saving...');
        try {
            const resp = await fetch('/api/effects', {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ preset: selected }),
            });
            if (!resp.ok) {
                const err = await resp.json().catch(() => ({}));
                setStatus('Error: ' + (err.detail || resp.statusText));
                return;
            }
            const data = await resp.json();
            setCurrent(data.current);
            setStatus(data.message || 'Saved.');
            setTimeout(() => setStatus(''), 3000);
        } catch (e) {
            setStatus('Save failed: ' + e.message);
        }
    };

    const trySample = async () => {
        setStatus('Playing sample...');
        try {
            await fetch('/api/effects/try', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ preset: selected }),
            });
            setStatus('Sample queued.');
            setTimeout(() => setStatus(''), 3000);
        } catch (e) {
            setStatus('Try failed: ' + e.message);
        }
    };

    const description =
        (presets.find(p => p.name === selected) || {}).description || '';
    const selectedIsBuiltin = (presets.find(p => p.name === selected) || {}).builtin;

    return (
        <>
            <div className="field-row">
                <label>Preset</label>
                <select value={selected} onChange={e => setSelected(e.target.value)}>
                    {presets.map(p => (
                        <option key={p.name} value={p.name}>
                            {p.name}{p.builtin ? '' : ' (custom)'}
                            {p.effect_count > 0 ? `  (${p.effect_count} effect${p.effect_count === 1 ? '' : 's'})` : ''}
                        </option>
                    ))}
                </select>
            </div>
            {description && (
                <div className="help" style={{ marginTop: 6, marginBottom: 14 }}>
                    {description}
                </div>
            )}
            <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
                <button className="btn-try" onClick={trySample}>Try sample</button>
                <button
                    className="btn"
                    onClick={save}
                    disabled={selected === current}
                    title={selected === current ? 'No unsaved change' : 'Persist selection'}
                >Save</button>
                <button
                    className="btn subtle"
                    onClick={() => setEditorOpen(o => !o)}
                    title="Add / edit / delete custom presets"
                >{editorOpen ? 'Hide editor' : 'Open editor'}</button>
                <span className="save-success">{status}</span>
                {selected !== current && (
                    <span className="filter-label" style={{ marginLeft: 6 }}>
                        Saved preset: <code>{current}</code>
                    </span>
                )}
            </div>
            {editorOpen && (
                <PresetEditor
                    presets={presets}
                    plugins={plugins}
                    onPresetsChanged={reloadPresets}
                    selectedName={selected}
                    onSelectPreset={setSelected}
                />
            )}
        </>
    );
}

// ----- Effects Preset Editor.
//
// CRUD over custom presets. Built-in presets are read-only -- selecting
// one shows its chain but disables save/rename/delete. Editing a built-in
// is done via "Clone as custom..." which seeds a new editable preset.
//
// Storage shape (per preset, matches the API):
//   {name: string, effects: [{name: pluginName, params: {paramName: number}}]}
function PresetEditor({ presets, plugins, onPresetsChanged, selectedName, onSelectPreset }) {
    const [editName, setEditName] = useState(selectedName);
    const [chain, setChain] = useState([]);  // [{name, params}]
    const [origName, setOrigName] = useState(selectedName);
    const [origChain, setOrigChain] = useState([]);
    const [isBuiltin, setIsBuiltin] = useState(true);
    const [renameTo, setRenameTo] = useState('');
    const [addType, setAddType] = useState('');
    const [status, setStatus] = useState('');

    // Load the selected preset's chain whenever the user picks a different
    // one from the parent's dropdown. We track its origin chain so the
    // Save button can stay disabled when nothing has changed.
    useEffect(() => {
        if (!selectedName) return;
        fetch('/api/effects/presets/' + encodeURIComponent(selectedName))
            .then(r => r.json())
            .then(d => {
                setEditName(d.name);
                setOrigName(d.name);
                setChain(d.effects || []);
                setOrigChain(JSON.parse(JSON.stringify(d.effects || [])));
                setIsBuiltin(!!d.builtin);
                setRenameTo('');
                setStatus('');
            });
    }, [selectedName]);

    const isDirty = JSON.stringify(chain) !== JSON.stringify(origChain)
        || (renameTo.trim() && renameTo.trim() !== origName);

    const setEffectParam = (idx, key, value) => {
        setChain(c => c.map((eff, i) => i === idx
            ? { ...eff, params: { ...eff.params, [key]: parseFloat(value) } }
            : eff
        ));
    };

    const moveEffect = (idx, delta) => {
        const j = idx + delta;
        if (j < 0 || j >= chain.length) return;
        setChain(c => {
            const next = c.slice();
            [next[idx], next[j]] = [next[j], next[idx]];
            return next;
        });
    };

    const removeEffect = (idx) => setChain(c => c.filter((_, i) => i !== idx));

    const addEffect = () => {
        if (!addType || !plugins[addType]) return;
        // Seed with the catalog's defaults so the chain audibly does
        // something the moment it's added.
        const params = {};
        for (const p of plugins[addType]) {
            params[p.name] = p.default;
        }
        setChain(c => [...c, { name: addType, params }]);
        setAddType('');
    };

    const saveChain = async () => {
        setStatus('Saving...');
        try {
            const url = '/api/effects/presets/' + encodeURIComponent(origName);
            const body = { effects: chain };
            if (renameTo.trim() && renameTo.trim() !== origName) {
                body.rename = renameTo.trim();
            }
            const resp = await fetch(url, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            });
            if (!resp.ok) {
                const err = await resp.json().catch(() => ({}));
                setStatus('Error: ' + (err.detail || resp.statusText));
                return;
            }
            const data = await resp.json();
            setOrigChain(JSON.parse(JSON.stringify(data.effects || [])));
            setOrigName(data.name);
            setEditName(data.name);
            setRenameTo('');
            setStatus(data.message || 'Saved.');
            onPresetsChanged();
            if (data.name !== origName) onSelectPreset(data.name);
            setTimeout(() => setStatus(''), 2500);
        } catch (e) {
            setStatus('Save failed: ' + e.message);
        }
    };

    const deletePreset = async () => {
        if (!window.confirm('Delete preset "' + origName + '"? This cannot be undone.')) return;
        const resp = await fetch('/api/effects/presets/' + encodeURIComponent(origName), { method: 'DELETE' });
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            setStatus('Error: ' + (err.detail || resp.statusText));
            return;
        }
        const data = await resp.json();
        setStatus(data.message || 'Deleted.');
        onPresetsChanged();
        // Snap selection back to 'off' to avoid pointing at the deleted name.
        onSelectPreset(data.default_reset_to_off ? 'off' : 'off');
    };

    const cloneAsCustom = async () => {
        const base = origName;
        let suggested = base + ' copy';
        // Avoid collision with existing names.
        const existing = new Set(presets.map(p => p.name));
        let i = 1;
        while (existing.has(suggested)) {
            i += 1;
            suggested = base + ' copy ' + i;
        }
        const name = window.prompt('Name for the new custom preset:', suggested);
        if (!name) return;
        const resp = await fetch('/api/effects/presets', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, effects: chain }),
        });
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            setStatus('Error: ' + (err.detail || resp.statusText));
            return;
        }
        const data = await resp.json();
        setStatus(data.message || 'Created.');
        onPresetsChanged();
        onSelectPreset(data.name);
    };

    return (
        <div className="preset-editor">
            <div className="preset-editor-header">
                <div>
                    <h3 style={{ margin: '0 0 4px', fontSize: '1em' }}>
                        Editing: <code>{origName}</code>
                        {isBuiltin && <span className="badge-builtin" title="Built-in presets are read-only -- use Clone as custom to edit">built-in</span>}
                    </h3>
                    {isBuiltin
                        ? <div className="help" style={{ margin: 0 }}>
                            Read-only. Use <strong>Clone as custom</strong> to start a new editable preset from this chain.
                        </div>
                        : <div className="field-row" style={{ margin: '4px 0 0' }}>
                            <label style={{ minWidth: 60 }}>Rename</label>
                            <input
                                type="text"
                                placeholder={origName}
                                value={renameTo}
                                onChange={e => setRenameTo(e.target.value)}
                            />
                        </div>
                    }
                </div>
            </div>
            <div className="preset-editor-chain">
                {chain.length === 0 && (
                    <div style={{ color: 'var(--text-3)', padding: '12px 0' }}>
                        Empty chain. Add an effect below.
                    </div>
                )}
                {chain.map((eff, idx) => {
                    const params = plugins[eff.name] || [];
                    return (
                        <div className="effect-card" key={idx}>
                            <div className="effect-card-header">
                                <span className="effect-name">{eff.name}</span>
                                <div className="effect-card-actions">
                                    <button
                                        className="btn-try"
                                        disabled={idx === 0 || isBuiltin}
                                        onClick={() => moveEffect(idx, -1)}
                                        title="Move up"
                                    >▲</button>
                                    <button
                                        className="btn-try"
                                        disabled={idx === chain.length - 1 || isBuiltin}
                                        onClick={() => moveEffect(idx, 1)}
                                        title="Move down"
                                    >▼</button>
                                    <button
                                        className="btn-try"
                                        disabled={isBuiltin}
                                        onClick={() => removeEffect(idx)}
                                        title="Remove"
                                    >×</button>
                                </div>
                            </div>
                            <div className="effect-params">
                                {params.map(p => {
                                    const value = eff.params[p.name];
                                    const display = (value === undefined || value === null) ? p.default : value;
                                    return (
                                        <div className="effect-param" key={p.name}>
                                            <label title={`min ${p.min} max ${p.max}`}>{p.name}</label>
                                            <input
                                                type="range"
                                                min={p.min}
                                                max={p.max}
                                                step={p.step}
                                                value={display}
                                                disabled={isBuiltin}
                                                onChange={e => setEffectParam(idx, p.name, e.target.value)}
                                            />
                                            <input
                                                type="number"
                                                min={p.min}
                                                max={p.max}
                                                step={p.step}
                                                value={display}
                                                disabled={isBuiltin}
                                                onChange={e => setEffectParam(idx, p.name, e.target.value)}
                                                className="param-num"
                                            />
                                        </div>
                                    );
                                })}
                            </div>
                        </div>
                    );
                })}
            </div>
            <div className="field-row" style={{ marginTop: 12 }}>
                <label style={{ minWidth: 60 }}>Add effect</label>
                <select
                    value={addType}
                    onChange={e => setAddType(e.target.value)}
                    disabled={isBuiltin}
                >
                    <option value="">(choose plugin...)</option>
                    {Object.keys(plugins).sort().map(name => (
                        <option key={name} value={name}>{name}</option>
                    ))}
                </select>
                <button
                    className="btn"
                    onClick={addEffect}
                    disabled={!addType || isBuiltin}
                >Add</button>
            </div>
            <div style={{ display: 'flex', gap: 8, marginTop: 14, alignItems: 'center', flexWrap: 'wrap' }}>
                <button
                    className="btn"
                    onClick={saveChain}
                    disabled={!isDirty || isBuiltin}
                    title={isBuiltin ? 'Built-in presets are read-only' : (isDirty ? 'Save changes' : 'No unsaved change')}
                >Save</button>
                <button
                    className="btn subtle"
                    onClick={cloneAsCustom}
                    title="Create a new custom preset starting from this chain"
                >Clone as custom...</button>
                <button
                    className="btn subtle"
                    onClick={deletePreset}
                    disabled={isBuiltin}
                    title={isBuiltin ? 'Built-in presets cannot be deleted' : 'Delete this custom preset'}
                    style={{ color: 'var(--warn)' }}
                >Delete</button>
                <span className="save-success">{status}</span>
            </div>
        </div>
    );
}

// ----- SSML support matrix + short guide.
// Simple SSML/XML syntax highlighter: tokenize into tags, attribute
// names, attribute values, comments, and text content. Returns React
// nodes so spans live in JSX. Not a full XML parser -- it's intentionally
// permissive so a half-typed tag still renders something useful while
// the user is editing.
function highlightSSML(src) {
    const out = [];
    let i = 0;
    let key = 0;
    while (i < src.length) {
        if (src.startsWith('<!--', i)) {
            const end = src.indexOf('-->', i + 4);
            const stop = end < 0 ? src.length : end + 3;
            out.push(<span key={key++} className="ssml-tok-comment">{src.slice(i, stop)}</span>);
            i = stop;
            continue;
        }
        if (src[i] === '<') {
            // Walk to the matching '>', emitting attr/string colors inside.
            const end = src.indexOf('>', i);
            if (end < 0) {
                out.push(<span key={key++} className="ssml-tok-tag">{src.slice(i)}</span>);
                i = src.length;
                continue;
            }
            const tag = src.slice(i, end + 1);
            // Split the tag into pieces: tag-name, then attr=val pairs.
            const parts = [];
            // Match: tag name first; then runs of attr="value" or attr='value' or attr=value
            let m;
            const tagRe = /(\s+)([a-zA-Z_:][\w:.-]*)(\s*=\s*)("[^"]*"|'[^']*'|[^\s>]+)/g;
            let last = 0;
            // The tag-name portion is everything up to the first space or '>'.
            const nameEnd = tag.search(/[\s>\/]/);
            parts.push({ type: 'tag', text: tag.slice(0, nameEnd > 0 ? nameEnd : tag.length) });
            last = nameEnd > 0 ? nameEnd : tag.length;
            while ((m = tagRe.exec(tag)) && m.index >= last) {
                parts.push({ type: 'tag', text: m[1] });             // whitespace
                parts.push({ type: 'attr', text: m[2] });            // attr name
                parts.push({ type: 'tag', text: m[3] });             // =
                parts.push({ type: 'string', text: m[4] });          // value
                last = tagRe.lastIndex;
            }
            parts.push({ type: 'tag', text: tag.slice(last) });
            for (const p of parts) {
                if (p.text) {
                    out.push(<span key={key++} className={'ssml-tok-' + p.type}>{p.text}</span>);
                }
            }
            i = end + 1;
            continue;
        }
        // Text run up to next '<'
        const next = src.indexOf('<', i);
        const stop = next < 0 ? src.length : next;
        out.push(<span key={key++} className="ssml-tok-text">{src.slice(i, stop)}</span>);
        i = stop;
    }
    return out;
}

// Default SSML when the editor mounts. Backtick (template literal) so
// the source can span multiple lines safely -- a single-quoted JS
// string with newline escapes does not survive the Python triple-
// quoted HTML_TEMPLATE wrapper (Python interprets the escape into a
// real newline, which then closes the JS string).
const DEFAULT_SSML = `<speak>
  Hello, this is <prosody rate="slow">very slow</prosody> speech.
  <break time="400ms"/>
  And the word <phoneme alphabet="ipa" ph="ˈkɒm.pəs">compass</phoneme> is pronounced with IPA.
</speak>`;

function SSMLGuide({ engines }) {
    const [ssml, setSsml] = useState(DEFAULT_SSML);
    const [lint, setLint] = useState(null);
    const [status, setStatus] = useState('');

    // Live lint with a small debounce so we don't hammer the server on
    // every keystroke. 400ms is short enough to feel responsive without
    // being chatty.
    useEffect(() => {
        const id = setTimeout(async () => {
            try {
                const resp = await fetch('/api/ssml/lint', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ ssml }),
                });
                setLint(await resp.json());
            } catch (e) { /* ignore network blips */ }
        }, 400);
        return () => clearTimeout(id);
    }, [ssml]);

    const tryIt = async () => {
        setStatus('Speaking...');
        try {
            const resp = await fetch('/api/ssml/try', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ ssml }),
            });
            if (!resp.ok) {
                const err = await resp.json().catch(() => ({}));
                setStatus('Error: ' + (typeof err.detail === 'object' ? (err.detail.errors?.[0]?.message || JSON.stringify(err.detail)) : err.detail));
                return;
            }
            setStatus('Queued.');
            setTimeout(() => setStatus(''), 2500);
        } catch (e) {
            setStatus('Try failed: ' + e.message);
        }
    };

    return (
        <div className="ssml-guide">
            <div className="help">
                SSML is the only path to deterministic pronunciation. Only Polly honors it.
            </div>

            <div className="ssml-editor">
                <pre aria-hidden="true">{highlightSSML(ssml)}</pre>
                <textarea
                    value={ssml}
                    spellCheck={false}
                    onChange={e => setSsml(e.target.value)}
                />
            </div>
            <div className="ssml-feedback">
                {lint && lint.errors && lint.errors.length > 0 && lint.errors.map((e, i) => (
                    <div key={'e' + i} className="ssml-error">
                        {e.line ? `Line ${e.line}, col ${e.col}: ` : ''}{e.message}
                    </div>
                ))}
                {lint && lint.warnings && lint.warnings.map((w, i) => (
                    <div key={'w' + i} className="ssml-warning">{w.message}</div>
                ))}
                {lint && lint.ok && (lint.errors || []).length === 0 && (lint.warnings || []).length === 0 && (
                    <span className="ssml-ok">Valid SSML.</span>
                )}
            </div>

            <div style={{ display: 'flex', gap: 10, marginTop: 14, alignItems: 'center' }}>
                <button
                    className="btn"
                    onClick={tryIt}
                    disabled={!lint || !lint.ok}
                    title={lint && lint.ok ? 'Send to Polly and play' : 'Fix the errors above first'}
                >Try it</button>
                <span className="save-success">{status}</span>
            </div>

            <table style={{ marginTop: 22 }}>
                <thead>
                    <tr><th>Engine</th><th>SSML</th><th>How to use</th></tr>
                </thead>
                <tbody>
                    {engines.map(e => (
                        <tr key={e.name}>
                            <td><code>{e.name}</code></td>
                            <td>{e.supports_ssml ? '✓' : '—'}</td>
                            <td>
                                {e.supports_ssml
                                    ? <>Send <code>?ssml=true</code> on <code>/speak</code>, or text starting with <code>&lt;speak&gt;</code>.</>
                                    : <>No SSML. Use the pronunciation overrides.</>}
                            </td>
                        </tr>
                    ))}
                </tbody>
            </table>
            <p style={{ marginTop: 14 }}>
                <a href="https://docs.aws.amazon.com/polly/latest/dg/supportedtags.html" target="_blank" rel="noreferrer" style={{ color: 'var(--accent)' }}>
                    Polly SSML reference
                </a>
            </p>
        </div>
    );
}

// ----- Per-queue overrides table.
// Per-queue overrides: cards-or-table dual view, fuzzy filter, sort.
// Mirrors the Queue History pattern (filter input + view toggle) so the
// two main collections in the app feel cohesive.
function PerQueueSection({ engines }) {
    const [queues, setQueues] = useState([]);
    const [editing, setEditing] = useState(null);
    const [pending, setPending] = useState({});
    const [saveStatus, setSaveStatus] = useState({});
    const [presets, setPresets] = useState([]);  // [{name, description, builtin}]
    // Set of queue names included in the filter. Empty Set = show all.
    // Shape matches HistoryView's selectedQueues so ProjectPicker can be
    // reused as-is.
    const [selectedQueues, setSelectedQueues] = useState(() => new Set());
    const [sortKey, setSortKey] = useState(() => localStorage.getItem('speeker.perqueueSort') || 'recent');
    useEffect(() => { localStorage.setItem('speeker.perqueueSort', sortKey); }, [sortKey]);
    const [viewMode, setViewMode] = useState(() => localStorage.getItem('speeker.perqueueView') || 'cards');
    useEffect(() => { localStorage.setItem('speeker.perqueueView', viewMode); }, [viewMode]);
    // Page-local sample phrase used by every row's Try button. Persisted
    // so the user can A/B voices on a phrase they actually care about
    // without retyping it each visit.
    const [samplePhrase, setSamplePhrase] = useState(() =>
        localStorage.getItem('speeker.perqueueSample') || 'The quick brown fox jumps over the lazy dog.'
    );
    useEffect(() => { localStorage.setItem('speeker.perqueueSample', samplePhrase); }, [samplePhrase]);
    const [tryStatus, setTryStatus] = useState({});  // per-queue transient status

    const reload = useCallback(() => {
        fetch('/api/queues').then(r => r.json()).then(d => setQueues(d.queues));
    }, []);
    useEffect(() => { reload(); }, [reload]);
    // Load preset list once for the per-queue effects dropdown.
    useEffect(() => {
        fetch('/api/effects').then(r => r.json()).then(d => setPresets(d.presets || []));
    }, []);

    const startEdit = (q) => {
        setEditing(q.queue);
        setPending(prev => ({ ...prev, [q.queue]: { ...q.settings } }));
    };
    const cancelEdit = () => setEditing(null);
    const setField = (queue, key, value) => {
        setPending(prev => ({ ...prev, [queue]: { ...prev[queue], [key]: value } }));
    };
    const save = async (queue) => {
        const patch = pending[queue];
        setSaveStatus(prev => ({ ...prev, [queue]: 'Saving...' }));
        const body = {
            session: queue,
            engine: patch.engine,
            voice: patch.voice,
        };
        // Color: send only when changed from the current setting; empty
        // string is the explicit "clear override" sentinel honored by
        // the backend (so user-removed colors fall back to auto-derived).
        const origColor = (queues.find(q => q.queue === queue) || {}).settings?.color || '';
        if ((patch.color || '') !== origColor) {
            body.color = patch.color || '';
        }
        // Effects preset: same pattern. ``"inherit"`` is the UI-only
        // sentinel for "fall back to global"; map it to empty string
        // before sending so the backend stores NULL.
        const origPreset = (queues.find(q => q.queue === queue) || {}).settings?.effects_preset || '';
        const newPreset = patch.effects_preset === 'inherit' ? '' : (patch.effects_preset || '');
        if (newPreset !== origPreset) {
            body.effects_preset = newPreset;
        }
        await fetch('/api/settings', {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        setSaveStatus(prev => ({ ...prev, [queue]: 'Saved' }));
        setEditing(null);
        reload();
        setTimeout(() => setSaveStatus(prev => ({ ...prev, [queue]: '' })), 2000);
    };

    // Speak the sample phrase using THIS queue's resolved engine/voice.
    // Uses the same /api/engines/try metadata path the Engine&Voice
    // section uses, so the daemon picks up the override for this one
    // utterance only. Status indicator is per-queue so concurrent
    // tries on different rows don't fight.
    const tryRow = async (q) => {
        const phrase = samplePhrase.trim();
        if (!phrase) {
            setTryStatus(prev => ({ ...prev, [q.queue]: 'Type a sample first.' }));
            setTimeout(() => setTryStatus(prev => ({ ...prev, [q.queue]: '' })), 2000);
            return;
        }
        setTryStatus(prev => ({ ...prev, [q.queue]: 'Playing...' }));
        try {
            await fetch('/api/engines/try', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    engine: q.settings.engine,
                    voice: q.settings.voice,
                    text: phrase,
                }),
            });
            setTryStatus(prev => ({ ...prev, [q.queue]: 'Queued' }));
        } catch (e) {
            setTryStatus(prev => ({ ...prev, [q.queue]: 'Failed' }));
        }
        setTimeout(() => setTryStatus(prev => ({ ...prev, [q.queue]: '' })), 2000);
    };

    // ProjectPicker uses the same {name, count, color} shape as the
    // History sidebar. Count comes from the queue's total_messages so
    // the user can quickly tell which projects are most active. Color
    // comes from the per-queue setting -- explicit value or the
    // auto-derived default (settings.color is always populated).
    const projectOptions = useMemo(
        () => queues.map(q => ({
            name: q.queue,
            count: q.total_messages,
            color: q.settings && q.settings.color,
        })),
        [queues],
    );
    const toggleQueue = (name) => {
        setSelectedQueues(prev => {
            const next = new Set(prev);
            if (next.has(name)) next.delete(name); else next.add(name);
            return next;
        });
    };

    const filteredSorted = useMemo(() => {
        const arr = selectedQueues.size === 0
            ? queues
            : queues.filter(q => selectedQueues.has(q.queue));
        const cmp = {
            'name':     (a, b) => a.queue.localeCompare(b.queue),
            'name-rev': (a, b) => b.queue.localeCompare(a.queue),
            'count':    (a, b) => b.total_messages - a.total_messages,
            'recent':   (a, b) => (b.last_activity || '').localeCompare(a.last_activity || ''),
        }[sortKey] || ((a, b) => 0);
        return [...arr].sort(cmp);
    }, [queues, selectedQueues, sortKey]);

    if (queues.length === 0) return <div className="help">No queues with history yet.</div>;

    return (
        <>
            {/* Page-local sample phrase. Each row's "Try" button speaks
                this phrase with the queue's resolved engine + voice. */}
            <div className="field-row" style={{ marginBottom: 14 }}>
                <label>Sample phrase</label>
                <input
                    type="text"
                    value={samplePhrase}
                    placeholder="Phrase each row's Try button will speak..."
                    onChange={e => setSamplePhrase(e.target.value)}
                />
            </div>
            <div className="layout-split">
                {/* Side panel: project filter, mirrors Queue History's
                    sidebar pattern. Inset variant skips the sticky
                    positioning because we're inside a Section card. */}
                <aside className="sidebar sidebar-inset">
                    <h3 style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                        <span>Projects</span>
                        {selectedQueues.size > 0 && (
                            <button
                                onClick={() => setSelectedQueues(new Set())}
                                style={{
                                    background: 'transparent',
                                    border: 'none',
                                    color: 'var(--accent)',
                                    fontSize: '0.85em',
                                    cursor: 'pointer',
                                    padding: 0,
                                    fontWeight: 600,
                                }}
                            >Clear ({selectedQueues.size})</button>
                        )}
                    </h3>
                    <ProjectPicker
                        projects={projectOptions}
                        selected={selectedQueues}
                        onToggle={toggleQueue}
                    />
                </aside>

                {/* Main column: sort + view toggle on top, cards/table below. */}
                <main className="main">
                    <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginBottom: 16 }}>
                        <select
                            value={sortKey}
                            onChange={e => setSortKey(e.target.value)}
                            style={{
                                flex: '0 0 220px',
                                padding: '9px 12px',
                                background: 'var(--surface-3)',
                                border: '1px solid var(--border)',
                                color: 'var(--text-1)',
                                borderRadius: 6,
                                fontSize: 15,
                            }}
                            aria-label="Sort"
                        >
                            <option value="recent">Most recent activity</option>
                            <option value="count">Most messages</option>
                            <option value="name">Name A→Z</option>
                            <option value="name-rev">Name Z→A</option>
                        </select>
                        <div className="view-toggle">
                            <button
                                className={viewMode === 'cards' ? 'active' : ''}
                                onClick={() => setViewMode('cards')}
                            >Cards</button>
                            <button
                                className={viewMode === 'table' ? 'active' : ''}
                                onClick={() => setViewMode('table')}
                            >Table</button>
                        </div>
                        <span className="filter-summary" style={{ marginLeft: 'auto' }}>
                            {selectedQueues.size === 0
                                ? `${filteredSorted.length} project${filteredSorted.length === 1 ? '' : 's'}`
                                : `${filteredSorted.length} of ${queues.length} project${queues.length === 1 ? '' : 's'}`}
                        </span>
                    </div>
                    {filteredSorted.length === 0
                        ? <div className="no-results">No projects match the filter.</div>
                        : viewMode === 'cards'
                            ? <PerQueueCards
                                queues={filteredSorted}
                                engines={engines}
                                presets={presets}
                                editing={editing}
                                pending={pending}
                                saveStatus={saveStatus}
                                setField={setField}
                                startEdit={startEdit}
                                cancelEdit={cancelEdit}
                                save={save}
                                tryRow={tryRow}
                                tryStatus={tryStatus}
                            />
                            : <PerQueueTable
                                queues={filteredSorted}
                                engines={engines}
                                presets={presets}
                                editing={editing}
                                pending={pending}
                                saveStatus={saveStatus}
                                setField={setField}
                                startEdit={startEdit}
                                cancelEdit={cancelEdit}
                                save={save}
                                tryRow={tryRow}
                                tryStatus={tryStatus}
                            />
                    }
                </main>
            </div>
        </>
    );
}

// Card grid for per-queue overrides. Each card shows the queue name,
// current overrides, message count, and an edit/save row.
function PerQueueCards({ queues, engines, presets, editing, pending, saveStatus, setField, startEdit, cancelEdit, save, tryRow, tryStatus }) {
    return (
        <div className="cards-grid" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))' }}>
            {queues.map(q => {
                const isEd = editing === q.queue;
                const ed = pending[q.queue] || q.settings;
                const engineVoices = engines.find(e => e.name === (ed.engine || q.settings.engine))?.voices || [];
                const color = ed.color || q.settings.color || '#7a8694';
                const cardStyle = { '--queue-color': color };
                // The "inherit" value is a UI sentinel meaning "fall back
                // to the global effects preset"; the backend stores it
                // as NULL.
                const presetValue = ed.effects_preset || 'inherit';
                return (
                    <div className="card" key={q.queue} style={cardStyle}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                            <span style={{ fontWeight: 600, color: 'var(--text-1)', wordBreak: 'break-all' }} title={q.queue}>
                                {q.queue}
                            </span>
                            <span className="time">{q.total_messages} msg</span>
                        </div>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                            <div className="field-row" style={{ margin: 0 }}>
                                <label style={{ minWidth: 60 }}>Engine</label>
                                {isEd ? (
                                    <select value={ed.engine || ''}
                                        onChange={e => setField(q.queue, 'engine', e.target.value)}>
                                        {engines.map(en => <option key={en.name} value={en.name}>{en.name}</option>)}
                                    </select>
                                ) : <span style={{ color: 'var(--text-2)' }}>{q.settings.engine || <em style={{ color: 'var(--text-mute)' }}>inherit</em>}</span>}
                            </div>
                            <div className="field-row" style={{ margin: 0 }}>
                                <label style={{ minWidth: 60 }}>Voice</label>
                                {isEd ? (
                                    <select value={ed.voice || ''}
                                        onChange={e => setField(q.queue, 'voice', e.target.value)}>
                                        {engineVoices.map(v => <option key={v.id} value={v.id}>{v.id}</option>)}
                                    </select>
                                ) : <span style={{ color: 'var(--text-2)' }}>{q.settings.voice || <em style={{ color: 'var(--text-mute)' }}>inherit</em>}</span>}
                            </div>
                            <div className="field-row" style={{ margin: 0, alignItems: 'center' }}>
                                <label style={{ minWidth: 60 }}>Color</label>
                                {isEd ? (
                                    <>
                                        <input
                                            type="color"
                                            value={color}
                                            onChange={e => setField(q.queue, 'color', e.target.value)}
                                            style={{ flex: '0 0 44px', height: 28, padding: 0, border: 'none', background: 'transparent', cursor: 'pointer' }}
                                            title="Per-queue accent color"
                                        />
                                        <button
                                            className="btn subtle"
                                            onClick={() => setField(q.queue, 'color', '')}
                                            title="Clear -- fall back to auto-derived color"
                                            style={{ marginLeft: 6 }}
                                        >Auto</button>
                                    </>
                                ) : (
                                    <span
                                        title={q.settings.color}
                                        style={{
                                            display: 'inline-block',
                                            width: 24, height: 24, borderRadius: 5,
                                            background: q.settings.color,
                                            border: '1px solid var(--border)',
                                        }}
                                    ></span>
                                )}
                            </div>
                            <div className="field-row" style={{ margin: 0 }}>
                                <label style={{ minWidth: 60 }}>Effects</label>
                                {isEd ? (
                                    <select
                                        value={presetValue}
                                        onChange={e => setField(q.queue, 'effects_preset', e.target.value)}
                                    >
                                        <option value="inherit">(global default)</option>
                                        {presets.map(p => (
                                            <option key={p.name} value={p.name}>
                                                {p.name}{p.builtin ? '' : ' (custom)'}
                                            </option>
                                        ))}
                                    </select>
                                ) : <span style={{ color: 'var(--text-2)' }}>
                                    {q.settings.effects_preset || <em style={{ color: 'var(--text-mute)' }}>(global)</em>}
                                </span>}
                            </div>
                        </div>
                        <div className="card-footer" style={{ marginTop: 12 }}>
                            <span className="save-success">{saveStatus[q.queue] || tryStatus[q.queue] || ''}</span>
                            <div style={{ display: 'flex', gap: 6, marginLeft: 'auto' }}>
                                {!isEd && (
                                    <button
                                        className="btn-try"
                                        onClick={() => tryRow(q)}
                                        title="Speak the sample phrase using this queue's voice"
                                    >Try</button>
                                )}
                                {isEd ? (
                                    <>
                                        <button className="btn subtle" onClick={cancelEdit}>Cancel</button>
                                        <button className="btn" onClick={() => save(q.queue)}>Save</button>
                                    </>
                                ) : (
                                    <button className="btn subtle" onClick={() => startEdit(q)}>Edit</button>
                                )}
                            </div>
                        </div>
                    </div>
                );
            })}
        </div>
    );
}

function PerQueueTable({ queues, engines, presets, editing, pending, saveStatus, setField, startEdit, cancelEdit, save, tryRow, tryStatus }) {
    return (
        <table className="history-table">
            <thead>
                <tr>
                    <th>Queue</th>
                    <th style={{ width: 130 }}>Engine</th>
                    <th style={{ width: 160 }}>Voice</th>
                    <th style={{ width: 70 }}>Color</th>
                    <th style={{ width: 140 }}>Effects</th>
                    <th style={{ width: 70 }}>Msgs</th>
                    <th style={{ width: 230 }}></th>
                </tr>
            </thead>
            <tbody>
                {queues.map(q => {
                    const isEd = editing === q.queue;
                    const ed = pending[q.queue] || q.settings;
                    const engineVoices = engines.find(e => e.name === (ed.engine || q.settings.engine))?.voices || [];
                    const color = ed.color || q.settings.color || '#7a8694';
                    const rowStyle = { '--queue-color': color };
                    const presetValue = ed.effects_preset || 'inherit';
                    return (
                        <tr key={q.queue} style={rowStyle}>
                            <td>
                                {/* Per-queue overrides: full queue id is shown
                                    untruncated since this view IS the place
                                    where the user identifies and edits a
                                    queue's settings -- truncating defeats
                                    the purpose. The chip's left stripe still
                                    carries the queue color. */}
                                <code
                                    className="queue-chip queue-chip-full"
                                    title={q.queue}
                                    style={{ borderLeftColor: color }}
                                >{q.queue}</code>
                            </td>
                            <td>
                                {isEd ? (
                                    <select value={ed.engine || ''}
                                        onChange={e => setField(q.queue, 'engine', e.target.value)}>
                                        {engines.map(en => <option key={en.name} value={en.name}>{en.name}</option>)}
                                    </select>
                                ) : <span>{q.settings.engine || <em style={{ color: 'var(--text-mute)' }}>inherit</em>}</span>}
                            </td>
                            <td>
                                {isEd ? (
                                    <select value={ed.voice || ''}
                                        onChange={e => setField(q.queue, 'voice', e.target.value)}>
                                        {engineVoices.map(v => <option key={v.id} value={v.id}>{v.id}</option>)}
                                    </select>
                                ) : <span>{q.settings.voice || <em style={{ color: 'var(--text-mute)' }}>inherit</em>}</span>}
                            </td>
                            <td>
                                {isEd ? (
                                    <input
                                        type="color"
                                        value={color}
                                        onChange={e => setField(q.queue, 'color', e.target.value)}
                                        style={{ width: 36, height: 26, padding: 0, border: 'none', background: 'transparent', cursor: 'pointer' }}
                                        title="Per-queue accent color"
                                    />
                                ) : (
                                    <span
                                        title={q.settings.color}
                                        style={{
                                            display: 'inline-block',
                                            width: 20, height: 20, borderRadius: 4,
                                            background: q.settings.color,
                                            border: '1px solid var(--border)',
                                            verticalAlign: 'middle',
                                        }}
                                    ></span>
                                )}
                            </td>
                            <td>
                                {isEd ? (
                                    <select
                                        value={presetValue}
                                        onChange={e => setField(q.queue, 'effects_preset', e.target.value)}
                                    >
                                        <option value="inherit">(global)</option>
                                        {presets.map(p => (
                                            <option key={p.name} value={p.name}>
                                                {p.name}{p.builtin ? '' : ' (custom)'}
                                            </option>
                                        ))}
                                    </select>
                                ) : <span>{q.settings.effects_preset || <em style={{ color: 'var(--text-mute)' }}>(global)</em>}</span>}
                            </td>
                            <td>{q.total_messages}</td>
                            <td>
                                {!isEd && (
                                    <button
                                        className="btn-try"
                                        onClick={() => tryRow(q)}
                                        style={{ marginRight: 4 }}
                                        title="Speak the sample phrase using this queue's voice"
                                    >Try</button>
                                )}
                                {isEd ? (
                                    <>
                                        <button className="btn" onClick={() => save(q.queue)}>Save</button>
                                        <button className="btn subtle" onClick={cancelEdit} style={{ marginLeft: 4 }}>Cancel</button>
                                    </>
                                ) : (
                                    <button className="btn subtle" onClick={() => startEdit(q)}>Edit</button>
                                )}
                                <span className="save-success" style={{ marginLeft: 6 }}>
                                    {saveStatus[q.queue] || tryStatus[q.queue] || ''}
                                </span>
                            </td>
                        </tr>
                    );
                })}
            </tbody>
        </table>
    );
}

ReactDOM.createRoot(document.getElementById('root')).render(<App />);
    </script>
</body>
</html>
"""


def format_time(iso_str: str | None) -> str:
    """Format ISO timestamp for display."""
    if not iso_str:
        return "-"
    try:
        dt = datetime.fromisoformat(iso_str)
        return dt.strftime("%b %d %H:%M")
    except (ValueError, TypeError):
        return iso_str


def escape_html(text: str) -> str:
    """Escape text for HTML display."""
    return (
        text.replace('&', '&amp;')
        .replace('<', '&lt;')
        .replace('>', '&gt;')
        .replace('"', '&quot;')
    )


def sanitize_key(key: str) -> str:
    """Sanitize metadata key for display."""
    return escape_html(str(key))


def sanitize_value(value) -> str:
    """Sanitize metadata value for display."""
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        import json
        return escape_html(json.dumps(value, default=str))
    return escape_html(str(value))


def interpretation_class(metadata: dict | None) -> str:
    """Map the item's interpretation outcome to a CSS modifier class.

    Returns one of ``"interp-success"``, ``"interp-error"``, ``"interp-other"``,
    or ``""`` (no accent). The mapping is intentionally case-insensitive on
    the built-in SUCCESS/ERROR names; any other configured interpretation
    name shows as the neutral amber accent so the user can still tell it
    carried *some* outcome cue.
    """
    if not metadata:
        return ""
    interp = metadata.get("interpretation")
    if not isinstance(interp, str) or not interp.strip():
        return ""
    name = interp.strip().upper()
    if name == "SUCCESS":
        return "interp-success"
    if name == "ERROR":
        return "interp-error"
    return "interp-other"


@lru_cache(maxsize=256)
def _queue_color_for(queue: str | None) -> str:
    """Per-queue accent color, cached. Resolves through get_settings so
    explicit colors win over the auto-derived default.

    The cache size of 256 fits most projects; collisions just trigger an
    extra DB read. Cache is process-local so a daemon restart picks up
    edits via the next page load.
    """
    s = get_settings(queue if queue != "default" else None)
    return s.get("color") or "#7a8694"


def render_metadata(metadata: dict | None) -> str:
    """Render the message's spoken label for the card footer.

    Only the ``display_name`` (or, as fallback, ``queue``) is shown, because
    interpretation already has a colored-border treatment and the other
    metadata keys (``engine``, ``voice``, ``polly_engine``, ``ssml`` etc.)
    are implementation details that just added noise to every row.

    A whitespace-only display_name is treated as absent so the queue id can
    surface instead -- callers that pass an empty placeholder shouldn't
    erase the meaningful queue label.
    """
    if not metadata:
        return '<span class="no-data">-</span>'

    def _clean(v) -> str | None:
        if not isinstance(v, str):
            return None
        v = v.strip()
        return v or None

    label = _clean(metadata.get("display_name")) or _clean(metadata.get("queue"))
    if label is None:
        return '<span class="no-data">-</span>'

    return f'<span class="kv"><span class="value">{sanitize_value(label)}</span></span>'


@router.get("/", response_class=HTMLResponse)
async def index(q: str | None = None):
    """Serve the React single-page app shell.

    The page is a static React mount; the app fetches initial data from
    ``/api/items`` (and the other ``/api/*`` endpoints) so this handler
    does no server-side row rendering anymore. The legacy ``?q=`` query
    param is preserved as a no-op for backward-compatible deep links --
    search now happens client-side in the React UI.
    """
    del q  # legacy compatibility; search is client-side now.
    return HTMLResponse(content=HTML_TEMPLATE)


class GenerateRequest(BaseModel):
    """Ad-hoc TTS generation request.

    All fields except ``text`` are optional and fall back to the saved
    global defaults. Generation is fully synchronous and bypasses the
    daemon (no audio plays through the system speakers) -- the endpoint
    writes the WAV to disk, records it in the queue history, and
    returns URLs for preview + download.
    """
    text: str
    engine: str | None = None
    voice: str | None = None
    speed: float | None = None


@router.post("/api/generate")
async def api_generate(body: GenerateRequest):
    """Ad-hoc text-to-speech.

    Synthesizes synchronously (no daemon, no playback). Creates a queue
    row in the ``generated`` session so the result shows up in queue
    history and benefits from the standard /audio + /download endpoints.
    Returns ``queue_id`` and convenience URLs.
    """
    from datetime import datetime, timezone

    from .cli import start_player  # noqa: F401 - import for parity with other endpoints
    from .player import generate_tts, get_audio_save_path
    from .queue_db import enqueue, get_connection, get_settings

    text = (body.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text cannot be empty")
    if len(text) > 5000:
        raise HTTPException(status_code=413, detail="text exceeds the 5000 character limit")
    if body.engine and body.engine not in _KNOWN_ENGINES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown engine {body.engine!r}. Known: {', '.join(sorted(_KNOWN_ENGINES))}",
        )
    if body.voice and body.voice not in _known_voices():
        raise HTTPException(status_code=400, detail=f"Unknown voice {body.voice!r}")

    # Resolve missing axes from the saved global defaults so the
    # generated audio uses the user's preferred engine/voice even when
    # they don't specify them in the request.
    settings = get_settings(None)
    engine = body.engine or settings.get("engine") or "polly"
    voice = body.voice or settings.get("voice") or settings.get("voice")
    speed = body.speed if isinstance(body.speed, (int, float)) and body.speed > 0 else float(settings.get("speed", 1.0))

    metadata: dict[str, str] = {"queue": "generated", "engine": engine, "voice": voice}

    # Enqueue first so we get an id; the per-item audio path is derived
    # from the id. We then update the row with the file path after
    # synthesis succeeds (or mark it played without a path on failure).
    queue_id = enqueue(text, metadata=metadata)
    save_path = get_audio_save_path(queue_id)

    try:
        result = generate_tts(
            text,
            engine=engine,
            voice=voice,
            speed=speed,
            save_path=save_path,
            verbose=False,
        )
    except Exception as e:
        with get_connection() as conn:
            conn.execute(
                "UPDATE queue SET played_at = ? WHERE id = ?",
                (datetime.now(timezone.utc).isoformat(), queue_id),
            )
            conn.commit()
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    if result is None or not Path(save_path).exists():
        with get_connection() as conn:
            conn.execute(
                "UPDATE queue SET played_at = ? WHERE id = ?",
                (datetime.now(timezone.utc).isoformat(), queue_id),
            )
            conn.commit()
        raise HTTPException(status_code=500, detail="Generation produced no audio")

    # Mark played + record audio_path so /audio/<id> and /download/<id>
    # both serve the file. The daemon never sees this row.
    with get_connection() as conn:
        conn.execute(
            "UPDATE queue SET played_at = ?, audio_path = ? WHERE id = ?",
            (datetime.now(timezone.utc).isoformat(), str(save_path), queue_id),
        )
        conn.commit()

    size = Path(save_path).stat().st_size
    return JSONResponse({
        "queue_id": queue_id,
        "engine": engine,
        "voice": voice,
        "speed": speed,
        "char_count": len(text),
        "size_bytes": size,
        "audio_url": f"/audio/{queue_id}",
        "download_wav": f"/download/{queue_id}?format=wav",
        "download_mp3": f"/download/{queue_id}?format=mp3",
    })


def _resolve_audio_for_item(item: dict) -> Path | None:
    """Resolve a playable audio file for a queue row.

    Mirrors /audio/<id>'s fallback chain so /download/<id> can stream
    the same bytes:

    1. Stored ``audio_path`` if the file exists on disk.
    2. Tone-only synthesis (cached chord WAV) for rows whose text is
       just ``$Note`` tokens.
    3. ``None`` if neither resolves.
    """
    raw = item.get("audio_path")
    if raw:
        p = Path(raw)
        if p.exists():
            return p
    try:
        from .player import extract_tone_tokens, generate_combined_tones_from_tokens
        text = item.get("text") or ""
        leading, body, _trailing = extract_tone_tokens(text)
        if leading and not body:
            meta = item.get("metadata") or {}
            td = meta.get("tone_duration")
            duration = (
                float(td)
                if isinstance(td, (int, float)) and not isinstance(td, bool) and td > 0
                else 0.8
            )
            cache_path = generate_combined_tones_from_tokens(leading, duration=duration)
            if cache_path and Path(cache_path).exists():
                return Path(cache_path)
    except Exception:
        pass
    return None


_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _download_filename(item: dict, item_id: int, ext: str) -> str:
    """Build a friendly download filename, e.g. ``speeker-14482-compass-docs-2026-05-30.wav``."""
    queue = (item.get("session_id") or "default")
    queue = _SAFE_FILENAME_RE.sub("-", queue).strip("-") or "default"
    created = item.get("created_at") or ""
    date_part = created[:10] if created and len(created) >= 10 else "unknown"
    return f"speeker-{item_id}-{queue}-{date_part}.{ext}"


@router.get("/download/{item_id}")
async def download_audio(item_id: int, format: str = "wav"):
    """Download a queue item's audio with attachment disposition.

    ``format=wav`` (default) streams the source WAV unchanged.
    ``format=mp3`` requires ``ffmpeg`` on PATH; the WAV is transcoded
    in-memory and streamed. Other formats return 400.

    Returns 404 for rows that don't have a playable audio resolution
    (TTS generation failed, daemon never finished, etc.).
    """
    history = get_history(limit=1000)
    item = next((it for it in history if it["id"] == item_id), None)
    if item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    src = _resolve_audio_for_item(item)
    if src is None:
        raise HTTPException(status_code=404, detail="No audio for this item")

    fmt = (format or "wav").lower().strip()
    if fmt == "wav":
        return FileResponse(
            src,
            media_type="audio/wav",
            filename=_download_filename(item, item_id, "wav"),
        )

    if fmt == "mp3":
        import shutil
        import subprocess
        from fastapi.responses import StreamingResponse

        # launchd-managed processes get a minimal PATH that often
        # doesn't include /opt/homebrew/bin or /usr/local/bin where
        # Homebrew installs ffmpeg. Check the well-known macOS install
        # locations explicitly before giving up.
        ffmpeg = (
            shutil.which("ffmpeg")
            or next(
                (p for p in (
                    "/opt/homebrew/bin/ffmpeg",
                    "/usr/local/bin/ffmpeg",
                    "/usr/bin/ffmpeg",
                ) if Path(p).exists()),
                None,
            )
        )
        if not ffmpeg:
            raise HTTPException(
                status_code=415,
                detail="MP3 export needs ffmpeg (brew install ffmpeg).",
            )
        proc = subprocess.Popen(
            [
                ffmpeg, "-loglevel", "error", "-i", str(src),
                "-codec:a", "libmp3lame", "-q:a", "4", "-f", "mp3", "pipe:1",
            ],
            stdout=subprocess.PIPE,
        )
        filename = _download_filename(item, item_id, "mp3")
        return StreamingResponse(
            proc.stdout,
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
            },
        )

    raise HTTPException(
        status_code=400,
        detail=f"Unknown format {fmt!r}. Use 'wav' or 'mp3'.",
    )


@router.get("/audio/{item_id}")
async def get_audio(item_id: int):
    """Serve audio for a queue item.

    Resolution order:

    1. ``audio_path`` from the DB if the file exists on disk (the common
       TTS case -- per-item WAV under the speeker audio dir).
    2. For tone-only items (text is just ``$Note`` tokens), the cached
       tone WAV under ``~/Library/Caches/speeker/tones/`` is rendered
       on demand via ``generate_combined_tones_from_tokens`` and served.
       Multiple rows with the same notes share the same cache file, so
       the call is virtually free after the first hit.

    Returns 404 only when neither resolves -- e.g. a row that was
    marked played without the daemon ever producing audio (the
    emergency purge case).
    """
    history = get_history(limit=1000)
    item = next((it for it in history if it["id"] == item_id), None)
    if item is None:
        return HTMLResponse(content="Audio not found", status_code=404)
    src = _resolve_audio_for_item(item)
    if src is None:
        return HTMLResponse(content="Audio not found", status_code=404)
    return FileResponse(src, media_type="audio/wav")


@router.get("/api/items")
async def api_items():
    """JSON endpoint for real-time updates.

    Also reports ``speaking_id`` -- the queue row the daemon is reading
    aloud at this moment. The hash includes it so the React poll picks up
    the highlight transitions without waiting for the next item-set change.
    """
    items = get_history(limit=200)
    speaking_id = get_currently_playing()

    hash_input = json.dumps(
        [(i["id"], i["played_at"]) for i in items] + [("speaking", speaking_id)]
    ).encode()
    items_hash = hashlib.md5(hash_input).hexdigest()[:8]

    result = []
    for item in items:
        # has_audio is true when /audio/<id> would return a real WAV --
        # either the stored audio_path exists, OR the row is tone-only
        # and the cached chord WAV can be served on demand. We don't
        # actually synthesize here; we just check that the text would
        # take the tone-only branch in /audio/{item_id}.
        has_audio = False
        if item["audio_path"] and Path(item["audio_path"]).exists():
            has_audio = True
        elif item.get("text"):
            try:
                from .player import extract_tone_tokens
                leading, body, _trail = extract_tone_tokens(item["text"])
                if leading and not body:
                    has_audio = True
            except Exception:
                pass
        queue = item.get("session_id") or "default"

        # Epoch milliseconds for the client-side date-range filter. Cheaper
        # than re-parsing item.time on every render, and ISO strings can be
        # ambiguous about timezone.
        created_at_ms = None
        if item.get("created_at"):
            try:
                ts = datetime.fromisoformat(item["created_at"])
                created_at_ms = int(ts.timestamp() * 1000)
            except (ValueError, TypeError):
                pass

        # TTS failure surfaced from the daemon's retry policy. Present
        # only after the attempt cap is reached -- before that, the row
        # stays pending and retries on each poll. The UI uses this to
        # render a "TTS failed" badge next to the (disabled) Play button
        # so users understand why audio is missing.
        meta_dict = item.get("metadata") or {}
        tts_error_msg = meta_dict.get("tts_error") if isinstance(meta_dict, dict) else None

        # Per-queue accent color. Used by the UI for the card's left
        # stripe and the table row's queue chip. Caching keyed on the
        # queue id so we don't query settings per item.
        queue_color = _queue_color_for(queue)

        result.append({
            "id": item["id"],
            "text": escape_html(strip_tone_tokens(item["text"])),
            # Raw text (without tone tokens stripped) for the copy button.
            # Kept separate from ``text`` so the displayed version can
            # have HTML-escaped highlighting without polluting clipboard.
            "raw_text": strip_tone_tokens(item["text"]),
            "time": format_time(item["created_at"]),
            "created_at_ms": created_at_ms,
            "played": bool(item["played_at"]),
            "has_audio": has_audio,
            "metadata": render_metadata(item.get("metadata")),
            "queue": queue,
            "queue_color": queue_color,
            # CSS modifier for the interpretation accent stripe (left border).
            # Empty string when no outcome cue was set on the item.
            "interp_class": interpretation_class(item.get("metadata")),
            "tts_error": tts_error_msg if isinstance(tts_error_msg, str) else None,
        })

    return JSONResponse({
        "hash": items_hash,
        "items": result,
        "speaking_id": speaking_id,
    })


# ---------------------------------------------------------------------------
# JSON API used by the React UI. Kept separate from the legacy /settings
# server-rendered form so the React app can be added without breaking
# existing form-post callers.
# ---------------------------------------------------------------------------


def _engine_supports_ssml(engine: str) -> bool:
    """Which engines support real SSML (vs only flat preprocessing)."""
    return engine == "polly"  # only Polly currently honors SSML phoneme tags


@router.get("/api/engines")
async def api_engines():
    """List available TTS engines with their voices and SSML support."""
    return JSONResponse({
        "engines": [
            {
                "name": "polly",
                "label": "Amazon Polly (cloud, SSML-capable)",
                "supports_ssml": True,
                "default_voice": get_polly_config().get("voice") or "Joanna",
                "voices": [{"id": k, "label": v} for k, v in POLLY_VOICES.items()],
                "polly_engine": get_polly_config().get("engine") or "neural",
            },
            {
                "name": "pocket-tts",
                "label": "pocket-tts (local, no SSML)",
                "supports_ssml": False,
                "default_voice": "azelma",
                "voices": [{"id": k, "label": v} for k, v in POCKET_TTS_VOICES.items()],
                "polly_engine": None,
            },
            {
                "name": "kokoro",
                "label": "kokoro (local, no SSML)",
                "supports_ssml": False,
                "default_voice": "am_liam",
                "voices": [{"id": k, "label": v} for k, v in KOKORO_VOICES.items()],
                "polly_engine": None,
            },
        ],
    })


@router.get("/api/pronunciation")
async def api_get_pronunciation():
    """Return current user-supplied pronunciation overrides."""
    return JSONResponse({"overrides": get_pronunciation_overrides()})


class PronunciationUpdate(BaseModel):
    """Replace the entire overrides dict with a new mapping.

    Each value may be either a single string (applies to every engine) or
    a ``{engine: replacement}`` dict with an optional ``"default"`` key
    for the fallback.
    """
    overrides: dict[str, str | dict[str, str]]


@router.put("/api/pronunciation")
async def api_put_pronunciation(body: PronunciationUpdate):
    """Persist a new pronunciation overrides dict to config.json.

    Accepts two value shapes per entry:

    - ``"kom-piss"``                                  -- universal
    - ``{"polly": "...", "default": "kom-piss"}``     -- per-engine

    The dict fully replaces the existing one -- callers should send the
    complete desired state, not a partial patch. Empty keys/values are
    discarded so a half-typed row in the UI doesn't poison the config.
    """
    cleaned: dict[str, str | dict[str, str]] = {}
    for k, v in body.overrides.items():
        if not isinstance(k, str) or not k.strip():
            continue
        word = k.strip()
        if isinstance(v, str):
            if v.strip():
                cleaned[word] = v.strip()
        elif isinstance(v, dict):
            inner: dict[str, str] = {}
            for ek, ev in v.items():
                if not isinstance(ek, str) or not isinstance(ev, str):
                    continue
                if not ek.strip() or not ev.strip():
                    continue
                inner[ek.strip()] = ev.strip()
            if inner:
                cleaned[word] = inner

    cfg = get_config()
    cfg.setdefault("pronunciation", {})["overrides"] = cleaned
    save_config(cfg)
    return JSONResponse({
        "overrides": cleaned,
        "restart_required": False,
        "message": "Saved. Takes effect on the next utterance.",
    })


# Curated public-domain tunes / short signals for intro/outro tones.
# Notes use the inline ``[A-G][b#]?[0-8]`` notation already enforced by
# ``_NOTE_RE``. Kept short (≤16 notes) so an intro/outro stays under
# ~2 seconds at the default 0.12s/note duration. Names are
# user-recognizable; descriptions hint at origin where useful.
# Each note is ``<pitch><accidental?><octave>[:<multiplier>]``. The
# multiplier scales that note's duration relative to the base (1.0 by
# default). Several iconic tunes have a long final note -- NBC chimes,
# Beethoven 5th, Charge fanfare -- and need that explicitly.
TONE_TUNES: dict[str, dict] = {
    "Rising major triad":   {"notes": ["E4", "G4", "C5"],          "note": "default intro"},
    "Falling major triad":  {"notes": ["C5", "G4", "E4"],          "note": "default outro"},
    # NBC: G E with the C ringing out (~3x). Without the multiplier, all
    # three notes are equal length and the signature isn't recognizable.
    "NBC chimes":           {"notes": ["G4", "E4", "C5:6"],        "note": "three-note signature (last note sustained)"},
    # Westminster: long-long-long-LONG (final G3 holds longer).
    "Westminster quarters": {"notes": ["E4", "C4:1.5", "D4:1.5", "G3:3"], "note": "first phrase of the Cambridge chime"},
    # Three short Gs, sustained Eb -- the classic da-da-da-DUM.
    "Beethoven 5th":        {"notes": ["G4:.5", "G4:.5", "G4:.5", "Eb4:3"], "note": "da-da-da-DUM"},
    # Charge: building short notes ending in two longer ones.
    "Charge fanfare":       {"notes": ["G4:.5", "C5:.5", "E5:.5", "G5", "E5", "G5:2"], "note": "sports stadium"},
    "Mary had a little lamb": {"notes": ["E5", "D5", "C5", "D5", "E5", "E5", "E5:2"], "note": "first phrase"},
    "Twinkle, twinkle":     {"notes": ["C5", "C5", "G5", "G5", "A5", "A5", "G5:2"],   "note": "first phrase"},
    "Ode to Joy":           {"notes": ["E5", "E5", "F5", "G5", "G5", "F5", "E5", "D5"], "note": "Beethoven 9th"},
    "Big Ben":              {"notes": ["E4", "C4", "D4", "G3:2", "G3", "D4", "E4", "C4:2"], "note": "full chime"},
}


@router.get("/api/tones/tunes")
async def api_get_tone_tunes():
    """Catalog of public-domain tunes for intro/outro presets.

    Each entry returns its note list plus a short descriptor so the UI
    can render a dropdown without baking the data into JS.
    """
    return JSONResponse({
        "tunes": [
            {"name": name, "notes": data["notes"], "note": data.get("note", "")}
            for name, data in TONE_TUNES.items()
        ],
    })


class TonePlay(BaseModel):
    """Inputs for /api/tones/play.

    Either:
      - ``kind: "intro"|"outro"`` -- play the *currently saved* notes for
        that role at the saved duration (the user can hear how the
        configured chord sounds right now), OR
      - ``notes`` (and optional ``duration``) -- play an arbitrary
        sequence with optional per-note duration (used by the UI to
        preview unsaved edits and tune-preset picks).

    When both are provided, the explicit ``notes`` wins.
    """
    kind: str | None = None
    notes: list[str] | None = None
    duration: float | None = None


@router.post("/api/tones/play")
async def api_tones_play(body: TonePlay):
    """Synthesize a sequence of notes and enqueue for immediate playback.

    Notes are validated as ``[A-G][b#]?[0-8]``; bad inputs return 400.
    The enqueued item has empty body text -- the player extracts the
    leading ``$Note`` tokens, plays them via ``play_tone_tokens`` at the
    requested duration (passed through ``metadata.tone_duration``), and
    skips TTS entirely.
    """
    from .queue_db import enqueue
    from .cli import start_player

    notes: list[str] = []
    duration = body.duration

    if body.notes:
        notes = _validate_notes(body.notes)
    elif body.kind in ("intro", "outro"):
        cfg = get_tones_config()
        raw = cfg.get(body.kind, [])
        if not isinstance(raw, list) or not raw:
            raise HTTPException(
                status_code=400,
                detail=f"No notes configured for {body.kind!r}.",
            )
        notes = _validate_notes(raw)
        if duration is None:
            try:
                duration = float(cfg.get("duration_seconds", 0.12))
            except (TypeError, ValueError):
                duration = 0.12
    else:
        raise HTTPException(
            status_code=400,
            detail="Supply either notes=[...] or kind='intro'|'outro'.",
        )

    if not notes:
        raise HTTPException(status_code=400, detail="No valid notes to play.")

    text = " ".join(f"${n}" for n in notes)
    metadata: dict = {"queue": "default"}
    if duration is not None:
        # Clamp to a sane range so a user mis-entering 100 doesn't lock
        # the daemon into a multi-minute tone.
        metadata["tone_duration"] = max(0.02, min(2.0, float(duration)))
    queue_id = enqueue(text, metadata=metadata)
    start_player()
    return JSONResponse({
        "queue_id": queue_id,
        "notes": notes,
        "duration": metadata.get("tone_duration"),
    })


class EngineTry(BaseModel):
    """Preview a (possibly unsaved) engine + voice + speed selection.

    All fields optional -- omitted axes fall back to the saved global
    setting via the daemon's normal metadata-overrides path in
    ``process_queue``. When ``include_tones`` is true, the preview is
    wrapped with the configured intro and outro $Note tokens so the
    user hears them around the voice line. When ``text`` is supplied,
    it replaces the default preview phrase -- used by the per-queue
    overrides section so the listener can A/B voices on a phrase
    they actually care about.
    """
    engine: str | None = None
    voice: str | None = None
    speed: float | None = None
    include_tones: bool = False
    text: str | None = None


_VOICE_TRY_PHRASE = "This is a preview of the selected voice."


# Tags + attributes Polly supports. Used by the SSML lint helper below;
# kept narrow because flagging "Polly may not honor <foo>" is most of
# what makes the lint useful. Source: the SSML reference page linked in
# the UI.
_POLLY_SSML_TAGS: set[str] = {
    "speak", "p", "s", "break", "phoneme", "say-as",
    "sub", "emphasis", "lang", "mark", "prosody",
    "amazon:auto-breaths", "amazon:breath", "amazon:domain",
    "amazon:effect", "w",
}


class SSMLTry(BaseModel):
    """Try a chunk of SSML through speeker -> Polly."""
    ssml: str
    engine: str | None = None
    voice: str | None = None


def _lint_ssml(text: str) -> dict:
    """Best-effort SSML lint. Returns a JSON-friendly summary with errors,
    warnings, the parsed text content, and a structural outline.

    XML parse errors are surfaced with their line/column so the editor
    can highlight the problem. Tag warnings are issued for elements that
    Polly doesn't honor (e.g. <voice>). The function never raises;
    callers can rely on the returned ``ok`` flag.
    """
    import xml.etree.ElementTree as ET

    raw = (text or "").strip()
    if not raw:
        return {"ok": False, "errors": [{"message": "Empty input."}], "warnings": []}
    if not raw.lstrip().startswith("<"):
        return {
            "ok": False,
            "errors": [{"message": "SSML must start with a tag, e.g. <speak>...</speak>."}],
            "warnings": [],
        }
    # Polly requires a <speak> root.
    wrapped = raw if raw.lstrip().startswith("<speak") else f"<speak>{raw}</speak>"
    errors: list[dict] = []
    warnings: list[dict] = []
    root = None
    try:
        root = ET.fromstring(wrapped)
    except ET.ParseError as e:
        line, col = (e.position if hasattr(e, "position") else (None, None))
        errors.append({"message": str(e), "line": line, "col": col})
        return {"ok": False, "errors": errors, "warnings": warnings}

    if root.tag != "speak":
        errors.append({"message": f"Root element must be <speak>, not <{root.tag}>"})
        return {"ok": False, "errors": errors, "warnings": warnings}

    # Walk and lint.
    def _walk(el):
        tag = el.tag
        # Strip namespace if any (we accept and ignore it for the lint).
        if isinstance(tag, str) and "}" in tag:
            tag = tag.split("}", 1)[1]
        if tag not in _POLLY_SSML_TAGS:
            warnings.append({
                "message": f"<{tag}> is not in the list of tags Polly is known to honor; output may strip it.",
            })
        for child in el:
            _walk(child)
    _walk(root)

    # Approximate "what will be spoken": all text content concatenated.
    spoken = " ".join(t.strip() for t in (root.itertext()) if t and t.strip())
    return {
        "ok": True,
        "errors": errors,
        "warnings": warnings,
        "spoken_preview": spoken[:240],
    }


@router.post("/api/ssml/lint")
async def api_ssml_lint(body: SSMLTry):
    """Validate SSML without enqueueing. Returns ok/errors/warnings."""
    return JSONResponse(_lint_ssml(body.ssml))


@router.post("/api/ssml/try")
async def api_ssml_try(body: SSMLTry):
    """Lint, then enqueue the SSML for playback through Polly.

    Returns 400 with the parse error details if the SSML doesn't validate.
    Engine and voice are validated against the known sets when supplied
    -- only Polly honors SSML, so the default goes to Polly + Joanna
    even if the user hasn't set those globally.
    """
    from .queue_db import enqueue
    from .cli import start_player

    lint = _lint_ssml(body.ssml)
    if not lint.get("ok"):
        raise HTTPException(status_code=400, detail=lint)

    if body.engine and body.engine not in _KNOWN_ENGINES:
        raise HTTPException(status_code=400, detail=f"Unknown engine {body.engine!r}")
    if body.voice and body.voice not in _known_voices():
        raise HTTPException(status_code=400, detail=f"Unknown voice {body.voice!r}")

    engine = body.engine or "polly"
    voice = body.voice or get_polly_config().get("voice") or "Joanna"

    # Make sure the body is wrapped in <speak>. Polly insists on it; the
    # lint already accepted both wrapped and unwrapped forms, so we
    # normalize here so the daemon's SSML detector picks it up reliably.
    text = body.ssml.strip()
    if not text.lstrip().startswith("<speak"):
        text = f"<speak>{text}</speak>"

    metadata = {
        "queue": "default",
        "engine": engine,
        "voice": voice,
        "ssml": True,
    }
    queue_id = enqueue(text, metadata=metadata)
    start_player()
    return JSONResponse({
        "queue_id": queue_id,
        "engine": engine,
        "voice": voice,
        "warnings": lint.get("warnings", []),
    })


_KNOWN_ENGINES: frozenset[str] = frozenset(["polly", "pocket-tts", "kokoro"])


def _known_voices() -> set[str]:
    """Union of every voice id across all engines + any custom voices."""
    voices = set(POLLY_VOICES) | set(POCKET_TTS_VOICES) | set(KOKORO_VOICES)
    # Custom (cloned) voices: ``voices.get_voices()`` reports them under a
    # "custom" key. They're valid for pocket-tts; we accept them as-is.
    try:
        from .voices import get_voices
        custom = get_voices().get("custom", {}) or {}
        voices.update(custom.keys())
    except Exception:
        pass
    return voices


@router.post("/api/engines/try")
async def api_engines_try(body: EngineTry):
    """Enqueue a fixed phrase with per-item engine/voice/speed overrides.

    The daemon's per-item metadata lookup (``meta.get("engine")`` /
    ``meta.get("voice")`` in ``process_queue``) already takes precedence
    over the saved session settings, so a sample posted here speaks with
    the requested config for *this one utterance only* -- no config
    write, no daemon restart.

    Validates engine/voice up front so an unknown value returns 400 with
    a helpful list of options instead of silently enqueuing an item the
    daemon can't process.
    """
    from .queue_db import enqueue
    from .cli import start_player

    if body.engine and body.engine not in _KNOWN_ENGINES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown engine {body.engine!r}. "
                f"Known: {', '.join(sorted(_KNOWN_ENGINES))}."
            ),
        )
    if body.voice and body.voice not in _known_voices():
        raise HTTPException(
            status_code=400,
            detail=f"Unknown voice {body.voice!r}.",
        )

    metadata: dict[str, str] = {"queue": "default"}
    if body.engine:
        metadata["engine"] = body.engine
    if body.voice:
        metadata["voice"] = body.voice
    # Speed is on the per-queue settings row, not per-item metadata, so a
    # one-shot speed preview would require temporarily writing the global
    # speed setting. Skip for now.

    # Caller-supplied phrase (e.g. the per-queue overrides "Try" button)
    # overrides the default preview phrase. Trim + length-cap + repeat
    # collapse as cheap guards against accidentally enqueueing a giant
    # body or a wall of one character. The repeat collapse exists
    # because a 500-X payload spoken by Polly is half a minute of "ex"
    # repeated -- which audibly resembles a slur and is never anyone's
    # actual test goal.
    base_phrase = _VOICE_TRY_PHRASE
    if isinstance(body.text, str) and body.text.strip():
        candidate = body.text.strip()[:500]
        # Collapse any run of the same character > 5 to a single instance.
        # Real prose never repeats one character five times; test inputs
        # like "XXXXX..." do.
        candidate = re.sub(r"(.)\1{5,}", r"\1", candidate)
        if candidate.strip():
            base_phrase = candidate
    text = base_phrase
    if body.include_tones:
        cfg = get_tones_config()
        intro = cfg.get("intro", []) or []
        outro = cfg.get("outro", []) or []
        duration = cfg.get("duration_seconds", 0.12)
        intro_part = " ".join(f"${n}" for n in intro if isinstance(n, str))
        outro_part = " ".join(f"${n}" for n in outro if isinstance(n, str))
        text = " ".join(p for p in (intro_part, base_phrase, outro_part) if p)
        if isinstance(duration, (int, float)) and not isinstance(duration, bool) and duration > 0:
            metadata["tone_duration"] = max(0.02, min(2.0, float(duration)))

    queue_id = enqueue(text, metadata=metadata)
    start_player()
    return JSONResponse({
        "queue_id": queue_id,
        "spoken": text,
        "engine": body.engine,
        "voice": body.voice,
    })


@router.get("/api/settings/global")
async def api_global_settings():
    """Return the global default settings (engine, voice, speed, intro_sound)."""
    return JSONResponse(get_settings(None))


class SettingsUpdate(BaseModel):
    """Partial update: only the fields supplied are written.

    ``color`` accepts a hex string (``#aabbcc``) or empty string to clear
    the override (auto-derived color resumes). ``effects_preset`` accepts
    a known preset name or empty string to clear (global preset resumes).
    """
    engine: str | None = None
    voice: str | None = None
    speed: float | None = None
    intro_sound: bool | None = None
    color: str | None = None
    effects_preset: str | None = None
    session: str | None = None  # None -> __global__


@router.put("/api/settings")
async def api_put_settings(body: SettingsUpdate):
    """Update global or per-queue settings.

    ``session=None`` writes the global default. Pass a queue id to write a
    per-queue override (matches the existing /settings POST behavior).
    """
    # Validate color format if provided: hex like #rrggbb / #rgb, or empty
    # string to clear. Other inputs reject early so the UI surfaces a
    # consistent error before saving.
    if body.color is not None and body.color.strip():
        if not re.match(r"^#[0-9a-fA-F]{3}([0-9a-fA-F]{3})?$", body.color.strip()):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid color {body.color!r}; expected #rgb or #rrggbb hex.",
            )
    # Validate effects preset against known names (built-in + custom).
    if body.effects_preset is not None and body.effects_preset.strip():
        from .effects import preset_names
        if body.effects_preset not in preset_names():
            raise HTTPException(
                status_code=400,
                detail=f"Unknown preset {body.effects_preset!r}. Known: {', '.join(preset_names())}",
            )
    set_settings(
        session_id=body.session,
        engine=body.engine,
        voice=body.voice,
        speed=body.speed,
        intro_sound=body.intro_sound,
        color=body.color,
        effects_preset=body.effects_preset,
    )
    # Invalidate the per-queue color cache so /api/items immediately
    # picks up a color change without waiting for cache eviction.
    _queue_color_for.cache_clear()
    return JSONResponse(get_settings(body.session))


@router.get("/api/queues")
async def api_queues():
    """List queues that have any history, with their per-queue settings."""
    rows = get_all_sessions()
    result = []
    for row in rows:
        sid = row["session_id"]
        if sid in (None, "__global__"):
            continue
        result.append({
            "queue": sid,
            "total_messages": row["total_messages"],
            "pending": row["pending"],
            "last_activity": row["last_activity"],
            "settings": get_settings(sid),
        })
    return JSONResponse({"queues": result})


@router.get("/api/tones")
async def api_get_tones():
    """Get current intro/outro tone notes, duration, and effects toggle."""
    cfg = get_tones_config()
    return JSONResponse({
        "intro": cfg.get("intro", ["E4", "G4", "C5"]),
        "outro": cfg.get("outro", ["C5", "G4", "E4"]),
        "duration_seconds": cfg.get("duration_seconds", 0.12),
        "apply_effects": bool(cfg.get("apply_effects", True)),
    })


_NOTE_RE = re.compile(r"^[A-Ga-g][b#]?[0-8](?::[0-9]*\.?[0-9]+)?$")


class TonesUpdate(BaseModel):
    intro: list[str] | None = None
    outro: list[str] | None = None
    duration_seconds: float | None = None
    # When True, the active effects preset is baked into synthesized
    # tones (intro/outro/$Note/cue). When False, tones bypass the chain.
    apply_effects: bool | None = None


def _validate_notes(notes: list[str]) -> list[str]:
    """Strip whitespace, validate each note matches the inline notation."""
    cleaned = []
    for n in notes:
        if not isinstance(n, str):
            continue
        n = n.strip()
        if not n:
            continue
        if not _NOTE_RE.match(n):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid note {n!r} -- expected [A-G][b#]?[0-8] (e.g. 'E4', 'G#5', 'Bb3').",
            )
        cleaned.append(n)
    return cleaned


@router.put("/api/tones")
async def api_put_tones(body: TonesUpdate):
    """Persist intro/outro tone notes and duration.

    Note notation: ``[A-G][b#]?[0-8]`` -- e.g. ``E4``, ``G#5``, ``Bb3``.
    A trailing octave digit is required. Empty lists are accepted as a
    "no tone" signal. Duration is in seconds; values outside (0.02..2.0)
    are clamped to that range.
    """
    cfg = get_config()
    tones = cfg.setdefault("tones", {})
    if body.intro is not None:
        tones["intro"] = _validate_notes(body.intro)
    if body.outro is not None:
        tones["outro"] = _validate_notes(body.outro)
    if body.duration_seconds is not None:
        tones["duration_seconds"] = max(0.02, min(2.0, float(body.duration_seconds)))
    if body.apply_effects is not None:
        tones["apply_effects"] = bool(body.apply_effects)
    save_config(cfg)
    return JSONResponse({
        "intro": tones.get("intro", []),
        "outro": tones.get("outro", []),
        "duration_seconds": tones.get("duration_seconds", 0.12),
        "apply_effects": bool(tones.get("apply_effects", True)),
        "message": "Saved. Takes effect on the next intro/outro batch.",
    })


# ----- Per-queue / per-interpretation tone rules.
#
# A rule overrides the default intro/outro/cue notes for a specific queue,
# interpretation, or queue+interpretation pair. Resolution at speech time
# picks the highest-specificity matching rule; see tone_rules.py.

@router.get("/api/tone-rules")
async def api_get_tone_rules():
    """Return the saved tone_rules list plus catalog data for the editor.

    ``interpretations`` is the set of valid interpretation names (built-in
    + config map). ``queues`` is the known session ids from queue_db --
    handy for autocompletion but not enforced (the rule's queue field
    accepts any string, regex or exact).
    """
    queues = []
    for row in get_all_sessions():
        sid = row["session_id"]
        if sid in (None, "__global__"):
            continue
        queues.append(sid)
    return JSONResponse({
        "rules": get_tone_rules(),
        "interpretations": interpretation_names(),
        "queues": sorted(set(queues)),
    })


class ToneRulesUpdate(BaseModel):
    rules: list[dict]


@router.put("/api/tone-rules")
async def api_put_tone_rules(body: ToneRulesUpdate):
    """Replace the saved tone_rules with the supplied list.

    Each rule is validated and normalized: bad rules return 400 with the
    offending index, so a malformed UI submission never silently drops
    rows. Saving takes effect on the next utterance -- the resolver
    re-reads config per call.
    """
    normalized: list[dict] = []
    for idx, raw in enumerate(body.rules):
        ok, err = validate_rule(raw)
        if not ok:
            raise HTTPException(
                status_code=400,
                detail=f"Rule {idx}: {err}",
            )
        normalized.append(normalize_rule(raw))
    cfg = get_config()
    cfg["tone_rules"] = normalized
    save_config(cfg)
    return JSONResponse({
        "rules": normalized,
        "message": "Saved. Takes effect on the next utterance.",
    })


@router.get("/api/interpretations")
async def api_get_interpretations():
    """List known interpretation names (built-in SUCCESS/ERROR plus map keys)."""
    return JSONResponse({"interpretations": interpretation_names()})


@router.get("/api/effects")
async def api_get_effects():
    """List built-in effect presets and the currently-active one.

    The chain itself is not exposed verbatim -- the UI only renders names
    + descriptions. Power users can still inspect ``effects.PRESETS`` in
    source.
    """
    from .effects import all_presets, is_builtin_preset
    current = get_effects_config().get("preset") or "off"
    presets_map = all_presets()
    if current not in presets_map:
        current = "off"
    return JSONResponse({
        "current": current,
        "presets": [
            {
                "name": name,
                "description": PRESET_DESCRIPTIONS.get(name, "Custom preset."),
                "effect_count": len(presets_map[name]),
                "builtin": is_builtin_preset(name),
            }
            for name in preset_names()
        ],
    })


class EffectsUpdate(BaseModel):
    preset: str


@router.put("/api/effects")
async def api_put_effects(body: EffectsUpdate):
    """Persist the active preset name to config.json.

    Validates against both built-in PRESETS and custom_presets. Takes
    effect on the next utterance -- no daemon restart needed because
    ``apply_effects`` re-reads ``config.effects`` per call and the
    ``_build_board`` LRU cache rebuilds when the preset changes.
    """
    if body.preset not in preset_names():
        raise HTTPException(
            status_code=400,
            detail=f"Unknown preset {body.preset!r}. Known: {', '.join(preset_names())}",
        )
    cfg = get_config()
    cfg.setdefault("effects", {})["preset"] = body.preset
    save_config(cfg)
    return JSONResponse({
        "current": body.preset,
        "restart_required": False,
        "message": "Saved. Takes effect on the next utterance.",
    })


# ----- Effects Preset Editor (CRUD for custom presets).
#
# Built-in presets are read-only -- listed but never mutated. Custom
# presets live in ``config.effects.custom_presets`` as ``{name: [{name,
# params}, ...]}``. The merged set (builtins + customs) is what
# ``apply_effects`` resolves against; a custom that shadows a built-in
# name wins.

@router.get("/api/effects/plugins")
async def api_effects_plugins():
    """Catalog of pedalboard plugins available for the editor.

    Each entry includes its constructor parameters (name, type, default,
    min/max, step) so the UI can render parameter sliders/inputs
    consistently. The list is curated rather than introspected -- many
    pedalboard parameters are read-only state that shouldn't appear in
    a user editor.
    """
    from .effects import plugin_catalog
    return JSONResponse({"plugins": plugin_catalog()})


def _normalize_preset_effects(effects: list) -> list[dict]:
    """Validate and normalize an effects chain for storage.

    Each entry must be ``{"name": str, "params": dict}`` where ``name``
    is a known plugin. Unknown plugin names or non-dict params raise 400.
    Missing parameters fall back to their plugin defaults at instantiate
    time (pedalboard's responsibility), so the stored chain only carries
    the values the user actually set.
    """
    from .effects import plugin_catalog
    catalog = plugin_catalog()
    normalized: list[dict] = []
    for idx, entry in enumerate(effects):
        if not isinstance(entry, dict):
            raise HTTPException(
                status_code=400,
                detail=f"Effect {idx}: must be an object.",
            )
        cls_name = entry.get("name")
        params = entry.get("params") or {}
        if not isinstance(cls_name, str) or cls_name not in catalog:
            raise HTTPException(
                status_code=400,
                detail=f"Effect {idx}: unknown plugin {cls_name!r}. Known: {', '.join(sorted(catalog))}",
            )
        if not isinstance(params, dict):
            raise HTTPException(
                status_code=400,
                detail=f"Effect {idx}: params must be an object.",
            )
        # Coerce numeric params and reject unknown keys so a typo doesn't
        # silently round-trip through the editor as dead data.
        param_specs = {p["name"]: p for p in catalog[cls_name]}
        cleaned_params: dict = {}
        for k, v in params.items():
            if k not in param_specs:
                raise HTTPException(
                    status_code=400,
                    detail=f"Effect {idx} ({cls_name}): unknown parameter {k!r}.",
                )
            try:
                cleaned_params[k] = float(v)
            except (TypeError, ValueError):
                raise HTTPException(
                    status_code=400,
                    detail=f"Effect {idx} ({cls_name}.{k}): expected number, got {v!r}.",
                )
        normalized.append({"name": cls_name, "params": cleaned_params})
    return normalized


def _custom_preset_name_ok(name: str) -> tuple[bool, str]:
    """Validate a custom preset name. Returns (ok, error_message)."""
    if not isinstance(name, str) or not name.strip():
        return False, "name must be a non-empty string"
    name = name.strip()
    if len(name) > 40:
        return False, "name too long (max 40 chars)"
    if not re.match(r"^[A-Za-z0-9 _\-]+$", name):
        return False, "name may only contain letters, digits, spaces, _ and -"
    # Built-ins are read-only -- protect 'off' especially since it's the
    # passthrough sentinel.
    from .effects import is_builtin_preset
    if is_builtin_preset(name):
        return False, f"{name!r} is a built-in preset and cannot be overwritten"
    return True, ""


class CustomPresetBody(BaseModel):
    name: str
    effects: list[dict]


@router.post("/api/effects/presets")
async def api_create_custom_preset(body: CustomPresetBody):
    """Create a new custom preset. 409 if the name already exists."""
    ok, err = _custom_preset_name_ok(body.name)
    if not ok:
        raise HTTPException(status_code=400, detail=err)
    name = body.name.strip()
    cfg = get_config()
    custom = cfg.setdefault("effects", {}).setdefault("custom_presets", {})
    if name in custom:
        raise HTTPException(status_code=409, detail=f"Preset {name!r} already exists; use PUT to update.")
    custom[name] = _normalize_preset_effects(body.effects)
    save_config(cfg)
    return JSONResponse({"name": name, "effects": custom[name], "message": "Created."})


class CustomPresetUpdate(BaseModel):
    """PUT body. ``rename`` lets the editor change a preset's name as
    part of the same save (avoids a delete+create dance the UI would
    otherwise have to orchestrate)."""
    effects: list[dict]
    rename: str | None = None


@router.put("/api/effects/presets/{name}")
async def api_update_custom_preset(name: str, body: CustomPresetUpdate):
    """Update an existing custom preset (effects chain and optionally name).

    404 if the preset doesn't exist. 400 if validation fails. Rename
    target must pass the same name rules; existing target name returns 409.
    """
    cfg = get_config()
    custom = cfg.setdefault("effects", {}).setdefault("custom_presets", {})
    if name not in custom:
        raise HTTPException(status_code=404, detail=f"Custom preset {name!r} not found.")
    new_effects = _normalize_preset_effects(body.effects)
    target_name = name
    if body.rename and body.rename != name:
        ok, err = _custom_preset_name_ok(body.rename)
        if not ok:
            raise HTTPException(status_code=400, detail=err)
        target_name = body.rename.strip()
        if target_name in custom:
            raise HTTPException(status_code=409, detail=f"Preset {target_name!r} already exists.")
        del custom[name]
        # If the renamed preset was the active default, follow the rename
        # so the user doesn't silently fall off it.
        if cfg.get("effects", {}).get("preset") == name:
            cfg["effects"]["preset"] = target_name
    custom[target_name] = new_effects
    save_config(cfg)
    return JSONResponse({
        "name": target_name,
        "effects": new_effects,
        "renamed_from": name if target_name != name else None,
        "message": "Saved.",
    })


@router.delete("/api/effects/presets/{name}")
async def api_delete_custom_preset(name: str):
    """Delete a custom preset. 404 if missing, 400 for built-ins.

    If the deleted preset was the active default, fall back to ``off``
    so the next utterance doesn't try to resolve a now-missing name.
    """
    from .effects import is_builtin_preset
    if is_builtin_preset(name):
        raise HTTPException(status_code=400, detail=f"{name!r} is a built-in preset.")
    cfg = get_config()
    custom = cfg.setdefault("effects", {}).setdefault("custom_presets", {})
    if name not in custom:
        raise HTTPException(status_code=404, detail=f"Custom preset {name!r} not found.")
    del custom[name]
    fell_back = False
    if cfg.get("effects", {}).get("preset") == name:
        cfg["effects"]["preset"] = "off"
        fell_back = True
    save_config(cfg)
    return JSONResponse({
        "deleted": name,
        "default_reset_to_off": fell_back,
        "message": "Deleted." + (" Default fell back to 'off'." if fell_back else ""),
    })


@router.get("/api/effects/presets/{name}")
async def api_get_custom_preset(name: str):
    """Return the effects chain for one preset (built-in or custom)."""
    from .effects import all_presets
    presets = all_presets()
    if name not in presets:
        raise HTTPException(status_code=404, detail=f"Preset {name!r} not found.")
    chain = [
        {"name": spec.name, "params": dict(spec.params)}
        for spec in presets[name]
    ]
    from .effects import is_builtin_preset
    return JSONResponse({
        "name": name,
        "builtin": is_builtin_preset(name),
        "effects": chain,
    })


class EffectsTry(BaseModel):
    """Optional preset override for a one-shot preview that doesn't touch
    the saved configuration. When ``preset`` is omitted, the saved
    preset is used."""
    preset: str | None = None


_EFFECTS_TRY_PHRASE = "The quick brown fox jumps over the lazy dog."


@router.post("/api/effects/try")
async def api_effects_try(body: EffectsTry):
    """Enqueue a fixed phrase for immediate playback through the chain.

    When *preset* is supplied, the preset is attached to the queue item
    as ``metadata.effects_preset``. The daemon's ``process_queue``
    extracts it and passes through to ``apply_effects`` as a one-shot
    override -- no mutation of ``config.json``, no race with the daemon's
    polling interval. When *preset* is omitted, the saved preset is used.

    Before enqueueing, the requested chain is exercised against a tiny
    zero buffer so a broken preset (e.g. unknown plugin name or bad
    parameter introduced by a hand-edit to PRESETS) returns 500 with the
    detail synchronously rather than producing silently-passthrough
    audio inside the daemon (which is the right behavior for production
    but unhelpful for previewing).
    """
    from .queue_db import enqueue
    from .cli import start_player
    from .effects import apply_effects
    import numpy as np

    requested = body.preset
    if requested is not None and requested not in PRESETS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown preset {requested!r}. Known: {', '.join(preset_names())}",
        )

    # Eager probe: run the chain on a 100ms-of-silence buffer. apply_effects
    # has a defensive catch in the daemon hot path that swallows exceptions
    # (right call there -- a bad chain should never break TTS), so we test
    # the construction here where we CAN surface failures. The build_board
    # cache means this probe is virtually free after the first call per
    # preset.
    try:
        from .effects import _build_board, PRESETS as _PRESETS
        probe_preset = requested or get_effects_config().get("preset", "off") or "off"
        if probe_preset in _PRESETS and _PRESETS[probe_preset]:
            board = _build_board(_PRESETS[probe_preset])
            if board is not None:
                silence = np.zeros(1600, dtype=np.float32)  # 100ms @ 16kHz
                board(silence, 16000)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Effect chain {probe_preset!r} failed at construction: {e}",
        )

    metadata: dict[str, str] = {"queue": "default"}
    if requested is not None:
        metadata["effects_preset"] = requested
    queue_id = enqueue(_EFFECTS_TRY_PHRASE, metadata=metadata)
    start_player()

    saved = get_effects_config().get("preset", "off")
    return JSONResponse({
        "queue_id": queue_id,
        "spoken": _EFFECTS_TRY_PHRASE,
        "preset": requested or saved,
    })


class PronunciationTry(BaseModel):
    """Speak a single word with an optional in-progress replacement.

    When *replacement* is supplied, the server enqueues that string
    verbatim so the user hears the unsaved edit before committing it.
    Without it, the server enqueues the original *word* and lets the
    standard preprocessor apply any saved override.
    """
    word: str
    replacement: str | None = None


@router.post("/api/pronunciation/try")
async def api_pronunciation_try(body: PronunciationTry):
    """Enqueue a sample for immediate playback to preview an override.

    If *replacement* is provided, that string is spoken directly --
    mimics exactly what the override would produce for *word*, without
    touching config.json. Lets the UI's TRY button reflect the live
    (unsaved) edit instead of the previously-persisted value.

    Without *replacement*, the original *word* is enqueued so the saved
    override (if any) applies through normal preprocessing.
    """
    from .queue_db import enqueue
    from .cli import start_player

    word = (body.word or "").strip()
    if not word:
        raise HTTPException(status_code=400, detail="word cannot be empty")
    spoken = (body.replacement or "").strip() or word
    queue_id = enqueue(spoken, metadata={"queue": "default"})
    start_player()
    return JSONResponse({"queue_id": queue_id, "spoken": spoken})


def _signal_restart_needed() -> None:
    """Write the restart-needed sentinel. Called by endpoints that change
    something cached at the daemon's startup."""
    try:
        from .paths import restart_sentinel_path
        p = restart_sentinel_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
    except OSError:
        pass


@router.get("/api/restart-needed")
async def api_restart_needed():
    """Whether a daemon restart is pending. Driven by a sentinel file
    that endpoints touching startup-cached config write, and that the
    daemon clears on boot. UI polls this and shows a pill in the top tab
    bar when true."""
    from .paths import restart_sentinel_path
    return JSONResponse({"required": restart_sentinel_path().exists()})


@router.post("/api/restart-player")
async def api_restart_player():
    """Kick the player daemon so it picks up new config (overrides, voice, ...).

    Uses ``launchctl kickstart -k`` against the user's launchd domain. Returns
    early if launchctl is unavailable (e.g. the daemon isn't launchd-managed
    here) -- the UI shows the response and the user can restart manually.
    """
    uid = os.getuid()
    target = f"gui/{uid}/com.speeker.player"
    try:
        result = subprocess.run(
            ["launchctl", "kickstart", "-k", target],
            capture_output=True, text=True, timeout=10,
        )
        return JSONResponse({
            "status": "ok" if result.returncode == 0 else "warn",
            "stdout": result.stdout,
            "stderr": result.stderr,
            "target": target,
        })
    except FileNotFoundError:
        raise HTTPException(status_code=501, detail="launchctl not available")
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="launchctl timed out")


@router.get("/settings")
async def settings_page(session: str | None = None):
    """Settings page."""
    settings = get_settings(session)
    target = session or "Global"

    # Build voice options for each engine
    pocket_options = []
    for voice, desc in POCKET_TTS_VOICES.items():
        selected = 'selected' if settings['voice'] == voice else ''
        pocket_options.append(f'<option value="{voice}" {selected}>{voice} - {desc}</option>')

    kokoro_options = []
    for voice, desc in KOKORO_VOICES.items():
        selected = 'selected' if settings['voice'] == voice else ''
        kokoro_options.append(f'<option value="{voice}" {selected}>{voice} - {desc}</option>')

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Speeker Settings</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, sans-serif;
                max-width: 600px;
                margin: 50px auto;
                padding: 20px;
                background: #0a0a0f;
                color: rgba(255, 255, 255, 0.75);
            }}
            h1 {{ color: #00d9ff; }}
            form {{ background: #151520; padding: 20px; border-radius: 8px; }}
            .field {{ margin-bottom: 20px; }}
            .field label {{ display: block; margin-bottom: 8px; font-weight: bold; }}
            input, select {{
                padding: 8px;
                background: #0a0a0f;
                border: 1px solid #222;
                color: rgba(255, 255, 255, 0.75);
                border-radius: 4px;
                width: 100%;
            }}
            select {{ width: 100%; }}
            input[type="checkbox"] {{ width: auto; }}
            button {{
                background: #00d9ff;
                color: #000;
                border: none;
                padding: 10px 20px;
                border-radius: 4px;
                cursor: pointer;
            }}
            a {{ color: #00d9ff; }}
            optgroup {{ font-weight: bold; }}
        </style>
    </head>
    <body>
        <h1>Settings: {target}</h1>
        <form method="POST">
            <div class="field">
                <label>Intro/Outro Sound:</label>
                <input type="checkbox" name="intro_sound" {'checked' if settings['intro_sound'] else ''}>
            </div>
            <div class="field">
                <label>Speed:</label>
                <input type="number" name="speed" value="{settings['speed']}" min="0.5" max="2.0" step="0.1">
            </div>
            <div class="field">
                <label>Engine:</label>
                <select name="engine">
                    <option value="pocket-tts" {'selected' if settings.get('engine') == 'pocket-tts' else ''}>pocket-tts (faster)</option>
                    <option value="kokoro" {'selected' if settings.get('engine') == 'kokoro' else ''}>kokoro (higher quality)</option>
                </select>
            </div>
            <div class="field">
                <label>Voice:</label>
                <select name="voice">
                    <optgroup label="pocket-tts">
                        {''.join(pocket_options)}
                    </optgroup>
                    <optgroup label="kokoro">
                        {''.join(kokoro_options)}
                    </optgroup>
                </select>
            </div>
            <button type="submit">Save</button>
        </form>
        <p><a href="/">Back to history</a> | <a href="/settings">Global Settings</a></p>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


@router.post("/settings")
async def save_settings(request: Request, session: str | None = None):
    """Save settings."""
    form = await request.form()
    set_settings(
        session_id=session,
        intro_sound="intro_sound" in form,
        speed=float(form.get("speed", 1.0)),
        voice=form.get("voice"),
        engine=form.get("engine"),
    )
    return HTMLResponse(
        content='<script>alert("Settings saved!"); window.location="/";</script>'
    )
