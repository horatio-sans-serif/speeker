# Amazon Polly + SSML Support — Design

**Date:** 2026-05-23
**Status:** Approved (design), pending implementation plan

## Goal

Add Amazon Polly as a third TTS engine and add SSML (Speech Synthesis Markup
Language) support to Speeker. Polly supports SSML natively; the local engines
(pocket-tts, kokoro) get a best-effort SSML emulation gated behind a flag.

## Background — current architecture

Two generation paths exist today and they do not share code:

- **Main flow (server + MCP):** `POST /speak` → `enqueue(text, metadata)` into
  the SQLite queue → the **player daemon** polls and generates TTS at playback
  time. `player.py:generate_tts` is **hardcoded to pocket-tts** and ignores the
  `engine` value stored in the `settings` table (and the `engine` passed by the
  MCP tool). So today neither kokoro nor anything else actually plays through the
  daemon path that the server and MCP use.
- **CLI direct flow:** `speeker speak` has its own `pocket-tts`/`kokoro`
  if/else dispatch in `cli.py:speak_text`, pre-generates audio, and writes file
  paths to a file-based queue.

Consequences this design addresses:

- pocket-tts model handling is duplicated in `cli.py` and `player.py`.
- The daemon must be changed to honor the stored `engine` setting, or Polly can
  never play through the server/MCP path (the path everything actually uses).
- `preprocess_for_tts` rewrites text aggressively (`.` → " dot ", symbol
  expansion). SSML markup fed through it would be destroyed, so SSML must bypass
  or be handled before preprocessing.

## Decisions

### SSML behavior across engines

- **Polly:** native SSML via `TextType='ssml'`.
- **Local engines (pocket-tts, kokoro):** when best-effort emulation is enabled,
  transform SSML into plain text approximating the intent. When disabled, naive
  tag-stripping to text content.
- Emulation is gated behind `--best-effort-ssml-emulation` (CLI) /
  `ssml.emulate_for_local` (config). CLI flag overrides config for that run.

Emulation transforms (rule-based, deterministic, offline, no LLM in the hot path):

- `<say-as interpret-as="characters">PHI</say-as>` /
  `interpret-as="spell-out"` → `P-H-I`.
- `<sub alias="...">text</sub>` → the alias text.
- `<break .../>` and prosody pauses → punctuation chosen by duration
  (short → comma, medium → period, long → ellipsis).
- All-caps runs and `<emphasis>` content → case-normalized so local engines do
  not shout or mis-spell. A word in the **spell-out set** is rendered `P-H-I`
  instead of normalized.
- Any other tag → replaced by its text content.

### Acronym (spell-out) set

- Built-in `COMMON_ACRONYMS` set.
- Plus a user file pointed to by `ssml.acronyms_file` config. File tokens are
  split on `[,\s|;]+` (whitespace, commas, pipes, semicolons), merged into the
  built-in set.

### SSML signaling

- A caller signals SSML by **explicit flag** (`--ssml` CLI, `ssml=true` in
  `/speak` body or `?ssml=true` query, `ssml=True` MCP tool) **OR** by
  **auto-detection** of a leading `<speak>` wrapper.
- Propagated through the queue via the existing `metadata` JSON column as
  `{"ssml": true}`. No schema migration. Auto-detect covers `<speak>`-prefixed
  text even without the flag.

### Polly specifics

- One Speeker engine: `-e polly`. Its variant is a sub-option:
  `--polly-engine={standard,neural,long-form,generative}` and
  `--polly-voice=VOICE`.
- Per-variant default voices (config-overridable; `describe_voices` is the real
  source of truth, catalog varies by region): standard→Joanna, neural→Joanna,
  long-form→Danielle, generative→Ruth.
- Credentials via boto3 default chain (`~/.aws/credentials` profiles,
  `AWS_PROFILE`, env). `polly.region` / `polly.profile` only override when set.
- Audio: request `OutputFormat='pcm'` (16-bit / 16 kHz mono). Convert int16 →
  float32 in `[-1, 1]` so it flows through the existing normalize → int16 → WAV
  pipeline with no ffmpeg decode step.

### Engine abstraction (Approach A)

`engines.py` defines an `Engine` interface and a registry. Both `cli.py` and
`player.py` call through it, removing the duplicated model handling and the
daemon's hardcoding.

Interface:

- `name: str`
- `supports_ssml: bool`
- `default_voice: str`
- `list_voices() -> dict[str, str]`
- `validate_voice(voice: str) -> bool`
- `generate(text: str, voice: str, *, is_ssml: bool) -> tuple[np.ndarray, int]`
  returns `(float32 audio in [-1, 1], sample_rate)`.
- `warm() -> None` and `unload() -> None` so the daemon's idle-timeout memory
  logic keeps working. For Polly both are no-ops (holds only a cheap boto3
  client).

Implementations: `PocketTTSEngine`, `KokoroEngine`, `PollyEngine`. Lazy imports
per engine so `boto3` / `kokoro` / `pocket_tts` only load when that engine is
used. `get_engine(name)` returns a cached singleton (each holds its own warm
model state where applicable).

## SSML flow (per utterance)

1. Extract leading `$Note` tone tokens first (preserves current behavior; SSML
   survives because tone tokens strip away before the `<speak>` wrapper).
2. `is_ssml` = explicit flag OR `looks_like_ssml(remaining_text)`.
3. If engine `supports_ssml` (Polly): `ensure_speak_wrapped`, send with
   `TextType='ssml'`, **bypass** `preprocess_for_tts`.
4. Else (local): if emulation enabled → `emulate_ssml(text, acronyms)`; else →
   `strip_ssml(text)`. Result is the final spoken text — `preprocess_for_tts` is
   **skipped** for the SSML path to avoid re-mangling spelled-out letters.
   Non-SSML text is unchanged: normal `preprocess_for_tts`.

## Config additions

```jsonc
"polly": {
  "region": null,      // null = boto3 default
  "profile": null,     // null = default credential chain
  "engine": "neural",  // default Polly variant
  "voice": "Joanna"    // default Polly VoiceId
},
"ssml": {
  "emulate_for_local": false,  // CLI --best-effort-ssml-emulation enables
  "acronyms_file": null        // path to extra acronyms file
}
```

Accessors `get_polly_config()` and `get_ssml_config()` follow the existing
section-merge pattern in `config.py`.

## Surface-area changes

- **`engines.py`** (new): `Engine` interface, three implementations,
  `get_engine()` registry.
- **`ssml.py`** (new): `looks_like_ssml`, `ensure_speak_wrapped`, `strip_ssml`,
  `emulate_ssml`, `load_acronyms`, `COMMON_ACRONYMS`.
- **`voices.py`**: add `POLLY_VOICES` (curated), `DEFAULT_POLLY_VOICE`,
  `DEFAULT_POLLY_ENGINE`; lenient `validate_voice("polly", ...)` (Polly is the
  authority); register `"polly"` in `get_voices`.
- **`cli.py`**: `-e polly`, `--polly-engine`, `--polly-voice`, `--ssml`,
  `--best-effort-ssml-emulation`; route generation through `get_engine()`.
- **`player.py`**: replace hardcoded `generate_tts` with
  `get_engine(settings["engine"]).generate(...)`; warm/unload the active engine;
  read `is_ssml` from queue metadata (extend `get_pending_for_session` to return
  `metadata`).
- **`server.py`**: `/speak` accepts `ssml` (body field + `?ssml=true`); add
  `polly` to `/voices`.
- **`mcp/server.py`**: `speak(..., ssml=False)`, allow `engine="polly"` with
  optional `polly_engine` / `polly_voice` passed via metadata.
- **`pyproject.toml`**: `[project.optional-dependencies] polly = ["boto3"]`.

## Explicitly out of scope (kept minor on purpose)

- Polly voices appear in `speeker voices` and are selectable, but are **not**
  added to the `voice-prefs` sample-ranking UI in this change — generating
  ranking samples there would fire billable Polly calls for every voice.
- No LLM-assisted SSML emulation. Case normalization is rule-based using the
  acronym spell-out set.
- `summarize` produces plain text; it does not emit or accept SSML.

## Testing

- `ssml.py`: detect / strip / emulate; acronym-file parsing across all
  separators (`[,\s|;]+`); break-duration → punctuation mapping; say-as and sub
  handling; all-caps normalization with and without spell-out membership.
- `engines.py`: registry returns cached singletons; `PollyEngine` with `boto3`
  mocked (no live AWS) — verify `TextType`, `Engine`, `VoiceId`, `OutputFormat`
  args and PCM → float32 conversion; `warm`/`unload` no-ops for Polly.
- `config.py`: `get_polly_config` / `get_ssml_config` defaults and merge.
- `player.py`: daemon dispatches by the stored `engine` setting (proves the
  hardcoding is fixed); `is_ssml` read from queue metadata.
- All Polly tests mock boto3; no test requires AWS credentials or network.
