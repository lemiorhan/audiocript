# Per-recording project folders + language selection

**Date:** 2026-06-10
**File affected:** `meeting-transcriptor.py` (plus new `config.json` at runtime)

## Goal

Stop appending every transcription to a single shared file. Instead, give each
recording its own self-contained project folder, and let the user pick the
transcription language (Turkish or English), remembered across runs.

## Behavior

### Config

A `config.json` file lives next to the script and stores:

```json
{ "language": "tr", "base_path": "/abs/path/to/recordings" }
```

- Created on first run, updated whenever the user changes language or base path.
- `language` is one of `"tr"` or `"en"`.

### Startup flow

1. Load `config.json` if present.
2. **Confirm base project folder.** Prompt shows the saved path, or the default
   `./recordings` on first run. Enter accepts; typing a new path overrides it.
   The folder is created if it does not exist. Saved to config.
3. **Transcription language.** On first run (no saved language), ask the user to
   pick `tr` or `en`. Otherwise use the saved language. Saved to config.

### Per recording

- Create a timestamped subfolder `<base>/YYYY-MM-DD_HH-MM-SS/`.
- Record audio and save as `audio.wav` inside that subfolder.
- Transcribe with Whisper using the selected language:
  `generate_kwargs={"language": "turkish"|"english", "task": "transcribe"}`.
- Save the transcription to `transcription.txt` inside the same subfolder.

### Menu after each recording (replaces the y/n prompt)

- `[Enter]` — start a new recording
- `l` — change language (toggle/select tr or en); updates config immediately
- `q` — quit

## Code structure

New/changed helpers in `meeting-transcriptor.py`:

- `load_config()` / `save_config(cfg)` — read/write `config.json`.
- `confirm_base_path(cfg)` — prompt and persist the base folder.
- `select_language(cfg)` — prompt for tr/en and persist; returns code.
- `lang_name(code)` — map `tr`→`turkish`, `en`→`english` for Whisper.
- `record_audio(filepath, fs)` — now takes a full output path.
- `transcribe_audio(filepath, language_name)` — now takes the language.
- `main()` — rewired for startup flow + per-recording subfolder + menu.

The old top-level `recorded_audio.wav` / `transcription.txt` are no longer
written. Existing untracked copies are left as-is.

## Testing

The interactive audio/Whisper paths are not unit-testable here. The pure helpers
are: config load/save round-trip, default handling when config is missing, and
`lang_name` mapping. These get covered with a small test using a temp dir.

## Out of scope

- No translation to a non-English target (Whisper translates reliably only to
  English).
- No changes to `mp4-transcriptor.py`.
