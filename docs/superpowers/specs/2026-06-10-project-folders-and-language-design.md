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
{
  "language": "tr",
  "base_path": "/abs/path/to/recordings",
  "input_device": "MX Brio",
  "system_device": "BlackHole 2ch"
}
```

- Created on first run, updated whenever the user changes language, base path,
  input (mic) device, or the system-audio source.
- `language` is one of `"tr"` or `"en"`.
- `input_device` / `system_device` are device **names** (not indices), since
  indices change between sessions as devices connect/disconnect.
- `system_device` is optional/absent. When absent, recording is mic-only.

### Startup flow

1. Load `config.json` if present.
2. **Confirm base project folder.** Prompt shows the saved path, or the default
   `./recordings` on first run. Enter accepts; typing a new path overrides it.
   The folder is created if it does not exist. Saved to config.
3. **Transcription language.** On first run (no saved language), ask the user to
   pick `tr` or `en`. Otherwise use the saved language. Saved to config.
4. **Input device.** Resolve the saved device name to a current index. If none is
   saved or it is no longer present, list all input-capable devices (physical mics
   plus virtual/app-audio devices like BlackHole/Zoom/Teams) and let the user
   pick one. Saved to config by name.
5. **System audio source (optional).** Resolve the saved `system_device` name. If
   set and present, the app records mic + system audio together (see below). If
   unset or missing, recording is mic-only.

### Input device

- The macOS system default input may be a virtual device (e.g. BlackHole), which
  records silence unless audio is routed into it — the selector lets the user
  avoid that. The app exposes virtual/loopback devices so Zoom/computer audio can
  be captured once routed at the OS level, but it does not create that routing.
- If a device cannot be opened (e.g. unsupported sample rate), `record_audio`
  reports the error and returns `False` so the user can pick another via the menu.
- Menu gains `d` (mic) and `s` (system source) options to change devices anytime.

### Mic + system audio (record both)

- `record_audio(filepath, devices, fs)` takes a **list** of device indices
  (`[mic]` or `[mic, system]`) and opens one `sd.InputStream` per device
  concurrently (via `contextlib.ExitStack`), each at `channels=1, dtype=int16,
  fs=16000`, accumulating into its own buffer.
- On stop, buffers are concatenated per device and combined by `mix_to_mono`:
  trim all signals to the shortest length (independent device clocks drift),
  sum as int32, then scale down if the peak exceeds the int16 range (soft limit
  instead of hard clipping). Result is written as one mono WAV.
- Purpose is transcription, not production audio: minor inter-stream drift is
  acceptable since both voices remain intelligible in the mix.
- The system source is optional; absent → mic-only (`[mic]`), preserving the
  original single-source behavior. `BlackHole` is suggested as the default when
  present. Capturing Zoom audio still requires the user to route Zoom/system
  output into that device at the OS level (e.g. a Multi-Output Device).

### Per recording

- Create a timestamped subfolder `<base>/YYYY-MM-DD_HH-MM-SS/`.
- Record audio and save as `audio.wav` inside that subfolder.
- Transcribe with the model chosen by the selected language (see "Models").
- Save the transcription to `transcription.txt` inside the same subfolder.

## Models (per language)

`distil-large-v3` is English-only, so each language uses a different model and
runtime:

- **Turkish (`tr`):** `selimc/whisper-large-v3-turbo-turkish` via the Hugging
  Face `transformers` pipeline (`generate_kwargs={"language": "turkish",
  "task": "transcribe"}`). Downloaded/cached by `transformers` on first use.
- **English (`en`):** `ggml-distil-large-v3` via whisper.cpp (`pywhispercpp`).
  The `.bin` (~1.5 GB) is fetched from `distil-whisper/distil-large-v3-ggml`
  with `huggingface_hub.hf_hub_download` and cached on first use.

Device selection picks CUDA → Apple Silicon (MPS/Metal) → CPU automatically.
(The previous code hardcoded `device=0`/CUDA, which fails on non-CUDA machines.)
Each model is lazy-loaded once and cached for the session.

### Menu after each recording (replaces the y/n prompt)

- `[Enter]` — start a new recording
- `l` — change language (toggle/select tr or en); updates config immediately
- `d` — change mic (input) device; updates config immediately
- `s` — set/clear the system-audio source (e.g. BlackHole); updates config
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
- Distil-Whisper is English-only; Turkish quality relies on the separate
  `selimc` model, not on `distil-large-v3`.
