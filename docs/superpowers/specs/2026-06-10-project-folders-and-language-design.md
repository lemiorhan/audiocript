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
  "speaker_device": "BlackHole 2ch"
}
```

- Created on first run, updated whenever the user changes language, base path,
  mic device, or the speaker (system-audio) source.
- `language` is one of `"tr"` or `"en"`.
- `input_device` / `speaker_device` are device **names** (not indices), since
  indices change between sessions as devices connect/disconnect.
- `speaker_device` is optional/absent. When absent, recording is mic-only.

### Startup flow

Every launch prompts for each setting **with the saved value pre-filled as the
default** — pressing Enter keeps the last choice, typing/selecting changes it.

1. Load `config.json` if present.
2. **Confirm base project folder.** Default = saved path, or `./recordings` on
   first run. Created if missing. Saved to config.
3. **Transcription language.** Default = saved language (or `tr`). Saved to config.
4. **Mic device.** List all input-capable devices (physical mics plus
   virtual/app-audio devices like BlackHole/Zoom/Teams); default = saved device's
   position (or the first device if the saved one is gone). Saved by name.
5. **Speaker (system audio) source — optional.** List input-capable devices plus
   an "off (mic only)" choice; default = saved choice, else BlackHole if present,
   else off. When set, the app records mic + speaker audio together (see below).
   Saved by name (key `speaker_device`); choosing "off" clears it.

### Input device

- The macOS system default input may be a virtual device (e.g. BlackHole), which
  records silence unless audio is routed into it — the selector lets the user
  avoid that. The app exposes virtual/loopback devices so Zoom/computer audio can
  be captured once routed at the OS level, but it does not create that routing.
- If a device cannot be opened (e.g. unsupported sample rate), `record_audio`
  reports the error and returns `False` so the user can pick another via the menu.
- Menu gains `d` (mic) and `s` (speaker / system source) options to change
  devices anytime.

### Mic + speaker audio (record both)

macOS does not allow capturing an output/speaker device directly, so speaker
audio is captured from a loopback **input** device (e.g. BlackHole) that the
user's system/Zoom output is routed into. The picker is labeled "Speaker (system
audio)" but selects an input device.

- `record_audio(filepath, devices, fs)` takes a **list** of device indices
  (`[mic]` or `[mic, speaker]`) and opens one `sd.InputStream` per device
  concurrently (via `contextlib.ExitStack`), each at `channels=1, dtype=int16,
  fs=16000`, accumulating into its own buffer.
- On stop, buffers are concatenated per device and combined by `mix_to_mono`:
  trim all signals to the shortest length (independent device clocks drift),
  sum as int32, then scale down if the peak exceeds the int16 range (soft limit
  instead of hard clipping). Result is written as one mono WAV.
- Purpose is transcription, not production audio: minor inter-stream drift is
  acceptable since both voices remain intelligible in the mix.
- The speaker source is optional; absent → mic-only (`[mic]`), preserving the
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
- `s` — set/clear the speaker (system-audio) source (e.g. BlackHole); updates config
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
