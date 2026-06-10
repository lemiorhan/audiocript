# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Full-screen terminal UI (Rich) with single-key controls: record, language,
  microphone picker, system-audio toggle, folder edit, quit.
- Record the **microphone and system audio together**, mixed into one 16 kHz mono
  track for transcription.
- System-audio capture via a native macOS **Core Audio process tap** (Swift
  helper compiled on first use) — playback stays audible, no BlackHole or
  output rerouting required.
- Live VU level meters for each source while recording (shows `no signal` for a
  silent source).
- Per-language transcription models: `selimc/whisper-large-v3-turbo-turkish`
  (Turkish, Transformers) and `ggml-distil-large-v3` (English, whisper.cpp).
- Automatic device selection for models (CUDA → Apple Silicon MPS/Metal → CPU).
- Per-recording project folders (`audio.wav` + `transcription.txt`).
- Persisted preferences in `config.json` (language, mic by name, system-audio
  toggle, recordings folder).
- One-command launcher `run.sh` (venv + dependency install + run).
- Open-source repository files: README, CONTRIBUTING, CODE_OF_CONDUCT, SECURITY,
  issue/PR templates.

### Changed
- Capture each source at its **native sample rate** and resample to 16 kHz in
  software (anti-aliased), so connecting to a device never changes its global
  sample rate or interrupts playback.
- UI text and logs are in English; noisy library logs are suppressed during
  transcription.

### Fixed
- Graceful `Ctrl-C` / `kill` handling (no traceback; recorders and the tap
  subprocess are always torn down).
- Empty/zero-length recordings no longer crash.
