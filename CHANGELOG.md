# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Fixed
- **The system-audio tap is now always torn down cleanly.** The Swift helper
  finishes its Core Audio teardown (destroying the process tap and the private
  aggregate device) only once its IOProc returns — and that IOProc is blocked
  writing to stdout. If the reader thread stopped draining that pipe, `SIGTERM`
  could never complete and the app `SIGKILL`ed the helper after 2 s, skipping the
  teardown entirely. Two causes, both fixed:
  - The reader's streaming loop was unguarded, and `np.frombuffer(data, "<f4")`
    raises on any pipe read whose length is not a multiple of 4 — silently killing
    the drain thread. The loop now meters only whole float32 samples and can no
    longer die on a bad chunk.
  - `stop()` now keeps stdout drained while the helper exits (draining itself if
    the reader thread is gone) instead of blindly killing it after 2 s, and always
    reaps the process. Measured: clean exit with status 0 in ~0.1 s, where it
    previously took 2 s and ended in `SIGKILL`.

- **The tap helper now cleans up on every signal that can reach it.** It only
  handled `SIGTERM` and `SIGINT`, so closing the terminal window (`SIGHUP`) or the
  reader going away (`SIGPIPE`) killed it outright and the process tap and
  aggregate device were left behind. Both now run the teardown and exit 0, and
  `cleanup()` is idempotent so the signal, write-error and start-up failure paths
  can no longer destroy an object twice.

- **Pressing `q` now shows that transcription has started.** Stopping a recording
  used to freeze the UI on the last recording frame — mixing the capture into
  `audio.wav` and transcribing it both ran on the main loop — so nothing on screen
  changed until the finished transcript appeared. That work now runs on a worker
  thread and the screen switches immediately to a live progress panel showing the
  step (Save audio → Label speakers → Transcribe), a spinner or a real percentage,
  and the elapsed time. The same panel is used by `t` (transcribe an existing
  recording), which had the same problem.

### Removed
- Deleted the unused legacy `mp4-transcriptor.py` (Turkish-only, superseded by the
  in-app **Import file** feature).

### Changed
- **All UI text, prompts, logs, comments and docstrings are in English** — including
  the system-audio source's labels (`System audio (Core Audio tap)`), the last
  Turkish strings left in the interface.
- **Renamed the app to Audiocript** (logo, header banner, all UI text and the
  main script `audiocript.py`).
- Default recordings folder is now `~/Audiocript/recordings` (was inside the repo
  folder), so the app name — not the repo path — is shown in the UI. Existing
  saved `base_path` values are unaffected; change it anytime in Settings.

### Branding / docs
- Added a parrot **logo** and a **header banner**, and a professional README with
  badges and screenshots.
- **Attribution & license:** Audiocript is a derivative of
  [voice_transcriptor](https://github.com/semihshn/voice_transcriptor) by
  Semih Şahan, distributed under the original **MIT License** (original copyright
  retained in `LICENSE`, credited in the README).

### Added
- **Arrow-key navigation** with a grouped, collapsible menu (New recording,
  Import file, Recordings ▸ list, Settings ▸ language/mic/system-audio/open-with/
  folder, Quit) — replaces the long single-key shortcut row.
- **Named recordings**: name a recording/import when creating it, and rename any
  recording at any time (`r`). Names are stored per project in `meta.json`.
- **Recordings list**: browse all recordings (name, date, language) in the menu.
- **In-app transcript viewer**: open a recording to read its transcript with
  scrolling (↑/↓, PgUp/PgDn, Home/End); `Enter` opens it in your external app.
- **Per-recording actions** from the menu and viewer: **play/stop** the audio
  (`p`, via afplay), **delete** a recording with confirmation (`d`), in addition
  to rename (`r`) and view.
- Full-screen terminal UI (Rich).
- Record the **microphone and system audio together**, mixed into one 16 kHz mono
  track for transcription.
- System-audio capture via a native macOS **Core Audio process tap** (Swift
  helper compiled on first use) — playback stays audible, no BlackHole or
  output rerouting required.
- Live VU level meters for each source while recording (shows `no signal` for a
  silent source).
- "Preparing to record…" screen with step-by-step status while initializing
  (opening the mic, starting the system-audio tap, first-run helper compile /
  permission), then a clear "Recording started — speak now." once capture begins.
- Background **model pre-warming**: the current language's model loads in a
  background thread at startup (and when you switch language), so the first
  transcript is fast. The header shows `Model: loading… / ready`.
- **Open the transcript in an app** after each run, chosen from a filterable list
  of installed apps (`o`). Defaults to Sublime Text if installed; can be disabled.
  Stored as `open_app` in `config.json`.
- **Import an existing media file** (`f`): pick an `mp4`/`mov`/`wav`/`mp3`/`m4a`
  via the native file picker; its audio is extracted with ffmpeg into a new
  project folder (`audio.wav`, 16 kHz mono) and transcribed like a recording.
  Runs in the background with a **live progress panel** (real % for extraction
  and for English transcription; spinner + elapsed otherwise).
- `run.sh` now checks for the optional external tools (`ffmpeg`, `swiftc`) and
  offers to install any that are missing (skip with `SKIP_DEP_CHECK=1`).
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
