# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- **A background dictation daemon.** `./run.sh --dictate` runs a global hotkey
  (`<cmd>+<alt>+d` by default) that starts and stops a recording from anywhere on
  macOS; on stop, the audio is transcribed locally, corrected by an OpenAI model,
  and the result replaces the clipboard contents. Nothing is saved to disk — the
  audio and its transcript exist only for the duration of one dictation. The
  language follows the existing Language setting. Two new `config.json` keys:
  `dictation_hotkeys` (default `{"toggle": "<cmd>+<alt>+d"}`) and
  `dictation_max_seconds` (default `300`, after which a forgotten recording stops
  itself). A new environment variable, `DICTATION_MODEL` (default
  `gpt-4.1-mini`), picks the correction model; `OPENAI_API_KEY` (shared with
  publishing) is required and the daemon refuses to start without it. Requires
  the **Input Monitoring** permission for the global hotkey; without it,
  `dictate.py --toggle`/`--stop` still work from another shell, a macOS
  Shortcut, or `skhd`. This is not offline: every dictation sends its transcript
  to OpenAI for correction. Uses no new dependencies.

- **Combined microphone and system-audio captures now receive automatic acoustic
  echo cancellation at finalization.** When the optional `pywebrtc-audio` processor
  is available, speaker leakage is removed from the aligned microphone before the
  sources are mixed; saved `mic.wav` is the cleaned channel when speaker labels are
  enabled. The optional native dependency is installed best-effort, and an
  unavailable or failed processor safely falls back to the aligned original
  microphone without losing the recording. Headphones remain the cleanest option.

- **Long transcripts can be published with one keypress.** With
  `GITHUB_REPO_FOR_TRANSCRIPTS`, `GITHUB_TOKEN` and `OPENAI_API_KEY` set in `.env`,
  **`u`** on a recording sends a transcript longer than `TRANSCRIPT_MIN_CHARS` (2000
  by default) through `prompts/transcript_prompt.md`, turns that into a document with
  `prompts/documentation_prompt.md`, and commits both to that repository — raw, edited
  and documented, in a single commit named after the recording. Nothing publishes
  automatically: saving a transcript never starts two paid model calls on its own. It
  runs behind the menu on the status line, so the app stays usable. Both model outputs
  are written next to the recording before anything is pushed, so a failed push costs
  only the retry, and a response the model had to cut short fails the step rather than
  filing half a document. Without those three keys the feature is simply off. Uses no
  new dependencies.

### Fixed
- **A stray keystroke can no longer end a recording.** Space, `r` and `q` were all
  wired to the same irreversible action, so a space bar pressed mid-sentence stopped
  the capture and went straight into transcription with nothing to undo it. Only `q`
  stops now, and it asks first: the confirmation is the recording screen with a
  question added, so the clock keeps climbing and the meters keep moving while you
  decide. `y` stops and transcribes; anything else — a second `q` included — keeps
  recording. Every other key during a recording is ignored.

- **A recording no longer disappears when the microphone is busy for a moment.**
  CoreAudio turns down a device that is busy or still settling, briefly and flatly,
  and a single refused open was fatal: the folder was thrown away and the user — who
  had just typed a name — was dropped back at the menu with the reason on the status
  line, which the footer hands to any background job that wants it. Publishing holds
  that line for minutes after each recording, which is exactly when the next one gets
  started, so recording twice in a row could look like the app was ignoring the
  keypress. Eight named, silent folders on disk were the only trace. The microphone now
  gets three attempts, re-reading the device list between them (a stale device index
  is the other way this has failed), and a start that really cannot happen stops on a
  screen that says why and keeps the typed name, so Enter is a retry.

- **Failures are written to `audiocript.log`.** Next to `config.json`, with the
  traceback. The status line is volatile — the next message wipes it — so something
  that goes wrong from time to time used to leave nothing behind to diagnose it from.

- **A model that fails to load now says why.** The header has one word for every way
  the load can go wrong — `error` — and the message the warm-up built was kept only in
  memory, so a missing package, a download the network refused and a full disk were
  indistinguishable and nothing on disk could tell them apart. The reason and its
  traceback go to `audiocript.log` now, written before the header changes, so the log
  already has the answer by the time anyone goes looking. A load that works still
  writes nothing, and a failure stays retryable.

- **A recording that never started leaves nothing behind.** `meta.json` was written
  before the microphone was opened and a refused open left an empty `mic.raw`, so the
  `rmdir()` meant to clean up could not possibly succeed. The recorders now clean up
  their own raw file when an open is refused (which also stops leaking the file
  handle), the metadata is written once audio is really arriving, and the discard
  refuses to delete a folder that holds any capture.

- **A pasted repository URL now works.** `GITHUB_REPO_FOR_TRANSCRIPTS` was dropped
  into the API path verbatim, so anything but a bare `owner/repo` produced
  `/repos/https://github.com/owner/repo` and could never publish — while the natural
  thing to configure is the address from the browser. Browser URLs, SSH remotes and
  `owner/repo` are all accepted now.

- **A publish can be retried.** Publishing only ever ran when a transcript was first
  saved, so one that failed — or that was killed by quitting the app mid-run, which
  takes minutes — left a recording that nothing could ever publish again. Press `u` on
  a recording to run it, and stages that already produced output are reused instead of
  being paid for a second time. `u` also says why it declined (too short, already
  published, not configured), where the automatic pass is deliberately silent.

- **A failed publish leaves a trace.** The only report was the status line, which is
  volatile and is lost entirely when the app closes — so the first real failure left
  nothing anywhere to say what had happened. The error is now written to `meta.json`
  as `publish_error`.

- **The app no longer hangs on startup after an interrupted recording.** Crash
  recovery resampled the whole capture in one `torchaudio.resample` call. At
  48 kHz → 16 kHz that reduces to a 411-tap, stride-3 single-channel `conv1d`, and
  PyTorch's CPU path materialises an im2col buffer of ~550 bytes per input sample —
  so a 48-minute capture asked for ~76 GB and the process thrashed instead of
  finishing. Because recovery ran before the UI existed, the terminal stayed blank
  and the app looked dead on every launch. Resampling is now done in blocks with
  the overlap the filter needs, so the output is unchanged: 5 minutes of audio went
  from 5.46 s / 6.75 GB to 0.63 s / 0.07 GB, and the 48-minute recording that
  triggered this rebuilt in 12 s.

- **Recovery now runs behind the UI and reports progress.** It used to block
  `main()` before `Live` started, so rebuilding a long recording meant seconds of
  blank terminal. It now runs on a worker thread once the UI is up, showing
  `Recovering "name"…  43%` on the status line, and the menu — including starting a
  new recording — stays usable throughout. Measured with a 10-minute interrupted
  capture: the first frame lands at 0.73 s while recovery finishes at 5.77 s.
  Recovery borrows the status line only while it runs, so it never overwrites the
  user's own messages.

- **Finalizing a recording no longer loads it into memory.** Both the `q` path and
  recovery read each raw capture with `Path.read_bytes()` and inflated it again via
  `astype(float32)`, peaking at 5.4 GB for a 48-minute two-source capture. Raw
  captures are now read block by block straight off disk and resampled, mixed and
  written as streams: a 30-minute two-source finalize now peaks at effectively no
  extra memory. Audio output is byte-identical, peak normalisation included.

- **An interrupted finalize can no longer destroy a recording.** Output was written
  directly to `audio.wav`, so a process killed mid-write left a truncated file while
  `capture.in_progress` was still set — and the next launch took that file as proof
  the recording had finished and deleted the raw capture. Everything is now written
  to `.part` files and renamed once complete, with `audio.wav` renamed last, so an
  interrupted finalize leaves the raw files untouched and simply runs again. A
  recovery that fails now keeps its folder too; only a folder with nothing to
  recover is removed.

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
- **Python 3.12 is now the minimum**, and `run.sh` checks it before doing anything
  else — both the interpreter it is given and the one an existing `.venv` was built
  with. An older Python used to fail much later, in the middle of a dependency
  build, as an error about something else entirely.
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
