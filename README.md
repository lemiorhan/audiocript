<div align="center">

<img src="assets/header.png" alt="Audiocript" width="820">

**Record & transcribe audio locally with Whisper — on macOS.**
Capture your microphone *and* your system audio at the same time, watch live level
meters, and get a saved transcript. Everything runs on your machine.

[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E.svg)](LICENSE)
[![Platform: macOS 14.4+](https://img.shields.io/badge/platform-macOS%2014.4%2B-0EA5E9.svg)](#requirements)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-3776AB.svg)](#requirements)
[![Transcription: Whisper](https://img.shields.io/badge/transcription-Whisper-8B5CF6.svg)](#models-per-language)
[![Runs offline](https://img.shields.io/badge/privacy-runs%20offline-14B8A6.svg)](#how-it-works)

</div>

---

<div align="center">

<img src="assets/tui-menu.png" alt="Audiocript main menu" width="48%">
<img src="assets/tui-recording.png" alt="Recording with live VU meters" width="48%">
<img src="assets/tui-viewer.png" alt="In-app transcript viewer" width="48%">
<img src="assets/tui-import.png" alt="Importing a file with progress" width="48%">

</div>

---

## Features

- **Full-screen terminal app** with **arrow-key navigation** and a grouped,
  collapsible menu — no commands to memorize.
- **Record mic + system audio together** — your voice and the computer's output
  (a call, a video) are captured simultaneously and mixed into one track.
- **Automatic acoustic echo cancellation** — when recording both sources,
  speaker leakage is removed from the microphone at finalization when available.
- **System audio without BlackHole** — a native macOS **Core Audio process tap**
  keeps your audio playing through your speakers while it's captured. No virtual
  cable, no Multi-Output Device, no rerouting.
- **Live level meters (VU)** — see whether each source is actually being captured
  (`no signal` is shown for a silent one).
- **Named recordings** — name a recording when you create it, and rename any
  recording at any time.
- **Browse & read transcripts in-app** — a scrollable viewer; press `Enter` to
  open the transcript in your preferred external app.
- **Import existing media** — transcribe an `mp4`, `mov`, `wav`, `mp3` or `m4a`
  file with a live progress bar.
- **Speaker labels (optional)** — tag who's talking: your mic becomes **`[Me]`**
  and remote people on the call become **`[Speaker 1]`, `[Speaker 2]`, …** See
  [below](#speaker-labels-who-is-talking).
- **Per-language models** for quality (Turkish & English — see [below](#models-per-language)).
- **Runs on the best device automatically** — CUDA → Apple Silicon (MPS/Metal) → CPU.
- **Background model pre-warming** so your first transcript is fast.
- **System-wide dictation (optional)** — a global hotkey records anywhere on
  macOS and delivers a corrected transcript straight to your clipboard, no app
  window needed. Sends transcript text to an AI provider, so it is not fully
  offline — see [below](#dictation-optional-off-by-default).
- **One command to run** — `./run.sh` sets everything up.

---

## Requirements

- **macOS 14.4+** (Core Audio process taps are used for system-audio capture).
- **Python 3.12+** (`run.sh` refuses to go on with anything older).
- **Xcode / Command Line Tools (`swiftc`)** — only for the optional system-audio
  capture (a tiny Swift helper is compiled on first use).
- **ffmpeg** — only to import existing media files (`brew install ffmpeg`).

`run.sh` checks for these and offers to install the missing ones. The first run
downloads the transcription models (≈1.5–1.6 GB) and caches them.

---

## Quick start

```bash
./run.sh
```

`run.sh` creates the virtual environment, installs Python dependencies the first
time (and only when its requirements files change), offers to install the optional
external tools if missing, then launches the app. It also tries to install optional
WebRTC acoustic echo cancellation (`pywebrtc-audio`). That package can need a native
compiler on Python versions without a prebuilt wheel; if installation fails,
Audiocript warns and starts normally. Retry it later with
`.venv/bin/python -m pip install -r requirements-aec.txt`.

> Permission denied? Run `chmod +x run.sh` once, or use `bash run.sh`.
> Pick a Python with `PYTHON=python3.12 ./run.sh`. Skip the tool check with
> `SKIP_DEP_CHECK=1`.

<details>
<summary>Manual setup</summary>

```bash
python3 -m venv .venv   # Python 3.12+
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-aec.txt  # optional WebRTC acoustic echo cancellation
python audiocript.py
```
</details>

---

## Usage

Audiocript is a full-screen app you drive with the **arrow keys**.

| Context | Keys |
|---------|------|
| **Menu** | `↑`/`↓` move · `Enter` open/expand · `→`/`←` expand/collapse · `q` quit |
| **Menu — on a recording** | `Enter` view transcript · `p` play/stop audio · `t` transcribe (if it has none) · `u` publish · `r` rename · `d` delete |
| **Transcript viewer** | `↑`/`↓` scroll · `PgUp`/`PgDn` page · `Enter` open in external app · `p` play/stop · `r` rename · `d` delete · `Esc` back |
| **Text fields** (name / rename / folder / app filter) | type · `Enter` confirm · `Esc` cancel |
| **Microphone / app pickers** | `1`–`9` select · `Esc` cancel |
| **Recording** | `q` stop & transcribe (asks first) — every other key is ignored |
| **Stop / delete confirmation** | `y` confirm · any other key cancels |

The menu groups everything:

- **New recording** / **Import file** — you're asked for a name first (optional),
  then it records (or opens a file picker for `mp4`/`mov`/`wav`/`mp3`/`m4a`).
- **Recordings** — every project (name · date · language). On a selected recording
  you can **view** the transcript (`Enter`), **play/stop** its audio (`p`),
  **rename** it (`r`), or **delete** it (`d`, with confirmation). Inside the viewer
  the transcript scrolls and `Enter` opens it in your external app.
- **Settings** — language (TR/EN), microphone, system-audio capture, **speaker
  labels**, the "open with" app, and the recordings folder.

`Ctrl-C` exits cleanly at any time.

### While recording

A **"Preparing…"** screen appears while the mic and system-audio tap start up
(the first run may compile the helper or ask for a permission), then it switches
to live **VU meters** and shows **"Recording started — speak now."** Press `q` to
stop and transcribe — it asks first, and keeps recording (clock running, meters
moving) until you answer `y`. No other key does anything while recording, so a
stray space bar cannot end the capture.

### Importing a file

Audio is extracted (via ffmpeg) into a new project folder as `audio.wav`
(16 kHz mono) and transcribed like a recording, with a **live progress panel**
(real % for extraction and for English transcription).

---

## How it works

### Recording both mic and system audio

macOS doesn't let an app record an output device directly. Instead of a virtual
cable (BlackHole) + Multi-Output Device, Audiocript uses a **Core Audio process
tap** (`CATapDescription` with `muteBehavior = .unmuted`) via a small Swift helper
(`mac_audio_tap/system_audio_tap.swift`, compiled on first use). It captures the
whole system mix **while you keep hearing it**. The mic is captured with
`sounddevice` at its native rate (never forced), so connecting never changes a
device's global sample rate or interrupts playback.

### Mixing & transcription

Each source is resampled to **16 kHz mono** with `torchaudio` (Kaiser
anti-aliasing), trimmed to the shortest, summed with peak-limiting, and written as
one mono `audio.wav` — exactly what Whisper expects.

When both microphone and system audio were captured, Audiocript first aligns their
start times, then automatically applies **acoustic echo cancellation** to remove
speaker output leaking back into the microphone before mixing. This is most useful
for calls played through speakers; **headphones** remain the cleanest option. If the
optional WebRTC processor is unavailable or fails, finalization safely uses the
aligned original microphone instead, so the recording is still saved and can be
transcribed.

### Models (per language)

`distil-large-v3` is English-only, so each language uses a dedicated model:

| Language | Model | Runtime |
|----------|-------|---------|
| Turkish (`tr`) | [`selimc/whisper-large-v3-turbo-turkish`](https://huggingface.co/selimc/whisper-large-v3-turbo-turkish) | Transformers |
| English (`en`) | [`ggml-distil-large-v3`](https://huggingface.co/distil-whisper/distil-large-v3-ggml) | whisper.cpp (`pywhispercpp`) |

### Speaker labels (who is talking)

Off by default. Turn it on in **Settings → Speaker labels**. When on, transcripts
are tagged by speaker:

```
[Me] Can you walk me through the rollout plan?

[Speaker 1] Sure — we ship to 10% on Friday, then ramp.

[Speaker 2] And the rollback path is the feature flag.
```

How it works: Whisper transcribes *what* was said, not *who* said it, so labels
come from two sources combined:

- **Your mic is always `[Me]`** — it's a separate channel, so no guessing.
- **Remote people (system audio)** are told apart by acoustic **diarization**
  ([`pyannote.audio`](https://github.com/pyannote/pyannote-audio)): the system
  channel is split into `Speaker 1`, `Speaker 2`, … and each line is matched to a
  speaker by timestamp overlap.

To enable it (one-time):

1. `pip install pyannote.audio` (already in `requirements.txt`).
2. Create a free [Hugging Face token](https://huggingface.co/settings/tokens) and
   **accept the terms** of the gated model at
   [`pyannote/speaker-diarization-3.1`](https://huggingface.co/pyannote/speaker-diarization-3.1).
3. Make the token available as `HF_TOKEN` (env var) or `"hf_token"` in
   `config.json`.

Notes & limits:

- Diarization adds a processing pass and is **slower**; the model downloads once,
  then runs **fully offline**. If the package or token is missing, Audiocript
  **falls back** to a normal unlabeled transcript and tells you why.
- It's **acoustic clustering**, so the remote speaker count can occasionally be off
  and names are generic (`Speaker 1`) until you rename them.
- **Imported files** have no separate mic channel, so the whole file is diarized
  into `Speaker 1/2/…` with no `[Me]`.
- **Punctuation & casing** are produced by the Whisper models themselves and are
  always present, regardless of this setting.

### Output & configuration

```
<recordings folder>/
└── 2026-06-10_14-30-15/
    ├── audio.wav          # 16 kHz mono mix
    ├── mic.wav            # echo-cleaned mic channel when AEC succeeds
    │                      # (only when speaker labels are on)
    ├── system.wav         # system-only channel (only when speaker labels are on)
    ├── transcription.txt  # transcript (labeled by speaker when enabled)
    └── meta.json          # { "name": "Daily standup", "language": "tr",
                           #   "speakers": ["Speaker 1", "Speaker 2"] }
```

Preferences are saved in `config.json` (language, microphone by name, system-audio
toggle, speaker-labels toggle + optional `hf_token`, recordings folder, and the
"open with" app).

### Publishing transcripts (optional, off by default)

Audiocript can turn a long transcript into two more useful forms and file all three in
a GitHub repository — when you ask it to, by pressing **`u`** on a recording.

Create a `.env` next to `audiocript.py` with three values:

```
GITHUB_REPO_FOR_TRANSCRIPTS=owner/repo
GITHUB_TOKEN=…               # needs contents:write on that repository
OPENAI_API_KEY=…
```

Any of these can be an exported environment variable instead, which takes precedence
over the file — so an `OPENAI_API_KEY` you already have in your shell just works. The
repository can be given as `owner/repo`, as the address from your browser, or as an
SSH remote; they all mean the same thing.

A transcript longer than `TRANSCRIPT_MIN_CHARS` (2000 characters by default) is then
sent through `prompts/transcript_prompt.md`, which cleans up the speech without
flattening who said what or how sure they were, and that result through
`prompts/documentation_prompt.md`, which reorganises it into a document by topic. Both
prompts are written to lose nothing, so expect output about as long as the input.

The repository gets one commit per recording, named after it:

```
2026-06-10_14-30-15-daily-standup/
├── transcript.raw.md      # exactly what Whisper produced
├── transcript.edited.md   # first pass
└── documentation.md       # second pass
```

Both model outputs are also written next to the recording, before anything is pushed,
so a failed push costs nothing but the retry. Progress shows on the status line while
the menu stays usable. Two optional settings: `OPENAI_MODEL` (default `gpt-4.1` — it
needs a large output budget, since the response is as long as the transcript) and
`TRANSCRIPT_MIN_CHARS`.

Nothing publishes by itself. Transcribing saves the transcript and stops there —
two paid model calls are not something to start on your behalf — so publishing waits
for **`u`**. Press it again after a failure, after the app was closed mid-publish, or
after setting the keys up later: anything already generated is reused rather than paid
for again. `u` also tells you why it declined (too short, already published, not
configured) instead of doing nothing quietly. A failure is written to `meta.json` as
`publish_error` as well, so it is still there after the status line has moved on.

Leave any of the three required values out and nothing happens: this is off unless you
configure it. Never commit `.env` — it is gitignored for that reason.

---

### Dictation (optional, off by default)

Audiocript can also run as a background dictation daemon instead of the TUI:
press a global hotkey anywhere on macOS, speak, press it again, and the
corrected text lands on your clipboard — no need to bring the app to the
foreground, or even have it open.

```bash
./run.sh --dictate
```

**This is not offline.** The badge at the top of this README is earned by
transcription, which stays local (the same Whisper models used everywhere else
in the app). Dictation adds a correction pass on top: every dictation, with no
length threshold and no separate opt-in beyond starting the daemon, has its
transcript sent as text to OpenAI. Do not dictate anything you would not want
to leave your machine — a password included.

Press **`<cmd>+<alt>+d`** (Cmd+Option+D) anywhere to start a recording; press
it again to stop, correct, and copy the result — it replaces whatever was
already on your clipboard. Forgetting to press it again is covered too: a
recording stops itself after `dictation_max_seconds` (5 minutes by default).
Nothing is saved to disk — the audio and its transcript exist only for the
duration of one dictation and are discarded once the clipboard is written.

Dictation transcribes in whichever language is set in **Settings → language**
— the same setting the rest of the app uses, so a session is Turkish or
English, never a per-dictation choice.

Two `config.json` keys, both optional:

| Key | Default | Meaning |
|-----|---------|---------|
| `dictation_hotkeys` | `{"toggle": "<cmd>+<alt>+d"}` | The hotkey that starts and stops a dictation. |
| `dictation_max_seconds` | `300` | A recording is stopped automatically after this many seconds. |

It also needs `OPENAI_API_KEY` — the same key set up for
[publishing](#publishing-transcripts-optional-off-by-default) above, in `.env`
or the environment — and refuses to start without it. One more optional
environment variable: `DICTATION_MODEL` (default `gpt-4.1-mini`) picks the
model used for the correction pass.

Like any global hotkey on macOS, it needs the **Input Monitoring** permission:
grant it under **System Settings → Privacy & Security → Input Monitoring** to
the terminal (or Python interpreter) running the daemon, then start it again.
Without it the daemon still runs, but the hotkey does nothing — use
`.venv/bin/python dictate.py --toggle` and `--stop` instead, which work with no
permission at all, from another shell, a macOS Shortcut, or `skhd`.

---

## Permissions (macOS)

On first use, grant these under **System Settings → Privacy & Security**:

- **Microphone** — for recording.
- **System Audio Recording** — for the tap. If denied (or `swiftc` is missing),
  the app continues mic-only.
- **Input Monitoring** — only for [dictation](#dictation-optional-off-by-default)'s
  global hotkey. If denied, the daemon still runs; `--toggle`/`--stop` work
  without it.

---

## Troubleshooting

- **A level bar stays empty / `no signal`** — that source isn't capturing. Pick a
  real microphone in Settings; for system audio make sure something is playing and
  the permission was granted.
- **System audio unavailable** — install the Xcode Command Line Tools
  (`xcode-select --install`) and grant *System Audio Recording*.
- **Can't import a file** — install ffmpeg (`brew install ffmpeg`).
- **First transcription is slow** — the model loads on first use, then it's cached.
- **Echo cancellation is unavailable** — the recording was still saved using the
  aligned original microphone. Retry the optional install with
  `.venv/bin/python -m pip install -r requirements-aec.txt`; a native compiler may
  be needed when `pywebrtc-audio` has no wheel for your Python version. Headphones
  are the cleanest way to avoid speaker leakage.
- **The dictation hotkey does nothing** — grant *Input Monitoring* (see
  [Permissions](#permissions-macos)) to the terminal or Python interpreter
  running `./run.sh --dictate`, then start the daemon again. Until then,
  `.venv/bin/python dictate.py --toggle` (or `--stop`) works without any
  permission at all.

---

## Project structure

```
audiocript.py                  # the app (TUI + recording + transcription)
dictate.py                     # the dictation daemon: state machine, hotkey, CLI
dictation.py                   # dictation config, clipboard, status sinks, correction
publish.py                     # optional: OpenAI passes + the GitHub commit
prompts/                       # transcript_prompt.md + documentation_prompt.md
                                #   (publishing), dictation_prompt.md (dictation)
mac_audio_tap/
  └── system_audio_tap.swift   # Core Audio process-tap helper (compiled on first run)
tests/                         # plain scripts; see tests/run_all.py
run.sh                         # one-command launcher
requirements.txt
requirements-aec.txt           # optional WebRTC acoustic echo cancellation
assets/                        # logo, header, screenshots
```

---

## Contributing

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) and the
[Code of Conduct](CODE_OF_CONDUCT.md). For security reports, see
[SECURITY.md](SECURITY.md).

## Credits & attribution

Audiocript is a derivative of **[voice_transcriptor](https://github.com/semihshn/voice_transcriptor)
by Semih Şahan**, used and distributed under the terms of its **MIT License**. The
original copyright and license are retained in [LICENSE](LICENSE), and attributions
for the original work and third-party components are listed in [NOTICE.md](NOTICE.md).
Thank you to the original author for releasing the project under a permissive license.

## License

Released under the **MIT License** — see [LICENSE](LICENSE).
Copyright © 2025 Semih Şahan (original author of voice_transcriptor) and the
Audiocript contributors. As required by the MIT License, the original copyright
notice and permission notice are preserved.

## Acknowledgements

- [OpenAI Whisper](https://github.com/openai/whisper) · [Distil-Whisper](https://github.com/huggingface/distil-whisper)
- [`selimc/whisper-large-v3-turbo-turkish`](https://huggingface.co/selimc/whisper-large-v3-turbo-turkish)
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) · [`pywhispercpp`](https://github.com/absadiki/pywhispercpp)
- [Rich](https://github.com/Textualize/rich) · [sounddevice](https://python-sounddevice.readthedocs.io/)
- [`pywebrtc-audio`](https://pypi.org/project/pywebrtc-audio/) for optional WebRTC acoustic echo cancellation
