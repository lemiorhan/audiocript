# Voice Transcriptor - Terminal App Edition

A full-screen terminal app for **recording audio and transcribing it locally** with
Whisper — on macOS. Capture your **microphone and the system audio (Zoom, Meet,
YouTube, …) at the same time**, watch live input/output level meters while you
record, and get a transcript saved next to the recording. Everything runs on your
machine; no audio leaves your computer.

> The interface is in English. The **transcription language** is selectable
> (Turkish or English) and each language uses a dedicated, high-quality model.

![Voice Transcriptor — home](assets/tui-home.png)

![Voice Transcriptor — recording with live level meters](assets/tui-recording.png)

---

## Features

- **Full-screen terminal UI** — header (current settings + clock), a main panel,
  and a footer with single-key shortcuts. No typing commands, no Enter needed.
- **Record mic + system audio together** — your voice (microphone) and the
  computer's output (a call, a video) are captured simultaneously and mixed into
  one mono track for transcription.
- **Transcribe an existing file** — import an `mp4`, `mov`, `wav`, `mp3` or `m4a`
  file (native file picker); its audio is extracted into a new project folder and
  transcribed just like a recording.
- **System audio without BlackHole** — uses a native macOS **Core Audio process
  tap**, so playback keeps playing through your speakers/headphones normally. No
  virtual cable, no Multi-Output Device, no rerouting.
- **Live level meters (VU)** — speak and the mic bar moves; play audio and the
  system bar moves. Instantly see whether capture is actually working (`no signal`
  is shown for a silent source).
- **Per-language models** for the best quality in each language:
  - **Turkish** → [`selimc/whisper-large-v3-turbo-turkish`](https://huggingface.co/selimc/whisper-large-v3-turbo-turkish) (Hugging Face Transformers)
  - **English** → [`ggml-distil-large-v3`](https://huggingface.co/distil-whisper/distil-large-v3-ggml) (whisper.cpp via `pywhispercpp`)
- **Runs on the best device automatically** — CUDA → Apple Silicon (MPS/Metal) → CPU.
- **Per-recording project folders** — each recording gets its own timestamped
  folder containing `audio.wav` and `transcription.txt`.
- **Opens the transcript automatically** — after each run the transcript is opened
  in an app you pick from your installed apps (defaults to **Sublime Text** if
  installed; can be turned off).
- **Remembers your preferences** — language, microphone, system-audio toggle,
  recordings folder and "open with" app are saved to `config.json`.
- **One command to run** — `./run.sh` sets up the environment and launches the app.

---

## Requirements

- **macOS 14.4+** (Core Audio process taps are used for system-audio capture).
- **Python 3.10+**.
- **Xcode / Command Line Tools (`swiftc`)** — only needed for the optional
  system-audio capture (a tiny Swift helper is compiled on first use). Plain
  microphone recording does not require it.
- **ffmpeg** — only needed to import/transcribe existing media files
  (`brew install ffmpeg`). Recording does not require it.

The first run downloads the transcription models (English ≈ 1.5 GB, Turkish
≈ 1.6 GB) and caches them; later runs are offline-capable for cached models.

---

## Quick start (one command)

```bash
./run.sh
```

`run.sh` creates the virtual environment, installs dependencies from
`requirements.txt` the first time (and only re-installs when that file changes),
then launches the app.

> If you get `./run.sh: Permission denied`, either run `chmod +x run.sh` once or
> start it with `bash run.sh`. To pick a specific interpreter:
> `PYTHON=python3.11 ./run.sh`.

### Manual setup (alternative)

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
python meeting-transcriptor.py
```

---

## Usage

The app opens a full-screen interface. Controls are single keypresses (no Enter):

| Key | Action |
|-----|--------|
| `r` (or `Space`) | Start recording |
| `f` | Import an existing audio/video file and transcribe it |
| `q` | While recording: **stop & transcribe**. On the home screen: **quit** |
| `l` | Toggle transcription language (TR ⇄ EN) |
| `d` | Open the microphone picker (`1`–`9` to choose, `Esc` to cancel) |
| `s` | Toggle system-audio capture on/off |
| `o` | Choose which app opens the transcript (type to filter, `0` = off) |
| `p` | Edit the recordings folder (`Enter` to save, `Esc` to cancel) |

`Ctrl-C` exits cleanly at any time.

When you press `r`, the app shows a **"Preparing to record…"** screen while it
opens the microphone and starts system-audio capture (the first run may compile a
small helper or ask for a permission). It switches to the recording view and shows
**"Recording started — speak now."** only once capture has actually begun — so you
never start talking before it's listening.

While recording, the panel shows a live **VU meter** for each source:

![Microphone picker](assets/tui-mic-picker.png)

When you stop, the app shows a "Transcribing…" panel (the model loads on first
use), then displays the transcript, saves it, and (optionally) opens it in your
chosen app.

### Transcribe an existing file

Press `f` to transcribe a file you already have. A native file picker opens; choose
an `mp4`, `mov`, `wav`, `mp3` or `m4a` file. Its audio is extracted (via ffmpeg)
into a **new project folder** as `audio.wav` (16 kHz mono) and transcribed using
the current language — identical to a recording, including the saved
`transcription.txt` and auto-open.

### Open the transcript in an app

After each transcript is saved it is opened in an app of your choice (e.g.
**Sublime Text**, VS Code, TextEdit). Press `o` to pick from your installed apps —
just start typing to filter the list, press `1`–`9` to select (or `Enter` for the
first match), or `0` to turn auto-open off. Your choice is saved. On first run the
app defaults to **Sublime Text** if it is installed, otherwise auto-open is off.

![Open transcript with — app picker](assets/tui-open-with.png)

---

## How it works

### Recording both mic and system audio

macOS does not let an app record an output (speaker) device directly. Many tools
work around this with a virtual cable (BlackHole) and a Multi-Output Device, but
that mutes your own playback or requires fiddly routing. Instead, this app uses a
**Core Audio process tap** (`CATapDescription` with `muteBehavior = .unmuted`)
via a small Swift helper (`mac_audio_tap/system_audio_tap.swift`, compiled on
first use). The tap captures the whole system mix **while you keep hearing it**.

The mic is captured with `sounddevice`. Each source is opened at its **native
sample rate** (never forced to 16 kHz) with
`CoreAudioSettings(change_device_parameters=False)`, so the app never changes a
device's global sample rate — which previously could interrupt playing audio.

### Mixing and transcription

Each source is resampled to **16 kHz mono** with `torchaudio` (Kaiser
anti-aliasing filter), the sources are trimmed to the shortest and mixed (summed
with peak-limiting), and written as a single 16 kHz mono `audio.wav` — exactly
what Whisper expects. The transcript is produced by the model for the selected
language and written to `transcription.txt`.

### Output layout

```
<recordings folder>/
└── 2026-06-10_14-30-15/
    ├── audio.wav          # 16 kHz mono mix
    └── transcription.txt  # transcript
```

### Configuration (`config.json`)

Saved next to the script and updated as you change settings:

```json
{
  "language": "tr",
  "base_path": "/abs/path/to/recordings",
  "input_device": "Yeti Stereo Microphone",
  "capture_system_audio": true,
  "open_app": "Sublime Text"
}
```

`input_device` is stored by **name** (device indices change between sessions).

---

## Permissions (macOS)

On first use macOS will prompt for permissions — grant them under
**System Settings → Privacy & Security**:

- **Microphone** — for mic recording.
- **System Audio Recording** — for the system-audio tap. If denied (or if
  `swiftc` is unavailable), the app automatically continues with mic-only.
- **Accessibility** — you may see a one-time notice; not required for normal use.

---

## Troubleshooting

- **A level bar stays empty / shows `no signal`** — that source isn't capturing.
  For the mic, pick a real microphone with `d` (not a virtual device). For system
  audio, make sure something is actually playing and the permission was granted.
- **System audio is off / "system audio unavailable"** — install Xcode Command
  Line Tools (`xcode-select --install`) so the helper can compile, and grant the
  *System Audio Recording* permission.
- **First transcription is slow** — the model is downloaded and loaded on first
  use; subsequent runs reuse the cache.

---

## Project structure

```
meeting-transcriptor.py        # the app (TUI + recording + transcription)
mac_audio_tap/
  └── system_audio_tap.swift   # Core Audio process-tap helper (compiled on first run)
run.sh                         # one-command launcher
requirements.txt
docs/superpowers/specs/        # design notes
```

There is also a standalone `mp4-transcriptor.py` helper that predates the in-app
**Import file** feature (`f`); for most uses, importing a file in the app is the
simpler path.

---

## Contributing

Contributions are welcome — please read [CONTRIBUTING.md](CONTRIBUTING.md) and the
[Code of Conduct](CODE_OF_CONDUCT.md). For security issues, see
[SECURITY.md](SECURITY.md).

## License

Released under the [MIT License](LICENSE).

## Acknowledgements

- [OpenAI Whisper](https://github.com/openai/whisper) and
  [Distil-Whisper](https://github.com/huggingface/distil-whisper)
- [`selimc/whisper-large-v3-turbo-turkish`](https://huggingface.co/selimc/whisper-large-v3-turbo-turkish)
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) /
  [`pywhispercpp`](https://github.com/absadiki/pywhispercpp)
- [Rich](https://github.com/Textualize/rich), [sounddevice](https://python-sounddevice.readthedocs.io/)
- Semih Şahan as the owner of original idea and [base repository](https://github.com/semihshn/voice_transcriptor)
