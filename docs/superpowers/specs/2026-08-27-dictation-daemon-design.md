# Dictation Daemon Design

## Goal

Let the user dictate short text into any application — a terminal, Slack, WhatsApp —
without leaving it. One global hotkey starts the microphone, the same hotkey stops it.
Audiocript then transcribes the speech locally, sends the transcript to an AI provider
for a **minimal correction pass only**, and puts the result on the clipboard. The user
pastes.

The value is the clipboard being ready, not a saved recording. Nothing is filed in the
recordings folder.

## Scope

A new long-lived process, split across two modules that reuse Audiocript's existing
capture and transcription code by importing it:

- `dictation.py` — no threads, no global state: configuration resolution and
  validation, the clipboard, the status sinks, the correction and delivery pipeline.
- `dictate.py` — the state machine, the capture, the hotkey listener, the signal
  handling and the CLI.

The line between them is testability: everything in `dictation.py` can be exercised
without audio hardware, a loaded model, or a running daemon, which is where most of
this feature's rules live. `dictate.py` imports `dictation.py`; not the reverse.

The full-screen TUI in `audiocript.py` is not changed; only `run.sh` gains a
`--dictate` switch so the daemon can be launched without naming the virtualenv's
interpreter. `config.json` gains keys the daemon reads and the TUI ignores.

**One language at a time.** The daemon reads the language from the existing
`language` key in `config.json` and warms exactly one model. There is no per-hotkey
language and no automatic detection: the configuration is either entirely Turkish or
entirely English.

Out of scope for this change:

- A menu bar indicator. The architecture reserves the main thread for it (see
  *Status feedback*) but it is a separate piece of work.
- Streaming/partial transcription. The transcript is produced after the recording stops.
- Windows or Linux. Clipboard, notifications and the event tap are all macOS-specific.
- Saving dictations. Audio and transcript live in a temporary directory and are deleted.

## Chosen Approach

A single process holding everything: the hotkey listener, the state machine, and the
warm model.

`pynput` is already a declared dependency and is already imported by `audiocript.py`,
though nothing uses it yet. Its `keyboard.GlobalHotKeys` parses a combination string
directly, so no key-tracking code is needed.

Two properties of `pynput`'s macOS backend shape the design and were confirmed by
reading `pynput/_util/darwin.py` and `pynput/_util/__init__.py`:

- `AbstractListener` extends `threading.Thread`, and the darwin listener calls
  `CFRunLoopGetCurrent()` inside its own `_run`. The listener therefore builds a run
  loop on **its own thread** and never claims the process's main run loop. The main
  thread stays free, which is what lets a menu bar indicator be added later without
  restructuring anything.
- `ListenerMixin.IS_TRUSTED` is assigned from `HIServices.AXIsProcessTrusted()` at the
  top of `_run`, and `_create_event_tap` returning `None` causes an immediate
  `_mark_ready()` and return. Missing Input Monitoring permission is therefore
  *detectable* rather than silent: after `listener.wait()` the daemon can read
  `IS_TRUSTED` and check `is_alive()`.

**An external trigger is kept as an escape hatch.** The daemon writes a PID file, and
`dictate.py --toggle` signals it. Granting Input Monitoring to a virtualenv interpreter
is the one genuinely unknown risk in this design; if it cannot be granted, the user
binds the hotkey in macOS Shortcuts or `skhd` to `--toggle` and the rest of the system
is unaffected.

### Rejected alternatives

- **`pynput` with no external trigger.** Saves roughly ten lines and removes the only
  fallback for the design's single unknown. Not worth it.
- **No `pynput`, hotkey delegated entirely to macOS Shortcuts / `skhd`.** Avoids the
  permission question but pushes a setup step onto the user and leaves an already
  installed dependency unused. The chosen approach contains this option as a fallback.
- **Two processes (thin listener + worker) communicating over a socket.** The worker
  must stay warm anyway — reloading a Whisper model per dictation is far too slow —
  so this buys a second lifecycle to manage and nothing else.
- **Adding a dictation mode inside `audiocript.py`.** That module is already near three
  thousand lines and carries the TUI, capture, mixing, transcription and diarization.
  A separate module that imports it keeps the dependency one-directional.

## Architecture

### Threads

| Thread | Responsibility |
|---|---|
| main | Owns the state machine; blocks on an `Event`. Deliberately left otherwise idle so a menu bar loop can take it later. |
| hotkey listener (`GlobalHotKeys`) | Translates the combination into a toggle event. Started once, lives for the process. |
| sounddevice callback | Existing behaviour of `DeviceRecorder`; writes native-rate raw PCM to disk. |
| worker (one per dictation) | Transcribe, correct, copy, notify, clean up. Keeps the listener responsive. |

### State machine

Three states behind one lock:

```
IDLE ──toggle──▶ RECORDING ──toggle──▶ PROCESSING ──▶ IDLE
                                  ▲                    │
                                  └─ toggle: ignored ──┘
```

- **IDLE → RECORDING.** Create a temporary directory, open the microphone through
  `audiocript._start_mic`, announce `recording` on the status sink.
- **RECORDING → PROCESSING.** Stop the recorder, convert the raw PCM to a 16 kHz mono
  WAV through `audiocript._stream_source_to_wav`, hand it to a worker thread, announce
  `processing`.
- **PROCESSING + toggle.** Ignored, with a notification saying so. A second dictation
  must not start while the first is still being corrected.
- **Worker.** Transcribe, correct, copy to the clipboard, announce the result, delete
  the temporary directory. Return to IDLE whatever happened.

### Reuse from existing modules

Cited by symbol; coordinates must be re-resolved when the work starts.

| Need | Existing symbol |
|---|---|
| Open the microphone, retrying a refusal | `audiocript._start_mic` |
| Microphone capture to raw PCM | `audiocript.DeviceRecorder` |
| Raw PCM → 16 kHz mono WAV | `audiocript._stream_source_to_wav` |
| Resolve the configured mic name to an index | `audiocript._resolve_mic_index` |
| Read `config.json` | `audiocript.load_config` |
| Load and cache the language's model | `audiocript._ensure_model` |
| Audio → text | `audiocript.transcribe_audio` |
| Valid language codes | `audiocript.LANGUAGES` |
| Read `.env` with environment override | `publish.env_value` |
| Inline a body into a prompt file's placeholder | `publish.render_prompt` |
| One chat completion, returned as text | `publish.openai_complete` |

`_start_mic` takes a state object but reads only `.mic_index` and `.cfg` from it. The
daemon passes a small two-field shim rather than constructing a `_TuiState`, so no TUI
state reaches the daemon. This reuse is deliberate: that function's retry logic exists
because refused microphone opens cost two real recordings, and duplicating the call
would lose it.

`publish.config` is **not** reused. It returns `None` unless the two GitHub values are
present alongside `OPENAI_API_KEY`, and dictation needs no GitHub access. `dictate.py`
defines its own `dictation_config()` built on `publish.env_value`, so `.env` parsing
still lives in one place.

## Configuration

Configuration is split along the line the project already uses: `config.json` holds
user preferences, `.env` holds provider credentials and provider settings.

### Preferences — `config.json`

The language comes from the existing `language` key, the same one the TUI's Settings
screen writes. It currently defaults to `tr` when the key is absent. A consequence
worth stating: switching the TUI language switches dictation too. That is the intended
single-language behaviour.

Two keys are added, read through `audiocript.load_config`:

| Key | Default | Meaning |
|---|---|---|
| `dictation_hotkeys` | `{"toggle": "<cmd>+<alt>+d"}` | Hotkey combinations, by action |
| `dictation_max_seconds` | `300` | Upper bound on one recording |

`dictation_hotkeys` is an object rather than a single string so that a second binding —
a cancel key, for instance — is a new entry rather than a schema change. Only `toggle`
is read in this change; an unrecognised action name is reported at startup rather than
ignored, since a silently dead hotkey is indistinguishable from a missing permission.

Each combination is passed to `pynput`'s own syntax unchanged: modifiers in angle
brackets, joined by `+`. `keyboard.HotKey.parse` raises `ValueError` on anything it
does not understand (`cmd+alt+d` and `<zort>+d` both fail), so an unusable combination
is caught at startup with the syntax explained, never at the moment the user first
presses it.

**At least one modifier is required.** `parse` accepts a bare `d`, which would fire on
every `d` typed anywhere on the machine. A combination with no modifier is refused.

A `language` value outside `audiocript.LANGUAGES` is refused the same way. Falling back
to a default would transcribe with a model the user did not ask for and blame the
microphone for the result.

`dictation_max_seconds` exists because a forgotten second keypress otherwise records
until the disk fills. Reaching the bound stops the capture and processes it exactly as
a keypress would, and the notification says the limit was reached rather than
pretending the user stopped it.

These keys are edited in the file. Adding rows to the TUI's Settings screen would mean
changing `audiocript.py`, which this design keeps out of scope; it is a small, separate
change once the daemon works.

### Provider — `.env` or the environment

Read through `publish.env_value`, so an exported variable wins over `.env`:

| Variable | Required | Default |
|---|---|---|
| `OPENAI_API_KEY` | yes | — (shared with publishing) |
| `DICTATION_MODEL` | no | `gpt-4.1-mini` |

`DICTATION_MODEL` sits here beside the existing `OPENAI_MODEL` rather than in
`config.json`: publishing's default is chosen for documents as long as their input, and
punctuating two sentences should not pay for that.

## Correction Contract

A new `prompts/dictation_prompt.md`, written in Turkish to match the two existing
prompts, ending in a bracketed placeholder line so `publish.render_prompt` can inline
the transcript.

**One prompt serves both languages.** The rules it states are about punctuation,
fillers and stutters, which do not differ by language, and a model given Turkish
instructions corrects English text correctly. The configured language selects the
Whisper model, not the prompt. A second prompt would be two things to keep in step for
no gain.

The prompt instructs the model to add punctuation and capitalisation, drop meaningless
fillers, remove stutters and repetitions, and fix technical terms Whisper mishears. It
forbids changing word choice, reordering sentences, summarising, shortening, adding
anything, emitting markdown, headings or quotes, and writing commentary.

**The contract is also enforced in code.** A corrected result longer than twice the
transcript, or shorter than forty percent of it, is treated as a violated contract: the
raw transcript goes to the clipboard instead and the notification says the correction
was skipped. Length is a coarse signal, but a model that answers with an apology or an
essay fails it, and the alternative is that text reaching the clipboard silently.

## Status Feedback

The daemon never calls `osascript` directly. It talks to a status sink:

```
recording()          # capture started
processing()         # capture stopped, work started
done(text)           # clipboard holds text; a short preview is shown
failed(reason)       # nothing usable, with the reason
```

`NotifySink` implements it with `osascript display notification` plus a short
`afplay` cue on start and stop. Both were confirmed present and working on this
machine, including a `pbcopy`/`pbpaste` round trip that preserved Turkish characters
byte for byte.

A future `MenuBarSink` is a new implementation of the same four calls. The daemon does
not change to gain a menu bar.

The clipboard is likewise behind a small interface, so tests never shell out.

## Error Handling

| Situation | Behaviour |
|---|---|
| `OPENAI_API_KEY` absent | Refuse to start; print the reason; exit non-zero. |
| Input Monitoring not granted (`IS_TRUSTED` false) | Warn clearly, name the permission and where to grant it, and point at `--toggle` as the working alternative. |
| Event tap failed for another reason (listener not alive after `wait()`) | Same warning. |
| Microphone refused after every attempt | Notify; delete the temporary directory; return to IDLE. |
| Transcript empty | Notify that no speech was detected. **The clipboard is not touched** — silently overwriting what the user had is the worst outcome available. |
| Provider call fails or times out | **Put the raw transcript on the clipboard** and say it was not corrected, with the reason. The user has already spoken and is waiting; unpunctuated text beats nothing. This deliberately inverts publishing's rule, where half a document is worse than none. |
| Correction violates the length guard | Raw transcript on the clipboard; notification says the correction was skipped. |
| Toggle pressed while PROCESSING | Ignored, with a notification. |
| A daemon is already running (PID file) | The second copy explains and exits. |
| `--toggle` or `--stop` with no daemon running | Say so and exit non-zero, rather than exiting silently as if it worked. |
| `language` not in `audiocript.LANGUAGES` | Refuse to start; name the accepted values. |
| A hotkey combination `HotKey.parse` rejects | Refuse to start; show the offending value and the expected syntax. |
| A hotkey combination with no modifier | Refuse to start; explain that it would fire on every such keypress. |
| An unknown action name in `dictation_hotkeys` | Refuse to start; name the actions that are understood. |
| Recording reached `dictation_max_seconds` | Stop and process it; the notification says the limit was reached. |

No failure leaves the state machine outside IDLE, and no failure leaves a temporary
directory behind.

## CLI

```
./run.sh --dictate          # set up the venv if needed, then run the daemon in the foreground
python dictate.py --toggle  # signal the running daemon (the Shortcuts / skhd path)
python dictate.py --stop    # ask the running daemon to exit
```

## Acceptance Criteria

1. With the daemon running and the hotkey pressed twice around a spoken sentence, the
   clipboard holds that sentence, punctuated, in the configured language.
2. The text on the clipboard says what was spoken: no summary, no reordering, no added
   sentences.
3. The user learns from a notification that recording started, that work is in
   progress, and that the clipboard is ready.
4. With the provider unreachable, the clipboard still holds the raw transcript and the
   notification says the correction did not happen.
5. Speaking nothing leaves the clipboard's previous contents intact.
6. Pressing the hotkey during processing does not start a second recording.
7. Without Input Monitoring, the daemon says so on startup instead of appearing to work.
8. `--toggle` drives a full dictation with no hotkey involved.
9. No dictation leaves a file behind, in the recordings folder or in the temporary one.
10. A recording left running stops itself at the configured bound and the clipboard
    still receives what was said up to that point.
11. Changing `dictation_hotkeys.toggle` in `config.json` and restarting the daemon
    makes the new combination drive dictation and the old one do nothing.
12. An unusable hotkey or language value stops the daemon at startup with a message
    that names the offending value.
13. The TUI behaves exactly as before; `tests/run_all.py` still passes.

## Testing

`tests/test_dictate.py`, following the conventions already in `tests/`: a plain script
using `support.run`, discovered by `run_all.py`'s `test_*.py` glob, runnable on its own.

`tests/test_publish.py` already stands up a local `HTTPServer` as a fake OpenAI and
uses an `isolated` context manager that clears the publishing variables and repoints
`publish.ROOT`, so a developer's real `.env` can never be spent. Dictation tests reuse
both. No test reaches a real API, opens a microphone, or touches the real clipboard.

Each of these must fail if the line it protects is removed:

1. The state transitions, and that a toggle during PROCESSING is ignored.
2. A failing provider call still puts the **raw transcript** on the clipboard.
3. An empty transcript leaves the clipboard's previous contents unchanged.
4. A response that violates the length guard falls back to the raw transcript.
5. `publish.render_prompt` finds the dictation prompt's placeholder — a prompt file
   edited into a shape without one would otherwise append the transcript silently.
6. `dictation_config()` works with no GitHub variables set, where `publish.config()`
   returns `None`.
7. `IS_TRUSTED` false produces the startup warning.
8. The language is read from `config.json` and selects the matching Whisper model.
9. A completed dictation leaves no temporary directory.
10. A `language` value outside `LANGUAGES` is refused at startup.
11. Reaching the duration bound stops the capture and processes what was captured.
12. `dictation_hotkeys.toggle` from `config.json` is the combination actually
    registered, and its absence falls back to the documented default.
13. A combination `HotKey.parse` rejects, and a combination with no modifier, are both
    refused at startup rather than registered.

The `mutation-gate` skill is used to prove each of these fails when its subject is
removed, rather than assuming it.

## Unverified Assumptions

These are not yet established and must be settled early in implementation:

- **`gpt-4.1-mini` exists and is appropriate.** It is a sibling of publishing's
  default, but no request has been made against it. The first implementation step is
  one real call. If it is wrong, only the default value changes.
- **Input Monitoring can be granted to the virtualenv interpreter.** Unknown. The
  `--toggle` path exists because of this.
- **Two processes can hold the same input device.** Whether the TUI recording and a
  dictation can overlap is untested. No lock is being designed: `_start_mic` already
  handles a refusal by retrying and then failing cleanly, and the daemon holds the
  microphone only while recording. If overlap turns out to fail in practice, a lock is
  a later, separate change.
- **End-to-end latency.** The Turkish path is a `transformers` pipeline with no
  incremental hook, so the transcript is produced after the recording ends, and a
  provider round trip follows. The delay has not been measured. If it proves too long
  to feel like dictation, the model choice and the warm-up strategy are where to look.

## Documentation

README currently carries a "runs offline" badge, earned by local transcription.
Dictation sends every utterance to an AI provider. The README section for this feature
must state that plainly rather than let the badge cover it, in the same way the
publishing section already declares its two paid calls.
