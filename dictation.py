"""The parts of dictation holding no threads and no global state: configuration,
the clipboard, the status sinks, and the correction pipeline that decides what
reaches the clipboard. Kept apart from dictate.py's state machine and hotkey
listener so all of it can be tested without audio hardware or a running daemon.
"""
import pathlib
import subprocess
from dataclasses import dataclass, field

from pynput.keyboard import HotKey, Key

import audiocript as A
import publish

HOTKEY_ACTIONS = ("toggle",)
DEFAULT_HOTKEYS = {"toggle": "<cmd>+<alt>+d"}
DEFAULT_MAX_SECONDS = 300
DEFAULT_MODEL = "gpt-4.1-mini"


class ConfigError(Exception):
    """Raised by resolve_config when a configured value cannot be used as given."""


@dataclass
class DictationConfig:
    language: str
    hotkeys: dict
    max_seconds: int
    model: str
    # repr=False: a traceback, a log line, or a careless print/f-string must not
    # be able to render (and so leak) the API key held here.
    openai_token: str = field(repr=False)
    openai_base: str


def _validated_hotkeys(cfg):
    """DEFAULT_HOTKEYS with any configured combinations laid over it, so a partial
    `dictation_hotkeys` object keeps the actions it did not mention."""
    hotkeys = {**DEFAULT_HOTKEYS, **(cfg.get("dictation_hotkeys") or {})}
    for action, combination in hotkeys.items():
        if action not in HOTKEY_ACTIONS:
            raise ConfigError(
                f"unknown hotkey action {action!r}: expected one of {HOTKEY_ACTIONS}")
        try:
            parsed = HotKey.parse(combination)
        except ValueError:
            raise ConfigError(
                f"hotkey {combination!r} for the {action!r} action is not a "
                "valid key combination") from None
        if not any(isinstance(key, Key) for key in parsed):
            raise ConfigError(
                f"hotkey {combination!r} for the {action!r} action needs a "
                "modifier such as <cmd>, <alt> or <shift> — otherwise it would "
                "fire on every matching key typed anywhere")
    return hotkeys


def _validated_max_seconds(cfg):
    value = cfg.get("dictation_max_seconds", DEFAULT_MAX_SECONDS)
    if not isinstance(value, int) or value <= 0:
        raise ConfigError(
            f"dictation_max_seconds must be a positive integer, got {value!r}")
    return value


def resolve_config(cfg, env=None):
    """Turn the dict from audiocript.load_config() plus the environment into a
    validated DictationConfig, or raise ConfigError naming what is wrong.

    `env` defaults to publish.env_value; tests pass a fake one so nothing here
    ever touches a developer's real .env."""
    env = env or publish.env_value

    language = cfg.get("language", "tr")
    if language not in A.LANGUAGES:
        raise ConfigError(
            f"unknown language {language!r}: expected one of {sorted(A.LANGUAGES)}")

    hotkeys = _validated_hotkeys(cfg)
    max_seconds = _validated_max_seconds(cfg)

    openai_token = env("OPENAI_API_KEY")
    if not openai_token:
        raise ConfigError("OPENAI_API_KEY is required (set it in the environment "
                          "or in .env)")

    return DictationConfig(
        language=language,
        hotkeys=hotkeys,
        max_seconds=max_seconds,
        model=env("DICTATION_MODEL", DEFAULT_MODEL),
        openai_token=openai_token,
        openai_base=env("OPENAI_API_BASE", "https://api.openai.com"))


# A message this long or shorter is shown in full on `done`; anything longer is
# truncated with an ellipsis so a notification never grows to the size of the
# dictation itself.
PREVIEW_CHARS = 60

# Confirmed present on the target machine, same as afplay/osascript themselves.
DEFAULT_SOUND = "/System/Library/Sounds/Pop.aiff"

# A notification or a sound cue that has not happened in this long has already
# failed at its purpose; bounding it keeps a stuck osascript/afplay from
# freezing the caller — Task 4's state machine, on the hotkey thread.
NOTIFY_TIMEOUT_SECONDS = 2


class Clipboard:
    """Puts text on the system clipboard, for the user to paste into any app."""

    def copy(self, text):
        """Pipe `text` through pbcopy as UTF-8. Raises subprocess.CalledProcessError
        if pbcopy exits non-zero."""
        subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True)


class StatusSink:
    """The four moments dictate.py's state machine reports, and nothing else it
    is allowed to say about output. A documented contract, not an abstract base
    class: NotifySink is today's only implementation, and the menu bar
    indicator the spec defers becomes a second one rather than a change to the
    daemon that calls these four methods."""

    def recording(self):
        """The hotkey fired; the microphone has started capturing."""

    def processing(self):
        """The hotkey fired again: recording stopped, transcription and
        correction are running."""

    def done(self, text):
        """`text` is corrected and already on the clipboard, ready to paste."""

    def failed(self, reason):
        """The attempt produced no text; `reason` explains why, for the user."""


def _osascript_notify(message):
    """Show `message` as a macOS notification titled Audiocript.

    `message` is passed as an argv item to an `on run argv` handler rather than
    interpolated into the script text, so nothing about it needs AppleScript's
    own escaping rules — a prior version built the script with json.dumps(),
    which escapes non-ASCII characters (Turkish letters included) as \\uXXXX
    sequences AppleScript does not understand, and reproduces quotes/backslashes
    /newlines by coincidence rather than by contract."""
    subprocess.run(
        ["osascript",
         "-e", "on run argv",
         "-e", 'display notification (item 1 of argv) with title "Audiocript"',
         "-e", "end run",
         "--", message],
        check=True, timeout=NOTIFY_TIMEOUT_SECONDS)


def _afplay_sound():
    """Play a short cue, so a stage change can be felt without looking at
    anything."""
    subprocess.run(["afplay", DEFAULT_SOUND], check=True, timeout=NOTIFY_TIMEOUT_SECONDS)


class NotifySink(StatusSink):
    """StatusSink backed by a macOS notification and a sound cue.

    `notify` and `sound` are injected so tests substitute collectors instead of
    shelling out to osascript/afplay; unset, they default to helpers that do
    exactly that. Every call to either is wrapped: the status report is the
    least important thing happening, and losing it must never cost a dictation
    whose text is already on the clipboard. A swallowed exception is sent to
    audiocript._log_problem so it is not lost entirely."""

    def __init__(self, notify=None, sound=None):
        self._notify = notify or _osascript_notify
        self._sound = sound or _afplay_sound

    def _report(self, message):
        try:
            self._notify(message)
        except Exception as e:
            A._log_problem(f"dictation notification failed: {message!r}", e)

    def _cue(self):
        try:
            self._sound()
        except Exception as e:
            A._log_problem("dictation sound cue failed", e)

    def recording(self):
        self._report("Recording…")
        self._cue()

    def processing(self):
        self._report("Processing…")
        self._cue()

    def done(self, text):
        preview = text if len(text) <= PREVIEW_CHARS else text[:PREVIEW_CHARS] + "…"
        self._report(f"Copied to clipboard: {preview}")

    def failed(self, reason):
        self._report(f"Dictation failed: {reason}")


# =========================== The correction pipeline ===========================

CORRECTED = "corrected"        # the provider's text is on the clipboard
UNCORRECTED = "uncorrected"    # the provider failed; the raw transcript is on it
SKIPPED = "skipped"            # the reply broke the length contract; raw transcript
EMPTY = "empty"                # nothing was said; the clipboard was not touched

# The prompt asks for punctuation, dropped fillers and nothing else, so a faithful
# reply is about as long as the transcript. A reply well outside this band is a
# rewrite, a summary, or a commentary about the text — the one failure the user
# cannot spot before pasting, because it reads perfectly well. Measured in
# characters rather than words: dictations are short, so a word ratio moves in
# steps of 1/N and a single dropped filler can look like a breach, and a
# character count also sees a prepended preamble or a code fence that adds few
# words. Punctuation is a few percent of the length, nowhere near these bounds.
MAX_GROWTH = 2.0               # a reply longer than this multiple of the input is refused
MIN_SHRINK = 0.4               # a reply shorter than this fraction of the input is refused

# Anchored on this file, never on the working directory: the daemon runs from
# wherever the user launched it.
PROMPT_PATH = pathlib.Path(__file__).resolve().parent / "prompts" / "dictation_prompt.md"


@dataclass
class Delivery:
    """What deliver() did, for the caller to report and for a test to assert on."""
    status: str                # one of the four constants above
    text: str                  # what reached the clipboard; "" for EMPTY
    detail: str = ""           # why, when it was not a plain correction


def deliver(transcript, config, clipboard, sink, complete=None):
    """Correct `transcript` with the provider, put the result on the clipboard, and
    return a Delivery saying what happened.

    Every path but EMPTY copies exactly once: the user has already spoken and is
    waiting, so a failed correction still delivers the raw transcript rather than
    nothing. EMPTY copies nothing at all — silently replacing whatever the user
    already had on the clipboard is worse than doing nothing.

    `complete` is a callable(prompt, cfg_dict) -> str, defaulting to
    publish.openai_complete, so the daemon's tests need no socket."""
    complete = complete or publish.openai_complete
    transcript = (transcript or "").strip()
    if not transcript:
        reason = "no speech was detected"
        sink.failed(reason)
        return Delivery(EMPTY, "", reason)

    def raw(status, detail):
        """The transcript itself on the clipboard, unpunctuated but pasteable."""
        clipboard.copy(transcript)
        sink.done(transcript)
        return Delivery(status, transcript, detail)

    # Built here as a local and never logged, returned or put in a message: it
    # carries the API key, and DictationConfig hides it from repr() for a reason.
    cfg_dict = {"openai_base": config.openai_base,
                "openai_token": config.openai_token,
                "model": config.model}
    try:
        corrected = complete(publish.render_prompt(PROMPT_PATH, transcript),
                             cfg_dict).strip()
    except Exception as e:
        A._log_problem("dictation correction failed", e)
        return raw(UNCORRECTED, f"the correction failed: {e}")

    ratio = len(corrected) / len(transcript)
    if not MIN_SHRINK <= ratio <= MAX_GROWTH:
        # An empty reply lands here too, at 0.00x.
        return raw(SKIPPED, f"the correction came back {ratio:.2f}x the length of "
                            "the dictation, so it was not used")
    clipboard.copy(corrected)
    sink.done(corrected)
    return Delivery(CORRECTED, corrected)
