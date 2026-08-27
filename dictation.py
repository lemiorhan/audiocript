"""The parts of dictation holding no threads and no global state: configuration,
the clipboard, and the status sinks today, and — as a later task adds it — the
correction pipeline. Kept apart from dictate.py's state machine and hotkey
listener so all of it can be tested without audio hardware or a running daemon.
"""
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
