"""The parts of dictation that hold no daemon threads and no global state:
configuration, the clipboard, the status sinks, the history log, the menu model and
the correction pipeline that decides what reaches the clipboard.

Kept apart from dictate.py's state machines and from menubar.py's AppKit layer so all
of it can be tested without audio hardware, a loaded model, a running daemon or a
menu bar — which is where most of this feature's rules live.

`History` starts no threads either, but it does serialise its own file access with a
lock: the worker appends while the main thread reads the menu.
"""
import json
import pathlib
import subprocess
import threading
from dataclasses import dataclass, field
from datetime import datetime

import audiocript as A
import publish

DEFAULT_MAX_SECONDS = 300
DEFAULT_MODEL = "gpt-4.1-mini"

# The daemon's two axes of state. `state` is one dictation's progress; `power` is
# whether the daemon is up at all, and means the model is loaded and a recording can
# start. They are separate because "recording" and "off" are not comparable: folding
# them into one enum would make every existing `state == IDLE` guard quietly start
# meaning "powered on and idle".
#
# They live here rather than in dictate.py, which owns both state machines, because
# the menu model below is pure and may not import dictate: the vocabulary both layers
# branch on belongs to the module underneath. dictate.py refers to them through this
# module and keeps no aliases of its own.
IDLE, RECORDING, PROCESSING = "idle", "recording", "processing"
POWER_OFF, POWER_LOADING, POWER_ON = "off", "loading", "on"

# Every value of each axis, so a caller — or a test — can cover all of them without
# naming them. A fourth value added to either tuple widens every scan that iterates
# it, which is the only way an added state does not go silently unhandled.
STATES = (IDLE, RECORDING, PROCESSING)
POWERS = (POWER_OFF, POWER_LOADING, POWER_ON)


class ConfigError(Exception):
    """Raised by resolve_config when a configured value cannot be used as given."""


@dataclass
class DictationConfig:
    language: str
    max_seconds: int
    model: str
    # repr=False: a traceback, a log line, or a careless print/f-string must not
    # be able to render (and so leak) the API key held here.
    openai_token: str = field(repr=False)
    openai_base: str


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

    max_seconds = _validated_max_seconds(cfg)

    openai_token = env("OPENAI_API_KEY")
    if not openai_token:
        raise ConfigError("OPENAI_API_KEY is required (set it in the environment "
                          "or in .env)")

    return DictationConfig(
        language=language,
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

    def done(self, text, note=""):
        """`text` is already on the clipboard, ready to paste. `note` says why it
        is not the corrected text — the provider failed, or the reply broke the
        length contract — and is empty when the correction went through, which is
        the only case where `text` is a corrected text."""

    def failed(self, reason):
        """The attempt produced no text; `reason` explains why, for the user."""

    def power_changed(self, power):
        """The daemon's power axis moved to `power` — one of POWER_OFF,
        POWER_LOADING, POWER_ON.

        The fifth moment exists because `enable()` returns before the model is
        loaded: the load runs on a thread of its own, so LOADING becoming ON is
        something only a sink call can tell a caller about. MenuBarSink renders it as
        an icon title; NotifySink says it out loud.

        Named `power_changed` rather than `power` so the parameter is not called
        `state`, which is the other axis's word."""


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

    def done(self, text, note=""):
        preview = text if len(text) <= PREVIEW_CHARS else text[:PREVIEW_CHARS] + "…"
        # The note is not truncated the way the preview is: a correction that was
        # refused has to be able to say so, or the guard is invisible to the only
        # person who can act on it — the user about to paste.
        self._report(f"Copied to clipboard: {preview}"
                     + (f" — {note}" if note else ""))

    def failed(self, reason):
        self._report(f"Dictation failed: {reason}")

    def power_changed(self, power):
        # No cue. The two moments that cue are the start and the stop of speaking,
        # where the user is not looking at the screen; a power change is something
        # they just asked for by clicking.
        #
        # .get rather than [power]: an unexpected value is worth reporting as itself,
        # where a KeyError here would raise on a status report — outside the
        # try/except in _report, which guards the notification, not this lookup.
        self._report(POWER_MESSAGES.get(power, f"Dictation daemon: {power}"))


# What each power value says to the user. English, like NotifySink's other four
# messages; the Turkish in this feature is in the menu titles and in the duration
# bound's notification, which dictate.py explains where it builds it.
POWER_MESSAGES = {POWER_OFF: "Dictation off",
                  POWER_LOADING: "Loading the model…",
                  POWER_ON: "Dictation ready"}


# =========================== The correction pipeline ===========================

CORRECTED = "corrected"        # the provider's text is on the clipboard
UNCORRECTED = "uncorrected"    # the provider failed; the raw transcript is on it
SKIPPED = "skipped"            # the reply broke the length contract; raw transcript
EMPTY = "empty"                # nothing was said; the clipboard was not touched

# All four, so a caller — or a test — can cover them without naming them. A fifth
# status added here widens every scan that iterates this, which is the only way an
# added status does not go silently unhandled by the history log or by a report.
DELIVERY_STATUSES = (CORRECTED, UNCORRECTED, SKIPPED, EMPTY)

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
        sink.done(transcript, detail)
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
        # A reply this short from the real provider never reaches here:
        # publish.openai_complete itself raises on an empty or whitespace reply,
        # which takes the UNCORRECTED path above instead. Only a
        # too-short-but-nonempty reply, or an empty one from a test's injected
        # `complete`, lands here.
        return raw(SKIPPED, f"the correction came back {ratio:.2f}x the length of "
                            "the dictation, so it was not used")
    clipboard.copy(corrected)
    sink.done(corrected)
    return Delivery(CORRECTED, corrected)


# ================================= The history =================================

# Beside the pid file the daemon already writes there.
HISTORY_PATH = pathlib.Path("~/.audiocript/dictations.jsonl").expanduser()

# How many entries the menu lists.
HISTORY_ITEMS = 10

# How much of the end of the log `recent` reads. The file is unbounded and the menu
# reads it every time it opens, so what is bounded is the read rather than the file.
# At the length a menu entry shows, this window holds several hundred dictations —
# far more than HISTORY_ITEMS ever asks for.
HISTORY_TAIL_BYTES = 256 * 1024


@dataclass
class HistoryEntry:
    """One delivered dictation, read back from the log."""
    at: str                    # local time with offset, seconds resolution
    status: str                # CORRECTED, UNCORRECTED or SKIPPED
    text: str                  # exactly what reached the clipboard


class History:
    """The log of what reached the clipboard: one JSON object per line, appended.

    The point is that a dictation outlives the clipboard. Every delivered text used to
    live exactly as long as it took the next one to replace it; here the newest ten
    are one click away and the whole log is a file the user can grep.

    Written with `ensure_ascii=False`: the primary language here is Turkish, and a log
    full of `\u011f` is a log nobody reads. Local time with an offset rather than UTC,
    for the same reason — it is read by the person who spoke it.

    `append` swallows its failures and `recent` does not, deliberately. By the time
    append runs, the text is already on the clipboard and the user has been told; a
    log that cannot be written must not cost them the dictation. But a `recent` that
    answered `[]` for an unreadable file would look exactly like "no dictations yet"
    in the menu, which is the one thing the menu must not say wrongly — so the layer
    that draws the menu decides what to show, and gets the error to decide with.

    `tail_bytes` is a parameter so a test can shrink the window and observe that the
    older entries are genuinely not read, without timing anything.
    """

    def __init__(self, path=HISTORY_PATH, tail_bytes=HISTORY_TAIL_BYTES):
        self._path = pathlib.Path(path)
        self._tail_bytes = tail_bytes
        # The worker thread appends while the main thread reads the menu, so both
        # ends of the file go through one lock. Without it two appends can land
        # inside each other, and the line that destroys is a dictation the user has
        # already spoken.
        self._lock = threading.Lock()

    @property
    def path(self):
        """Where the log is — for the menu item that reveals it in Finder."""
        return self._path

    def append(self, text, status):
        """Add one entry. Never raises: see the class docstring."""
        line = json.dumps(
            {"at": datetime.now().astimezone().isoformat(timespec="seconds"),
             "status": status, "text": text}, ensure_ascii=False) + "\n"
        try:
            with self._lock:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with open(self._path, "a", encoding="utf-8") as f:
                    f.write(line)
        except Exception as e:
            A._log_problem(f"a dictation could not be added to {self._path}", e)

    def recent(self, limit=HISTORY_ITEMS):
        """The newest `limit` entries, newest first; `[]` when there is no log yet.

        Anything else that goes wrong is raised, not swallowed. A line that will not
        parse costs that line and nothing more — a file a user can open can hold
        anything, and the last line of a file being appended to is routinely half
        written."""
        try:
            with self._lock:
                start = max(0, self._path.stat().st_size - self._tail_bytes)
                with open(self._path, "rb") as f:
                    f.seek(start)
                    block = f.read()
        except FileNotFoundError:
            return []
        lines = block.split(b"\n")
        if start:
            del lines[0]              # the window opened in the middle of a line
        entries = []
        for line in reversed(lines):
            if len(entries) >= limit:
                break
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                entries.append(HistoryEntry(at=entry["at"], status=entry["status"],
                                            text=entry["text"]))
            except (ValueError, KeyError, TypeError):
                continue
        return entries


# =============================== The menu model ===============================
#
# What the two menus contain, as data. This is the whole of the menu bar's
# decision-making: menubar.py turns these lists into NSMenu objects and decides
# nothing. The split is not tidiness — there is no headless AppKit to test against,
# so anything that branches on power or state has to live on this side of the line to
# be testable at all.

# How much of a dictation a menu item shows. The user asked for 250 characters. How
# wide macOS actually draws a title of that length, and where it puts its own
# ellipsis, is not ours to control; the full text is the item's tooltip and is what a
# click copies, so nothing is lost to that.
MENU_PREVIEW_CHARS = 250

# Every action a menu item can carry. A closed set because menubar.py dispatches on
# these strings: a typo would otherwise produce an item that looks live and silently
# does nothing, and a test scans this against what the builders below actually offer.
MENU_ACTIONS = ("power_on", "power_off", "record_toggle", "copy", "reveal", "quit")

# One icon carries both axes, because one icon is all there is. While the daemon is not
# up its power is the whole story — a microphone glyph over a daemon that cannot record
# is a lie — and once it is up, the glyph is the dictation's stage.
#
# Glyphs rather than colours: the menu bar is monochrome in both appearances, so a
# hollow/dotted distinction survives where a red/grey one would not.
POWER_ICONS = {POWER_OFF: "◯", POWER_LOADING: "◌"}
STAGE_ICONS = {IDLE: "🎙", RECORDING: "🔴", PROCESSING: "⏳"}

# Said by both menus while the model loads, in the same words on purpose: one name per
# thing the user is waiting for.
LOADING_TITLE = "Model yükleniyor…"


@dataclass
class MenuItem:
    """One row of a menu. `payload` is the full text for a `copy` row and nothing for
    the rest — the title is cut for reading, and a click must put back everything
    that was on the clipboard."""
    title: str
    action: str = None
    enabled: bool = True
    payload: str = None
    separator: bool = False


def _separator():
    return MenuItem("", separator=True)


def menu_preview(text, limit=MENU_PREVIEW_CHARS):
    """`text` as one line, cut to `limit` characters with an ellipsis if it was cut.

    Whitespace is collapsed because a menu item is a single line: a newline in a title
    renders as a box or swallows the rest of the row, and dictated text has newlines
    in it as a matter of course."""
    one_line = " ".join(text.split())
    if len(one_line) <= limit:
        return one_line
    return one_line[:limit] + "…"


def icon(power, state):
    """The status item's title: one glyph for both axes.

    An unexpected value draws the off glyph rather than raising. This is called from a
    menu callback, where an exception is a wedged menu bar."""
    if power != POWER_ON:
        return POWER_ICONS.get(power, POWER_ICONS[POWER_OFF])
    return STAGE_ICONS.get(state, STAGE_ICONS[IDLE])


def _dictation_row(power, state):
    """The row that starts and stops a dictation — the reason the menu is opened, so it
    goes first."""
    if power == POWER_LOADING:
        return MenuItem(LOADING_TITLE, enabled=False)
    if power != POWER_ON:
        return MenuItem("Daemon kapalı", enabled=False)
    if state == IDLE:
        return MenuItem("Kaydı başlat", "record_toggle")
    if state == RECORDING:
        return MenuItem("Kaydı bitir", "record_toggle")
    return MenuItem("İşleniyor…", enabled=False)


def _power_row(power, state):
    """The row that brings the daemon up and down.

    Stopping is refused while a dictation is running. The invariant itself is in
    dictate.Daemon.disable — this is the menu saying so in advance rather than offering
    a row whose click would be turned down.

    Stopping is offered while the model is still loading, and on purpose: disable()
    accepts it (the load's own stale path drops what it produced), so this is the only
    way to call off a load started by mistake — a wait of 11.3s otherwise. It also
    keeps this row from repeating what the row above it already says while loading."""
    if power == POWER_OFF:
        return MenuItem("Daemon'ı başlat", "power_on")
    return MenuItem("Daemon'ı durdur", "power_off", enabled=state == IDLE)


def _history_rows(entries):
    """The last dictations, or one row saying why there are none to show.

    `entries` are HistoryEntry objects, newest first — the order History.recent returns
    and the order the menu shows; `None` means the log could not be read. An unreadable
    log is not an empty one and the menu must not say the one when the other is true:
    History.recent raises rather than answering [] for exactly this, because "Henüz
    dictation yok" over a file that could not be read would send the user looking for a
    bug in the recording."""
    if entries is None:
        return [MenuItem("Geçmiş okunamadı", enabled=False)]
    if not entries:
        return [MenuItem("Henüz dictation yok", enabled=False)]
    return [MenuItem(menu_preview(e.text), "copy", payload=e.text) for e in entries]


def menu(power, state, entries):
    """The whole menu, in a fixed order.

    Fixed rather than rearranged for the current state: this menu is opened many times a
    day, and a row that moves depending on what the daemon is doing costs more than the
    dead row it would save. The dictation row is first because it is why the menu is
    opened; the daemon row sits under it because it is clicked twice a day.

    Quit follows the same rule as stopping the daemon — a dictation the user has already
    spoken is not thrown away by a menu click.

    Takes `entries` rather than a History so it stays pure: reading the file is the
    caller's job, and so is deciding what an unreadable one means."""
    return [_dictation_row(power, state),
            _power_row(power, state),
            _separator(),
            *_history_rows(entries),
            _separator(),
            MenuItem("Geçmiş dosyasını aç", "reveal"),
            MenuItem("Çıkış", "quit", enabled=state == IDLE)]
