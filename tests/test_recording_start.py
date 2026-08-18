"""Starting a recording must survive a microphone that is briefly unavailable.

Eight folders on disk showed what the app did instead: `meta.json` plus an empty
`mic.raw` and nothing else — CoreAudio refused the stream, `_start_recording`
returned to the menu, and the reason went to a status line that a running publish
job was covering. From the user's seat the app had simply ignored the keypress.

These tests drive `_start_recording` with a stand-in recorder, so a failing mic
open is something we can ask for rather than wait for.
"""
import json

from rich.console import Console

from support import REPO_ROOT, run, workdir       # noqa: F401  (REPO_ROOT via support)
import audiocript as A


def plain_render(state):
    """The whole TUI as the text a user would see, for asserting on what is shown."""
    console = Console(width=100, height=40, record=True, legacy_windows=False)
    console.print(A._tui_render(state))
    return console.export_text()


class FakeLive:
    """The Live object `_start_recording` announces through."""

    def __init__(self):
        self.frames = 0

    def update(self, _renderable):
        self.frames += 1


def fake_state(home, **over):
    """A _TuiState whose recordings live under `home`, with no real devices in play."""
    cfg = {"base_path": str(home / "recordings"), "language": "tr",
           "open_app": None, "capture_system_audio": False, "diarize": False}
    state = A._TuiState(cfg)
    state.mic_index = 7                       # never used: DeviceRecorder is stubbed
    state.pending_name = "pm1"
    for k, v in over.items():
        setattr(state, k, v)
    return state


class FlakyMic:
    """Stands in for DeviceRecorder: refuses to start `fails` times, then works.

    `leaves_junk` reproduces what the real recorder did on a failed open — the raw
    file created, then the stream refused — so cleanup is tested against the mess
    an interrupted open can actually leave, not just the tidy case."""

    attempts = 0
    fails = 0
    leaves_junk = False

    def __init__(self, index):
        self.index = index
        self.rate, self.channels = 48000, 1
        self.label = "Fake mic (48000 Hz)"
        self.meter_name = "🎤 Fake mic"
        self._raw_path = None

    @classmethod
    def reset(cls, fails=0, leaves_junk=False):
        cls.attempts, cls.fails, cls.leaves_junk = 0, fails, leaves_junk

    def start(self, raw_path):
        type(self).attempts += 1
        if type(self).attempts <= type(self).fails:
            if type(self).leaves_junk:
                open(raw_path, "wb").close()
            raise RuntimeError("Error opening InputStream: Device unavailable [-9986]")
        self._raw_path = str(raw_path)
        open(raw_path, "wb").close()

    def manifest(self):
        return {"kind": "mic", "file": "mic.raw", "rate": self.rate,
                "channels": self.channels, "dtype": "int16"}

    def level(self):
        return 0.0

    def stop(self):
        pass


class _Stub:
    """Swap module attributes for the duration of a test."""

    def __init__(self, **attrs):
        self.attrs = attrs
        self.saved = {}

    def __enter__(self):
        for k, v in self.attrs.items():
            self.saved[k] = getattr(A, k)
            setattr(A, k, v)
        return self

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            setattr(A, k, v)


def recording_dirs(home):
    base = home / "recordings"
    return sorted(d.name for d in base.iterdir() if d.is_dir()) if base.exists() else []


def test_a_transient_mic_refusal_still_starts_the_recording():
    """The tap gets ten seconds of patience; the mic got none. One refusal must not
    cost the user the recording they just named."""
    with workdir("start-retry") as home:
        FlakyMic.reset(fails=1)
        state, live = fake_state(home), FakeLive()
        with _Stub(DeviceRecorder=FlakyMic, device_name=lambda i: "Fake mic"):
            A._start_recording(state, live)
        print(f"  open attempts: {FlakyMic.attempts}, mode: {state.mode!r}")
        assert FlakyMic.attempts >= 2, "the mic open was never retried"
        assert state.mode == "recording", (
            f"one refusal dropped the user back to {state.mode!r}: {state.status!r}")
        assert state.recorders, "no recorder was kept on the state"


def test_a_recording_that_never_started_leaves_nothing_on_disk():
    """Eight of these folders were sitting in the recordings directories: named,
    empty, and unremovable, because meta.json was written before the mic opened."""
    with workdir("start-junk") as home:
        FlakyMic.reset(fails=99, leaves_junk=True)
        state, live = fake_state(home), FakeLive()
        with _Stub(DeviceRecorder=FlakyMic, device_name=lambda i: "Fake mic"):
            A._start_recording(state, live)
        left = recording_dirs(home)
        print(f"  folders left behind: {left}")
        assert state.mode != "recording", "the recording claims to have started"
        assert left == [], f"a failed start left {left} behind"


def test_the_reason_is_shown_where_a_background_job_cannot_cover_it():
    """The footer prints `state.bg_msg or state.status`, and publishing owns bg_msg
    for minutes after each recording — which is exactly when the user starts the
    next one. The failure has to reach the screen anyway."""
    with workdir("start-loud") as home:
        FlakyMic.reset(fails=99)
        state, live = fake_state(home), FakeLive()
        state.bg_msg = "Editing transcript “pm0”…"      # a publish job is running
        with _Stub(DeviceRecorder=FlakyMic, device_name=lambda i: "Fake mic"):
            A._start_recording(state, live)
        screen = plain_render(state)
        print(f"  mode after failure: {state.mode!r}")
        assert "Device unavailable" in screen, (
            "the microphone error never reached the screen:\n" + screen)
        assert state.mode != "menu", (
            "dropped straight to the menu, where the error is only a status line "
            "that bg_msg is already covering")


def test_the_typed_name_survives_a_failed_start():
    """So retrying is one keypress, not the whole naming dance again."""
    with workdir("start-name") as home:
        FlakyMic.reset(fails=99)
        state, live = fake_state(home), FakeLive()
        with _Stub(DeviceRecorder=FlakyMic, device_name=lambda i: "Fake mic"):
            A._start_recording(state, live)
        assert state.pending_name == "pm1", f"lost the name: {state.pending_name!r}"
        screen = plain_render(state)
        assert "pm1" in screen, "the kept name is not shown:\n" + screen

        FlakyMic.reset(fails=0)                          # the mic comes back
        with _Stub(DeviceRecorder=FlakyMic, device_name=lambda i: "Fake mic"):
            assert A._tui_handle_key("ENTER", state, live) is True
        assert state.mode == "recording", (
            f"Enter did not retry the start (mode {state.mode!r})")
        meta = json.loads((state.project_dir / "meta.json").read_text())
        assert meta.get("name") == "pm1", f"the retry lost the name: {meta}"


def test_the_failure_is_written_to_a_log():
    """The status line is volatile, so an intermittent failure left no trace at all
    and nothing to diagnose it from afterwards."""
    with workdir("start-log") as home:
        FlakyMic.reset(fails=99)
        state, live = fake_state(home), FakeLive()
        A.LOG_PATH.unlink(missing_ok=True)
        try:
            with _Stub(DeviceRecorder=FlakyMic, device_name=lambda i: "Fake mic"):
                A._start_recording(state, live)
            assert A.LOG_PATH.exists(), f"nothing was logged to {A.LOG_PATH}"
            text = A.LOG_PATH.read_text(encoding="utf-8")
            print(f"  logged: {text.strip().splitlines()[0][:100]}")
            assert "Device unavailable" in text, f"the error text is missing:\n{text}"
            assert "mic" in text.lower(), f"the log does not say what failed:\n{text}"
        finally:
            A.LOG_PATH.unlink(missing_ok=True)


def test_a_refused_open_leaves_no_raw_file_and_no_open_handle():
    """The real recorder opened mic.raw before the stream, so a refused open left an
    empty file (which is what defeated the folder cleanup) and a leaked handle.

    Uses a real device with an impossible channel count so the failure comes from
    CoreAudio itself rather than from a stand-in."""
    devices = A.list_input_devices()
    if not devices:
        print("  no input device on this machine — skipped")
        return
    with workdir("start-raw") as home:
        rec = A.DeviceRecorder(devices[0][0])
        rec.channels = 4096                      # no device offers this
        raw = home / "mic.raw"
        try:
            rec.start(raw)
        except Exception as e:
            print(f"  refused as expected: {type(e).__name__}")
        else:
            rec.stop()
            print("  the device accepted 4096 channels — skipped")
            return
        assert not raw.exists(), "a refused open left an empty mic.raw behind"
        assert rec._raw is None, "a refused open leaked the raw file handle"


if __name__ == "__main__":
    run(["test_a_transient_mic_refusal_still_starts_the_recording",
         "test_a_recording_that_never_started_leaves_nothing_on_disk",
         "test_the_reason_is_shown_where_a_background_job_cannot_cover_it",
         "test_the_typed_name_survives_a_failed_start",
         "test_the_failure_is_written_to_a_log",
         "test_a_refused_open_leaves_no_raw_file_and_no_open_handle"], globals())
