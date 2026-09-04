"""Stopping a recording must be something the user asked for, not something they
brushed against.

A recording was lost this way: mid-sentence the space bar got pressed, the app read
it as "stop", and the capture went straight into transcription with no way back.
Space, [r] and [q] were all wired to the same irreversible action.

These tests drive `_tui_handle_key` with stand-in recorders, so what a keystroke does
to a live recording is something we can assert on without opening a microphone.
"""
from rich.console import Console

from support import run, workdir       # noqa: F401
import audiocript as A


def plain_render(state):
    """The whole TUI as the text a user would see, for asserting on what is shown."""
    console = Console(width=100, height=40, record=True, legacy_windows=False)
    console.print(A._tui_render(state))
    return console.export_text()


class FakeLive:
    def update(self, _renderable):
        pass


class FakeRecorder:
    """A recorder that remembers whether anything stopped it."""

    def __init__(self):
        self.stopped = False
        self.meter_name = "🎤 Fake mic"

    def level(self):
        return 0.4

    def manifest(self):
        return {"kind": "mic", "file": "mic.raw", "rate": 48000,
                "channels": 1, "dtype": "int16"}

    def stop(self):
        self.stopped = True


class StubWorker:
    """Replaces `_stop_worker` so confirming does not run the real pipeline."""

    def __init__(self):
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1


def recording_state(home):
    """A _TuiState mid-recording, as `_start_recording` leaves it."""
    cfg = {"base_path": str(home / "recordings"), "language": "tr",
           "open_app": None, "capture_system_audio": False, "diarize": False}
    state = A._TuiState(cfg)
    project = home / "recordings" / "2026-08-18_09-00-00"
    project.mkdir(parents=True, exist_ok=True)
    state.project_dir = project
    state.pending_name = "stand-up"
    state.recorders = [FakeRecorder()]
    state.rec_start = A.time.monotonic() - 42
    state.mode = "recording"
    return state


def press(state, key, live=None):
    """One keystroke through the real dispatcher, with the worker stubbed out."""
    worker = StubWorker()
    saved = A._stop_worker
    A._stop_worker = worker
    try:
        assert A._tui_handle_key(key, state, live or FakeLive()) is True, (
            f"{key!r} quit the app")
    finally:
        A._stop_worker = saved
    return worker


def test_space_no_longer_stops_the_recording():
    """The keystroke that cost a recording. It must do nothing at all."""
    with workdir("stop-space") as home:
        state = recording_state(home)
        worker = press(state, " ")
        print(f"  mode after space: {state.mode!r}")
        assert state.mode == "recording", (
            f"space took the recording to {state.mode!r}")
        assert worker.calls == 0, "space started the stop worker"
        assert not state.recorders[0].stopped, "space stopped the recorder"


def test_stray_letters_no_longer_stop_the_recording():
    """[r] sat next to [q] on the same irreversible action, and every other letter
    has to be inert too — a recording is the one screen where a typo is expensive."""
    with workdir("stop-letters") as home:
        for key in ("r", "R", "t", "d", "ENTER", "ESC"):
            state = recording_state(home)
            worker = press(state, key)
            assert state.mode == "recording", f"{key!r} left recording: {state.mode!r}"
            assert worker.calls == 0, f"{key!r} started the stop worker"
        print("  r R t d Enter Esc: all inert")


def test_q_asks_before_it_stops():
    """[q] is the stop key, but it must ask first and keep capturing while it waits."""
    with workdir("stop-ask") as home:
        state = recording_state(home)
        worker = press(state, "q")
        print(f"  mode after q: {state.mode!r}")
        assert state.mode == "confirm_stop", (
            f"q stopped without asking (mode {state.mode!r})")
        assert worker.calls == 0, "q started the stop worker before the confirmation"
        assert not state.recorders[0].stopped, (
            "the recorder was stopped while the confirmation was still on screen")


def test_the_question_shows_the_recording_is_still_running():
    """Otherwise the confirmation reads as "already stopped", and the honest answer
    to "did I lose the last ten seconds?" has to be visible on the screen itself."""
    with workdir("stop-screen") as home:
        state = recording_state(home)
        press(state, "q")
        screen = plain_render(state)
        assert "REC" in screen, "the confirmation hides that capture continues:\n" + screen
        assert "00:42" in screen, "the elapsed time is not on the confirmation:\n" + screen
        assert "Fake mic" in screen, "the level meters are gone:\n" + screen
        low = screen.lower()
        assert "still recording" in low, (
            "nothing says the recording is still running:\n" + screen)
        assert "[y]" in low or " y " in low, "the confirming key is not shown:\n" + screen


def test_confirming_stops_and_transcribes():
    """Confirming hands the work to a job and gives the menu straight back. It used to
    switch to a transcription screen that had no key handler at all, so nothing could
    be done — least of all starting the next recording — until it finished."""
    with workdir("stop-yes") as home:
        state = recording_state(home)
        project = state.project_dir
        press(state, "q")
        worker = press(state, "y")
        jobs = A._jobs_snapshot(state)
        print(f"  mode after y: {state.mode!r}, worker calls: {worker.calls}, "
              f"jobs: {jobs}")
        assert state.mode == "menu", (
            f"y left the app on {state.mode!r} instead of handing the menu back")
        assert worker.calls == 1, "the stop worker never ran"
        assert len(jobs) == 1 and jobs[0].kind == "transcribe", (
            f"the work was not registered as a job: {jobs}")
        assert jobs[0].project_dir == project, "the job points at the wrong recording"
        assert state.recorders == [], (
            "the recorders were left on the state for the worker to re-read")


def test_declining_goes_back_to_the_recording():
    """Any key other than the confirming one keeps the recording — including a second
    [q], so a double tap can never stop what one tap only asked about."""
    with workdir("stop-no") as home:
        for key in ("n", "N", "ESC", "q", " "):
            state = recording_state(home)
            press(state, "q")
            worker = press(state, key)
            assert state.mode == "recording", (
                f"{key!r} did not return to the recording (mode {state.mode!r})")
            assert worker.calls == 0, f"{key!r} stopped the recording anyway"
            assert not state.recorders[0].stopped, f"{key!r} stopped the recorder"
        print("  n N Esc q space: all keep recording")


def test_the_footer_tells_the_user_what_q_does():
    with workdir("stop-footer") as home:
        state = recording_state(home)
        screen = plain_render(state).lower()
        assert "q" in screen and "stop" in screen, (
            "the recording screen does not name the stop key:\n" + screen)
        press(state, "q")
        screen = plain_render(state).lower()
        assert "keep recording" in screen, (
            "the confirmation does not offer a way back:\n" + screen)
        print("  both footers name their keys")


if __name__ == "__main__":
    run(["test_space_no_longer_stops_the_recording",
         "test_stray_letters_no_longer_stop_the_recording",
         "test_q_asks_before_it_stops",
         "test_the_question_shows_the_recording_is_still_running",
         "test_confirming_stops_and_transcribes",
         "test_declining_goes_back_to_the_recording",
         "test_the_footer_tells_the_user_what_q_does"], globals())
