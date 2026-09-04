"""What the user asked for: start the next recording while the last one is still
transcribing, and let several jobs run at once.

The old shape made that impossible in two ways at once. `state.mode` meant both
"which screen" and "what is running", so a transcription put the app on a screen with
no key handler and every keystroke was dropped until it finished. And the in-flight
work lived in one set of fields, so a second job would have overwritten the first's
progress even if it could have been started.

These drive the real dispatcher, so what a keystroke does is what these assert.
"""
import threading
import time

from rich.console import Console

from support import A, run, workdir

TIMEOUT = 30


def plain_render(state):
    console = Console(width=100, height=40, record=True, legacy_windows=False)
    console.print(A._tui_render(state))
    return console.export_text()


class FakeLive:
    def update(self, _renderable):
        pass


class FakeRecorder:
    KIND = "mic"

    def __init__(self, raw_name="mic.raw"):
        self.raw_name = raw_name
        self.stopped = False

    def level(self):
        return 0.0

    def manifest(self):
        return {"kind": "mic", "file": self.raw_name, "rate": 48000,
                "channels": 1, "dtype": "int16"}

    def stop(self):
        self.stopped = True


def a_state(home, **over):
    cfg = {"base_path": str(home / "recordings"), "language": "tr",
           "open_app": None, "capture_system_audio": False, "diarize": False}
    state = A._TuiState(cfg)
    state.mic_index = 7
    for k, v in over.items():
        setattr(state, k, v)
    return state


def recording_state(home):
    """Mid-recording, as _start_recording leaves it."""
    state = a_state(home)
    project = home / "recordings" / "2026-08-18_09-00-00"
    project.mkdir(parents=True, exist_ok=True)
    state.project_dir = project
    state.pending_name = "stand-up"
    state.recorders = [FakeRecorder()]
    state.rec_start = A.time.monotonic() - 42
    state.mode = "recording"
    return state


def a_recording_on_disk(base, stamp, text="hello", name="rec"):
    """A finished recording the menu will list."""
    d = base / stamp
    d.mkdir(parents=True, exist_ok=True)
    (d / "audio.wav").touch()
    if text is not None:
        (d / "transcription.txt").write_text(text, encoding="utf-8")
    A._write_meta(d, name=name, language="tr")
    return d


def press(state, key, live=None, items=None):
    return A._tui_handle_key(key, state, live or FakeLive(), items)


def row_for(state, predicate):
    """Move the menu cursor onto the first row matching `predicate`."""
    items = A._menu_items(state)
    for i, it in enumerate(items):
        if predicate(it):
            state.menu_index = i
            return items, it
    raise AssertionError(f"no menu row matched among {[i.get('kind') for i in items]}")


# ============ the headline: the app is usable while a job runs ============

def test_confirming_a_stop_hands_the_menu_straight_back():
    hold = threading.Event()
    started = threading.Event()

    def slow_finalize(project_dir, manifests, save_channels=False, on_progress=None):
        started.set()
        hold.wait(TIMEOUT)
        return 1

    real = A._finalize_sources
    A._finalize_sources = slow_finalize
    try:
        with workdir("flow-menu-back") as home:
            state = recording_state(home)
            press(state, "q")
            press(state, "y")
            assert started.wait(TIMEOUT), "the stop worker never began"

            print(f"  mode while the job runs: {state.mode!r}, "
                  f"jobs: {len(A._active_jobs(state))}")
            assert state.mode == "menu", (
                f"the app is on {state.mode!r} while the job runs — the screen it used "
                "to sit on had no key handler at all")
            assert len(A._active_jobs(state)) == 1

            # And the menu really answers keys, which is the whole point.
            before = state.menu_index
            press(state, "DOWN")
            assert state.menu_index != before, (
                "the menu did not respond to a keystroke while a job was running")
            hold.set()
    finally:
        A._finalize_sources = real


def test_a_new_recording_starts_while_the_last_one_is_still_transcribing():
    """The user's own words: a recording is finished, it is transcribing and
    publishing, and the next recording has to be able to start."""
    hold = threading.Event()
    started = threading.Event()

    def slow_finalize(project_dir, manifests, save_channels=False, on_progress=None):
        started.set()
        hold.wait(TIMEOUT)
        return 1

    real_finalize, real_mic = A._finalize_sources, A._start_mic
    A._finalize_sources = slow_finalize
    A._start_mic = lambda state, raw_path, announce: FakeRecorder()
    try:
        with workdir("flow-second-recording") as home:
            state = recording_state(home)
            first_dir = state.project_dir
            press(state, "q")
            press(state, "y")
            assert started.wait(TIMEOUT), "the first recording never began finalizing"

            # Enter on "New recording", a name, Enter.
            row_for(state, lambda it: it.get("action") == "record")
            press(state, "ENTER")
            assert state.mode == "name_input", state.mode
            for ch in "second":
                press(state, ch)
            press(state, "ENTER")

            print(f"  mode={state.mode!r}  second dir={getattr(state.project_dir, 'name', None)!r}"
                  f"  jobs={len(A._active_jobs(state))}")
            assert state.mode == "recording", (
                f"the second recording did not start (mode {state.mode!r}, "
                f"status {state.status!r})")
            assert state.project_dir != first_dir, (
                "the second recording landed in the first one's folder")
            assert state.recorders, "the second recording has no recorder"
            assert len(A._active_jobs(state)) == 1, (
                "the first recording's job was lost when the second started")
            hold.set()
    finally:
        A._finalize_sources, A._start_mic = real_finalize, real_mic


def test_a_finishing_job_does_not_move_the_user_off_their_screen():
    """Workers used to write state.mode. A job finishing while the user was naming the
    next recording threw them back to the menu mid-word."""
    with workdir("flow-no-jump") as home:
        state = a_state(home)
        d = a_recording_on_disk(state.base_path, "2026-09-03_10-00-00")
        job = A._job_add(state, A.Job("publish", d, "rec", base_path=state.base_path))

        state.mode = "name_input"
        state.name_buffer = "half-typed"
        A._job_finish(state, job, "Published “rec” → https://example/commit/abc")

        print(f"  mode after the job finished: {state.mode!r}, "
              f"buffer {state.name_buffer!r}")
        assert state.mode == "name_input", (
            f"a finishing job moved the user to {state.mode!r}")
        assert state.name_buffer == "half-typed", "the half-typed name was lost"


def test_several_jobs_run_and_finish_together():
    release = threading.Event()
    in_flight = []
    lock = threading.Lock()

    def slow_finalize(project_dir, manifests, save_channels=False, on_progress=None):
        with lock:
            in_flight.append(project_dir)
        release.wait(TIMEOUT)
        return 1

    real_finalize, real_transcribe = A._finalize_sources, A._transcribe_and_save
    A._finalize_sources = slow_finalize
    A._transcribe_and_save = lambda st, job, **kw: A._job_finish(st, job, "saved")
    try:
        with workdir("flow-many") as home:
            state = a_state(home)
            jobs = []
            for i in range(3):
                d = state.base_path / f"2026-09-03_10-00-0{i}"
                d.mkdir(parents=True)
                job = A._job_add(state, A.Job(
                    "transcribe", d, f"rec{i}", language="tr",
                    base_path=state.base_path, audio_path=d / "audio.wav",
                    manifests=[], steps=["finalize", "transcribe"]))
                jobs.append(job)
                threading.Thread(target=A._stop_worker,
                                 args=(state, job, [], False), daemon=True).start()

            end = time.monotonic() + TIMEOUT
            while len(in_flight) < 3 and time.monotonic() < end:
                time.sleep(0.02)
            print(f"  {len(in_flight)} jobs in flight at once")
            assert len(in_flight) == 3, (
                f"only {len(in_flight)} of 3 jobs were running together")

            release.set()
            end = time.monotonic() + TIMEOUT
            while A._active_jobs(state) and time.monotonic() < end:
                time.sleep(0.02)
            states = [j.state for j in jobs]
            print(f"  final states: {states}")
            assert all(j.state == A._JOB_DONE for j in jobs), states
    finally:
        A._finalize_sources, A._transcribe_and_save = real_finalize, real_transcribe


# ============ one recording, one job ============

def test_a_second_transcribe_on_the_same_recording_is_refused():
    """[t] gates on has_transcript, which only flips at the job's very last step — so
    two presses used to start two jobs writing the same transcription.txt."""
    with workdir("flow-double-t") as home:
        state = a_state(home)
        a_recording_on_disk(state.base_path, "2026-09-03_10-00-00", text=None)
        state.recordings = A.list_recordings(state.base_path)

        real = A._transcribe_and_save
        A._transcribe_and_save = lambda *a, **kw: time.sleep(5)
        try:
            row_for(state, lambda it: it["kind"] == "recording")
            press(state, "t")
            press(state, "t")
            jobs = A._jobs_snapshot(state)
            print(f"  jobs after two [t]: {len(jobs)}, status={state.status!r}")
            assert len(jobs) == 1, f"two [t] presses started {len(jobs)} jobs"
            assert "already transcribing" in state.status.lower(), state.status
        finally:
            A._transcribe_and_save = real


def test_publish_and_transcribe_will_not_share_a_recording():
    """A transcribe job rewrites transcription.txt; a publish reads it and files it.
    Running both on one recording commits text that no longer matches."""
    with workdir("flow-t-vs-u") as home:
        state = a_state(home)
        d = a_recording_on_disk(state.base_path, "2026-09-03_10-00-00", text="x" * 5000)
        state.recordings = A.list_recordings(state.base_path)
        A._job_add(state, A.Job("transcribe", d, "rec", base_path=state.base_path))

        thread = A._publish_existing(state, {"dir": d, "name": "rec"})
        print(f"  publish refused: {thread is None}, status={state.status!r}")
        assert thread is None, "a publish started on a recording being transcribed"
        assert "already transcribing" in state.status.lower(), state.status


def test_delete_is_refused_while_a_job_owns_the_folder():
    """The refusal that costs money if it is missed: a publish writes its edited
    transcript and its documentation into the folder after paying for them."""
    with workdir("flow-delete") as home:
        state = a_state(home)
        d = a_recording_on_disk(state.base_path, "2026-09-03_10-00-00")
        state.recordings = A.list_recordings(state.base_path)
        A._job_add(state, A.Job("publish", d, "rec", base_path=state.base_path))

        _, row = row_for(state, lambda it: it["kind"] == "recording")
        press(state, "d")
        print(f"  mode={state.mode!r} status={state.status!r}")
        assert state.mode == "menu", (
            f"[d] opened the delete confirmation for a folder a publish is writing to "
            f"(mode {state.mode!r})")
        assert "already publishing" in state.status.lower(), state.status
        assert d.exists(), "the recording was deleted out from under a publish"


def test_delete_still_works_when_nothing_is_running():
    with workdir("flow-delete-ok") as home:
        state = a_state(home)
        a_recording_on_disk(state.base_path, "2026-09-03_10-00-00")
        state.recordings = A.list_recordings(state.base_path)
        row_for(state, lambda it: it["kind"] == "recording")
        press(state, "d")
        assert state.mode == "confirm_delete", (
            f"[d] no longer asks at all (mode {state.mode!r})")


# ============ quitting ============

def test_quitting_with_jobs_running_asks_first():
    with workdir("flow-quit") as home:
        state = a_state(home)
        d = a_recording_on_disk(state.base_path, "2026-09-03_10-00-00")
        job = A._job_add(state, A.Job("transcribe", d, "stand-up",
                                      base_path=state.base_path))

        assert press(state, "q") is True, "the app quit with a job in flight"
        assert state.mode == "confirm_quit", state.mode
        screen = plain_render(state)
        assert "stand-up" in screen, "the confirmation does not say what is running:\n" + screen
        assert "abandon" in screen.lower(), "it does not say what quitting costs:\n" + screen

        assert press(state, "n") is True, "any key but y should keep the app running"
        assert state.mode == "menu"

        state.mode = "confirm_quit"
        assert press(state, "y") is False, "[y] did not quit"

        A._job_finish(state, job, "saved")
        state.mode = "menu"
        assert press(state, "q") is False, (
            "the app refuses to quit with nothing running")


# ============ the rows the frame drew ============

def test_the_key_acts_on_the_row_the_frame_drew():
    """The item list is built once per frame and handed to the key handler.

    Jobs refresh state.recordings when they finish — _stop_worker does it as soon as
    audio.wav is safe — and the list is newest first, so a recording that finalizes
    while the user is choosing pushes every row down by one. Rebuilding the rows here
    instead of using the frame's means [d] deletes the recording *below* the one the
    user was looking at."""
    with workdir("flow-rows") as home:
        state = a_state(home)
        chosen = a_recording_on_disk(state.base_path, "2026-09-03_09-00-00",
                                     name="the one on screen")
        state.recordings = A.list_recordings(state.base_path)

        items, row = row_for(state, lambda it: it["kind"] == "recording")
        assert row["rec"]["dir"] == chosen, "the highlighted row is not the one meant"

        # A recording finishes finalizing between the frame and the keystroke. It is
        # newer, so it goes in above — exactly what a running job does on completion.
        a_recording_on_disk(state.base_path, "2026-09-03_12-00-00", name="just landed")
        state.recordings = A.list_recordings(state.base_path)
        assert state.recordings[0]["dir"] != chosen, "the fixture did not shift the rows"

        press(state, "d", items=items)
        print(f"  highlighted {chosen.name!r} -> delete target "
              f"{getattr(state.delete_target, 'name', None)!r}")
        assert state.delete_target == chosen, (
            f"[d] targeted {getattr(state.delete_target, 'name', None)!r} rather than "
            f"the highlighted {chosen.name!r} — the rows shifted between the frame and "
            "the keystroke")


def test_a_running_job_is_shown_on_its_recordings_row_and_in_the_footer():
    with workdir("flow-badge") as home:
        state = a_state(home)
        d = a_recording_on_disk(state.base_path, "2026-09-03_10-00-00", name="stand-up")
        state.recordings = A.list_recordings(state.base_path)
        job = A._job_add(state, A.Job("transcribe", d, "stand-up", language="tr",
                                      base_path=state.base_path,
                                      steps=["transcribe"]))
        job.pct = 0.42

        screen = plain_render(state)
        print("  " + next(ln.strip() for ln in screen.splitlines() if "stand-up" in ln))
        assert "transcribing 42%" in screen, (
            "the recording's row does not say a job is working on it:\n" + screen)
        assert "Active jobs (1)" in screen, (
            "the jobs group is not offered:\n" + screen)

        A._job_finish(state, job, "Saved “stand-up”")
        screen = plain_render(state)
        assert "transcribing 42%" not in screen, (
            "a finished job still claims the row:\n" + screen)


if __name__ == "__main__":
    run(["test_confirming_a_stop_hands_the_menu_straight_back",
         "test_a_new_recording_starts_while_the_last_one_is_still_transcribing",
         "test_a_finishing_job_does_not_move_the_user_off_their_screen",
         "test_several_jobs_run_and_finish_together",
         "test_a_second_transcribe_on_the_same_recording_is_refused",
         "test_publish_and_transcribe_will_not_share_a_recording",
         "test_delete_is_refused_while_a_job_owns_the_folder",
         "test_delete_still_works_when_nothing_is_running",
         "test_quitting_with_jobs_running_asks_first",
         "test_the_key_acts_on_the_row_the_frame_drew",
         "test_a_running_job_is_shown_on_its_recordings_row_and_in_the_footer"],
        globals())
