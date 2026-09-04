"""Background workers must not read the state the user is still changing.

Every one of these was unreachable while the transcription screen locked the keyboard:
the app could only ever have one job, so a worker reading `state.recorders`,
`state.pending_name` or `state.language` was reading its own. Once a job runs behind
the menu those reads find the *next* recording's values, and the failures are not
subtle — one of them deletes the recording the worker was called to save.
"""
import threading
import time
from pathlib import Path

import numpy as np

from support import A, run, workdir, write_wav

TIMEOUT = 30


class FakeLive:
    def update(self, _renderable):
        pass


class FakeRecorder:
    """A recorder whose stop() can be held open, so the test can interleave a second
    recording exactly where the worker used to re-read the state."""

    KIND = "mic"

    def __init__(self, raw_name="mic.raw", hold=None):
        self.raw_name = raw_name
        self.stopped = False
        self._hold = hold

    def level(self):
        return 0.0

    def manifest(self):
        return {"kind": "mic", "file": self.raw_name, "rate": 48000,
                "channels": 1, "dtype": "int16"}

    def stop(self):
        if self._hold is not None:
            self._hold.wait(TIMEOUT)
        self.stopped = True


def recording_state(home, recorder):
    """A _TuiState mid-recording, as _start_recording leaves it."""
    cfg = {"base_path": str(home / "recordings"), "language": "tr",
           "open_app": None, "capture_system_audio": False, "diarize": False}
    state = A._TuiState(cfg)
    project = home / "recordings" / "2026-08-18_09-00-00"
    project.mkdir(parents=True, exist_ok=True)
    state.project_dir = project
    state.pending_name = "stand-up"
    state.recorders = [recorder]
    state.rec_start = A.time.monotonic() - 42
    state.mode = "recording"
    return state


def _join_worker(state, deadline=TIMEOUT):
    """Wait for the daemon worker _stop_and_transcribe spawned to leave."""
    end = time.monotonic() + deadline
    while time.monotonic() < end:
        if not any(t.is_alive() and t.daemon and t.name.startswith("Thread")
                   for t in threading.enumerate()
                   if t is not threading.current_thread()):
            return
        time.sleep(0.02)


# =================== the worker must not see the next recording ===================

def test_the_stop_worker_never_reaches_the_next_recordings_recorders():
    """The one that loses a recording.

    Old order: stop the recorders, then build the manifest from `state.recorders`
    again, then clear it. Hold the first stop() open, start a second recording in that
    window, and the manifest read finds the new recorder — so finalize looks for the
    wrong raw file, finds nothing, and the caller deletes the folder as an empty take.
    """
    seen = {}
    hold = threading.Event()
    entered = threading.Event()
    real_finalize = A._finalize_sources

    def fake_finalize(project_dir, manifests, save_channels=False, on_progress=None):
        seen["manifests"] = [dict(m) for m in manifests]
        entered.set()
        return 1                                  # pretend a take was written

    with workdir("stop-isolation") as home:
        old = FakeRecorder("mic.raw", hold=hold)
        # A distinct raw name, so a manifest built from the wrong list is visible
        # rather than coincidentally identical.
        new = FakeRecorder("second-recording-mic.raw")
        state = recording_state(home, old)
        project_dir = state.project_dir
        (project_dir / "audio.wav").touch()

        A._finalize_sources = fake_finalize
        real_transcribe = A._transcribe_and_save
        A._transcribe_and_save = lambda st, job, **kw: A._job_finish(st, job, "done")
        try:
            A._stop_and_transcribe(state, FakeLive())
            assert state.recorders == [], (
                "the recorders were left on the state for the worker to re-read; "
                f"still there: {state.recorders!r}")

            # A second recording starts while the first is still being stopped.
            state.recorders = [new]
            hold.set()
            assert entered.wait(TIMEOUT), "the worker never reached finalize"
            _join_worker(state)
        finally:
            A._finalize_sources = real_finalize
            A._transcribe_and_save = real_transcribe

        print(f"  manifests seen: {seen.get('manifests')}")
        assert seen.get("manifests") == [old.manifest()], (
            "finalize was handed a manifest that is not the stopped recording's: "
            f"{seen.get('manifests')}")
        assert old.stopped, "the recording being stopped was not stopped"
        assert not new.stopped, "the worker stopped the recording that had just started"
        assert state.recorders == [new], (
            "the worker cleared the next recording's recorders — its meters go blank "
            f"and nothing can stop it at exit; left: {state.recorders!r}")


def test_a_worker_that_raises_still_hands_the_screen_back():
    """A second, separate cause of the same symptom the user reported: an exception
    escaping the worker killed the thread silently and left mode on "transcribing"
    forever, with every keystroke dropped and nothing on screen to say why."""
    real_finalize = A._finalize_sources

    def exploding_finalize(*a, **kw):
        raise OSError("disk went away")

    with workdir("stop-raises") as home:
        state = recording_state(home, FakeRecorder())
        A._finalize_sources = exploding_finalize
        try:
            job = A._stop_and_transcribe(state, FakeLive())
            end = time.monotonic() + TIMEOUT
            while not job.is_terminal() and time.monotonic() < end:
                time.sleep(0.02)
        finally:
            A._finalize_sources = real_finalize

        print(f"  mode={state.mode!r} job={job.state!r} message={job.message!r}")
        assert state.mode == "menu", (
            f"a failed stop left the app on {state.mode!r} instead of the menu")
        assert job.state == A._JOB_FAILED, (
            f"the job never reached a terminal state ({job.state!r}) — it would hold "
            "its recording's guard and sit in Active jobs for the rest of the session")
        assert "disk went away" in job.message, (
            f"the failure was not reported: {job.message!r}")
        assert A._job_working_in(state, state.project_dir) is None, (
            "the failed job still blocks its recording; [t] would stay refused")


# =========================== snapshots, not live state ===========================

def test_a_transcription_saves_under_the_name_its_job_began_with():
    """`_save_and_open` used to fall back to state.pending_name / state.language."""
    hold = threading.Event()
    started = threading.Event()

    def slow_transcript(job):
        started.set()
        hold.wait(TIMEOUT)
        return "the transcript"

    real = A._plain_transcript
    A._plain_transcript = slow_transcript
    try:
        with workdir("save-snapshot") as home:
            state = recording_state(home, FakeRecorder())
            d = state.project_dir
            write_wav(d / "audio.wav", np.zeros(16000, dtype=np.int16))
            job = A.Job("transcribe", d, "job-name", language="tr", name="job-name",
                        open_app=None, base_path=state.base_path,
                        audio_path=d / "audio.wav", steps=["transcribe"])
            A._job_add(state, job)

            t = threading.Thread(target=A._transcribe_and_save, args=(state, job),
                                 daemon=True)
            t.start()
            assert started.wait(TIMEOUT), "the transcription never began"

            # The user names the next recording and flips the language meanwhile.
            state.pending_name = "THE NEXT RECORDING"
            state.language = "en"
            state.open_app = "Some Other App"
            hold.set()
            t.join(TIMEOUT)

            meta = A._read_meta(d)
            print(f"  meta name={meta.get('name')!r} language={meta.get('language')!r}"
                  f" job={job.state!r}")
            assert meta.get("name") == "job-name", (
                f"the transcript was saved under a name the user typed after it "
                f"started: {meta.get('name')!r}")
            assert meta.get("language") == "tr", (
                f"a language toggle mid-job relabelled the finished transcript: "
                f"{meta.get('language')!r}")
            assert job.state == A._JOB_DONE, f"the job did not finish: {job.state!r}"
    finally:
        A._plain_transcript = real


def test_a_rename_during_a_job_is_not_overwritten_when_it_finishes():
    """The lost update a lock cannot fix. The job carries the name the recording had
    when it started; a rename while it runs is the user's more recent intent, and
    writing the job's name unconditionally at the end silently undid it — reliably,
    once the write became atomic."""
    hold = threading.Event()
    started = threading.Event()

    def slow_transcript(job):
        started.set()
        hold.wait(TIMEOUT)
        return "the transcript"

    real = A._plain_transcript
    A._plain_transcript = slow_transcript
    try:
        with workdir("rename-race") as home:
            state = recording_state(home, FakeRecorder())
            d = state.project_dir
            write_wav(d / "audio.wav", np.zeros(16000, dtype=np.int16))
            A._write_meta(d, name="before", language="tr")
            job = A.Job("transcribe", d, "before", language="tr", name="before",
                        open_app=None, base_path=state.base_path,
                        audio_path=d / "audio.wav", steps=["transcribe"])
            A._job_add(state, job)

            t = threading.Thread(target=A._transcribe_and_save, args=(state, job),
                                 daemon=True)
            t.start()
            assert started.wait(TIMEOUT), "the transcription never began"

            A._write_meta(d, name="renamed while it ran")   # the [r] path's write
            hold.set()
            t.join(TIMEOUT)

            name = A._read_meta(d).get("name")
            print(f"  meta name after the job finished: {name!r}")
            assert name == "renamed while it ran", (
                f"the job wrote its own stale name over the rename: {name!r}")
    finally:
        A._plain_transcript = real


def test_an_import_uses_one_language_from_end_to_end():
    """The import worker read state.language twice — once to decide whether the panel
    could show a real percentage, once to pick the model."""
    hold = threading.Event()
    started = threading.Event()
    used = []

    def slow_transcribe(path, language, on_progress=None, duration=None, on_wait=None):
        used.append(language)
        started.set()
        hold.wait(TIMEOUT)
        return "imported text"

    real_transcribe = A.transcribe_audio
    real_extract = A._extract_audio
    real_duration = A._media_duration
    A.transcribe_audio = slow_transcribe
    A._extract_audio = lambda src, dst, duration=None, on_pct=None: Path(dst).touch()
    A._media_duration = lambda p: 12.0
    try:
        with workdir("import-snapshot") as home:
            state = recording_state(home, FakeRecorder())
            d = state.project_dir
            state.language = "tr"
            job = A.Job("import", d, "clip.mp4", language="tr", name="clip",
                        open_app=None, base_path=state.base_path,
                        audio_path=d / "audio.wav", steps=["extract", "transcribe"])
            A._job_add(state, job)

            t = threading.Thread(
                target=A._import_worker,
                args=(state, job, str(home / "clip.mp4"), False), daemon=True)
            t.start()
            assert started.wait(TIMEOUT), "the import never reached transcription"

            state.language = "en"           # the user toggles mid-import
            hold.set()
            t.join(TIMEOUT)

            meta = A._read_meta(d)
            print(f"  transcribed as {used}, meta language={meta.get('language')!r}")
            assert used == ["tr"], (
                f"the import switched models mid-job: {used}")
            assert meta.get("language") == "tr", (
                f"the import was filed under the language the user switched to: "
                f"{meta.get('language')!r}")
            assert job.state == A._JOB_DONE, f"the job did not finish: {job.state!r}"
    finally:
        A.transcribe_audio = real_transcribe
        A._extract_audio = real_extract
        A._media_duration = real_duration


# =========================== names are not markup ===========================

def test_a_recording_named_like_a_closing_tag_still_renders():
    """The jobs panel interpolates a name into a markup string and hands it to
    Text.from_markup, which raises on a tag it cannot match. That raise happens inside
    Live's render, so one badly named recording takes the whole app down — and a
    recovery job carrying such a name appears at startup, before anything can be done
    about it."""
    hostile_names = ("[/x]", "[bold]", "[", "] [/]", "[red]on[/blue]")
    with workdir("markup") as home:
        state = recording_state(home, FakeRecorder())
        for hostile in hostile_names:
            for kind, steps in (("transcribe", ["transcribe"]),
                                ("import", ["extract", "transcribe"]),
                                ("recover", ["finalize"])):
                job = A.Job(kind, state.project_dir, hostile, language="tr",
                            steps=steps)
                job.pct = 0.5
                A._jobs_panel(state, [job])               # must not raise
                A._job_summary(job)
                A._job_badge(job)
                A._menu_row_text({"kind": "job", "job": job}, 80)
                A._job_finish(state, job, f"failed on {hostile}", failed=True)
                A._jobs_panel(state, [job])               # terminal rows too
        print("  rendered: " + ", ".join(repr(h) for h in hostile_names))

        # Escaping must show the name, not swallow it: the panel's renderable is the
        # already-parsed Text, so its plain form is what the user sees.
        job = A.Job("transcribe", state.project_dir, "[/x]", language="tr",
                    steps=["transcribe"])
        text = A._jobs_panel(state, [job]).renderable.plain
        assert "[/x]" in text, f"the name was swallowed instead of shown: {text!r}"


if __name__ == "__main__":
    run(["test_the_stop_worker_never_reaches_the_next_recordings_recorders",
         "test_a_worker_that_raises_still_hands_the_screen_back",
         "test_a_transcription_saves_under_the_name_its_job_began_with",
         "test_a_rename_during_a_job_is_not_overwritten_when_it_finishes",
         "test_an_import_uses_one_language_from_end_to_end",
         "test_a_recording_named_like_a_closing_tag_still_renders"], globals())
