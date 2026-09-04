"""The job list: what runs behind the menu, and the rules that keep it honest.

Three of these pin decisions that are easy to undo without noticing:
  - a finished job's result stays readable for a while, but pruning happens once per
    frame and never inside the snapshot the renderers share;
  - a job that raises still reaches a terminal state, or it holds its recording's
    guard for the rest of the session;
  - one recording, one job — the check has to be a check-and-insert under the lock,
    not two reads that can both pass.
"""
import threading
import types

from support import A, run, workdir

TIMEOUT = 30


def a_state(base):
    cfg = {"base_path": str(base), "language": "tr", "open_app": None,
           "capture_system_audio": False, "diarize": False}
    return A._TuiState(cfg)


def a_job(project_dir, kind="transcribe", label="stand-up", **kw):
    return A.Job(kind, project_dir, label, **kw)


# =========================== the snapshot and the prune ===========================

def test_the_snapshot_does_not_prune():
    """Body, footer and the key handler each ask for the list inside one frame. A
    prune in the snapshot would hand them different lists, and could take a job's
    result away between the frame the user read and the key they answered with."""
    with workdir("jobs-snapshot") as base:
        state = a_state(base)
        job = A._job_add(state, a_job(base / "r1"))
        A._job_finish(state, job, "done")
        job.finished_at = A.time.monotonic() - (A._JOB_KEEP_SECONDS * 10)

        first = A._jobs_snapshot(state)
        second = A._jobs_snapshot(state)
        print(f"  two snapshots: {len(first)} / {len(second)} rows")
        assert first == second, "two snapshots in one frame disagreed"
        assert len(first) == 1, (
            "the snapshot pruned a long-finished job — renderers in the same frame "
            "would then see different lists")


def test_pruning_keeps_a_result_on_screen_then_drops_it():
    with workdir("jobs-prune") as base:
        state = a_state(base)
        fresh = A._job_add(state, a_job(base / "fresh", label="fresh"))
        stale = A._job_add(state, a_job(base / "stale", label="stale"))
        running = A._job_add(state, a_job(base / "running", label="running"))
        A._job_finish(state, fresh, "saved")
        A._job_finish(state, stale, "saved")
        stale.finished_at = A.time.monotonic() - (A._JOB_KEEP_SECONDS + 1)

        A._prune_jobs(state)
        left = [j.label for j in A._jobs_snapshot(state)]
        print(f"  left after prune: {left}")
        assert "stale" not in left, "a long-finished job was never dropped"
        assert "fresh" in left, (
            "a just-finished job was dropped before its result could be read")
        assert "running" in left, "a running job was pruned"


def test_a_job_that_never_finishes_is_never_pruned():
    """Which is why every worker has to reach _job_finish on every path."""
    with workdir("jobs-stuck") as base:
        state = a_state(base)
        A._job_add(state, a_job(base / "stuck"))
        A._prune_jobs(state, now=A.time.monotonic() + 10_000)
        assert len(A._jobs_snapshot(state)) == 1
        assert A._active_jobs(state), "a running job dropped off the active list"


# =========================== terminal states ===========================

def test_finishing_records_the_outcome_and_the_time():
    with workdir("jobs-finish") as base:
        state = a_state(base)
        ok = A._job_add(state, a_job(base / "a"))
        bad = A._job_add(state, a_job(base / "b"))
        A._job_finish(state, ok, "Saved “stand-up”")
        A._job_finish(state, bad, "disk went away", failed=True)

        assert ok.state == A._JOB_DONE and ok.is_terminal()
        assert bad.state == A._JOB_FAILED and bad.is_terminal()
        assert bad.message == "disk went away"
        assert ok.finished_at is not None and bad.finished_at is not None
        assert A._active_jobs(state) == [], "a finished job stayed active"


def test_a_queued_job_reads_as_queued_not_as_stalled():
    """_job_waiting is what transcribe_audio calls: True while the job waits for the
    model, False once it has it."""
    with workdir("jobs-queued") as base:
        state = a_state(base)
        job = A._job_add(state, a_job(base / "q", steps=["transcribe"]))
        told = A._job_waiting(job)

        told(True)
        assert job.state == A._JOB_QUEUED and job.queued
        assert not job.is_terminal(), "a queued job counted as finished"
        assert A._active_jobs(state) == [job], "a queued job dropped off the active list"

        told(False)
        assert job.state == A._JOB_RUNNING and not job.queued


def test_beginning_a_step_restarts_its_clock_and_clears_queued():
    with workdir("jobs-steps") as base:
        state = a_state(base)
        job = A._job_add(state, a_job(base / "s", steps=["finalize", "transcribe"]))
        A._job_waiting(job)(True)
        job.phase_start = A.time.monotonic() - 500

        A._job_begin(job, "transcribe", pct=0.0)
        assert job.phase == "transcribe"
        assert job.pct == 0.0
        assert not job.queued, "entering a step left the job marked as queued"
        assert A.time.monotonic() - job.phase_start < 5, "the step clock was not restarted"

        A._job_begin(job, "diarize", steps=["diarize", "transcribe"])
        assert job.steps == ["diarize", "transcribe"], "the step plan did not change"


# =========================== one recording, one job ===========================

def test_a_second_job_on_the_same_recording_is_visible_to_the_guard():
    with workdir("jobs-guard") as base:
        state = a_state(base)
        d = base / "2026-09-03_10-00-00"
        other = base / "2026-09-03_11-00-00"
        job = A._job_add(state, a_job(d, kind="transcribe"))

        assert A._job_working_in(state, d) is job
        assert A._job_working_in(state, other) is None, "the guard matched the wrong folder"
        assert A._job_working_in(state, d, kinds=("publish",)) is None
        assert A._job_working_in(state, d, kinds=("transcribe", "publish")) is job

        A._job_finish(state, job, "saved")
        assert A._job_working_in(state, d) is None, (
            "a finished job still blocks its recording — [t] and [u] would stay "
            "refused for the rest of the session")


def test_the_guard_ignores_jobs_with_no_folder():
    with workdir("jobs-nodir") as base:
        state = a_state(base)
        A._job_add(state, a_job(None, kind="recover", label="recovery"))
        assert A._job_working_in(state, base / "anything") is None
        assert A._job_working_in(state, None) is None


def test_the_key_is_the_kind_and_the_recording():
    with workdir("jobs-key") as base:
        d = base / "r"
        assert a_job(d, kind="transcribe").key == a_job(d, kind="transcribe").key
        assert a_job(d, kind="transcribe").key != a_job(d, kind="publish").key
        assert a_job(d, kind="publish").key != a_job(base / "s", kind="publish").key


# =========================== the list is shared state ===========================

def test_jobs_added_from_many_threads_all_arrive():
    """state.jobs is appended from workers and read from the render loop."""
    n = 40
    with workdir("jobs-threads") as base:
        state = a_state(base)
        barrier = threading.Barrier(n)

        def add(i):
            barrier.wait()
            A._job_add(state, a_job(base / f"r{i}", label=f"r{i}"))

        threads = [threading.Thread(target=add, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(TIMEOUT)

        labels = sorted(j.label for j in A._jobs_snapshot(state))
        print(f"  {len(labels)} of {n} arrived")
        assert len(labels) == n, f"only {len(labels)} of {n} jobs were kept"
        assert len(set(labels)) == n


def test_a_job_carries_its_snapshot_not_a_reference_to_the_state():
    with workdir("jobs-snapshot-fields") as base:
        state = a_state(base)
        state.pending_name = "typed later"
        job = a_job(base / "r", language="tr", name="at the time",
                    open_app="Sublime Text", diarize=False)
        state.language = "en"
        state.open_app = None
        state.diarize = True

        print(f"  job language={job.language!r} name={job.name!r} open_app={job.open_app!r}")
        assert job.language == "tr"
        assert job.name == "at the time"
        assert job.open_app == "Sublime Text"
        assert job.diarize is False


if __name__ == "__main__":
    run(["test_the_snapshot_does_not_prune",
         "test_pruning_keeps_a_result_on_screen_then_drops_it",
         "test_a_job_that_never_finishes_is_never_pruned",
         "test_finishing_records_the_outcome_and_the_time",
         "test_a_queued_job_reads_as_queued_not_as_stalled",
         "test_beginning_a_step_restarts_its_clock_and_clears_queued",
         "test_a_second_job_on_the_same_recording_is_visible_to_the_guard",
         "test_the_guard_ignores_jobs_with_no_folder",
         "test_the_key_is_the_kind_and_the_recording",
         "test_jobs_added_from_many_threads_all_arrive",
         "test_a_job_carries_its_snapshot_not_a_reference_to_the_state"], globals())
