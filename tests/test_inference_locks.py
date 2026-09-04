"""One inference at a time, and a lock order that stays the way it is.

Each backend is a single cached instance: one whisper.cpp context, one transformers
pipeline, one pyannote pipeline. While the app could only run one job these were
reached one at a time by construction. Jobs behind the menu remove that, so the
serialization has to be real.

The lock order is pinned here too, because the mistake that would break it looks like
a fix: guarding unload_model against an inference in flight.
"""
import threading
import time

from support import A, run, workdir

TIMEOUT = 30


class OverlapDetector:
    """Records whether two calls were ever inside at the same time."""

    def __init__(self, dwell=0.05):
        self.lock = threading.Lock()
        self.inside = 0
        self.max_inside = 0
        self.calls = 0
        self.dwell = dwell

    def enter(self):
        with self.lock:
            self.inside += 1
            self.calls += 1
            self.max_inside = max(self.max_inside, self.inside)
        time.sleep(self.dwell)
        with self.lock:
            self.inside -= 1


def in_parallel(target, n=4):
    """Run `target` on n threads released together; returns when all have finished."""
    start = threading.Event()

    def body():
        start.wait(TIMEOUT)
        target()

    threads = [threading.Thread(target=body, daemon=True) for _ in range(n)]
    for t in threads:
        t.start()
    start.set()
    for t in threads:
        t.join(TIMEOUT)
        assert not t.is_alive(), "a worker never finished"


# =========================== one inference at a time ===========================

def test_two_transcriptions_never_run_at_once():
    detector = OverlapDetector()

    class FakePipe:
        def __call__(self, path, **kw):
            detector.enter()
            return {"text": "hello"}

    saved_pipe, saved_ensure = A._hf_pipe, A._ensure_model
    A._hf_pipe = FakePipe()
    A._ensure_model = lambda code: None
    try:
        in_parallel(lambda: A.transcribe_audio("x.wav", "tr"), n=4)
    finally:
        A._hf_pipe, A._ensure_model = saved_pipe, saved_ensure

    print(f"  {detector.calls} transcriptions, max concurrent {detector.max_inside}")
    assert detector.calls == 4, f"only {detector.calls} calls arrived"
    assert detector.max_inside == 1, (
        f"{detector.max_inside} transcriptions were inside the model at once")


def test_transcribe_segments_shares_the_same_lock():
    """Both entry points reach the same instance, so they must queue against each
    other and not only against themselves."""
    detector = OverlapDetector()

    class FakePipe:
        def __call__(self, path, **kw):
            detector.enter()
            return {"text": "hello", "chunks": []}

    saved_pipe, saved_ensure = A._hf_pipe, A._ensure_model
    A._hf_pipe = FakePipe()
    A._ensure_model = lambda code: None
    try:
        both = [lambda: A.transcribe_audio("x.wav", "tr"),
                lambda: A.transcribe_segments("x.wav", "tr")]
        start = threading.Event()

        def body(fn):
            start.wait(TIMEOUT)
            fn()

        threads = [threading.Thread(target=body, args=(f,), daemon=True)
                   for f in both * 2]
        for t in threads:
            t.start()
        start.set()
        for t in threads:
            t.join(TIMEOUT)
    finally:
        A._hf_pipe, A._ensure_model = saved_pipe, saved_ensure

    print(f"  {detector.calls} calls, max concurrent {detector.max_inside}")
    assert detector.calls == 4
    assert detector.max_inside == 1, (
        f"transcribe_audio and transcribe_segments overlapped ({detector.max_inside})")


def test_two_diarizations_never_run_at_once():
    detector = OverlapDetector()

    class FakeAnnotation:
        def itertracks(self, yield_label=False):
            return []

    class FakeDiarPipe:
        def __call__(self, path):
            detector.enter()
            return FakeAnnotation()

    saved = A._diar_pipe
    A._diar_pipe = FakeDiarPipe()
    try:
        in_parallel(lambda: A.diarize("x.wav"), n=4)
    finally:
        A._diar_pipe = saved

    print(f"  {detector.calls} diarizations, max concurrent {detector.max_inside}")
    assert detector.calls == 4
    assert detector.max_inside == 1, (
        f"{detector.max_inside} diarizations were inside the pipeline at once")


def test_a_queued_job_is_told_it_is_queued():
    """Without this a job waiting behind a long transcription shows an active step
    with a bar that never moves, which reads as a hang."""
    waited = []
    holding = threading.Event()
    release = threading.Event()

    class FakePipe:
        def __call__(self, path, **kw):
            holding.set()
            release.wait(TIMEOUT)
            return {"text": "hello"}

    saved_pipe, saved_ensure = A._hf_pipe, A._ensure_model
    A._hf_pipe = FakePipe()
    A._ensure_model = lambda code: None
    try:
        first = threading.Thread(
            target=lambda: A.transcribe_audio("a.wav", "tr"), daemon=True)
        first.start()
        assert holding.wait(TIMEOUT), "the first transcription never started"

        second = threading.Thread(
            target=lambda: A.transcribe_audio(
                "b.wav", "tr", on_wait=waited.append), daemon=True)
        second.start()
        time.sleep(0.2)                       # long enough to have queued
        print(f"  on_wait told: {waited} (while the first still holds the model)")
        assert waited == [True], (
            f"the queued job was not told it was waiting: {waited}")

        release.set()
        first.join(TIMEOUT)
        second.join(TIMEOUT)
    finally:
        release.set()
        A._hf_pipe, A._ensure_model = saved_pipe, saved_ensure

    print(f"  on_wait told, after the model came free: {waited}")
    assert waited == [True, False], (
        f"a job that queued was never told the model was its own: {waited} — it would "
        f"keep drawing as queued for the rest of the run")


def test_a_job_that_does_not_wait_is_not_told_it_waited():
    waited = []

    class FakePipe:
        def __call__(self, path, **kw):
            return {"text": "hello"}

    saved_pipe, saved_ensure = A._hf_pipe, A._ensure_model
    A._hf_pipe = FakePipe()
    A._ensure_model = lambda code: None
    try:
        A.transcribe_audio("a.wav", "tr", on_wait=waited.append)
    finally:
        A._hf_pipe, A._ensure_model = saved_pipe, saved_ensure
    print(f"  on_wait told: {waited}")
    assert waited == [False], (
        f"an uncontended transcription reported itself as queued: {waited}")


# =========================== lock order ===========================

def test_unload_model_does_not_take_the_inference_lock():
    """The mistake this pins looks like a fix: 'don't drop a model mid-inference'.
    Taking _INFER_LOCK here would block the daemon's disable() for the length of a
    transcription, and would invert the order every inference path uses."""
    saved = A._hf_pipe
    A._hf_pipe = object()
    A._INFER_LOCK.acquire()
    try:
        done = threading.Event()

        def unload():
            A.unload_model("tr")
            done.set()

        t = threading.Thread(target=unload, daemon=True)
        t.start()
        finished = done.wait(5)
        print(f"  unload finished while the inference lock was held: {finished}")
        assert finished, (
            "unload_model blocked on _INFER_LOCK — that inverts the lock order and "
            "stalls the daemon's disable() for the length of a transcription")
    finally:
        A._INFER_LOCK.release()
        A._hf_pipe = saved


def test_the_inference_lock_is_not_held_across_a_model_load():
    """_ensure_model holds _MODEL_LOCK across a download. Holding _INFER_LOCK over it
    would make one job's cold start block every other job's inference, and would put
    the two locks in a nest that has no reason to exist."""
    observed = []
    saved_ensure, saved_pipe = A._ensure_model, A._hf_pipe

    def watching_ensure(code):
        observed.append(A._INFER_LOCK.locked())

    class FakePipe:
        def __call__(self, path, **kw):
            return {"text": "hi"}

    A._ensure_model = watching_ensure
    A._hf_pipe = FakePipe()
    try:
        A.transcribe_audio("a.wav", "tr")
    finally:
        A._ensure_model, A._hf_pipe = saved_ensure, saved_pipe

    print(f"  inference lock held during the load: {observed}")
    assert observed == [False], (
        f"_INFER_LOCK was already held when the model load ran: {observed}")


def test_the_diarizer_load_does_not_block_a_transcription_model_load():
    """They used to share _MODEL_LOCK, so a gated ~1 GB diarizer download stalled
    every transcription in both languages for as long as it took.

    Both loads have to be real calls into the ensure_* functions with the cache empty,
    or neither takes a lock and the test proves nothing: _ensure_model returns at the
    top when the pipeline is already there."""
    import sys
    import types

    diarizer_loading = threading.Event()
    release = threading.Event()

    class FakeDiarPipe:
        def to(self, device):
            return self

    class FakePipelineClass:
        @staticmethod
        def from_pretrained(name, use_auth_token=None):
            diarizer_loading.set()
            release.wait(TIMEOUT)          # stands in for the gated download
            return FakeDiarPipe()

    fake_pyannote = types.ModuleType("pyannote")
    fake_audio = types.ModuleType("pyannote.audio")
    fake_audio.Pipeline = FakePipelineClass
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.pipeline = lambda *a, **kw: object()

    saved_modules = {k: sys.modules.get(k)
                     for k in ("pyannote", "pyannote.audio", "transformers")}
    saved_hf, saved_diar = A._hf_pipe, A._diar_pipe
    sys.modules["pyannote"] = fake_pyannote
    sys.modules["pyannote.audio"] = fake_audio
    sys.modules["transformers"] = fake_transformers
    A._hf_pipe = None
    A._diar_pipe = None
    try:
        diar = threading.Thread(
            target=lambda: A._ensure_diarizer({"hf_token": "t"}), daemon=True)
        diar.start()
        assert diarizer_loading.wait(TIMEOUT), "the diarizer load never began"

        done = threading.Event()
        threading.Thread(
            target=lambda: (A._ensure_model("tr"), done.set()), daemon=True).start()
        finished = done.wait(5)
        print(f"  transcription model loaded while the diarizer was downloading: "
              f"{finished}")
        assert finished, (
            "_ensure_model waited on the diarizer's load — one job's gated download "
            "blocks every other job's transcription, in both languages")
    finally:
        release.set()
        diar.join(TIMEOUT)
        A._hf_pipe, A._diar_pipe = saved_hf, saved_diar
        for k, v in saved_modules.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


# =================== recovery must not touch a live capture ===================

def test_recovery_skips_the_folder_the_live_recording_is_writing():
    """A live recording carries the same in_progress flag as an abandoned one.
    Recovery reads that as permission to finalize the raw files and delete them."""
    with workdir("recover-skip") as base:
        live = base / "2026-09-03_10-00-00"
        dead = base / "2026-09-03_09-00-00"
        for d in (live, dead):
            d.mkdir(parents=True)
            (d / "mic.raw").write_bytes(b"\0" * 1024)
            A._write_meta(d, name=d.name, capture={
                "in_progress": True,
                "sources": [{"kind": "mic", "file": "mic.raw", "rate": 48000,
                             "channels": 1, "dtype": "int16"}]})

        found = A._interrupted_dirs(base, skip=[live])
        print(f"  interrupted: {[d.name for d in found]} (live: {live.name})")
        assert dead in found, "the genuinely interrupted recording was not offered"
        assert live not in found, (
            "recovery would finalize the recording that is still being captured, "
            "then unlink the raw file the recorder still holds open")

        # And with nothing excluded, it is offered — so the skip is what did it.
        assert live in A._interrupted_dirs(base), (
            "the live folder is not interrupted-looking at all; the test proves nothing")


def test_busy_dirs_names_the_live_recording_only_while_it_records():
    import types
    project = object()
    recording = types.SimpleNamespace(project_dir=project, recorders=["mic"])
    idle = types.SimpleNamespace(project_dir=project, recorders=[])
    assert A._busy_dirs(recording) == [project]
    assert A._busy_dirs(idle) == [], "an idle app claimed to be writing to a folder"


if __name__ == "__main__":
    run(["test_two_transcriptions_never_run_at_once",
         "test_transcribe_segments_shares_the_same_lock",
         "test_two_diarizations_never_run_at_once",
         "test_a_queued_job_is_told_it_is_queued",
         "test_a_job_that_does_not_wait_is_not_told_it_waited",
         "test_unload_model_does_not_take_the_inference_lock",
         "test_the_inference_lock_is_not_held_across_a_model_load",
         "test_the_diarizer_load_does_not_block_a_transcription_model_load",
         "test_recovery_skips_the_folder_the_live_recording_is_writing",
         "test_busy_dirs_names_the_live_recording_only_while_it_records"], globals())
