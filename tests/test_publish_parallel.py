"""Publishing in parallel: what may overlap, and the one thing that may not.

Two publishes were never possible before — the app refused a second one while any
background message was on screen — so nothing here was reachable. Now they run
together, and the split matters: the two model calls are the slow, expensive part and
should overlap, while the push cannot, because both commits would branch from the same
head and only one of them can fast-forward the branch.
"""
import threading
import time

from support import run, workdir

import publish

TIMEOUT = 30


class OverlapDetector:
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


def in_parallel(target, n=3):
    errors = []
    start = threading.Event()

    def body():
        start.wait(TIMEOUT)
        try:
            target()
        except Exception as e:                       # pragma: no cover - reported
            errors.append(e)

    threads = [threading.Thread(target=body, daemon=True) for _ in range(n)]
    for t in threads:
        t.start()
    start.set()
    for t in threads:
        t.join(TIMEOUT)
        assert not t.is_alive(), "a publish never finished"
    assert not errors, f"a publish raised: {errors[0]!r}"


def test_two_pushes_never_overlap():
    """The head read, the commit built on it and the ref move are one sequence. Two of
    them interleaved means the second PATCH is not a fast-forward, GitHub answers 422,
    and a job that did everything right reports a failure."""
    detector = OverlapDetector()

    def fake_gh(method, path, cfg, body=None):
        detector.enter()
        if path.startswith("/repos/o/r/git/ref/"):
            return 200, {"object": {"sha": "head-sha"}}
        if path.startswith("/repos/o/r/git/commits/"):
            return 200, {"tree": {"sha": "tree-sha"}}
        if path == "/repos/o/r/git/blobs":
            return 201, {"sha": "blob-sha"}
        if path == "/repos/o/r/git/trees":
            return 201, {"sha": "new-tree"}
        if path == "/repos/o/r/git/commits":
            return 201, {"sha": "new-sha", "html_url": "https://x/commit/new-sha"}
        if path.startswith("/repos/o/r/git/refs/"):
            return 200, {}
        return 200, {"default_branch": "main"}

    real = publish._gh
    publish._gh = fake_gh
    try:
        cfg = {"repo": "o/r", "token": "t"}
        in_parallel(lambda: publish.publish_files(
            cfg, "folder", {"a.md": "x"}, "message"), n=3)
    finally:
        publish._gh = real

    print(f"  {detector.calls} GitHub calls, max concurrent {detector.max_inside}")
    assert detector.calls > 3, "the fixture never reached the API"
    assert detector.max_inside == 1, (
        f"{detector.max_inside} pushes were inside the head-read-to-ref-move sequence "
        "at once; the losers would come back as a 422")


def test_the_model_calls_do_overlap():
    """The part worth parallelising: two calls that take minutes each and cost money.
    Serializing these would make a second publish wait for the whole of the first."""
    detector = OverlapDetector()

    def fake_complete(prompt, cfg):
        detector.enter()
        return "written"

    real_complete, real_files = publish.openai_complete, publish.publish_files
    publish.openai_complete = fake_complete
    publish.publish_files = lambda cfg, folder, files, message: "https://x/commit/abc"
    try:
        with workdir("publish-parallel") as base:
            dirs = []
            for i in range(3):
                d = base / f"rec{i}"
                d.mkdir()
                dirs.append(d)
            cfg = {"repo": "o/r", "token": "t", "min_chars": 1}
            index = iter(dirs)
            lock = threading.Lock()

            def one_publish():
                with lock:
                    d = next(index)
                publish.run(cfg, d, "some transcript", "title")

            in_parallel(one_publish, n=3)
    finally:
        publish.openai_complete = real_complete
        publish.publish_files = real_files

    print(f"  {detector.calls} model calls, max concurrent {detector.max_inside}")
    assert detector.calls == 6, f"expected two calls per publish, got {detector.calls}"
    assert detector.max_inside > 1, (
        "the model calls were serialized — a second publish now waits out the first's "
        "two multi-minute calls for no reason")


def test_an_already_written_stage_is_not_paid_for_twice():
    """Guards the idempotence the quit confirmation promises: abandoning a publish and
    pressing [u] again must not re-run the model calls that already produced output."""
    calls = []

    real_complete, real_files = publish.openai_complete, publish.publish_files
    publish.openai_complete = lambda prompt, cfg: calls.append(1) or "written"
    publish.publish_files = lambda cfg, folder, files, message: "https://x/commit/abc"
    try:
        with workdir("publish-idempotent") as base:
            d = base / "rec"
            d.mkdir()
            cfg = {"repo": "o/r", "token": "t", "min_chars": 1}
            publish.run(cfg, d, "some transcript", "title")
            first = len(calls)
            publish.run(cfg, d, "some transcript", "title")
            print(f"  model calls: {first} then {len(calls) - first}")
            assert first == 2, f"expected two model calls on the first run, got {first}"
            assert len(calls) == first, (
                "a retry paid for the edit and the documentation a second time")
            assert (d / publish.EDITED_NAME).exists()
            assert (d / publish.DOC_NAME).exists()
    finally:
        publish.openai_complete = real_complete
        publish.publish_files = real_files


if __name__ == "__main__":
    run(["test_two_pushes_never_overlap",
         "test_the_model_calls_do_overlap",
         "test_an_already_written_stage_is_not_paid_for_twice"], globals())
