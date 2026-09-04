"""Project folders and meta.json under concurrency.

Both of these were safe only because the app could do one thing at a time. The TUI
locked its own screen while a recording was being transcribed, so a second recording
could not be started and a second job could not write meta.json. Once jobs run behind
the menu, two recordings can be named in the same second and two writers can reach one
meta.json — and every filename inside a project folder is fixed.
"""
import json
import threading
from datetime import datetime
from pathlib import Path

from support import A, run, workdir


class FrozenDatetime:
    """Stands in for audiocript's `datetime` so every call lands in one second."""

    def __init__(self, when):
        self._when = when

    def now(self):
        return self._when

    def strptime(self, *args):
        return datetime.strptime(*args)


def frozen_clock(when):
    """Patch A.datetime for the duration; returns the stamp _new_project_dir will use."""
    A.datetime = FrozenDatetime(when)
    return when.strftime(A._PROJECT_DIR_FORMAT)


def restore_clock():
    A.datetime = datetime


# =========================== _new_project_dir ===========================

def test_a_second_recording_in_the_same_second_gets_its_own_folder():
    with workdir("project-dir") as base:
        stamp = frozen_clock(datetime(2026, 9, 3, 10, 0, 0))
        try:
            first = A._new_project_dir(base)
            second = A._new_project_dir(base)
        finally:
            restore_clock()
        print(f"  {first.name} / {second.name}")
        assert first != second, (
            f"both recordings were handed the same folder: {first}")
        assert first.name == stamp, f"unexpected first name {first.name!r}"
        assert second.name == f"{stamp}-02", f"unexpected second name {second.name!r}"
        assert first.is_dir() and second.is_dir(), "a folder was not created"


def test_concurrent_starts_never_share_a_folder():
    """The claim is the mkdir itself, so threads asking at once cannot collide."""
    threads_n = 12
    with workdir("project-dir-race") as base:
        frozen_clock(datetime(2026, 9, 3, 10, 0, 0))
        got, errors = [], []
        lock = threading.Lock()
        start = threading.Event()

        def claim():
            start.wait()
            try:
                d = A._new_project_dir(base)
            except Exception as e:                       # pragma: no cover - reported
                with lock:
                    errors.append(e)
                return
            with lock:
                got.append(d)

        workers = [threading.Thread(target=claim) for _ in range(threads_n)]
        for t in workers:
            t.start()
        start.set()
        for t in workers:
            t.join(30)
        restore_clock()

        assert not errors, f"a claim raised: {errors[0]!r}"
        names = sorted(d.name for d in got)
        print(f"  {len(got)} claims, {len(set(names))} distinct: {names[0]} … {names[-1]}")
        assert len(got) == threads_n, f"only {len(got)}/{threads_n} threads returned"
        assert len(set(names)) == threads_n, (
            f"two threads were handed the same folder: {names}")
        on_disk = sorted(p.name for p in Path(base).iterdir() if p.is_dir())
        assert on_disk == names, f"folders on disk {on_disk} != claimed {names}"


def test_the_suffix_sorts_in_the_order_the_recordings_were_made():
    """list_recordings sorts on the folder name as a string, so an unpadded `-10`
    would come back before `-2` and the list would show them out of order."""
    with workdir("project-dir-sort") as base:
        frozen_clock(datetime(2026, 9, 3, 10, 0, 0))
        try:
            made = [A._new_project_dir(base).name for _ in range(12)]
        finally:
            restore_clock()
        print(f"  made: {made[0]} … {made[-1]}")
        assert len(set(made)) == len(made), (
            f"the names collapsed, so their order proves nothing: {made}")
        assert sorted(made) == made, (
            f"folder names do not sort in creation order:\n"
            f"    made:   {made}\n    sorted: {sorted(made)}")


def test_the_suffix_does_not_reach_the_created_column():
    with workdir("project-dir-fmt") as base:
        stamp = frozen_clock(datetime(2026, 9, 3, 10, 0, 0))
        try:
            plain = A._new_project_dir(base).name
            suffixed = A._new_project_dir(base).name
        finally:
            restore_clock()
        shown, shown_suffixed = A._fmt_created(plain), A._fmt_created(suffixed)
        print(f"  {plain!r} -> {shown!r}   {suffixed!r} -> {shown_suffixed!r}")
        assert shown == "2026-09-03 10:00", f"plain name formatted as {shown!r}"
        assert shown_suffixed == shown, (
            f"the -NN suffix leaked into the created column: {shown_suffixed!r}")
        assert stamp in plain


def test_a_folder_that_is_not_a_recording_is_shown_as_is():
    """_fmt_created falls back to the raw name, and must not mistake an arbitrary
    folder whose first 19 characters happen to parse for one of ours."""
    for name in ("notes", "2026-09-03_10-00-00-draft", "2026-13-45_99-99-99"):
        assert A._fmt_created(name) == name, f"{name!r} was reformatted"


# =========================== _write_meta ===========================

def test_concurrent_writers_lose_no_field():
    """Read-modify-write: without the lock, one writer's merge is built on a copy that
    predates another's and lands on top of it."""
    writers, per_writer = 8, 25
    with workdir("meta-merge") as d:
        A._write_meta(d, name="start")
        barrier = threading.Barrier(writers)

        def write(i):
            barrier.wait()
            for n in range(per_writer):
                A._write_meta(d, **{f"key{i}": n})

        threads = [threading.Thread(target=write, args=(i,)) for i in range(writers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(60)

        meta = A._read_meta(d)
        missing = [f"key{i}" for i in range(writers) if f"key{i}" not in meta]
        stale = {f"key{i}": meta.get(f"key{i}") for i in range(writers)
                 if meta.get(f"key{i}") != per_writer - 1}
        print(f"  keys: {sorted(k for k in meta if k.startswith('key'))}")
        assert meta.get("name") == "start", "the field written before the run was lost"
        assert not missing, f"{len(missing)} writers' fields were lost: {missing}"
        assert not stale, f"a writer's last value was overwritten by an older one: {stale}"


def test_a_reader_never_sees_a_half_written_file():
    """_read_meta treats an unparseable file as {}. A plain open(…, "w") truncates
    first, so a reader landing in that window sees a recording with no name, no
    language and no capture flag — a live capture read as one that never started."""
    payload = "x" * 200_000
    with workdir("meta-torn") as d:
        A._write_meta(d, name="stand-up", filler=payload)
        stop = threading.Event()
        empties, seen = [], []

        def writer():
            n = 0
            while not stop.is_set():
                n += 1
                A._write_meta(d, filler=payload, counter=n)

        def reader():
            while not stop.is_set():
                m = A._read_meta(d)
                seen.append(1)
                if not m or "name" not in m:
                    empties.append(m)

        threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
        for t in threads:
            t.start()
        threading.Event().wait(2.0)
        stop.set()
        for t in threads:
            t.join(30)

        print(f"  {len(seen)} reads, {len(empties)} of them torn")
        assert len(seen) > 50, f"the reader only managed {len(seen)} reads; too few to prove anything"
        assert not empties, (
            f"{len(empties)} of {len(seen)} reads saw a truncated meta.json")


def test_a_falsy_value_is_written_and_none_is_not():
    """The filter is `is not None`, not truthiness. `publish_error=""` is how a
    successful retry clears an earlier failure, and `speakers=None` is how a
    transcription without diarization declines to touch the field."""
    with workdir("meta-falsy") as d:
        A._write_meta(d, name="retro", publish_error="the push failed")
        A._write_meta(d, publish_error="")
        A._write_meta(d, name=None, speakers=None)

        meta = A._read_meta(d)
        print(f"  {meta}")
        assert meta.get("publish_error") == "", (
            f"an empty string did not land: {meta.get('publish_error')!r} — a truthiness "
            "filter leaves the old failure on a recording that published fine")
        assert meta.get("name") == "retro", "None wiped a field it should have skipped"
        assert "speakers" not in meta, "None was written as a field"


def test_a_write_into_a_deleted_recording_is_survivable():
    """The common case is a job writing meta for a folder the user deleted meanwhile."""
    with workdir("meta-gone") as parent:
        d = Path(parent) / "removed"
        d.mkdir()
        A._write_meta(d, name="doomed")
        import shutil
        shutil.rmtree(d)
        A._write_meta(d, published={"url": "https://example.invalid/c"})   # must not raise
        assert not d.exists(), "the write recreated the deleted recording"


def test_a_failed_write_leaves_no_temp_behind():
    """A temp that survives would be picked up as a stale file, and — had it been
    named `*.part` — deleted out from under a concurrent finalize."""
    with workdir("meta-temp") as d:
        A._write_meta(d, name="kept")
        A._write_meta(d, broken=object())        # not JSON-serializable; must not raise

        leftovers = sorted(p.name for p in Path(d).iterdir() if p.name != "meta.json")
        print(f"  leftovers: {leftovers}")
        assert not leftovers, f"a temp file survived a failed write: {leftovers}"
        assert A._read_meta(d).get("name") == "kept", "the failed write damaged meta.json"
        assert not A._META_TMP_NAME.endswith(".part"), (
            "the meta temp file is named *.part, which _finalize_sources glob-deletes")


if __name__ == "__main__":
    run(["test_a_second_recording_in_the_same_second_gets_its_own_folder",
         "test_concurrent_starts_never_share_a_folder",
         "test_the_suffix_sorts_in_the_order_the_recordings_were_made",
         "test_the_suffix_does_not_reach_the_created_column",
         "test_a_folder_that_is_not_a_recording_is_shown_as_is",
         "test_concurrent_writers_lose_no_field",
         "test_a_reader_never_sees_a_half_written_file",
         "test_a_falsy_value_is_written_and_none_is_not",
         "test_a_write_into_a_deleted_recording_is_survivable",
         "test_a_failed_write_leaves_no_temp_behind"], globals())
