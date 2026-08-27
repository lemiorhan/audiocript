"""audiocript.unload_model: giving a loaded model's memory back.

Measured on this machine before the function existed: a loaded Turkish model holds
3085.6 MB of MPS memory, and dropping the pipeline reference plus
torch.mps.empty_cache() returns all of it (allocated 3085.6 MB -> 0.0 MB, driver
3208.4 MB -> 0.4 MB). Host RSS does not fall — 454.5 MB -> 488.7 MB — because the
weights live GPU-side and the host figure is torch's own arena. Freeing the host
arena is not what this function is for.

No test here loads a model. Each one puts a sentinel in the module global a real load
would fill and asserts on what unload_model does with it, so the file runs in well
under a second and needs no GPU.
"""
import contextlib
import subprocess
import sys
import threading

from support import A, REPO_ROOT, run

SENTINEL_TR = object()
SENTINEL_EN = object()


@contextlib.contextmanager
def loaded(tr=None, en=None):
    """Put sentinels in the two model globals, and restore them afterwards whatever
    the test did — a leaked sentinel would make a later test in this file assert
    against a model it never set."""
    saved = A._hf_pipe, A._cpp_model
    A._hf_pipe, A._cpp_model = tr, en
    try:
        yield
    finally:
        A._hf_pipe, A._cpp_model = saved


def test_the_turkish_model_is_dropped():
    with loaded(tr=SENTINEL_TR):
        A.unload_model("tr")
        assert A._hf_pipe is None, f"_hf_pipe survived the unload: {A._hf_pipe!r}"


def test_the_english_model_is_dropped():
    with loaded(en=SENTINEL_EN):
        A.unload_model("en")
        assert A._cpp_model is None, f"_cpp_model survived the unload: {A._cpp_model!r}"


def test_each_language_leaves_the_other_alone():
    """Both can be resident at once — `_ensure_model` caches per language and never
    clears the other one — so unloading one must not take the other's memory with it.
    """
    with loaded(tr=SENTINEL_TR, en=SENTINEL_EN):
        A.unload_model("tr")
        assert A._cpp_model is SENTINEL_EN, "unloading tr dropped the english model too"
    with loaded(tr=SENTINEL_TR, en=SENTINEL_EN):
        A.unload_model("en")
        assert A._hf_pipe is SENTINEL_TR, "unloading en dropped the turkish model too"


def test_the_lock_is_held_while_it_drops():
    """`_ensure_model` publishes its pipeline under `_MODEL_LOCK`. An unload that does
    not take the same lock can drop a pipeline a load is still assembling, and the
    load would then publish a model nobody asked for over an unload that already
    told the user the daemon is off."""
    with loaded(tr=SENTINEL_TR):
        A._MODEL_LOCK.acquire()
        done = threading.Event()

        def unload():
            A.unload_model("tr")
            done.set()

        thread = threading.Thread(target=unload, name="unload-under-lock", daemon=True)
        thread.start()
        try:
            assert not done.wait(0.5), \
                "unload_model finished while _MODEL_LOCK was held by another thread"
            assert A._hf_pipe is SENTINEL_TR, "the model was dropped without the lock"
        finally:
            A._MODEL_LOCK.release()
        thread.join(5)
        assert not thread.is_alive(), \
            "unload_model never returned after _MODEL_LOCK was released"
        assert A._hf_pipe is None, "the model survived an unload that did hold the lock"


def test_torch_is_not_imported_in_order_to_unload():
    """The English model never brings torch in at all, and importing it to free
    memory would cost far more than it returns. So the MPS cache drop is conditional
    on torch already being in sys.modules.

    Measured in a child interpreter rather than this one, for the same reason
    `support.child_peak_rss_mb` uses one: `sys.modules` is process-global. An
    in-process version of this test was written first, and a mutation proved it
    useless — with the guard removed, an earlier test in this file imported torch,
    so this one failed on its own precondition ("something before me did it")
    instead of naming the defect."""
    source = (
        "import sys\n"
        f"sys.path.insert(0, {str(REPO_ROOT)!r})\n"
        "import audiocript as A\n"
        "assert 'torch' not in sys.modules, 'importing audiocript brought torch in "
        "on its own, so this check cannot mean anything'\n"
        "A._hf_pipe = object()\n"
        "A.unload_model('tr')\n"
        "print('TORCH_IMPORTED', 'torch' in sys.modules)\n"
    )
    out = subprocess.run([sys.executable, "-c", source],
                         capture_output=True, text=True, cwd=str(REPO_ROOT))
    assert out.returncode == 0, f"the child failed:\n{out.stdout}\n{out.stderr}"
    verdicts = [line.split()[1] for line in out.stdout.splitlines()
                if line.startswith("TORCH_IMPORTED")]
    assert verdicts, f"the child printed no verdict:\n{out.stdout}"
    print(f"  child reported torch in sys.modules: {verdicts[-1]}")
    assert verdicts[-1] == "False", \
        "unload_model imported torch in order to unload a model"


def test_an_unknown_language_is_refused():
    """`_ensure_model` treats every code that is not 'en' as Turkish. Copying that
    here would let `unload_model("rt")` free nothing at all while its caller believes
    the model is gone and reports the daemon as off."""
    with loaded(tr=SENTINEL_TR, en=SENTINEL_EN):
        try:
            A.unload_model("de")
        except ValueError as e:
            message = str(e)
        else:
            raise AssertionError("unload_model('de') was accepted")
        print(f"  refused with: {message}")
        for code in sorted(A.LANGUAGES):
            assert code in message, f"the refusal does not name {code!r}: {message}"
        assert A._hf_pipe is SENTINEL_TR and A._cpp_model is SENTINEL_EN, \
            "a refused unload dropped a model anyway"


if __name__ == "__main__":
    run([n for n in sorted(globals()) if n.startswith("test_")], globals())
