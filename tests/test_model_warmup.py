"""A model that fails to load must leave its reason behind.

The header has one word for every way the load can go wrong: `error`. A missing
package, a download the network refused and a full disk are indistinguishable
there, and the message `_warm_model_async` builds was kept only in memory — so the
first question anyone asks ("why?") had no answer anywhere on disk.

These tests drive the warm-up with a stand-in `_ensure_model`, so a failure can be
asserted on without touching the real model.
"""
import time

from support import run, workdir       # noqa: F401
import audiocript as A


def warm_and_wait(state, language, timeout=5.0):
    """Start the background warm-up and wait for it to settle."""
    A._warm_model_async(state, language)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if state.model_state.get(language) not in (None, "loading"):
            return state.model_state[language]
        time.sleep(0.01)
    raise AssertionError(f"warm-up never finished: {state.model_state!r}")


def fresh_state(home):
    cfg = {"base_path": str(home / "recordings"), "language": "tr",
           "open_app": None, "capture_system_audio": False, "diarize": False}
    return A._TuiState(cfg)


class Stub:
    """Replaces `_ensure_model`, raising what the caller asked for."""

    def __init__(self, exc=None):
        self.exc = exc
        self.calls = []

    def __call__(self, language):
        self.calls.append(language)
        if self.exc is not None:
            raise self.exc


def with_stub(home, exc, language="tr"):
    """Run one warm-up against a stubbed model loader and a throwaway log."""
    log = home / "audiocript.log"
    stub = Stub(exc)
    saved_ensure, saved_log = A._ensure_model, A.LOG_PATH
    A._ensure_model, A.LOG_PATH = stub, log
    try:
        outcome = warm_and_wait(fresh_state(home), language)
    finally:
        A._ensure_model, A.LOG_PATH = saved_ensure, saved_log
    return outcome, log, stub


def test_a_failed_warm_up_writes_its_reason_to_the_log():
    """The defect this file exists for: the header said `error` and the log was silent."""
    with workdir("warm-fail") as home:
        exc = RuntimeError("could not reach huggingface.co")
        outcome, log, _ = with_stub(home, exc)
        assert outcome.startswith("error"), f"the failure was not recorded: {outcome!r}"
        assert log.exists(), "a failed warm-up left nothing in the log"
        text = log.read_text(encoding="utf-8")
        print(f"  logged {len(text.splitlines())} lines")
        assert "model warm-up failed" in text, "the log does not name what failed:\n" + text
        assert "tr" in text, "the log does not say which language:\n" + text
        assert "could not reach huggingface.co" in text, (
            "the reason itself is missing:\n" + text)
        assert "Traceback" in text and "RuntimeError" in text, (
            "the traceback is missing, so the failing call is unknown:\n" + text)


def test_the_log_survives_the_kinds_of_failure_that_are_not_exceptions_about_text():
    """A missing package and a full disk are the two real ones. Both must be readable."""
    for exc in (ModuleNotFoundError("No module named 'pywhispercpp'"),
                OSError(28, "No space left on device")):
        with workdir("warm-kinds") as home:
            _, log, _ = with_stub(home, exc)
            assert log.exists(), f"{type(exc).__name__} left nothing in the log"
            text = log.read_text(encoding="utf-8")
            assert type(exc).__name__ in text, (
                f"{type(exc).__name__} is not in the log:\n" + text)
    print("  ModuleNotFoundError and OSError: both logged")


def test_a_successful_warm_up_writes_nothing():
    """The log is for failures. A working install must not grow it every start."""
    with workdir("warm-ok") as home:
        outcome, log, stub = with_stub(home, None)
        print(f"  outcome: {outcome!r}, model loads: {stub.calls}")
        assert outcome == "ready", f"a clean load did not report ready: {outcome!r}"
        assert not log.exists(), "a successful warm-up wrote to the log:\n" + \
            log.read_text(encoding="utf-8")


def test_the_failure_is_left_retryable():
    """`error` is not `loading` or `ready`, so pressing on can load the model again —
    otherwise one flaky download would need a restart to recover from."""
    with workdir("warm-retry") as home:
        log = home / "audiocript.log"
        stub = Stub(RuntimeError("first attempt"))
        saved_ensure, saved_log = A._ensure_model, A.LOG_PATH
        A._ensure_model, A.LOG_PATH = stub, log
        try:
            state = fresh_state(home)
            warm_and_wait(state, "tr")
            stub.exc = None                       # the network came back
            outcome = warm_and_wait(state, "tr")
        finally:
            A._ensure_model, A.LOG_PATH = saved_ensure, saved_log
        print(f"  attempts: {stub.calls}, outcome: {outcome!r}")
        assert len(stub.calls) == 2, f"the retry never reached the loader: {stub.calls}"
        assert outcome == "ready", f"the retry did not recover: {outcome!r}"


if __name__ == "__main__":
    run(["test_a_failed_warm_up_writes_its_reason_to_the_log",
         "test_the_log_survives_the_kinds_of_failure_that_are_not_exceptions_about_text",
         "test_a_successful_warm_up_writes_nothing",
         "test_the_failure_is_left_retryable"], globals())
