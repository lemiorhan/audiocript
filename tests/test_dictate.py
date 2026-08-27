"""dictate: the state machine that drives one dictation from hotkey to clipboard,
the hotkey that starts it, and the daemon's lifecycle.

No test here opens a microphone, loads a model, reaches a provider or draws a menu
bar — the daemon takes its capture, transcription, delivery, model load and model
unload as parameters, for exactly that reason.
The doubles come from support, not test_dictation. Importing a test file is
safe either way: every file here guards its `run(...)` behind
`if __name__ == "__main__":`, which is what makes it importable without
running its suite — test_dictation already relies on this, importing FakeAPI
and completion from test_publish.
"""
import contextlib
import io
import os
import pathlib
import re
import signal
import subprocess
import sys
import threading
import time
import wave

import numpy as np

from support import A, run, fake_env, FakeClipboard, RecordingSink, workdir

import dictate
import dictation


class FakeCapture:
    """Stands in for MicCapture. Records what the daemon asked it to do."""
    instances = []

    def __init__(self, cfg=None, wav="/tmp/never-read.wav", fail=None):
        self.events, self.wav, self.fail = [], pathlib.Path(wav), fail
        FakeCapture.instances.append(self)

    def start(self):
        self.events.append("start")
        if self.fail:
            raise self.fail

    def stop(self):
        self.events.append("stop")
        return self.wav

    def discard(self):
        self.events.append("discard")


class StopFailsCapture(FakeCapture):
    """A capture that opened but cannot produce a WAV — a raw file the resampler
    refuses, say. The recording is lost either way; the daemon must not be."""

    def stop(self):
        self.events.append("stop")
        raise OSError("the raw capture could not be converted")


def build(transcript="bir cumle", deliver=None, capture=None, max_seconds=300,
          transcribe=None, sink=None, power=dictation.POWER_ON,
          ensure_model=None, unload_model=None, history=None):
    """A daemon with every edge faked. Returns (daemon, clipboard, sink, delivered).

    `power` defaults to ON, and gets there through the real `enable()` with the model
    load faked — a daemon that is off refuses every toggle, and almost every test here
    drives a dictation. The sink's record is cleared afterwards, so a case that
    asserts on the exact sequence of reports sees the dictation's moments and not the
    fixture's two power reports. Pass POWER_OFF to test the power axis itself."""
    FakeCapture.instances = []
    clip, sink, delivered = FakeClipboard(), sink or RecordingSink(), []
    config = dictation.resolve_config(
        {"dictation_max_seconds": max_seconds},
        fake_env(OPENAI_API_KEY="sk-test", OPENAI_API_BASE="http://127.0.0.1:1"))

    def default_deliver(text, cfg, clipboard, status_sink, **kw):
        delivered.append(text)
        clipboard.copy(text)
        status_sink.done(text)
        return dictation.Delivery(dictation.CORRECTED, text, "")

    daemon = dictate.Daemon(config, clip, sink, cfg={},
                            capture_factory=capture or FakeCapture,
                            transcribe=transcribe or (lambda wav, lang: transcript),
                            deliver=deliver or default_deliver,
                            history=history,
                            ensure_model=ensure_model or (lambda _language: None),
                            unload_model=unload_model or (lambda _language: None))
    if power == dictation.POWER_ON:
        daemon.enable()
        daemon.join_power(5)
        assert daemon.power == dictation.POWER_ON, \
            f"the fixture could not power the daemon up: {daemon.power!r}"
        sink.calls.clear()
        sink.notes.clear()
    return daemon, clip, sink, delivered


@contextlib.contextmanager
def _stubbed(**attrs):
    """Swap audiocript module attributes for the duration of a block, the way
    tests/test_recording_start.py does. No microphone is opened and no device list
    is enumerated, but everything between MicCapture and the recorder runs for
    real — including _start_mic's own device_name(index) lookup, which asks
    PortAudio about index 7 and, since that need not exist, gets "7" back from its
    own except branch."""
    saved = {name: getattr(A, name) for name in attrs}
    for name, value in attrs.items():
        setattr(A, name, value)
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(A, name, value)


class FakeRecorder:
    """Stands in for DeviceRecorder: writes one second of raw int16 capture as the
    real one streams it, and describes it through the same manifest()."""
    instances = []
    RAW_DTYPE = "int16"
    RATE, CHANNELS = 44100, 1

    def __init__(self, index):
        self.index, self.rate, self.channels = index, self.RATE, self.CHANNELS
        self._raw_path = None
        FakeRecorder.instances.append(self)

    def start(self, raw_path):
        self._raw_path = str(raw_path)
        t = np.arange(self.rate) / self.rate
        pathlib.Path(raw_path).write_bytes(
            (np.sin(2 * np.pi * 300 * t) * 8000).astype(np.int16).tobytes())

    def stop(self):
        pass

    def manifest(self):
        return {"kind": "mic", "file": pathlib.Path(self._raw_path).name,
                "rate": self.rate, "channels": self.channels,
                "dtype": self.RAW_DTYPE}


def _wait_for(predicate, seconds=5):
    """Poll until `predicate` holds; the state machine runs on its own threads."""
    deadline = time.time() + seconds
    while not predicate() and time.time() < deadline:
        time.sleep(0.01)
    return predicate()


def test_a_capture_turns_its_raw_recording_into_a_16_khz_wav():
    """The one thing in this module the state machine's fakes cannot reach: the
    real chain through audiocript._start_mic, _RawPcmReader and
    _stream_source_to_wav. A renamed or re-signed internal there would otherwise
    first be discovered by a dictation the user has already spoken."""
    FakeRecorder.instances = []
    with _stubbed(DeviceRecorder=FakeRecorder, _resolve_mic_index=lambda cfg: 7):
        capture = dictate.MicCapture({})
        capture.start()
        wav = capture.stop()
    assert FakeRecorder.instances[-1].index == 7, "the configured device was not used"
    assert wav.name == dictate.WAV_NAME and wav.is_file(), f"{wav}"
    with wave.open(str(wav)) as wf:
        shape = (wf.getnchannels(), wf.getsampwidth(), wf.getframerate())
        frames = wf.getnframes()
    assert shape == (1, 2, 16000), f"channels/width/rate are {shape}"
    assert abs(frames - 16000) <= 200, f"{frames} frames from one second of capture"
    directory = wav.parent
    capture.discard()
    assert not directory.exists(), f"{directory} outlived the dictation"
    print(f"  {frames} frames at {shape[2]} Hz, then the directory was gone")


def test_discard_never_raises_and_leaves_nothing_behind():
    """Every failure path in the state machine ends in discard(), so a recorder
    that refuses to stop must not take the daemon down with it."""
    class StubbornRecorder(FakeRecorder):
        def stop(self):
            raise OSError("the device is wedged")

    FakeRecorder.instances = []
    with _stubbed(DeviceRecorder=StubbornRecorder, _resolve_mic_index=lambda cfg: 7):
        capture = dictate.MicCapture({})
        capture.start()
        directory = pathlib.Path(FakeRecorder.instances[-1]._raw_path).parent
        try:
            capture.discard()
            capture.discard()         # again, as a second failure path would
        except Exception as e:
            raise AssertionError(f"discard() raised {e!r}") from None
    assert not directory.exists(), f"{directory} outlived a failed dictation"
    print(f"  {directory} was removed despite the recorder refusing to stop")


def test_a_full_cycle_returns_to_idle():
    d, clip, sink, delivered = build()
    assert d.state == dictation.IDLE, f"state {d.state!r} before the first toggle"
    d.toggle()
    assert d.state == dictation.RECORDING, f"state {d.state!r} after one toggle"
    d.toggle()
    d.join_worker(timeout=5)
    assert d.state == dictation.IDLE, f"state {d.state!r} after the cycle"
    assert delivered == ["bir cumle"], f"delivered {delivered!r}"
    assert clip.text == "bir cumle", f"clipboard {clip.text!r}"
    print(f"  {sink.calls}")


def test_the_sink_hears_recording_then_processing():
    d, _, sink, _ = build()
    d.toggle()
    d.toggle()
    d.join_worker(timeout=5)
    kinds = [kind for kind, _ in sink.calls]
    assert kinds[0] == "recording", f"the first report was {kinds!r}"
    assert kinds.index("recording") < kinds.index("processing"), f"{kinds!r}"
    # deliver reports on the sink itself, so exactly one done() may reach it: a
    # worker that reported again would notify the user twice for one dictation.
    assert kinds == ["recording", "processing", "done"], f"{kinds!r}"
    print(f"  {kinds}")


def test_a_microphone_that_cannot_open_returns_to_idle():
    def refused(cfg):
        return FakeCapture(cfg, fail=OSError("device refused"))

    d, clip, sink, _ = build(capture=refused)
    d.toggle()
    assert d.state == dictation.IDLE, f"state {d.state!r} after a refused mic"
    assert any(kind == "failed" for kind, _ in sink.calls), \
        f"the user was not told: {sink.calls!r}"
    assert clip.writes == 0, f"the clipboard was written {clip.writes} times"
    assert "discard" in FakeCapture.instances[-1].events, \
        f"events {FakeCapture.instances[-1].events!r}"
    print(f"  {sink.calls}")


def test_a_capture_that_cannot_be_finished_returns_to_idle():
    """A stop() that raises must not leave the daemon in RECORDING: nothing but
    another toggle could move it, and that toggle would call the same failing
    stop() again."""
    d, clip, sink, delivered = build(capture=StopFailsCapture)
    d.toggle()
    d.toggle()
    assert d.state == dictation.IDLE, f"state {d.state!r} after a failed stop"
    assert any(kind == "failed" for kind, _ in sink.calls), \
        f"the user was not told: {sink.calls!r}"
    assert delivered == [], f"delivered {delivered!r} from a capture with no WAV"
    assert clip.writes == 0, f"the clipboard was written {clip.writes} times"
    assert "discard" in FakeCapture.instances[-1].events, \
        f"events {FakeCapture.instances[-1].events!r}"
    print(f"  {sink.calls}")


def test_stop_without_a_started_recorder_says_which_call_is_missing():
    """MicCapture.stop is public and the state machine shows what it raises to the
    user, so it names the precondition instead of leaving an AttributeError about
    NoneType to stand for it. discard() is the one that must never raise."""
    capture = dictate.MicCapture({})
    try:
        capture.stop()
    except Exception as e:
        assert isinstance(e, RuntimeError), f"stop() raised {e!r}"
        assert "start()" in str(e), f"the message does not name start(): {e}"
        print(f"  {e}")
        return
    raise AssertionError("stop() without a start() did not raise")


def test_a_report_between_the_two_states_cannot_strand_the_daemon():
    """The window between `state = PROCESSING` and a running worker: from there
    only the worker can move the state, so anything raising in it would leave the
    daemon answering "still working" to every hotkey press for the rest of the
    session. Thread.start() under thread exhaustion is the live case; a sink that
    does not swallow its own failures — as NotifySink does — stands in for it."""
    class ExplodingSink(RecordingSink):
        def processing(self):
            RecordingSink.processing(self)
            raise RuntimeError("the notification blew up")

    d, clip, sink, delivered = build(sink=ExplodingSink())
    d.toggle()
    try:
        d.toggle()
    except Exception as e:
        # toggle runs on the thread that carries the hotkey; an exception reaching
        # that caller takes the listener with it.
        raise AssertionError(f"toggle() let {e!r} reach its caller") from None
    d.join_worker(timeout=5)
    assert d.state == dictation.IDLE, f"state {d.state!r} after a report blew up"
    assert "discard" in FakeCapture.instances[-1].events, \
        f"the capture was left behind: {FakeCapture.instances[-1].events!r}"
    assert delivered == [], f"delivered {delivered!r} with no worker running"
    assert clip.writes == 0, f"the clipboard was written {clip.writes} times"
    assert any(kind == "failed" for kind, _ in sink.calls), \
        f"the user was not told: {sink.calls!r}"
    print(f"  {sink.calls}")


def test_a_crashing_worker_returns_to_idle():
    def exploding_deliver(text, cfg, clipboard, status_sink, **kw):
        raise RuntimeError("deliver blew up")

    d, _, sink, _ = build(deliver=exploding_deliver)
    d.toggle()
    d.toggle()
    d.join_worker(timeout=5)
    assert d.state == dictation.IDLE, f"state {d.state!r} after a crashing worker"
    assert any(kind == "failed" for kind, _ in sink.calls), \
        f"the user was not told: {sink.calls!r}"
    print(f"  {sink.calls}")


def test_the_cleanup_survives_what_the_worker_cannot_catch():
    """The cleanup is in a `finally` and not merely after the `except`, so it also
    runs for what `except Exception` does not see — a BaseException, or a status
    report that itself raised. SystemExit stands in for those: threading discards
    it silently, so this test needs no excepthook of its own."""
    def exits(wav, lang):
        raise SystemExit("the interpreter is going down")

    d, _, sink, delivered = build(transcribe=exits)
    d.toggle()
    d.toggle()
    d.join_worker(timeout=5)
    assert d.state == dictation.IDLE, f"state {d.state!r} after an uncatchable exit"
    assert "discard" in FakeCapture.instances[-1].events, \
        f"events {FakeCapture.instances[-1].events!r}"
    assert delivered == [], f"delivered {delivered!r}"
    print(f"  {sink.calls}")


def test_the_capture_is_always_discarded():
    d, _, _, _ = build()
    d.toggle()
    d.toggle()
    d.join_worker(timeout=5)
    capture = FakeCapture.instances[-1]
    assert "discard" in capture.events, f"events {capture.events!r}"
    print(f"  {capture.events}")


def test_a_toggle_while_processing_is_ignored():
    # A second dictation must not start on top of one still being corrected.
    gate = threading.Event()

    def slow_deliver(text, cfg, clipboard, status_sink, **kw):
        gate.wait(5)
        clipboard.copy(text)
        return dictation.Delivery(dictation.CORRECTED, text, "")

    d, _, sink, _ = build(deliver=slow_deliver)
    d.toggle()
    d.toggle()
    deadline = time.time() + 5
    while d.state != dictation.PROCESSING and time.time() < deadline:
        time.sleep(0.01)
    assert d.state == dictation.PROCESSING, f"state {d.state!r}"
    before = len(FakeCapture.instances)
    d.toggle()
    assert d.state == dictation.PROCESSING, f"the toggle was not ignored: {d.state!r}"
    assert len(FakeCapture.instances) == before, "a second capture was started"
    assert any(k == "failed" for k, _ in sink.calls), (
        f"the user was not told it was still working: {sink.calls!r}")
    gate.set()
    d.join_worker(timeout=5)


def test_the_duration_bound_stops_the_recording_by_itself():
    d, clip, sink, delivered = build(max_seconds=1)
    d.toggle()
    assert _wait_for(lambda: d.state == dictation.IDLE, 10), \
        f"the bound did not finish the dictation: state {d.state!r}"
    assert delivered == ["bir cumle"], f"delivered {delivered!r}"
    assert clip.text == "bir cumle", f"clipboard {clip.text!r}"
    assert any("sınır" in (text or "") for _, text in sink.calls), \
        f"no report said the limit was reached: {sink.calls!r}"
    print(f"  {sink.calls}")


def test_a_stale_duration_timer_cannot_start_a_recording():
    """The race Timer.cancel() cannot win: a bound whose thread has already left
    its wait when the user presses the hotkey. By the time it runs the recording
    is over, and it must do nothing at all — not start one nobody asked for."""
    d, clip, sink, _ = build()
    d.toggle()
    epoch = d._epoch                      # the recording that timer belongs to
    d.toggle()
    d.join_worker(timeout=5)
    before, calls = len(FakeCapture.instances), len(sink.calls)

    d._on_duration_bound(epoch)           # the timer thread, arriving late

    assert d.state == dictation.IDLE, f"state {d.state!r} after a stale bound"
    assert len(FakeCapture.instances) == before, "the stale bound started a recording"
    assert len(sink.calls) == calls, f"the stale bound reported: {sink.calls[calls:]!r}"


def test_a_stale_duration_timer_cannot_stop_the_next_recording():
    """Same race, one step later: the user stopped and started again before the
    late bound ran. It belongs to the finished recording, not to this one."""
    d, _, sink, delivered = build()
    d.toggle()
    epoch = d._epoch
    d.toggle()
    d.join_worker(timeout=5)
    d.toggle()                            # a second recording, still going

    d._on_duration_bound(epoch)           # the first recording's timer, late

    assert d.state == dictation.RECORDING, \
        f"the stale bound cut the next recording short: {d.state!r}"
    d.toggle()
    d.join_worker(timeout=5)
    assert delivered == ["bir cumle", "bir cumle"], f"delivered {delivered!r}"
    assert not any("sınır" in (text or "") for _, text in sink.calls), \
        f"the stale bound reported a limit: {sink.calls!r}"


# ================== The hotkey, the permission, the lifecycle ==================


def _config(**cfg):
    """A resolved DictationConfig from raw config keys, with a fake environment so
    nothing here reads the developer's own .env."""
    return dictation.resolve_config(cfg, fake_env(OPENAI_API_KEY="sk-test"))


def test_the_pid_file_round_trips():
    with workdir("pid") as d:
        path = d / "dictate.pid"
        dictate.write_pid(path)
        assert dictate.read_pid(path) == os.getpid(), \
            f"read_pid returned {dictate.read_pid(path)!r}, not {os.getpid()}"


def test_a_stale_pid_file_reads_as_absent_and_does_not_block():
    """A daemon killed with SIGKILL leaves its pid file behind. If that blocked the
    next start, the feature would be dead until the user found the file."""
    with workdir("pid") as d:
        path = d / "dictate.pid"
        path.write_text("999999")            # past macOS's pid ceiling: never live
        assert dictate.read_pid(path) is None, "a pid nothing is running read as live"
        dictate.write_pid(path)
        assert dictate.read_pid(path) == os.getpid(), \
            f"the stale file was not taken over: {dictate.read_pid(path)!r}"


def test_a_live_pid_file_blocks_a_second_daemon():
    with workdir("pid") as d:
        path = d / "dictate.pid"
        dictate.write_pid(path)
        try:
            dictate.write_pid(path)
        except Exception as e:
            assert str(os.getpid()) in str(e), \
                f"the refusal does not name the daemon that holds it: {e}"
            print(f"  {e}")
            return
        raise AssertionError("a second daemon wrote over a live pid file")


def test_only_one_of_several_daemons_started_at_once_claims_the_pid_file():
    """Read-then-write let every daemon started in the same instant pass the
    liveness check before any of them wrote, so all of them believed they owned
    the file. The damage is not the duplicate: it is that the loser's own
    `_shutdown` unlinks the *winner's* pid file on the way out, after which a
    daemon is still running and `--toggle` and `--stop` can no longer find it.

    Real processes, because real pids are the whole point — threads would share
    ours and read as live, and every child stays alive to the end, so a loser
    probing the winner always finds a live owner.

    Several rounds, not one: the window the old code loses is a few microseconds
    wide, so a single round catches read-then-write only about half the time
    (measured). Each round is its own barrier and its own path, and the children
    busy-spin on the go marker so they arrive together. Correct code cannot fail
    this — `os.link` admits exactly one, and every child outlives every round, so a
    loser's probe always finds a live owner — while read-then-write has to survive
    every round to pass."""
    workers, rounds = 4, 8
    repo = str(pathlib.Path(dictate.__file__).resolve().parent)
    child = f"""
import sys, time, pathlib
sys.path.insert(0, {repo!r})
import dictate
d, n, rounds = pathlib.Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
(d / ("ready-" + n)).write_text("")
answers = []
for r in range(rounds):
    while not (d / ("go-%d" % r)).exists():
        time.sleep(0)                    # yield, do not oversleep the window
    try:
        dictate.write_pid(d / ("pid-%d" % r))
    except Exception:
        answers.append("REFUSED")
    else:
        answers.append("CLAIMED")
    (d / ("done-%d-%s" % (r, n))).write_text("")
print(" ".join(answers))
"""
    with workdir("pid") as d:
        started = [subprocess.Popen(
            [sys.executable, "-c", child, str(d), str(n), str(rounds)],
            stdout=subprocess.PIPE, text=True) for n in range(workers)]
        try:
            assert _wait_for(lambda: len(list(d.glob("ready-*"))) == workers, 60), \
                f"only {len(list(d.glob('ready-*')))} of {workers} children started"
            for r in range(rounds):
                (d / f"go-{r}").write_text("")
                assert _wait_for(
                    lambda: len(list(d.glob(f"done-{r}-*"))) == workers, 60), \
                    f"round {r}: only {len(list(d.glob(f'done-{r}-*')))} answered"
            answers = [p.communicate(timeout=60)[0].split() for p in started]
        finally:
            for p in started:
                if p.poll() is None:
                    p.kill()
    for r in range(rounds):
        got = [a[r] for a in answers if len(a) > r]
        assert len(got) == workers, f"round {r}: {len(got)} of {workers} reported"
        assert got.count("CLAIMED") == 1, \
            f"round {r}: {got.count('CLAIMED')} of {workers} claimed it: {got!r}"
    print(f"  {rounds} rounds x {workers} simultaneous starts, one winner each")


def test_the_launcher_flag_matches_what_signal_daemon_names():
    """run.sh's --dictate switch and signal_daemon's "no daemon is running"
    message are two independent spellings of the same flag, and nothing else
    pins them together — a rename on either side would go silently stale.
    Extracting both and comparing catches a change made to only one of them."""
    run_sh = pathlib.Path(dictate.__file__).resolve().parent / "run.sh"
    text = run_sh.read_text()
    match = re.search(r'"\$\{1:-\}"\s*=\s*"(--[\w-]+)"', text)
    assert match, f"run.sh's launcher switch was not found in {run_sh}"
    launcher_flag = match.group(1)

    with workdir("pid") as d:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            dictate.signal_daemon("toggle", d / "absent.pid")
    message = out.getvalue()
    assert launcher_flag in message, (
        f"run.sh consumes {launcher_flag!r} but the daemon's message says "
        f"{message.strip()!r}")
    print(f"  run.sh consumes {launcher_flag!r}, the daemon's message names it too")


def test_toggle_with_no_daemon_exits_non_zero():
    """Exiting zero would make a Shortcuts binding look like it worked."""
    with workdir("pid") as d:
        status = dictate.signal_daemon("toggle", d / "absent.pid")
        assert status != 0, "signalling a daemon that is not running exited zero"


def test_each_action_sends_its_own_signal():
    """A --toggle that sent SIGTERM would kill the daemon the user only meant to
    talk to, and a --stop that sent SIGUSR1 would start a dictation instead of
    stopping anything. Both signals are handled here so the test process receives
    them harmlessly, and the handlers are restored afterwards."""
    received = []
    saved = {s: signal.getsignal(s) for s in (signal.SIGUSR1, signal.SIGTERM)}
    try:
        for s in saved:
            signal.signal(s, lambda num, frame: received.append(num))
        with workdir("pid") as d:
            path = d / "dictate.pid"
            dictate.write_pid(path)
            for action, expected in (("toggle", signal.SIGUSR1),
                                     ("stop", signal.SIGTERM)):
                received.clear()
                assert dictate.signal_daemon(action, path) == 0, \
                    f"{action} exited non-zero against a live daemon"
                assert _wait_for(lambda: received == [expected], 2), \
                    f"{action} sent {received!r}, expected [{int(expected)}]"
    finally:
        for s, handler in saved.items():
            signal.signal(s, handler)
    print("  toggle sent SIGUSR1, stop sent SIGTERM")


def test_the_signal_handler_does_not_wait_for_the_toggle():
    """The SIGUSR1 handler runs on the main thread, which is the AppKit run loop. A
    toggle run there can hold the daemon's lock for as long as a WAV conversion or a
    microphone retry takes — not bounded by dictation.NOTIFY_TIMEOUT_SECONDS — and
    that is the thread drawing the menu bar. It also must not be re-entered: a second
    --toggle arriving inside the first would deadlock on a lock its own thread already
    holds."""
    started, release = threading.Event(), threading.Event()

    def blocking_toggle():
        started.set()
        release.wait(5)

    handler = dictate._deferred(blocking_toggle, "test-signal")
    begin = time.monotonic()
    handler(signal.SIGUSR1, None)          # exactly how a signal handler is called
    elapsed = time.monotonic() - begin
    ran = started.wait(5)
    release.set()
    assert ran, "the toggle never ran at all"
    assert elapsed < 0.5, f"the handler waited {elapsed:.2f}s for the toggle"
    print(f"  the handler returned in {elapsed * 1000:.1f} ms")


def test_a_toggle_that_raises_does_not_kill_the_pump():
    """Daemon._begin_recording's trailing sink.recording() is outside its guard by
    design, so toggle can raise on the recording path. The thread carrying the toggle
    off the signal handler has to survive that: a pump that dies leaves --toggle
    silently dead for the rest of the session, with nothing on screen and nothing in
    any log. The second request raises SystemExit, which a plain `except Exception`
    would let through."""
    presses = []
    third = threading.Event()

    def angry_toggle():
        presses.append(len(presses) + 1)
        if len(presses) == 1:
            raise RuntimeError("the recording notification blew up")
        if len(presses) == 2:
            raise SystemExit("something called sys.exit on the toggle's thread")
        third.set()

    submit = dictate._deferred(angry_toggle, "test-pump")
    for press in (1, 2, 3):
        try:
            submit(signal.SIGUSR1, None)
        except BaseException as e:
            raise AssertionError(
                f"request {press} let {e!r} reach the signal handler") from None
        assert _wait_for(lambda: len(presses) >= press, 5), \
            f"press {press} was never delivered: {presses!r}"
    assert third.wait(5), "the third press never reached the toggle"
    print(f"  {len(presses)} presses delivered through two exceptions")


def test_shutdown_leaves_no_pid_file_and_no_recording_behind():
    """A pid file that outlives its daemon refuses the next start; a recording that
    outlives it leaves the microphone open and a temporary directory on disk."""
    with workdir("pid") as d:
        path = d / "dictate.pid"
        dictate.write_pid(path)
        daemon, clip, sink, _ = build()
        daemon.toggle()                        # a recording still going
        dictate._shutdown(daemon, path)
        assert daemon.state == dictation.IDLE, f"state {daemon.state!r} after shutdown"
        assert "discard" in FakeCapture.instances[-1].events, \
            f"the recording was left behind: {FakeCapture.instances[-1].events!r}"
        assert not path.exists(), f"{path} outlived the daemon"
        assert clip.writes == 0, f"the clipboard was written {clip.writes} times"
    print(f"  {sink.calls}")


def test_shutdown_does_not_take_the_capture_away_from_a_running_worker():
    """During PROCESSING the worker owns the capture and discards it in its own
    finally. Discarding it here would delete the WAV out from under the
    transcription of a dictation the user has already spoken."""
    gate = threading.Event()

    def slow_deliver(text, cfg, clipboard, status_sink, **kw):
        gate.wait(5)
        clipboard.copy(text)
        return dictation.Delivery(dictation.CORRECTED, text, "")

    daemon, clip, _, _ = build(deliver=slow_deliver)
    daemon.toggle()
    daemon.toggle()
    assert _wait_for(lambda: daemon.state == dictation.PROCESSING, 5), \
        f"state {daemon.state!r}"
    capture = FakeCapture.instances[-1]

    daemon.abandon_recording()

    assert "discard" not in capture.events, \
        f"the worker's capture was discarded under it: {capture.events!r}"
    gate.set()
    daemon.join_worker(timeout=5)
    assert clip.text == "bir cumle", f"clipboard {clip.text!r}"
    assert "discard" in capture.events, \
        f"the worker did not clean up: {capture.events!r}"


def test_no_arguments_runs_the_daemon():
    """The primary entry point, and the one thing here that cannot be tried for
    real in a test: _run_daemon takes the pid file, loads a model and puts up an
    event tap. So the branch is what is pinned — without it `dictate.py` with no
    arguments prints the usage and exits 2, which is a feature that does not
    start. _run_daemon is replaced for the duration; nothing of it runs."""
    called, saved = [], dictate._run_daemon
    try:
        dictate._run_daemon = lambda: (called.append("ran"), 3)[1]
        status = dictate.main([])
    finally:
        dictate._run_daemon = saved
    assert called == ["ran"], "main([]) never reached _run_daemon"
    assert status == 3, f"main([]) returned {status!r}, not _run_daemon's status"
    print("  main([]) -> _run_daemon()")


def test_an_unknown_argument_is_refused():
    """It must not fall through to the daemon path: a typo would then start a
    daemon, load a model and take the pid file."""
    status = dictate.main(["--halt"])
    assert status != 0, "an unknown argument exited zero"


def test_every_action_with_a_signal_has_a_flag_that_reaches_it():
    """The one piece of wiring between the CLI and the signals, and nothing else
    covers it: signal_daemon looks its action up in ACTION_SIGNALS only after
    read_pid, so a flag forwarded with its dashes still on exits non-zero — with a
    message blaming a daemon that is not running — whenever none is running, and
    raises KeyError when one is.

    Driven off ACTION_SIGNALS rather than off a list written out here, so an action
    given a signal but no way in from the CLI fails this instead of shipping.
    signal_daemon is replaced for the duration so no signal is sent and the real
    pid file is never read."""
    asked, saved = [], dictate.signal_daemon
    try:
        dictate.signal_daemon = lambda action, *a, **kw: (asked.append(action), 7)[1]
        for action in dictate.ACTION_SIGNALS:
            status = dictate.main([f"--{action}"])
            assert status == 7, \
                f"--{action} returned {status!r}, not what signal_daemon gave it"
            assert asked[-1] == action, f"--{action} asked for {asked[-1]!r}"
    finally:
        dictate.signal_daemon = saved
    assert asked == list(dictate.ACTION_SIGNALS), \
        f"{len(dictate.ACTION_SIGNALS)} actions have signals, {asked!r} got there"
    print(f"  {asked}")


# =============================== the power axis ===============================


class Loader:
    """Stands in for audiocript._ensure_model with the timing under the test's
    control. `block=True` holds the load open until release() is called, so LOADING
    can be observed rather than raced; `fail` is settable afterwards, so one loader
    can fail and then succeed."""

    def __init__(self, fail=None, block=False):
        self.calls = []
        self.fail = fail
        self.entered = threading.Event()
        self._release = threading.Event() if block else None

    def __call__(self, language):
        self.calls.append(language)
        self.entered.set()
        if self._release is not None:
            assert self._release.wait(10), "the test never released the load"
        if self.fail is not None:
            raise self.fail

    def release(self):
        self._release.set()


def build_off(loader=None, unload_model=None, **kw):
    """A daemon that has not been powered up. Returns (daemon, sink, loader, unloaded).

    `unloaded` records the languages the default unload was asked for; a case that
    needs the unload to block passes its own `unload_model` and ignores it."""
    loader = loader or Loader()
    unloaded = []
    daemon, _clip, sink, _delivered = build(power=dictation.POWER_OFF,
                                            ensure_model=loader,
                                            unload_model=unload_model or unloaded.append,
                                            **kw)
    return daemon, sink, loader, unloaded


def _powers(sink):
    return [value for kind, value in sink.calls if kind == "power_changed"]


def _reasons(sink):
    return [text for kind, text in sink.calls if kind == "failed"]


def test_enable_reaches_on_through_loading():
    daemon, sink, loader, _ = build_off(loader=Loader(block=True))
    assert daemon.power == dictation.POWER_OFF, \
        f"a new daemon starts at {daemon.power!r}, not off"
    daemon.enable()
    assert loader.entered.wait(5), "the load never started"
    assert daemon.power == dictation.POWER_LOADING, \
        f"power is {daemon.power!r} while the model is still loading"
    loader.release()
    daemon.join_power(5)
    assert daemon.power == dictation.POWER_ON, \
        f"power is {daemon.power!r} after the load returned"
    assert _powers(sink) == [dictation.POWER_LOADING, dictation.POWER_ON], \
        f"the sink saw {_powers(sink)!r}, in that order"
    assert loader.calls == ["tr"], f"the model was loaded for {loader.calls!r}"
    print(f"  {_powers(sink)}")


def test_enable_does_not_block_the_caller():
    """It is called from the AppKit main thread, which has to keep drawing the menu
    bar while the model loads — measured at 11.3s cold and 4.1s warm."""
    daemon, _sink, loader, _ = build_off(loader=Loader(block=True))
    started = time.monotonic()
    daemon.enable()
    elapsed = time.monotonic() - started
    assert loader.entered.wait(5), "the load never started"
    assert elapsed < 1.0, \
        f"enable() took {elapsed:.2f}s while the load was still blocked"
    loader.release()
    daemon.join_power(5)
    print(f"  enable() returned in {elapsed * 1000:.0f} ms, load still blocked")


def test_a_load_that_raises_returns_to_off_and_is_retryable():
    """A daemon that cannot come back from a network blip is dead to the user: the
    icon would say off and no click would ever change it."""
    logged = []
    saved = A._log_problem
    A._log_problem = lambda what, exc=None: logged.append((what, exc))
    try:
        loader = Loader(fail=RuntimeError("could not reach huggingface.co"))
        daemon, sink, _loader, unloaded = build_off(loader=loader)
        daemon.enable()
        daemon.join_power(5)
        assert daemon.power == dictation.POWER_OFF, \
            f"a failed load left power at {daemon.power!r}"
        assert any("huggingface" in (text or "") for text in _reasons(sink)), \
            f"the reason did not reach the user: {sink.calls!r}"
        assert logged, "the failure was not logged"
        assert unloaded == [], f"a load that never finished unloaded anyway: {unloaded}"
        loader.fail = None
        daemon.enable()
        daemon.join_power(5)
        assert daemon.power == dictation.POWER_ON, \
            f"a second enable() after a failure reached {daemon.power!r}"
    finally:
        A._log_problem = saved
    print(f"  failed, reported, and came back: {_powers(sink)}")


def test_enable_is_idempotent():
    """Two clicks on "Daemon'ı başlat", or a click while the model is loading, must
    not start a second load — _ensure_model would serialise them on its own lock and
    the second would be pure waiting."""
    daemon, _sink, loader, _ = build_off(loader=Loader(block=True))
    daemon.enable()
    assert loader.entered.wait(5), "the load never started"
    daemon.enable()                                   # while LOADING
    loader.release()
    daemon.join_power(5)
    daemon.enable()                                   # while ON
    daemon.join_power(5)
    assert loader.calls == ["tr"], \
        f"the model was loaded {len(loader.calls)} times: {loader.calls!r}"


def test_disable_unloads_the_configured_language():
    daemon, sink, _loader, unloaded = build_off()
    daemon.enable()
    daemon.join_power(5)
    daemon.disable()
    assert daemon.power == dictation.POWER_OFF, \
        f"disable() left power at {daemon.power!r}"
    assert unloaded == ["tr"], \
        f"unload_model was called with {unloaded!r}, not the configured language"
    assert _powers(sink)[-1] == dictation.POWER_OFF, _powers(sink)
    print(f"  {_powers(sink)}, unloaded {unloaded}")


def test_disable_is_refused_while_recording():
    """The rule the user asked for: the daemon cannot be stopped before the recording
    ends. Enforced here rather than only by a greyed-out menu item, so it holds for
    the CLI and for any other caller too."""
    daemon, sink, _loader, unloaded = build_off()
    daemon.enable()
    daemon.join_power(5)
    daemon.toggle()
    assert daemon.state == dictation.RECORDING, f"state {daemon.state!r}"
    daemon.disable()
    assert daemon.power == dictation.POWER_ON, \
        f"the daemon was stopped mid-recording: power {daemon.power!r}"
    assert unloaded == [], f"the model was unloaded mid-recording: {unloaded!r}"
    assert _reasons(sink), f"the user was not told why: {sink.calls!r}"
    daemon.toggle()
    daemon.join_worker(timeout=5)
    print(f"  refused with: {_reasons(sink)[-1]!r}")


def test_disable_is_refused_while_processing():
    """The worse half of the same rule: here the text has already been spoken and is
    being transcribed, so stopping would throw away work the user is waiting for."""
    gate = threading.Event()

    def slow_deliver(text, cfg, clipboard, status_sink, **kw):
        gate.wait(5)
        clipboard.copy(text)
        return dictation.Delivery(dictation.CORRECTED, text, "")

    daemon, sink, _loader, unloaded = build_off(deliver=slow_deliver)
    daemon.enable()
    daemon.join_power(5)
    daemon.toggle()
    daemon.toggle()
    assert _wait_for(lambda: daemon.state == dictation.PROCESSING, 5), \
        f"state {daemon.state!r}"
    daemon.disable()
    assert daemon.power == dictation.POWER_ON, \
        f"the daemon was stopped mid-transcription: power {daemon.power!r}"
    assert unloaded == [], f"the model was unloaded under the worker: {unloaded!r}"
    assert _reasons(sink), f"the user was not told why: {sink.calls!r}"
    gate.set()
    daemon.join_worker(timeout=5)


def test_disable_is_idempotent():
    daemon, _sink, _loader, unloaded = build_off()
    daemon.disable()
    daemon.disable()
    assert unloaded == [], f"a daemon that was never up unloaded {unloaded!r}"
    assert daemon.power == dictation.POWER_OFF, daemon.power


def test_toggle_is_refused_while_the_daemon_is_down():
    """Both doors, both powers. The menu greys its item out, but SIGUSR1 from
    `dictate.py --toggle` reaches toggle() directly and must be told the same thing
    instead of opening a microphone with no model behind it."""
    for power, arrange in (("off", lambda d, l: None),
                           ("loading", lambda d, l: (d.enable(),
                                                     l.entered.wait(5)))):
        loader = Loader(block=True)
        daemon, sink, _loader, _ = build_off(loader=loader)
        arrange(daemon, loader)
        before = len(FakeCapture.instances)
        daemon.toggle()
        assert len(FakeCapture.instances) == before, \
            f"a microphone was opened with the daemon {power}"
        assert daemon.state == dictation.IDLE, \
            f"state {daemon.state!r} after a refused toggle ({power})"
        assert _reasons(sink), f"the user was not told ({power}): {sink.calls!r}"
        loader.release()
        daemon.join_power(5)
    print("  refused from off and from loading, both reported")


def test_power_off_implies_idle():
    """The invariant the rest of the code is allowed to assume: there is no way to
    reach OFF while a dictation is in flight, because disable() refuses. Nothing has
    to handle "off while recording"."""
    daemon, _sink, _loader, _ = build_off()
    daemon.enable()
    daemon.join_power(5)
    daemon.toggle()
    daemon.disable()                     # refused: RECORDING
    assert daemon.power == dictation.POWER_ON, daemon.power
    daemon.toggle()
    daemon.join_worker(timeout=5)
    assert daemon.state == dictation.IDLE, daemon.state
    daemon.disable()                     # accepted: IDLE
    assert (daemon.power, daemon.state) == (dictation.POWER_OFF, dictation.IDLE), \
        f"({daemon.power!r}, {daemon.state!r})"


def test_a_disable_during_the_load_does_not_leave_the_model_resident():
    """The race the two axes make possible: the user clicks start, changes their mind
    while the model is still loading, and the load then finishes into a daemon whose
    icon says off. Without this the 3 GB stays resident behind an off icon, and the
    next enable() would find _ensure_model already satisfied and never reload."""
    loader = Loader(block=True)
    daemon, sink, _loader, unloaded = build_off(loader=loader)
    daemon.enable()
    assert loader.entered.wait(5), "the load never started"
    daemon.disable()
    assert daemon.power == dictation.POWER_OFF, \
        f"disable() during a load left power at {daemon.power!r}"
    loader.release()
    daemon.join_power(5)
    assert daemon.power == dictation.POWER_OFF, \
        f"the finished load powered the daemon up behind the user: {daemon.power!r}"
    assert unloaded == ["tr"], (
        f"the model was unloaded {len(unloaded)} times ({unloaded!r}); exactly one "
        "drop belongs here, from _load's stale path — disable() must not take "
        "A._MODEL_LOCK while the load in flight is holding it, or it blocks the "
        "AppKit main thread for the rest of the load")
    assert _powers(sink)[-1] == dictation.POWER_OFF, _powers(sink)
    print(f"  {_powers(sink)}, unloaded {unloaded}")


def test_disable_does_not_wait_for_a_load_in_flight():
    """A.unload_model takes A._MODEL_LOCK, and the load in flight is holding it. A
    disable() that dropped the model itself would therefore block until the load
    finished — on the AppKit main thread, for the 11.3s a cold load takes. The drop
    belongs to _load's stale path instead, on the loader's own thread.

    The count in the test above is the visible half of this; this case pins the timing
    directly, with an unload that blocks if it is ever reached from here."""
    loader = Loader(block=True)
    reached_unload = threading.Event()
    release_unload = threading.Event()

    def blocking_unload(_language):
        reached_unload.set()
        assert release_unload.wait(10), "the test never released the unload"

    daemon, _sink, _loader, _unloaded = build_off(loader=loader,
                                                 unload_model=blocking_unload)
    daemon.enable()
    assert loader.entered.wait(5), "the load never started"
    started = time.monotonic()
    daemon.disable()
    elapsed = time.monotonic() - started
    assert not reached_unload.is_set(), \
        "disable() dropped the model while the load in flight held A._MODEL_LOCK"
    assert elapsed < 1.0, f"disable() took {elapsed:.2f}s during a load"
    print(f"  disable() returned in {elapsed * 1000:.0f} ms, load still blocked")
    loader.release()
    release_unload.set()
    daemon.join_power(5)


# ============================== the history log ==============================


class FakeHistory:
    """Records what the daemon asked to be logged. `fail` makes append raise, which
    the real History never does — it swallows — so only a broken one gets here."""

    def __init__(self, fail=None):
        self.entries, self.fail = [], fail

    def append(self, text, status):
        if self.fail is not None:
            raise self.fail
        self.entries.append((text, status))


def _delivering(status, text):
    """A deliver double that behaves the way the real one does for `status`: copies
    and reports done when there is text, reports failed when there is not."""
    def deliver(_transcript, _cfg, clipboard, sink, **_kw):
        if text:
            clipboard.copy(text)
            sink.done(text)
        else:
            sink.failed("no speech was detected")
        return dictation.Delivery(status, text, "")
    return deliver


def test_every_delivery_status_is_logged_by_the_same_rule():
    """One rule for all of them: what reached the clipboard is what is logged, and
    EMPTY is the only status that reaches it with nothing.

    Scanned over dictation.DELIVERY_STATUSES rather than naming the four, because a
    fifth status added to the pipeline would otherwise be silently logged or silently
    dropped with no test noticing — and the guard in _work is a test on
    delivery.text, not a list of statuses, for exactly the same reason."""
    for status in dictation.DELIVERY_STATUSES:
        text = "" if status == dictation.EMPTY else f"bir cumle ({status})"
        log = FakeHistory()
        daemon, clip, _sink, _ = build(deliver=_delivering(status, text), history=log)
        daemon.toggle()
        daemon.toggle()
        daemon.join_worker(timeout=5)
        expected = [(text, status)] if text else []
        assert log.entries == expected, \
            f"status {status!r} logged {log.entries!r}, expected {expected!r}"
        if text:
            assert clip.text == text, f"status {status!r} left {clip.text!r} on the clipboard"
    print(f"  {len(dictation.DELIVERY_STATUSES)} statuses, one rule: "
          f"{[s for s in dictation.DELIVERY_STATUSES if s != dictation.EMPTY]} logged")


def test_a_history_failure_does_not_cost_the_dictation():
    """The log is the least important thing happening. By the time it is written the
    text is on the clipboard and the user has been told — so a failure here must not
    turn a dictation that worked into a "processing failed" report, and must not
    strand the daemon."""
    logged = []
    saved = A._log_problem
    A._log_problem = lambda what, exc=None: logged.append((what, exc))
    try:
        daemon, clip, sink, _ = build(history=FakeHistory(fail=OSError("disk full")))
        daemon.toggle()
        daemon.toggle()
        daemon.join_worker(timeout=5)
    finally:
        A._log_problem = saved
    kinds = [kind for kind, _ in sink.calls]
    assert daemon.state == dictation.IDLE, f"state {daemon.state!r}"
    assert clip.text == "bir cumle", f"the clipboard holds {clip.text!r}"
    assert "done" in kinds, f"the user was not told it worked: {sink.calls!r}"
    assert "failed" not in kinds, \
        f"a log failure was reported as the dictation failing: {sink.calls!r}"
    assert logged, "the log failure was swallowed without a trace"
    print(f"  reported done, logged the problem: {logged[0][0]}")


def test_a_daemon_with_no_history_still_dictates():
    """history is optional: the CLI path and every test that predates the log pass
    nothing, and a dictation must not depend on there being somewhere to record it."""
    daemon, clip, sink, delivered = build(history=None)
    daemon.toggle()
    daemon.toggle()
    daemon.join_worker(timeout=5)
    assert delivered == ["bir cumle"], f"delivered {delivered!r}"
    assert daemon.state == dictation.IDLE, f"state {daemon.state!r}"


if __name__ == "__main__":
    run([n for n in sorted(globals()) if n.startswith("test_")], globals())
