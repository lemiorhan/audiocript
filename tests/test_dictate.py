"""dictate: the state machine that drives one dictation from hotkey to clipboard,
the hotkey that starts it, and the daemon's lifecycle.

No test here opens a microphone, loads a model, reaches a provider or creates a
real event tap — the daemon takes its capture, transcription and delivery as
parameters, and build_listener takes the hotkeys class, for exactly that reason.
The doubles come from support, never from test_dictation: importing a test file
runs its suite and exits, which would look like a pass.
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
          transcribe=None, sink=None):
    """A daemon with every edge faked. Returns (daemon, clipboard, sink, delivered)."""
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
                            deliver=deliver or default_deliver)
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
    assert d.state == dictate.IDLE, f"state {d.state!r} before the first toggle"
    d.toggle()
    assert d.state == dictate.RECORDING, f"state {d.state!r} after one toggle"
    d.toggle()
    d.join_worker(timeout=5)
    assert d.state == dictate.IDLE, f"state {d.state!r} after the cycle"
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
    assert d.state == dictate.IDLE, f"state {d.state!r} after a refused mic"
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
    assert d.state == dictate.IDLE, f"state {d.state!r} after a failed stop"
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
    assert d.state == dictate.IDLE, f"state {d.state!r} after a report blew up"
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
    assert d.state == dictate.IDLE, f"state {d.state!r} after a crashing worker"
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
    assert d.state == dictate.IDLE, f"state {d.state!r} after an uncatchable exit"
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
    while d.state != dictate.PROCESSING and time.time() < deadline:
        time.sleep(0.01)
    assert d.state == dictate.PROCESSING, f"state {d.state!r}"
    before = len(FakeCapture.instances)
    d.toggle()
    assert d.state == dictate.PROCESSING, f"the toggle was not ignored: {d.state!r}"
    assert len(FakeCapture.instances) == before, "a second capture was started"
    assert any(k == "failed" for k, _ in sink.calls), (
        f"the user was not told it was still working: {sink.calls!r}")
    gate.set()
    d.join_worker(timeout=5)


def test_the_duration_bound_stops_the_recording_by_itself():
    d, clip, sink, delivered = build(max_seconds=1)
    d.toggle()
    assert _wait_for(lambda: d.state == dictate.IDLE, 10), \
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

    assert d.state == dictate.IDLE, f"state {d.state!r} after a stale bound"
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

    assert d.state == dictate.RECORDING, \
        f"the stale bound cut the next recording short: {d.state!r}"
    d.toggle()
    d.join_worker(timeout=5)
    assert delivered == ["bir cumle", "bir cumle"], f"delivered {delivered!r}"
    assert not any("sınır" in (text or "") for _, text in sink.calls), \
        f"the stale bound reported a limit: {sink.calls!r}"


# ================== The hotkey, the permission, the lifecycle ==================


class FakeListener:
    """The two health signals check_listener reads off a started listener, with no
    event tap behind either of them."""

    def __init__(self, trusted=True, alive=True):
        self.IS_TRUSTED, self._alive = trusted, alive

    def wait(self): pass
    def is_alive(self): return self._alive


class SpyHotKeys:
    """Captures the mapping build_listener registered, without creating an event
    tap. `registered` is a class attribute so a test can read it without holding
    the listener build_listener returns."""
    registered = {}

    def __init__(self, mapping):
        SpyHotKeys.registered = dict(mapping)
        self.IS_TRUSTED = True

    def start(self): pass
    def wait(self): pass
    def is_alive(self): return True
    def stop(self): pass


def _config(**cfg):
    """A resolved DictationConfig from raw config keys, with a fake environment so
    nothing here reads the developer's own .env."""
    return dictation.resolve_config(cfg, fake_env(OPENAI_API_KEY="sk-test"))


def _registered_callback(config):
    """The callback build_listener actually gave pynput for the toggle hotkey."""
    return SpyHotKeys.registered[config.hotkeys["toggle"]]


def test_a_healthy_listener_raises_no_warning():
    warning = dictate.check_listener(FakeListener())
    assert warning is None, f"a working listener warned anyway: {warning!r}"


def test_a_listener_without_permission_warns_and_names_the_fallback():
    """Without this the daemon looks like it is working while the hotkey silently
    does nothing — indistinguishable from a broken install."""
    warning = dictate.check_listener(FakeListener(trusted=False))
    assert warning, "an untrusted listener did not warn"
    assert "Input Monitoring" in warning, \
        f"the warning does not name the permission: {warning}"
    assert "--toggle" in warning, f"the warning does not name the fallback: {warning}"
    print(f"  {warning}")


def test_a_dead_listener_warns_too():
    """_create_event_tap returning None marks the listener ready and then exits its
    thread, so IS_TRUSTED alone does not cover every failure."""
    warning = dictate.check_listener(FakeListener(alive=False))
    assert warning, "a listener whose thread has exited did not warn"
    print(f"  {warning}")


def test_the_configured_hotkey_is_the_one_registered():
    combination = "<cmd>+<shift>+<alt>+k"
    config = _config(dictation_hotkeys={"toggle": combination})
    dictate.build_listener(config, lambda: None, hotkeys_class=SpyHotKeys)
    assert combination in SpyHotKeys.registered, \
        f"registered {sorted(SpyHotKeys.registered)!r}"
    assert len(SpyHotKeys.registered) == 1, \
        f"registered more than the configured hotkey: {sorted(SpyHotKeys.registered)!r}"


def test_the_default_hotkey_is_registered_when_config_is_silent():
    config = _config()
    dictate.build_listener(config, lambda: None, hotkeys_class=SpyHotKeys)
    assert dictation.DEFAULT_HOTKEYS["toggle"] in SpyHotKeys.registered, \
        f"registered {sorted(SpyHotKeys.registered)!r}"


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


def test_the_listener_callback_does_not_wait_for_the_toggle():
    """macOS disables an event tap whose callback is too slow, and pynput never
    switches it back on: CGEventTapEnable(tap, True) is called once, at listener
    startup, and pynput/_util/darwin.py holds no other reference to it. So a toggle
    run on the callback thread — which can wait 2 x dictation.NOTIFY_TIMEOUT_SECONDS
    on the daemon's lock — kills the hotkey for the rest of the session, with
    IS_TRUSTED still true and the listener still alive."""
    started, release = threading.Event(), threading.Event()

    def blocking_toggle():
        started.set()
        release.wait(5)

    config = _config()
    dictate.build_listener(config, blocking_toggle, hotkeys_class=SpyHotKeys)
    begin = time.monotonic()
    _registered_callback(config)()
    elapsed = time.monotonic() - begin
    ran = started.wait(5)
    release.set()
    assert ran, "the toggle never ran at all"
    assert elapsed < 0.5, f"the callback waited {elapsed:.2f}s for the toggle"
    print(f"  the callback returned in {elapsed * 1000:.1f} ms")


def test_a_toggle_that_raises_does_not_kill_the_hotkey():
    """Daemon._begin_recording's trailing sink.recording() is outside its guard by
    design, so toggle can raise on the recording path. Whatever carries the toggle
    off the callback thread has to survive that — a carrier that dies is the same
    silently dead hotkey as a disabled tap. The second press raises SystemExit,
    which a plain `except Exception` would let through."""
    presses = []
    third = threading.Event()

    def angry_toggle():
        presses.append(len(presses) + 1)
        if len(presses) == 1:
            raise RuntimeError("the recording notification blew up")
        if len(presses) == 2:
            raise SystemExit("something called sys.exit on the toggle's thread")
        third.set()

    config = _config()
    dictate.build_listener(config, angry_toggle, hotkeys_class=SpyHotKeys)
    callback = _registered_callback(config)
    for press in (1, 2, 3):
        try:
            callback()
        except BaseException as e:
            raise AssertionError(
                f"press {press} let {e!r} reach the listener") from None
        assert _wait_for(lambda: len(presses) >= press, 5), \
            f"press {press} was never delivered: {presses!r}"
    assert third.wait(5), "the third press never reached the toggle"
    print(f"  {len(presses)} presses delivered through two exceptions")


def test_shutdown_leaves_no_pid_file_and_no_recording_behind():
    """A pid file that outlives its daemon refuses the next start; a recording that
    outlives it leaves the microphone open and a temporary directory on disk."""
    stopped = []

    class StoppableListener:
        def stop(self): stopped.append("stop")

    with workdir("pid") as d:
        path = d / "dictate.pid"
        dictate.write_pid(path)
        daemon, clip, sink, _ = build()
        daemon.toggle()                        # a recording still going
        dictate._shutdown(StoppableListener(), daemon, path)
        assert stopped == ["stop"], f"the listener was left running: {stopped!r}"
        assert daemon.state == dictate.IDLE, f"state {daemon.state!r} after shutdown"
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
    assert _wait_for(lambda: daemon.state == dictate.PROCESSING, 5), \
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


if __name__ == "__main__":
    run([n for n in sorted(globals()) if n.startswith("test_")], globals())
