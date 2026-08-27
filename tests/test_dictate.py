"""dictate: the state machine that drives one dictation from hotkey to clipboard.

No test here opens a microphone, loads a model, or reaches a provider — the
daemon takes its capture, transcription and delivery as parameters for exactly
that reason. The doubles come from support, never from test_dictation: importing
a test file runs its suite and exits, which would look like a pass.
"""
import contextlib
import pathlib
import threading
import time
import wave

import numpy as np

from support import A, run, fake_env, FakeClipboard, RecordingSink

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
          transcribe=None):
    """A daemon with every edge faked. Returns (daemon, clipboard, sink, delivered)."""
    FakeCapture.instances = []
    clip, sink, delivered = FakeClipboard(), RecordingSink(), []
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
    tests/test_recording_start.py does: no microphone is opened and no device list
    is queried, but everything between MicCapture and the recorder runs for real."""
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


if __name__ == "__main__":
    run([n for n in sorted(globals()) if n.startswith("test_")], globals())
