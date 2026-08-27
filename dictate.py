"""The parts of dictation that hold threads and state: the microphone capture and
the state machine that drives one dictation from hotkey to clipboard.

dictation.py holds everything that can be tested without either — configuration,
the clipboard, the status sinks and the correction pipeline. Here the state is
touched from three threads (the hotkey listener, the duration timer and the
worker), so `toggle` is the only way in and one lock guards `state`.
"""
import pathlib
import shutil
import tempfile
import threading

import audiocript as A
import dictation

IDLE, RECORDING, PROCESSING = "idle", "recording", "processing"

# The WAV a capture converts itself into, inside its own temporary directory.
WAV_NAME = "dictation.wav"
RAW_NAME = "mic.raw"


def _announce(_message):
    """Where audiocript._start_mic's progress notes go: nowhere.

    They are written for a status line the daemon does not have, and the one thing
    in them worth keeping is already kept — _start_mic itself sends every refused
    attempt, with its traceback and the device index, to _log_problem. Routing
    these here as well would add a line to the log for every dictation ever made
    and say nothing new.

    Dropping them here does not make them free: _start_mic builds the first note's
    text before it calls this, and that text holds device_name(index), which asks
    PortAudio for the device. So every dictation pays one exception-safe device
    lookup for a string thrown away here. Nothing this side of the call can avoid
    it — the f-string is evaluated inside audiocript, which this plan does not
    modify — and the cost is a read of a device table built at import."""


class _MicState:
    """The two attributes audiocript._start_mic reads off its `state` argument:
    the resolved device index — which it rewrites when a retry re-resolves the
    configured device — and the config dict it re-resolves from. Building a real
    _TuiState would drag the whole TUI into the daemon for two fields."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.mic_index = A._resolve_mic_index(cfg)


class MicCapture:
    """Owns one recording's temporary directory and microphone.

    Nothing survives a dictation: the raw capture and the WAV live in a directory
    of their own that `discard` removes, so a dictation leaves no folder in the
    recordings library and no file to clean up later."""

    def __init__(self, cfg):
        self._cfg = cfg or {}
        self._dir = None
        self._recorder = None

    def start(self):
        """Open the microphone into a fresh temporary directory.

        Goes through audiocript._start_mic rather than DeviceRecorder directly for
        its retry loop: CoreAudio turns down a device that is busy or still
        settling, and two real recordings were lost to a refusal that passed in a
        second."""
        self._dir = pathlib.Path(tempfile.mkdtemp(prefix="audiocript-dictation-"))
        self._recorder = A._start_mic(_MicState(self._cfg),
                                      self._dir / RAW_NAME, _announce)

    def stop(self):
        """Stop the recorder and return the path of a 16 kHz mono WAV of what it
        captured, converted a block at a time straight off disk.

        Valid only after a start() that returned. Unlike discard, this one does
        raise — the state machine puts whatever it raises in front of the user, and
        `RuntimeError: stop() without ...` is something a reader can act on where
        `AttributeError: 'NoneType' object has no attribute 'stop'` is not."""
        if self._recorder is None:
            raise RuntimeError("stop() without a recorder: start() has to have "
                               "returned first")
        recorder, self._recorder = self._recorder, None
        recorder.stop()
        # manifest() already describes the raw file the way it has to be re-read;
        # assembling the rate/channels/dtype by hand here would be a second, and
        # eventually disagreeing, copy of that.
        manifest = recorder.manifest()
        raw_path = self._dir / manifest["file"]
        wav_path = self._dir / WAV_NAME
        with A._RawPcmReader(raw_path, manifest["dtype"],
                             manifest["channels"]) as reader:
            A._stream_source_to_wav(reader, manifest["rate"], wav_path)
        return wav_path

    def discard(self):
        """Stop a running recorder and remove the directory. Never raises: every
        failure path in the state machine ends here, and a cleanup that throws
        would take the state machine down with it."""
        recorder, self._recorder = self._recorder, None
        if recorder is not None:
            try:
                recorder.stop()
            except Exception as e:
                A._log_problem("dictation recorder could not be stopped", e)
        if self._dir is not None:
            shutil.rmtree(self._dir, ignore_errors=True)
            self._dir = None


class Daemon:
    """One dictation at a time: hotkey → microphone → transcript → clipboard.

    `toggle` is the only public way in, so the hotkey listener, the signal handler
    and the duration timer cannot interleave. `capture_factory(cfg)` returns
    something with start()/stop()/discard(); `transcribe(wav, language)` and
    `deliver(transcript, config, clipboard, sink)` default to the real ones, and
    are parameters so the tests need neither a microphone, nor a model, nor a
    socket.
    """

    def __init__(self, config, clipboard, sink, cfg=None,
                 capture_factory=MicCapture, transcribe=None, deliver=None):
        self._config = config
        self._clipboard = clipboard
        self._sink = sink
        self._cfg = {} if cfg is None else cfg
        self._capture_factory = capture_factory
        self._transcribe = transcribe or A.transcribe_audio
        self._deliver = deliver or dictation.deliver

        # `state` is read from anywhere and written only under `_lock`.
        self.state = IDLE
        self._lock = threading.Lock()
        self._capture = None
        self._worker = None
        self._timer = None
        # Which recording the live duration timer belongs to; see _leave_recording.
        self._epoch = 0

    # ------------------------------- the way in -------------------------------

    def toggle(self):
        """Start a recording, stop one, or refuse — whichever the current state
        means. Returns as soon as the transition is made: the transcription and
        the correction run on a worker thread, because this is called from the
        hotkey listener."""
        with self._lock:
            if self.state == IDLE:
                self._begin_recording()
            elif self.state == RECORDING:
                self._begin_processing()
            else:
                # Nothing else. A second dictation on top of one still being
                # corrected would race for the clipboard, and the first one is
                # the one the user is waiting for.
                self._sink.failed("still working on the previous dictation")

    def join_worker(self, timeout=None):
        """Wait for the worker thread, if there is one. For the tests only — the
        daemon never sequences its own work this way, because every call above
        happens on a thread that has to return at once."""
        with self._lock:
            worker = self._worker
        if worker is not None:
            worker.join(timeout)

    # --------------------------- the transitions ---------------------------
    # All three run with `_lock` held, so the *transitions* serialise: no two
    # threads can read the same state and both act on it. That is the whole
    # guarantee. The reports are not ordered by it — the worker's own sink.failed
    # and every sink call inside dictation.deliver run with no lock held, so a
    # "still working" from toggle can interleave with a worker's report.
    #
    # Holding the lock across a sink call is safe only because a sink cannot block
    # indefinitely: NotifySink.recording() and processing() each run *two* bounded
    # subprocesses, the notification and the sound cue, so the worst case in here
    # is 2 x dictation.NOTIFY_TIMEOUT_SECONDS, not one.

    def _begin_recording(self):
        capture = self._capture_factory(self._cfg)
        try:
            capture.start()
        except Exception as e:
            A._log_problem("dictation microphone could not be opened", e)
            capture.discard()
            # The clipboard is not touched: the user still has whatever they
            # copied before, and no dictation happened to replace it with.
            self._sink.failed(f"the microphone could not be opened: {e}")
            return
        self._capture = capture
        self.state = RECORDING
        self._timer = threading.Timer(self._config.max_seconds,
                                      self._on_duration_bound, args=(self._epoch,))
        # daemon: a recording that is never stopped must not keep the process
        # alive on its way out.
        self._timer.daemon = True
        self._timer.start()
        self._sink.recording()

    def _begin_processing(self):
        """Stop the recording and hand the WAV to a worker.

        Every step is inside the guard, not just the stop(): once the state says
        PROCESSING, only the worker can move it, so anything that goes wrong
        before the worker is running strands the daemon exactly the way _work's
        `finally` exists to prevent — one step earlier, where nothing catches it.
        Thread.start() raising under thread exhaustion is the live case; a status
        sink that does not swallow its own failures the way NotifySink does is the
        other."""
        capture = self._capture
        self._leave_recording()
        try:
            wav_path = capture.stop()
            self.state = PROCESSING
            self._sink.processing()
            # Published only once it is actually running: join()ing a thread that
            # never started raises, and join_worker is what the tests wait on.
            worker = threading.Thread(target=self._work, args=(wav_path, capture),
                                      name="dictation-worker", daemon=True)
            worker.start()
            self._worker = worker
        except Exception as e:
            A._log_problem("dictation could not be handed to a worker", e)
            # The state is put right before the user is told, so even a sink that
            # raises in here cannot leave the daemon in PROCESSING.
            capture.discard()
            self._capture = None
            self.state = IDLE
            self._sink.failed(f"the dictation could not be finished: {e}")

    def _leave_recording(self):
        """Close the recording window: cancel the duration timer and make sure a
        bound that fires anyway is ignored.

        cancel() alone is not enough. A Timer whose thread has already left its
        wait cannot be cancelled, so a bound reached in the same moment the user
        presses the hotkey still runs — and would then either start a recording
        nobody asked for or cut the *next* one short. `_epoch` names the recording
        each timer belongs to, and this bump orphans the one that just ended."""
        self._epoch += 1
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _on_duration_bound(self, epoch):
        """The duration bound was reached: stop the recording as a toggle would,
        but say that the limit stopped it rather than the user."""
        with self._lock:
            if epoch != self._epoch or self.state != RECORDING:
                return                    # this timer's recording is already over
            # Turkish, unlike the rest of this module's English, because it is
            # read by the user in a notification alongside the sink's own
            # messages. `failed` is the only method of the four-method sink
            # contract that carries a reason, and the dictation *was* cut short:
            # what the microphone did capture is still delivered below.
            self._sink.failed(f"kayıt {self._config.max_seconds} saniye sınırına "
                              "ulaştı ve durduruldu")
            self._begin_processing()

    # ------------------------------- the worker -------------------------------

    def _work(self, wav_path, capture):
        """Transcribe, deliver, and return the daemon to IDLE whatever happens."""
        try:
            transcript = self._transcribe(wav_path, self._config.language)
            # deliver reports on the sink itself on every one of its paths, so
            # nothing here reports again: that would notify the user twice for
            # one dictation.
            self._deliver(transcript, self._config, self._clipboard, self._sink)
        except Exception as e:
            A._log_problem("dictation could not be processed", e)
            self._sink.failed(f"processing failed: {e}")
        finally:
            # A daemon left in PROCESSING is dead to the user with no way to tell:
            # the hotkey answers "still working" forever.
            capture.discard()
            with self._lock:
                self._capture = None
                self.state = IDLE
