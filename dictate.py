"""The dictation daemon: the microphone capture, the state machine that drives one
dictation from a click to the clipboard, the power axis that loads and unloads the
model, and the CLI that runs and signals it.

dictation.py holds everything that can be tested without threads or hardware —
configuration, the clipboard, the status sinks, the history log, the menu model and
the correction pipeline. menubar.py holds the AppKit layer and decides nothing. Here
the state is touched from five threads — the menu bar's main thread, the signal pump,
the model loader, the duration timer and the worker — so `toggle`, `enable` and
`disable` are the only ways in, and one lock guards both axes.

    dictate.py             run the menu bar in the foreground
    dictate.py --toggle    start or stop a dictation in the running app
    dictate.py --stop      stop the running app
"""
import os
import pathlib
import queue
import shutil
import signal
import sys
import tempfile
import threading

import audiocript as A
import dictation

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
                 capture_factory=MicCapture, transcribe=None, deliver=None,
                 history=None, ensure_model=None, unload_model=None):
        self._config = config
        self._clipboard = clipboard
        self._sink = sink
        self._cfg = {} if cfg is None else cfg
        self._capture_factory = capture_factory
        self._transcribe = transcribe or A.transcribe_audio
        self._deliver = deliver or dictation.deliver
        self._history = history
        # Injected for the same reason `transcribe` and `deliver` are: no test here
        # may load or unload a real model.
        self._ensure_model = ensure_model or A._ensure_model
        self._unload_model = unload_model or A.unload_model

        # `state` is read from anywhere and written only under `_lock`.
        self.state = dictation.IDLE
        # `power` likewise — and every power_changed report is made *under* the lock
        # too, unlike the dictation reports. enable() reports LOADING and the loader
        # thread reports ON; reported outside the lock they can arrive in the other
        # order, which leaves an icon saying "loading" over a daemon that is up.
        self.power = dictation.POWER_OFF
        self._loader = None
        self._lock = threading.Lock()
        self._capture = None
        self._worker = None
        self._timer = None
        # Which recording the live duration timer belongs to; see _leave_recording.
        self._epoch = 0

    # ------------------------------ the power axis ------------------------------

    def enable(self):
        """Bring the daemon up: load the model, then start accepting recordings.

        Returns as soon as the transition is made. The load runs on a thread of its
        own because it takes seconds — measured 11.3s cold, 4.1s warm on this machine
        — and this is called from the AppKit main thread, which has to keep drawing
        the menu bar while it happens.

        A second call while the daemon is coming up or already up does nothing.
        A._ensure_model would serialise two loads on its own lock anyway, so the
        second would be pure waiting behind an icon that already says what is
        happening."""
        with self._lock:
            if self.power != dictation.POWER_OFF:
                return
            loader = threading.Thread(target=self._load, name="dictation-model",
                                      daemon=True)
            try:
                loader.start()
            except Exception as e:
                # Thread exhaustion. Power has not moved, so the next click retries.
                A._log_problem("the dictation model loader could not be started", e)
                self._sink.failed(f"the daemon could not be started: {e}")
                return
            # After start(), so a loader that could not start leaves power alone —
            # and safe in that order because _load reaches for this lock before it
            # looks at `power`, so it cannot see OFF here and call itself stale.
            self.power = dictation.POWER_LOADING
            self._loader = loader
            self._sink.power_changed(dictation.POWER_LOADING)

    def _load(self):
        """Load the model, then open the daemon for recordings. The slow half of the
        power axis, on a thread of its own."""
        try:
            self._ensure_model(self._config.language)
        except BaseException as e:                                     # noqa: BLE001
            # BaseException for the reason _deferred catches it: this is a thread of
            # its own, and anything escaping here strands power at LOADING behind an
            # icon that never resolves, with nothing on screen and nothing in a log.
            A._log_problem("the dictation model could not be loaded", e)
            with self._lock:
                self.power = dictation.POWER_OFF
                self._sink.power_changed(dictation.POWER_OFF)
            self._sink.failed(f"the model could not be loaded: {e}")
            return
        with self._lock:
            stale = self.power != dictation.POWER_LOADING
            if not stale:
                self.power = dictation.POWER_ON
                self._sink.power_changed(dictation.POWER_ON)
        if stale:
            # disable() arrived while this load was running, and the user has already
            # been told the daemon is off. Two things go wrong if what the load
            # produced is left alone: the memory stays resident behind an icon that
            # says off, and the next enable() finds A._ensure_model already satisfied
            # and returns without loading anything the user waited for.
            self._drop_model()

    def disable(self):
        """Take the daemon down: refuse new recordings, and give the model's memory
        back. Measured: unloading returns 3085.6 MB of MPS memory.

        Refused while a dictation is in flight. That is the rule this feature was
        asked for, and it lives here rather than only in the menu's greyed-out item —
        so it holds for `--toggle`'s door too, and so the invariant below is true
        rather than merely likely:

        **Power OFF implies state IDLE.** No caller anywhere has to handle "off while
        recording", because there is no way to reach it."""
        with self._lock:
            if self.power == dictation.POWER_OFF:
                return
            if self.state != dictation.IDLE:
                self._sink.failed("a dictation is in progress; the daemon cannot be "
                                  "stopped until it finishes")
                return
            # Whether a load is still in flight decides who drops the model; see
            # below. Read under the lock, because _load moves it out of LOADING.
            loading = self.power == dictation.POWER_LOADING
            self.power = dictation.POWER_OFF
            self._sink.power_changed(dictation.POWER_OFF)
        if loading:
            # Nothing to drop yet, and dropping anyway would be worse than useless:
            # the load in flight holds A._MODEL_LOCK, which A.unload_model takes, so
            # this thread — the AppKit main thread — would block for the rest of the
            # load. Measured: 11.3s for a cold one. _load's own stale path drops what
            # the load produced, on the loader's thread, the moment it has it.
            return
        # Outside the lock: a gc pass and a GPU cache drop are not instant, and with
        # power already OFF nothing can start a recording behind this call.
        self._drop_model()

    def _drop_model(self):
        """Unload the configured language's model, reporting a refusal rather than
        raising. The reference the daemon held is gone by the time this runs either
        way, and a model that will not drop is not a reason to leave the daemon on."""
        try:
            self._unload_model(self._config.language)
        except Exception as e:
            A._log_problem("the dictation model could not be unloaded", e)

    def join_power(self, timeout=None):
        """Wait for a model load, if one is running. Called by the tests and by the
        shutdown path, never by the daemon's own threads — the mirror of
        join_worker."""
        with self._lock:
            loader = self._loader
        if loader is not None:
            loader.join(timeout)

    # ------------------------------- the way in -------------------------------

    def toggle(self):
        """Start a recording, stop one, or refuse — whichever the current state
        means. Returns as soon as the transition is made: the transcription and
        the correction run on a worker thread, because this is called from the
        hotkey listener."""
        with self._lock:
            if self.power != dictation.POWER_ON:
                # Both doors arrive here. The menu greys its own item out, but
                # SIGUSR1 from `dictate.py --toggle` reaches this directly, and
                # opening a microphone with no model behind it would record a
                # dictation nothing can transcribe.
                self._sink.failed("the dictation daemon is not running")
            elif self.state == dictation.IDLE:
                self._begin_recording()
            elif self.state == dictation.RECORDING:
                self._begin_processing()
            else:
                # Nothing else. A second dictation on top of one still being
                # corrected would race for the clipboard, and the first one is
                # the one the user is waiting for.
                self._sink.failed("still working on the previous dictation")

    def join_worker(self, timeout=None):
        """Wait for the worker thread, if there is one. Called by the tests and by
        the shutdown path, never by the daemon's own threads: every call above
        happens on a thread that has to return at once."""
        with self._lock:
            worker = self._worker
        if worker is not None:
            worker.join(timeout)

    def abandon_recording(self):
        """Throw away a recording that is still going, for the way out.

        A recording only ends through a toggle, so a daemon shutting down mid
        recording would otherwise leave the microphone open and its temporary
        directory on disk. PROCESSING is deliberately left alone: there the worker
        owns the capture and discards it in its own `finally`, and discarding it
        here would delete the WAV out from under a transcription of something the
        user has already spoken."""
        with self._lock:
            if self.state != dictation.RECORDING:
                return
            capture, self._capture = self._capture, None
            self._leave_recording()
            self.state = dictation.IDLE
        # Outside the lock, as the worker's own cleanup is: discard() stops a
        # recorder and removes a directory, and neither needs the state guarded.
        capture.discard()

    # --------------------------- the transitions ---------------------------
    # All three run with `_lock` held, so the *transitions* serialise: no two
    # threads can read the same state and both act on it. That is the whole
    # guarantee. The reports are not ordered by it — the worker's own sink.failed
    # and every sink call inside dictation.deliver run with no lock held, so a
    # "still working" from toggle can interleave with a worker's report.
    #
    # The hold is not bounded by dictation.NOTIFY_TIMEOUT_SECONDS.
    # _begin_processing holds the lock across
    # capture.stop()'s WAV conversion, whose cost scales with how long the
    # recording ran (measured on this machine: roughly 0.1-1.1s for a synthetic
    # 300s recording, the low end once torchaudio's one-time import has already
    # run this process) — before either of processing()'s subprocesses even
    # starts. _begin_recording holds it across audiocript._start_mic's retry
    # loop: up to audiocript._MIC_OPEN_BACKOFF's 2s of sleep plus two
    # audiocript._refresh_audio_devices() calls. _on_duration_bound holds it
    # across both a sink.failed() and a full _begin_processing. None of that is
    # bounded by the notify timeout.

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
        self.state = dictation.RECORDING
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
            self.state = dictation.PROCESSING
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
            self.state = dictation.IDLE
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
            if epoch != self._epoch or self.state != dictation.RECORDING:
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
            delivery = self._deliver(transcript, self._config, self._clipboard,
                                     self._sink)
            self._record(delivery)
        except Exception as e:
            A._log_problem("dictation could not be processed", e)
            self._sink.failed(f"processing failed: {e}")
        finally:
            # A daemon left in PROCESSING is dead to the user with no way to tell:
            # the hotkey answers "still working" forever.
            capture.discard()
            with self._lock:
                self._capture = None
                self.state = dictation.IDLE

    def _record(self, delivery):
        """Add what reached the clipboard to the history log, if there is one.

        The condition is `delivery.text`, not a list of statuses: EMPTY is the only
        path that copies nothing, and testing the text means a status added to the
        pipeline later cannot be silently logged or silently dropped.

        Guarded on its own rather than left to _work's `except`. The real History
        already swallows its own failures, but a failure that reached _work's handler
        would report "processing failed" for a dictation whose text is on the
        clipboard and which the user has already been told about — the log is the
        least important thing happening here.

        `delivery` may be None: only an injected test double returns that, and
        nothing to record is not an error."""
        if self._history is None or delivery is None or not delivery.text:
            return
        try:
            self._history.append(delivery.text, delivery.status)
        except Exception as e:
            A._log_problem("a dictation could not be recorded in the history", e)


# ============================ The global hotkey ============================


def _deferred(action, name):
    """Return a callable that hands `action` to a thread of its own and returns at
    once, and start that thread. Every press ends up on the same thread, so two of
    them are answered in the order they arrived.

    The SIGUSR1 handler is the one caller, and two things make deferring necessary
    there:

    - The handler runs on the main thread and is not re-entrant. A second --toggle
      arriving while the first is inside `toggle` would deadlock on a lock its own
      thread already holds.
    - That main thread is the AppKit run loop. `toggle` can hold the daemon's lock
      for as long as the transition in progress takes — _begin_processing holds it
      across a WAV conversion whose cost scales with the recording's length,
      _begin_recording across _start_mic's retry backoff — and none of that may
      happen on the thread drawing the menu bar.

    A failure here would be invisible: --toggle would stop working for the rest of
    the session with nothing on screen and nothing in any log. So the guard below
    catches BaseException rather than Exception, and the pump loop is the one thing
    in this module that may not end.
    """
    requests = queue.SimpleQueue()

    def pump():
        while True:
            requests.get()
            try:
                action()
            except BaseException as e:                            # noqa: BLE001
                A._log_problem("the dictation toggle failed", e)

    # daemon: the pump is blocked on an empty queue for almost all of its life and
    # must not keep the process alive on its way out.
    threading.Thread(target=pump, name=name, daemon=True).start()

    def submit(*_signal_arguments):
        """Ignores its arguments so it can serve as a signal handler, which is called
        with two, as well as a plain callable, which is called with none."""
        requests.put(None)

    return submit


# ======================= The daemon's own lifecycle =======================

PID_PATH = pathlib.Path("~/.audiocript/dictate.pid").expanduser()

# Which signal each CLI action sends. SIGUSR1 has no meaning of its own, and
# SIGTERM is what the shutdown handler is installed for.
ACTION_SIGNALS = {"toggle": signal.SIGUSR1, "stop": signal.SIGTERM}

# How long the way out waits for a dictation already being transcribed. The worker
# is a daemon thread, so without this wait a Ctrl-C during transcription loses a
# dictation the user has already spoken; with it unbounded, a wedged provider call
# would keep the process alive instead.
DRAIN_SECONDS = 15

USAGE = ("usage: dictate.py [--toggle | --stop]\n"
         "  (no arguments)  run the dictation daemon in the foreground\n"
         "  --toggle        start or stop a dictation in the running daemon\n"
         "  --stop          stop the running daemon")


def read_pid(path=PID_PATH):
    """The pid of a daemon that is actually running, or None — which also covers an
    absent file, an unreadable one, one holding something that is not a pid, and one
    left behind by a daemon that is gone."""
    try:
        pid = int(path.read_text().strip())
    except (OSError, ValueError):
        return None
    if pid <= 0:
        # os.kill(0, sig) signals every process in our own group; 0 and negatives
        # are not pids this file may act on.
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None                 # stale: the daemon that wrote it is gone
    except PermissionError:
        return pid                  # running, under another user
    except OSError:
        return None
    return pid


def write_pid(path=PID_PATH):
    """Claim `path` for this process, or raise if a daemon is already running.

    A stale file is taken over rather than refused: a daemon killed with SIGKILL
    leaves one behind, and refusing would make the feature dead until the user
    found the file.

    The claim is atomic rather than a read followed by a write, so of two daemons
    started in the same instant exactly one wins. Read-then-write let both pass the
    liveness check before either wrote, and both then believed they owned the file
    — at which point the loser's own `_shutdown` unlinks the *winner's* pid file on
    the way out, leaving a daemon running that `--toggle` and `--stop` can no
    longer find.

    Two things have to be atomic, and the second is easy to miss. This is why the
    pid goes into a temporary file first and `os.link` is what publishes it:

    - **The claim.** `link` fails with FileExistsError if the name is taken, so
      only one caller can create it.
    - **Its contents.** An `O_CREAT|O_EXCL` create is exclusive but leaves the file
      *empty* until the write lands, and in that window a rival's `read_pid` gets
      an unparseable file, reads it as stale, and unlinks the winner's claim. The
      linked file already holds the pid the instant it is visible. Measured: with
      the empty window, four simultaneous starts produced two winners.

    Taking a stale file over is unlink-and-retry, and it keeps one residual race
    rather than pretending otherwise. Between the `read_pid` that answers "stale"
    and the unlink below, another daemon can link a live claim over that name; the
    unlink then removes *its* file and both processes believe they own the pid —
    the same failure the atomic claim above exists to prevent. Two things have to
    be true at once for the window to open, and it is microseconds wide: a stale
    file has to be on disk already, *and* two daemons have to start on top of it in
    that instant. With no stale file — the ordinary case, however many daemons race
    — the claim is exact.

    Closing it needs the claim verified after the fact: rename-then-fstat, or an
    open-and-compare loop. That is real concurrency work on a path already correct
    for every case the daemon meets, so it is not done here.

    Not worth reintroducing: an earlier version moved the stale file aside with
    `os.rename` first and called that safer. It is not. `rename` identifies the
    *name*, not the inode `read_pid` just examined, so it carries a rival's fresh
    claim off exactly as the unlink does — measured, by inode, before it was
    deleted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{os.getpid()}"
    tmp.write_text(f"{os.getpid()}\n")
    try:
        for _ in range(3):
            try:
                os.link(tmp, path)
                return
            except FileExistsError:
                pass
            live = read_pid(path)
            if live is not None:
                raise RuntimeError(
                    f"a dictation daemon is already running (pid {live}) — use "
                    "`dictate.py --toggle` to dictate, or `dictate.py --stop` to "
                    "stop it") from None
            A._unlink_quietly(path)         # stale; drop it and race for the name
        # Three rounds of losing a race we then found nobody had won. Refusing is
        # the safe answer: starting anyway is what leaves two daemons behind.
        raise RuntimeError(f"{path} kept changing hands; no daemon was started")
    finally:
        A._unlink_quietly(tmp)


def signal_daemon(action, path=PID_PATH):
    """Send the running daemon the signal for `action` and return a process exit
    status. Non-zero with no daemon running: a Shortcuts binding or a shell alias
    that exited zero would look as if it had worked."""
    pid = read_pid(path)
    if pid is None:
        print("no dictation daemon is running (start one with ./run.sh --dictate)")
        return 1
    try:
        os.kill(pid, ACTION_SIGNALS[action])
    except OSError as e:
        print(f"the dictation daemon (pid {pid}) could not be signalled: {e}")
        return 1
    return 0


def _shutdown(daemon, path=PID_PATH, drain_seconds=DRAIN_SECONDS):
    """Take the daemon down in order: no new dictations, then the one in flight, then
    the pid file.

    The pid file is removed in a `finally` because it is the one step whose absence
    is felt after the process is gone — a file left behind refuses the next start.

    A model load still running is deliberately *not* waited for. It holds no work of
    the user's — unlike the worker, which holds a dictation they have already spoken —
    and a cold load takes 11.3s, which is far too long for Ctrl-C to appear to do
    nothing. Its thread is a daemon thread for exactly this reason.
    """
    try:
        daemon.abandon_recording()
        daemon.join_worker(timeout=drain_seconds)
        if daemon.state != dictation.IDLE:
            print(f"a dictation was still being processed after {drain_seconds}s; "
                  "its text did not reach the clipboard")
    except Exception as e:
        A._log_problem("the dictation daemon did not shut down cleanly", e)
    finally:
        A._unlink_quietly(path)


def _run_daemon():
    """Run the menu bar in the foreground until it is asked to stop."""
    cfg = A.load_config()
    try:
        # Both are needed: the resolved config drives the daemon, and the raw dict
        # is what MicCapture hands to audiocript._resolve_mic_index to find the
        # configured microphone by name.
        config = dictation.resolve_config(cfg)
    except dictation.ConfigError as e:
        # The message already names the offending value; nothing to add, and the
        # config itself is never printed — it holds the API key.
        print(f"dictation cannot start: {e}")
        return 1
    try:
        write_pid()
    except (OSError, RuntimeError) as e:
        print(f"dictation cannot start: {e}")
        return 1

    # Imported here rather than at the top of the module so `--toggle` and `--stop`
    # do not pay for AppKit: they send a signal and exit, and this is the only path
    # that draws anything.
    import menubar

    history = dictation.History()
    clipboard = dictation.Clipboard()
    # Power starts OFF and nothing is loaded here. The icons have to be in the menu
    # bar at once; the first icon is what loads the model, when the user asks it to.
    daemon = Daemon(config, clipboard, dictation.NotifySink(), cfg=cfg,
                    history=history)
    status = 0
    try:
        # Before the model load, not after: the default action for SIGUSR1 is to
        # terminate the process, so a --toggle during a first-run download would
        # otherwise kill the daemon outright.
        #
        # Note what it does *not* buy. The pump thread is live the moment
        # _deferred builds it, so a press arriving here opens the microphone at
        # once, next to the load running on this thread — it does not wait for
        # the model. That is safe rather than intended: A.transcribe_audio calls
        # A._ensure_model itself and _ensure_model holds A._MODEL_LOCK, so it is
        # the worker's transcription that waits, not the recording.
        signal.signal(signal.SIGUSR1, _deferred(daemon.toggle, "dictation-signal"))
        # Both stop signals end the run loop rather than the process, so the shutdown
        # in the `finally` below actually runs — see menubar.stop_the_loop. They
        # reach Python between bytecodes on the main thread, which is what menubar's
        # heartbeat timer is there to provide: measured 4996 ms late with no timer
        # scheduled, 197 ms with it.
        for stop_signal in (signal.SIGTERM, signal.SIGINT):
            signal.signal(stop_signal, lambda *_: menubar.stop_the_loop())
        print("Two icons are now in the menu bar. Click the left one to start the "
              f"daemon; the first start loads the {A.lang_name(config.language)} "
              "model, and the very first one downloads it, which takes a while.")
        print("Ctrl-C stops the app.", flush=True)
        menubar.run(daemon, history, clipboard)
    except KeyboardInterrupt:
        pass                      # Ctrl-C before the handlers above were installed
    except Exception as e:
        A._log_problem("the dictation daemon stopped", e)
        print(f"dictation stopped: {e}")
        status = 1
    finally:
        print("\nStopping the dictation daemon…", flush=True)
        # No new dictations from either door on the way out: a --toggle arriving
        # now would open the microphone as the process is leaving.
        signal.signal(signal.SIGUSR1, signal.SIG_IGN)
        _shutdown(daemon)
    return status


def main(argv=None):
    """The CLI. Returns a process exit status; the caller below is what exits."""
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv:
        return _run_daemon()
    if len(argv) == 1 and argv[0] in ("--toggle", "--stop"):
        return signal_daemon(argv[0].removeprefix("--"))
    print(USAGE)
    return 2


if __name__ == "__main__":
    sys.exit(main())
