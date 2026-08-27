"""The dictation daemon: the microphone capture, the state machine that drives one
dictation from hotkey to clipboard, the global hotkey that starts it, and the CLI
that runs and signals the daemon.

dictation.py holds everything that can be tested without threads or hardware —
configuration, the clipboard, the status sinks and the correction pipeline. Here
the state is touched from four threads (the hotkey, the signal, the duration timer
and the worker), so `toggle` is the only way in and one lock guards `state`.

    dictate.py             run the daemon in the foreground
    dictate.py --toggle    start or stop a dictation in the running daemon
    dictate.py --stop      stop the running daemon
"""
import os
import pathlib
import queue
import shutil
import signal
import sys
import tempfile
import threading

from pynput import keyboard

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
                 capture_factory=MicCapture, transcribe=None, deliver=None):
        self._config = config
        self._clipboard = clipboard
        self._sink = sink
        self._cfg = {} if cfg is None else cfg
        self._capture_factory = capture_factory
        self._transcribe = transcribe or A.transcribe_audio
        self._deliver = deliver or dictation.deliver

        # `state` is read from anywhere and written only under `_lock`.
        self.state = dictation.IDLE
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
            if self.state == dictation.IDLE:
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
                self.state = dictation.IDLE


# ============================ The global hotkey ============================


def _deferred(action, name):
    """Return a callable that hands `action` to a thread of its own and returns at
    once, and start that thread. Every press ends up on the same thread, so two of
    them are answered in the order they arrived.

    Nothing may call Daemon.toggle on the thread that delivered the event, and
    nothing may let it raise there:

    - pynput calls the hotkey callback from its CGEventTap callback (see
      `_handler` in pynput/_util/darwin.py, and `_emitter` in _util/__init__.py),
      and macOS switches off a tap whose callback is too slow. pynput calls
      `CGEventTapEnable(tap, True)` exactly once, at listener startup, and holds
      no other reference to it — so a tap the system disables is never switched
      back on, while IS_TRUSTED stays true and the listener stays alive. `toggle`
      can hold the daemon's lock for as long as the transition in progress takes
      — not bounded by dictation.NOTIFY_TIMEOUT_SECONDS.
    - `_emitter` calls `listener.stop()` on any exception the callback lets
      through, which ends the listener for good. Daemon._begin_recording's
      trailing sink.recording() is outside its try/except by design, so `toggle` can
      raise on the recording path.
    - the SIGUSR1 handler runs on the main thread and is not re-entrant: a second
      --toggle arriving while the first one is inside `toggle` would deadlock on a
      lock its own thread already holds. Deferring makes that impossible too.

    Both failures look the same to a user — the hotkey stops working for the rest
    of the session with nothing on screen and nothing in any log — so the guard
    below catches BaseException rather than Exception, and the pump loop is the
    one thing in this module that may not end.
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
        """Ignores its arguments so it can serve as a signal handler as well as a
        pynput callback, which is called with none."""
        requests.put(None)

    return submit


def build_listener(config, on_toggle, hotkeys_class=keyboard.GlobalHotKeys):
    """Map each configured hotkey to what it does and return the listener,
    unstarted — the caller decides when the event tap goes up.

    The thread that carries `on_toggle` off the callback is running by the time
    this returns, because the callback registered here closes over it. Wrapping
    happens in here rather than at the call site so the guarantee cannot be lost
    by a caller passing `daemon.toggle` straight through.

    `hotkeys_class` is a parameter so the tests can watch what was registered
    without creating a real event tap.
    """
    callbacks = {"toggle": _deferred(on_toggle, "dictation-hotkey")}
    return hotkeys_class({combination: callbacks[action]
                          for action, combination in config.hotkeys.items()})


def check_listener(listener):
    """None when the hotkey can work, otherwise the warning to put in front of the
    user. Call it after start() and wait().

    Two signals, because they fail differently. IS_TRUSTED is assigned from
    AXIsProcessTrusted() at the top of the darwin backend's `_run`, so it is
    meaningful only once wait() has returned; and a tap that fails for any other
    reason marks the listener ready and then exits its thread, leaving IS_TRUSTED
    true. Either way the daemon looks healthy while the hotkey does nothing, which
    is why this is reported at startup instead of being discovered by a user
    pressing a key and getting silence.
    """
    if listener.IS_TRUSTED and listener.is_alive():
        return None
    return ("WARNING: the dictation hotkey cannot be registered — this process is "
            "not allowed to watch the keyboard.\n"
            "  Grant it in System Settings → Privacy & Security → Input "
            "Monitoring (and in Accessibility, which is the list macOS reports "
            "through AXIsProcessTrusted) for the terminal or the Python "
            "interpreter running this daemon, then start it again.\n"
            "  Until then `dictate.py --toggle` starts and stops a dictation "
            "without any permission at all, from another shell, a macOS Shortcut "
            "or skhd.")


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


def _shutdown(listener, daemon, path=PID_PATH, drain_seconds=DRAIN_SECONDS):
    """Take the daemon down in order: no new dictations, then the one in flight,
    then the pid file.

    The pid file is removed in a `finally` because it is the one step whose absence
    is felt after the process is gone — a file left behind refuses the next start.
    """
    try:
        if listener is not None:
            listener.stop()
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
    """Run the daemon in the foreground until it is asked to stop."""
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

    daemon = Daemon(config, dictation.Clipboard(), dictation.NotifySink(), cfg=cfg)
    listener = None
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
        print(f"Loading the local {A.lang_name(config.language)} model. The first "
              "run downloads it, which takes a while.", flush=True)
        A._ensure_model(config.language)

        listener = build_listener(config, daemon.toggle)
        listener.start()
        listener.wait()
        warning = check_listener(listener)
        if warning:
            print(warning)
        else:
            print(f"Ready: press {config.hotkeys['toggle']} to start a dictation "
                  "and again to finish it.")
        print("Ctrl-C stops the daemon.", flush=True)

        # Installed only now, so Ctrl-C during the model load above still raises
        # KeyboardInterrupt and interrupts it instead of being swallowed by a
        # handler while the download carries on.
        stopping = threading.Event()
        for stop_signal in (signal.SIGTERM, signal.SIGINT):
            signal.signal(stop_signal, lambda *_: stopping.set())
        # The main thread does nothing else on purpose: the spec reserves it for a
        # menu bar indicator. A handler setting the event wakes this up — verified
        # on this interpreter, both for SIGTERM and for Ctrl-C.
        stopping.wait()
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
        _shutdown(listener, daemon)
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
