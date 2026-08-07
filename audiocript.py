import os
import sys
import json
import time
import shutil
import tty
import select
import contextlib
import threading
import subprocess
import termios  # available on Unix-based systems.
import wave
from datetime import datetime
from pathlib import Path
import numpy as np
import sounddevice as sd
import warnings
from pynput import keyboard
from rich.console import Console, Group
from rich.panel import Panel
from rich.live import Live
from rich.text import Text
from rich.layout import Layout

# Suppress warnings (FutureWarning, DeprecationWarning, UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Quiet transformers / huggingface_hub noise (must be set BEFORE importing them;
# they are imported lazily, so setting it here is enough).
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _silence_ml_logging():
    """Silence transformers progress bars and transformers/huggingface_hub warning
    logs (e.g. 'unauthenticated requests', logits processor warnings)."""
    import logging
    try:
        import transformers
        transformers.logging.set_verbosity_error()
        transformers.utils.logging.disable_progress_bar()
    except Exception:
        pass
    for name in ("transformers", "huggingface_hub"):
        logging.getLogger(name).setLevel(logging.ERROR)


# Rich console object
console = Console()

# While the full-screen TUI is active, library/app prints are suppressed so they
# don't corrupt the screen (status is shown in the UI instead).
_QUIET = False

# Config file, stored next to the script.
CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
DEFAULT_BASE_PATH = Path.home() / "Audiocript" / "recordings"

# Supported languages: code -> Whisper language name
LANGUAGES = {"tr": "turkish", "en": "english"}

# Model and runtime used per language.
#  - Turkish: a Turkish fine-tuned model via Hugging Face transformers.
#  - English: ggml-distil-large-v3 via whisper.cpp (pywhispercpp).
TR_HF_MODEL = "selimc/whisper-large-v3-turbo-turkish"
EN_GGML_REPO = "distil-whisper/distil-large-v3-ggml"
EN_GGML_FILE = "ggml-distil-large-v3.bin"

# Speaker diarization (optional, opt-in). A gated Hugging Face model; needs a
# token and the user must accept its terms once at huggingface.co.
PYANNOTE_MODEL = "pyannote/speaker-diarization-3.1"

# Cache loaded models so they are not reloaded.
_hf_pipe = None
_cpp_model = None
_diar_pipe = None
# Lock so the model is not loaded twice at once (e.g. background pre-warm + transcription).
_MODEL_LOCK = threading.Lock()


def _free_cpp_model_quietly():
    """
    Free the whisper.cpp model while stderr (fd 2) is redirected to /dev/null, to
    hide the C/Metal teardown log ('ggml_metal_free: deallocating') it prints on
    release. Called at exit (atexit).
    """
    global _cpp_model
    if _cpp_model is None:
        return
    saved = devnull = None
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        saved = os.dup(2)
        os.dup2(devnull, 2)
        _cpp_model = None  # release logs go to /dev/null
    except Exception:
        _cpp_model = None
    finally:
        if saved is not None:
            try:
                os.dup2(saved, 2)
                os.close(saved)
            except Exception:
                pass
        if devnull is not None:
            try:
                os.close(devnull)
            except Exception:
                pass


import atexit as _atexit
_atexit.register(_free_cpp_model_quietly)


def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')


def flush_stdin():
    """
    Clear any pending characters left in sys.stdin (using termios.tcflush()
    on Unix-based systems).
    """
    try:
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except Exception:
        pass


def load_config():
    """Read config.json; return an empty dict if missing or invalid."""
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg if isinstance(cfg, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_config(cfg):
    """Write the config to config.json."""
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def lang_name(code):
    """Map a language code ('tr'/'en') to the name Whisper expects."""
    return LANGUAGES.get(code, "turkish")


def list_input_devices():
    """Return (index, name) of audio devices that can capture input."""
    result = []
    for idx, dev in enumerate(sd.query_devices()):
        if dev.get("max_input_channels", 0) > 0:
            result.append((idx, dev["name"]))
    return result


def device_name(index):
    """Return a device's name from its index, or the index as text if unknown."""
    try:
        return sd.query_devices(index)["name"]
    except Exception:
        return str(index)


def mix_to_mono(arrays):
    """
    Mix one or more mono int16 signals into a single int16 signal. Trims streams
    to the shortest length (device clocks drift slightly), sums them, and scales
    down by the peak if needed to avoid clipping.
    """
    arrays = [a for a in arrays if a is not None and len(a) > 0]
    if not arrays:
        return None
    if len(arrays) == 1:
        return arrays[0].astype(np.int16)
    n = min(len(a) for a in arrays)
    acc = np.zeros(n, dtype=np.int32)
    for a in arrays:
        acc += a[:n].astype(np.int32)
    peak = int(np.max(np.abs(acc))) if n else 0
    if peak > 32767:
        acc = (acc * (32767.0 / peak)).astype(np.int32)
    return acc.astype(np.int16)


def resample_to_target(arr, src_fs, target_fs=16000):
    """
    Convert a signal (int16 or float32; mono or multi-channel) to mono int16 at
    the target sample rate (16000 Hz for Whisper). Averages channels if multi-
    channel. If src_fs == target_fs, just downmix to mono without resampling.
    Resampling uses torchaudio (anti-aliased).
    """
    if arr is None or len(arr) == 0:
        return None
    if np.issubdtype(arr.dtype, np.floating):
        f = arr.astype(np.float32)            # already in [-1, 1]
    else:
        f = arr.astype(np.float32) / 32768.0  # int16 -> [-1, 1]
    if f.ndim == 2:                       # (N, kanal) -> ortalama ile mono
        f = f.mean(axis=1)
    if src_fs != target_fs:
        import torch
        import torchaudio.functional as AF
        w = torch.from_numpy(np.ascontiguousarray(f))
        # High-quality anti-aliased resampling (close to soxr-VHQ). The default
        # parameters don't suppress tones above the 8 kHz Nyquist enough; the
        # Kaiser window provides a narrow transition band.
        f = AF.resample(
            w, orig_freq=src_fs, new_freq=target_fs,
            lowpass_filter_width=64, rolloff=0.945,
            resampling_method="sinc_interp_kaiser", beta=14.769656459379492,
        ).numpy()
    # float -> int16: scale, round, and clip to avoid overflow.
    i16 = np.clip(np.round(f * 32768.0), -32768, 32767).astype(np.int16)
    return i16


def _coreaudio_extra_settings():
    """
    Return the setting that stops PortAudio from changing a device's global
    sample rate on macOS (if available). This keeps audio playing from a device
    from being interrupted when we connect to it. Returns None if unsupported.
    """
    settings_cls = getattr(sd, "CoreAudioSettings", None)
    if settings_cls is None:
        return None
    try:
        return settings_cls(change_device_parameters=False)
    except Exception:
        return None


# --- Core Audio system-audio capture (Swift helper) ---
TAP_DIR = Path(__file__).resolve().parent / "mac_audio_tap"
TAP_SRC = TAP_DIR / "system_audio_tap.swift"
TAP_BIN = TAP_DIR / "system_audio_tap"


def build_tap_binary():
    """
    Compile the Swift system-audio helper if needed and return its path. Raises
    RuntimeError if swiftc is missing or compilation fails.
    """
    if not TAP_SRC.exists():
        raise RuntimeError(f"tap source file not found: {TAP_SRC}")
    if TAP_BIN.exists() and TAP_BIN.stat().st_mtime >= TAP_SRC.stat().st_mtime:
        return TAP_BIN
    swiftc = shutil.which("swiftc")
    if not swiftc:
        raise RuntimeError("swiftc not found (Xcode / Command Line Tools required).")
    if not _QUIET:
        console.print("[dim]Building the system-audio helper (first run)…[/dim]")
    res = subprocess.run(
        [swiftc, "-O", str(TAP_SRC), "-o", str(TAP_BIN),
         "-framework", "CoreAudio", "-framework", "AudioToolbox", "-framework", "Foundation"],
        capture_output=True, text=True,
    )
    if res.returncode != 0 or not TAP_BIN.exists():
        raise RuntimeError(f"Swift build failed:\n{res.stderr.strip()}")
    return TAP_BIN


class DeviceRecorder:
    """Record from one input device (microphone) at its native rate via sounddevice.

    Audio is streamed straight to a raw PCM file (`raw_path`, native-rate int16) as
    it arrives, so an interrupted recording (crash / kill) can be recovered from
    disk. Nothing is buffered in memory."""

    KIND = "mic"
    RAW_DTYPE = "int16"          # native-endian 16-bit PCM, as written by tobytes()

    def __init__(self, index):
        self.index = index
        self.name = device_name(index)
        info = sd.query_devices(index, 'input')
        self.rate = int(round(info['default_samplerate']))
        self.channels = max(1, min(2, int(info['max_input_channels'])))
        self._stream = None
        self._raw = None
        self._raw_path = None
        self._level = 0.0  # instantaneous level for the live VU meter (0..1)
        self.meter_name = f"🎤 {self.name}"

    @property
    def label(self):
        return f"{self.name} ({self.rate} Hz)"

    def level(self):
        return self._level

    def manifest(self):
        """Describe the raw file so it can be re-read after an interruption."""
        return {"kind": self.KIND, "file": Path(self._raw_path).name,
                "rate": self.rate, "channels": self.channels, "dtype": self.RAW_DTYPE}

    def start(self, raw_path):
        # Unbuffered so every block reaches the kernel immediately; a killed
        # process then loses nothing already written (only a hard power-loss would).
        self._raw_path = str(raw_path)
        self._raw = open(self._raw_path, "wb", buffering=0)

        def callback(indata, n, t, status):
            if status and not _QUIET:
                console.log(f"[red]{status}[/red]")
            if self._raw is not None:
                try:
                    self._raw.write(indata.tobytes())
                except Exception:
                    pass
            if indata.size:
                peak = float(np.max(np.abs(indata))) / 32768.0
                # peak-hold + decay: the VU bar reacts to sound and falls smoothly.
                self._level = max(peak, self._level * 0.85)
        kwargs = dict(samplerate=self.rate, channels=self.channels,
                      dtype='int16', device=self.index, callback=callback)
        extra = _coreaudio_extra_settings()
        if extra is not None:
            kwargs['extra_settings'] = extra
        self._stream = sd.InputStream(**kwargs)
        self._stream.start()

    def stop(self):
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._raw is not None:
            try:
                self._raw.close()
            except Exception:
                pass
            self._raw = None

    def result(self):
        return _read_raw_pcm(self._raw_path, self.RAW_DTYPE, self.channels), self.rate


class TapRecorder:
    """
    Capture the ENTIRE system audio (speakers/Zoom/YouTube) via a macOS Core
    Audio process tap. Runs the Swift helper as a subprocess; playback KEEPS
    PLAYING (unmuted), and no BlackHole / rerouting is needed.
    """

    KIND = "system"
    RAW_DTYPE = "<f4"           # 32-bit little-endian float, as sent by the helper

    def __init__(self):
        self.label = "Sistem sesi (Core Audio tap)"
        self.meter_name = "🔊 Sistem sesi"
        self.rate = None
        self.channels = None
        self._proc = None
        self._reader = None
        self._stderr_reader = None
        self._raw = None
        self._raw_path = None
        self._stderr = []
        self._ready = threading.Event()
        self._error = None
        self._level = 0.0  # instantaneous level for the live VU meter (0..1)

    def level(self):
        return self._level

    def manifest(self):
        """Describe the raw file so it can be re-read after an interruption."""
        return {"kind": self.KIND, "file": Path(self._raw_path).name,
                "rate": self.rate, "channels": self.channels, "dtype": self.RAW_DTYPE}

    def start(self, raw_path):
        # Stream the tap's PCM straight to disk, unbuffered (see DeviceRecorder).
        self._raw_path = str(raw_path)
        self._raw = open(self._raw_path, "wb", buffering=0)
        binpath = build_tap_binary()  # compiles if needed; raises on failure
        self._proc = subprocess.Popen(
            [str(binpath)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0,
        )
        self._stderr_reader = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_reader.start()
        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader.start()
        # Wait for the header (sample rate/channels); if it never arrives, likely a permission issue.
        if not self._ready.wait(timeout=10):
            self.stop()
            raise RuntimeError(
                "system audio did not start (10s). 'System Audio Recording' "
                "permission may be needed: System Settings → Privacy & Security."
            )
        if self._error:
            self.stop()
            detail = " ".join(self._stderr).strip()
            raise RuntimeError(f"{self._error}{(': ' + detail) if detail else ''}")

    def _drain_stderr(self):
        try:
            for line in iter(self._proc.stderr.readline, b""):
                self._stderr.append(line.decode("utf-8", "replace").strip())
        except Exception:
            pass

    def _read_stdout(self):
        f = self._proc.stdout
        # 1) Read the header line: "samplerate=<r> channels=<c> format=f32le\n"
        header = b""
        while not header.endswith(b"\n"):
            b = f.read(1)
            if not b:
                self._error = "helper exited before sending a header"
                self._ready.set()
                return
            header += b
        try:
            parts = dict(p.split("=", 1) for p in header.decode().split())
            self.rate = int(parts["samplerate"])
            self.channels = int(parts["channels"])
        except Exception:
            self._error = f"invalid header: {header!r}"
            self._ready.set()
            return
        self._ready.set()
        # 2) Stream the remaining data (float32 PCM) to disk and update the level.
        while True:
            data = f.read(8192)
            if not data:
                break
            if self._raw is not None:
                try:
                    self._raw.write(data)
                except Exception:
                    pass
            arr = np.frombuffer(data, dtype="<f4")
            if arr.size:
                peak = float(np.max(np.abs(arr)))
                self._level = max(peak, self._level * 0.85)

    def stop(self):
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except Exception:
                self._proc.kill()
        if self._reader is not None:
            self._reader.join(timeout=2)
        if self._stderr_reader is not None:
            self._stderr_reader.join(timeout=1)
        # Close the raw file only after the reader thread has stopped writing.
        if self._raw is not None:
            try:
                self._raw.close()
            except Exception:
                pass
            self._raw = None

    def result(self):
        if not self.rate:
            return None, self.rate or 16000
        return _read_raw_pcm(self._raw_path, self.RAW_DTYPE, self.channels), self.rate


@contextlib.contextmanager
def _cbreak_mode():
    """Read single keystrokes (no Enter, no echo) while keeping Ctrl-C working."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        new = termios.tcgetattr(fd)
        new[3] &= ~(termios.ICANON | termios.ECHO)  # raw-ish, keep ISIG for Ctrl-C
        termios.tcsetattr(fd, termios.TCSADRAIN, new)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _read_key(timeout=0.1):
    """Return one keystroke or None on timeout. Special: ENTER, BACKSPACE, ESC."""
    try:
        r, _, _ = select.select([sys.stdin], [], [], timeout)
    except Exception:
        return None
    if not r:
        return None
    fd = sys.stdin.fileno()
    ch = os.read(fd, 1)
    if not ch:
        return None
    if ch == b"\x1b":                 # ESC, or an arrow/navigation sequence
        r2, _, _ = select.select([sys.stdin], [], [], 0.0008)
        if not r2:
            return "ESC"
        seq = os.read(fd, 8).decode("ascii", "ignore")
        nav = {
            "[A": "UP", "[B": "DOWN", "[C": "RIGHT", "[D": "LEFT",
            "OA": "UP", "OB": "DOWN", "OC": "RIGHT", "OD": "LEFT",
            "[5~": "PGUP", "[6~": "PGDN", "[H": "HOME", "[F": "END",
            "[1~": "HOME", "[4~": "END",
        }
        return nav.get(seq, "ESC_SEQ")
    if ch in (b"\r", b"\n"):
        return "ENTER"
    if ch in (b"\x7f", b"\x08"):
        return "BACKSPACE"
    try:
        return ch.decode("utf-8", "ignore")
    except Exception:
        return None


def _write_mono_wav(path, samples, target_fs=16000):
    """Write a mono int16 numpy array as a 16-bit WAV at target_fs."""
    with wave.open(str(path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)            # 16-bit
        wf.setframerate(target_fs)    # Whisper expects 16 kHz mono
        wf.writeframes(samples.tobytes())


def _wav_duration(path):
    """Duration of a WAV file in seconds (None if it cannot be read). Used to turn
    transcription progress into a real percentage without shelling out to ffprobe."""
    try:
        with wave.open(str(path), "rb") as wf:
            rate = wf.getframerate()
            return (wf.getnframes() / float(rate)) if rate else None
    except Exception:
        return None


def _read_raw_pcm(raw_path, dtype, channels):
    """Read a raw PCM capture file back into a numpy array (None if empty/missing).

    `dtype` is a numpy dtype string ('int16', '<f4'); `channels` reshapes the flat
    stream into interleaved frames. Shared by the live and recovery finalize paths."""
    if not raw_path or not Path(raw_path).exists():
        return None
    try:
        data = Path(raw_path).read_bytes()
    except Exception:
        return None
    if not data:
        return None
    arr = np.frombuffer(data, dtype=dtype)
    ch = int(channels) if channels else 1
    if ch > 1:
        arr = arr[:len(arr) // ch * ch].reshape(-1, ch)
    return arr


def _mix_write(pairs, audio_path, target_fs=16000, save_channels=False):
    """Resample each (kind, array, rate) source to 16k mono, mix, and write a mono
    WAV. Returns the number of samples written (0 if nothing captured).

    When `save_channels` is set, also write the per-source channels next to
    audio_path — `mic.wav` (mic) and `system.wav` (system) — so the speaker-labeling
    path can transcribe and diarize them separately."""
    resampled, mic_arr, sys_arr = [], None, None
    for kind, arr, rate in pairs:
        r = resample_to_target(arr, rate, target_fs)
        resampled.append(r)
        if kind == "system":
            sys_arr = r
        elif kind == "mic":
            mic_arr = r
    mixed = mix_to_mono(resampled)
    if mixed is None or len(mixed) == 0:
        return 0
    _write_mono_wav(audio_path, mixed, target_fs)
    if save_channels:
        d = Path(audio_path).parent
        if mic_arr is not None and len(mic_arr) > 0:
            _write_mono_wav(d / "mic.wav", mic_arr, target_fs)
        if sys_arr is not None and len(sys_arr) > 0:
            _write_mono_wav(d / "system.wav", sys_arr, target_fs)
    return len(mixed)


def _finalize_to_wav(started, audio_path, target_fs=16000, save_channels=False):
    """Finalize the live recorders (already stopped) into a mono WAV from their raw
    capture files. Returns the number of samples written (0 if nothing captured)."""
    pairs = []
    for src in started:
        arr, rate = src.result()
        pairs.append((getattr(src, "KIND", "mic"), arr, rate))
    return _mix_write(pairs, audio_path, target_fs, save_channels)


def _finalize_from_manifest(project_dir, sources, target_fs=16000):
    """Rebuild audio.wav from raw capture files described by a meta `capture`
    manifest (recovery of an interrupted recording). Writes per-source channel WAVs
    when more than one source is present, so a later diarized transcript is possible.
    Returns the number of samples written (0 if nothing recoverable)."""
    pairs = []
    for s in sources:
        arr = _read_raw_pcm(Path(project_dir) / s.get("file", ""),
                            s.get("dtype", "int16"), s.get("channels", 1))
        if arr is None:
            continue
        pairs.append((s.get("kind", "mic"), arr, int(s.get("rate", 16000))))
    save_channels = len({kind for kind, _, _ in pairs}) > 1
    return _mix_write(pairs, Path(project_dir) / "audio.wav", target_fs, save_channels)


def _clear_capture(project_dir):
    """Mark a recording as finalized: delete its raw capture files and clear the
    `capture.in_progress` flag in meta.json."""
    meta = _read_meta(project_dir)
    cap = meta.get("capture")
    if not isinstance(cap, dict):
        return
    for s in cap.get("sources", []):
        try:
            (Path(project_dir) / s.get("file", "")).unlink()
        except Exception:
            pass
    cap["in_progress"] = False
    _write_meta(project_dir, capture=cap)


def list_installed_apps():
    """Return sorted names of installed .app bundles (for the 'open with' picker)."""
    bases = ["/Applications", "/Applications/Utilities", "/System/Applications",
             "/System/Applications/Utilities", os.path.expanduser("~/Applications")]
    names = set()
    for b in bases:
        try:
            for entry in os.listdir(b):
                if entry.endswith(".app"):
                    names.add(entry[:-4])
        except Exception:
            pass
    return sorted(names, key=str.lower)


def _open_in_app(app, path):
    """Open `path` in the macOS app named `app`. Returns None on success or an
    error string on failure."""
    try:
        subprocess.run(["open", "-a", app, str(path)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return None
    except Exception as e:
        return str(e)


# Importable external media: audio is extracted from these and transcribed.
IMPORT_EXTS = (".mp4", ".mov", ".wav", ".mp3", ".m4a")


def _choose_media_file():
    """Open the native macOS file picker for an audio/video file.
    Returns the chosen POSIX path, or None if cancelled/unavailable."""
    osa = shutil.which("osascript")
    if not osa:
        return None
    script = (
        'set theFile to choose file with prompt "Select an audio or video file to transcribe" '
        'of type {"mp4","mov","wav","mp3","m4a","public.movie","public.audio"}\n'
        'POSIX path of theFile'
    )
    try:
        res = subprocess.run([osa, "-e", script], capture_output=True, text=True)
        if res.returncode != 0:
            return None  # user cancelled or an error occurred
        path = res.stdout.strip()
        return path or None
    except Exception:
        return None


def _media_duration(src):
    """Return media duration in seconds via ffprobe, or None if unavailable."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    try:
        res = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nokey=1:noprint_wrappers=1", str(src)],
            capture_output=True, text=True,
        )
        return float(res.stdout.strip())
    except Exception:
        return None


def _extract_audio(src, dest_wav, target_fs=16000, on_pct=None, duration=None):
    """Extract/convert `src` media to a 16 kHz mono PCM WAV at `dest_wav` using
    ffmpeg. If `on_pct` and `duration` are given, report extraction progress
    (0..1) by parsing ffmpeg's `-progress` output. Returns None on success or an
    error string."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return "ffmpeg not found — install it (e.g. 'brew install ffmpeg')"
    cmd = [ffmpeg, "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", str(target_fs),
           "-c:a", "pcm_s16le"]
    stream = bool(on_pct and duration)
    if stream:
        cmd += ["-progress", "pipe:1", "-nostats"]
    cmd += [str(dest_wav)]
    try:
        if stream:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.DEVNULL, text=True)
            for line in proc.stdout or []:
                line = line.strip()
                if line.startswith("out_time_us="):
                    try:
                        on_pct(min(1.0, int(line.split("=", 1)[1]) / 1e6 / duration))
                    except Exception:
                        pass
                elif line == "progress=end":
                    on_pct(1.0)
            proc.wait()
            rc = proc.returncode
        else:
            res = subprocess.run(cmd, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.PIPE, text=True)
            rc = res.returncode
        if rc != 0:
            return f"ffmpeg failed (code {rc})"
        if not Path(dest_wav).exists() or Path(dest_wav).stat().st_size == 0:
            return "ffmpeg produced no audio (no audio track?)"
        return None
    except Exception as e:
        return str(e)


def pick_device():
    """Pick a suitable device for the transformers pipeline (CUDA > MPS > CPU)."""
    import torch
    if torch.cuda.is_available():
        return "cuda:0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _ensure_model(language_code):
    """
    Load and cache the model for the given language if not already loaded.
    Thread-safe (locked) so background pre-warming and an actual transcription
    cannot load the same model twice. Returns when the model is ready.
    """
    global _hf_pipe, _cpp_model
    if language_code == "en":
        if _cpp_model is not None:
            return
        with _MODEL_LOCK:
            if _cpp_model is not None:
                return
            _silence_ml_logging()
            from huggingface_hub import hf_hub_download
            from pywhispercpp.model import Model
            if not _QUIET:
                console.print(f"[dim]Preparing English model ({EN_GGML_FILE})…[/dim]")
            model_path = hf_hub_download(repo_id=EN_GGML_REPO, filename=EN_GGML_FILE)
            # redirect_whispercpp_logs_to=None -> whisper.cpp C/Metal logs to /dev/null.
            _cpp_model = Model(
                model_path, print_progress=False, print_realtime=False,
                redirect_whispercpp_logs_to=None,
            )
    else:
        if _hf_pipe is not None:
            return
        with _MODEL_LOCK:
            if _hf_pipe is not None:
                return
            from transformers import pipeline
            _silence_ml_logging()
            device = pick_device()
            if not _QUIET:
                console.print(f"[dim]Preparing Turkish model ({TR_HF_MODEL}, {device})…[/dim]")
            _hf_pipe = pipeline("automatic-speech-recognition", model=TR_HF_MODEL, device=device)


def transcribe_audio(filepath, language_code, on_progress=None, duration=None):
    """
    Transcribe with the model for the selected language and return the text.
    If `on_progress` (a callable taking 0..1) and `duration` are given, report
    progress: for English (whisper.cpp) via each segment's end time / duration.
    The Turkish (transformers) path has no incremental hook, so it stays
    indeterminate.
    """
    _ensure_model(language_code)
    if language_code == "en":
        cb = None
        if on_progress and duration:
            def cb(seg):
                t1 = getattr(seg, "t1", None)            # centiseconds (10 ms units)
                if t1 is not None:
                    try:
                        on_progress(min(1.0, (t1 / 100.0) / duration))
                    except Exception:
                        pass
        segments = _cpp_model.transcribe(str(filepath), language="en",
                                         new_segment_callback=cb)
        return "".join(segment.text for segment in segments).strip()
    result = _hf_pipe(
        str(filepath),
        return_timestamps=True,
        generate_kwargs={"language": "turkish", "task": "transcribe"},
    )
    return result.get("text", "")


def transcribe_segments(filepath, language_code, on_progress=None, duration=None):
    """
    Like transcribe_audio, but return timestamped segments:
        [{"start": seconds, "end": seconds, "text": str}, …]
    Empty-text segments are dropped. Used by the speaker-labeling path, which needs
    timestamps to align transcript text with diarization turns.
    """
    _ensure_model(language_code)
    if language_code == "en":
        cb = None
        if on_progress and duration:
            def cb(seg):
                t1 = getattr(seg, "t1", None)            # centiseconds (10 ms units)
                if t1 is not None:
                    try:
                        on_progress(min(1.0, (t1 / 100.0) / duration))
                    except Exception:
                        pass
        segments = _cpp_model.transcribe(str(filepath), language="en",
                                         new_segment_callback=cb)
        out = []
        for s in segments:
            text = (s.text or "").strip()
            if not text:
                continue
            t0 = getattr(s, "t0", None)                  # centiseconds
            t1 = getattr(s, "t1", None)
            start = (t0 / 100.0) if t0 is not None else 0.0
            end = (t1 / 100.0) if t1 is not None else start
            out.append({"start": start, "end": end, "text": text})
        return out
    result = _hf_pipe(
        str(filepath),
        return_timestamps=True,
        generate_kwargs={"language": "turkish", "task": "transcribe"},
    )
    out = []
    for ch in result.get("chunks", []):
        text = (ch.get("text") or "").strip()
        if not text:
            continue
        ts = ch.get("timestamp") or (None, None)
        start = float(ts[0]) if ts[0] is not None else 0.0
        end = float(ts[1]) if ts[1] is not None else start
        out.append({"start": start, "end": end, "text": text})
    return out


# =========================== Speaker diarization (optional) ===========================

def _hf_token(cfg=None):
    """Hugging Face token from config ('hf_token') or the HF_TOKEN/HUGGINGFACE_TOKEN
    env vars. None if unset."""
    return ((cfg or {}).get("hf_token")
            or os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGINGFACE_TOKEN"))


def _ensure_diarizer(cfg=None):
    """
    Load and cache the pyannote diarization pipeline. Thread-safe (locked). Raises
    RuntimeError with an actionable message if pyannote, the token, or the gated
    model are unavailable — callers fail soft on that.
    """
    global _diar_pipe
    if _diar_pipe is not None:
        return _diar_pipe
    with _MODEL_LOCK:
        if _diar_pipe is not None:
            return _diar_pipe
        _silence_ml_logging()
        try:
            from pyannote.audio import Pipeline
        except Exception as e:
            raise RuntimeError("pyannote.audio not installed "
                               "(pip install pyannote.audio)") from e
        import torch
        token = _hf_token(cfg)
        if not _QUIET:
            console.print(f"[dim]Preparing speaker diarization ({PYANNOTE_MODEL})…[/dim]")
        pipe = Pipeline.from_pretrained(PYANNOTE_MODEL, use_auth_token=token)
        if pipe is None:
            raise RuntimeError(
                "could not load the diarization model — set a Hugging Face token "
                "(config 'hf_token' or HF_TOKEN env) and accept the terms at "
                f"huggingface.co/{PYANNOTE_MODEL}")
        try:
            pipe.to(torch.device(pick_device()))
        except Exception:
            pass                                  # CPU fallback is fine
        _diar_pipe = pipe
    return _diar_pipe


def diarize(wav_path, cfg=None):
    """Return speaker turns [(start, end, raw_label), …] for a mono WAV, sorted by
    start time. Raises on failure; the caller decides the fallback."""
    pipe = _ensure_diarizer(cfg)
    annotation = pipe(str(wav_path))
    turns = [(float(seg.start), float(seg.end), str(spk))
             for seg, _, spk in annotation.itertracks(yield_label=True)]
    turns.sort(key=lambda t: t[0])
    return turns


def _overlap(a0, a1, b0, b1):
    """Length of the overlap between intervals [a0,a1] and [b0,b1] (0 if disjoint)."""
    return max(0.0, min(a1, b1) - max(a0, b0))


def assign_speakers(segments, turns, default=None):
    """Tag each transcript segment with the diarization turn it overlaps most.
    Returns new segments with an added 'speaker' (raw label, or `default`)."""
    out = []
    for seg in segments:
        best, best_ov = default, 0.0
        for (t0, t1, spk) in turns:
            ov = _overlap(seg["start"], seg["end"], t0, t1)
            if ov > best_ov:
                best, best_ov = spk, ov
        s = dict(seg)
        s["speaker"] = best
        out.append(s)
    return out


def _friendly_speaker_map(segments):
    """Map raw diarization labels to 'Speaker 1 / Speaker 2 / …' in first-appearance
    order (by start time)."""
    mapping, n = {}, 0
    for seg in sorted(segments, key=lambda s: s["start"]):
        raw = seg.get("speaker")
        if raw is None or raw in mapping:
            continue
        n += 1
        mapping[raw] = f"Speaker {n}"
    return mapping


def build_labeled_recording(mic_segments, system_segments, system_turns):
    """Combine mic segments (labeled 'Me') with diarized system segments
    (labeled 'Speaker N'), ordered by start time. Returns (labeled_segments,
    speaker_map) where each segment has a 'label'."""
    labeled = [dict(s, label="Me") for s in mic_segments]
    assigned = assign_speakers(system_segments, system_turns)
    smap = _friendly_speaker_map(assigned)
    for s in assigned:
        labeled.append(dict(s, label=smap.get(s.get("speaker"), "Speaker ?")))
    labeled.sort(key=lambda s: s["start"])
    return labeled, smap


def build_labeled_diarized(segments, turns):
    """Label a single diarized file's segments as 'Speaker N' (no 'Me'). Used for
    imports. Returns (labeled_segments, speaker_map)."""
    assigned = assign_speakers(segments, turns)
    smap = _friendly_speaker_map(assigned)
    labeled = [dict(s, label=smap.get(s.get("speaker"), "Speaker ?")) for s in assigned]
    labeled.sort(key=lambda s: s["start"])
    return labeled, smap


def render_labeled_transcript(labeled):
    """Render labeled segments to text with [Label] prefixes, merging consecutive
    same-label segments into one paragraph separated by blank lines."""
    blocks, cur_label, parts = [], None, []
    for s in labeled:
        text = s["text"].strip()
        if not text:
            continue
        label = s.get("label", "")
        if label != cur_label:
            if parts:
                blocks.append((cur_label, " ".join(parts)))
            cur_label, parts = label, [text]
        else:
            parts.append(text)
    if parts:
        blocks.append((cur_label, " ".join(parts)))
    return "\n\n".join(f"[{lbl}] {txt}" for lbl, txt in blocks)


# =========================== Recordings (metadata + listing) ===========================

def _read_meta(project_dir):
    """Read a recording's meta.json ({name, language, …}); {} if missing/bad."""
    try:
        with open(Path(project_dir) / "meta.json", encoding="utf-8") as f:
            m = json.load(f)
        return m if isinstance(m, dict) else {}
    except Exception:
        return {}


def _write_meta(project_dir, **fields):
    """Merge non-None `fields` into the recording's meta.json."""
    meta = _read_meta(project_dir)
    meta.update({k: v for k, v in fields.items() if v is not None})
    try:
        with open(Path(project_dir) / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def _fmt_created(dirname):
    """Format a 'YYYY-MM-DD_HH-MM-SS' folder name as 'YYYY-MM-DD HH:MM'."""
    try:
        return datetime.strptime(dirname, "%Y-%m-%d_%H-%M-%S").strftime("%Y-%m-%d %H:%M")
    except Exception:
        return dirname


def list_recordings(base_path):
    """List recording projects under base_path (newest first). Each item:
    {dir, name, language, created, sortkey, has_transcript}."""
    recs = []
    try:
        entries = [d for d in Path(base_path).iterdir() if d.is_dir()]
    except Exception:
        entries = []
    for d in entries:
        has_tx = (d / "transcription.txt").exists()
        if not has_tx and not (d / "audio.wav").exists():
            continue
        meta = _read_meta(d)
        recs.append({
            "dir": d,
            "name": meta.get("name", "") or "",
            "language": meta.get("language", "") or "",
            "created": _fmt_created(d.name),
            "sortkey": d.name,
            "has_transcript": has_tx,
        })
    recs.sort(key=lambda r: r["sortkey"], reverse=True)
    return recs


def _recover_interrupted(base_path):
    """Rebuild audio.wav for recordings that were killed mid-capture (their meta
    still has capture.in_progress). Returns the number recovered. Folders with no
    usable raw data are removed. Run once at startup."""
    recovered = 0
    try:
        entries = [d for d in Path(base_path).iterdir() if d.is_dir()]
    except Exception:
        return 0
    for d in entries:
        cap = _read_meta(d).get("capture")
        if not isinstance(cap, dict) or not cap.get("in_progress"):
            continue
        if (d / "audio.wav").exists():
            _clear_capture(d)                 # finalized but flag never cleared
            continue
        sources = cap.get("sources", [])
        if not any((d / s.get("file", "")).exists() for s in sources):
            shutil.rmtree(d, ignore_errors=True)   # nothing captured before the kill
            continue
        try:
            n = _finalize_from_manifest(d, sources)
        except Exception:
            n = 0
        if n:
            _clear_capture(d)
            recovered += 1
        else:
            shutil.rmtree(d, ignore_errors=True)
    return recovered


def _rec_display_name(rec):
    return rec["name"] or "Untitled"


# =========================== Full-screen TUI ===========================

def _resolve_mic_index(cfg):
    """Resolve the saved mic device name to a current index; fall back to first."""
    devices = list_input_devices()
    name = cfg.get("input_device")
    if name:
        for idx, dev_name in devices:
            if dev_name == name:
                return idx
    return devices[0][0] if devices else None


class _TuiState:
    """All UI/app state for the full-screen interface."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.base_path = Path(os.path.expanduser(cfg.get("base_path") or str(DEFAULT_BASE_PATH)))
        self.language = cfg.get("language") or "tr"
        self.mic_index = _resolve_mic_index(cfg)
        self.capture_system = bool(cfg.get("capture_system_audio", False))
        self.diarize = bool(cfg.get("diarize", False))
        # modes: menu | recording | preparing | transcribing | importing |
        #        viewer | name_input | rename | mic_picker | app_picker | path_edit
        self.mode = "menu"
        self.status = "Ready."
        self.last_transcript = ""
        self.last_project = None
        # recording-time
        self.recorders = []
        self.project_dir = None
        self.rec_start = 0.0
        # pickers
        self.devices = []
        self.path_buffer = ""
        self.apps = []
        self.app_filter = ""
        # App used to open the transcript after each run. Defaults to Sublime Text
        # if installed (the user's stated preference); changeable via the picker.
        if "open_app" in cfg:
            self.open_app = cfg.get("open_app")
        else:
            self.open_app = "Sublime Text" if "Sublime Text" in list_installed_apps() else None
        # background model pre-warming: lang -> "loading" | "ready" | "error: …"
        self.model_state = {}
        # background file-import job progress
        self.import_src = ""
        self.import_phase = ""        # "extract" | "transcribe"
        self.import_pct = None        # 0..1, or None for indeterminate
        self.import_phase_start = 0.0
        # background transcription job progress (record → [q], and [t] on a
        # recording without a transcript); rendered by _transcribe_panel
        self.tx_name = ""
        self.tx_language = self.language
        self.tx_steps = []            # ordered subset of _TX_LABELS' keys
        self.tx_phase = ""            # the step currently running
        self.tx_pct = None            # 0..1, or None for indeterminate
        self.tx_phase_start = 0.0
        try:
            self.base_path.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        # menu navigation
        self.menu_index = 0
        self.menu_expanded = {"recordings"}     # which groups are open
        self.recordings = list_recordings(self.base_path)
        # transcript viewer
        self.viewer_rec = None
        self.viewer_text = ""
        self.viewer_scroll = 0
        # naming a new recording/import, and renaming existing ones
        self.name_buffer = ""
        self.pending_action = None              # "record" | "import"
        self.pending_name = ""
        self.rename_buffer = ""
        self.rename_target = None               # project dir being renamed
        self.rename_return = "menu"
        # audio playback (afplay) + delete confirmation
        self.player_proc = None
        self.player_rec_dir = None
        self.player_name = ""
        self.delete_target = None
        self.delete_name = ""
        self.delete_return = "menu"


def _player_active(state):
    return state.player_proc is not None and state.player_proc.poll() is None


def _stop_play(state):
    if state.player_proc is not None:
        try:
            if state.player_proc.poll() is None:
                state.player_proc.terminate()
        except Exception:
            pass
    state.player_proc = None
    state.player_rec_dir = None


def _toggle_play(state, rec):
    """Play the recording's audio with afplay, or stop it if it's already playing."""
    if _player_active(state) and state.player_rec_dir == rec["dir"]:
        _stop_play(state)
        state.status = "Playback stopped"
        return
    _stop_play(state)                       # stop any other playback first
    ap = rec["dir"] / "audio.wav"
    if not ap.exists():
        state.status = "No audio file to play"
        return
    afplay = shutil.which("afplay")
    if not afplay:
        state.status = "afplay not found"
        return
    try:
        state.player_proc = subprocess.Popen(
            [afplay, str(ap)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        state.player_rec_dir = rec["dir"]
        state.player_name = _rec_display_name(rec)
        state.status = f"Playing “{state.player_name}”"
    except Exception as e:
        state.status = f"Play error: {e}"


def _delete_recording(state, target):
    if _player_active(state) and state.player_rec_dir == target:
        _stop_play(state)
    try:
        shutil.rmtree(target)
        state.status = "Recording deleted"
    except Exception as e:
        state.status = f"Delete error: {e}"
    state.recordings = list_recordings(state.base_path)
    state.menu_index = min(state.menu_index, max(0, len(_menu_items(state)) - 1))


def _warm_model_async(state, language):
    """Load the transcription model for `language` in the background so the first
    transcript is fast. Updates state.model_state for the header indicator."""
    if state.model_state.get(language) in ("loading", "ready"):
        return
    state.model_state[language] = "loading"

    def run():
        try:
            _ensure_model(language)
            state.model_state[language] = "ready"
        except Exception as e:
            state.model_state[language] = f"error: {e}"

    threading.Thread(target=run, daemon=True).start()


def _meters_markup(recorders):
    width = 30
    rows = []
    for r in recorders:
        lvl = max(0.0, min(1.0, r.level()))
        filled = int(round(lvl * width))
        color = "red" if lvl >= 0.85 else "yellow" if lvl >= 0.5 else "green"
        name = getattr(r, "meter_name", getattr(r, "label", "?"))
        bar = f"[{color}]{'█' * filled}[/{color}][dim]{'░' * (width - filled)}[/dim]"
        tag = "" if lvl > 0.01 else "  [dim]no signal[/dim]"
        rows.append(f"{name:<18}{bar} {int(lvl * 100):3d}%{tag}")
    return "\n".join(rows) if rows else "[dim](no sources)[/dim]"


_SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def _spinner(elapsed):
    return _SPIN[int(elapsed * 10) % len(_SPIN)]


def _progress_bar(pct, width=30):
    pct = max(0.0, min(1.0, pct))
    filled = int(round(pct * width))
    return f"[cyan]{'█' * filled}[/cyan][dim]{'░' * (width - filled)}[/dim]"


_MODEL_NOTE = "  [dim](model loads on first use)[/dim]"


def _step_line(label, status, pct=None, elapsed=0.0, note=""):
    """One row of a step list: "done", "pending" (not started), or "active" — a
    progress bar when `pct` is known, a spinner with the elapsed seconds when it is
    not. Shared by the import and transcription progress panels."""
    if status == "done":
        return f"  [green]{label:<26}✓ done[/green]"
    if status == "pending":
        return f"  [dim]{label}[/dim]"
    meter = (f"{_spinner(elapsed)}  {elapsed:4.0f}s" if pct is None
             else f"{_progress_bar(pct)} {int(pct * 100):3d}%  {elapsed:4.0f}s")
    return f"  {label:<26}{meter}{note}"


def _import_panel(state):
    """Two-phase import progress: extract (real %), then transcribe (real % for
    English, spinner+elapsed otherwise)."""
    lines = [f"[bold]Importing[/bold] {state.import_src}", ""]
    elapsed = time.monotonic() - state.import_phase_start if state.import_phase_start else 0.0

    if state.import_phase == "extract":
        lines.append(_step_line("Step 1/2  Extract audio", "active", state.import_pct, elapsed))
        lines.append(_step_line("Step 2/2  Transcribe", "pending"))
    elif state.import_phase == "transcribe":
        lines.append(_step_line("Step 1/2  Extract audio", "done"))
        lines.append(_step_line("Step 2/2  Transcribe", "active", state.import_pct, elapsed,
                                _MODEL_NOTE if state.import_pct is None else ""))
    elif state.import_phase == "diarize":
        lines.append(_step_line("Step 1  Extract audio", "done"))
        lines.append(_step_line("Step 2  Transcribe", "done"))
        lines.append(_step_line("Step 3  Label speakers", "active", None, elapsed))
    else:
        lines.append(f"  Preparing…  {_spinner(elapsed)}")
    return Panel(Text.from_markup("\n".join(lines)), title="Import", border_style="yellow")


_TX_LABELS = {"finalize": "Save audio", "diarize": "Label speakers",
              "transcribe": "Transcribe"}


def _tx_begin(state, phase, pct=None, steps=None):
    """Enter a step of the background transcription job, restarting its clock.
    Pass `steps` (the full ordered key list) when the plan is set or changes."""
    if steps is not None:
        state.tx_steps = steps
    state.tx_phase = phase
    state.tx_pct = pct
    state.tx_phase_start = time.monotonic()


def _transcribe_panel(state):
    """Live progress for the transcription job — which step is running, for how
    long, and (English) how far in. The work happens on a worker thread, so this
    keeps animating: pressing [q] switches to it on the very next frame."""
    elapsed = time.monotonic() - state.tx_phase_start if state.tx_phase_start else 0.0
    steps = state.tx_steps or ["transcribe"]
    active = steps.index(state.tx_phase) if state.tx_phase in steps else 0
    lines = [f"[bold]Transcribing[/bold] “{state.tx_name or 'recording'}”"
             f"     [dim]Language[/dim] {(state.tx_language or '').upper()}", ""]
    for i, key in enumerate(steps):
        label = f"Step {i + 1}/{len(steps)}  {_TX_LABELS.get(key, key)}"
        if i < active:
            lines.append(_step_line(label, "done"))
        elif i > active:
            lines.append(_step_line(label, "pending"))
        else:
            note = _MODEL_NOTE if key == "transcribe" and state.tx_pct is None else ""
            lines.append(_step_line(label, "active", state.tx_pct, elapsed, note))
    return Panel(Text.from_markup("\n".join(lines)),
                 title="Please wait", border_style="yellow")


def _model_label(state):
    ms = state.model_state.get(state.language)
    if ms == "ready":
        return "[green]ready[/green]"
    if ms == "loading":
        return "[yellow]loading…[/yellow]"
    if ms and ms.startswith("error"):
        return "[red]error[/red]"
    return "[dim]—[/dim]"


def _tui_header(state):
    clock = datetime.now().strftime("%H:%M:%S")
    mic = device_name(state.mic_index) if state.mic_index is not None else "—"
    sysv = "[green]on[/green]" if state.capture_system else "[dim]off[/dim]"
    diar = "[green]on[/green]" if state.diarize else "[dim]off[/dim]"
    line1 = (f"[bold]Language[/bold] {state.language.upper()}     "
             f"[bold]Mic[/bold] {mic}     [bold]System audio[/bold] {sysv}     "
             f"[bold]Speakers[/bold] {diar}     "
             f"[bold]Model[/bold] {_model_label(state)}")
    open_with = state.open_app or "[dim]off[/dim]"
    line2 = f"[dim]Folder[/dim] {state.base_path}     [dim]Open with[/dim] {open_with}"
    return Panel(Text.from_markup(line1 + "\n" + line2),
                 title="🎙  Audiocript", title_align="left",
                 subtitle=clock, subtitle_align="right", border_style="cyan")


def _menu_items(state):
    """Build the flat list of visible menu rows from the group/expanded state."""
    items = [
        {"kind": "action", "action": "record", "label": "  New recording"},
        {"kind": "action", "action": "import", "label": "  Import file"},
    ]
    rec_open = "recordings" in state.menu_expanded
    items.append({"kind": "group", "group": "recordings",
                  "label": f"{'▾' if rec_open else '▸'} Recordings ({len(state.recordings)})"})
    if rec_open:
        if state.recordings:
            for rec in state.recordings:
                items.append({"kind": "recording", "rec": rec})
        else:
            items.append({"kind": "info", "label": "      (no recordings yet)"})
    set_open = "settings" in state.menu_expanded
    items.append({"kind": "group", "group": "settings",
                  "label": f"{'▾' if set_open else '▸'} Settings"})
    if set_open:
        mic = device_name(state.mic_index) if state.mic_index is not None else "—"
        items += [
            {"kind": "setting", "setting": "language", "label": f"      Language: {state.language.upper()}"},
            {"kind": "setting", "setting": "mic", "label": f"      Microphone: {mic}"},
            {"kind": "setting", "setting": "system", "label": f"      System audio: {'on' if state.capture_system else 'off'}"},
            {"kind": "setting", "setting": "diarize", "label": f"      Speaker labels: {'on' if state.diarize else 'off'}"},
            {"kind": "setting", "setting": "openwith", "label": f"      Open with: {state.open_app or 'off'}"},
            {"kind": "setting", "setting": "folder", "label": f"      Folder: {state.base_path}"},
        ]
    items.append({"kind": "action", "action": "quit", "label": "  Quit"})
    return items


def _menu_row_text(it, width):
    """Plain display text for a menu row (no markup; safe for arbitrary names)."""
    if it["kind"] == "recording":
        rec = it["rec"]
        nm = _rec_display_name(rec)
        lang = (rec["language"] or "").upper()
        tx = "" if rec["has_transcript"] else " (no transcript)"
        return f"      {nm[:30]:<30}  {rec['created']}  {lang}{tx}"
    return it["label"]


def _render_menu(state):
    items = _menu_items(state)
    if state.menu_index >= len(items):
        state.menu_index = len(items) - 1
    if state.menu_index < 0:
        state.menu_index = 0
    # Window the list so the selected row stays visible on small terminals.
    h = console.size.height
    visible = max(5, h - 12)
    n = len(items)
    start = 0 if n <= visible else min(max(0, state.menu_index - visible // 2), n - visible)
    window = items[start:start + visible]
    width = max(20, console.size.width - 6)
    lines = []
    if start > 0:
        lines.append(Text("  ↑ more", style="dim"))
    for i, it in enumerate(window, start=start):
        t = Text(_menu_row_text(it, width))
        if it["kind"] == "group":
            t.stylize("bold")
        if i == state.menu_index:
            t.stylize("reverse")
        lines.append(t)
    if start + visible < n:
        lines.append(Text("  ↓ more", style="dim"))
    return Panel(Group(*lines) if lines else Text(""),
                 title="Menu", title_align="left", border_style="cyan")


def _render_viewer(state):
    import textwrap
    rec = state.viewer_rec or {}
    title = f"{_rec_display_name(rec)} — transcript"
    width = max(20, console.size.width - 6)
    raw = (state.viewer_text or "").splitlines() or ["(empty transcript)"]
    wrapped = []
    for ln in raw:
        wrapped += textwrap.wrap(ln, width) or [""]
    h = console.size.height
    visible = max(5, h - 12)
    total = len(wrapped)
    state.viewer_scroll = max(0, min(state.viewer_scroll, max(0, total - visible)))
    window = wrapped[state.viewer_scroll:state.viewer_scroll + visible]
    lines = [Text(ln) for ln in window]
    if total > visible:
        pos = f"  [{state.viewer_scroll + 1}-{min(total, state.viewer_scroll + visible)}/{total}]"
        title += pos
    return Panel(Group(*lines) if lines else Text(""),
                 title=title, title_align="left", border_style="green")


def _tui_body(state):
    if state.mode == "importing":
        return _import_panel(state)
    if state.mode == "recording":
        elapsed = time.monotonic() - state.rec_start
        mm, ss = divmod(int(elapsed), 60)
        title = f"[blink bold red]●[/] [bold]REC[/] {mm:02d}:{ss:02d}"
        return Panel(Text.from_markup(_meters_markup(state.recorders)),
                     title=title, title_align="left", border_style="red")
    if state.mode == "preparing":
        msg = (f"[bold yellow]Preparing…[/bold yellow]\n\n"
               f"[dim]{state.status}[/dim]\n\n"
               f"[dim]The first run may compile a helper, ask for a permission, or "
               f"load a model; this can take a moment.[/dim]")
        return Panel(Text.from_markup(msg), title="Please wait", border_style="yellow")
    if state.mode == "transcribing":
        return _transcribe_panel(state)
    if state.mode == "mic_picker":
        lines = []
        for i, (idx, name) in enumerate(state.devices, start=1):
            marker = "[green]›[/green]" if idx == state.mic_index else " "
            sel = i if i <= 9 else "·"
            lines.append(f"{marker} [cyan]{sel}[/cyan]  {name}")
        note = "" if len(state.devices) <= 9 else "\n[dim](only 1–9 selectable)[/dim]"
        body = "\n".join(lines) if lines else "[dim]no input devices[/dim]"
        return Panel(Text.from_markup(body + note),
                     title="Select microphone", border_style="cyan")
    if state.mode == "path_edit":
        body = f"Recordings folder:\n\n[bold]{state.path_buffer}[/bold][blink]▏[/blink]"
        return Panel(Text.from_markup(body),
                     title="Edit folder", border_style="cyan")
    if state.mode == "app_picker":
        flt = state.app_filter.lower()
        matches = [a for a in state.apps if flt in a.lower()]
        shown = matches[:9]
        lines = ["[cyan]0[/cyan]  [dim]Off (don't auto-open)[/dim]"]
        for i, a in enumerate(shown, start=1):
            marker = "[green]›[/green]" if a == state.open_app else " "
            lines.append(f"{marker} [cyan]{i}[/cyan]  {a}")
        extra = f"\n[dim]…and {len(matches) - 9} more — type to filter[/dim]" if len(matches) > 9 else ""
        filt = f"\n\n[dim]filter:[/dim] {state.app_filter}[blink]▏[/blink]"
        return Panel(Text.from_markup("\n".join(lines) + extra + filt),
                     title="Open transcript with", border_style="cyan")
    if state.mode == "viewer":
        return _render_viewer(state)
    if state.mode == "name_input":
        body = Group(
            Text.from_markup("Name this recording [dim](optional — Enter to skip)[/dim]:"),
            Text(""),
            Text(state.name_buffer + "▏"),
        )
        return Panel(body, title="Name", border_style="cyan")
    if state.mode == "rename":
        body = Group(
            Text("New name:"),
            Text(""),
            Text(state.rename_buffer + "▏"),
        )
        return Panel(body, title="Rename recording", border_style="cyan")
    if state.mode == "confirm_delete":
        body = Group(
            Text.from_markup(f"Delete [bold]“{state.delete_name or 'Untitled'}”[/bold] and its files?"),
            Text(""),
            Text.from_markup("[red]This cannot be undone.[/red]   Press [bold]y[/bold] to delete, any other key to cancel."),
        )
        return Panel(body, title="Delete recording", border_style="red")
    # default: the main menu
    return _render_menu(state)


def _tui_footer(state):
    keymap = {
        "preparing": "please wait…",
        "importing": "importing… please wait",
        "recording": "[q] Stop & transcribe",
        "transcribing": "transcribing… please wait",
        "viewer": "↑/↓ scroll   PgUp/PgDn page   Enter open in app   p play/stop   t transcribe   r rename   d delete   Esc back",
        "name_input": "[Enter] Start   [Esc] Cancel",
        "rename": "[Enter] Save   [Backspace] Delete   [Esc] Cancel",
        "confirm_delete": "[y] Delete   any other key: Cancel",
        "mic_picker": "[1-9] Select   [Esc] Cancel",
        "path_edit": "[Enter] Save   [Backspace] Delete   [Esc] Cancel",
        "app_picker": "[1-9] Select   [0] Off   [Enter] First match   type to filter   [Esc] Cancel",
    }
    if state.mode == "menu":
        items = _menu_items(state)
        sel = items[state.menu_index] if 0 <= state.menu_index < len(items) else None
        if sel and sel["kind"] == "recording":
            tx = "" if sel["rec"]["has_transcript"] else "t transcribe   "
            keystr = f"↑/↓ move   Enter view   p play/stop   {tx}r rename   d delete   q quit"
        else:
            keystr = "↑/↓ move   Enter open/expand   →/← expand/collapse   q quit"
    else:
        keystr = keymap.get(state.mode, "")
    keys = Text(keystr)                               # plain Text: brackets shown literally
    if _player_active(state):
        status = Text(f"▶ playing “{state.player_name}”  ·  p to stop", style="green")
    else:
        status = Text(state.status or "", style="dim")
    return Panel(Group(keys, status), border_style="blue")


def _tui_render(state):
    layout = Layout()
    layout.split_column(
        Layout(_tui_header(state), name="header", size=4),
        Layout(_tui_body(state), name="body"),
        Layout(_tui_footer(state), name="footer", size=4),
    )
    return layout


def _start_recording(state, live):
    """
    Set up the recording sources, keeping the user informed during the (possibly
    slow) initialization — opening the mic, and especially starting the system
    audio tap, which on the first run may compile a helper or wait on a macOS
    permission. Shows a "Preparing…" screen, then switches to the recording view
    only once capture has actually begun.
    """
    def announce(msg):
        state.status = msg
        live.update(_tui_render(state))

    _stop_play(state)                 # don't capture our own playback
    state.mode = "preparing"
    announce("Creating project folder…")

    project_dir = state.base_path / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    try:
        project_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        state.status = f"Folder error: {e}"
        state.mode = "menu"
        return
    _write_meta(project_dir, name=state.pending_name, language=state.language)

    started = []
    sys_failed = False

    # Microphone — critical: abort the recording if it can't open.
    if state.mic_index is not None:
        announce(f"Opening microphone ({device_name(state.mic_index)})…")
        try:
            mic = DeviceRecorder(state.mic_index)
            mic.start(project_dir / "mic.raw")
            started.append(mic)
        except Exception as e:
            for r in started:
                try:
                    r.stop()
                except Exception:
                    pass
            state.status = f"Microphone error: {e}"
            state.mode = "menu"
            try:
                project_dir.rmdir()
            except Exception:
                pass
            return

    # System audio — best effort: continue mic-only if it can't start.
    if state.capture_system:
        announce("Starting system audio… (first run may compile a helper or ask for permission)")
        try:
            tap = TapRecorder()
            tap.start(project_dir / "system.raw")
            started.append(tap)
        except Exception as e:
            sys_failed = True
            announce(f"System audio unavailable ({e}); continuing mic-only")

    if not started:
        state.status = "No microphone available."
        state.mode = "menu"
        try:
            project_dir.rmdir()
        except Exception:
            pass
        return

    # Record how the raw files can be re-read if this recording is interrupted
    # (crash / kill) before it is finalized. Recovered on the next launch.
    _write_meta(project_dir, capture={
        "in_progress": True,
        "sources": [r.manifest() for r in started],
    })

    state.recorders = started
    state.project_dir = project_dir
    state.rec_start = time.monotonic()
    state.mode = "recording"
    if not sys_failed:
        state.status = "Recording started — speak now."


def _stop_and_transcribe(state, live):
    """Handle [q] during a recording. Switches to the transcription progress screen
    right away and hands the slow part — mixing the raw capture into audio.wav, then
    transcribing it — to a worker thread, so the UI keeps animating instead of
    freezing on the last recording frame with no sign that anything started."""
    diarize_system = state.diarize and any(isinstance(r, TapRecorder)
                                           for r in state.recorders)
    project_dir, language = state.project_dir, state.language
    name, save_channels = state.pending_name, state.diarize
    state.tx_name = name or project_dir.name
    state.tx_language = language
    _tx_begin(state, "finalize",
              steps=["finalize"] + (["diarize"] if diarize_system else []) + ["transcribe"])
    state.status = "Stopping the recording…"
    state.mode = "transcribing"
    live.update(_tui_render(state))       # visible feedback on the keypress itself
    threading.Thread(
        target=_stop_worker,
        args=(state, project_dir, language, name, save_channels, diarize_system),
        daemon=True).start()


def _stop_worker(state, project_dir, language, name, save_channels, diarize_system):
    """Background half of [q]: stop the recorders, mix their raw capture files into
    audio.wav, then transcribe and save. Reports progress through state.tx_*."""
    for r in state.recorders:
        try:
            r.stop()
        except Exception:
            pass
    audio_path = project_dir / "audio.wav"
    n = _finalize_to_wav(state.recorders, audio_path, save_channels=save_channels)
    state.recorders = []
    if not n:
        state.status = "No audio captured; recording discarded."
        shutil.rmtree(project_dir, ignore_errors=True)
        state.mode = "menu"
        return
    # audio.wav is safely on disk now — drop the larger raw files and clear the flag.
    _clear_capture(project_dir)
    _transcribe_and_save(state, project_dir, audio_path, language, name,
                         diarize_system=diarize_system)


def _save_and_open(state, project_dir, text, name=None, language=None, speaker_map=None):
    """Write transcription.txt + meta, refresh the recordings list, and open the
    transcript in the chosen app. Sets state.status; does not change state.mode.

    `name`/`language` default to the pending ones on state (the import flow); the
    transcription jobs pass the recording's own, since they run in the background
    and must not depend on settings the user may change meanwhile.

    `speaker_map` (raw label -> 'Speaker N'), when present, is the diarization
    result; its friendly names are stored in meta.json for possible later renaming."""
    name = state.pending_name if name is None else name
    language = state.language if language is None else language
    transcript_path = project_dir / "transcription.txt"
    try:
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    except Exception as e:
        state.status = f"Save error: {e}"
        return
    speakers = sorted(speaker_map.values()) if speaker_map else None
    _write_meta(project_dir, name=name, language=language, speakers=speakers)
    state.recordings = list_recordings(state.base_path)
    label = name or project_dir.name
    state.status = f"Saved “{label}”"
    if state.open_app:
        err = _open_in_app(state.open_app, transcript_path)
        state.status = (f"Saved — could not open in {state.open_app}"
                        if err else f"Saved “{label}” & opened in {state.open_app}")


def _transcribe_and_save(state, project_dir, audio_path, language, name,
                         diarize_system=False):
    """Transcribe `audio_path`, save, and open. Runs on a worker thread and reports
    its step/percentage through state.tx_* for _transcribe_panel to render.

    When `diarize_system` is set, transcribe the mic channel as 'Me' and diarize +
    transcribe the system channel into 'Speaker N', producing a labeled transcript.
    Falls back to the plain single-file transcription if diarization is unavailable
    or the channel files are missing."""
    try:
        if diarize_system:
            text, speaker_map = _labeled_recording_text(state, project_dir, language)
        else:
            text, speaker_map = _plain_transcript(state, audio_path, language), None
    except Exception as e:
        state.status = f"Transcription error: {e}"
        state.mode = "menu"
        return
    _save_and_open(state, project_dir, text, name=name, language=language,
                   speaker_map=speaker_map)
    state.mode = "menu"


def _plain_transcript(state, audio_path, language):
    """Transcribe a single WAV, driving the panel's Transcribe step (a real
    percentage for English, a spinner for Turkish — see transcribe_audio)."""
    dur = _wav_duration(audio_path)
    _tx_begin(state, "transcribe", pct=0.0 if (language == "en" and dur) else None)
    return transcribe_audio(audio_path, language, duration=dur,
                            on_progress=lambda p: setattr(state, "tx_pct", p))


def _transcribe_existing(state, live, rec):
    """Transcribe an already-recorded audio.wav that has no transcript yet (e.g. a
    recording recovered after an interrupted capture). Uses the recording's own
    saved language/name, and diarizes if speaker labels are on and channel files
    are present. Runs in a worker thread so the progress screen stays live."""
    project_dir = rec["dir"]
    audio_path = project_dir / "audio.wav"
    if not audio_path.exists():
        state.status = "No audio to transcribe"
        return
    language = rec.get("language") or state.language
    name = rec.get("name") or ""
    diarize_system = state.diarize and (project_dir / "system.wav").exists()
    state.tx_name = name or project_dir.name
    state.tx_language = language
    _tx_begin(state, "diarize" if diarize_system else "transcribe",
              steps=(["diarize"] if diarize_system else []) + ["transcribe"])
    state.status = "Starting transcription…"
    state.mode = "transcribing"
    live.update(_tui_render(state))
    threading.Thread(
        target=_transcribe_and_save,
        args=(state, project_dir, audio_path, language, name),
        kwargs={"diarize_system": diarize_system}, daemon=True).start()


def _labeled_recording_text(state, project_dir, language):
    """Build the labeled transcript for a recording: mic → 'Me', system → diarized
    'Speaker N'. Returns (text, speaker_map). Falls back to a plain transcript of
    audio.wav (no labels) if diarization fails or the channel files are absent."""
    mic_wav = project_dir / "mic.wav"
    sys_wav = project_dir / "system.wav"

    def plain():
        # No speaker labels after all — drop that step so the panel stops showing it.
        state.tx_steps = [s for s in state.tx_steps if s != "diarize"]
        return _plain_transcript(state, project_dir / "audio.wav", language), None

    if not sys_wav.exists():
        return plain()
    _tx_begin(state, "diarize")
    try:
        turns = diarize(sys_wav, state.cfg)
    except Exception as e:
        state.status = f"Speaker labels off (diarization unavailable: {e})"
        return plain()
    # Both channels come from the same take, so each is half of the Transcribe step.
    has_mic = mic_wav.exists()
    dur = _wav_duration(sys_wav)
    _tx_begin(state, "transcribe", pct=0.0 if (language == "en" and dur) else None)
    span = 0.5 if has_mic else 1.0
    sys_segments = transcribe_segments(
        sys_wav, language, duration=dur,
        on_progress=lambda p: setattr(state, "tx_pct", p * span))
    mic_segments = []
    if has_mic:
        state.tx_pct = span if state.tx_pct is not None else None
        mic_segments = transcribe_segments(
            mic_wav, language, duration=_wav_duration(mic_wav),
            on_progress=lambda p: setattr(state, "tx_pct", span + p * span))
    labeled, smap = build_labeled_recording(mic_segments, sys_segments, turns)
    return render_labeled_transcript(labeled), smap


def _import_worker(state, src, project_dir):
    """Background worker: extract audio (with progress), then transcribe (with
    progress), then save+open. Updates state.import_* for the UI to render."""
    audio_path = project_dir / "audio.wav"
    # Phase 1: extract audio (real % from ffmpeg when duration is known).
    dur = _media_duration(src)
    state.import_phase = "extract"
    state.import_pct = 0.0 if dur else None
    state.import_phase_start = time.monotonic()
    err = _extract_audio(
        src, audio_path, duration=dur,
        on_pct=(lambda p: setattr(state, "import_pct", p)) if dur else None,
    )
    if err:
        state.status = f"Import failed: {err}"
        try:
            project_dir.rmdir()
        except Exception:
            pass
        state.mode = "menu"
        return

    # Phase 2: transcribe (real % for English via segments; indeterminate for tr).
    tdur = _media_duration(audio_path) or dur
    state.import_phase = "transcribe"
    state.import_pct = 0.0 if (state.language == "en" and tdur) else None
    state.import_phase_start = time.monotonic()
    try:
        if state.diarize:
            text, speaker_map = _labeled_import_text(state, audio_path, tdur)
        else:
            text = transcribe_audio(
                audio_path, state.language,
                on_progress=(lambda p: setattr(state, "import_pct", p)),
                duration=tdur,
            )
            speaker_map = None
    except Exception as e:
        state.status = f"Transcription error: {e}"
        state.mode = "menu"
        return

    _save_and_open(state, project_dir, text, speaker_map=speaker_map)
    state.mode = "menu"


def _labeled_import_text(state, audio_path, tdur):
    """Labeled transcript for an imported file: diarize the whole mixed file into
    'Speaker N' (no 'Me', since there is no separate mic channel). Returns
    (text, speaker_map). Falls back to a plain transcript if diarization fails."""
    segments = transcribe_segments(
        audio_path, state.language,
        on_progress=(lambda p: setattr(state, "import_pct", p)),
        duration=tdur,
    )
    state.import_phase = "diarize"
    state.import_pct = None
    state.import_phase_start = time.monotonic()
    try:
        turns = diarize(audio_path, state.cfg)
    except Exception as e:
        state.status = f"Speaker labels off (diarization unavailable: {e})"
        return " ".join(s["text"] for s in segments), None
    labeled, smap = build_labeled_diarized(segments, turns)
    return render_labeled_transcript(labeled), smap


def _import_file(state, live):
    """Pick an external media file (native dialog), then run extraction +
    transcription in a background worker so the UI can show live progress."""
    state.mode = "preparing"
    state.status = "Opening file picker…"
    live.update(_tui_render(state))

    src = _choose_media_file()
    if not src:
        state.status = "Import cancelled."
        state.mode = "menu"
        return
    ext = os.path.splitext(src)[1].lower()
    if ext not in IMPORT_EXTS:
        state.status = f"Unsupported file type '{ext}' (use mp4/mov/wav/mp3/m4a)."
        state.mode = "menu"
        return

    project_dir = state.base_path / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    try:
        project_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        state.status = f"Folder error: {e}"
        state.mode = "menu"
        return
    # Default an unnamed import to the source file's base name.
    if not state.pending_name:
        state.pending_name = os.path.splitext(os.path.basename(src))[0]
    _write_meta(project_dir, name=state.pending_name, language=state.language)

    state.import_src = os.path.basename(src)
    state.import_phase = ""
    state.import_pct = None
    state.status = "Importing…"
    state.mode = "importing"
    threading.Thread(target=_import_worker, args=(state, src, project_dir),
                     daemon=True).start()


def _begin_named(state, action):
    """Ask for a name, then run `action` ('record' or 'import')."""
    state.pending_action = action
    state.name_buffer = ""
    state.mode = "name_input"


def _open_viewer(state, rec):
    state.viewer_rec = rec
    state.viewer_scroll = 0
    tx = rec["dir"] / "transcription.txt"
    try:
        state.viewer_text = tx.read_text(encoding="utf-8") if tx.exists() else "(no transcript)"
    except Exception as e:
        state.viewer_text = f"(could not read transcript: {e})"
    state.mode = "viewer"


def _activate_setting(state, live, setting):
    if setting == "language":
        state.language = "en" if state.language == "tr" else "tr"
        state.cfg["language"] = state.language
        save_config(state.cfg)
        state.status = f"Language set to {state.language.upper()}"
        _warm_model_async(state, state.language)
    elif setting == "system":
        state.capture_system = not state.capture_system
        state.cfg["capture_system_audio"] = state.capture_system
        save_config(state.cfg)
        state.status = f"System audio {'on' if state.capture_system else 'off'}"
    elif setting == "diarize":
        state.diarize = not state.diarize
        state.cfg["diarize"] = state.diarize
        save_config(state.cfg)
        state.status = ("Speaker labels on (needs a Hugging Face token; see README)"
                        if state.diarize else "Speaker labels off")
    elif setting == "mic":
        state.devices = list_input_devices()
        state.mode = "mic_picker"
    elif setting == "openwith":
        state.apps = list_installed_apps()
        state.app_filter = ""
        state.mode = "app_picker"
    elif setting == "folder":
        state.path_buffer = str(state.base_path)
        state.mode = "path_edit"


def _menu_activate(state, live, cur):
    kind = cur["kind"]
    if kind == "group":
        g = cur["group"]
        if g in state.menu_expanded:
            state.menu_expanded.discard(g)
        else:
            state.menu_expanded.add(g)
            if g == "recordings":
                state.recordings = list_recordings(state.base_path)
    elif kind == "action":
        if cur["action"] in ("record", "import"):
            _begin_named(state, cur["action"])
    elif kind == "recording":
        _open_viewer(state, cur["rec"])
    elif kind == "setting":
        _activate_setting(state, live, cur["setting"])


def _start_rename(state, rec, return_mode):
    state.rename_target = rec["dir"]
    state.rename_buffer = rec["name"]
    state.rename_return = return_mode
    state.mode = "rename"


def _tui_handle_key(key, state, live):
    """Dispatch a keystroke. Returns False to quit, True to keep running."""
    mode = state.mode
    if mode == "menu":
        items = _menu_items(state)
        state.menu_index = max(0, min(state.menu_index, len(items) - 1))
        cur = items[state.menu_index]
        if key in ("q", "Q"):
            return False
        elif key in ("UP", "k"):
            state.menu_index = (state.menu_index - 1) % len(items)
        elif key in ("DOWN", "j"):
            state.menu_index = (state.menu_index + 1) % len(items)
        elif key == "RIGHT":
            if cur["kind"] == "group":
                state.menu_expanded.add(cur["group"])
                if cur["group"] == "recordings":
                    state.recordings = list_recordings(state.base_path)
        elif key == "LEFT":
            if cur["kind"] == "group":
                state.menu_expanded.discard(cur["group"])
        elif key == "ENTER":
            if cur["kind"] == "action" and cur["action"] == "quit":
                return False
            _menu_activate(state, live, cur)
        elif cur["kind"] == "recording":
            if key in ("r", "R"):
                _start_rename(state, cur["rec"], "menu")
            elif key in ("p", "P"):
                _toggle_play(state, cur["rec"])
            elif key in ("t", "T"):
                if not cur["rec"]["has_transcript"]:
                    _transcribe_existing(state, live, cur["rec"])
            elif key in ("d", "D"):
                state.delete_target = cur["rec"]["dir"]
                state.delete_name = _rec_display_name(cur["rec"])
                state.delete_return = "menu"
                state.mode = "confirm_delete"
    elif mode == "name_input":
        if key == "ESC":
            state.mode = "menu"
        elif key == "ENTER":
            state.pending_name = state.name_buffer.strip()
            action = state.pending_action
            state.pending_action = None
            if action == "record":
                _start_recording(state, live)
            elif action == "import":
                _import_file(state, live)
            else:
                state.mode = "menu"
        elif key == "BACKSPACE":
            state.name_buffer = state.name_buffer[:-1]
        elif key and len(key) == 1 and key.isprintable():
            state.name_buffer += key
    elif mode == "viewer":
        page = max(1, console.size.height - 12)
        if key in ("ESC", "LEFT", "q", "Q"):
            state.mode = "menu"
        elif key in ("UP", "k"):
            state.viewer_scroll = max(0, state.viewer_scroll - 1)
        elif key in ("DOWN", "j"):
            state.viewer_scroll += 1
        elif key == "PGUP":
            state.viewer_scroll = max(0, state.viewer_scroll - page)
        elif key == "PGDN":
            state.viewer_scroll += page
        elif key == "HOME":
            state.viewer_scroll = 0
        elif key == "END":
            state.viewer_scroll = 10 ** 9   # clamped at render
        elif key == "ENTER":
            tx = state.viewer_rec["dir"] / "transcription.txt"
            if state.open_app:
                err = _open_in_app(state.open_app, tx)
                state.status = (f"Could not open in {state.open_app}" if err
                                else f"Opened in {state.open_app}")
            else:
                try:
                    subprocess.run(["open", str(tx)], check=False)
                    state.status = "Opened transcript"
                except Exception as e:
                    state.status = f"Open error: {e}"
        elif key in ("p", "P"):
            _toggle_play(state, state.viewer_rec)
        elif key in ("t", "T"):
            if state.viewer_rec and not state.viewer_rec.get("has_transcript"):
                _transcribe_existing(state, live, state.viewer_rec)
        elif key in ("d", "D"):
            state.delete_target = state.viewer_rec["dir"]
            state.delete_name = _rec_display_name(state.viewer_rec)
            state.delete_return = "menu"          # the recording is gone after delete
            state.mode = "confirm_delete"
        elif key in ("r", "R"):
            _start_rename(state, state.viewer_rec, "viewer")
    elif mode == "rename":
        if key == "ESC":
            state.mode = state.rename_return
        elif key == "ENTER":
            new = state.rename_buffer.strip()
            if state.rename_target is not None:
                _write_meta(state.rename_target, name=new)
            state.recordings = list_recordings(state.base_path)
            if state.viewer_rec and state.viewer_rec.get("dir") == state.rename_target:
                state.viewer_rec["name"] = new
            state.status = f"Renamed to “{new or 'Untitled'}”"
            state.mode = state.rename_return
        elif key == "BACKSPACE":
            state.rename_buffer = state.rename_buffer[:-1]
        elif key and len(key) == 1 and key.isprintable():
            state.rename_buffer += key
    elif mode == "confirm_delete":
        if key in ("y", "Y"):
            _delete_recording(state, state.delete_target)
            if state.viewer_rec and state.viewer_rec.get("dir") == state.delete_target:
                state.viewer_rec = None
            state.mode = "menu"
        else:
            state.mode = state.delete_return
    elif mode == "recording":
        if key in ("q", "Q", "r", "R", " "):
            _stop_and_transcribe(state, live)
    elif mode == "mic_picker":
        if key in ("ESC", "q", "Q"):
            state.mode = "menu"
        elif key and key.isdigit() and key != "0":
            i = int(key)
            if 1 <= i <= len(state.devices):
                idx, name = state.devices[i - 1]
                state.mic_index = idx
                state.cfg["input_device"] = name
                save_config(state.cfg)
                state.status = f"Microphone: {name}"
                state.mode = "menu"
    elif mode == "app_picker":
        flt = state.app_filter.lower()
        matches = [a for a in state.apps if flt in a.lower()]
        if key == "ESC":
            state.mode = "menu"
        elif key == "0":
            state.open_app = None
            state.cfg["open_app"] = None
            save_config(state.cfg)
            state.status = "Auto-open disabled"
            state.mode = "menu"
        elif key and key.isdigit():
            i = int(key)
            if 1 <= i <= min(9, len(matches)):
                state.open_app = matches[i - 1]
                state.cfg["open_app"] = state.open_app
                save_config(state.cfg)
                state.status = f"Open transcripts with: {state.open_app}"
                state.mode = "menu"
        elif key == "ENTER":
            if matches:
                state.open_app = matches[0]
                state.cfg["open_app"] = state.open_app
                save_config(state.cfg)
                state.status = f"Open transcripts with: {state.open_app}"
                state.mode = "menu"
        elif key == "BACKSPACE":
            state.app_filter = state.app_filter[:-1]
        elif key and len(key) == 1 and key.isprintable():
            state.app_filter += key
    elif mode == "path_edit":
        if key == "ESC":
            state.mode = "menu"
        elif key == "ENTER":
            raw = state.path_buffer.strip() or str(state.base_path)
            p = Path(os.path.expanduser(raw)).resolve()
            try:
                p.mkdir(parents=True, exist_ok=True)
                state.base_path = p
                state.cfg["base_path"] = str(p)
                save_config(state.cfg)
                state.recordings = list_recordings(p)
                state.menu_index = 0
                state.status = f"Folder: {p}"
            except Exception as e:
                state.status = f"Folder error: {e}"
            state.mode = "menu"
        elif key == "BACKSPACE":
            state.path_buffer = state.path_buffer[:-1]
        elif key and len(key) == 1 and key.isprintable():
            state.path_buffer += key
    return True


def main():
    global _QUIET
    cfg = load_config()
    state = _TuiState(cfg)

    if not sys.stdin.isatty() or not sys.stdout.isatty():
        console.print("Audiocript needs an interactive terminal.")
        return

    # Rebuild audio.wav for any recording killed mid-capture last time, so it is
    # never lost — it reappears in the list ready to transcribe with [t].
    try:
        recovered = _recover_interrupted(state.base_path)
        if recovered:
            state.recordings = list_recordings(state.base_path)
            state.status = (f"Recovered {recovered} interrupted "
                            f"recording{'s' if recovered > 1 else ''}")
    except Exception:
        pass

    _QUIET = True  # keep library/app prints off the full-screen UI
    # Pre-warm the current language's model in the background so the first
    # transcript is fast (header shows Model: loading… → ready).
    _warm_model_async(state, state.language)
    try:
        with _cbreak_mode(), Live(_tui_render(state), screen=True, console=console,
                                  refresh_per_second=15) as live:
            running = True
            while running:
                live.update(_tui_render(state))
                key = _read_key(0.07)
                if key is None:
                    continue
                running = _tui_handle_key(key, state, live)
    finally:
        # Stop any active recorders (e.g. Ctrl-C mid-recording) so the tap
        # subprocess never lingers, and stop any audio playback.
        for r in getattr(state, "recorders", []) or []:
            try:
                r.stop()
            except Exception:
                pass
        _stop_play(state)


def _handle_sigterm(signum, frame):
    # On `kill` (SIGTERM), take the same clean path as Ctrl-C.
    raise KeyboardInterrupt


if __name__ == "__main__":
    import signal
    try:
        signal.signal(signal.SIGTERM, _handle_sigterm)
    except Exception:
        pass
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        console.print("\n[bold yellow]Exiting…[/bold yellow]")
        sys.exit(0)
