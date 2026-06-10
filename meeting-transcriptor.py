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
import termios  # Unix tabanlı sistemlerde çalışır.
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

# Uyarıları bastır (FutureWarning, DeprecationWarning, UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# transformers / huggingface_hub gürültüsünü kapat (import'tan ÖNCE ayarlanmalı;
# bu kütüphaneler tembel (lazy) import edildiği için burada ayarlamak yeterli).
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _silence_ml_logging():
    """transformers ilerleme çubuklarını ve transformers/huggingface_hub uyarı
    loglarını (ör. 'unauthenticated requests', logits processor uyarıları) kapatır."""
    import logging
    try:
        import transformers
        transformers.logging.set_verbosity_error()
        transformers.utils.logging.disable_progress_bar()
    except Exception:
        pass
    for name in ("transformers", "huggingface_hub"):
        logging.getLogger(name).setLevel(logging.ERROR)


# Rich için konsol nesnesi oluşturuyoruz
console = Console()

# Tam ekran TUI etkinken kütüphane/uygulama yazdırmaları ekranı bozmasın diye
# bastırılır (durum bilgisi arayüzde gösterilir).
_QUIET = False

# Yapılandırma dosyası, scriptin yanında saklanır.
CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
DEFAULT_BASE_PATH = Path(__file__).resolve().parent / "recordings"

# Desteklenen diller: kod -> Whisper dil adı
LANGUAGES = {"tr": "turkish", "en": "english"}

# Her dil için kullanılacak model ve çalışma zamanı (runtime).
#  - Türkçe: Hugging Face transformers ile Türkçe'ye ince ayarlı model.
#  - İngilizce: whisper.cpp (pywhispercpp) ile ggml-distil-large-v3 modeli.
TR_HF_MODEL = "selimc/whisper-large-v3-turbo-turkish"
EN_GGML_REPO = "distil-whisper/distil-large-v3-ggml"
EN_GGML_FILE = "ggml-distil-large-v3.bin"

# Yüklenen modelleri tekrar tekrar yüklememek için önbellek.
_hf_pipe = None
_cpp_model = None
# Modeli aynı anda iki kez (ör. arka plan ön-ısıtma + transkripsiyon) yüklememek için kilit.
_MODEL_LOCK = threading.Lock()


def _free_cpp_model_quietly():
    """
    whisper.cpp modeli serbest bırakılırken çıkardığı C/Metal teardown logunu
    ('ggml_metal_free: deallocating') gizlemek için, modeli stderr (fd 2)
    /dev/null'a yönlendirilmişken serbest bırakır. Çıkışta (atexit) çağrılır.
    """
    global _cpp_model
    if _cpp_model is None:
        return
    saved = devnull = None
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        saved = os.dup(2)
        os.dup2(devnull, 2)
        _cpp_model = None  # serbest bırakma logları /dev/null'a gider
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
    sys.stdin'de bekleyen (kalan) karakterleri temizler.
    Unix tabanlı sistemlerde termios.tcflush() kullanarak giriş tamponunu temizler.
    """
    try:
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except Exception:
        pass


def load_config():
    """config.json'u okur. Dosya yoksa veya bozuksa boş bir sözlük döndürür."""
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg if isinstance(cfg, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_config(cfg):
    """Yapılandırmayı config.json'a yazar."""
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def lang_name(code):
    """Map a language code ('tr'/'en') to the name Whisper expects."""
    return LANGUAGES.get(code, "turkish")


def list_input_devices():
    """Giriş (input) yapabilen ses cihazlarının (index, name) listesini döndürür."""
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
    Bir veya daha fazla mono int16 sinyali tek bir int16 sinyalde birleştirir.
    Akışları en kısa olanın uzunluğuna göre kırpar (cihazların saatleri biraz
    kayabilir). Toplama yapar ve gerekirse kırpılmayı (clipping) önlemek için
    tepe değere göre ölçekler.
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
    Bir sinyali (int16 veya float32; mono ya da çok kanallı) hedef örnekleme
    hızında (Whisper için 16000 Hz) mono int16'ya dönüştürür. Çok kanallıysa
    kanalları ortalar. src_fs == target_fs ise yeniden örnekleme yapmadan sadece
    mono'ya indirger. Yeniden örnekleme torchaudio ile yapılır (anti-aliasing'li).
    """
    if arr is None or len(arr) == 0:
        return None
    if np.issubdtype(arr.dtype, np.floating):
        f = arr.astype(np.float32)            # zaten [-1, 1] aralığında
    else:
        f = arr.astype(np.float32) / 32768.0  # int16 -> [-1, 1]
    if f.ndim == 2:                       # (N, kanal) -> ortalama ile mono
        f = f.mean(axis=1)
    if src_fs != target_fs:
        import torch
        import torchaudio.functional as AF
        w = torch.from_numpy(np.ascontiguousarray(f))
        # Yüksek kaliteli, anti-aliasing'li yeniden örnekleme (soxr-VHQ'ya yakın).
        # Varsayılan parametreler 8 kHz Nyquist üzerindeki tonları yeterince
        # bastırmaz; Kaiser penceresi ile dar geçiş bandı sağlanır.
        f = AF.resample(
            w, orig_freq=src_fs, new_freq=target_fs,
            lowpass_filter_width=64, rolloff=0.945,
            resampling_method="sinc_interp_kaiser", beta=14.769656459379492,
        ).numpy()
    # float -> int16: ölçekle, yuvarla ve taşmayı önlemek için kırp.
    i16 = np.clip(np.round(f * 32768.0), -32768, 32767).astype(np.int16)
    return i16


def _coreaudio_extra_settings():
    """
    macOS'ta PortAudio'nun cihazın global örnekleme hızını değiştirmesini
    engelleyen ayarı döndürür (varsa). Bu, bir cihaza bağlanırken o cihazdan
    çalan sesin kesilmesini önler. Desteklenmiyorsa None döner.
    """
    settings_cls = getattr(sd, "CoreAudioSettings", None)
    if settings_cls is None:
        return None
    try:
        return settings_cls(change_device_parameters=False)
    except Exception:
        return None


# --- Core Audio sistem sesi yakalayıcı (Swift yardımcı program) ---
TAP_DIR = Path(__file__).resolve().parent / "mac_audio_tap"
TAP_SRC = TAP_DIR / "system_audio_tap.swift"
TAP_BIN = TAP_DIR / "system_audio_tap"


def build_tap_binary():
    """
    Sistem sesi yakalayan Swift yardımcı programını gerekiyorsa derler ve yolunu
    döndürür. swiftc yoksa veya derleme başarısızsa RuntimeError fırlatır.
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
    """sounddevice ile bir giriş cihazından (mikrofon) doğal hızda kayıt yapar."""

    def __init__(self, index):
        self.index = index
        self.name = device_name(index)
        info = sd.query_devices(index, 'input')
        self.rate = int(round(info['default_samplerate']))
        self.channels = max(1, min(2, int(info['max_input_channels'])))
        self._frames = []
        self._stream = None
        self._level = 0.0  # canlı VU göstergesi için anlık seviye (0..1)
        self.meter_name = f"🎤 {self.name}"

    @property
    def label(self):
        return f"{self.name} ({self.rate} Hz)"

    def level(self):
        return self._level

    def start(self):
        def callback(indata, n, t, status):
            if status and not _QUIET:
                console.log(f"[red]{status}[/red]")
            self._frames.append(indata.copy())
            if indata.size:
                peak = float(np.max(np.abs(indata))) / 32768.0
                # peak-hold + sönümleme: VU çubuğu sese tepki verir, yumuşak iner.
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

    def result(self):
        if not self._frames:
            return None, self.rate
        return np.concatenate(self._frames, axis=0), self.rate


class TapRecorder:
    """
    macOS Core Audio process tap ile TÜM sistem sesini (hoparlör/Zoom/YouTube)
    yakalar. Swift yardımcı programını alt süreç olarak çalıştırır; ses çalmaya
    DEVAM eder (unmuted), BlackHole/yeniden yönlendirme gerekmez.
    """

    def __init__(self):
        self.label = "Sistem sesi (Core Audio tap)"
        self.meter_name = "🔊 Sistem sesi"
        self.rate = None
        self.channels = None
        self._proc = None
        self._reader = None
        self._stderr_reader = None
        self._chunks = []
        self._stderr = []
        self._ready = threading.Event()
        self._error = None
        self._level = 0.0  # canlı VU göstergesi için anlık seviye (0..1)

    def level(self):
        return self._level

    def start(self):
        binpath = build_tap_binary()  # gerekiyorsa derler; başarısızsa RuntimeError
        self._proc = subprocess.Popen(
            [str(binpath)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0,
        )
        self._stderr_reader = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_reader.start()
        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader.start()
        # Başlık satırını (örnekleme hızı/kanal) bekle; gelmezse izin sorunu olabilir.
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
        # 1) Başlık satırını oku: "samplerate=<r> channels=<c> format=f32le\n"
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
        # 2) Kalan veriyi (float32 PCM) topla ve canlı seviyeyi güncelle.
        while True:
            data = f.read(8192)
            if not data:
                break
            self._chunks.append(data)
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

    def result(self):
        if not self._chunks or not self.rate:
            return None, self.rate or 16000
        raw = b"".join(self._chunks)
        arr = np.frombuffer(raw[:len(raw) // 4 * 4], dtype="<f4")
        ch = self.channels or 1
        if ch > 1:
            arr = arr[:len(arr) // ch * ch].reshape(-1, ch)
        return arr, self.rate


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
    if ch == b"\x1b":                 # ESC or an arrow/escape sequence
        r2, _, _ = select.select([sys.stdin], [], [], 0.0008)
        if r2:
            os.read(fd, 8)            # swallow the rest of the sequence
            return "ESC_SEQ"
        return "ESC"
    if ch in (b"\r", b"\n"):
        return "ENTER"
    if ch in (b"\x7f", b"\x08"):
        return "BACKSPACE"
    try:
        return ch.decode("utf-8", "ignore")
    except Exception:
        return None


def _finalize_to_wav(started, audio_path, target_fs=16000):
    """Stop already-stopped sources' data: resample each to 16k mono, mix, write
    a mono WAV. Returns the number of samples written (0 if nothing captured)."""
    resampled = [resample_to_target(*src.result(), target_fs) for src in started]
    mixed = mix_to_mono(resampled)
    if mixed is None or len(mixed) == 0:
        return 0
    with wave.open(str(audio_path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)            # 16-bit
        wf.setframerate(target_fs)    # Whisper expects 16 kHz mono
        wf.writeframes(mixed.tobytes())
    return len(mixed)


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


def _extract_audio(src, dest_wav, target_fs=16000):
    """Extract/convert `src` media to a 16 kHz mono PCM WAV at `dest_wav` using
    ffmpeg. Returns None on success or an error string."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return "ffmpeg not found — install it (e.g. 'brew install ffmpeg')"
    try:
        res = subprocess.run(
            [ffmpeg, "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", str(target_fs),
             "-c:a", "pcm_s16le", str(dest_wav)],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
        )
        if res.returncode != 0:
            tail = (res.stderr or "").strip().splitlines()
            return f"ffmpeg failed: {tail[-1] if tail else res.returncode}"
        if not Path(dest_wav).exists() or Path(dest_wav).stat().st_size == 0:
            return "ffmpeg produced no audio (no audio track?)"
        return None
    except Exception as e:
        return str(e)


def pick_device():
    """transformers pipeline için uygun cihazı seçer (CUDA > MPS > CPU)."""
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


def transcribe_audio(filepath, language_code):
    """Transcribe with the model for the selected language and return the text."""
    _ensure_model(language_code)
    if language_code == "en":
        segments = _cpp_model.transcribe(str(filepath), language="en")
        return "".join(segment.text for segment in segments).strip()
    result = _hf_pipe(
        str(filepath),
        return_timestamps=True,
        generate_kwargs={"language": "turkish", "task": "transcribe"},
    )
    return result.get("text", "")


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
        self.mode = "home"           # home | recording | transcribing | mic_picker | path_edit
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
        try:
            self.base_path.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass


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
    line1 = (f"[bold]Language[/bold] {state.language.upper()}     "
             f"[bold]Mic[/bold] {mic}     [bold]System audio[/bold] {sysv}     "
             f"[bold]Model[/bold] {_model_label(state)}")
    open_with = state.open_app or "[dim]off[/dim]"
    line2 = f"[dim]Folder[/dim] {state.base_path}     [dim]Open with[/dim] {open_with}"
    return Panel(Text.from_markup(line1 + "\n" + line2),
                 title="🎙  Voice Transcriptor", title_align="left",
                 subtitle=clock, subtitle_align="right", border_style="cyan")


def _tui_body(state):
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
        msg = ("[bold yellow]Transcribing…[/bold yellow]\n\n"
               "[dim]The model loads on first use — this can take a moment.[/dim]")
        return Panel(Text.from_markup(msg), title="Please wait", border_style="yellow")
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
    # home
    if state.last_transcript:
        name = state.last_project.name if state.last_project else ""
        head = Text.from_markup(f"[bold green]Last transcript[/bold green]  [dim]{name}[/dim]")
        return Panel(Group(head, Text(""), Text(state.last_transcript.strip() or "(empty)")),
                     border_style="green")
    return Panel(Text.from_markup("[dim]Press [/dim][bold]r[/bold][dim] to start recording.[/dim]"),
                 title="Transcript", border_style="green")


def _tui_footer(state):
    keymap = {
        "home": "[r] Record  [f] Import file  [l] Language  [d] Microphone  [s] System audio  [o] Open-with  [p] Folder  [q] Quit",
        "preparing": "please wait…",
        "recording": "[q] Stop & transcribe",
        "transcribing": "working…",
        "mic_picker": "[1-9] Select   [Esc] Cancel",
        "path_edit": "[Enter] Save   [Backspace] Delete   [Esc] Cancel",
        "app_picker": "[1-9] Select   [0] Off   [Enter] First match   type to filter   [Esc] Cancel",
    }
    keys = Text(keymap.get(state.mode, ""))          # plain Text: brackets shown literally
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

    state.mode = "preparing"
    announce("Creating project folder…")

    project_dir = state.base_path / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    try:
        project_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        state.status = f"Folder error: {e}"
        state.mode = "home"
        return

    started = []
    sys_failed = False

    # Microphone — critical: abort the recording if it can't open.
    if state.mic_index is not None:
        announce(f"Opening microphone ({device_name(state.mic_index)})…")
        try:
            mic = DeviceRecorder(state.mic_index)
            mic.start()
            started.append(mic)
        except Exception as e:
            for r in started:
                try:
                    r.stop()
                except Exception:
                    pass
            state.status = f"Microphone error: {e}"
            state.mode = "home"
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
            tap.start()
            started.append(tap)
        except Exception as e:
            sys_failed = True
            announce(f"System audio unavailable ({e}); continuing mic-only")

    if not started:
        state.status = "No microphone available."
        state.mode = "home"
        try:
            project_dir.rmdir()
        except Exception:
            pass
        return

    state.recorders = started
    state.project_dir = project_dir
    state.rec_start = time.monotonic()
    state.mode = "recording"
    if not sys_failed:
        state.status = "Recording started — speak now."


def _stop_and_transcribe(state, live):
    for r in state.recorders:
        try:
            r.stop()
        except Exception:
            pass
    audio_path = state.project_dir / "audio.wav"
    n = _finalize_to_wav(state.recorders, audio_path)
    state.recorders = []
    if not n:
        state.status = "No audio captured; recording discarded."
        try:
            state.project_dir.rmdir()
        except Exception:
            pass
        state.mode = "home"
        return
    _transcribe_and_save(state, live, state.project_dir, audio_path)


def _transcribe_and_save(state, live, project_dir, audio_path):
    """Transcribe `audio_path` (in `project_dir`), save transcription.txt, update
    the UI, and open the transcript in the chosen app. Shared by recording and
    file-import flows."""
    state.mode = "transcribing"
    state.status = "Transcribing…"
    live.update(_tui_render(state))
    try:
        text = transcribe_audio(audio_path, state.language)
    except Exception as e:
        state.status = f"Transcription error: {e}"
        state.mode = "home"
        return
    transcript_path = project_dir / "transcription.txt"
    try:
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    except Exception as e:
        state.status = f"Save error: {e}"
        state.mode = "home"
        return
    state.last_transcript = text
    state.last_project = project_dir
    state.status = f"Saved to {project_dir.name}"
    if state.open_app:
        err = _open_in_app(state.open_app, transcript_path)
        state.status = (f"Saved — could not open in {state.open_app}"
                        if err else f"Saved & opened in {state.open_app}")
    state.mode = "home"


def _import_file(state, live):
    """Let the user pick an external media file, extract its audio into a new
    project folder, and transcribe it like a recording."""
    state.mode = "preparing"
    state.status = "Opening file picker…"
    live.update(_tui_render(state))

    src = _choose_media_file()
    if not src:
        state.status = "Import cancelled."
        state.mode = "home"
        return
    ext = os.path.splitext(src)[1].lower()
    if ext not in IMPORT_EXTS:
        state.status = f"Unsupported file type '{ext}' (use mp4/mov/wav/mp3/m4a)."
        state.mode = "home"
        return

    project_dir = state.base_path / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    try:
        project_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        state.status = f"Folder error: {e}"
        state.mode = "home"
        return

    state.status = f"Extracting audio from {os.path.basename(src)}…"
    live.update(_tui_render(state))
    audio_path = project_dir / "audio.wav"
    err = _extract_audio(src, audio_path)
    if err:
        state.status = f"Import failed: {err}"
        try:
            project_dir.rmdir()
        except Exception:
            pass
        state.mode = "home"
        return

    _transcribe_and_save(state, live, project_dir, audio_path)


def _tui_handle_key(key, state, live):
    """Dispatch a keystroke. Returns False to quit, True to keep running."""
    mode = state.mode
    if mode == "home":
        if key in ("q", "Q"):
            return False
        if key in ("r", "R", " "):
            _start_recording(state, live)
        elif key in ("f", "F"):
            _import_file(state, live)
        elif key in ("l", "L"):
            state.language = "en" if state.language == "tr" else "tr"
            state.cfg["language"] = state.language
            save_config(state.cfg)
            state.status = f"Language set to {state.language.upper()}"
            _warm_model_async(state, state.language)  # pre-load the new language
        elif key in ("s", "S"):
            state.capture_system = not state.capture_system
            state.cfg["capture_system_audio"] = state.capture_system
            save_config(state.cfg)
            state.status = f"System audio {'on' if state.capture_system else 'off'}"
        elif key in ("d", "D"):
            state.devices = list_input_devices()
            state.mode = "mic_picker"
        elif key in ("o", "O"):
            state.apps = list_installed_apps()
            state.app_filter = ""
            state.mode = "app_picker"
        elif key in ("p", "P"):
            state.path_buffer = str(state.base_path)
            state.mode = "path_edit"
    elif mode == "recording":
        if key in ("q", "Q", "r", "R", " "):
            _stop_and_transcribe(state, live)
    elif mode == "mic_picker":
        if key in ("ESC", "q", "Q", "d", "D"):
            state.mode = "home"
        elif key and key.isdigit() and key != "0":
            i = int(key)
            if 1 <= i <= len(state.devices):
                idx, name = state.devices[i - 1]
                state.mic_index = idx
                state.cfg["input_device"] = name
                save_config(state.cfg)
                state.status = f"Microphone: {name}"
                state.mode = "home"
    elif mode == "app_picker":
        flt = state.app_filter.lower()
        matches = [a for a in state.apps if flt in a.lower()]
        if key == "ESC":
            state.mode = "home"
        elif key == "0":
            state.open_app = None
            state.cfg["open_app"] = None
            save_config(state.cfg)
            state.status = "Auto-open disabled"
            state.mode = "home"
        elif key and key.isdigit():
            i = int(key)
            if 1 <= i <= min(9, len(matches)):
                state.open_app = matches[i - 1]
                state.cfg["open_app"] = state.open_app
                save_config(state.cfg)
                state.status = f"Open transcripts with: {state.open_app}"
                state.mode = "home"
        elif key == "ENTER":
            if matches:
                state.open_app = matches[0]
                state.cfg["open_app"] = state.open_app
                save_config(state.cfg)
                state.status = f"Open transcripts with: {state.open_app}"
                state.mode = "home"
        elif key == "BACKSPACE":
            state.app_filter = state.app_filter[:-1]
        elif key and len(key) == 1 and key.isprintable():
            state.app_filter += key
    elif mode == "path_edit":
        if key == "ESC":
            state.mode = "home"
        elif key == "ENTER":
            raw = state.path_buffer.strip() or str(state.base_path)
            p = Path(os.path.expanduser(raw)).resolve()
            try:
                p.mkdir(parents=True, exist_ok=True)
                state.base_path = p
                state.cfg["base_path"] = str(p)
                save_config(state.cfg)
                state.status = f"Folder: {p}"
            except Exception as e:
                state.status = f"Folder error: {e}"
            state.mode = "home"
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
        console.print("Voice Transcriptor needs an interactive terminal.")
        return

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
        # subprocess never lingers.
        for r in getattr(state, "recorders", []) or []:
            try:
                r.stop()
            except Exception:
                pass


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
