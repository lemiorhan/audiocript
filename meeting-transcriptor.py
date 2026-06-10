import os
import sys
import json
import shutil
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
from rich.prompt import Prompt
from rich.console import Console
from rich.panel import Panel

# Uyarıları bastır (FutureWarning, DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Rich için konsol nesnesi oluşturuyoruz
console = Console()

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
    """Dil kodunu ('tr'/'en') Whisper'ın beklediği dil adına çevirir."""
    return LANGUAGES.get(code, "turkish")


def confirm_base_path(cfg):
    """
    Proje klasörünün yolunu kullanıcıya onaylatır.
    Kaydedilmiş yol varsa onu, yoksa varsayılanı önerir. Klasörü oluşturur.
    """
    current = cfg.get("base_path") or str(DEFAULT_BASE_PATH)
    console.print(
        f"[bold blue]Kayıtların saklanacağı proje klasörü:[/bold blue] [cyan]{current}[/cyan]"
    )
    answer = Prompt.ask(
        "Bu yolu kullanmak için Enter'a basın veya yeni bir yol girin",
        default=current,
    )
    base_path = Path(os.path.expanduser(answer.strip() or current)).resolve()
    base_path.mkdir(parents=True, exist_ok=True)
    cfg["base_path"] = str(base_path)
    save_config(cfg)
    console.print(f"[green]Proje klasörü: {base_path}[/green]\n")
    return base_path


def select_language(cfg):
    """tr/en arasından dil seçtirir ve yapılandırmaya kaydeder. Dil kodunu döndürür."""
    choice = Prompt.ask(
        "Transkripsiyon dili seçin",
        choices=list(LANGUAGES.keys()),
        default=cfg.get("language", "tr"),
    )
    cfg["language"] = choice
    save_config(cfg)
    console.print(f"[green]Seçilen dil: {choice} ({lang_name(choice)})[/green]\n")
    return choice


def list_input_devices():
    """Giriş (input) yapabilen ses cihazlarının (index, name) listesini döndürür."""
    result = []
    for idx, dev in enumerate(sd.query_devices()):
        if dev.get("max_input_channels", 0) > 0:
            result.append((idx, dev["name"]))
    return result


def device_name(index):
    """Cihaz index'inden ismini döndürür; bulunamazsa index'i string olarak verir."""
    try:
        return sd.query_devices(index)["name"]
    except Exception:
        return str(index)


def select_input_device(cfg):
    """
    Kullanıcıya giriş cihazı seçtirir, ismini config'e kaydeder ve index'i döndürür.
    Sanal/uygulama cihazları (BlackHole, Zoom, Teams) da listelenir; böylece
    yönlendirilmiş (loopback) bir kaynaktan da kayıt yapılabilir.
    """
    devices = list_input_devices()
    if not devices:
        console.print("[bold red]Giriş yapabilen ses cihazı bulunamadı![/bold red]")
        return None

    console.print("[bold blue]Kullanılabilir giriş (mikrofon/ses) cihazları:[/bold blue]")
    for n, (idx, name) in enumerate(devices):
        console.print(f"  [cyan]{n}[/cyan]) {name}")

    # Önceki seçimi varsayılan yap (varsa).
    current_name = cfg.get("input_device")
    default_choice = "0"
    for n, (_, name) in enumerate(devices):
        if name == current_name:
            default_choice = str(n)
            break

    choice = Prompt.ask(
        "Bir cihaz seçin (numara)",
        choices=[str(n) for n in range(len(devices))],
        default=default_choice,
    )
    idx, name = devices[int(choice)]
    cfg["input_device"] = name
    save_config(cfg)
    console.print(f"[green]Seçilen giriş cihazı: {name} (index {idx})[/green]\n")
    return idx


def select_system_capture(cfg):
    """
    Sistem sesinin (hoparlör/Zoom/YouTube) de kaydedilip kaydedilmeyeceğini
    sorar ve config'e (capture_system_audio: bool) kaydeder. macOS Core Audio
    process tap ile yakalanır; ses çalmaya DEVAM eder, BlackHole gerekmez.
    Açık (True) / kapalı (False) döndürür.
    """
    current = bool(cfg.get("capture_system_audio", False))
    default = "e" if current else "h"
    console.print(
        "[bold blue]Sistem sesini (hoparlör/Zoom/YouTube) de kaydedeyim mi?[/bold blue] "
        "[dim](Core Audio tap; sesi duymaya devam edersiniz, BlackHole gerekmez. "
        "İlk kullanımda 'Sistem Sesi Kaydı' izni istenebilir.)[/dim]"
    )
    choice = Prompt.ask("Evet/Hayır", choices=["e", "h"], default=default)
    val = (choice == "e")
    cfg["capture_system_audio"] = val
    save_config(cfg)
    console.print(f"[green]Sistem sesi kaydı: {'açık' if val else 'kapalı'}[/green]\n")
    return val


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
        raise RuntimeError(f"Tap kaynak dosyası bulunamadı: {TAP_SRC}")
    if TAP_BIN.exists() and TAP_BIN.stat().st_mtime >= TAP_SRC.stat().st_mtime:
        return TAP_BIN
    swiftc = shutil.which("swiftc")
    if not swiftc:
        raise RuntimeError("swiftc bulunamadı (Xcode / Command Line Tools gerekli).")
    console.print("[dim]Sistem sesi yardımcı programı derleniyor (ilk kullanım)...[/dim]")
    res = subprocess.run(
        [swiftc, "-O", str(TAP_SRC), "-o", str(TAP_BIN),
         "-framework", "CoreAudio", "-framework", "AudioToolbox", "-framework", "Foundation"],
        capture_output=True, text=True,
    )
    if res.returncode != 0 or not TAP_BIN.exists():
        raise RuntimeError(f"Swift derlemesi başarısız:\n{res.stderr.strip()}")
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

    @property
    def label(self):
        return f"{self.name} ({self.rate} Hz)"

    def start(self):
        def callback(indata, n, t, status):
            if status:
                console.log(f"[red]{status}[/red]")
            self._frames.append(indata.copy())
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
        self.rate = None
        self.channels = None
        self._proc = None
        self._reader = None
        self._stderr_reader = None
        self._chunks = []
        self._stderr = []
        self._ready = threading.Event()
        self._error = None

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
                "Sistem sesi başlatılamadı (10 sn). 'Sistem Sesi Kaydı' izni "
                "gerekebilir: Sistem Ayarları → Gizlilik ve Güvenlik."
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
                self._error = "Yardımcı program başlık vermeden kapandı"
                self._ready.set()
                return
            header += b
        try:
            parts = dict(p.split("=", 1) for p in header.decode().split())
            self.rate = int(parts["samplerate"])
            self.channels = int(parts["channels"])
        except Exception:
            self._error = f"Geçersiz başlık: {header!r}"
            self._ready.set()
            return
        self._ready.set()
        # 2) Kalan veriyi (float32 PCM) topla.
        while True:
            data = f.read(8192)
            if not data:
                break
            self._chunks.append(data)

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


def record_audio(filepath, mic_device, capture_system=False, target_fs=16000):
    """
    Mikrofondan (sounddevice) ve isteğe bağlı olarak sistem sesinden (Core Audio
    tap) EŞ ZAMANLI kayıt yapar; her kaynağı 16000 Hz mono'ya yeniden örnekleyip
    tek bir mono WAV'da birleştirir. Başarılıysa True döndürür.

    Mikrofon kendi doğal hızında açılır (cihazın global hızını değiştirmemek
    için). Sistem sesi, hoparlörden ses kesilmeden yakalanır.
    """
    recorders = []
    if mic_device is not None:
        recorders.append(DeviceRecorder(mic_device))

    # Sistem sesi seçiliyse yardımcı programı ÖNDEN derle (Enter'dan önce), ki
    # derleme mesajı kayıt akışını bölmesin. Derleme/izin sorunu olursa uyar.
    tap = None
    if capture_system:
        try:
            build_tap_binary()
            tap = TapRecorder()
        except Exception as e:
            console.print(f"[yellow]Sistem sesi kullanılamıyor, sadece mikrofon: {e}[/yellow]")
            tap = None

    clear_console()
    console.print(Panel("Ses Transkripsiyon Uygulamasına Hoşgeldiniz!", style="bold green"), justify="center")
    labels = [r.label for r in recorders] + ([tap.label] if tap else [])
    console.print(f"[dim]Kaynak(lar): {', '.join(labels)}[/dim]")
    console.print("[bold blue]Kayda başlamak için 'Enter'a basın. Kayıt sırasında durdurmak için 'q' tuşuna basın.[/bold blue]\n")
    input()  # Kullanıcı Enter'a bastığında devam eder

    console.print("[bold yellow]Recording... (Press 'q' to stop)[/bold yellow]")

    # Mikrofon akışlarını başlat (kritik: açılamazsa kaydı iptal et).
    started = []
    try:
        for r in recorders:
            r.start()
            started.append(r)
    except Exception as e:
        for r in started:
            r.stop()
        console.print(f"[bold red]Mikrofon açılamadı: {e}[/bold red]")
        console.print("[yellow]Menüden 'd' ile farklı bir mikrofon seçmeyi deneyin.[/yellow]")
        return False

    # Sistem sesi akışını başlat (kritik değil: başarısızsa sadece mikrofonla devam).
    if tap is not None:
        try:
            tap.start()
            started.append(tap)
        except Exception as e:
            console.print(f"[yellow]Sistem sesi başlatılamadı, sadece mikrofon ile devam: {e}[/yellow]")
            tap = None

    stop_flag = [False]

    def on_press(key):
        try:
            if key.char == 'q':
                console.print("[bold red]Recording stopped.[/bold red]")
                stop_flag[0] = True
                return False  # Dinleyiciyi durdur
        except AttributeError:
            pass

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    # try/finally: Ctrl-C (KeyboardInterrupt) veya başka bir hata olsa bile tüm
    # kayıt kaynaklarını (özellikle tap alt sürecini) mutlaka durdur.
    try:
        while not stop_flag[0]:
            sd.sleep(100)
    finally:
        try:
            listener.stop()
        except Exception:
            pass
        for r in started:
            r.stop()

    # Her kaynağı KENDİ hızından 16000 Hz mono'ya yeniden örnekle, sonra birleştir.
    resampled = []
    for r in started:
        arr, rate = r.result()
        resampled.append(resample_to_target(arr, rate, target_fs))
    mixed = mix_to_mono(resampled)
    if mixed is None or len(mixed) == 0:
        # 'q' kayıt başlamadan basılmış veya hiç ses gelmemiş olabilir.
        console.print("[bold red]Hiç ses kaydedilemedi; kayıt atlanıyor.[/bold red]")
        return False

    with wave.open(str(filepath), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit (2 byte)
        wf.setframerate(target_fs)  # Whisper 16000 Hz mono bekler
        wf.writeframes(mixed.tobytes())
    console.print(f"[green]Audio saved to {filepath} ({len(mixed)} samples @ {target_fs} Hz, {len(started)} kaynak)[/green]")
    return True


def pick_device():
    """transformers pipeline için uygun cihazı seçer (CUDA > MPS > CPU)."""
    import torch
    if torch.cuda.is_available():
        return "cuda:0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _transcribe_turkish(filepath):
    """Türkçe transkripsiyon: transformers + selimc/whisper-large-v3-turbo-turkish."""
    global _hf_pipe
    if _hf_pipe is None:
        from transformers import pipeline
        device = pick_device()
        console.print(f"[dim]Türkçe model yükleniyor ({TR_HF_MODEL}, {device})...[/dim]")
        _hf_pipe = pipeline("automatic-speech-recognition", model=TR_HF_MODEL, device=device)
    result = _hf_pipe(
        str(filepath),
        return_timestamps=True,
        generate_kwargs={"language": "turkish", "task": "transcribe"},
    )
    return result.get("text", "")


def _transcribe_english(filepath):
    """İngilizce transkripsiyon: whisper.cpp (pywhispercpp) + ggml-distil-large-v3."""
    global _cpp_model
    if _cpp_model is None:
        from huggingface_hub import hf_hub_download
        from pywhispercpp.model import Model
        console.print(f"[dim]İngilizce model hazırlanıyor ({EN_GGML_FILE})...[/dim]")
        model_path = hf_hub_download(repo_id=EN_GGML_REPO, filename=EN_GGML_FILE)
        _cpp_model = Model(model_path, print_progress=False, print_realtime=False)
    segments = _cpp_model.transcribe(str(filepath), language="en")
    return "".join(segment.text for segment in segments).strip()


def transcribe_audio(filepath, language_code):
    """Seçilen dile göre uygun model ile transkripsiyon yapar ve metni döndürür."""
    if language_code == "en":
        console.print("[bold cyan]Transcribing audio (en / ggml-distil-large-v3)...[/bold cyan]")
        return _transcribe_english(filepath)
    console.print("[bold cyan]Transcribing audio (tr / whisper-large-v3-turbo-turkish)...[/bold cyan]")
    return _transcribe_turkish(filepath)


def main():
    cfg = load_config()

    clear_console()
    console.print(Panel("Ses Transkripsiyon Uygulaması", style="bold magenta"), justify="center")

    # Başlangıç: her seçim, kayıtlı değer varsayılan olarak önerilerek sorulur.
    # (Enter'a basmak kayıtlı değeri korur.)
    base_path = confirm_base_path(cfg)
    language = select_language(cfg)
    device_index = select_input_device(cfg)
    capture_system = select_system_capture(cfg)

    while True:
        # Her kayıt için zaman damgalı bir alt klasör (proje) oluştur.
        project_dir = base_path / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        project_dir.mkdir(parents=True, exist_ok=True)

        audio_path = project_dir / "audio.wav"
        if not record_audio(audio_path, device_index, capture_system):
            # Kayıt alınamadı: boş proje klasörünü temizle ve baştan başla.
            try:
                project_dir.rmdir()
            except OSError:
                pass
            flush_stdin()
            continue

        transcribed_text = transcribe_audio(audio_path, language)
        console.print(Panel(f"[bold green]Transcription:[/bold green]\n{transcribed_text}", style="green"), justify="center")

        transcription_path = project_dir / "transcription.txt"
        with open(transcription_path, "w", encoding="utf-8") as file:
            file.write(transcribed_text + "\n")
        console.print(f"[bold green]Proje kaydedildi: {project_dir}[/bold green]\n")

        # Önceki kayıttan kalan karakterleri temizliyoruz.
        flush_stdin()

        # Kullanıcıya menüyü gösteriyoruz.
        system_label = "açık" if capture_system else "kapalı"
        console.print(
            f"[bold blue]Menü[/bold blue]\n"
            f"  [Enter] Yeni kayıt\n"
            f"  [l] Dili değiştir (mevcut: {language})\n"
            f"  [d] Mikrofonu değiştir (mevcut: {device_name(device_index)})\n"
            f"  [s] Sistem sesi kaydı (mevcut: {system_label})\n"
            f"  [q] Çıkış"
        )
        try:
            response = Prompt.ask("Seçiminiz", default="")
        except Exception:
            response = ""
        choice = response.strip().lower()

        if choice == "q":
            console.print("[bold yellow]Çıkılıyor...[/bold yellow]")
            break
        elif choice == "l":
            language = select_language(cfg)
        elif choice == "d":
            new_index = select_input_device(cfg)
            if new_index is not None:
                device_index = new_index
        elif choice == "s":
            capture_system = select_system_capture(cfg)
        # Diğer her durumda (Enter dahil) yeni kayda devam edilir.


def _handle_sigterm(signum, frame):
    # `kill` (SIGTERM) gelince Ctrl-C ile aynı temiz çıkış yolunu kullan:
    # KeyboardInterrupt fırlatınca record_audio'daki finally tüm kaynakları
    # (tap alt süreci dahil) durdurur ve aşağıdaki handler temiz mesaj basar.
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
        console.print("\n[bold yellow]Çıkılıyor...[/bold yellow]")
        sys.exit(0)
