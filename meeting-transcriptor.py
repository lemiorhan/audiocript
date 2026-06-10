import os
import sys
import json
import contextlib
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


def resolve_input_device(cfg):
    """Config'teki cihazı çözer; kayıtlı cihaz yoksa/bulunamazsa seçim ister."""
    name = cfg.get("input_device")
    if name:
        for idx, dev_name in list_input_devices():
            if dev_name == name:
                console.print(f"[blue]Giriş cihazı:[/blue] [cyan]{name}[/cyan] (index {idx})\n")
                return idx
        console.print(f"[yellow]Kayıtlı cihaz '{name}' bulunamadı; lütfen yeniden seçin.[/yellow]")
    return select_input_device(cfg)


def select_system_device(cfg):
    """
    İsteğe bağlı 'sistem sesi' (ör. Zoom çıkışı) kaynağını seçtirir. Kullanıcı
    'kapalı' seçerse mikrofon-tek kayda dönülür. Seçimi config'e isimle kaydeder
    ve index'i (veya kapalıysa None) döndürür.
    """
    devices = list_input_devices()
    if not devices:
        console.print("[bold red]Giriş yapabilen ses cihazı bulunamadı![/bold red]")
        return None

    console.print("[bold blue]Sistem sesi (Zoom/toplantı) kaynağı seçin:[/bold blue]")
    console.print("  [cyan]0[/cyan]) (Kapalı — sadece mikrofon)")
    for n, (idx, name) in enumerate(devices, start=1):
        hint = "  [dim]<- önerilen[/dim]" if "blackhole" in name.lower() else ""
        console.print(f"  [cyan]{n}[/cyan]) {name}{hint}")
    console.print("[dim]Not: Zoom sesini yakalamak için Zoom/sistem çıkışını bu cihaza (ör. BlackHole) yönlendirmiş olmalısınız.[/dim]")

    # Varsayılan: önceki seçim, yoksa BlackHole varsa onu öner, yoksa kapalı.
    current_name = cfg.get("system_device")
    default_choice = "0"
    for n, (_, name) in enumerate(devices, start=1):
        if name == current_name:
            default_choice = str(n)
            break
    else:
        for n, (_, name) in enumerate(devices, start=1):
            if "blackhole" in name.lower():
                default_choice = str(n)
                break

    choice = Prompt.ask(
        "Seçiminiz (numara)",
        choices=[str(n) for n in range(len(devices) + 1)],
        default=default_choice,
    )
    if choice == "0":
        cfg.pop("system_device", None)
        save_config(cfg)
        console.print("[green]Sistem sesi kaynağı kapalı (sadece mikrofon).[/green]\n")
        return None
    idx, name = devices[int(choice) - 1]
    cfg["system_device"] = name
    save_config(cfg)
    console.print(f"[green]Sistem sesi kaynağı: {name} (index {idx})[/green]\n")
    return idx


def resolve_system_device(cfg):
    """Config'teki sistem sesi cihazını çözer; ayarlı değilse/yoksa None döndürür."""
    name = cfg.get("system_device")
    if not name:
        return None
    for idx, dev_name in list_input_devices():
        if dev_name == name:
            console.print(f"[blue]Sistem sesi kaynağı:[/blue] [cyan]{name}[/cyan] (index {idx})\n")
            return idx
    console.print(f"[yellow]Kayıtlı sistem sesi cihazı '{name}' bulunamadı; sadece mikrofon kullanılacak.[/yellow]")
    return None


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


def record_audio(filepath, devices, fs=16000):
    """
    Bir veya iki giriş cihazından eşzamanlı kayıt yapar ve tek bir mono WAV'a
    birleştirir. `devices`: cihaz index'lerinin listesi (ör. [mic] veya
    [mic, sistem_sesi]). Başarılıysa True döndürür.
    """
    if not isinstance(devices, (list, tuple)):
        devices = [devices]
    devices = [d for d in devices if d is not None]
    names = [device_name(d) for d in devices]

    clear_console()
    console.print(Panel("Ses Transkripsiyon Uygulamasına Hoşgeldiniz!", style="bold green"), justify="center")
    console.print(f"[dim]Giriş cihaz(lar)ı: {', '.join(names)}[/dim]")
    console.print("[bold blue]Kayda başlamak için 'Enter'a basın. Kayıt sırasında durdurmak için 'q' tuşuna basın.[/bold blue]\n")
    input()  # Kullanıcı Enter'a bastığında devam eder

    console.print(f"[bold yellow]Recording from {', '.join(repr(n) for n in names)}... (Press 'q' to stop)[/bold yellow]")
    frames = [[] for _ in devices]   # her cihaz için ayrı tampon
    stop_flag = [False]

    def make_callback(i):
        def callback(indata, n, t, status):
            if status:
                console.log(f"[red]{status}[/red]")
            frames[i].append(indata.copy())
        return callback

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

    try:
        with contextlib.ExitStack() as stack:
            for i, dev in enumerate(devices):
                stack.enter_context(
                    sd.InputStream(samplerate=fs, channels=1, dtype='int16',
                                   device=dev, callback=make_callback(i))
                )
            while not stop_flag[0]:
                sd.sleep(100)
    except Exception as e:
        listener.stop()
        console.print(f"[bold red]Cihaz açılamadı ({', '.join(names)}): {e}[/bold red]")
        console.print("[yellow]Menüden 'd'/'s' ile farklı bir giriş cihazı seçmeyi deneyin.[/yellow]")
        return False

    listener.join()

    per_device = [
        np.concatenate(buf, axis=0).reshape(-1) if buf else None
        for buf in frames
    ]
    mixed = mix_to_mono(per_device)
    if mixed is None or len(mixed) == 0:
        # 'q' kayıt başlamadan (mikrofon ilk veriyi üretmeden) basılmış olabilir.
        console.print("[bold red]Hiç ses kaydedilemedi; kayıt atlanıyor.[/bold red]")
        return False

    with wave.open(str(filepath), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit (2 byte)
        wf.setframerate(fs)
        wf.writeframes(mixed.tobytes())
    console.print(f"[green]Audio saved to {filepath} ({len(mixed)} samples, {len(devices)} kaynak)[/green]")
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

    # Başlangıç: proje klasörü, dil, mikrofon ve (isteğe bağlı) sistem sesi
    base_path = confirm_base_path(cfg)
    language = cfg.get("language") or select_language(cfg)
    device_index = resolve_input_device(cfg)
    system_index = resolve_system_device(cfg)

    while True:
        # Her kayıt için zaman damgalı bir alt klasör (proje) oluştur.
        project_dir = base_path / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        project_dir.mkdir(parents=True, exist_ok=True)

        # Kayıt kaynakları: mikrofon + (varsa) sistem sesi.
        sources = [device_index] + ([system_index] if system_index is not None else [])

        audio_path = project_dir / "audio.wav"
        if not record_audio(audio_path, sources):
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
        system_label = device_name(system_index) if system_index is not None else "kapalı"
        console.print(
            f"[bold blue]Menü[/bold blue]\n"
            f"  [Enter] Yeni kayıt\n"
            f"  [l] Dili değiştir (mevcut: {language})\n"
            f"  [d] Mikrofonu değiştir (mevcut: {device_name(device_index)})\n"
            f"  [s] Sistem sesi kaynağı (mevcut: {system_label})\n"
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
            system_index = select_system_device(cfg)
        # Diğer her durumda (Enter dahil) yeni kayda devam edilir.


if __name__ == "__main__":
    main()
