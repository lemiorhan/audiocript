#!/usr/bin/env python3
"""Post-install check for the dictation menu bar: run it as often as you like.

    python3 doctor.py              # check, and fix what is safe to fix
    python3 doctor.py --check      # change nothing, only report
    python3 doctor.py --probe      # put a plain AC-TEST icon in the menu bar for 15s
    python3 doctor.py --restart    # stop a running daemon and start a fresh one

It ends with one line: "HERŞEY OK" and exit status 0, or the numbered list of what
is still wrong and exit status 1. Nothing here prints the API key, and nothing is
deleted — the only writes are `chmod +x run.sh`, creating `.venv`, `pip install`,
appending a placeholder to `.env`, and starting the daemon.

Two levels of check, because the interesting ones need the packages: the repo-level
ones run under any Python 3, and this same file is then re-executed by
`.venv/bin/python --venv-probe` to reach AppKit, the resolved config and the
microphone. Comments here are English like the rest of the codebase; the output is
Turkish like the menu.
"""
import argparse
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent
VENV = ROOT / ".venv"
VENV_PY = VENV / "bin" / "python"
ENV_FILE = ROOT / ".env"
MIN_PY = (3, 12)                    # keep in step with run.sh's MIN_PY

# Where a daemon this script starts sends its output. Not in the repo: it is a
# runtime artefact, and ~/.audiocript is where the pid file and the history already
# live.
START_LOG = pathlib.Path("~/.audiocript/daemon-start.log").expanduser()

# The line dictate.py prints once the icon is in the menu bar. Matching on it is how
# "the daemon started" is verified rather than assumed — the process being alive only
# means it has not exited yet.
READY_LINE = "is now in the menu bar"

# How long to wait for that line. A first-ever start downloads nothing here (the
# model is loaded later, by the menu), so this only covers interpreter start-up and
# the config; generous anyway, because a cold import of torch is not free.
START_TIMEOUT_SECONDS = 90

# Every import dictation needs. pyannote.audio is left out on purpose: it is the
# optional speaker-labels dependency the TUI uses, and the menu bar never touches it.
REQUIRED_MODULES = ("AppKit", "Foundation", "numpy", "sounddevice", "rich",
                    "transformers", "huggingface_hub", "pywhispercpp")

# Apps that hide status items. A running one is the most common reason the icon is
# "not there" while the daemon is fine — it is a warning rather than a failure
# because it does not stop anything, and the app cannot detect the hiding itself.
MENU_BAR_MANAGERS = ("Ice", "Bartender", "Barbee", "Hidden Bar", "Dozer", "Vanilla",
                     "SketchyBar", "TopNotch")

OK, FIXED, WARN, FAIL, SKIP = "ok", "fixed", "warn", "fail", "skip"
MARK = {OK: "✔", FIXED: "⟳", WARN: "!", FAIL: "✖", SKIP: "-"}


class Report:
    """The rows, and the verdict they add up to."""

    def __init__(self):
        self.rows = []

    def add(self, status, label, detail="", fix=""):
        self.rows.append((status, label, detail, fix))
        width = 24
        line = f" {MARK[status]}  {label:<{width}} {detail}".rstrip()
        print(line, flush=True)
        if fix:
            print(f"        → {fix}", flush=True)
        return status

    def of(self, status):
        return [r for r in self.rows if r[0] == status]


# --------------------------------- repo level ---------------------------------

def check_repo(rep):
    """The files this script needs to be next to. Everything below assumes them."""
    missing = [n for n in ("audiocript.py", "dictate.py", "dictation.py",
                           "menubar.py", "run.sh", "requirements.txt")
               if not (ROOT / n).exists()]
    if missing:
        rep.add(FAIL, "repo klasörü", f"eksik dosya: {', '.join(missing)}",
                "doctor.py'yi audiocript repo'sunun kökünde çalıştır")
        return False
    rep.add(OK, "repo klasörü", str(ROOT))
    return True


def check_macos(rep):
    """14.4+ is a TUI requirement (Core Audio process taps); dictation does not need
    it, so an older system is a warning and not a refusal."""
    try:
        ver = subprocess.run(["sw_vers", "-productVersion"], capture_output=True,
                             text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        rep.add(WARN, "macOS", "sürüm okunamadı")
        return
    parts = tuple(int(p) for p in re.findall(r"\d+", ver)[:2])
    if parts and parts < (14, 4):
        rep.add(WARN, "macOS", f"{ver} — dictation çalışır, sistem sesi capture'ı için 14.4+ gerekiyor")
    else:
        rep.add(OK, "macOS", ver)


def check_run_sh(rep, fix):
    """run.sh without its executable bit is the documented `chmod +x` case."""
    path = ROOT / "run.sh"
    if os.access(path, os.X_OK):
        rep.add(OK, "run.sh", "çalıştırılabilir")
        return
    if not fix:
        rep.add(FAIL, "run.sh", "çalıştırma izni yok", "chmod +x run.sh")
        return
    path.chmod(path.stat().st_mode | 0o111)
    rep.add(FIXED, "run.sh", "chmod +x uygulandı")


def _version_of(python):
    """(major, minor, micro) for an interpreter, or None if it cannot be asked."""
    try:
        out = subprocess.run([str(python), "-c",
                              "import sys;print('.'.join(map(str,sys.version_info[:3])))"],
                             capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    try:
        return tuple(int(p) for p in out.stdout.strip().split("."))
    except ValueError:
        return None


def check_system_python(rep):
    """Which interpreter would build the venv. Returns its path, or None."""
    for name in ("python3.14", "python3.13", "python3.12", "python3"):
        found = shutil.which(name)
        if not found:
            continue
        ver = _version_of(found)
        if ver and ver[:2] >= MIN_PY:
            rep.add(OK, "python3", f"{'.'.join(map(str, ver))} ({found})")
            return found
    rep.add(FAIL, "python3", f"{'.'.join(map(str, MIN_PY))}+ bulunamadı",
            f"brew install python@{'.'.join(map(str, MIN_PY))}")
    return None


def check_venv(rep, fix, system_python):
    """The venv, and its interpreter's version — an environment built by an older
    Python is the case run.sh's second require_python call exists for."""
    if VENV_PY.exists():
        ver = _version_of(VENV_PY)
        if ver is None:
            rep.add(FAIL, ".venv", "interpreter cevap vermiyor",
                    "rm -rf .venv && ./run.sh --dictate")
            return False
        if ver[:2] < MIN_PY:
            rep.add(FAIL, ".venv", f"{'.'.join(map(str, ver))} — çok eski",
                    "rm -rf .venv && ./run.sh --dictate")
            return False
        rep.add(OK, ".venv", ".".join(map(str, ver)))
        return True
    if not fix:
        rep.add(FAIL, ".venv", "yok", "./run.sh --dictate")
        return False
    if not system_python:
        rep.add(FAIL, ".venv", "yok ve kuracak python3 de yok")
        return False
    print("        … .venv kuruluyor, ilk seferde uzun sürer", flush=True)
    try:
        subprocess.run([system_python, "-m", "venv", str(VENV)], check=True,
                       timeout=600)
    except (OSError, subprocess.SubprocessError) as e:
        rep.add(FAIL, ".venv", f"kurulamadı: {e}", "./run.sh --dictate")
        return False
    rep.add(FIXED, ".venv", "oluşturuldu")
    return True


def pip_install(rep):
    """Install requirements.txt into the venv. Idempotent: pip says 'already
    satisfied' and returns 0 when there is nothing to do.

    run.sh's own stamp (.venv/.requirements.sha) is deliberately not written here.
    Duplicating how it is computed is how the two drift apart; leaving it alone
    costs one no-op pip run on the next ./run.sh and cannot go wrong."""
    print("        … pip install -r requirements.txt (uzun sürebilir)", flush=True)
    try:
        out = subprocess.run([str(VENV_PY), "-m", "pip", "install", "-r",
                              str(ROOT / "requirements.txt")],
                             capture_output=True, text=True, timeout=3600)
    except (OSError, subprocess.SubprocessError) as e:
        rep.add(FAIL, "bağımlılıklar", f"pip çalışmadı: {e}")
        return False
    if out.returncode != 0:
        tail = (out.stderr or out.stdout).strip().splitlines()[-1:] or [""]
        rep.add(FAIL, "bağımlılıklar", f"pip hata verdi: {tail[0][:120]}",
                ".venv/bin/python -m pip install -r requirements.txt")
        return False
    return True


def check_env_file(rep, fix):
    """OPENAI_API_KEY has to exist for the daemon to reach the menu bar at all
    (dictation.resolve_config). The exported variable wins over .env, matching
    publish.env_value, so both count. The value is never read out loud."""
    if os.environ.get("OPENAI_API_KEY"):
        rep.add(OK, "OPENAI_API_KEY", "shell environment'ında tanımlı")
        return True
    has_key = placeholder = False
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8", errors="replace").splitlines():
            name, _, value = line.strip().partition("=")
            if name != "OPENAI_API_KEY":
                continue
            # The skeleton this script writes counts as absent, not as configured:
            # reporting it OK is how a setup passes the check and then refuses to
            # start.
            if value.strip() and "sk-..." not in value:
                has_key = True
                break
            placeholder = True
    if has_key:
        rep.add(OK, "OPENAI_API_KEY", ".env içinde tanımlı")
        return True
    if placeholder:
        rep.add(FAIL, "OPENAI_API_KEY", ".env'de sadece sk-... placeholder'ı var",
                f"{ENV_FILE} içindeki sk-... yerine gerçek key'i yaz")
        return False
    if not ENV_FILE.exists() and fix:
        # A skeleton, never a key: .env is gitignored precisely so that no key is
        # ever written by anything but the person who owns it.
        ENV_FILE.write_text("OPENAI_API_KEY=sk-...\n", encoding="utf-8")
        rep.add(FAIL, "OPENAI_API_KEY", ".env oluşturuldu, key'i sen yazacaksın",
                f"{ENV_FILE} dosyasındaki sk-... yerine gerçek key'i koy")
        return False
    rep.add(FAIL, "OPENAI_API_KEY", "ne environment'ta ne .env'de var",
            "printf 'OPENAI_API_KEY=sk-...\\n' > .env  (sonra gerçek key'i yaz)")
    return False


def check_menu_bar_managers(rep):
    """A running hider is reported by name, because "drag it back" is only actionable
    once the user knows which app is doing it."""
    running = []
    for app in MENU_BAR_MANAGERS:
        try:
            out = subprocess.run(["pgrep", "-x", app], capture_output=True, text=True,
                                 timeout=10)
        except (OSError, subprocess.SubprocessError):
            continue
        if out.returncode == 0 and out.stdout.strip():
            running.append(f"{app} ({out.stdout.split()[0]})")
    if running:
        rep.add(WARN, "menü bar yöneticisi", ", ".join(running) + " çalışıyor",
                "ikon gizlenmiş olabilir: Cmd basılı tutup menü bar'da sürükleyerek görünür bölüme al")
    else:
        rep.add(OK, "menü bar yöneticisi", "çalışan yok")


# --------------------------------- venv level ---------------------------------

def run_venv_probe(mic):
    """Re-execute this file inside the venv and bring back its JSON.

    Only the last '{'-line is parsed: an import that writes a warning to stdout
    would otherwise make the whole probe unreadable."""
    argv = [str(VENV_PY), str(ROOT / "doctor.py"), "--venv-probe"]
    if mic:
        argv.append("--mic")
    try:
        out = subprocess.run(argv, capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.SubprocessError) as e:
        return {"error": f"probe çalıştırılamadı: {e}"}
    for line in reversed(out.stdout.splitlines()):
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                break
    detail = (out.stderr or out.stdout).strip().splitlines()[-1:] or ["çıktı yok"]
    return {"error": detail[0][:200]}


def venv_probe(mic):
    """What only the venv can answer. Prints one JSON object to stdout and nothing
    else; every failure is carried in the object rather than raised, so the parent
    always gets a readable answer."""
    result = {"modules": {}, "screens": []}
    for name in REQUIRED_MODULES:
        try:
            __import__(name)
            result["modules"][name] = None
        except BaseException as e:                      # noqa: BLE001 - reported, not handled
            result["modules"][name] = f"{type(e).__name__}: {e}"

    try:
        import audiocript as A
        import dictation
        import dictate
        try:
            dictation.resolve_config(A.load_config())
            result["config"] = {"ok": True}
        except Exception as e:
            result["config"] = {"error": str(e)}
        result["language"] = A.load_config().get("language", "tr")
        result["idle_icon"] = dictation.icon(dictation.POWER_OFF, dictation.IDLE)
        result["ready_icon"] = dictation.icon(dictation.POWER_ON, dictation.IDLE)
        result["history_path"] = str(dictation.HISTORY_PATH)
        result["pid_path"] = str(dictate.PID_PATH)
        result["pid"] = dictate.read_pid()
    except BaseException as e:                          # noqa: BLE001
        result["config"] = {"error": f"import: {type(e).__name__}: {e}"}

    try:
        import AppKit
        app = AppKit.NSApplication.sharedApplication()
        app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)
        bar = AppKit.NSStatusBar.systemStatusBar()
        item = bar.statusItemWithLength_(AppKit.NSVariableStatusItemLength)
        item.button().setTitle_("…")
        result["status_item"] = {"created": True, "visible": bool(item.isVisible())}
        # Give it back at once: this probe must not leave an icon behind.
        bar.removeStatusItem_(item)
        for screen in AppKit.NSScreen.screens():
            frame = screen.frame()
            result["screens"].append({
                "w": int(frame.size.width), "h": int(frame.size.height),
                # A non-zero top safe area is the notch. Measured on this machine:
                # 0.0 on an external 1920x1080, 38.0 on the built-in display.
                "notch": float(screen.safeAreaInsets().top) > 0})
    except BaseException as e:                          # noqa: BLE001
        result["status_item"] = {"created": False, "error": f"{type(e).__name__}: {e}"}

    if mic:
        try:
            import numpy
            import sounddevice
            device = sounddevice.query_devices(kind="input")
            frames = int(0.5 * 16000)
            rec = sounddevice.rec(frames, samplerate=16000, channels=1,
                                  dtype="float32")
            sounddevice.wait()
            # Exactly zero is the signal, not "quiet". A microphone macOS refuses
            # still opens and delivers zeros; a live one in a silent room does not
            # (measured 0.0029 peak here).
            result["mic"] = {"device": device["name"],
                             "peak": float(numpy.max(numpy.abs(rec))),
                             "zero": bool(numpy.all(rec == 0))}
        except BaseException as e:                      # noqa: BLE001
            result["mic"] = {"error": f"{type(e).__name__}: {e}"}

    print(json.dumps(result))
    return 0


def report_probe(rep, probe):
    """Turn the probe's JSON into rows. Returns True when nothing in it failed."""
    if "error" in probe and "modules" not in probe:
        rep.add(FAIL, "venv kontrolü", probe["error"],
                ".venv/bin/python -m pip install -r requirements.txt")
        return False
    good = True
    broken = {n: e for n, e in probe.get("modules", {}).items() if e}
    if broken:
        good = False
        rep.add(FAIL, "python paketleri", "eksik: " + ", ".join(sorted(broken)),
                ".venv/bin/python -m pip install -r requirements.txt")
    else:
        rep.add(OK, "python paketleri", f"{len(REQUIRED_MODULES)} paket tamam")

    config = probe.get("config") or {"error": "kontrol edilemedi"}
    if config.get("ok"):
        rep.add(OK, "dictation config", f"dil: {probe.get('language', '?')}")
    else:
        good = False
        error = config["error"]
        # An import that failed is a broken checkout or a broken venv, not a value
        # the user typed wrong; sending them to config.json would waste their time.
        hint = (".venv/bin/python -m pip install -r requirements.txt  (ya da eksik "
                "repo dosyası var: git status)" if error.startswith("import:")
                else "config.json / .env değerlerini düzelt")
        rep.add(FAIL, "dictation config", error, hint)

    item = probe.get("status_item") or {}
    if item.get("created"):
        rep.add(OK, "menü bar status item", f"oluşturulabiliyor (visible={item.get('visible')})")
    else:
        good = False
        rep.add(FAIL, "menü bar status item", item.get("error", "oluşturulamadı"),
                "GUI session'da mı çalışıyorsun? ssh/tmux üzerinden menü bar açılmaz")

    notched = [s for s in probe.get("screens", []) if s.get("notch")]
    if notched:
        rep.add(WARN, "çentikli ekran", f"{len(notched)} ekranda notch var",
                "menü bar doluysa ikon çentiğin altına düşer; birkaç ikon kapatıp yer aç")

    mic = probe.get("mic")
    if mic and "error" in mic:
        good = False
        rep.add(FAIL, "mikrofon", mic["error"])
    elif mic and mic.get("zero"):
        good = False
        rep.add(FAIL, "mikrofon", f"{mic['device']} — hep sıfır geliyor",
                "System Settings → Privacy & Security → Microphone: terminal uygulamasına izin ver")
    elif mic:
        rep.add(OK, "mikrofon", f"{mic['device']} (peak {mic['peak']:.4f})")
    return good


# --------------------------------- the daemon ---------------------------------

def daemon_pid():
    """The running daemon's pid, or None. Uses `ps` rather than the venv, so it works
    even when the venv probe could not run: a recycled pid that is not a dictate.py
    is not our daemon."""
    path = pathlib.Path("~/.audiocript/dictate.pid").expanduser()
    try:
        pid = int(path.read_text().strip())
    except (OSError, ValueError):
        return None
    try:
        out = subprocess.run(["ps", "-p", str(pid), "-o", "command="],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0 or "dictate.py" not in out.stdout:
        return None
    return pid


def stop_daemon(rep):
    """The documented clean stop: dictate.py --stop, which is what Çıkış does."""
    try:
        subprocess.run([str(VENV_PY), str(ROOT / "dictate.py"), "--stop"],
                       cwd=ROOT, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as e:
        rep.add(WARN, "daemon durdurma", str(e))
        return
    for _ in range(60):                                # up to ~30s to drain and exit
        if daemon_pid() is None:
            rep.add(FIXED, "daemon", "durduruldu")
            return
        time.sleep(0.5)
    rep.add(WARN, "daemon", "durdurma isteği gönderildi ama hâlâ çalışıyor")


def start_daemon(rep):
    """Start ./run.sh --dictate detached, then wait for the line that means the icon
    is up. Detached on purpose: this is a menu bar app, so it has to outlive the
    check that started it."""
    START_LOG.parent.mkdir(parents=True, exist_ok=True)
    marker = f"\n===== doctor.py başlattı: {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n"
    with open(START_LOG, "a", encoding="utf-8") as log:
        log.write(marker)
        log.flush()
        try:
            subprocess.Popen([str(ROOT / "run.sh"), "--dictate"], cwd=ROOT,
                             stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                             start_new_session=True)
        except OSError as e:
            rep.add(FAIL, "daemon", f"başlatılamadı: {e}", "./run.sh --dictate")
            return False
    print(f"        … daemon başlatıldı, log: {START_LOG}", flush=True)
    deadline = time.time() + START_TIMEOUT_SECONDS
    while time.time() < deadline:
        text = START_LOG.read_text(encoding="utf-8", errors="replace")
        after = text.rsplit(marker, 1)[-1]
        if READY_LINE in after:
            rep.add(FIXED, "daemon", f"başladı (pid {daemon_pid()}), ikon menü bar'da")
            return True
        if "dictation cannot start" in after:
            line = next(l for l in after.splitlines() if "dictation cannot start" in l)
            rep.add(FAIL, "daemon", line.strip())
            return False
        time.sleep(1)
    rep.add(FAIL, "daemon", f"{START_TIMEOUT_SECONDS}s içinde hazır olmadı",
            f"log'a bak: tail -30 {START_LOG}")
    return False


def check_daemon(rep, fix, restart, probe_ok):
    """The last gate: is there a daemon holding an icon right now?"""
    pid = daemon_pid()
    if pid and restart:
        stop_daemon(rep)
        pid = daemon_pid()
    if pid:
        rep.add(OK, "daemon", f"çalışıyor (pid {pid})")
        return True
    if not fix:
        rep.add(FAIL, "daemon", "çalışmıyor", "./run.sh --dictate")
        return False
    if not probe_ok:
        # Starting on top of a known-bad config is how the log fills with the same
        # refusal every run; the rows above already say what to fix.
        rep.add(FAIL, "daemon", "çalışmıyor — yukarıdaki sorunlar düzelmeden başlatılmadı")
        return False
    return start_daemon(rep)


# --------------------------------- visual probe ---------------------------------

def visual_probe():
    """A plain AppKit status item for 15 seconds, with nothing of Audiocript in it.

    This is the one thing no check can decide: if AC-TEST is not on screen while
    this runs, the menu bar is hiding items and the app is not at fault."""
    import AppKit
    import Foundation
    app = AppKit.NSApplication.sharedApplication()
    app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)
    item = AppKit.NSStatusBar.systemStatusBar().statusItemWithLength_(
        AppKit.NSVariableStatusItemLength)
    item.button().setTitle_("AC-TEST")
    print(f"        … AC-TEST menü bar'da (visible={item.isVisible()}), 15 saniye bak",
          flush=True)

    def stop(_timer):
        app.stop_(None)
        # stop_ alone does not end the loop — see menubar.stop_the_loop.
        app.postEvent_atStart_(
            AppKit.NSEvent.otherEventWithType_location_modifierFlags_timestamp_windowNumber_context_subtype_data1_data2_(
                AppKit.NSEventTypeApplicationDefined, Foundation.NSZeroPoint, 0, 0, 0,
                None, 0, 0, 0), True)

    Foundation.NSTimer.scheduledTimerWithTimeInterval_repeats_block_(15.0, False, stop)
    app.run()
    return 0


# ------------------------------------ main ------------------------------------

def run_visual_probe(wanted):
    """The eye check, after the verdict: it blocks for 15 seconds, and the verdict is
    what the user is waiting for."""
    if not wanted:
        return
    print()
    subprocess.run([str(VENV_PY), str(ROOT / "doctor.py"), "--visual-probe"],
                   cwd=ROOT, timeout=120)
    print("AC-TEST'i gördüysen menü bar Audiocript'i de gösterir; görmediysen "
          "menü bar ikonu gizliyor.")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Audiocript menü bar kurulum kontrolü")
    parser.add_argument("--check", "-n", action="store_true",
                        help="hiçbir şeyi değiştirme, sadece raporla")
    parser.add_argument("--restart", action="store_true",
                        help="çalışan daemon'ı durdurup yeniden başlat")
    parser.add_argument("--probe", action="store_true",
                        help="menü bar'a 15 saniyelik AC-TEST ikonu koy")
    parser.add_argument("--no-mic", action="store_true",
                        help="mikrofon testini atla")
    parser.add_argument("--venv-probe", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--mic", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--visual-probe", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.venv_probe:                     # re-executed inside the venv
        return venv_probe(args.mic)
    if args.visual_probe:
        return visual_probe()

    fix = not args.check
    rep = Report()
    print("Audiocript menü bar kurulum kontrolü")
    print("─" * 60)

    if not check_repo(rep):
        return 1
    check_macos(rep)
    check_run_sh(rep, fix)
    system_python = check_system_python(rep)
    have_venv = check_venv(rep, fix, system_python)
    check_env_file(rep, fix)
    check_menu_bar_managers(rep)

    probe_ok = False
    if have_venv:
        probe = run_venv_probe(mic=not args.no_mic)
        if probe.get("modules") and any(probe["modules"].values()) and fix:
            # Missing packages are the one venv failure worth fixing and re-asking:
            # everything after them reads as broken while they are absent.
            if pip_install(rep):
                probe = run_venv_probe(mic=not args.no_mic)
        probe_ok = report_probe(rep, probe)
    else:
        rep.add(SKIP, "venv kontrolleri", ".venv olmadan yapılamaz")

    daemon_ok = check_daemon(rep, fix, args.restart, probe_ok)

    print("─" * 60)
    fails, warns = rep.of(FAIL), rep.of(WARN)
    if not fails and daemon_ok:
        icon = "🎙"
        if warns:
            print(f"HERŞEY OK — daemon ayakta. Menü bar'da {icon} ikonunu ara "
                  f"(daemon kapalıysa ◯).")
            print(f"Ama {len(warns)} uyarı var, ikonu göremiyorsan sebebi bunlar:")
            for i, (_, label, detail, hint) in enumerate(warns, 1):
                print(f"  {i}. {label}: {detail}")
                if hint:
                    print(f"     → {hint}")
            if not args.probe:
                print("Göz kontrolü için: python3 doctor.py --probe")
        else:
            print(f"HERŞEY OK — daemon ayakta, menü bar'da ◯ ikonu var. "
                  f"Tıkla → Daemon'ı başlat → {icon}")
        run_visual_probe(args.probe and have_venv)
        return 0

    print(f"{len(fails)} SORUN kaldı:")
    for i, (_, label, detail, hint) in enumerate(fails, 1):
        print(f"  {i}. {label}: {detail}")
        if hint:
            print(f"     → {hint}")
    if warns:
        print(f"({len(warns)} uyarı da var, yukarıda işaretli.)")
    run_visual_probe(args.probe and have_venv)
    return 1


if __name__ == "__main__":
    sys.exit(main())
