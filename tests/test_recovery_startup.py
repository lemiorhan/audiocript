"""Recovery must run behind the UI, not in front of it.

Rebuilding an interrupted recording used to block main() before Live started, so the
terminal stayed blank for as long as it took — which read as a hang. These tests
drive the real app on a pty against a throwaway HOME.
"""
import json
import os
import pty
import re
import select
import signal
import subprocess
import sys
import time

import numpy as np

from support import REPO_ROOT, SOURCES, build_capture, run, workdir


def launch(home, run_for):
    """Run the app on a pty until run_for seconds pass, then interrupt it.

    Returns (first_frame_s, recovered_s, text): when the UI first painted and when
    the recovery completion message appeared, both relative to launch."""
    env = dict(os.environ, HOME=str(home), TERM="xterm-256color")
    master, slave = pty.openpty()
    p = subprocess.Popen([sys.executable, "audiocript.py"], stdin=slave, stdout=slave,
                         stderr=slave, cwd=str(REPO_ROOT), env=env)
    os.close(slave)
    t0, first, recovered, buf = time.time(), None, None, b""
    try:
        while time.time() - t0 < run_for:
            r, _, _ = select.select([master], [], [], 0.2)
            if not r:
                continue
            try:
                chunk = os.read(master, 65536)
            except OSError:
                break
            if not chunk:
                break
            if first is None and b"Audiocript" in chunk:
                first = time.time() - t0
            buf += chunk
            if recovered is None and "Recovered 1 interrupted" in buf.decode("utf-8", "ignore"):
                recovered = time.time() - t0
    finally:
        p.send_signal(signal.SIGINT)
        time.sleep(2)
        if p.poll() is None:
            p.kill()
        p.wait()
        os.close(master)
    return first, recovered, buf.decode("utf-8", "ignore")


def interrupted_home(home, seconds):
    """A fake HOME holding one recording that was killed mid-capture."""
    d = home / "Audiocript" / "recordings" / "2026-08-07_13-47-05"
    build_capture(d, seconds, seed=2)
    meta = json.loads((d / "meta.json").read_text())
    meta["name"] = "killed-meeting"
    (d / "meta.json").write_text(json.dumps(meta))
    return d


def test_ui_paints_before_recovery_finishes():
    """The size-independent proof that recovery no longer blocks startup: the first
    frame must land before recovery is done, not after it."""
    with workdir("startup") as home:
        interrupted_home(home, 600)
        first, recovered, text = launch(home, 30)
        print(f"  first frame {first if first is None else round(first, 2)}s, "
              f"recovery done {recovered if recovered is None else round(recovered, 2)}s")
        assert first is not None, "the UI never painted"
        assert recovered is not None, "recovery never finished inside the window"
        assert first < recovered - 0.5, (
            f"first frame {first:.2f}s vs recovery {recovered:.2f}s — still blocking")
        assert "Recovering" in text, "the status line never showed recovery"
        assert re.search(r"Recovering.*\d+%", text), "no percentage on the status line"


def test_recovery_completes_and_the_recording_appears():
    with workdir("complete") as home:
        d = interrupted_home(home, 20)
        first, recovered, text = launch(home, 60)
        print(f"  first frame {round(first, 2)}s, recovery done {round(recovered, 2)}s")
        assert recovered is not None, "no completion message"
        assert (d / "audio.wav").exists(), "audio.wav missing after recovery"
        assert not (d / "mic.raw").exists(), "raw files not cleared after success"
        assert not list(d.glob("*.part")), "left .part files behind"
        meta = json.loads((d / "meta.json").read_text())
        assert meta["capture"]["in_progress"] is False, "in_progress flag not cleared"
        assert "killed-meeting" in text, "the recovered recording never reached the list"


if __name__ == "__main__":
    run(["test_ui_paints_before_recovery_finishes",
         "test_recovery_completes_and_the_recording_appears"], globals())
