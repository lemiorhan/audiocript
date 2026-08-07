"""Shared helpers for the test scripts.

The project has no test framework, so each test file is a plain script that runs
its checks and exits non-zero on failure. This module carries what they all need:
importing the app, building throwaway capture fixtures, and measuring peak memory
honestly.
"""
import contextlib
import json
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import audiocript as A                                        # noqa: E402

# The capture manifest a mic + system-audio recording writes into meta.json.
SOURCES = [
    {"kind": "mic", "file": "mic.raw", "rate": 44100, "channels": 2, "dtype": "int16"},
    {"kind": "system", "file": "system.raw", "rate": 48000, "channels": 2, "dtype": "<f4"},
]


@contextlib.contextmanager
def workdir(name):
    """A temporary directory, removed afterwards. Fixtures here get large (a
    30-minute two-source capture is about 1 GB), so they never live in the repo."""
    path = Path(tempfile.mkdtemp(prefix=f"audiocript-{name}-"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def build_capture(d, seconds, seed=0, loud=False, sources=SOURCES, chunk_seconds=10):
    """Write a raw capture and its meta.json into d, as an interrupted recording
    would have left them. `loud` makes the sources sum past int16 so the mix has to
    scale itself down.

    Written in chunks: a 30-minute two-source fixture is about 1 GB, and building it
    in one array would cost more memory than the code under test is allowed to use."""
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    amp = 0.95 if loud else 0.3
    for s in sources:
        dtype, total = np.dtype(s["dtype"]), int(s["rate"] * seconds)
        with open(d / s["file"], "wb") as fh:
            for start in range(0, total, s["rate"] * chunk_seconds):
                n = min(s["rate"] * chunk_seconds, total - start)
                t = (np.arange(start, start + n, dtype=np.float32) / s["rate"])
                sig = amp * np.sin(2 * np.pi * 300 * t)
                sig += 0.05 * rng.standard_normal(n).astype(np.float32)
                arr = np.stack([sig, sig * 0.9], axis=1)[:, :s["channels"]]
                if dtype.kind != "f":
                    arr = np.clip(np.round(arr * 32000), -32768, 32767)
                fh.write(arr.astype(dtype).tobytes())
    (d / "meta.json").write_text(json.dumps(
        {"capture": {"in_progress": True, "sources": sources}}))
    return d


def wav_samples(path):
    """Every sample of a mono 16-bit WAV as an int16 array."""
    with wave.open(str(path), "rb") as wf:
        return np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)


def write_wav(path, samples, rate=16000):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(samples.tobytes())


def reference_mix(arrays):
    """The mixing the app did before it streamed: truncate to the shortest source,
    sum as int32, and scale by 32767/peak if that clips.

    Deliberately a frozen copy rather than a call into audiocript — it is the thing
    the streaming implementation is checked against, so it must not follow it."""
    n = min(len(a) for a in arrays)
    acc = np.zeros(n, dtype=np.int32)
    for a in arrays:
        acc += a[:n].astype(np.int32)
    peak = int(np.max(np.abs(acc))) if n else 0
    if peak > 32767:
        acc = (acc * (32767.0 / peak)).astype(np.int32)
    return acc.astype(np.int16)


def child_peak_rss_mb(source):
    """Run `source` in a fresh interpreter and return the peak RSS it reached, in MB.

    Memory has to be measured in a child: ru_maxrss is a high-water mark that never
    falls, so a second measurement in the same process reports whatever the first one
    peaked at and a regression would pass unnoticed."""
    prelude = (
        "import resource, sys\n"
        f"sys.path.insert(0, {str(REPO_ROOT / 'tests')!r})\n"
        "from support import *\n"
    )
    epilogue = (
        "\nprint('PEAK_RSS_MB', "
        "resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6)\n"
    )
    out = subprocess.run([sys.executable, "-c", prelude + source + epilogue],
                         capture_output=True, text=True, cwd=str(REPO_ROOT))
    if out.returncode != 0:
        raise AssertionError(f"child failed:\n{out.stdout}\n{out.stderr}")
    for line in reversed(out.stdout.splitlines()):
        if line.startswith("PEAK_RSS_MB"):
            return float(line.split()[1])
    raise AssertionError(f"child printed no peak RSS:\n{out.stdout}")


def run(names, namespace):
    """Run the named test functions, print a line each, and exit non-zero if any
    failed. Keeps every test file's tail down to one call."""
    failed = []
    for name in names:
        print(f"{name}:")
        try:
            namespace[name]()
            print("  PASS")
        except AssertionError as e:
            print(f"  FAIL: {e}")
            failed.append(name)
    if failed:
        print(f"\n{len(failed)} failed: {', '.join(failed)}")
    sys.exit(1 if failed else 0)
