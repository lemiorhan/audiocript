"""_finalize_sources: identical output, bounded memory, and no way to lose a take.

The last two matter most. Finalize used to read whole recordings into memory, and it
wrote straight to audio.wav — so a process killed mid-write left a truncated file
that the next launch accepted as finished and deleted the raw capture behind.
"""
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from support import (A, REPO_ROOT, SOURCES, build_capture, child_peak_rss_mb, run,
                     wav_samples, workdir)

# What a 60x longer recording may add to peak memory. Finalize used to read each
# capture whole, so memory tracked the recording's length; the growth is the signal.
MEM_GROWTH_BUDGET_MB = 300


def reference_audio(d, sources=SOURCES):
    """What the pre-streaming code produced: read each source whole, resample, mix."""
    parts = []
    for s in sources:
        raw = np.fromfile(Path(d) / s["file"], dtype=np.dtype(s["dtype"]))
        parts.append(A.resample_to_target(raw.reshape(-1, s["channels"]), s["rate"], 16000))
    n = min(len(p) for p in parts)
    acc = np.zeros(n, dtype=np.int32)
    for p in parts:
        acc += p[:n].astype(np.int32)
    peak = int(np.max(np.abs(acc)))
    if peak > 32767:
        acc = (acc * (32767.0 / peak)).astype(np.int32)
    return acc.astype(np.int16)


def test_output_matches_the_in_memory_result():
    for loud in (False, True):
        with workdir(f"eq-{'loud' if loud else 'quiet'}") as d:
            build_capture(d, 6, seed=1, loud=loud)
            ref = reference_audio(d)
            n = A._finalize_sources(d, SOURCES, save_channels=True)
            got = wav_samples(d / "audio.wav")
            print(f"  loud={loud}: {n} samples, peak {int(np.abs(got).max())}")
            assert n == len(ref) and np.array_equal(got, ref), f"loud={loud}: output differs"
            assert (d / "mic.wav").exists() and (d / "system.wav").exists()
            assert not list(d.glob("*.part")), "left .part files behind"


def test_save_channels_off_removes_the_channel_wavs():
    with workdir("nochan") as d:
        build_capture(d, 3, seed=2)
        A._finalize_sources(d, SOURCES, save_channels=False)
        assert (d / "audio.wav").exists()
        assert not (d / "mic.wav").exists() and not (d / "system.wav").exists()
        assert not list(d.glob("*.part"))
        print("  channel WAVs removed, audio.wav kept")


def test_a_missing_source_is_skipped():
    with workdir("onesrc") as d:
        build_capture(d, 3, seed=4)
        (d / "system.raw").unlink()
        n = A._finalize_sources(d, SOURCES, save_channels=False)
        raw = np.fromfile(d / "mic.raw", dtype=np.int16).reshape(-1, 2)
        ref = A.resample_to_target(raw, 44100, 16000)
        print(f"  single source: {n} samples")
        assert n == len(ref) and np.array_equal(wav_samples(d / "audio.wav"), ref)


def test_single_source_with_channels_writes_both():
    with workdir("onesrc-chan") as d:
        build_capture(d, 3, seed=9)
        (d / "system.raw").unlink()
        A._finalize_sources(d, SOURCES, save_channels=True)
        assert (d / "audio.wav").exists() and (d / "mic.wav").exists()
        assert np.array_equal(wav_samples(d / "audio.wav"), wav_samples(d / "mic.wav"))
        assert not list(d.glob("*.part"))
        print("  audio.wav and mic.wav both written")


def test_a_capture_with_no_data_writes_nothing():
    with workdir("empty") as d:
        build_capture(d, 0, seed=10)
        for s in SOURCES:
            (d / s["file"]).write_bytes(b"")
        assert A._finalize_sources(d, SOURCES) == 0
        assert not (d / "audio.wav").exists()
        print("  empty capture -> 0, no audio.wav")


def test_progress_climbs_to_one():
    """The status line shows this fraction, so it must be monotonic and finish at 1."""
    with workdir("progress") as d:
        build_capture(d, 4, seed=11)
        seen = []
        A._finalize_sources(d, SOURCES, save_channels=False, on_progress=seen.append)
        monotonic = all(b >= a - 1e-9 for a, b in zip(seen, seen[1:]))
        print(f"  {len(seen)} updates, last {seen[-1]:.2f}, monotonic={monotonic}")
        assert seen and seen[-1] == 1.0, f"ended at {seen[-1] if seen else None}"
        assert all(0.0 <= v <= 1.0 for v in seen), "progress left [0, 1]"
        assert monotonic, "progress went backwards"


def test_a_killed_finalize_leaves_the_capture_recoverable():
    """Kill mid-finalize: no audio.wav, raw files intact, only .part debris.

    The child drops a marker as soon as finalize reports progress, so the kill lands
    while it is genuinely working instead of racing a sleep."""
    with workdir("kill") as d:
        build_capture(d, 600, seed=6)
        marker, runner = d / "started.marker", d / "runner.py"
        runner.write_text(
            "import sys, json\n"
            f"sys.path.insert(0, {str(REPO_ROOT)!r})\n"
            "import audiocript as A\n"
            f"d, marker = {str(d)!r}, {str(marker)!r}\n"
            "src = json.load(open(d + '/meta.json'))['capture']['sources']\n"
            "A._finalize_sources(d, src, save_channels=True,\n"
            "                    on_progress=lambda f: open(marker, 'w').write(str(f)))\n")
        p = subprocess.Popen([sys.executable, str(runner)], cwd=str(REPO_ROOT))
        deadline = time.time() + 120
        while not marker.exists() and p.poll() is None and time.time() < deadline:
            time.sleep(0.05)
        still_running = p.poll() is None
        p.kill()
        p.wait()
        print(f"  killed mid-run={still_running}; left {sorted(f.name for f in d.iterdir())}")
        assert marker.exists(), "finalize never reported progress"
        assert still_running, "finalize finished before the kill — test proved nothing"
        assert not (d / "audio.wav").exists(), "a truncated audio.wav looks finished"
        assert (d / "mic.raw").exists() and (d / "system.raw").exists(), "raw capture lost"


def test_stale_part_files_are_cleared():
    with workdir("stale") as d:
        build_capture(d, 3, seed=12)
        (d / "audio.wav.part").write_bytes(b"garbage")
        (d / "mic.wav.part").write_bytes(b"garbage")
        n = A._finalize_sources(d, SOURCES, save_channels=True)
        assert n and (d / "audio.wav").exists() and not list(d.glob("*.part"))
        print("  stale .part files cleared, finalize succeeded")


def test_memory_does_not_scale_with_length():
    """A 30-minute two-source capture used to peak at gigabytes, against a few hundred
    MB for a short one — finalize read every source into memory whole."""
    peaks = {}
    for seconds in (30, 1800):
        with workdir(f"mem-{seconds}") as d:
            build_capture(d, seconds, seed=8)
            peaks[seconds] = child_peak_rss_mb(
                f"A._finalize_sources({str(d)!r}, SOURCES, save_channels=True)\n")
    growth = peaks[1800] - peaks[30]
    print(f"  30 s: {peaks[30]:.0f} MB, 30 min: {peaks[1800]:.0f} MB, "
          f"growth {growth:.0f} MB")
    assert growth < MEM_GROWTH_BUDGET_MB, (
        f"60x the recording added {growth:.0f} MB, budget {MEM_GROWTH_BUDGET_MB} MB")


if __name__ == "__main__":
    run(["test_output_matches_the_in_memory_result",
         "test_save_channels_off_removes_the_channel_wavs",
         "test_a_missing_source_is_skipped",
         "test_single_source_with_channels_writes_both",
         "test_a_capture_with_no_data_writes_nothing",
         "test_progress_climbs_to_one",
         "test_a_killed_finalize_leaves_the_capture_recoverable",
         "test_stale_part_files_are_cleared",
         "test_memory_does_not_scale_with_length"], globals())
