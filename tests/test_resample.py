"""resample_to_target: bounded memory on long input, unchanged output.

Guards the startup hang. Resampling a whole recording in one torchaudio call made
PyTorch's CPU conv1d materialise an im2col buffer of hundreds of bytes per input
sample, so a 48-minute capture asked for tens of GB and never finished.
"""
import time

import numpy as np
import torch
import torchaudio.functional as AF

from support import A, child_peak_rss_mb, run

# Peak memory this may cost per input sample. resample_to_target is handed an array,
# so it can never be flat in the recording's length — a few copies of the samples are
# unavoidable. What it must not do is what the old code did: torchaudio's conv1d
# allocated an im2col buffer of ~470 bytes per sample, which is how a 48-minute
# capture came to ask for tens of GB. Measured as growth between two lengths, so the
# interpreter's own weight (torch alone is a few hundred MB) cancels out.
MEM_PER_SAMPLE_BUDGET = 100
TIME_BUDGET_S = 30.0


def noise(n, rate, seed=0):
    """A deterministic stereo float32 signal: a tone plus a little noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float32) / rate
    sig = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.1 * rng.standard_normal(n).astype(np.float32)
    return np.stack([sig, sig * 0.8], axis=1).astype(np.float32)


def _resample_seconds(seconds):
    """Peak RSS of a child that resamples `seconds` of 48 kHz stereo audio."""
    return child_peak_rss_mb(
        "import time\n"
        "from test_resample import noise\n"
        f"arr = noise(48000 * {seconds}, 48000)\n"
        "t = time.time()\n"
        "out = A.resample_to_target(arr, 48000, 16000)\n"
        "dt = time.time() - t\n"
        f"assert len(out) == 16000 * {seconds}, f'bad length {{len(out)}}'\n"
        f"assert dt < {TIME_BUDGET_S}, f'took {{dt:.1f}}s'\n")


def test_memory_per_sample_stays_small():
    """Five minutes at 48 kHz used to cost ~6.8 GB. It now costs a few copies of the
    samples themselves, which is the floor for an array-in, array-out function."""
    small, large = _resample_seconds(5), _resample_seconds(300)
    added_samples = 48000 * (300 - 5)
    per_sample = (large - small) * 1e6 / added_samples
    print(f"  5 s: {small:.0f} MB, 300 s: {large:.0f} MB "
          f"-> {per_sample:.0f} bytes per added sample")
    assert per_sample < MEM_PER_SAMPLE_BUDGET, (
        f"{per_sample:.0f} bytes/sample, budget {MEM_PER_SAMPLE_BUDGET}")


def test_matches_a_single_torchaudio_call():
    """Blocking the resample must not change the samples it produces."""
    kw = dict(lowpass_filter_width=64, rolloff=0.945,
              resampling_method="sinc_interp_kaiser", beta=14.769656459379492)
    for rate in (48000, 44100):
        arr = noise(int(rate * 20), rate, seed=1)
        got = A.resample_to_target(arr, rate, 16000)
        mono = arr.mean(axis=1)
        ref_f = AF.resample(torch.from_numpy(np.ascontiguousarray(mono)),
                            orig_freq=rate, new_freq=16000, **kw).numpy()
        ref = np.clip(np.round(ref_f * 32768.0), -32768, 32767).astype(np.int16)
        assert len(got) == len(ref), f"{rate}: length {len(got)} != {len(ref)}"
        diff = int(np.abs(got.astype(np.int32) - ref.astype(np.int32)).max())
        print(f"  {rate} -> 16000: {len(got)} samples, max diff {diff} LSB")
        # A single LSB can differ where float rounding lands exactly on .5.
        assert diff <= 1, f"{rate}: max diff {diff}"


if __name__ == "__main__":
    run(["test_memory_per_sample_stays_small",
         "test_matches_a_single_torchaudio_call"], globals())
