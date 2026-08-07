"""The streaming writers must match what the in-memory mix produced.

Mixing is where a regression would be audible rather than obvious: the peak scaling
is global, so getting it wrong quietly clips or over-attenuates the whole file.
"""
import numpy as np

from support import A, reference_mix, run, wav_samples, workdir, write_wav


def test_stream_source_matches_resample():
    """_stream_source_to_wav writes exactly what resample_to_target would return."""
    with workdir("source") as d:
        rng = np.random.default_rng(7)
        frames, rate = 48000 * 9, 48000
        arr = (0.5 * rng.standard_normal((frames, 2))).astype(np.float32)
        p = d / "src.raw"
        arr.tofile(p)
        ref = A.resample_to_target(arr, rate, 16000)
        out = d / "src.wav"
        with A._RawPcmReader(p, "<f4", 2) as r:
            n = A._stream_source_to_wav(r, rate, out)
        got = wav_samples(out)
        print(f"  wrote {n} samples, reference {len(ref)}")
        assert n == len(ref) and len(got) == len(ref), f"length {n}/{len(got)} != {len(ref)}"
        assert int(np.abs(got.astype(np.int32) - ref.astype(np.int32)).max()) == 0


def test_mix_matches_reference_without_clipping():
    with workdir("mix") as d:
        a = (np.sin(np.linspace(0, 900, 160000)) * 8000).astype(np.int16)
        b = (np.sin(np.linspace(0, 500, 150000)) * 6000).astype(np.int16)
        write_wav(d / "a.wav", a)
        write_wav(d / "b.wav", b)
        n = A._mix_wavs_to_wav([d / "a.wav", d / "b.wav"], d / "mix.wav")
        ref = reference_mix([a, b])
        got = wav_samples(d / "mix.wav")
        print(f"  {n} samples (shortest source), peak {int(np.abs(got).max())}")
        assert n == len(ref) and np.array_equal(got, ref), "mix differs from reference"


def test_mix_matches_reference_when_the_sum_clips():
    """Two loud sources whose sum passes int16 — the peak-scaling branch."""
    with workdir("mixclip") as d:
        a = (np.sin(np.linspace(0, 900, 300000)) * 30000).astype(np.int16)
        b = (np.sin(np.linspace(0, 901, 300000)) * 28000).astype(np.int16)
        write_wav(d / "a.wav", a)
        write_wav(d / "b.wav", b)
        n = A._mix_wavs_to_wav([d / "a.wav", d / "b.wav"], d / "mix.wav")
        ref = reference_mix([a, b])
        got = wav_samples(d / "mix.wav")
        print(f"  {n} samples, peak {int(np.abs(got).max())} (must be <= 32767)")
        assert int(np.abs(got).max()) <= 32767, "mix clipped"
        assert n == len(ref) and np.array_equal(got, ref), "scaled mix differs from reference"


if __name__ == "__main__":
    run(["test_stream_source_matches_resample", "test_mix_matches_reference_without_clipping",
         "test_mix_matches_reference_when_the_sum_clips"], globals())
