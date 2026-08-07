"""_RawPcmReader + _resample_stream must reproduce resample_to_target exactly.

The reader is what lets finalize work off disk instead of holding a whole recording
in memory, so its samples have to be indistinguishable from the in-memory path.
"""
import numpy as np

from support import A, run, workdir


def make_raw(path, frames, rate, dtype, channels, seed=0):
    """Write a deterministic interleaved raw PCM file and return the array."""
    rng = np.random.default_rng(seed)
    t = np.arange(frames, dtype=np.float32) / rate
    sig = 0.4 * np.sin(2 * np.pi * 440 * t) + 0.15 * rng.standard_normal(frames).astype(np.float32)
    arr = np.stack([sig, sig * 0.7], axis=1)[:, :channels]
    arr = (arr.astype(np.float32) if np.dtype(dtype).kind == "f"
           else np.clip(np.round(arr * 32000), -32768, 32767).astype(np.int16))
    arr.tofile(path)
    return arr


def test_reader_matches_in_memory():
    with workdir("reader") as d:
        for rate, dtype, ch in ((48000, "<f4", 2), (44100, "int16", 2), (44100, "int16", 1)):
            p = d / f"{rate}_{np.dtype(dtype).kind}{ch}.raw"
            arr = make_raw(p, rate * 7, rate, dtype, ch)
            ref = A.resample_to_target(arr, rate, 16000)
            out = []
            with A._RawPcmReader(p, dtype, ch) as r:
                assert len(r) == rate * 7, f"{rate}: frames {len(r)} != {rate * 7}"
                A._resample_stream(r.read, len(r), rate, 16000, out.append)
            got = np.concatenate(out)
            assert len(got) == len(ref), f"{rate}/{dtype}/{ch}: len {len(got)} != {len(ref)}"
            diff = int(np.abs(got.astype(np.int32) - ref.astype(np.int32)).max())
            print(f"  {rate} {dtype} {ch}ch: {len(got)} samples, max diff {diff}")
            assert diff == 0, f"{rate}/{dtype}/{ch}: differs by {diff}"


def test_same_rate_passthrough():
    """A source already at 16 kHz skips the filter but still streams."""
    with workdir("passthrough") as d:
        p = d / "pass.raw"
        arr = make_raw(p, 16000 * 3, 16000, "int16", 2, seed=3)
        ref = A.resample_to_target(arr, 16000, 16000)
        out = []
        with A._RawPcmReader(p, "int16", 2) as r:
            A._resample_stream(r.read, len(r), 16000, 16000, out.append)
        got = np.concatenate(out)
        assert len(got) == len(ref), f"len {len(got)} != {len(ref)}"
        assert int(np.abs(got.astype(np.int32) - ref.astype(np.int32)).max()) == 0
        print(f"  16000 passthrough: {len(got)} samples, exact")


def test_progress_counts_every_input_frame():
    """Progress drives the percentage on the status line, so it has to add up."""
    with workdir("progress") as d:
        p = d / "prog.raw"
        make_raw(p, 48000 * 12, 48000, "<f4", 2, seed=5)
        seen = []
        with A._RawPcmReader(p, "<f4", 2) as r:
            A._resample_stream(r.read, len(r), 48000, 16000, lambda b: None, seen.append)
        print(f"  {len(seen)} blocks, {sum(seen)} frames reported")
        assert sum(seen) == 48000 * 12, f"reported {sum(seen)}, want {48000 * 12}"


if __name__ == "__main__":
    run(["test_reader_matches_in_memory", "test_same_rate_passthrough",
         "test_progress_counts_every_input_frame"], globals())
